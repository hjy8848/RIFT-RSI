"""Selection mechanism tests; deterministic fixtures, no online experiments."""
import json

import pytest

from vast_coevo.evolution.evolvers import CustomerEvolver
from vast_coevo.evolution.gates import DynamicGate
from vast_coevo.evolution.models import GatePolicy, RunConfig
from vast_coevo.evaluation.customer import CustomerQualityPolicy
from vast_coevo.reflection.reflector import CustomerReflector
from vast_coevo.schema import digest
from vast_coevo.skills.library import apply_patch
from test_vast_evolution import MODEL, PatchClient, customer_skill, proposal, setup
from test_vast_controller import setup_run


def test_worse_customer_is_selected_and_later_negative_diagnostics_cannot_rollback(tmp_path):
    adapter, archive, store, customer, service, case, _ = setup(tmp_path)
    adapter.customer_mode = "regress"
    record = CustomerEvolver(adapter, CustomerReflector(PatchClient(
        proposal(customer, skill=customer_skill())), MODEL), store, archive).update(
            0, customer, service, (), (case,))
    assert record.outcome == "accepted" and record.static_validation.passed
    assert record.reasons == ("static_gate_passed_snapshot_updated",)
    assert record.dynamic_validation is None and adapter.calls == []
    assert record.consumption.llm_calls == 1 and record.consumption.tool_calls == 0
    selected = store.load(record.selected_snapshot_hash)
    before = digest(record), store.hashes()
    diagnosis = DynamicGate(adapter, archive).customer(customer, selected, service, (case,), "worse-diagnostic")
    assert not diagnosis.passed
    assert "valid_actions_decreased" in diagnosis.reasons
    assert (digest(record), store.hashes()) == before
    assert record.selected_snapshot_hash == selected.snapshot_hash != customer.snapshot_hash


def test_diagnostic_policy_cannot_impose_customer_adapter_or_sample_requirements(tmp_path):
    adapter, archive, store, customer, service, case, _ = setup(tmp_path)
    # Generic adapter intentionally lacks QGD hooks. Performance preconditions
    # belong to explicit diagnostics, never to static publication.
    policy = GatePolicy(customer_quality=CustomerQualityPolicy())
    record = CustomerEvolver(adapter, CustomerReflector(PatchClient(
        proposal(customer, skill=customer_skill())), MODEL), store, archive, policy).update(
            0, customer, service, (), (case,))
    assert record.outcome == "accepted" and not adapter.calls
    selected_hash = record.selected_snapshot_hash
    with pytest.raises(ValueError, match="CloudOps"):
        DynamicGate(adapter, archive, policy).customer(customer, store.load(selected_hash), service, (case,), "unavailable")
    assert record.selected_snapshot_hash == selected_hash
    assert store.load(selected_hash).parent_hash == customer.snapshot_hash


def test_two_successive_customer_updates_reach_service_and_next_generation(tmp_path, monkeypatch):
    def forbidden_diagnostic(*args, **kwargs):
        pytest.fail("Controller must not invoke Customer performance diagnostics")
    monkeypatch.setattr(DynamicGate, "customer", forbidden_diagnostic)
    controller, config, adapter, customer_client, service_client, libraries, archive, runs = setup_run(
        tmp_path, customer_modes=("add", "add"))
    report = controller.run(config)
    assert report.status == "completed", report.error
    first, second = report.generations
    assert first.selected_customer_hash == second.parent_customer_hash
    assert second.selected_customer_hash != first.selected_customer_hash
    assert libraries.load(second.selected_customer_hash).skills[0].version == 2
    for generation in range(2):
        selected = report.generations[generation].selected_customer_hash
        service_input = json.loads(service_client.calls[generation]["messages"][1]["content"])
        selected_evidence = [row for row in service_input["evidence"] if ".current." in row["episode_id"]]
        assert selected_evidence and all(row["customer_snapshot_hash"] == report.generations[generation].parent_customer_hash for row in selected_evidence)
        assert runs.load_evolution(config.run_id, generation, "service").parent_customer_hash == selected
        assert all(row.customer_hash == selected for row in report.generations[generation].crossings
                   if row.pairing.startswith("selected_"))
    assert report.consumption.reflection_calls == 4
    assert json.loads(customer_client.calls[1]["messages"][1]["content"])["library"]["snapshot_hash"] == first.selected_customer_hash
    assert report.consumption.tool_calls == len(adapter.calls)
    calls = len(adapter.calls), len(customer_client.calls), len(service_client.calls)
    assert controller.run(config) == report  # every completed operation gets exact Replay
    assert calls == (len(adapter.calls), len(customer_client.calls), len(service_client.calls))


def test_historical_config_roundtrips_but_cannot_resume_under_new_selection(tmp_path):
    controller, current, adapter, customer, service, _, _, runs = setup_run(tmp_path)
    raw = current.model_dump(mode="json")
    raw["schema_version"] = 1
    raw.pop("customer_selection")
    raw.pop("training_protocol", None)
    historical = RunConfig.model_validate(raw)
    assert historical.model_dump(mode="json") == raw
    assert digest(historical) == digest(raw)
    runs.initialize(historical)
    before = (runs.run_path(current.run_id) / "config.json").read_bytes()
    with pytest.raises(ValueError, match="original code"):
        controller.run(historical)
    assert not adapter.calls and not customer.calls and not service.calls
    assert (runs.run_path(current.run_id) / "config.json").read_bytes() == before


@pytest.mark.parametrize("failed_replay", [False, True])
def test_optional_diagnostic_cli_records_costs_without_publishing_candidate(tmp_path, monkeypatch, capsys, failed_replay):
    from integrations.usami.cli import ExperimentPlan, main
    from test_vast_cli import plan_for
    from test_vast_usami import adapter_for, libraries, SPEC
    from vast_coevo.evidence.archive import EvidenceArchive
    from vast_coevo.skills.library import LibraryStore
    from vast_coevo.skills.models import SkillPatch
    from vast_coevo.storage.runs import RunStore
    adapter, config, clients = adapter_for(tmp_path / "native", "cloudops")
    customer, service = libraries("cloudops")
    root, path = plan_for(tmp_path, adapter, config, customer, service)
    plan = ExperimentPlan.model_validate_json(path.read_text())
    skill = customer.skills[0].model_dump(mode="json", exclude={"content_hash"})
    skill["version"] = 2
    patch = SkillPatch(action="UPDATE", role="customer", base_snapshot_hash=customer.snapshot_hash,
        skill=type(customer.skills[0]).model_validate(skill), rationale="Deterministic diagnostic isolation fixture.")
    candidate = apply_patch(customer, patch, "fixture")
    store, archive = LibraryStore(root / "libraries"), EvidenceArchive(root / "evidence")
    store.save(candidate, publish=False)
    published = tuple((store.root / "skills").rglob("*.json"))
    monkeypatch.setattr("integrations.usami.cli.context", lambda *_: (
        plan, adapter, store, archive, RunStore(root / "runs"), customer, service,
        {role: SPEC for role in ("customer", "service", "customer_reflector", "service_reflector")}))
    args = ["--root", str(root), "--plan", str(path), "diagnose-customer", "--id", "explicit",
        "--baseline", customer.snapshot_hash, "--candidate", candidate.snapshot_hash, "--service", service.snapshot_hash]
    if failed_replay:
        from vast_coevo.models import ReplayResult
        monkeypatch.setattr(adapter, "replay", lambda *_: ReplayResult(verified=False, reason="fixture integrity failure"))
        with pytest.raises(SystemExit):
            main(args)
        output = json.loads((root / 'diagnostics' / 'explicit' / 'result.json').read_text())
        assert output['status'] == 'failed' and 'replay' in output['error']
    else:
        assert main(args) == 0
        output = json.loads(capsys.readouterr().out)
        assert output["status"] == "completed"
        assert output["consumption"] == output["result"]["consumption"]
    assert output["selection_authority"] is False and "selected_snapshot_hash" not in output
    assert output["source_hashes"] and output["protocol"] == "legacy-v1"
    assert output["consumption"]["reflection_calls"] == 0
    assert output["consumption"]["customer_llm_calls"] > 0
    assert tuple((store.root / "skills").rglob("*.json")) == published
    assert not (root / "runs").exists()  # diagnostic journal is independent
    count = len(clients)
    with pytest.raises(SystemExit):
        main(args)
    assert len(clients) == count  # no hidden second run or best-of selection
