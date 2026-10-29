"""Executable integration CLI checks; no external LLM is invoked."""
import json
from pathlib import Path
import pytest

from integrations.usami.cli import ExperimentPlan, main, run_config
from integrations.usami import UsamiAdapter
from vast_coevo.schema import atomic_json
from vast_coevo.models import EpisodeEvidence, ModelSpec
from vast_coevo.skills.library import LibrarySnapshot, LibraryStore
from test_vast_usami import adapter_for, libraries, SPEC


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("filename,worlds,tasks", [("cloudops-pilot.json", 1, 2), ("four-domain-development.json", 4, 8)])
def test_shipped_plans_resolve_real_native_worlds_and_task_kinds(tmp_path, filename, worlds, tasks):
    path = ROOT / "experiments/vast_coevo/configs" / filename
    plan = ExperimentPlan.model_validate_json(path.read_text())
    customer = LibrarySnapshot.model_validate_json((path.parent / plan.customer_library).read_text())
    service = LibrarySnapshot.model_validate_json((path.parent / plan.service_library).read_text())
    adapter = UsamiAdapter(tmp_path, plan.profiles, plan.service_tasks)
    specs = {key: SPEC for key in ("customer", "service", "customer_reflector", "service_reflector")}
    config = run_config(plan, adapter, customer, service, specs, "plan-check")
    assert len(config.exploration_cases) == worlds
    assert len(config.service_task_cases) == tasks
    assert {case.purpose for case in config.service_task_cases} == {"legal_completion", "correct_refusal"}
    assert all(case.config.partition == "development" for case in config.exploration_cases)


def plan_for(tmp_path, adapter, config, customer, service):
    root = tmp_path / "cli"
    store = LibraryStore(root / "libraries")
    store.save(customer)
    store.save(service)
    atomic_json(tmp_path / "customer.json", customer.model_dump(mode="json"))
    atomic_json(tmp_path / "service.json", service.model_dump(mode="json"))
    raw = {"profiles": [p.model_dump(mode="json") for p in adapter._profiles.values()],
           "exploration": [config.config_id], "validation": [config.config_id],
           "customer_library": "customer.json", "service_library": "service.json"}
    raw.update({key + "_model": SPEC.model_dump(mode="json") for key in ("customer", "service", "customer_reflector", "service_reflector")})
    path = tmp_path / "plan.json"
    atomic_json(path, raw)
    return root, path


def test_offline_replay_cli_needs_no_model_env_auth_or_network(tmp_path, monkeypatch, capsys):
    adapter, config, _ = adapter_for(tmp_path / "native", "cloudops")
    customer, service = libraries("cloudops")
    row = adapter.run_episode(config, customer, service, "cli-replay")
    root, plan = plan_for(tmp_path, adapter, config, customer, service)
    raw = json.loads(plan.read_text())
    for key in list(raw):
        if key.endswith("_model"):
            del raw[key]
    atomic_json(plan, raw)
    for name in list(__import__("os").environ):
        if name.startswith(("VAST_", "USAMI_", "OPENAI_")):
            monkeypatch.delenv(name)
    evidence_path = tmp_path / "evidence.json"
    atomic_json(evidence_path, row.model_dump(mode="json"))
    assert main(["--root", str(root), "--plan", str(plan), "replay", "--evidence", str(evidence_path)]) == 0
    assert json.loads(capsys.readouterr().out)["verified"]


def test_reflection_cli_rejects_resealed_fake_oracle_summary_before_model_call(tmp_path, monkeypatch):
    adapter, config, _ = adapter_for(tmp_path / "native", "cloudops")
    customer, service = libraries("cloudops")
    row = adapter.run_episode(config, customer, service, "cli-reflect-verify")
    root, plan = plan_for(tmp_path, adapter, config, customer, service)
    raw = row.model_dump(mode="json", exclude={"evidence_hash"})
    raw["events"][0]["classification"] = "successful_business_violation"
    fake = EpisodeEvidence.model_validate(raw)
    path = tmp_path / "fake-evidence.json"
    atomic_json(path, fake.model_dump(mode="json"))
    monkeypatch.setattr("integrations.usami.cli.transport", lambda spec: pytest.fail("must verify before calling model"))
    with pytest.raises(SystemExit) as error:
        main(["--root", str(root), "--plan", str(plan), "reflect", "--role", "service", "--evidence", str(path),
              "--output", str(tmp_path / "reflection.json")])
    assert error.value.code == 2


def test_reflection_cli_program_fills_add_scope_from_plan(tmp_path, monkeypatch, capsys):
    adapter, config, _ = adapter_for(tmp_path / "native", "cloudops")
    customer, service = libraries("cloudops")
    row = adapter.run_episode(config, customer, service, "cli-reflect-managed-scope")
    root, plan = plan_for(tmp_path, adapter, config, customer, service)
    path = tmp_path / "evidence.json"
    atomic_json(path, row.model_dump(mode="json"))
    skill = customer.skills[0].model_dump(mode="json", exclude={"content_hash", "scope"})
    skill["id"] = "customer.new_exploration"
    raw = dict(action="ADD", role="customer", base_snapshot_hash=customer.snapshot_hash, skill=skill,
               rationale="One labelled CI proposal, not a live model result.", evidence_ids=[row.episode_id])
    from test_vast_reflection import Client, completion
    client = Client(completion(json.dumps(raw)))
    client.close = lambda: None
    monkeypatch.setattr("integrations.usami.cli.transport", lambda spec: client)
    assert main(["--root", str(root), "--plan", str(plan), "reflect", "--role", "customer", "--evidence", str(path),
                 "--output", str(tmp_path / "reflection.json")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["error"] is None and result["patch"]["skill"]["scope"]["domains"] == ["cloudops"]
    assert result["scope_resolution"]["source"] == "experiment_policy"
    assert len(client.requests) == 1


def test_published_episode_id_fails_before_any_duplicate_model_calls(tmp_path):
    adapter, config, clients = adapter_for(tmp_path, "cloudops")
    customer, service = libraries("cloudops")
    adapter.run_episode(config, customer, service, "same-id")
    count = len(clients)
    with pytest.raises(ValueError, match="already published"):
        adapter.run_episode(config, customer, service, "same-id")
    assert len(clients) == count


def test_each_actual_call_is_durable_and_offline_replay_does_not_write_call_journal(tmp_path):
    adapter, config, _ = adapter_for(tmp_path, "cloudops")
    customer, service = libraries("cloudops")
    row = adapter.run_episode(config, customer, service, "durable-calls")
    folder = Path(row.trace.path).parent / "calls"
    starts, completed = sorted(folder.glob("*.started.json")), sorted(folder.glob("*.completed.json"))
    assert len(starts) == len(completed) == len(row.model_calls)
    actual = {json.loads(path.read_text())["call"]["id"]: json.loads(path.read_text())["call"] for path in completed}
    assert all(actual[call.id] == call.model_dump(mode="json") for call in row.model_calls)
    times = {path: path.stat().st_mtime_ns for path in folder.glob("*.json")}
    assert adapter.replay(row, customer, service).verified
    assert {path: path.stat().st_mtime_ns for path in folder.glob("*.json")} == times


def test_in_flight_episode_marker_preserves_completed_requests_without_repeat(tmp_path):
    adapter, config, clients = adapter_for(tmp_path, "cloudops")
    customer, service = libraries("cloudops")
    episode_id = "unknown-outcome"
    from vast_coevo.schema import digest
    path = tmp_path / "episodes" / digest({"episode_id": episode_id})[:32] / "online-start.json"
    atomic_json(path, {"episode_id": episode_id, "outcome": "unknown"}, immutable=True)
    with pytest.raises(ValueError, match="in-flight"):
        adapter.run_episode(config, customer, service, episode_id)
    assert not clients


def test_reflection_cli_existing_marker_cannot_repeat_a_model_call(tmp_path, monkeypatch):
    output = tmp_path / "reflection.json"
    atomic_json(str(output) + ".started.json", {"input_hash": "unknown"})
    monkeypatch.setattr("integrations.usami.cli.context", lambda *args: pytest.fail("must not start another Reflection"))
    with pytest.raises(SystemExit) as error:
        main(["--plan", "unused-plan", "reflect", "--role", "customer", "--evidence", "unused", "--output", str(output)])
    assert error.value.code == 2
