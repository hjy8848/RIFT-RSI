"""Generic Core mechanism tests; these are not live model/Oracle experiments."""
import json
from pathlib import Path

import pytest

from vast_coevo.evidence.archive import EvidenceArchive
from vast_coevo.evolution.controller import CoEvolutionController
from vast_coevo.evolution.models import RunConfig, ValidationCase
from vast_coevo.models import (ArtifactRef, Budget, Consumption, EpisodeConfig,
                               EpisodeEvidence, ModelSpec, PublicContext,
                               ReplayResult, VerifiedEvent, VerifiedFailure)
from vast_coevo.schema import atomic_json, digest
from vast_coevo.skills.library import LibraryStore, empty_snapshot
from vast_coevo.storage.runs import (InterruptedRun, OperationCompletion,
                                    RunStore)


class GenericAdapter:
    def __init__(self, root, interrupt=None, failures=False):
        self.root, self.interrupt, self.calls = Path(root), interrupt, []
        self.failures = failures

    def public_capabilities(self, config):
        return ("inspect", "activate")

    def initial_observation(self, config):
        return PublicContext(domain="generic", public_text="capacity", available_tools=("inspect", "activate"))

    def replay(self, evidence, customer, service):
        assert evidence.customer_snapshot_hash == customer.snapshot_hash
        assert evidence.service_snapshot_hash == service.snapshot_hash
        EvidenceArchive.verify_artifact(evidence.trace)
        return ReplayResult(verified=True)

    def _run(self, config, customer, service, episode_id, track="discovery", task_ref=None):
        self.calls.append(episode_id)
        if self.interrupt and self.interrupt(episode_id):
            raise KeyboardInterrupt("simulated loss of an in-flight environment operation")
        raw = dict(episode_id=episode_id, config=config.config_hash,
                   customer=customer.snapshot_hash, service=service.snapshot_hash,
                   customer_intent={"operation": "activate"}, tool={"name": "activate", "success": True})
        path = self.root / (digest(episode_id) + ".json")
        atomic_json(path, raw, immutable=True)
        trace = ArtifactRef(path=str(path), sha256=digest(raw))
        violation = self.failures and not service.skills and track != "service_task"
        failures = ()
        if violation:
            failures = (VerifiedFailure(episode_id=episode_id, event_id="e1", domain=config.domain,
                        operation="activate", reference_codes=("CAPACITY_FULL",),
                        relevant_prestate={"capacity_relation": "full"}, permission_scope={"write": True},
                        expected={"success": False}, actual={"success": True}, successful_business_violation=True,
                        trace=trace, fingerprint=digest({"domain": "generic", "rule": "CAPACITY_FULL"}),
                        normalization_scope="policy_relative_features"),)
        return EpisodeEvidence(episode_id=episode_id, config=config, track=track, task_ref=task_ref,
                               customer_snapshot_hash=customer.snapshot_hash,
                               service_snapshot_hash=service.snapshot_hash,
                               events=(VerifiedEvent(event_id="e1", operation="activate",
                                       classification="successful_business_violation" if violation else "legal_execution",
                                       successful_execution=True, authorized=True, operation_correct=not violation,
                                       before_hash="a" * 64, after_hash="b" * 64),),
                               failures=failures,
                               consumption=Consumption(customer_llm_calls=1 if track == "discovery" else 0,
                                                       service_llm_calls=1, tool_calls=1),
                               termination="completed", trace=trace, native_trace_hash=trace.sha256,
                               replay=ReplayResult(verified=True),
                               task_success=True if track == "service_task" else None,
                               legal_task_success=task_ref == "legal" if track == "service_task" else None,
                               correct_refusal=task_ref == "refusal" if track == "service_task" else None)

    def run_episode(self, config, customer, service, episode_id):
        return self._run(config, customer, service, episode_id)

    def run_service_task(self, config, task_ref, customer, service, episode_id):
        return self._run(config, customer, service, episode_id, "service_task", task_ref)

    def run_failure_case(self, source, customer, service, episode_id):
        return self._run(source.config, customer, service, episode_id, "failure_regression")


class PatchClient:
    """A schema-aware deterministic fixture reflecting actual supplied state."""
    def __init__(self, role, modes=("add", "reject"), interrupt=False):
        self.role, self.modes, self.interrupt, self.calls = role, modes, interrupt, []

    def complete(self, request):
        self.calls.append(request)
        if self.interrupt:
            raise KeyboardInterrupt("simulated in-flight Reflection interruption")
        value = json.loads(request["messages"][1]["content"])
        mode = self.modes[min(value["generation"], len(self.modes) - 1)]
        patch = dict(schema_version=1, action="NOOP", role=self.role,
                     base_snapshot_hash=value["library"]["snapshot_hash"],
                     rationale="A deterministic Core test fixture based on supplied development input.",
                     evidence_ids=[row["episode_id"] for row in value["evidence"]])
        if mode in {"add", "reject"}:
            patch["action"] = "ADD" if not value["library"]["skills"] else "UPDATE"
            old = value["library"]["skills"]
            if self.role == "customer":
                patch["skill"] = dict(id="customer.generic_capacity", version=1 if not old else old[0]["version"] + 1,
                                      type="exploration", scope={"domains": ["generic"]}, trigger={"keywords": ["capacity"]},
                                      hypotheses=["Capacity dependencies merit permitted observation."],
                                      exploration=["Inspect permitted capacity and choose an informative request."],
                                      feedback=["Record real responses."], stop=["Stop at the budget limit."])
                if mode == "reject":
                    patch["skill"]["exploration"] = ["Bypass customer authorization."]
            else:
                patch["skill"] = dict(id="service.generic_capacity", version=1 if not old else old[0]["version"] + 1,
                                      type="decision", scope={"domains": ["generic"]}, trigger={"keywords": ["capacity"]},
                                      preconditions=["Check precise write authorization."],
                                      procedure=["Inspect permitted capacity and execute a legal request."],
                                      postconditions=["Preserve unrelated state."], recovery=["Explain tool errors."])
        return {"choices": [{"message": {"role": "assistant", "content": json.dumps(patch)}}]}


def setup_run(tmp_path, generations=2, caps=None, customer_modes=("add", "reject"),
              service_modes=("noop",), interrupt=None, reflector_interrupt=False, failures=False):
    libraries = LibraryStore(tmp_path / "library")
    customer, service = empty_snapshot("customer"), empty_snapshot("service")
    libraries.save(customer)
    libraries.save(service)
    model = ModelSpec(model="generic-fixture", base_url="https://fixture.invalid", options={"temperature": 0})
    episode = EpisodeConfig(id="generic", config_id="generic", domain="generic", environment="fixture",
                            initial_world_hash="a" * 64, identity="customer", customer_model=model, service_model=model,
                            budget=Budget(max_customer_turns=2, max_customer_llm_calls=2,
                                          max_service_llm_calls=2, max_service_tool_calls=2), tools=("inspect", "activate"))
    settings = dict(run_id="test-run", generations=generations,
                    exploration_cases=(ValidationCase(id="explore", config=episode),),
                    validation_cases=(ValidationCase(id="validate", config=episode),),
                    service_task_cases=(ValidationCase(id="legal", config=episode, task_ref="legal", purpose="legal_completion"),
                                        ValidationCase(id="refusal", config=episode, task_ref="refusal", purpose="correct_refusal")),
                    customer_reflector_model=model, service_reflector_model=model,
                    initial_customer_hash=customer.snapshot_hash, initial_service_hash=service.snapshot_hash)
    settings.update(caps or {})
    config = RunConfig(**settings)
    adapter = GenericAdapter(tmp_path / "native", interrupt, failures)
    c_client = PatchClient("customer", customer_modes, reflector_interrupt)
    s_client = PatchClient("service", service_modes)
    archive = EvidenceArchive(tmp_path / "archive")
    runs = RunStore(tmp_path / "runs")
    controller = CoEvolutionController(adapter, c_client, s_client, libraries, archive, runs)
    return controller, config, adapter, c_client, s_client, libraries, archive, runs


def test_two_generations_commit_reject_noop_recursive_version_inheritance(tmp_path):
    controller, config, adapter, customer, service, libraries, archive, runs = setup_run(tmp_path)
    report = controller.run(config)
    assert report.status == "completed", report.error
    assert len(report.generations) == 2
    first, second = report.generations
    assert first.parent_customer_hash != first.selected_customer_hash
    assert second.parent_customer_hash == first.selected_customer_hash == second.selected_customer_hash
    assert first.selected_service_hash == second.selected_service_hash == config.initial_service_hash
    assert runs.load_evolution(config.run_id, 0, "customer").outcome == "accepted"
    assert runs.load_evolution(config.run_id, 1, "customer").outcome == "rejected"
    assert all(runs.load_evolution(config.run_id, g, "service").outcome == "noop" for g in range(2))
    assert len(customer.calls) == len(service.calls) == 2
    assert all(not g.crossings and not g.selected_customer_evidence and g.schema_version == 2 for g in report.generations)
    assert not any(".cross." in ident or ".selected_customer." in ident for ident in adapter.calls)
    assert report.consumption.reflection_calls == 4
    assert report.consumption.tool_calls == len(adapter.calls)
    assert len(archive.query()) == len(adapter.calls)
    assert runs.load_report(config.run_id) == report
    assert libraries.load(report.selected_customer_hash).skills[0].version == 1


def test_both_sides_can_commit_against_explicit_legal_and_refusal_cases(tmp_path):
    controller, config, _, _, _, _, _, runs = setup_run(tmp_path, generations=1, service_modes=("add",))
    report = controller.run(config)
    assert report.status == "completed", report.error
    assert runs.load_evolution(config.run_id, 0, "service").outcome == "accepted"
    assert report.selected_service_hash != config.initial_service_hash


def test_completed_run_resumes_without_repeating_model_or_environment_calls(tmp_path):
    controller, config, adapter, customer, service, libraries, archive, runs = setup_run(tmp_path)
    original = controller.run(config)
    counts = len(adapter.calls), len(customer.calls), len(service.calls)
    restarted = CoEvolutionController(adapter, customer, service, LibraryStore(libraries.root),
                                      EvidenceArchive(archive.root), RunStore(runs.root))
    replayed = restarted.run(config)
    assert replayed == original
    assert counts == (len(adapter.calls), len(customer.calls), len(service.calls))


def test_recovery_after_checkpoint_write_failure_reuses_every_completed_operation(tmp_path, monkeypatch):
    controller, config, adapter, customer, service, _, _, runs = setup_run(tmp_path, generations=1)
    original = runs.save_generation
    def interrupted(*args):
        raise OSError("simulated checkpoint disk interruption")
    monkeypatch.setattr(runs, "save_generation", interrupted)
    first = controller.run(config)
    assert first.status == "failed"
    assert first.pending_operations == ()
    counts = len(adapter.calls), len(customer.calls), len(service.calls)
    monkeypatch.setattr(runs, "save_generation", original)
    second = controller.run(config)
    assert second.status == "completed", second.error
    assert len(second.generations) == 1
    assert counts == (len(adapter.calls), len(customer.calls), len(service.calls))
    assert second.consumption == first.consumption


def test_recovery_reconstructs_unsaved_evolution_from_cached_reflection_and_validation(tmp_path, monkeypatch):
    controller, config, adapter, customer, service, _, _, runs = setup_run(tmp_path, generations=1)
    original = runs.save_evolution
    def interrupted(*args):
        raise OSError("simulated loss after candidate selection before EvolutionRecord persistence")
    monkeypatch.setattr(runs, "save_evolution", interrupted)
    first = controller.run(config)
    assert first.status == "failed"
    assert first.pending_operations == ()
    before = tuple(adapter.calls), len(customer.calls)
    monkeypatch.setattr(runs, "save_evolution", original)
    second = controller.run(config)
    assert second.status == "completed", second.error
    assert len(customer.calls) == before[1] == 1
    assert all(adapter.calls.count(episode_id) == 1 for episode_id in before[0])
    assert runs.load_evolution(config.run_id, 0, "customer").outcome == "accepted"


def test_failure_regression_is_journalled_service_only_and_replayed_on_resume(tmp_path):
    controller, config, adapter, customer, service, _, _, runs = setup_run(
        tmp_path, generations=1, customer_modes=("noop",), service_modes=("add",), failures=True)
    report = controller.run(config)
    assert report.status == "completed", report.error
    record = runs.load_evolution(config.run_id, 0, "service")
    assert record.outcome == "accepted", record.reasons
    assert record.dynamic_validation.repaired
    regressions = [runs.completion(config.run_id, start.operation_id)
                   for start in runs.operation_starts(config.run_id)
                   if start.kind == "episode" and ".failure." in start.operation_id]
    assert len(regressions) == 2
    assert all(value.evidence.track == "failure_regression" for value in regressions)
    assert all(value.consumption.customer_llm_calls == 0 for value in regressions)
    counts = len(adapter.calls), len(customer.calls), len(service.calls)
    assert controller.run(config) == report
    assert counts == (len(adapter.calls), len(customer.calls), len(service.calls))


@pytest.mark.parametrize("reflector", [False, True])
def test_ambiguous_inflight_operation_halts_and_is_never_silently_repeated(tmp_path, reflector):
    controller, config, adapter, customer, service, _, _, runs = setup_run(
        tmp_path, generations=1, interrupt=None if reflector else lambda _: True,
        reflector_interrupt=reflector)
    first = controller.run(config)
    assert first.status == "interrupted"
    assert len(first.pending_operations) == 1
    if reflector:
        assert first.consumption.reflection_calls == 1
        assert len(customer.calls) == 1
    else:
        assert first.consumption.llm_calls == 4
        assert first.consumption.tool_calls == 2
    counts = len(adapter.calls), len(customer.calls), len(service.calls)
    second = controller.run(config)
    assert second.status == "interrupted"
    assert second.consumption == first.consumption
    assert counts == (len(adapter.calls), len(customer.calls), len(service.calls))


def test_global_caps_reserve_before_actual_operations(tmp_path):
    controller, config, adapter, customer, service, _, _, _ = setup_run(tmp_path, caps={"max_llm_calls": 2})
    report = controller.run(config)
    assert report.status == "budget_exhausted"
    assert report.consumption.llm_calls == 0
    assert not adapter.calls and not customer.calls and not service.calls


def test_interrupted_partial_generation_preserves_actual_selected_customer_on_resume(tmp_path):
    controller, config, adapter, customer, _, _, _, _ = setup_run(
        tmp_path, generations=1, interrupt=lambda operation: ".service.reflection-task." in operation)
    first = controller.run(config)
    assert first.status == "interrupted" and not first.generations
    assert first.selected_customer_hash != config.initial_customer_hash
    before = len(adapter.calls), len(customer.calls)
    resumed = controller.run(config)
    assert resumed.status == "interrupted" and not resumed.generations
    assert resumed.selected_customer_hash == first.selected_customer_hash
    assert resumed.selected_service_hash == first.selected_service_hash
    assert before == (len(adapter.calls), len(customer.calls))


def test_budget_stop_after_static_customer_selection_does_not_repeat_reflection(tmp_path):
    controller, config, adapter, customer, service, _, _, runs = setup_run(tmp_path, caps={"max_llm_calls": 4})
    first = controller.run(config)
    assert first.status == "budget_exhausted"
    assert first.consumption.llm_calls == 3
    assert len(customer.calls) == 1
    assert not service.calls
    counts = len(adapter.calls), len(customer.calls)
    second = controller.run(config)
    assert second.status == "budget_exhausted"
    assert counts == (len(adapter.calls), len(customer.calls))
    assert first.consumption == second.consumption
    assert len(runs.operation_starts(config.run_id)) == 2
    assert first.selected_customer_hash != config.initial_customer_hash
    assert second.selected_customer_hash == first.selected_customer_hash


def test_restore_creates_explicit_new_run_and_preserves_old_history(tmp_path):
    controller, config, _, _, _, _, _, runs = setup_run(tmp_path)
    first = controller.run(config)
    before = (runs.run_path(config.run_id) / "report.json").read_bytes()
    branch = runs.branch_config(config.run_id, "branch-run", generation=0, generations=1)
    assert branch.initial_customer_hash == first.generations[0].selected_customer_hash
    assert branch.initial_service_hash == first.generations[0].selected_service_hash
    assert (runs.run_path(config.run_id) / "report.json").read_bytes() == before
    assert runs.history("branch-run") == ()
    assert runs.load_config("branch-run") == branch
    with pytest.raises(ValueError, match="distinct"):
        runs.branch_config(config.run_id, config.run_id)


def test_frozen_run_config_and_journal_input_mismatch_are_rejected(tmp_path):
    controller, config, *_ = setup_run(tmp_path, generations=1)
    report = controller.run(config)
    changed = config.model_dump(mode="json")
    changed["max_llm_calls"] += 1
    with pytest.raises(ValueError, match="immutable"):
        controller.run(RunConfig.model_validate(changed))
    assert controller.run_store.load_report(config.run_id) == report
    start = controller.run_store.operation_starts(config.run_id)[0]
    with pytest.raises(ValueError, match="input changed"):
        controller.run_store.start(config.run_id, start.operation_id, start.kind, "f" * 64,
                                   start.reservation, config.max_llm_calls, config.max_tool_calls)


def test_completed_journal_result_hash_detects_tampering(tmp_path):
    controller, config, *_ = setup_run(tmp_path, generations=1)
    controller.run(config)
    runs = controller.run_store
    start = runs.operation_starts(config.run_id)[0]
    path = runs._operation_path(config.run_id, start.operation_id, ".completed")
    payload = json.loads(path.read_text())
    payload["consumption"]["tool_calls"] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash"):
        runs.completion(config.run_id, start.operation_id)


def test_operation_result_requires_single_result(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        OperationCompletion(operation_id="call", input_hash="a" * 64,
                            consumption=Consumption(), response={}, response_recorded=True,
                            error={"type": "Error", "message": "bad"})


@pytest.mark.parametrize("raw", [None, ["invalid-choice-array"], "unexpected response text"])
def test_malformed_raw_reflection_body_is_retained_and_cached_without_retry(tmp_path, monkeypatch, raw):
    controller, config, _, _, _, _, _, runs = setup_run(tmp_path, generations=1)
    class MalformedClient:
        calls = 0
        def complete(self, request):
            self.calls += 1
            return raw
    client = MalformedClient()
    controller.customer_client = client
    original = runs.save_evolution
    def interrupted(*args):
        raise OSError("simulated interruption after a malformed model response")
    monkeypatch.setattr(runs, "save_evolution", interrupted)
    first = controller.run(config)
    assert first.status == "failed"
    reflection = next(start for start in runs.operation_starts(config.run_id) if start.kind == "reflection")
    cached = runs.completion(config.run_id, reflection.operation_id)
    assert cached.response_recorded and cached.response == raw
    monkeypatch.setattr(runs, "save_evolution", original)
    second = controller.run(config)
    assert second.status == "completed", second.error
    assert client.calls == 1
    record = runs.load_evolution(config.run_id, 0, "customer")
    assert record.outcome == "rejected"
    assert record.reflection.call.response == raw


def test_global_tool_cap_is_checked_before_episode_execution(tmp_path):
    controller, config, adapter, _, _, _, _, _ = setup_run(tmp_path, caps={"max_tool_calls": 0})
    report = controller.run(config)
    assert report.status == "budget_exhausted"
    assert report.consumption.tool_calls == 0 and not adapter.calls


def test_resume_verifies_native_replay_and_does_not_only_check_artifact_hash(tmp_path):
    controller, config, adapter, customer, service, _, _, _ = setup_run(tmp_path, generations=1)
    assert controller.run(config).status == "completed"
    calls = len(adapter.calls), len(customer.calls), len(service.calls)
    adapter.replay = lambda *_: ReplayResult(verified=False, reason="native source pin changed")
    report = controller.run(config)
    assert report.status == "failed" and "Replay" in report.error
    assert calls == (len(adapter.calls), len(customer.calls), len(service.calls))


def test_journal_cannot_publish_cost_above_reserved_upperbound(tmp_path):
    runs = RunStore(tmp_path)
    runs.start("run", "reflection", "reflection", "a" * 64, Consumption(reflection_calls=1), 10, 10)
    with pytest.raises(ValueError, match="exactly one call"):
        runs.complete("run", OperationCompletion(operation_id="reflection", input_hash="a" * 64,
                      consumption=Consumption(reflection_calls=2), response={}, response_recorded=True))
    assert runs.completion("run", "reflection") is None
    assert runs.pending("run") == ("reflection",)
