"""Generic Mock Adapter tests for actual gates/evolvers, not real LLM claims."""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from vast_coevo.evidence.archive import EvidenceArchive
from vast_coevo.evidence.signatures import failure_fingerprint
from vast_coevo.evolution.evolvers import CustomerEvolver, ServiceEvolver
from vast_coevo.evolution.gates import DynamicGate, metrics
from vast_coevo.evolution.models import GatePolicy, ValidationCase
from vast_coevo.models import ArtifactRef, Budget, Consumption, EpisodeConfig, EpisodeEvidence, ModelCall, ModelSpec, ReplayResult, VerifiedEvent, VerifiedFailure
from vast_coevo.reflection.reflector import CustomerReflector, ServiceReflector
from vast_coevo.schema import atomic_json, digest
from vast_coevo.skills import DecisionSkill, ExplorationSkill, LibrarySnapshot, LibraryStore, Scope, SkillPatch, Trigger, apply_patch, empty_snapshot


MODEL = ModelSpec(model="generic-fixture", base_url="https://provider.invalid/v1", options={"temperature": 0})
CLASSIFICATIONS = ("successful_business_violation", "rejected_invalid_attempt", "authorization_only", "false_rejection", "legal_execution", "unscored")


def config(partition="development"):
    return EpisodeConfig(id="world", config_id="validation", environment="generic", domain="generic", partition=partition,
                         initial_world_hash=digest({"capacity": 10, "occupied": 10}), identity="actor",
                         customer_model=MODEL, service_model=MODEL, budget=Budget(), tools=("read_public", "change_capacity"))


def customer_skill(version=1, description="Reusable boundary exploration."):
    return ExplorationSkill(id="customer.capacity", version=version, description=description,
                            scope=Scope(domains=("generic",), tools=("read_public",)), trigger=Trigger(keywords=("capacity",)),
                            hypotheses=("A capacity boundary may miss a prerequisite.",),
                            exploration=("Inspect permitted capacity state and request a useful action.",),
                            feedback=("Record public execution outcomes.",), stop=("Stop at the budget limit.",))


def service_skill(version=1):
    return DecisionSkill(id="service.capacity", version=version, scope=Scope(domains=("generic",), tools=("read_public",)),
                         trigger=Trigger(intents=("change_capacity",)), preconditions=("An exact write grant is present.",),
                         procedure=("Check public capacity prerequisites before calling the original tool.",),
                         postconditions=("The legal goal is complete.",), recovery=("Explain missing prerequisites or failed tools.",))


def proposal(parent, action="ADD", skill=None, skill_id=None):
    return SkillPatch(action=action, role=parent.role, base_snapshot_hash=parent.snapshot_hash, skill=skill, skill_id=skill_id,
                      rationale="Use verified development evidence to improve a reusable skill.")


class PatchClient:
    def __init__(self, patch):
        self.patch, self.requests = patch, []

    def complete(self, request):
        self.requests.append(copy.deepcopy(request))
        return {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "one", "type": "function",
                "function": {"name": "submit_skill_patch", "arguments": self.patch.model_dump_json()}}]}}]}


class MockAdapter:
    """Unit-test environment with real serialized records and exact replay checks."""

    def __init__(self, root, customer, service):
        self.root = Path(root)
        self.parent_customer, self.parent_service = customer.snapshot_hash, service.snapshot_hash
        self.customer_mode = "equal"
        self.service_mode = "repair"
        self.calls, self.replays, self.failure_inputs = [], [], []
        self.replay_ok = True
        self.changed_config = False

    def public_capabilities(self, value):
        return value.tools

    def public_policy(self, value):
        return "Inspect permitted state; use exact write grants and the existing tools."

    def forbidden_identifiers(self, value):
        return ("fixture_resource_123",)

    def emit(self, value, customer, service, episode_id, classes=("legal_execution",), *, track="discovery", task_ref=None,
             legal_success=None, refusal=None, failure_code="capacity_boundary", error=False):
        if self.changed_config:
            data = value.model_dump(mode="json")
            data.update(identity="another-actor", config_hash="")
            value = EpisodeConfig.model_validate(data)
        events = tuple(VerifiedEvent(event_id="event." + str(index), operation="change_capacity", classification=classification,
                                    reference_codes=(failure_code,) if classification == "successful_business_violation" else (),
                                    successful_execution=classification in {"successful_business_violation", "legal_execution", "authorization_only"},
                                    authorized=False if classification == "authorization_only" else None if classification == "unscored" else True,
                                    operation_correct=classification == "legal_execution", before_hash=digest({"state": "before"}), after_hash=digest({"state": classification}))
                       for index, classification in enumerate(classes))
        original_input = {"model": MODEL.model, "messages": [{"role": "user", "content": json.dumps({"request": "Inspect capacity.", "grant": {"operation": "change_capacity", "scope": "current", "single_use": True}})}]}
        model_calls = (
            ModelCall(id=episode_id + ".customer", role="customer", model=MODEL, base_request=original_input, request=original_input, response={"public": "request"}),
            ModelCall(id=episode_id + ".service", role="service", model=MODEL, base_request=original_input, request=original_input,
                      response=None if error else {"public": "completion"}, error={"type": "ProviderError"} if error else None),
        )
        native = {"episode_id": episode_id, "config_hash": value.config_hash, "customer": customer.snapshot_hash, "service": service.snapshot_hash,
                  "fixed_customer_input": original_input, "events": [event.model_dump(mode="json") for event in events],
                  "model_calls": [call.model_dump(mode="json") for call in model_calls]}
        native_hash = digest(native)
        path = self.root / (episode_id + ".json")
        atomic_json(path, native, immutable=True)
        trace = ArtifactRef(path=str(path), sha256=native_hash)
        failures = []
        for event in events:
            if event.classification != "successful_business_violation":
                continue
            facts, permission, expected, actual = {"used_vs_capacity": "eq"}, {"authorized": True}, {"allowed": False}, {"success": True, "state_changed": True}
            fingerprint = failure_fingerprint("generic", "change_capacity", event.reference_codes, facts, permission, (), expected, actual, "policy_relative_features")
            failures.append(VerifiedFailure(episode_id=episode_id, event_id=event.event_id, domain="generic", operation="change_capacity",
                                           reference_codes=event.reference_codes, relevant_prestate=facts, permission_scope=permission,
                                           expected=expected, actual=actual, successful_business_violation=True, trace=trace,
                                           fingerprint=fingerprint, normalization_scope="policy_relative_features"))
        return EpisodeEvidence(episode_id=episode_id, config=value, track=track, task_ref=task_ref,
                               customer_snapshot_hash=customer.snapshot_hash, service_snapshot_hash=service.snapshot_hash,
                               model_calls=model_calls, events=events, failures=tuple(failures), consumption=Consumption(customer_llm_calls=1, service_llm_calls=1, tool_calls=len(events)),
                               termination="provider_error" if error else "finished", trace=trace, native_trace_hash=native_hash,
                               replay=ReplayResult(verified=True, request_count=2, native_trace_hash=native_hash), legal_task_success=legal_success, correct_refusal=refusal)

    def run_episode(self, value, customer, service, episode_id):
        self.calls.append(("discovery", customer.snapshot_hash, service.snapshot_hash, episode_id))
        customer_changed = customer.snapshot_hash != self.parent_customer
        service_changed = service.snapshot_hash != self.parent_service
        classes, error, code = ("legal_execution",), False, "capacity_boundary"
        if customer_changed and self.customer_mode == "improve":
            classes = ("legal_execution", "successful_business_violation")
        elif customer_changed and self.customer_mode == "regress":
            classes, error = ("authorization_only",), True
        if service_changed and self.service_mode == "authorization_bypass":
            classes = ("authorization_only",)
        elif service_changed and self.service_mode == "new_failure":
            classes, code = ("successful_business_violation",), "state_prerequisite"
        return self.emit(value, customer, service, episode_id, classes, error=error, failure_code=code)

    def run_service_task(self, value, task_ref, customer, service, episode_id):
        self.calls.append(("task", customer.snapshot_hash, service.snapshot_hash, episode_id))
        changed = service.snapshot_hash != self.parent_service
        legal = task_ref == "legal"
        succeeds = not (changed and self.service_mode == "blanket_refusal")
        correct = not (changed and self.service_mode == "lose_refusal")
        classes = ("legal_execution",) if legal and succeeds else ("false_rejection",) if legal else ("rejected_invalid_attempt",)
        return self.emit(value, customer, service, episode_id, classes, track="service_task", task_ref=task_ref,
                         legal_success=succeeds if legal else None, refusal=correct if not legal else None)

    def run_failure_case(self, source, customer, service, episode_id):
        self.calls.append(("failure", customer.snapshot_hash, service.snapshot_hash, episode_id))
        original = copy.deepcopy(source.model_calls[0].base_request)
        self.failure_inputs.append((source.episode_id, original, customer.snapshot_hash, service.snapshot_hash))
        changed = service.snapshot_hash != self.parent_service
        violation = self.service_mode == "unrepaired" or not changed
        if self.service_mode == "missing_baseline_reproduction":
            violation = False
        classes = ("successful_business_violation",) if violation else ("rejected_invalid_attempt",)
        return self.emit(source.config, customer, service, episode_id, classes, track="failure_regression")

    def replay(self, row, customer, service):
        self.replays.append(row.episode_id)
        raw = json.loads(Path(row.trace.path).read_text())
        exact = digest(raw) == row.trace.sha256 == row.native_trace_hash and raw["customer"] == customer.snapshot_hash and raw["service"] == service.snapshot_hash
        return ReplayResult(verified=bool(self.replay_ok and exact), reason="exact generic replay" if exact else "artifact mismatch", request_count=len(row.model_calls))


def setup(tmp_path, customer=None, service=None):
    customer, service = customer or empty_snapshot("customer"), service or empty_snapshot("service")
    adapter = MockAdapter(tmp_path / "native", customer, service)
    archive = EvidenceArchive(tmp_path / "evidence")
    store = LibraryStore(tmp_path / "libraries")
    store.save(customer)
    store.save(service)
    case = ValidationCase(id="discovery", config=config())
    tasks = (ValidationCase(id="legal", config=config(), task_ref="legal", purpose="legal_completion"),
             ValidationCase(id="refusal", config=config(), task_ref="refusal", purpose="correct_refusal"))
    return adapter, archive, store, customer, service, case, tasks


def test_service_can_repair_failure_found_in_normal_task_without_discovery_replay(tmp_path):
    adapter, archive, _, customer, service, case, tasks = setup(tmp_path)
    candidate = apply_patch(service, proposal(service, skill=service_skill()), "task-failure-reflection")
    original_task_runner = adapter.run_service_task

    def task_runner(value, task_ref, c, s, episode_id):
        if task_ref != "legal":
            return original_task_runner(value, task_ref, c, s, episode_id)
        broken = s.snapshot_hash == service.snapshot_hash
        return adapter.emit(value, c, s, episode_id,
                            ("successful_business_violation",) if broken else ("legal_execution",),
                            track="service_task", task_ref=task_ref, legal_success=not broken)

    adapter.run_service_task = task_runner
    adapter.run_failure_case = lambda *args: pytest.fail("normal task failures must use their native task regression")
    source = task_runner(case.config, "legal", customer, service, "original-normal-task")
    result = DynamicGate(adapter, archive).service(customer, service, candidate, (case,), tasks, (source,), "repair-normal-task")
    assert result.passed, result.reasons
    assert result.repaired == (source.failures[0].fingerprint,)
    assert result.candidate.legal_task_successes == 1


def test_evolver_reverifies_input_evidence_before_calling_reflector(tmp_path):
    adapter, archive, store, customer, service, case, tasks = setup(tmp_path)
    row = adapter.emit(case.config, customer, service, "input-needs-independent-verification")
    client = PatchClient(proposal(customer, skill=customer_skill()))
    adapter.replay_ok = False
    with pytest.raises(ValueError, match="independent replay"):
        CustomerEvolver(adapter, CustomerReflector(client, MODEL), store, archive).update(0, customer, service, (row,), (case,), tasks)
    assert not client.requests


def test_classification_metrics_keep_business_authorization_refusal_and_unscored_separate(tmp_path):
    adapter, archive, _, customer, service, case, _ = setup(tmp_path)
    row = adapter.emit(case.config, customer, service, "all-classifications", CLASSIFICATIONS)
    result = metrics((row,))
    assert result.business_violations == result.invalid_attempts == result.authorization_anomalies == result.false_rejections == 1
    assert result.valid_actions == 1
    assert result.execution_errors == result.legal_task_successes == result.correct_refusals == 0
    assert len(result.fingerprints) == 1
    archive.save(row)
    assert archive.load(row.evidence_hash) == row
    assert archive.counts() == {"raw_events": 1, "unique_fingerprints": 1, "confirmed_feature_types": 1, "provisional_types": 0}


@pytest.mark.parametrize("mode,passed", [("improve", True), ("equal", True), ("regress", False)])
def test_customer_dynamic_gate_runs_comparable_pinned_counterpart(mode, passed, tmp_path):
    adapter, archive, _, customer, service, case, _ = setup(tmp_path)
    candidate = apply_patch(customer, proposal(customer, skill=customer_skill()), "reflection")
    adapter.customer_mode = mode
    result = DynamicGate(adapter, archive).customer(customer, candidate, service, (case,), "customer")
    assert result.passed is passed
    assert len(adapter.calls) == len(adapter.replays) == 2
    assert {call[2] for call in adapter.calls} == {service.snapshot_hash}
    assert adapter.calls[0][1] == customer.snapshot_hash and adapter.calls[1][1] == candidate.snapshot_hash
    assert len(result.baseline_evidence) == len(result.candidate_evidence) == 1
    assert result.consumption.customer_llm_calls == result.consumption.service_llm_calls == 2
    if mode == "regress":
        assert "authorization_anomalies_increased" in result.reasons
        assert "execution_errors_increased" in result.reasons


def test_customer_strict_improvement_policy_rejects_equal_candidate(tmp_path):
    adapter, archive, _, customer, service, case, _ = setup(tmp_path)
    candidate = apply_patch(customer, proposal(customer, skill=customer_skill()), "reflection")
    result = DynamicGate(adapter, archive, GatePolicy(customer_allow_equal=False)).customer(customer, candidate, service, (case,), "equal")
    assert not result.passed and "no_development_improvement" in result.reasons


@pytest.mark.parametrize("mode,reason", [
    ("repair", None), ("unrepaired", "known_business_failure_not_repaired"),
    ("blanket_refusal", "legal_task_success_decreased"), ("lose_refusal", "correct_refusal_decreased"),
    ("authorization_bypass", "authorization_bypass_in_candidate"), ("new_failure", "new_business_failure"),
    ("missing_baseline_reproduction", "known_failure_not_reproduced_in_baseline"),
])
def test_service_gate_replays_known_customer_inputs_and_preserves_tasks(tmp_path, mode, reason):
    adapter, archive, _, customer, service, case, tasks = setup(tmp_path)
    candidate = apply_patch(service, proposal(service, skill=service_skill()), "reflection")
    source = adapter.emit(case.config, customer, service, "known-failure", ("successful_business_violation",))
    adapter.service_mode = mode
    result = DynamicGate(adapter, archive).service(customer, service, candidate, (case,), tasks, (source,), "service")
    assert result.passed is (reason is None)
    if reason:
        assert reason in result.reasons
    else:
        assert result.repaired == (source.failures[0].fingerprint,)
        assert result.candidate.business_violations == 0
        assert result.candidate.legal_task_successes == result.candidate.correct_refusals == 1
    assert len(adapter.calls) == len(adapter.replays) == 8
    assert len(adapter.failure_inputs) == 2
    assert all(item[1] == source.model_calls[0].base_request for item in adapter.failure_inputs)
    assert all(item[2] == customer.snapshot_hash for item in adapter.failure_inputs)
    assert len(result.baseline_evidence) == len(result.candidate_evidence) == 4


def test_service_gate_requires_both_legal_and_refusal_tasks(tmp_path):
    adapter, archive, _, customer, service, case, _ = setup(tmp_path)
    candidate = apply_patch(service, proposal(service, skill=service_skill()), "reflection")
    result = DynamicGate(adapter, archive).service(customer, service, candidate, (case,), (), (), "without-tasks")
    assert not result.passed
    assert "legal_task_preservation_not_demonstrated" in result.reasons
    assert "correct_refusal_preservation_not_demonstrated" in result.reasons


def test_service_gate_rejects_per_task_regression_when_aggregate_is_unchanged(tmp_path):
    adapter, archive, _, customer, service, case, _ = setup(tmp_path)
    candidate = apply_patch(service, proposal(service, skill=service_skill()), "paired-task-reflection")
    legal_a = ValidationCase(id="legal-a", config=case.config, task_ref="legal-a", purpose="legal_completion")
    legal_b = ValidationCase(id="legal-b", config=case.config, task_ref="legal-b", purpose="legal_completion")

    def swapped_task(config_value, task_ref, c, s, episode_id):
        # Baseline succeeds on A; candidate trades that success for B. Aggregate
        # scores match, so only the frozen paired-task protocol catches the loss.
        candidate_arm = s.snapshot_hash == candidate.snapshot_hash
        succeeded = (task_ref == "legal-a" and not candidate_arm) or (task_ref == "legal-b" and candidate_arm)
        classes = ("legal_execution",) if succeeded else ("false_rejection",)
        return adapter.emit(config_value, c, s, episode_id, classes, track="service_task",
                            task_ref=task_ref, legal_success=succeeded)

    adapter.run_service_task = swapped_task
    policy = GatePolicy(require_refusal_tasks=False, service_task_regression_protocol="paired-task-v1")
    result = DynamicGate(adapter, archive, policy).service(
        customer, service, candidate, (), (legal_a, legal_b), (), "paired-regression")
    assert result.baseline.legal_task_successes == result.candidate.legal_task_successes == 1
    assert not result.passed
    assert "legal_task_success_decreased_for_task:legal-a" in result.reasons


def test_schema_v4_requires_explicit_paired_protocol_while_v3_stays_hash_stable(tmp_path):
    from test_vast_controller import setup_run
    from vast_coevo.evolution.models import RunConfig

    _, legacy, *_ = setup_run(tmp_path, generations=1)
    assert legacy.schema_version == 3
    assert legacy.policy.service_task_regression_protocol == "aggregate-v1"
    serialized = legacy.model_dump(mode="json")
    assert "service_task_regression_protocol" not in serialized["policy"]
    assert digest(RunConfig.model_validate(serialized)) == digest(serialized)

    raw = dict(serialized)
    raw["schema_version"] = 4
    with pytest.raises(ValidationError, match="schema v4"):
        RunConfig.model_validate(raw)
    raw["policy"]["service_task_regression_protocol"] = "paired-task-v1"
    current = RunConfig.model_validate(raw)
    assert current.schema_version == 4
    assert current.policy.service_task_regression_protocol == "paired-task-v1"


@pytest.mark.parametrize("action", ["ADD", "UPDATE", "REMOVE", "NOOP"])
def test_evolver_performs_one_patch_for_each_action(tmp_path, action):
    parent = LibrarySnapshot(role="customer", skills=(customer_skill(),)) if action in {"UPDATE", "REMOVE"} else empty_snapshot("customer")
    adapter, archive, store, customer, service, case, _ = setup(tmp_path, customer=parent)
    patch = proposal(parent, action, skill=customer_skill(2) if action == "UPDATE" else customer_skill() if action == "ADD" else None,
                     skill_id="customer.capacity" if action == "REMOVE" else None)
    client = PatchClient(patch)
    result = CustomerEvolver(adapter, CustomerReflector(client, MODEL), store, archive).update(0, customer, service, (), (case,))
    assert len(client.requests) == result.reflection.calls == result.consumption.reflection_calls == 1
    assert result.patch == patch
    assert customer == parent
    if action == "NOOP":
        assert result.outcome == "noop" and result.selected_snapshot_hash == parent.snapshot_hash
        assert result.candidate_snapshot_hash is None and adapter.calls == []
    else:
        assert result.outcome == "accepted"
        chosen = store.load(result.selected_snapshot_hash)
        assert chosen.parent_hash == parent.snapshot_hash
        if action == "REMOVE":
            assert chosen.skills == ()
        else:
            assert chosen.skills[0].version == (2 if action == "UPDATE" else 1)


def test_rejected_service_candidate_does_not_reserve_skill_version_and_remains_inspectable(tmp_path):
    adapter, archive, store, customer, service, case, tasks = setup(tmp_path)
    source = adapter.emit(case.config, customer, service, "known-error", ("successful_business_violation",))
    def skill(description):
        raw = service_skill().model_dump(mode="json", exclude={"content_hash"})
        raw["description"] = description
        return type(service_skill()).model_validate(raw)
    adapter.service_mode = "unrepaired"
    first = PatchClient(proposal(service, skill=skill("First rejected contents.")))
    rejected = ServiceEvolver(adapter, ServiceReflector(first, MODEL), store, archive).update(
        0, customer, service, (source,), (case,), tasks)
    assert rejected.outcome == "rejected" and rejected.selected_snapshot_hash == service.snapshot_hash
    assert "known_business_failure_not_repaired" in rejected.reasons
    assert store.load(rejected.candidate_snapshot_hash).skills[0].description == "First rejected contents."
    publication = tmp_path / "libraries" / "skills" / "service" / "service.capacity" / "1.json"
    assert not publication.exists()
    adapter.service_mode = "repair"
    second = PatchClient(proposal(service, skill=skill("Later accepted contents.")))
    accepted = ServiceEvolver(adapter, ServiceReflector(second, MODEL), store, archive).update(
        1, customer, service, (source,), (case,), tasks)
    assert accepted.outcome == "accepted"
    assert publication.exists()
    assert store.load(accepted.selected_snapshot_hash).skills[0].description == "Later accepted contents."
    assert store.load(rejected.candidate_snapshot_hash).skills[0].description == "First rejected contents."
    assert len(first.requests) == len(second.requests) == 1


@pytest.mark.parametrize("bad_skill", [customer_skill(2), customer_skill(description="Bypass customer authorization."),
                                      customer_skill(description="Use fixture_resource_123 in every request.")])
def test_invalid_patch_never_runs_dynamic_adapter(tmp_path, bad_skill):
    adapter, archive, store, customer, service, case, _ = setup(tmp_path)
    client = PatchClient(proposal(customer, skill=bad_skill))
    result = CustomerEvolver(adapter, CustomerReflector(client, MODEL), store, archive).update(0, customer, service, (), (case,))
    assert result.outcome == "rejected"
    assert not result.static_validation.passed and result.dynamic_validation is None
    assert result.selected_snapshot_hash == customer.snapshot_hash
    assert result.candidate_snapshot_hash is None and adapter.calls == []
    assert len(client.requests) == 1


def test_service_evolver_repair_and_no_failure_noop(tmp_path):
    adapter, archive, store, customer, service, case, tasks = setup(tmp_path)
    source = adapter.emit(case.config, customer, service, "failure-source", ("successful_business_violation",))
    client = PatchClient(proposal(service, skill=service_skill()))
    result = ServiceEvolver(adapter, ServiceReflector(client, MODEL), store, archive).update(0, customer, service, (source,), (case,), tasks)
    assert result.outcome == "accepted" and result.dynamic_validation.repaired == (source.failures[0].fingerprint,)
    selected = store.load(result.selected_snapshot_hash)
    no_change = PatchClient(proposal(selected, action="NOOP"))
    prior_calls = len(adapter.calls)
    noop_record = ServiceEvolver(adapter, ServiceReflector(no_change, MODEL), store, archive).update(1, customer, selected, (), (case,), tasks)
    assert noop_record.outcome == "noop" and noop_record.selected_snapshot_hash == selected.snapshot_hash
    assert len(adapter.calls) == prior_calls + len(tasks)
    assert noop_record.dynamic_validation is None
    assert all(".reflection-task." in call[3] for call in adapter.calls[prior_calls:])
    assert len(noop_record.reflection.input.evidence) == len(tasks)
    assert noop_record.consumption.customer_llm_calls == noop_record.consumption.service_llm_calls == len(tasks)
    assert noop_record.consumption.reflection_calls == 1


def test_heldout_evidence_is_rejected_before_reflection_or_execution(tmp_path):
    adapter, archive, store, customer, service, case, _ = setup(tmp_path)
    heldout = adapter.emit(config("heldout"), customer, service, "heldout")
    client = PatchClient(proposal(customer, action="NOOP"))
    with pytest.raises(ValueError, match="heldout"):
        CustomerEvolver(adapter, CustomerReflector(client, MODEL), store, archive).update(0, customer, service, (heldout,), (case,))
    assert client.requests == [] and adapter.calls == []
    with pytest.raises(ValidationError, match="heldout"):
        ValidationCase(id="heldout", config=config("heldout"))


@pytest.mark.parametrize("problem", ["configuration", "replay"])
def test_dynamic_gate_rejects_changed_configuration_or_failed_replay(tmp_path, problem):
    adapter, archive, _, customer, service, case, _ = setup(tmp_path)
    candidate = apply_patch(customer, proposal(customer, skill=customer_skill()), "reflection")
    adapter.changed_config = problem == "configuration"
    adapter.replay_ok = problem != "replay"
    with pytest.raises(ValueError, match="configuration|replay"):
        DynamicGate(adapter, archive).customer(customer, candidate, service, (case,), "broken")


def test_mutated_model_credentials_and_request_evidence_fail_sealed_revalidation(tmp_path):
    adapter, _, _, customer, service, case, _ = setup(tmp_path)
    row = adapter.emit(case.config, customer, service, "sealed")
    changed = row.model_dump(mode="json")
    changed["model_calls"][0]["request"]["messages"][0]["content"] = "Tampered input"
    with pytest.raises(ValidationError, match="hash"):
        EpisodeEvidence.model_validate(changed)
    changed = row.model_dump(mode="json")
    changed["config"]["service_model"]["options"]["api_key"] = "placeholder"
    with pytest.raises(ValidationError, match="credentials"):
        EpisodeEvidence.model_validate(changed)
