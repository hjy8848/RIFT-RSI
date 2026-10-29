"""Explicit Designer ability choices and shared Failure Bank evaluation."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from rift.abilities import AbilityRegistry, AbilityStatus
from rift.banks.core import BankManager, public_projection
from rift.banks.records import EvidenceLevel, ObservedTrial, TaskEvidenceRecord
from rift.benchmark.public_feedback import PublicEvent, PublicSummary
from rift.loop import Stage
from rift.probes.selection import ProbeSelector
from rift.replay.splits import SplitRegistry
from rift.schemas import (Checkpoint, DataPurpose, Identity, Outcome, Split,
    INSTANCE_DISJOINT_WITHIN_FAMILY_V1)
from rift.tasks import Proposal


BEFORE = Checkpoint("service", "10", "tokenizer", "weights-before")
AFTER = Checkpoint("service", "11", "tokenizer", "weights-after")
PROTOCOL = "offline-protocol"


def task_record(task_ref, outcome=Outcome.FAILURE, checkpoint=BEFORE,
                split=Split.TRAIN, evidence_level=EvidenceLevel.REAL_SERVICE):
    identity = Identity("world-" + task_ref, "task-" + task_ref, "accounts",
        "lineage-" + task_ref, split, DataPurpose.BANK_CANDIDATE)
    outcomes = outcome if isinstance(outcome, tuple) else (outcome,) * 4
    trials = tuple(ObservedTrial(task_ref, f"{task_ref}-{index}", checkpoint, index,
        result, PROTOCOL, None if result is Outcome.UNKNOWN else f"evidence-{task_ref}-{index}",
        result is not Outcome.UNKNOWN, evidence_level,
        None if result is Outcome.UNKNOWN else PublicSummary(
            (PublicEvent("tool", "returned EVIDENCE_REQUIRED"),), result))
        for index, result in enumerate(outcomes))
    return TaskEvidenceRecord(task_ref, identity, checkpoint, PROTOCOL, trials, "fixture")


class RealEvidence:
    def verify_record(self, record):
        return (isinstance(record, TaskEvidenceRecord)
            and all(not row.valid or row.evidence_level is EvidenceLevel.REAL_SERVICE
                    for row in record.trials))


def test_designer_creates_ability_and_explicitly_links_failure_task():
    failure = task_record("F001")
    banks = BankManager()
    banks.observe(failure, event_id="bootstrap-F001")
    visible = (failure,)
    proposal = Proposal("p1", "check prerequisites before acting", "refund request", "{}",
        "NEW", ("F001",))
    registry = AbilityRegistry()

    created = registry.apply_designer_choice(ability_id=proposal.ability_id,
        hypothesis=proposal.hypothesis, add_failure_task_refs=proposal.diagnosis_task_refs,
        visible_failures=visible, evidence_verifier=banks, round_index=0,
        designer_rollout_id="rollout-created-A1")
    assert registry.apply_designer_choice(ability_id=proposal.ability_id,
        hypothesis=proposal.hypothesis, add_failure_task_refs=proposal.diagnosis_task_refs,
        visible_failures=visible, evidence_verifier=banks, round_index=0,
        designer_rollout_id="rollout-created-A1") == created

    assert created.ability_id == "A000001"
    assert created.failure_task_refs == ("F001",)
    assert registry.prompts()[0].failure_task_refs == ("F001",)
    assert public_projection(failure).task_ref == "F001"


def test_next_round_reuses_ability_and_seen_failure_is_recovery_candidate():
    f001, f002 = task_record("F001"), task_record("F002")
    histories = (f001, f002)
    banks = BankManager()
    for row in histories:
        banks.observe(row, event_id="bootstrap-" + row.task_ref)
    registry = AbilityRegistry()
    first = registry.apply_designer_choice(ability_id="NEW",
        hypothesis="check prerequisites", add_failure_task_refs=("F001",),
        visible_failures=histories, evidence_verifier=banks, round_index=0,
        designer_rollout_id="round-zero")
    # The same TaskRef remains in the Designer's visible history next round.
    second = registry.apply_designer_choice(ability_id=first.ability_id,
        hypothesis=first.description, add_failure_task_refs=(),
        visible_failures=histories, evidence_verifier=banks, round_index=1,
        designer_rollout_id="round-one")
    assert second.ability_id == first.ability_id
    assert second.failure_task_refs == ("F001",)

    retention = tuple(task_record(f"S{index}", Outcome.SUCCESS,
        split=Split.VALIDATION) for index in range(2))
    current = tuple(task_record(row.task_ref, Outcome.SUCCESS, checkpoint=AFTER,
        split=row.identity.split) for row in (f001, f002, *retention))
    split_registry = SplitRegistry((row.identity for row in (*histories, *retention)),
        policy_version=INSTANCE_DISJOINT_WITHIN_FAMILY_V1)
    selector = ProbeSelector(record_verifier=RealEvidence())
    args = dict(recovery_candidates=histories, retention_candidates=retention,
        train_package_identities=(), split_registry=split_registry,
        before_checkpoint=AFTER, recovery_tasks=1, retention_tasks=1, seed=73,
        protocol_ref="protocol", protocol_digest=PROTOCOL,
        target_ability_id=second.ability_id, ability_task_refs=second.failure_task_refs,
        current_baseline_candidates=current)

    recovery_a, retention_a = selector.freeze(**args)
    recovery_b, retention_b = selector.freeze(**args)
    assert recovery_a.probes[0].task_ref == "F001"
    assert recovery_a.priority_task_refs == ("F001",)
    assert recovery_a.content_digest == recovery_b.content_digest
    assert retention_a.content_digest == retention_b.content_digest
    assert recovery_a.probes[0].baseline_checkpoint == AFTER

    # The same failure cannot score the update that trained on it; random fill uses F002.
    training_identity = replace(f001.identity, purpose=DataPurpose.TRAIN)
    split_registry.register(training_identity)
    recovery_without_train = selector.freeze(**{**args,
        "train_package_identities": (training_identity,)})[0]
    assert recovery_without_train.probes[0].task_ref == "F002"
    assert recovery_without_train.priority_task_refs == ()

    revised = registry.apply_designer_choice(ability_id=second.ability_id,
        hypothesis="verify prerequisites before acting", add_failure_task_refs=(),
        remove_failure_task_refs=("F001",), visible_failures=histories,
        evidence_verifier=banks, round_index=2, designer_rollout_id="round-two")
    assert revised.failure_task_refs == ()
    assert revised.revisions[-1].removed_failure_task_refs == ("F001",)


def test_explicit_choice_rejects_unknown_id_unseen_ref_and_nonreal_failure():
    failure = task_record("F001")
    banks = BankManager()
    banks.observe(failure, event_id="bootstrap-F001")
    registry = AbilityRegistry()
    with pytest.raises(KeyError, match="unknown ability"):
        registry.apply_designer_choice(ability_id="A999999", hypothesis="gap",
            add_failure_task_refs=(), visible_failures=(failure,),
            evidence_verifier=banks, round_index=0, designer_rollout_id="unknown")
    with pytest.raises(ValueError, match="visible"):
        registry.apply_designer_choice(ability_id="NEW", hypothesis="gap",
            add_failure_task_refs=("not-shown",), visible_failures=(failure,),
            evidence_verifier=banks, round_index=0, designer_rollout_id="unseen")
    scripted = task_record("scripted", evidence_level=EvidenceLevel.SCRIPTED_USAMI)
    scripted_banks = BankManager()
    scripted_banks.observe(scripted, event_id="scripted")
    with pytest.raises(ValueError, match="real Service"):
        registry.apply_designer_choice(ability_id="NEW", hypothesis="gap",
            add_failure_task_refs=("scripted",), visible_failures=(scripted,),
            evidence_verifier=scripted_banks, round_index=0, designer_rollout_id="scripted")


def test_final_test_never_enters_bank_or_probe():
    test_task = task_record("T-TEST", split=Split.TEST)
    from rift.banks.bootstrap import BootstrapTask
    with pytest.raises(ValueError, match="non-TEST"):
        BootstrapTask("T-TEST", test_task.identity)
    with pytest.raises(ValueError, match="TEST"):
        BankManager().observe(test_task, event_id="test")

    other = task_record("S001", Outcome.SUCCESS, split=Split.VALIDATION)
    current_test = task_record("T-TEST", checkpoint=AFTER, split=Split.TEST)
    split_registry = SplitRegistry((test_task.identity, other.identity),
        policy_version=INSTANCE_DISJOINT_WITHIN_FAMILY_V1)
    with pytest.raises(ValueError, match="insufficient eligible"):
        ProbeSelector(record_verifier=RealEvidence()).freeze(
            recovery_candidates=(test_task,), retention_candidates=(other,),
            train_package_identities=(), split_registry=split_registry,
            before_checkpoint=AFTER, recovery_tasks=1, retention_tasks=1, seed=1,
            protocol_ref="protocol", protocol_digest=PROTOCOL,
            current_baseline_candidates=(current_test, other))


def test_ability_training_history_is_idempotent_and_restorable(tmp_path):
    registry = AbilityRegistry()
    ability = registry.create("invoice reconciliation", round_index=0)
    state = SimpleNamespace(stage=Stage.DESIGNER_TRAINING_COMPLETED,
        target_ability_id=ability.ability_id, round_index=1, package_id="pkg-1",
        service_update=SimpleNamespace(run_id="service-update-1", policy_after=AFTER),
        learning_reward=SimpleNamespace(value=0.4, recovery_before_sr=0.2,
            recovery_after_sr=0.8, retention_before_sr=0.9, retention_after_sr=0.85),
        recovery_snapshot=SimpleNamespace(baseline_checkpoint=BEFORE,
            probes=(SimpleNamespace(task_ref="probe-task"),), priority_task_refs=("probe-task",)),
        recovery_before=SimpleNamespace(results=(SimpleNamespace(task_ref="probe-task", success_rate=0.2),)),
        recovery_after=SimpleNamespace(results=(SimpleNamespace(task_ref="probe-task", success_rate=0.8,
            evidence_refs=("after-evidence",)),)),
        retention_snapshot=SimpleNamespace(probes=(SimpleNamespace(task_ref="retention-task"),)),
        retention_after=SimpleNamespace(results=(SimpleNamespace(evidence_refs=("retention-evidence",)),)))

    class Verified:
        def verify_update(self, _): return True
        def verify_reward(self, *_): return True

    first = registry.record_round(state, update_verifier=Verified(), reward_verifier=Verified())
    assert registry.record_round(state, update_verifier=Verified(), reward_verifier=Verified()) == first
    assert first.status is AbilityStatus.STABLE
    assert len(first.training_history) == len(first.evaluation_history) == 1
    path = tmp_path / "registry.json"
    registry.save(path)
    assert AbilityRegistry.load(path).get(ability.ability_id) == first


def test_loop_state_restores_action_without_host_selected_ability(tmp_path):
    from rift.loop import LoopState, PendingDesignerAction
    from rift.schemas import Tokens
    state = LoopState("package", ("proposal",), target_ability_id="A000001",
        designer_action=PendingDesignerAction(Checkpoint("designer", "2", "tokenizer", "weights"),
            "a" * 32, Tokens((1, 2), (3,), (True,))))
    path = tmp_path / "loop.json"
    state.save(path)

    class ActionVerifier:
        def verify_designer_action(self, tokens, checkpoint, package_id, rollout_id, group_id=None):
            return tokens == state.designer_action.tokens and package_id == state.package_id

    assert LoopState.load(path, action_verifier=ActionVerifier()) == state
