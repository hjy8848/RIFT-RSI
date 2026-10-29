"""Offline contracts for provisional abilities and checkpoint-relative probes."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from rift.abilities import AbilityRegistry, AbilityStatus
from rift.abilities.records import AbilityEvidenceLink
from rift.banks.records import EvidenceLevel, ObservedTrial, TaskEvidenceRecord
from rift.benchmark.public_feedback import PublicEvent, PublicSummary
from rift.loop import Stage
from rift.probes.selection import ProbeSelector
from rift.probes.storage import JsonProbeEvidenceStore
from rift.replay.splits import SplitRegistry
from rift.schemas import (Checkpoint, DataPurpose, Identity, Outcome, Split,
    INSTANCE_DISJOINT_WITHIN_FAMILY_V1, Tokens)


BEFORE = Checkpoint("service", "10", "tokenizer", "weights-before")
AFTER = Checkpoint("service", "11", "tokenizer", "weights-after")
PROTOCOL = "fixture-protocol"


def task_record(task_ref, purpose, split, text, outcome=Outcome.FAILURE,
                checkpoint=BEFORE, evidence_level=EvidenceLevel.REAL_SERVICE):
    identity = Identity("world-" + task_ref, "task-" + task_ref, "accounts",
        "lineage-" + task_ref, split, purpose)
    outcomes = outcome if isinstance(outcome, tuple) else (outcome,) * 4
    trials = tuple(ObservedTrial(task_ref, f"{task_ref}-trial-{index}", checkpoint,
        index, trial_outcome, PROTOCOL,
        None if trial_outcome is Outcome.UNKNOWN else f"private-evidence-{task_ref}-{index}",
        trial_outcome is not Outcome.UNKNOWN, evidence_level,
        None if trial_outcome is Outcome.UNKNOWN else PublicSummary(
            (PublicEvent("user", text),), trial_outcome))
        for index, trial_outcome in enumerate(outcomes))
    return TaskEvidenceRecord(task_ref, identity, purpose, checkpoint, PROTOCOL,
                              trials, "offline-test-fixture")


class AcceptRecordEvidence:
    def verify_record(self, record):
        return isinstance(record, TaskEvidenceRecord) and all(
            trial.valid for trial in record.trials)


def test_hypotheses_remain_provisional_without_failure_attribution(tmp_path):
    diagnosis = task_record("diagnosis-secret", DataPurpose.DIAGNOSIS,
        Split.TRAIN, "tool returned EVIDENCE_REQUIRED")
    registry = AbilityRegistry()
    registry.mark_diagnosis_seen(diagnosis.task_ref, diagnosis.identity)

    first = registry.create("missing prerequisite checks", round_index=0)
    same_words_next_round = registry.create("missing prerequisite checks", round_index=1)

    assert first.ability_id != same_words_next_round.ability_id
    assert first.status is AbilityStatus.PROVISIONAL
    assert first.failure_task_refs == ()
    assert registry.failure_task_refs(first.ability_id) == frozenset()
    assert registry.seen_diagnosis_identities == (diagnosis.identity,)

    path = tmp_path / "registry.json"
    registry.save(path)
    assert AbilityRegistry.load(path).records == registry.records
    path.write_text(path.read_text(encoding="utf-8").replace(
        "missing prerequisite checks", "tampered ability"), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        AbilityRegistry.load(path)


def test_explicit_evidence_link_accepts_uncertain_failure_history():
    uncertain = task_record("uncertain-recovery", DataPurpose.RECOVERY_FEEDBACK,
        Split.VALIDATION, "consent check was missing",
        (Outcome.FAILURE, Outcome.FAILURE, Outcome.SUCCESS, Outcome.SUCCESS))
    assert uncertain.state().value == "UNCERTAIN"
    registry = AbilityRegistry()
    ability = registry.create("check authorization before action", round_index=0)
    failure_refs = tuple(trial.evidence_ref for trial in uncertain.trials
                         if trial.outcome is Outcome.FAILURE)

    registry.link_failure_task(ability.ability_id, uncertain, failure_refs,
        evidence_verifier=AcceptRecordEvidence(), round_index=1)

    assert registry.failure_task_refs(ability.ability_id) == {uncertain.task_ref}
    link = registry.get(ability.ability_id).failure_links[0]
    assert link.evidence_refs == failure_refs
    assert link.confidence is None


def test_legacy_confidence_link_is_not_used_for_ability_priority():
    registry = AbilityRegistry()
    ability = registry.create("check authorization", round_index=0)
    legacy = replace(ability, failure_task_refs=("legacy-task",),
        failure_links=(AbilityEvidenceLink("legacy-task", ("legacy-evidence",),
            1.0, 0, "old string match"),))
    registry = AbilityRegistry((legacy,))

    assert registry.get(ability.ability_id).failure_task_refs == ("legacy-task",)
    assert registry.failure_task_refs(ability.ability_id) == frozenset()


def test_scripted_failure_cannot_be_linked_to_an_ability():
    scripted = task_record("scripted-recovery", DataPurpose.RECOVERY_FEEDBACK,
        Split.VALIDATION, "refund issue", evidence_level=EvidenceLevel.SCRIPTED_USAMI)
    registry = AbilityRegistry()
    ability = registry.create("refund handling", round_index=0)
    refs = tuple(trial.evidence_ref for trial in scripted.trials)

    with pytest.raises(ValueError, match="real Service failure"):
        registry.link_failure_task(ability.ability_id, scripted, refs,
            evidence_verifier=AcceptRecordEvidence(), round_index=1)


def test_recovery_prioritizes_explicit_links_then_random_fallback(tmp_path):
    diagnosis = task_record("seen-diagnosis", DataPurpose.DIAGNOSIS,
        Split.TRAIN, "refund handling failed")
    linked = task_record("linked-recovery", DataPurpose.RECOVERY_FEEDBACK,
        Split.VALIDATION, "refund handling failed", checkpoint=BEFORE)
    fallback = task_record("fallback-recovery", DataPurpose.RECOVERY_FEEDBACK,
        Split.VALIDATION, "unrelated issue failed", checkpoint=BEFORE)
    retention = [task_record(f"retention-{index}", DataPurpose.RETENTION_FEEDBACK,
        Split.VALIDATION, "refund resolved", Outcome.SUCCESS, BEFORE)
        for index in range(5)]
    split_registry = SplitRegistry(
        [diagnosis.identity, linked.identity, fallback.identity,
         *(row.identity for row in retention)],
        policy_version=INSTANCE_DISJOINT_WITHIN_FAMILY_V1)
    train_identity = Identity("world-train", "task-train", "accounts",
        "lineage-train", Split.TRAIN, DataPurpose.TRAIN)
    split_registry.register(train_identity)
    registry = AbilityRegistry()
    ability = registry.create("refund handling", round_index=0)
    registry.link_failure_task(ability.ability_id, linked,
        tuple(trial.evidence_ref for trial in linked.trials),
        evidence_verifier=AcceptRecordEvidence(), round_index=0)

    selector = ProbeSelector(record_verifier=AcceptRecordEvidence())
    common = dict(recovery_candidates=(linked, fallback),
        retention_candidates=retention, diagnosis_identities=(diagnosis.identity,),
        train_package_identities=(train_identity,), split_registry=split_registry,
        before_checkpoint=BEFORE, recovery_tasks=2, retention_tasks=2, seed=29,
        protocol_ref="protocol-v1", protocol_digest=PROTOCOL,
        ability_task_refs=registry.failure_task_refs(ability.ability_id))
    recovery, kept = selector.freeze(**common, target_ability_id=ability.ability_id)
    other_target, kept_other = selector.freeze(**common, target_ability_id="A999999")

    selected = {probe.task_ref for probe in recovery.probes}
    assert selected == {linked.task_ref, fallback.task_ref}
    assert recovery.priority_task_refs == (linked.task_ref,)
    assert recovery.target_ability_task_count == 1
    assert recovery.random_fallback_task_count == 1
    assert recovery.target_ability_id == ability.ability_id
    assert kept == kept_other
    assert len(selected) == len(recovery.probes)
    assert all(trial.public_summary is None for probe in recovery.probes
               for trial in probe.baseline_trials)

    evidence = JsonProbeEvidenceStore(tmp_path / "probe-evidence")
    evidence.save_snapshot(recovery)
    assert evidence.load_snapshot(recovery.content_digest) == recovery


def test_historical_failure_uses_new_checkpoint_baseline_for_recovery_and_retention():
    old_failure = task_record("old-failure", DataPurpose.RECOVERY_FEEDBACK,
        Split.VALIDATION, "authorization not checked", checkpoint=BEFORE)
    current_recovery = task_record("old-failure", DataPurpose.RECOVERY_FEEDBACK,
        Split.VALIDATION, "authorization now checked", Outcome.SUCCESS,
        checkpoint=AFTER)
    old_mastery = task_record("old-mastery", DataPurpose.RETENTION_FEEDBACK,
        Split.VALIDATION, "invoice reconciled", Outcome.SUCCESS, checkpoint=BEFORE)
    current_retention = task_record("old-mastery", DataPurpose.RETENTION_FEEDBACK,
        Split.VALIDATION, "invoice still reconciled", Outcome.SUCCESS,
        checkpoint=AFTER)
    split_registry = SplitRegistry(
        [old_failure.identity, old_mastery.identity],
        policy_version=INSTANCE_DISJOINT_WITHIN_FAMILY_V1)
    ability_registry = AbilityRegistry()
    ability = ability_registry.create("check authorization", round_index=0)
    ability_registry.link_failure_task(ability.ability_id, old_failure,
        tuple(trial.evidence_ref for trial in old_failure.trials),
        evidence_verifier=AcceptRecordEvidence(), round_index=0)
    selector = ProbeSelector(record_verifier=AcceptRecordEvidence())
    args = dict(recovery_candidates=(old_failure,), retention_candidates=(old_mastery,),
        diagnosis_identities=(), train_package_identities=(), split_registry=split_registry,
        before_checkpoint=AFTER, recovery_tasks=1, retention_tasks=1, seed=13,
        protocol_ref="protocol-v2", protocol_digest=PROTOCOL,
        target_ability_id=ability.ability_id,
        ability_task_refs=ability_registry.failure_task_refs(ability.ability_id),
        current_baseline_candidates=(current_recovery, current_retention))

    recovery, retention = selector.freeze(**args)

    assert recovery.baseline_checkpoint == retention.baseline_checkpoint == AFTER
    assert recovery.priority_task_refs == (old_failure.task_ref,)
    assert recovery.probes[0].task_ref == old_failure.task_ref
    assert all(trial.outcome is Outcome.SUCCESS
               for trial in recovery.probes[0].baseline_trials)
    assert retention.probes[0].task_ref == old_mastery.task_ref
    assert all(trial.checkpoint == AFTER for probe in recovery.probes + retention.probes
               for trial in probe.baseline_trials)

    with pytest.raises(ValueError, match="insufficient eligible"):
        selector.freeze(**{**args, "current_baseline_candidates": ()})


def test_ability_training_and_evaluation_history_is_idempotent_and_restorable(tmp_path):
    registry = AbilityRegistry()
    ability = registry.create("invoice reconciliation", round_index=0)
    state = SimpleNamespace(stage=Stage.DESIGNER_TRAINING_COMPLETED,
        target_ability_id=ability.ability_id, round_index=1, package_id="pkg-1",
        service_update=SimpleNamespace(run_id="service-update-1", policy_after=AFTER),
        learning_reward=SimpleNamespace(value=0.4, recovery_before_sr=0.2,
            recovery_after_sr=0.8, retention_before_sr=0.9, retention_after_sr=0.85),
        recovery_snapshot=SimpleNamespace(baseline_checkpoint=BEFORE,
            probes=(SimpleNamespace(task_ref="probe-task"),),
            priority_task_refs=("probe-task",)),
        recovery_before=SimpleNamespace(results=(SimpleNamespace(
            task_ref="probe-task", success_rate=0.2),)),
        recovery_after=SimpleNamespace(results=(SimpleNamespace(
            task_ref="probe-task", success_rate=0.8,
            evidence_refs=("after-evidence",)),)),
        retention_snapshot=SimpleNamespace(
            probes=(SimpleNamespace(task_ref="retention-task"),)),
        retention_after=SimpleNamespace(results=(
            SimpleNamespace(evidence_refs=("retention-after-evidence",)),)))

    class Verified:
        def verify_update(self, _): return True
        def verify_reward(self, *_): return True

    first = registry.record_round(state, update_verifier=Verified(),
                                  reward_verifier=Verified())
    second = registry.record_round(state, update_verifier=Verified(),
                                   reward_verifier=Verified())
    assert first == second
    assert first.status is AbilityStatus.STABLE
    assert len(first.training_history) == len(first.evaluation_history) == 1
    assert first.evaluation_history[0].checkpoint_before == BEFORE
    assert first.evaluation_history[0].retention_task_refs == ("retention-task",)
    assert first.evaluation_history[0].target_recovery_before_sr == pytest.approx(0.2)
    assert first.evaluation_history[0].target_recovery_after_sr == pytest.approx(0.8)
    path = tmp_path / "history.json"
    registry.save(path)
    assert AbilityRegistry.load(path).get(ability.ability_id) == first


def test_loop_state_persists_directed_designer_ability_context(tmp_path):
    from rift.loop import LoopState, PendingDesignerAction
    state = LoopState("package-targeted", ("proposal-1",),
        target_ability_id="A000001",
        designer_action=PendingDesignerAction(Checkpoint("designer", "2",
            "tokenizer", "designer-weights"), "a" * 32,
            Tokens((1, 2), (3,), (True,)), None, "A000001"))
    path = tmp_path / "loop-state.json"
    state.save(path)

    class ActionVerifier:
        def verify_designer_action(self, tokens, checkpoint, package_id,
                                   rollout_id, group_id=None, *, target_ability_id=None):
            return (tokens == state.designer_action.tokens
                    and checkpoint == state.designer_action.checkpoint
                    and package_id == state.package_id
                    and target_ability_id == "A000001")

    restored = LoopState.load(path, action_verifier=ActionVerifier())
    assert restored == state
