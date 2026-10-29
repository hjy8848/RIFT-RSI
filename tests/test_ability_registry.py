"""Offline contracts for durable Ability Registry attribution and Probe selection."""
from types import SimpleNamespace

import pytest

from rift.abilities import AbilityAttributor, AbilityRegistry, AbilityStatus
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
    trials = tuple(ObservedTrial(task_ref, f"{task_ref}-trial-{index}", checkpoint,
        index, outcome, PROTOCOL,
        None if outcome is Outcome.UNKNOWN else f"private-evidence-{task_ref}-{index}",
        outcome is not Outcome.UNKNOWN, evidence_level,
        None if outcome is Outcome.UNKNOWN else PublicSummary(
            (PublicEvent("user", text),), outcome)) for index in range(4))
    return TaskEvidenceRecord(task_ref, identity, purpose, checkpoint, PROTOCOL,
                              trials, "offline-test-fixture")


class AcceptRecordEvidence:
    def verify_record(self, record):
        return isinstance(record, TaskEvidenceRecord) and all(
            trial.valid for trial in record.trials)


def test_attribution_matches_existing_hypothesis_and_persists_verified_links(tmp_path):
    diagnosis = task_record("diagnosis-secret", DataPurpose.DIAGNOSIS,
        Split.TRAIN, "refund status lookup failed")
    recovery = task_record("recovery-secret", DataPurpose.RECOVERY_FEEDBACK,
        Split.VALIDATION, "refund status lookup still failed")
    registry = AbilityRegistry()
    ability = registry.create("refund status lookup", round_index=0)

    resolution = AbilityAttributor(record_verifier=AcceptRecordEvidence()).resolve(
        "refund status lookup", (diagnosis,), registry, round_index=1,
        failure_candidates=(recovery,))

    assert resolution.decision == "matched"
    assert resolution.ability_id == ability.ability_id
    assert set(registry.get(ability.ability_id).failure_task_refs) == {
        diagnosis.task_ref, recovery.task_ref}
    assert registry.failure_task_refs(ability.ability_id) == {
        diagnosis.task_ref, recovery.task_ref}
    assert set(resolution.evidence_refs) == {
        trial.evidence_ref for row in (diagnosis, recovery) for trial in row.trials}

    path = tmp_path / "registry.json"
    registry.save(path)
    restored = AbilityRegistry.load(path)
    assert restored.records == registry.records
    assert restored.attributions == registry.attributions
    assert restored.seen_diagnosis_identities == (diagnosis.identity,)
    path.write_text(path.read_text(encoding="utf-8").replace(
        "refund status lookup", "tampered ability"), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        AbilityRegistry.load(path)


def test_uncertain_or_ambiguous_hypotheses_create_provisional_ids():
    diagnosis = task_record("diagnosis-provisional", DataPurpose.DIAGNOSIS,
        Split.TRAIN, "transfer cancellation failed")
    registry = AbilityRegistry()
    attributor = AbilityAttributor(record_verifier=AcceptRecordEvidence())
    provisional = attributor.resolve("transfer cancellation", (diagnosis,),
        registry, round_index=0)
    assert provisional.decision == "provisional"
    assert registry.get(provisional.ability_id).status is AbilityStatus.PROVISIONAL
    assert registry.get(provisional.ability_id).failure_task_refs == (diagnosis.task_ref,)

    duplicate = registry.create("transfer cancellation", round_index=0)
    ambiguous = attributor.resolve("transfer cancellation", (diagnosis,),
        registry, round_index=1)
    assert ambiguous.decision == "provisional"
    assert ambiguous.ability_id not in {provisional.ability_id, duplicate.ability_id}


def test_attributor_only_passes_public_summaries_and_registered_descriptions():
    diagnosis = task_record("diagnosis-private", DataPurpose.DIAGNOSIS,
        Split.TRAIN, "transfer cancellation failed")

    class InspectingMatcher:
        def match(self, hypothesis, public_summaries, candidates, public_failure_summaries):
            self.inputs = repr((hypothesis, public_summaries, candidates,
                                public_failure_summaries))
            return ()

        def associate_failures(self, ability, public_failure_summaries):
            return ()

    matcher = InspectingMatcher()
    AbilityAttributor(matcher, record_verifier=AcceptRecordEvidence()).resolve(
        "transfer cancellation", (diagnosis,), AbilityRegistry(), round_index=0)
    assert "diagnosis-private" not in matcher.inputs
    assert "private-evidence" not in matcher.inputs
    assert "lineage-diagnosis-private" not in matcher.inputs
    assert "transfer cancellation failed" in matcher.inputs


def test_nonreal_failure_evidence_cannot_enter_ability_registry():
    scripted = task_record("scripted-diagnosis", DataPurpose.DIAGNOSIS,
        Split.TRAIN, "refund issue", evidence_level=EvidenceLevel.SCRIPTED_USAMI)
    with pytest.raises(ValueError, match="active Diagnosis"):
        AbilityAttributor(record_verifier=AcceptRecordEvidence()).resolve(
            "refund issue", (scripted,), AbilityRegistry(), round_index=0)


def test_recovery_prioritizes_linked_tasks_then_random_fallback_and_keeps_retention_random(tmp_path):
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
        evidence_verifier=AcceptRecordEvidence(), confidence=0.95,
        round_index=0, rationale="verified public failure evidence")

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
