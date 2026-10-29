"""Offline dual-bank protocol tests; scripted and fake records are not model evidence."""
from dataclasses import replace
import json

import pytest

from rift.banks.bootstrap import BootstrapRunner, BootstrapTask
from rift.banks.core import BankManager, JsonTaskEvidenceStore, public_projection
from rift.banks.records import (CapabilityState, EvidenceLevel, ObservedTrial,
    TaskEvidenceRecord)
from rift.benchmark.public_feedback import PublicEvent, PublicSummary
from rift.loop import LoopState, PendingDesignerAction, Stage
from rift.probes.records import ProbeSnapshot
from rift.probes.runner import ProbeRunner
from rift.probes.selection import ProbeSelector
from rift.probes.storage import JsonProbeEvidenceStore
from rift.rewards.learning_value import (DualBankOutcomeReward,
    InsufficientEvidenceError, dual_bank_formula, score_dual_bank)
from rift.schemas import (Checkpoint, DataPurpose, Identity, Outcome, Split,
    STRICT_FAMILY_DISJOINT_V1, INSTANCE_DISJOINT_WITHIN_FAMILY_V1, Tokens)
from rift.replay.splits import SplitRegistry
from rift.tasks import Certification, CertificationStatus
from rift.training.contracts import CompletedUpdate, UpdateRequest
from rift.training.records import (DesignerFeedback, comparable_designer_rollouts,
    prepare_designer_update, service_rollout)
from rift.verl.coordinator import prepare_dual_bank_round


BEFORE = Checkpoint('service', '10', 'tokenizer-v1', 'weights-before')
AFTER = Checkpoint('service', '11', 'tokenizer-v1', 'weights-after')
DESIGNER = Checkpoint('designer', '4', 'tokenizer-v1', 'designer-weights')
ACTION = Tokens((1, 2), (3, 4), (True, True))
PROTOCOL = 'protocol-digest-v1'


def identity(name, purpose, split):
    return Identity(f'world-{name}', f'task-{name}', 'refund-capability',
                    f'source-{name}', split, purpose)


class ScriptedBootstrapEvaluator:
    evidence_level = EvidenceLevel.SCRIPTED_USAMI

    def __init__(self, outcomes, purpose_by_task):
        self.outcomes, self.purpose_by_task, self.counts = outcomes, purpose_by_task, {}

    def evaluate(self, task_ref, checkpoint, seed, protocol_digest):
        index = self.counts.get(task_ref, 0)
        self.counts[task_ref] = index + 1
        outcome = self.outcomes[task_ref][index]
        summary = None
        if self.purpose_by_task[task_ref] is DataPurpose.DIAGNOSIS:
            summary = PublicSummary((PublicEvent('user', '公开退款申请'),), outcome)
        return ObservedTrial(task_ref, f'{task_ref}-trial-{index}', checkpoint, seed,
            outcome, protocol_digest, None if outcome is Outcome.UNKNOWN else f'scripted:{task_ref}:{index}',
            outcome is not Outcome.UNKNOWN, self.evidence_level, summary)


class AcceptBootstrapEvidence:
    def verify_trial(self, trial, task):
        return trial.task_ref == task.task_ref and trial.evidence_level is EvidenceLevel.SCRIPTED_USAMI


def make_bootstrap(tmp_path):
    refs = {
        'diag': (DataPurpose.DIAGNOSIS, Split.TRAIN, [Outcome.FAILURE] * 4),
        'recovery-a': (DataPurpose.RECOVERY_FEEDBACK, Split.VALIDATION,
                       [Outcome.SUCCESS, Outcome.FAILURE, Outcome.FAILURE, Outcome.FAILURE]),
        'recovery-b': (DataPurpose.RECOVERY_FEEDBACK, Split.VALIDATION, [Outcome.FAILURE] * 4),
        'recovery-c': (DataPurpose.RECOVERY_FEEDBACK, Split.VALIDATION,
                       [Outcome.SUCCESS, Outcome.FAILURE, Outcome.FAILURE, Outcome.FAILURE]),
        'retention-a': (DataPurpose.RETENTION_FEEDBACK, Split.VALIDATION,
                        [Outcome.SUCCESS] * 3 + [Outcome.FAILURE]),
        'retention-b': (DataPurpose.RETENTION_FEEDBACK, Split.VALIDATION, [Outcome.SUCCESS] * 4),
        'retention-c': (DataPurpose.RETENTION_FEEDBACK, Split.VALIDATION, [Outcome.SUCCESS] * 4),
        'unknown': (DataPurpose.DIAGNOSIS, Split.TRAIN, [Outcome.UNKNOWN] * 4),
    }
    manifest = tuple(BootstrapTask(key, identity(key, purpose, split), purpose)
                     for key, (purpose, split, _) in refs.items())
    evaluator = ScriptedBootstrapEvaluator({key: outcomes for key, (_, _, outcomes) in refs.items()},
        {key: purpose for key, (purpose, _, _) in refs.items()})
    result = BootstrapRunner(evaluator, AcceptBootstrapEvidence()).run(
        manifest, BEFORE, protocol={'customer': 'fixed', 'budget': 5}, trials_per_task=4, seed=17)
    store = JsonTaskEvidenceStore(tmp_path / 'task-store')
    for row in result.records:
        store.save(row)
        assert store.load(row.task_ref, row.checkpoint) == row
    return result


class AcceptRecordEvidence:
    def verify_record(self, record):
        return isinstance(record, TaskEvidenceRecord) and bool(record.trials)


class AcceptProbeEvidence:
    def verify_probe_trial(self, trial, probe, checkpoint):
        return trial.task_ref == probe.task_ref and trial.checkpoint == checkpoint

    def verify_snapshot(self, snapshot):
        return snapshot.verify()

    def verify_result(self, result, snapshot):
        return result.snapshot_digest == snapshot.content_digest

    def verify_frozen_before(self, snapshot, update):
        return snapshot.verify()


class ScriptedAfterEvaluator:
    def evaluate(self, task_ref, checkpoint, seed, protocol_ref, protocol_digest):
        return ObservedTrial(task_ref, f'after-{task_ref}-{seed}', checkpoint, seed,
            Outcome.SUCCESS, protocol_digest, f'scripted-after:{task_ref}:{seed}', True,
            EvidenceLevel.SCRIPTED_USAMI)


def test_bootstrap_keeps_unknown_separate_and_diagnosis_is_allowlisted(tmp_path):
    result = make_bootstrap(tmp_path)
    by_ref = {row.task_ref: row for row in result.records}
    assert by_ref['diag'].state() is CapabilityState.ACTIVE_WEAKNESS
    assert by_ref['retention-a'].state() is CapabilityState.MASTERED_CANDIDATE
    assert by_ref['unknown'].state() is CapabilityState.UNRESOLVED
    assert by_ref['unknown'].success_rate is None
    assert by_ref['unknown'] not in result.bank_manager.failure_bank.records
    diagnosis = public_projection(by_ref['diag'])
    assert 'diag' not in repr(diagnosis) and 'source-diag' not in repr(diagnosis)
    assert 'evidence_ref' not in repr(diagnosis) and 'strict_task_success' not in repr(diagnosis)
    assert diagnosis.summary.outcome is Outcome.FAILURE
    assert all(row.evidence_levels == {EvidenceLevel.SCRIPTED_USAMI} for row in result.records)
    path = tmp_path / 'banks.json'
    result.bank_manager.save(path)
    restored = BankManager.load(path)
    assert restored.records == result.bank_manager.records
    assert restored.events == result.bank_manager.events
    path.write_text(path.read_text(encoding='utf-8').replace('refund-capability', 'tampered'), encoding='utf-8')
    with pytest.raises(ValueError, match='digest'):
        BankManager.load(path)


def test_versioned_split_policy_preserves_legacy_and_allows_same_family_instances(tmp_path):
    training = identity('train', DataPurpose.TRAIN, Split.TRAIN)
    feedback = identity('feedback', DataPurpose.RECOVERY_FEEDBACK, Split.VALIDATION)
    with pytest.raises(ValueError, match='overlap'):
        SplitRegistry((training, feedback), policy_version=STRICT_FAMILY_DISJOINT_V1)
    registry = SplitRegistry((training, feedback),
        policy_version=INSTANCE_DISJOINT_WITHIN_FAMILY_V1)
    assert training.family == feedback.family and training.task != feedback.task
    same_instance_other_purpose = Identity(training.world, training.task, training.family,
        training.lineage, Split.VALIDATION, DataPurpose.RECOVERY_FEEDBACK)
    with pytest.raises(ValueError, match='overlap'):
        SplitRegistry((training, same_instance_other_purpose),
            policy_version=INSTANCE_DISJOINT_WITHIN_FAMILY_V1)
    path = tmp_path / 'split-registry.json'
    registry.save(path)
    assert SplitRegistry.load(path).policy_version == INSTANCE_DISJOINT_WITHIN_FAMILY_V1
    legacy = tmp_path / 'legacy-splits.json'
    legacy.write_text(json.dumps([{'world':'w','task':'t','family':'f','lineage':'l','split':'train'}]), encoding='utf-8')
    assert SplitRegistry.load(legacy).policy_version == STRICT_FAMILY_DISJOINT_V1


def test_random_probe_freeze_is_reproducible_private_and_instance_disjoint(tmp_path):
    bootstrap = make_bootstrap(tmp_path)
    registry = SplitRegistry((row.identity for row in bootstrap.records),
        policy_version=INSTANCE_DISJOINT_WITHIN_FAMILY_V1)
    training = identity('new-training', DataPurpose.TRAIN, Split.TRAIN)
    registry.register(training)
    rows = {row.task_ref: row for row in bootstrap.records}
    selector = ProbeSelector(record_verifier=AcceptRecordEvidence(), allow_nonreal=True)
    args = dict(recovery_candidates=[rows['recovery-a'], rows['recovery-b'], rows['recovery-c']],
        retention_candidates=[rows['retention-a'], rows['retention-b'], rows['retention-c']],
        diagnosis_identities=(rows['diag'].identity,), train_package_identities=(training,),
        split_registry=registry, before_checkpoint=BEFORE, recovery_tasks=1,
        retention_tasks=1, seed=0, protocol_ref='protocol-v1', protocol_digest=bootstrap.protocol_digest)
    recovery1, retention1 = selector.freeze(**args, frozen_at_ns=1)
    recovery2, retention2 = selector.freeze(**args, frozen_at_ns=99)
    assert recovery1.content_digest == recovery2.content_digest
    assert retention1.content_digest == retention2.content_digest
    assert recovery1.random_seed == retention1.random_seed == 0
    assert recovery1.probes[0].identity.family == training.family
    assert recovery1.probes[0].task_ref not in repr(public_projection(rows['diag']))
    with pytest.raises(ValueError, match='insufficient'):
        selector.freeze(**{**args, 'diagnosis_identities':
            (rows['diag'].identity, rows['recovery-a'].identity,
             rows['recovery-b'].identity, rows['recovery-c'].identity)})
    assert all(row.purpose is DataPurpose.RECOVERY_FEEDBACK for row in recovery1.probes)
    store = JsonProbeEvidenceStore(tmp_path / 'probe-store')
    store.save_snapshot(recovery1); store.save_snapshot(retention1)
    assert store.load_snapshot(recovery1.content_digest) == recovery1


def test_scripted_probe_runner_persists_trials_but_cannot_claim_learning_reward(tmp_path):
    bootstrap = make_bootstrap(tmp_path)
    registry = SplitRegistry((row.identity for row in bootstrap.records),
        policy_version=INSTANCE_DISJOINT_WITHIN_FAMILY_V1)
    training = identity('service-training', DataPurpose.TRAIN, Split.TRAIN)
    registry.register(training)
    records = {row.task_ref: row for row in bootstrap.records}
    recovery, retention = ProbeSelector(record_verifier=AcceptRecordEvidence(),
        allow_nonreal=True).freeze(recovery_candidates=[records['recovery-a'], records['recovery-b']],
        retention_candidates=[records['retention-a'], records['retention-b']],
        diagnosis_identities=(records['diag'].identity,), train_package_identities=(training,),
        split_registry=registry, before_checkpoint=BEFORE, recovery_tasks=1,
        retention_tasks=1, seed=3, protocol_ref='protocol-v1', protocol_digest=bootstrap.protocol_digest)
    runner = ProbeRunner(ScriptedAfterEvaluator(), AcceptProbeEvidence())
    recovery_before, retention_before = runner.baseline(recovery), runner.baseline(retention)
    recovery_after, retention_after = runner.after(recovery, AFTER), runner.after(retention, AFTER)
    store = JsonProbeEvidenceStore(tmp_path / 'evidence-store')
    for snapshot in (recovery, retention): store.save_snapshot(snapshot)
    for result in (recovery_before, retention_before, recovery_after, retention_after):
        assert store.load_results(store.save_results(result)) == result
        assert all(trial.evidence_level is EvidenceLevel.SCRIPTED_USAMI
                   for task_result in result.results for trial in task_result.trials)

    tokens = Tokens((1,), (2,), (True,))
    rollout = service_rollout(tokens, BEFORE, training.task, 'service-episode',
        Outcome.SUCCESS, response_id='service-response')
    update = CompletedUpdate(UpdateRequest(BEFORE, 'service-package', (rollout,), 'policy_gradient'),
                             AFTER, 'service-update', 'mock-verification-reference')

    class Verifier:
        def verify_update(self, _): return True
        def verify_frozen_before(self, snapshot, _): return snapshot.verify()
        def verify_probe_trial(self, trial, probe, checkpoint): return trial.task_ref == probe.task_ref
    class Checkpoints:
        def verify(self, _): return True

    with pytest.raises(ValueError, match='Mock or scripted'):
        score_dual_bank(update, recovery, retention, recovery_before, recovery_after,
            retention_before, retention_after, update_verifier=Verifier(),
            checkpoint_store=Checkpoints(), snapshot_verifier=Verifier(),
            trial_verifier=Verifier(), training_identities=(training,))


def test_dual_bank_formula_and_insufficient_evidence_status():
    gain, retention_delta, reward = dual_bank_formula(0.2, 0.6, 0.9, 0.8, 0.5)
    assert (gain, retention_delta, reward) == pytest.approx((0.4, -0.1, 0.35))
    assert dual_bank_formula(0.3, 0.3, 0.8, 0.9, 0.5) == pytest.approx((0.0, 0.1, 0.0))
    assert dual_bank_formula(0.5, 0.3, 0.8, 0.7, 0.5) == pytest.approx((-0.2, -0.1, -0.25))
    with pytest.raises(ValueError): dual_bank_formula(1.1, 0.0, 0.5, 0.5, 0.5)
    error = InsufficientEvidenceError('unknown coverage exceeded')
    assert error.status.value == 'INSUFFICIENT_EVIDENCE'


def test_mock_verl_designer_uses_ungrouped_sequential_policy_gradient():
    from rift.training.records import DesignerFeedback
    tokens = Tokens((11, 12), (13, 14), (True, True))
    train_identity = identity('designer-credit', DataPurpose.TRAIN, Split.TRAIN)
    service_rollout_row = service_rollout(tokens, BEFORE, train_identity.task,
        'service-episode', Outcome.SUCCESS, response_id='service-response')
    update = CompletedUpdate(UpdateRequest(BEFORE, 'package-a', (service_rollout_row,), 'policy_gradient'),
                             AFTER, 'update-a', 'fake-update-evidence')
    gain, retention, total = dual_bank_formula(0.2, 0.6, 0.9, 0.8, 0.5)
    mock_reward = DualBankOutcomeReward('package-a', 'update-a', BEFORE, AFTER,
        'recovery-snapshot', 'retention-snapshot', 'recovery-before', 'recovery-after',
        'retention-before', 'retention-after', 0.2, 0.6, 0.9, 0.8, gain, retention,
        0.5, total, 1.0, 1.0, 0.8, 3, INSTANCE_DISJOINT_WITHIN_FAMILY_V1,
        PROTOCOL, 'sequential_curriculum')

    class MockTrust:
        def verify_update(self, _): return True
        def verify_reward(self, *_): return True
        def verify_designer_action(self, *_): return True

    feedback = (DesignerFeedback(tokens, update, mock_reward, 'designer-rollout'),)
    request = prepare_designer_update('designer-batch', DESIGNER, feedback,
        update_verifier=MockTrust(), reward_verifier=MockTrust(), action_verifier=MockTrust(),
        reward_baseline=0.1)
    assert request.algorithm == 'reinforce_baseline'
    assert request.advantage_baseline == 0.1
    assert request.reward_protocol == 'dual_bank_outcome_v1'
    assert request.credit_assignment_mode == 'sequential_curriculum'
    assert request.rollouts[0].group_id is None
    assert request.rollouts[0].reward == pytest.approx(0.35)
    with pytest.raises(ValueError, match='explicit scalar baseline'):
        prepare_designer_update('designer-batch', DESIGNER, feedback,
            update_verifier=MockTrust(), reward_verifier=MockTrust(), action_verifier=MockTrust())


def test_dual_bank_loop_freeze_baseline_and_restore(tmp_path):
    bootstrap = make_bootstrap(tmp_path)
    registry = SplitRegistry((row.identity for row in bootstrap.records),
        policy_version=INSTANCE_DISJOINT_WITHIN_FAMILY_V1)
    training = identity('loop-training', DataPurpose.TRAIN, Split.TRAIN)
    registry.register(training)
    rows = {row.task_ref: row for row in bootstrap.records}
    recovery, retention = ProbeSelector(record_verifier=AcceptRecordEvidence(),
        allow_nonreal=True).freeze(recovery_candidates=[rows['recovery-a'], rows['recovery-b']],
        retention_candidates=[rows['retention-a'], rows['retention-b']],
        diagnosis_identities=(rows['diag'].identity,), train_package_identities=(training,),
        split_registry=registry, before_checkpoint=BEFORE, recovery_tasks=1,
        retention_tasks=1, seed=4, protocol_ref='protocol-v1', protocol_digest=bootstrap.protocol_digest)
    runner = ProbeRunner(ScriptedAfterEvaluator(), AcceptProbeEvidence())
    recovery_before, retention_before = runner.baseline(recovery), runner.baseline(retention)
    tokens = Tokens((1,), (2,), (True,))
    service_row = service_rollout(tokens, BEFORE, training.task, 'episode',
        Outcome.SUCCESS, response_id='response')
    request = UpdateRequest(BEFORE, 'package-loop', (service_row,), 'policy_gradient')

    class ActionVerifier:
        def verify_designer_action(self, *args): return True
    class SnapshotVerifier:
        def verify_snapshot(self, snapshot): return snapshot.verify()
    class ProbeVerifier(AcceptProbeEvidence):
        def verify_result(self, result, snapshot):
            return result.snapshot_digest == snapshot.content_digest

    state = LoopState('package-loop', ('proposal-1',),
        designer_action=PendingDesignerAction(DESIGNER, 'a' * 32, ACTION))
    state = state.certify((Certification('proposal-1', CertificationStatus.CERTIFIED, training.task),))
    state = state.freeze_probes(recovery, retention, snapshot_verifier=SnapshotVerifier())
    assert state.stage is Stage.PROBES_FROZEN
    state = state.baseline_ready(recovery_before, retention_before,
                                 probe_verifier=ProbeVerifier())
    state = state.ready(request)
    assert state.reward_mode == 'dual_bank_outcome_v1'
    path = tmp_path / 'loop.json'
    state.save(path)
    restored = LoopState.load(path, action_verifier=ActionVerifier(),
        probe_verifier=ProbeVerifier())
    assert restored == state and restored.stage is Stage.READY_FOR_TRAINING


def test_mock_coordinator_persists_frozen_probes_and_resumes_before_training(tmp_path):
    bootstrap = make_bootstrap(tmp_path)
    registry = SplitRegistry((row.identity for row in bootstrap.records),
        policy_version=INSTANCE_DISJOINT_WITHIN_FAMILY_V1)
    training = identity('coordinator-training', DataPurpose.TRAIN, Split.TRAIN)
    registry.register(training)
    rows = {row.task_ref: row for row in bootstrap.records}
    recovery, retention = ProbeSelector(record_verifier=AcceptRecordEvidence(),
        allow_nonreal=True).freeze(
            recovery_candidates=[rows['recovery-a'], rows['recovery-b']],
            retention_candidates=[rows['retention-a'], rows['retention-b']],
            diagnosis_identities=(rows['diag'].identity,),
            train_package_identities=(training,), split_registry=registry,
            before_checkpoint=BEFORE, recovery_tasks=1, retention_tasks=1,
            seed=31, protocol_ref='protocol-v1',
            protocol_digest=bootstrap.protocol_digest)
    runner = ProbeRunner(ScriptedAfterEvaluator(), AcceptProbeEvidence())
    tokens = Tokens((7,), (8,), (True,))
    rollouts = tuple(service_rollout(tokens, BEFORE, training.task, f'ep-{i}',
        Outcome.SUCCESS, 'group-coordinator', response_id=f'ep-{i}') for i in range(2))
    request = UpdateRequest(BEFORE, 'package-coordinator', rollouts, 'grpo')

    class ActionVerifier:
        def verify_designer_action(self, *args): return True
    class SnapshotVerifier:
        def verify_snapshot(self, snapshot): return snapshot.verify()
    class ProbeVerifier(AcceptProbeEvidence):
        def verify_result(self, result, snapshot):
            return result.snapshot_digest == snapshot.content_digest

    state = LoopState('package-coordinator', ('proposal-coordinator',),
        designer_action=PendingDesignerAction(DESIGNER, 'b' * 32, ACTION))
    state = state.certify((Certification('proposal-coordinator',
        CertificationStatus.CERTIFIED, training.task),))
    evidence_store = JsonProbeEvidenceStore(tmp_path / 'coordinator-probes')
    state_path = tmp_path / 'coordinator-loop.json'
    persisted = []
    result = prepare_dual_bank_round(state, recovery_snapshot=recovery,
        retention_snapshot=retention, service_request=request, probe_runner=runner,
        probe_evidence_store=evidence_store, probe_verifier=ProbeVerifier(),
        snapshot_verifier=SnapshotVerifier(),
        save_state=lambda value: (value.save(state_path), persisted.append(value.stage)))
    assert result.stage is Stage.READY_FOR_TRAINING
    assert persisted == [Stage.PROBES_FROZEN, Stage.BASELINE_READY,
                         Stage.READY_FOR_TRAINING]
    assert evidence_store.load_snapshot(recovery.content_digest) == recovery
    assert result.recovery_before.phase == result.retention_before.phase == 'before'

    restored = LoopState.load(state_path, action_verifier=ActionVerifier(),
        probe_verifier=ProbeVerifier())
    resumed = prepare_dual_bank_round(restored, recovery_snapshot=recovery,
        retention_snapshot=retention, service_request=request, probe_runner=runner,
        probe_evidence_store=evidence_store, probe_verifier=ProbeVerifier(),
        snapshot_verifier=SnapshotVerifier(), save_state=lambda value: None)
    assert resumed == result
