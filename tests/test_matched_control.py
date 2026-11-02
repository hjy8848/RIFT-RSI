from types import SimpleNamespace

import pytest

from rift.banks.records import EvidenceLevel, ObservedTrial
from rift.evolution import matched_control as matched
from rift.probes.records import ProbeSnapshot, TaskProbe
from rift.schemas import (Checkpoint, DataPurpose, Identity, INSTANCE_DISJOINT_WITHIN_FAMILY_V1,
                          Outcome, Split, Tokens)
from rift.training.contracts import CompletedUpdate, Rollout, UpdateRequest


def test_matched_control_uses_same_start_and_snapshots_and_reports_differences(monkeypatch):
    before = Checkpoint('service', '0', 'tok', 'before')
    identities = {}
    def request(name):
        task = 'train-' + name
        identities[task] = Identity('world-' + task, task, 'family', 'lineage-' + task,
                                    Split.TRAIN, DataPurpose.TRAIN)
        rows = tuple(Rollout(Tokens((1,), (i + 2,), (True,)), before, task, name + str(i),
                             float(i), 'outcome', name + '-group') for i in range(2))
        return UpdateRequest(before, name, rows, 'grpo')
    treatment, control = request('treatment'), request('control')
    def snapshot(name, purpose):
        ident = Identity('world-' + name, name, 'family', 'lineage-' + name,
                         Split.VALIDATION, DataPurpose.BANK_CANDIDATE)
        trial = ObservedTrial(name, name + '-baseline', before, 1, Outcome.SUCCESS, 'protocol',
                              'evidence', True, EvidenceLevel.REAL_SERVICE)
        probe = TaskProbe(name, ident, purpose, before, (trial,), 'protocol-ref', 'protocol')
        return ProbeSnapshot.create(snapshot_id=name, purpose=purpose, probes=(probe,), random_seed=1,
            split_policy_version=INSTANCE_DISJOINT_WITHIN_FAMILY_V1, baseline_checkpoint=before,
            protocol_digest='protocol', candidate_count=1)
    recovery = snapshot('recovery', DataPurpose.RECOVERY_FEEDBACK)
    retention = snapshot('retention', DataPurpose.RETENTION_FEEDBACK)
    seen = []
    class Runner:
        def baseline(self, snap): return 'baseline-' + snap.snapshot_id
        def after_many(self, snapshots, checkpoint):
            seen.append(('probes', snapshots, checkpoint))
            return 'recovery-after', 'retention-after'
    def backend_factory(name, req, budget):
        def train(actual):
            assert actual is req and actual.policy_before == before
            seen.append(('train', name, actual.policy_before, budget))
            return CompletedUpdate(actual, Checkpoint('service', name, 'tok', 'weights-' + name),
                                   'run-' + name, 'evidence')
        return SimpleNamespace(train=train, budget_for=lambda request: budget)
    def score(update, rec, ret, *args, **kwargs):
        assert rec is recovery and ret is retention
        assert {row.task for row in kwargs['training_identities']} == {r.task_id for r in update.request.rollouts}
        return SimpleNamespace(gain=.3 if update.run_id == 'run-treatment' else .1,
                               retention_delta=-.1 if update.run_id == 'run-treatment' else 0.,
                               total=.25 if update.run_id == 'run-treatment' else .1)
    monkeypatch.setattr(matched, 'score_dual_bank', score)
    verifier = SimpleNamespace(verify_update=lambda update: True,
                               verify_update_budget=lambda update, budget: True)
    budget = matched.UpdateBudget(2, 2, 1e-6)
    common = dict(recovery_snapshot=recovery, retention_snapshot=retention,
        treatment_budget=budget, control_budget=budget, backend_factory=backend_factory,
        probe_runner=Runner(), update_verifier=verifier, checkpoint_store=object(),
        snapshot_verifier=object(), trial_verifier=object(),
        training_identity_resolver=lambda req: (identities[req.rollouts[0].task_id],))
    result = matched.run_matched_control(treatment, control, **common)
    assert [row[1] for row in seen if row[0] == 'train'] == ['treatment', 'control']
    assert all(row[1] == (recovery, retention) for row in seen if row[0] == 'probes')
    assert result.optimizer_steps == 2
    assert result.gain_difference == pytest.approx(.2)
    assert result.retention_difference == pytest.approx(-.1)
    assert result.reward_difference == pytest.approx(.15)
    with pytest.raises(ValueError, match='identical'):
        matched.run_matched_control(treatment, control,
            **{**common, 'control_budget': matched.UpdateBudget(2, 1, 1e-6)})
    seen.clear()
    def wrong_backend(name, req, budget):
        return SimpleNamespace(train=lambda request: seen.append(name),
            budget_for=lambda request: budget if name == 'treatment' else matched.UpdateBudget(1, 2, 1e-6))
    with pytest.raises(ValueError, match='actual matched update budget'):
        matched.run_matched_control(treatment, control, **{**common, 'backend_factory': wrong_backend})
    assert seen == []


def test_matched_control_scores_frozen_trials_through_real_dual_bank_scorer():
    from rift.probes.runner import ProbeRunner
    before = Checkpoint('service', '0', 'tok', 'before')
    def make_request(name):
        task = f'train-{name}'
        rows = tuple(Rollout(Tokens((1,), (i + 2,), (True,)), before, task, f'{name}-{i}',
                             float(i), 'business_outcome_v2', name + '-group') for i in range(2))
        return UpdateRequest(before, name, rows, 'grpo')
    def identity(name, purpose, split):
        return Identity('world-' + name, name, 'family', 'lineage-' + name, split, purpose)
    def trial(name, cp, seed, success):
        return ObservedTrial(name, f'{name}-{cp.weights_digest}-{seed}', cp, seed,
            Outcome.SUCCESS if success else Outcome.FAILURE, 'protocol', 'verified-evidence',
            True, EvidenceLevel.REAL_SERVICE)
    def snapshot(name, purpose, values):
        ident = identity(name, DataPurpose.BANK_CANDIDATE, Split.VALIDATION)
        baseline = tuple(trial(name, before, i, value) for i, value in enumerate(values))
        probe = TaskProbe(name, ident, purpose, before, baseline, 'ref', 'protocol')
        return ProbeSnapshot.create(snapshot_id=name, purpose=purpose, probes=(probe,), random_seed=0,
            split_policy_version=INSTANCE_DISJOINT_WITHIN_FAMILY_V1, baseline_checkpoint=before,
            protocol_digest='protocol', candidate_count=1)
    recovery = snapshot('recovery', DataPurpose.RECOVERY_FEEDBACK, (True, False))
    retention = snapshot('retention', DataPurpose.RETENTION_FEEDBACK, (True, True))
    class Evaluator:
        def evaluate(self, task_ref, checkpoint, seed, protocol_ref, protocol_digest):
            if checkpoint.revision == 'treatment':
                success = task_ref == 'recovery' or seed == 0
            else:
                success = task_ref == 'retention' or seed == 0
            return trial(task_ref, checkpoint, seed, success)
    verifier = SimpleNamespace(verify_update=lambda update: True,
        verify_update_budget=lambda update, budget: True,
        verify_probe_trial=lambda trial, probe, checkpoint: True,
        verify_frozen_before=lambda snap, update: True)
    budget = matched.UpdateBudget(2, 1, 1e-6)
    def factory(name, request, budget):
        def train(req):
            return CompletedUpdate(req, Checkpoint('service', name, 'tok', 'after-' + name),
                                   'run-' + name, 'evidence-' + name)
        return SimpleNamespace(train=train, budget_for=lambda req: budget)
    result = matched.run_matched_control(make_request('treatment'), make_request('control'),
        recovery_snapshot=recovery, retention_snapshot=retention, treatment_budget=budget,
        control_budget=budget, backend_factory=factory, probe_runner=ProbeRunner(Evaluator(), verifier),
        update_verifier=verifier, checkpoint_store=SimpleNamespace(verify=lambda checkpoint: True),
        snapshot_verifier=verifier, trial_verifier=verifier, min_valid_fraction=1.0,
        training_identity_resolver=lambda req: (identity(req.rollouts[0].task_id, DataPurpose.TRAIN, Split.TRAIN),))
    assert result.treatment.gain == pytest.approx(.5)
    assert result.treatment.retention_delta == pytest.approx(-.5)
    assert result.treatment.reward == pytest.approx(.25)
    assert result.control.reward == 0
    assert result.reward_difference == pytest.approx(.25)
