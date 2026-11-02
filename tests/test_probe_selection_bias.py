"""The dual-bank reward must not reward a policy that did not change.

Tasks enter the Recovery set because a few trials happened to fail and the
Retention set because a few happened to succeed. If those same trials are then
reused as the frozen "before" baseline, any fresh re-measurement regresses to
the mean, so Recovery "improves" and Retention "forgets" with no learning at all.
These tests drive the real ProbeSelector and the real ProductionPlanFactory
baseline code against an evaluator whose policy never changes.
"""
from hashlib import sha256
from types import SimpleNamespace

import pytest

from rift.banks.core import JsonTaskEvidenceStore
from rift.banks.records import EvidenceLevel, ObservedTrial, TaskEvidenceRecord
from rift.probes.runner import ProbeRunner
from rift.probes.selection import ProbeSelector
from rift.replay.splits import SplitRegistry
from rift.runtime.plan_factory import ProductionPlanFactory
from rift.schemas import (Checkpoint, DataPurpose, Identity, INSTANCE_DISJOINT_WITHIN_FAMILY_V1,
                          Outcome, Split)

BEFORE = Checkpoint('service', '0', 'tok', 'weights-0')
AFTER = Checkpoint('service', '1', 'tok', 'weights-1')
PROTOCOL = 'f' * 64


def draw(task_ref, checkpoint, seed, p):
    """Deterministic Bernoulli(p) keyed by task, weights and seed.

    Keying on the weights models the worst case of a policy whose outputs fully
    diverge after any update; the success probability itself never changes.
    """
    digest = sha256(f'{task_ref}|{checkpoint.weights_digest}|{seed}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big') / 2 ** 64 < p


def trial(task_ref, checkpoint, seed, success):
    return ObservedTrial(task_ref, f'{task_ref}:{checkpoint.weights_digest}:{seed}', checkpoint, seed,
        Outcome.SUCCESS if success else Outcome.FAILURE, PROTOCOL, f'evidence:{task_ref}:{seed}',
        True, EvidenceLevel.REAL_SERVICE)


def identity(task_ref):
    return Identity('world-' + task_ref, 'task-' + task_ref, 'family', 'lineage-' + task_ref,
                    Split.VALIDATION, DataPurpose.BANK_CANDIDATE)


def history_record(task_ref, p, n=5):
    trials = tuple(trial(task_ref, BEFORE, 1000 + i, draw(task_ref, BEFORE, 1000 + i, p)) for i in range(n))
    return TaskEvidenceRecord(task_ref, identity(task_ref), BEFORE, PROTOCOL, trials, 'history')


def fixed_record(task_ref, outcomes, *, checkpoint=BEFORE, first_seed=1000):
    trials = tuple(trial(task_ref, checkpoint, first_seed + i, ok) for i, ok in enumerate(outcomes))
    return TaskEvidenceRecord(task_ref, identity(task_ref), checkpoint, PROTOCOL, trials, 'fixed')


WEAK, MASTERED = (False, False, False, False, True), (True, True, True, True, False)


class AcceptAll:
    def verify_record(self, record): return True
    def verify_probe_trial(self, trial_, probe, checkpoint): return True


def test_optional_fingerprint_filter_applies_to_preselection_and_freeze():
    from rift.probes.fingerprint import TaskFingerprint
    near = fixed_record('near', WEAK)
    other = fixed_record('other', WEAK)
    retained = fixed_record('retained', MASTERED)
    histories = (near, other, retained)
    registry = SplitRegistry(tuple(row.identity for row in histories),
                             policy_version=INSTANCE_DISJOINT_WITHIN_FAMILY_V1)
    train = Identity('train-world', 'train-task', 'family', 'train-lineage', Split.TRAIN,
                     DataPurpose.TRAIN)
    same = TaskFingerprint((('refund',),), ((1, 2, 3),))
    far = TaskFingerprint((('exchange',),), ((1, 2, 3),))
    values = {'train-task': same, 'near': same, 'other': far, 'retained': far}
    selector = ProbeSelector(record_verifier=AcceptAll())
    kwargs = dict(recovery_candidates=(near, other), retention_candidates=(retained,),
        train_package_identities=(train,), split_registry=registry, recovery_tasks=1,
        retention_tasks=1, seed=3, fingerprint_resolver=values.__getitem__, min_distance=.5)
    recovery, _ = selector.preselect(**kwargs)
    assert [row.task_ref for row in recovery] == ['other']
    baselines = tuple(fixed_record(row.task_ref, WEAK if row is not retained else MASTERED,
                                  first_seed=2000) for row in histories)
    recovery_snapshot, _ = selector.freeze(**kwargs, before_checkpoint=BEFORE,
        protocol_ref='protocol', protocol_digest=PROTOCOL, current_baseline_candidates=baselines)
    assert recovery_snapshot.probes[0].task_ref == 'other'
    assert dict(recovery_snapshot.exclusions)['near_duplicate_of_training'] == 1


class NullPolicyEvaluator:
    """Same success probability at every checkpoint."""
    def __init__(self, p_by_task): self.p = p_by_task

    def evaluate(self, task_ref, checkpoint, seed, protocol_ref, protocol_digest):
        return trial(task_ref, checkpoint, seed, draw(task_ref, checkpoint, seed, self.p[task_ref]))

    def evaluate_many(self, requests):
        return tuple(self.evaluate(*request) for request in requests)


def make_factory(tmp_path, registry, evaluator):
    factory = ProductionPlanFactory.__new__(ProductionPlanFactory)
    factory.split_registry = registry
    factory.task_evidence_store = JsonTaskEvidenceStore(tmp_path / 'task-evidence')
    factory.baseline_evidence_store = JsonTaskEvidenceStore(tmp_path / 'probe-baselines')
    factory.trial_verifier = AcceptAll()
    factory.probe_selector = ProbeSelector(record_verifier=AcceptAll())
    factory.protocol_ref, factory.protocol_digest = 'protocol', PROTOCOL
    factory.trials_per_probe, factory.probe_seed = 5, 4100
    factory.recovery_tasks = factory.retention_tasks = 6
    factory.metric_sink = None
    factory.evaluator = evaluator
    return factory


def one_round(tmp_path, pool_seed, round_index):
    import random
    rng = random.Random(pool_seed)
    p = {f'T{pool_seed}-{i:03d}': rng.random() for i in range(60)}
    histories = tuple(history_record(task, prob) for task, prob in p.items())
    registry = SplitRegistry((row.identity for row in histories),
                             policy_version=INSTANCE_DISJOINT_WITHIN_FAMILY_V1)
    evaluator = NullPolicyEvaluator(p)
    factory = make_factory(tmp_path / f'r{pool_seed}', registry, evaluator)
    # Deployment saves bootstrap and post-update records in the evidence store,
    # which is exactly where the baseline lookup used to find them again.
    for history in histories:
        factory.task_evidence_store.save(history)
    weak = [h for h in histories if h.success_rate <= 0.25]
    mastered = [h for h in histories if h.success_rate >= 0.75]
    if len(weak) < 6 or len(mastered) < 6:
        return None
    recovery, retention = factory._freeze_snapshots(
        round_index, BEFORE, 'A1', SimpleNamespace(diagnosis_task_refs=()), (), weak, mastered,
        SimpleNamespace(task_sources=()), evaluator, ())
    after_recovery, after_retention = ProbeRunner(evaluator, AcceptAll()).after_many(
        (recovery, retention), AFTER)

    def mean_rate(snapshot=None, result=None):
        rows = ([probe.successes / probe.valid_trials for probe in snapshot.probes] if snapshot
                else [row.success_rate for row in result.results])
        return sum(rows) / len(rows)

    gain = mean_rate(result=after_recovery) - mean_rate(snapshot=recovery)
    drop = mean_rate(result=after_retention) - mean_rate(snapshot=retention)
    # What the old pipeline measured: for the very same tasks, the selection trials as baseline.
    by_task = {h.task_ref: h for h in histories}
    selection_rate = sum(by_task[p.task_ref].success_rate for p in recovery.probes) / len(recovery.probes)
    naive_gain = mean_rate(result=after_recovery) - selection_rate
    return gain, drop, naive_gain, weak, mastered


def test_unchanged_policy_earns_no_recovery_gain_and_no_retention_drop(tmp_path):
    rows = [row for pool in range(60) if (row := one_round(tmp_path, pool, pool))]
    assert len(rows) >= 40
    gain = sum(r[0] for r in rows) / len(rows)
    drop = sum(r[1] for r in rows) / len(rows)
    assert abs(gain) < 0.05, gain
    assert abs(drop) < 0.05, drop


def test_the_old_reuse_of_selection_trials_would_have_shown_a_large_phantom_gain(tmp_path):
    # Guards the test above: with the selection trials as baseline the bias is clearly visible.
    rows = [row for pool in range(60) if (row := one_round(tmp_path, pool, pool))]
    naive = sum(r[2] for r in rows) / len(rows)
    assert naive > 0.08, naive


def test_selector_rejects_a_baseline_that_is_the_selection_sample(tmp_path):
    history = fixed_record('T-reuse', WEAK)
    other = fixed_record('T-other', MASTERED)
    registry = SplitRegistry((history.identity, other.identity),
                             policy_version=INSTANCE_DISJOINT_WITHIN_FAMILY_V1)
    selector = ProbeSelector(record_verifier=AcceptAll())
    with pytest.raises(ValueError, match='insufficient'):
        selector.freeze(recovery_candidates=(history,), retention_candidates=(other,),
            train_package_identities=(), split_registry=registry, before_checkpoint=BEFORE,
            recovery_tasks=1, retention_tasks=1, seed=1, protocol_ref='protocol',
            protocol_digest=PROTOCOL, current_baseline_candidates=(history, other))
    # no separate baseline at all is no longer silently replaced by the history record
    with pytest.raises(ValueError, match='insufficient'):
        selector.freeze(recovery_candidates=(history,), retention_candidates=(other,),
            train_package_identities=(), split_registry=registry, before_checkpoint=BEFORE,
            recovery_tasks=1, retention_tasks=1, seed=1, protocol_ref='protocol',
            protocol_digest=PROTOCOL)


def test_a_baseline_measured_with_fresh_seeds_is_accepted(tmp_path):
    history = fixed_record('T-fresh', WEAK)
    other = fixed_record('T-keep', MASTERED)
    fresh = fixed_record('T-fresh', (False,) * 5, first_seed=2000)
    fresh_other = fixed_record('T-keep', (True,) * 5, first_seed=2000)
    registry = SplitRegistry((history.identity, other.identity),
                             policy_version=INSTANCE_DISJOINT_WITHIN_FAMILY_V1)
    recovery, retention = ProbeSelector(record_verifier=AcceptAll()).freeze(
        recovery_candidates=(history,), retention_candidates=(other,), train_package_identities=(),
        split_registry=registry, before_checkpoint=BEFORE, recovery_tasks=1, retention_tasks=1,
        seed=1, protocol_ref='protocol', protocol_digest=PROTOCOL,
        current_baseline_candidates=(fresh, fresh_other))
    assert recovery.probes[0].seeds == tuple(2000 + i for i in range(5))
    assert not set(recovery.probes[0].seeds) & {t.seed for t in history.trials}
