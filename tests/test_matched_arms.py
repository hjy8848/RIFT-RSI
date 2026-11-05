import sys
import tempfile
import unittest
from unittest import mock

sys.path[:0] = ['src', 'third_party/usami']

from rift.evolution.matched_control import UpdateBudget
from rift.runtime import curriculum_run as run
from rift.runtime import matched_arms as arms_module
from rift.runtime.matched_arms import (ArmSpec, ProbeSet, compare_arms, parse_arms, parse_pairs,
                                       select_probes, step_tasks, task_counts)
from rift.schemas import Checkpoint


def counts(rates, n=8):
    """``{ref: (successes, valid, attempted)}`` with ``n`` valid samples at each given rate."""
    return {ref: (round(rate * n), n, n) for ref, rate in rates.items()}


class ProbeSelectionTests(unittest.TestCase):
    def test_recovery_takes_the_lowest_and_retention_only_mastered(self):
        screen = counts({'a': 0.0, 'b': 0.25, 'c': 0.5, 'd': 0.75, 'e': 1.0, 'f': 1.0}, n=4)
        probes = select_probes(screen, n_recovery=2, n_retention=2, seed=1)
        self.assertEqual(probes.recovery, ('a', 'b'))
        self.assertTrue(set(probes.retention) <= {'d', 'e', 'f'})
        self.assertEqual(probes, select_probes(screen, n_recovery=2, n_retention=2, seed=1))

    def test_tasks_without_a_valid_outcome_are_never_probes(self):
        screen = {'a': (0, 0, 4), 'b': (0, 4, 4), 'c': (4, 4, 4)}
        probes = select_probes(screen, n_recovery=1, n_retention=1, seed=0)
        self.assertEqual((probes.recovery, probes.retention), (('b',), ('c',)))

    def test_a_small_pool_is_refused(self):
        with self.assertRaises(ValueError):
            select_probes(counts({'a': 0.0, 'b': 1.0}), n_recovery=2, n_retention=1, seed=0)

    def test_banks_are_disjoint_and_unique(self):
        with self.assertRaises(ValueError):
            ProbeSet(('a',), ('a',))
        with self.assertRaises(ValueError):
            ProbeSet(('a', 'a'), ('b',))

    def test_unknown_outcomes_are_attempts_but_not_valid(self):
        rows = [{'task_ref': 't', 'outcome': o} for o in ('SUCCESS', 'failure', 'unknown')]
        self.assertEqual(task_counts(rows), {'t': (1, 2, 3)})


class CompareTests(unittest.TestCase):
    probes = ProbeSet(('r1', 'r2', 'r3', 'r4'), ('k1', 'k2', 'k3', 'k4'))
    before = counts({'r1': 0.0, 'r2': 0.25, 'r3': 0.0, 'r4': 0.25, 'k1': 1.0, 'k2': 1.0, 'k3': 0.875, 'k4': 1.0})

    def test_identical_arms_differ_by_zero(self):
        same = dict(self.before)
        report = compare_arms(self.before, {'B': same, 'A': same}, self.probes, (('B', 'A'),), n_boot=200)
        pair = report['pairs']['B-A']
        self.assertEqual(pair['reward_difference'], 0.0)
        self.assertEqual(pair['ci95'], [0.0, 0.0])

    def test_gain_and_retention_penalty(self):
        better = dict(self.before, r1=(4, 8, 8), r3=(4, 8, 8))          # recovery +0.25
        forgetful = dict(better, k1=(4, 8, 8), k2=(4, 8, 8))           # retention -0.25
        report = compare_arms(self.before, {'B': better, 'F': forgetful, 'A': dict(self.before)},
                              self.probes, (('B', 'A'), ('F', 'A')), lambda_retention=0.5, n_boot=200)
        self.assertAlmostEqual(report['arms']['B']['gain'], 0.25)
        self.assertAlmostEqual(report['arms']['B']['reward'], 0.25)
        self.assertAlmostEqual(report['arms']['F']['retention_delta'], -0.25)
        self.assertAlmostEqual(report['arms']['F']['reward'], 0.25 - 0.5 * 0.25)
        self.assertAlmostEqual(report['pairs']['B-A']['reward_difference'], 0.25)
        low, high = report['pairs']['B-A']['ci95']
        self.assertLessEqual(low, 0.25)
        self.assertGreaterEqual(high, 0.25)
        self.assertGreater(report['pairs']['B-A']['bootstrap_sd'], 0.0)

    def test_retention_gain_is_not_rewarded(self):
        improved = dict(self.before, k3=(8, 8, 8))
        report = compare_arms(self.before, {'B': improved, 'A': dict(self.before)}, self.probes,
                              (('B', 'A'),), n_boot=50)
        self.assertEqual(report['arms']['B']['reward'], 0.0)

    def test_low_coverage_gives_no_reward(self):
        unknown = dict(self.before, r1=(0, 0, 8), r2=(0, 0, 8))
        report = compare_arms(self.before, {'B': unknown, 'A': dict(self.before)}, self.probes,
                              (('B', 'A'),), min_valid_fraction=0.8, n_boot=50)
        self.assertFalse(report['valid'])
        self.assertNotIn('pairs', report)

    def test_tasks_missing_in_one_arm_drop_out_of_every_arm(self):
        partial = dict(self.before, r1=(1, 1, 8))           # one valid sample keeps r1
        report = compare_arms(self.before, {'B': partial, 'A': dict(self.before)}, self.probes,
                              (('B', 'A'),), min_valid_fraction=0.5, n_boot=50)
        self.assertEqual(report['usable_tasks']['recovery'], 4)
        none = dict(self.before, r1=(0, 0, 8))
        report = compare_arms(self.before, {'B': none, 'A': dict(self.before)}, self.probes,
                              (('B', 'A'),), min_valid_fraction=0.5, n_boot=50)
        self.assertEqual(report['usable_tasks']['recovery'], 3)


class ArmTests(unittest.TestCase):
    def test_arm_and_pair_parsing(self):
        arms = parse_arms([{'name': 'A', 'source': 'random', 'seed': 1},
                           {'name': 'B', 'source': 'designer', 'seed': 1}])
        self.assertEqual(parse_pairs('B-A', arms), (('B', 'A'),))
        for bad in ('B-B', 'C-A', 'BA', ''):
            with self.assertRaises(ValueError):
                parse_pairs(bad, arms)
        with self.assertRaises(ValueError):
            parse_arms([{'name': 'A', 'source': 'random', 'seed': 1}])
        with self.assertRaises(ValueError):
            parse_arms([{'name': 'A', 'source': 'x', 'seed': 1}, {'name': 'B', 'source': 'random', 'seed': 1}])

    def test_step_tasks_are_reproducible_and_mixed(self):
        train = [('F1', 't%d' % i) for i in range(40)]
        designer = ['d%d' % i for i in range(10)]
        arm = ArmSpec('B', 'designer', 3)
        first = step_tasks(arm, 0, train_pairs=train, designer_refs=designer, tasks_per_step=16, mix_fraction=0.5)
        self.assertEqual(first, step_tasks(arm, 0, train_pairs=train, designer_refs=designer,
                                           tasks_per_step=16, mix_fraction=0.5))
        self.assertEqual(sum(source == 'designer' for source, _ in first), 8)
        self.assertNotEqual(first, step_tasks(arm, 1, train_pairs=train, designer_refs=designer,
                                              tasks_per_step=16, mix_fraction=0.5))
        random_arm = step_tasks(ArmSpec('A', 'random', 3), 0, train_pairs=train, designer_refs=None,
                                tasks_per_step=16, mix_fraction=0.5)
        self.assertEqual(len(random_arm), 16)
        self.assertTrue(all(source == 'F1' for source, _ in random_arm))


class FakeStore:
    def __init__(self, virtual=()):
        self.virtual = set(virtual)


class RunnerTests(unittest.TestCase):
    start = Checkpoint('service', '0', 't', 'w0')

    def _context(self, virtual=()):
        calls = []

        class Context:
            checkpoint_store = FakeStore(virtual)

            def resync(self, checkpoint):
                calls.append(('resync', checkpoint.revision))

            def restore(self, checkpoint):
                calls.append(('restore', checkpoint.revision))
        return Context(), calls

    def _run(self, context, calls, **overrides):
        probe_pool = {'P': ['p%d' % i for i in range(8)]}
        screen_rates = {'p%d' % i: (0.0 if i < 4 else 1.0) for i in range(8)}
        measured = []

        def fake_sample(context_, by_key, checkpoint, *, groups, seed, label):
            measured.append((label, checkpoint.revision, seed))
            calls.append(('generate', label))
            refs = [ref for refs in by_key.values() for ref in refs]
            rate = (lambda ref: screen_rates[ref]) if label == 'screen' else (lambda ref: 0.5)
            return [{'task_ref': ref, 'outcome': 'success' if i < round(rate(ref) * 4) else 'failure'}
                    for ref in refs for i in range(4)]

        persisted = []

        def fake_train(context_, checkpoint, chosen, *, index, seed, persist, label):
            persisted.append((label, index, persist))
            calls.append(('update', label))
            after = Checkpoint('service', str(int(checkpoint.revision) + 1), 't', 'w-%s-%d' % (label, index))
            return after, {'index': index, 'revision': after.revision}

        kwargs = dict(start=self.start,
            arms=(ArmSpec('A', 'random', 1), ArmSpec('A2', 'random', 2), ArmSpec('B', 'designer', 1)),
            probe_pool=probe_pool, train_pairs=[('F1', 't%d' % i) for i in range(20)],
            designer_refs=['d1', 'd2'], steps=2, tasks_per_step=4, mix_fraction=0.5, n_recovery=4,
            n_retention=4, seed=7, pairs=(('B', 'A'), ('A2', 'A')), n_boot=20, persist_last=True)
        kwargs.update(overrides)
        with mock.patch.object(run, 'sample_rows', fake_sample), \
                mock.patch.object(arms_module, 'train_step', fake_train):
            result = arms_module.run_matched_arms(context, **kwargs)
        return result, measured, persisted

    def test_every_arm_forks_from_the_start_and_is_measured_with_paired_seeds(self):
        context, calls = self._context()
        result, measured, persisted = self._run(context, calls)
        self.assertEqual([c for c in calls if c[0] == 'restore'], [('restore', '0'), ('restore', '0')])
        self.assertEqual([label for label, _, _ in measured],
                         ['screen', 'before', 'arm-A', 'arm-A2', 'arm-B'])
        seeds = {label: seed for label, _, seed in measured}
        self.assertEqual(seeds['before'], seeds['arm-A'])
        self.assertNotEqual(seeds['screen'], seeds['before'])
        self.assertTrue(all(revision == '2' for label, revision, _ in measured if label.startswith('arm-')))
        self.assertEqual([p for p in persisted if p[2]], [('B', 5, True)])   # only the last arm's last step
        self.assertEqual(len({index for _, index, _ in persisted}), 6)       # unique update run ids
        self.assertEqual(result['final']['revision'], '2')
        self.assertTrue(result['report']['valid'])

    def test_vllm_is_woken_before_every_generate_that_follows_another(self):
        context, calls = self._context()
        self._run(context, calls)
        previous = None
        for call in calls:
            if call[0] == 'generate':
                self.assertNotEqual(previous, 'generate', calls)
            if call[0] in ('generate', 'update', 'resync', 'restore'):
                previous = call[0]

    def test_a_virtual_start_cannot_be_forked(self):
        context, calls = self._context(virtual={self.start})
        with self.assertRaises(ValueError):
            self._run(context, calls)

    def test_probe_tasks_may_not_be_training_tasks(self):
        context, calls = self._context()
        with self.assertRaises(ValueError):
            self._run(context, calls, designer_refs=['p1'])


class ContextForkTests(unittest.TestCase):
    def _context(self, root):
        from usami_v2.runtime.runner import EpisodeBudget
        log = self.log = []

        class Fake:
            def __init__(self, loaded):
                self.checkpoint, self._loaded_checkpoint, self.closed = loaded, loaded, False
                self.worker_group = object()
                self.restore_ok = True

            def restore_training_state(self, role, worker_group, checkpoint):
                log.append(('restore', role, checkpoint.revision))
                if self.restore_ok:
                    self._loaded_checkpoint = checkpoint
                return self.restore_ok

            def synchronize(self, checkpoint):
                log.append(('sync', checkpoint.revision))

            def close(self):
                self.closed = True

        context = run.ExperimentContext(root=root, service_model='m', verl_config_dir='.',
            episode_budget=EpisodeBudget(), update_budget=UpdateBudget(8, 1, 1e-6), runtime_factory=Fake)
        context._worker_manifest = lambda checkpoint: None
        return context

    def test_restore_reuses_the_runtime(self):
        start, later = Checkpoint('service', '0', 't', 'w0'), Checkpoint('service', '4', 't', 'w4')
        with tempfile.TemporaryDirectory() as root:
            context = self._context(root)
            runtime = context.ensure_runtime(start)
            runtime._loaded_checkpoint = later
            self.assertIs(context.restore(start), runtime)
            self.assertEqual(self.log, [('restore', 'service', '0'), ('sync', '0')])
            self.assertEqual(run.runtime_checkpoint(runtime), start)

    def test_a_failed_restore_releases_the_runtime(self):
        start = Checkpoint('service', '0', 't', 'w0')
        with tempfile.TemporaryDirectory() as root:
            context = self._context(root)
            runtime = context.ensure_runtime(start)
            runtime.restore_ok = False
            with self.assertRaises(ValueError):
                context.restore(start)
            self.assertTrue(runtime.closed and context.runtime is None)

    def test_virtual_checkpoints_cannot_be_restored(self):
        virtual = Checkpoint('service', '3', 't', 'v3')
        with tempfile.TemporaryDirectory() as root:
            context = self._context(root)
            context.checkpoint_store.virtual.add(virtual)
            with self.assertRaises(ValueError):
                context.restore(virtual)

    def test_resync_wakes_the_live_runtime(self):
        start = Checkpoint('service', '0', 't', 'w0')
        with tempfile.TemporaryDirectory() as root:
            context = self._context(root)
            runtime = context.ensure_runtime(start)
            self.assertIs(context.resync(start), runtime)
            self.assertEqual(self.log, [('sync', '0')])

    def test_round_backend_persist_override(self):
        from types import SimpleNamespace
        start = Checkpoint('service', '0', 't', 'w0')
        with tempfile.TemporaryDirectory() as root:
            context = self._context(root)
            runtime = context.ensure_runtime(start)
            seen = {}

            def create(request, **kwargs):
                seen['persist'] = kwargs['persist']
                return SimpleNamespace(train=lambda request: 'done')
            runtime.create_update_backend = create
            request = SimpleNamespace(policy_before=start)
            context.persist_every = 1                       # the schedule would persist ...
            self.assertEqual(run.RoundBackend(context, 0, start, persist=False).train(request), 'done')
            self.assertIs(seen['persist'], False)           # ... but the override wins
            run.RoundBackend(context, 0, start).train(request)
            self.assertIs(seen['persist'], True)


if __name__ == '__main__':
    unittest.main()
