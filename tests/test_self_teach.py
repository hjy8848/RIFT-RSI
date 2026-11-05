import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path[:0] = ['src', 'third_party/usami']

from rift.runtime import curriculum_run as run
from rift.runtime import self_teach as S
from rift.runtime import teacher as T
from rift.schemas import Checkpoint


def checkpoint(n):
    return Checkpoint('service', str(n), 't', 'w%d' % n)


class FakeWorld:
    task_registry = object()

    def certify_specs(self, specs):
        return {'F1': ['t0', 't1', 't2', 't3']}

    def save(self):
        pass


class FakeContext:
    group_size = 4

    def __init__(self, root):
        self.root = Path(root)
        self.world = FakeWorld()
        self.capture_store = object()
        self.checkpoint_store = SimpleNamespace(world_size=2)
        self.calls = []
        self.closed = False

    def initial_checkpoint(self):
        return checkpoint(0)

    def resync(self, cp):
        self.calls.append(('resync', cp.revision))
        return SimpleNamespace(tokenizer=None)

    def close_runtime(self):
        self.closed = True


def sample(sid, group, refs):
    row = {'status': 'certified', 'task_refs': list(refs), 'selected': [True] * len(refs), 'copy_score': 0.0}
    return T.TeacherSample(sid, group, checkpoint(0), (1, 2), (3, 4), (-0.1, -0.2), '{}', 'teacher-' + sid, 1, row)


class SelfTeachTests(unittest.TestCase):
    def _run(self, root, *, actions=2, calibrate_every=2, plan_refs=(('x1',), ('x2',)), **overrides):
        context = FakeContext(root)
        seen = {'matched': [], 'solver': [], 'updates': [], 'generate': []}

        def fake_generate(ctx, cp, failures, *, action, **kwargs):
            seen['generate'].append((action, cp.revision, list(failures)))
            return [sample('a%03d-g0-p%d' % (action, i), 'a%03d-g0' % action, refs)
                    for i, refs in enumerate(plan_refs)], [11]

        def fake_certify(world, members, *, prompt_seed, seen=None):
            return members

        def fake_matched(ctx, **kwargs):
            seen['matched'].append((kwargs['start'].revision, kwargs['index_start'], kwargs['label_prefix']))
            rows = [{'task_counts': {'x1': (2, 4), 'x2': (0, 4)}, 'episodes': [
                {'task_ref': 't0', 'outcome': 'failure', 'episode_id': 'e%d' % kwargs['index_start']}]}] * kwargs['steps']
            return {'history': {'T': rows}, 'report': {'pairs': {'T-C': {'reward_difference': 0.01}}},
                    'final': asdict(checkpoint(100 + kwargs['index_start'])),
                    'next_index': kwargs['index_start'] + 2 * kwargs['steps']}

        def fake_solver(ctx, cp, *, action, index, steps, **kwargs):
            seen['solver'].append((cp.revision, index))
            rows = [{'task_counts': {'x1': (2, 4), 'x2': (0, 4)}, 'episodes': [
                {'task_ref': 't1', 'outcome': 'failure', 'episode_id': 'f%d' % action}]}] * steps
            return checkpoint(200 + index), rows, index + steps

        def fake_update(ctx, samples, rewards, policy_before, ledger, index):
            seen['updates'].append((policy_before.revision, index, dict(rewards)))
            return SimpleNamespace(policy_after=checkpoint(int(len(seen['updates'])))), len(samples)

        defaults = dict(train_specs=[], probe_specs=[{}], actions=actions, groups=1, plans=2, steps=2,
                        tasks_per_step=4, mix_fraction=0.5, seed=3, calibrate_every=calibrate_every,
                        n_recovery=2, n_retention=2)
        defaults.update(overrides)
        with mock.patch.object(run, 'sample_rows', lambda *a, **k: []), \
                mock.patch.object(S, 'failures_from_rows', lambda ctx, rows, refs, n: [{'task_ref': 'seed-failure'}]
                                  if not rows else [{'task_ref': rows[-1]['task_ref']}]), \
                mock.patch.object(S, 'generate_plans', fake_generate), \
                mock.patch.object(T, 'certify_samples', fake_certify), \
                mock.patch.object(S, 'run_matched_arms', fake_matched), \
                mock.patch.object(S, 'run_solver_steps', fake_solver), \
                mock.patch.object(S, 'teacher_update', fake_update):
            result = S.self_teach(context, **defaults)
        return context, result, seen

    def test_calibration_runs_only_every_nth_action_and_indices_never_repeat(self):
        with tempfile.TemporaryDirectory() as root:
            context, result, seen = self._run(root, actions=3, calibrate_every=2)
            self.assertEqual([row['calibrated'] for row in result['actions']], [True, False, True])
            self.assertEqual(len(seen['matched']), 2)
            self.assertEqual(len(seen['solver']), 1)
            used = [index for _, index in seen['solver']] + [i for _, i, _ in seen['matched']] \
                + [index for _, index, _ in seen['updates']]
            self.assertEqual(len(used), len(set(used)))
            self.assertEqual([p for _, _, p in seen['matched']], ['a000-', 'a002-'])
            self.assertTrue(context.closed)

    def test_each_action_forks_from_the_previous_update_and_plans_are_rewarded_by_learnability(self):
        with tempfile.TemporaryDirectory() as root:
            _, result, seen = self._run(root, actions=2, calibrate_every=0)
            self.assertEqual([revision for _, revision, _ in seen['generate']], ['0', '1'])
            self.assertEqual(result['final_checkpoint']['revision'], '2')
            rewards = seen['updates'][0][2]
            self.assertAlmostEqual(rewards['a000-g0-p0'], 1.0)      # x1: 2 of 4 solved
            self.assertAlmostEqual(rewards['a000-g0-p1'], 0.0)      # x2: never solved
            row = result['actions'][0]
            self.assertEqual(row['reward_mean'], 0.5)
            self.assertEqual(row['measured_plans'], 2)

    def test_failures_for_the_next_prompt_come_from_the_solver_training_rows(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, seen = self._run(root, actions=2, calibrate_every=0)
            self.assertEqual(seen['generate'][0][2], [{'task_ref': 'seed-failure'}])
            self.assertEqual(seen['generate'][1][2], [{'task_ref': 't1'}])

    def test_an_action_with_no_certified_task_skips_the_solver_but_still_updates(self):
        with tempfile.TemporaryDirectory() as root:
            context, result, seen = self._run(root, actions=1, calibrate_every=1, plan_refs=((), ()))
            self.assertEqual((seen['solver'], seen['matched']), ([], []))
            self.assertEqual(len(seen['updates']), 1)
            self.assertEqual(seen['updates'][0][0], '0')          # updated from the fork itself

    def test_log_has_one_row_per_action(self):
        with tempfile.TemporaryDirectory() as root:
            self._run(root, actions=2, calibrate_every=0)
            lines = [json.loads(line) for line in (Path(root) / 'self-teach.jsonl').read_text().splitlines()]
            self.assertEqual([line['event'] for line in lines], ['start', 'action', 'action'])
            with self.assertRaises(FileExistsError):
                self._run(root)

    def test_pruning_keeps_recent_and_every_nth_fork(self):
        with tempfile.TemporaryDirectory() as root:
            dirs = []
            forks = [(0, None)]
            for action in range(1, 8):
                path = Path(root) / ('fork%d' % action)
                path.mkdir()
                forks.append((action, path))
            removed = S.prune_forks(forks, keep_last=2, keep_every=5)
            kept = sorted(p.name for p in Path(root).iterdir())
            self.assertEqual(kept, ['fork5', 'fork6', 'fork7'])
            self.assertEqual(len(removed), 4)
            self.assertEqual(S.prune_forks([(0, None), (1, Path(root) / 'fork5')], keep_last=0, keep_every=5), [])
            self.assertTrue((Path(root) / 'fork5').exists())


if __name__ == '__main__':
    unittest.main()
