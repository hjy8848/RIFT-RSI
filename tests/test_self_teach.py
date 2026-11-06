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
        if specs == 'EVAL':
            return {'E': ['e0', 'e1']}
        if specs == 'OVERLAP':
            return {'E': ['t0']}
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

    def forget_checkpoints(self, checkpoints):
        self.forgotten = list(checkpoints)


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

    def test_the_ability_curve_is_measured_on_fixed_tasks_after_every_action(self):
        calls = []

        def fake_eval(ctx, by_key, cp, *, groups, seed, label):
            calls.append((label, cp.revision, seed, sorted(by_key)))
            return {'success': 0.5, 'episodes': 8, 'tasks': 2, 'unknown': 0, 'classes': {'E': 0.5}, 'per_task': {}}
        with tempfile.TemporaryDirectory() as root, mock.patch.object(S, 'evaluate_checkpoint', fake_eval):
            _, result, _ = self._run(root, actions=2, calibrate_every=0, eval_specs='EVAL', eval_every=1)
            lines = [json.loads(line) for line in (Path(root) / 'self-teach.jsonl').read_text().splitlines()]
        self.assertEqual([line['event'] for line in lines], ['start', 'action', 'eval', 'action', 'eval'])
        self.assertEqual([c[1] for c in calls], ['1', '2'])                 # each evaluates the checkpoint just written
        self.assertEqual({c[2] for c in calls}, {7000})                     # the same seed every time
        self.assertEqual([c[0] for c in calls], ['eval-a000', 'eval-a001'])

    def test_saved_checkpoints_are_evaluated_in_turn_on_the_same_tasks(self):
        with tempfile.TemporaryDirectory() as old, tempfile.TemporaryDirectory() as new:
            old, new = Path(old), Path(new)
            (old / 'checkpoints').mkdir()
            for revision in (10, 20):
                (old / ('step%d' % revision)).mkdir()
            records = [{'checkpoint': asdict(checkpoint(r)), 'path': str(old / ('step%d' % r)), 'artifact_digest': 'd%d' % r}
                       for r in (10, 20)]
            for r in (10, 20):
                (old / 'checkpoints' / ('service-step-%d.json' % r)).write_text(json.dumps(
                    {'world_size': 4, 'records': [rec for rec in records if rec['checkpoint']['revision'] == str(r)]}))
            (new / 'checkpoints').mkdir()
            order, adopted = [], []
            context = FakeContext(new)
            context.adopt_manifest = lambda path: adopted.append(json.loads(Path(path).read_text()))
            context.restore = lambda cp: order.append(('restore', cp.revision))
            context.world.certify_specs = lambda specs: {'E': ['e0', 'e1']}

            def fake_eval(ctx, by_key, cp, *, groups, seed, label):
                order.append(('eval', cp.revision, seed))
                return {'success': 0.5 + int(cp.revision) / 100, 'episodes': 8, 'tasks': 2, 'unknown': 0,
                        'classes': {}, 'per_task': {}}
            with mock.patch.object(S, 'evaluate_checkpoint', fake_eval):
                result = S.eval_checkpoints(context, entries=['%s:20' % old, '%s:10' % old], eval_specs='EVAL',
                                            groups=2, seed=7000)
            self.assertEqual(order, [('restore', '20'), ('eval', '20', 7000), ('restore', '10'), ('eval', '10', 7000)])
            self.assertEqual(sorted(r['checkpoint']['revision'] for r in adopted[0]['records']), ['10', '20'])
            self.assertEqual([r['revision'] for r in result['results']], ['20', '10'])
            self.assertTrue(context.closed)
            with self.assertRaises(FileNotFoundError):              # a deleted checkpoint is reported, not skipped
                (old / 'step10').rmdir()
                S.eval_checkpoints(context, entries=['%s:10' % old], eval_specs='EVAL', groups=2, seed=1)

    def test_evaluation_tasks_may_not_overlap_the_training_pool(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                self._run(root, actions=1, eval_specs='OVERLAP')

    def test_pruning_keeps_recent_and_every_nth_fork(self):
        with tempfile.TemporaryDirectory() as root:
            forks = [(0, None, None)]
            for action in range(1, 8):
                path = Path(root) / ('fork%d' % action)
                path.mkdir()
                forks.append((action, path, checkpoint(action)))
            removed = S.prune_forks(forks, keep_last=2, keep_every=5)
            kept = sorted(p.name for p in Path(root).iterdir())
            self.assertEqual(kept, ['fork5', 'fork6', 'fork7'])
            self.assertEqual(sorted(cp.revision for _, cp in removed), ['1', '2', '3', '4'])    # the caller forgets these
            self.assertEqual([entry[0] for entry in forks], [0, 5, 6, 7])
            self.assertEqual(S.prune_forks([(0, None, None), (1, Path(root) / 'fork5', checkpoint(1))],
                                           keep_last=0, keep_every=5), [])
            self.assertTrue((Path(root) / 'fork5').exists())

    def test_the_run_forgets_what_pruning_deleted(self):
        """The step after a prune must not read a manifest that still lists the deleted files."""
        from rift.runtime.curriculum_run import ExperimentContext
        from rift.evolution.matched_control import UpdateBudget
        from usami_v2.runtime.runner import EpisodeBudget
        with tempfile.TemporaryDirectory() as root:
            context = ExperimentContext(root=root, service_model='m', verl_config_dir='.',
                episode_budget=EpisodeBudget(), update_budget=UpdateBudget(8, 1, 1e-6),
                runtime_factory=lambda cp: None)
            old, kept = checkpoint(5), checkpoint(10)
            store = context.checkpoint_store
            for cp in (old, kept):
                store.virtual.add(cp)
                store.artifact_digests[cp] = cp.weights_digest
            manifests = Path(root) / 'checkpoints'
            store.save_manifest(manifests / 'service-step-10.json')       # lists both, as the old run's did
            context.forget_checkpoints([old])
            path = context._worker_manifest(kept)
            listed = json.loads(json.loads(path.read_text())['checkpoint_manifest'] and
                                Path(json.loads(path.read_text())['checkpoint_manifest']).read_text())['records']
            revisions = sorted(row['checkpoint']['revision'] for row in listed)
            self.assertEqual(revisions, ['10'])

    def test_resume_takes_only_the_last_checkpoint_and_the_next_action_number(self):
        with tempfile.TemporaryDirectory() as old, tempfile.TemporaryDirectory() as new:
            old, new = Path(old), Path(new)
            (old / 'checkpoints').mkdir()
            (old / 'teacher').mkdir()
            rows = [{'event': 'start'}, {'event': 'action', 'action': 0, 'next_revision': '5'},
                    {'event': 'action', 'action': 1, 'next_revision': '10'}]
            (old / 'self-teach.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            records = [{'checkpoint': asdict(checkpoint(5)), 'path': 'gone', 'artifact_digest': 'x'},
                       {'checkpoint': asdict(checkpoint(10)), 'path': str(old / 'step10'), 'artifact_digest': 'y'}]
            (old / 'checkpoints' / 'service-step-10.json').write_text(json.dumps({'world_size': 4, 'records': records}))
            (old / 'teacher' / 'a001-g0-p00.json').write_text(json.dumps({'row': {
                'signatures': ['sigA', 'unmeasured:KeyError', 'sigB'], 'selected': [True, True, False]}}))
            adopted = []
            context = SimpleNamespace(root=new, adopt_manifest=lambda path: adopted.append(json.loads(Path(path).read_text())))
            (new / 'checkpoints').mkdir()
            cp, first_action, seen = S.resume_state(context, old)
            self.assertEqual((cp.revision, first_action, seen), ('10', 2, {'sigA'}))
            self.assertEqual([r['checkpoint']['revision'] for r in adopted[0]['records']], ['10'])
            with self.assertRaises(ValueError):                     # a run that finished no action
                (old / 'self-teach.jsonl').write_text('{"event": "start"}')
                S.resume_state(context, old)


if __name__ == '__main__':
    unittest.main()
