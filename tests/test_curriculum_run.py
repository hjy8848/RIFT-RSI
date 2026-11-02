import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path[:0] = ['src', 'third_party/usami']

from rift.benchmark.families import FAMILIES, FAMILY_IDS
from rift.evolution.matched_control import UpdateBudget
from rift.runtime import curriculum_run as run
from rift.schemas import Checkpoint


class SpecTests(unittest.TestCase):
    def test_specs_are_validated(self):
        parsed = run.parse_task_specs([{'family': 'F1', 'instances': 2, 'seed': 5}])
        self.assertEqual(parsed[0]['knobs'], {})
        for bad in ([], [{'family': 'F1', 'instances': 0, 'seed': 1}],
                    [{'family': 'F1', 'instances': 1, 'seed': -1}],
                    [{'family': 'F1', 'instances': 1, 'seed': 1, 'extra': 1}]):
            with self.assertRaises(ValueError):
                run.parse_task_specs(bad)

    def test_inline_json_and_file(self):
        self.assertEqual(run.load_json_argument('[{"a": 1}]'), [{'a': 1}])
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'specs.json'
            path.write_text('{"a": 2}', encoding='utf-8')
            self.assertEqual(run.load_json_argument(str(path)), {'a': 2})

    def test_summary_keeps_unknown_apart(self):
        rows = [('F1:{}', 't1', 'success')] * 3 + [('F1:{}', 't1', 'failure'), ('F1:{}', 't2', 'unknown')]
        report = run.summarize_outcomes(rows)['F1:{}']
        self.assertEqual((report['success'], report['failure'], report['unknown']), (3, 1, 1))
        self.assertAlmostEqual(report['success_rate'], 0.75)
        self.assertLess(report['wilson95'][0], 0.75)
        self.assertGreater(report['wilson95'][1], 0.75)

    def test_summary_accepts_the_real_outcome_values(self):
        """The Agent Loop reports Outcome.value, which is upper case."""
        rows = [('k', 't', 'SUCCESS')] * 3 + [('k', 't', 'FAILURE'), ('k', 't', 'UNKNOWN')]
        report = run.summarize_outcomes(rows)['k']
        self.assertEqual((report['success'], report['failure'], report['unknown']), (3, 1, 1))

    def test_wilson_bounds(self):
        low, high = run.wilson_interval(0, 10)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)
        self.assertEqual(run.wilson_interval(0, 0), (0.0, 1.0))


class WorldTests(unittest.TestCase):
    def test_every_family_certifies_as_a_designer_arm(self):
        """Arms use n_tasks > 1; each family must certify that way before a run depends on it."""
        with tempfile.TemporaryDirectory() as root:
            kit = run.WorldKit(root)
            failures = {}
            for family in FAMILY_IDS:
                try:
                    instance = FAMILIES[family].instance({}, 3000 + FAMILY_IDS.index(family), n_tasks=4)
                    result = kit.host(instance).certify_training_plan(
                        instance.proposal(n_tasks=4), world_seed=instance.world_seed)
                    if result.build.status != 'SAT' or len(result.tasks) != 4:
                        failures[family] = result.build.status
                except Exception as error:   # report all families, not only the first
                    failures[family] = repr(error)[:120]
            self.assertEqual(failures, {})

    def test_registry_survives_restart(self):
        with tempfile.TemporaryDirectory() as root:
            kit = run.WorldKit(root)
            refs = kit.certify_specs([{'family': 'F1', 'knobs': {}, 'instances': 2, 'seed': 10}])
            kit.save()
            again = run.WorldKit(root)
            self.assertEqual(set(refs['F1:{}']), set(again.task_registry.task_refs))


class EvalRequestTests(unittest.TestCase):
    def test_groups_are_complete_and_ids_unique(self):
        checkpoint = Checkpoint('service', '0', 'tok', 'w0')
        requests, owners = run.build_eval_requests({'F1:{}': ['a' * 64, 'b' * 64]}, checkpoint,
            groups=2, group_size=4, seed=7, label='x0')
        self.assertEqual(len(requests), 16)
        self.assertEqual(len({row['rift_request_id'] for row in requests}), 16)
        self.assertEqual(len({row['session_id'] for row in requests}), 16)
        by_group = {}
        for row in requests:
            by_group.setdefault(row['uid'], []).append(row)
        self.assertEqual({len(rows) for rows in by_group.values()}, {4})
        for rows in by_group.values():
            self.assertEqual(len({row['rift_task_ref'] for row in rows}), 1)
            self.assertEqual(len({row['rift_sampling_seed'] for row in rows}), 4)
            self.assertEqual(len({row['rift_customer_seed'] for row in rows}), 1)
        self.assertEqual(set(owners), {row['rift_request_id'] for row in requests})

    def test_labels_separate_runs(self):
        checkpoint = Checkpoint('service', '0', 'tok', 'w0')
        first, _ = run.build_eval_requests({'k': ['a' * 64]}, checkpoint, groups=1, group_size=4, seed=7, label='x0')
        second, _ = run.build_eval_requests({'k': ['a' * 64]}, checkpoint, groups=1, group_size=4, seed=7, label='x1')
        self.assertFalse({r['rift_request_id'] for r in first} & {r['rift_request_id'] for r in second})


class PolicyTests(unittest.TestCase):
    def test_every_named_policy_builds(self):
        from rift.evolution.curriculum import Selection
        arms = (Selection('F1', {}, 4), Selection('F4', {'age_days': 20}, 4))
        for name in run.POLICIES:
            policy = run.build_policy(name, arms, seed=1, window=5, fixed_family='F1')
            self.assertEqual(policy.propose(0, ()).n_tasks, 4)
        with self.assertRaises(ValueError):
            run.build_policy('fixed', arms, seed=1, window=5)
        with self.assertRaises(ValueError):
            run.build_policy('nope', arms, seed=1, window=5)


class FakeStore:
    def verify(self, checkpoint):
        return True


class RecoveryTests(unittest.TestCase):
    def test_no_receipt_means_train_and_mismatch_is_refused(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as root:
            ledger = SimpleNamespace(root=Path(root), verify_update=lambda update: False)
            context = SimpleNamespace(ledger=ledger)
            recover = run.build_recover_update(context)
            self.assertIsNone(recover(0, object()))
            after = Checkpoint('service', '1', 'tok', 'w1')
            (Path(root) / (run.run_id_for(0) + '.json')).write_text(
                json.dumps({'after': {'policy': 'service', 'revision': '1', 'tokenizer': 'tok',
                                      'weights_digest': 'w1'}}), encoding='utf-8')
            request = SimpleNamespace(policy_before=Checkpoint('service', '0', 'tok', 'w0'))
            with self.assertRaises(ValueError):
                recover(0, request)
            ledger.verify_update = lambda update: True
            update = recover(0, request)
            self.assertEqual(update.policy_after, after)
            self.assertEqual(update.run_id, run.run_id_for(0))


class ChatTemplateTests(unittest.TestCase):
    def test_runs_use_the_non_thinking_template(self):
        self.assertEqual(run.CHAT_TEMPLATE_KWARGS, {'enable_thinking': False})


class BackendBudgetTests(unittest.TestCase):
    def test_backend_reports_the_shared_budget(self):
        from types import SimpleNamespace
        budget = UpdateBudget(8, 1, 1e-6)
        backend = run.RoundBackend(SimpleNamespace(update_budget=budget), 0, None)
        self.assertEqual(backend.budget_for(None), budget)


if __name__ == '__main__':
    unittest.main()

