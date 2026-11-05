import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path[:0] = ['src', 'third_party/usami']

from rift.runtime.curriculum_run import WorldKit, seed_task_registry

SPECS = [{'generator': 'F2', 'instances': 2, 'seed': 424200, 'balanced': True},
         {'generator': 'F1', 'instances': 1, 'seed': 424300, 'balanced': True, 'near_distractors': 2}]


def no_proving(*args, **kwargs):
    raise AssertionError('a task that is already certified was proved again')


class TaskRegistrySeedTests(unittest.TestCase):
    def _certified(self, root):
        kit = WorldKit(root)
        refs = kit.certify_specs(SPECS)
        kit.save()
        return refs

    def test_a_seeded_run_does_not_prove_tasks_again(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            refs = self._certified(a)
            seed_task_registry(b, Path(a) / 'task_registry')
            kit = WorldKit(b)
            with mock.patch.object(WorldKit, 'certify_plan', no_proving):
                self.assertEqual(kit.certify_specs(SPECS), refs)

    def test_a_registry_from_before_the_cache_existed_is_rebuilt_from_task_identities(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            refs = self._certified(a)
            (Path(a) / 'task_registry' / 'spec-refs.json').unlink()
            seed_task_registry(b, a)                     # the run directory itself is accepted too
            kit = WorldKit(b)
            with mock.patch.object(WorldKit, 'certify_plan', no_proving):
                self.assertEqual(kit.certify_specs(SPECS), refs)

    def test_a_task_the_registry_lacks_is_still_proved(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            self._certified(a)
            seed_task_registry(b, Path(a) / 'task_registry')
            kit = WorldKit(b)
            more = SPECS + [{'generator': 'F2', 'instances': 1, 'seed': 424900, 'balanced': True}]
            refs = kit.certify_specs(more)
            self.assertEqual(len(refs), 2)         # keyed by spec class: F2 and near-distractor F1
            self.assertEqual(sum(len(v) for v in refs.values()), 4)

    def test_refusals(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            self._certified(a)
            source = Path(a) / 'task_registry'
            with self.assertRaises(FileNotFoundError):
                seed_task_registry(b, Path(b))
            seed_task_registry(b, source)
            with self.assertRaises(FileExistsError):     # never merge into a registry that already has tasks
                seed_task_registry(b, source)
            (source / 'spec-refs.json').unlink()
            with tempfile.TemporaryDirectory() as c, self.assertRaises(ValueError):
                seed_task_registry(c, source, patience_slack=2)    # unknown patience setting of the source

    def test_a_different_patience_setting_ignores_the_cache(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            self._certified(a)
            seed_task_registry(b, Path(a) / 'task_registry')
            kit = WorldKit(b, patience_slack=3)
            self.assertEqual(kit._spec_refs, {})


if __name__ == '__main__':
    unittest.main()
