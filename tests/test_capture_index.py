import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path[:0] = ['src', 'third_party/usami']

from rift.verl import capture_store as module
from rift.verl.capture_store import NativeCaptureStore


class Attestor:
    pass


class RolloutIndexTests(unittest.TestCase):
    def _store(self, root):
        return NativeCaptureStore(Path(root) / 'captures', serialization='fixture', attestor=Attestor())

    def _write(self, store, name, rollout_id):
        store.root.mkdir(parents=True, exist_ok=True)
        (store.root / (name + '.json')).write_text(
            json.dumps({'rollout_id': rollout_id, 'episode_id': name}), encoding='utf-8')

    def test_each_stored_file_is_parsed_once_however_many_lookups_there_are(self):
        """A lookup per request used to reparse everything captured so far."""
        with tempfile.TemporaryDirectory() as root:
            store = self._store(root)
            for index in range(40):
                self._write(store, 'ep%02d' % index, 'rollout-%d' % index)
            real, calls = module.json.loads, {'n': 0}

            def counting(*args, **kwargs):
                calls['n'] += 1
                return real(*args, **kwargs)
            module.json.loads = counting
            try:
                for _ in range(48):                       # one lookup per request in a round
                    self.assertEqual(store._paths_for_rollout('rollout-never-captured'), ())
            finally:
                module.json.loads = real
            self.assertEqual(calls['n'], 40)

    def test_files_that_appear_later_are_found(self):
        with tempfile.TemporaryDirectory() as root:
            store = self._store(root)
            self._write(store, 'first', 'r1')
            self.assertEqual([p.name for p in store._paths_for_rollout('r1')], ['first.json'])
            self.assertEqual(store._paths_for_rollout('r2'), ())
            self._write(store, 'second', 'r2')
            self.assertEqual([p.name for p in store._paths_for_rollout('r2')], ['second.json'])

    def test_an_unreadable_file_is_still_an_error(self):
        with tempfile.TemporaryDirectory() as root:
            store = self._store(root)
            store.root.mkdir(parents=True)
            (store.root / 'broken.json').write_text('{not json', encoding='utf-8')
            with self.assertRaises(ValueError):
                store._paths_for_rollout('anything')

    def test_two_files_for_one_rollout_are_both_reported(self):
        with tempfile.TemporaryDirectory() as root:
            store = self._store(root)
            self._write(store, 'a', 'same')
            self._write(store, 'b', 'same')
            self.assertEqual(len(store._paths_for_rollout('same')), 2)


if __name__ == '__main__':
    unittest.main()
