import sys
import tempfile
import unittest
from pathlib import Path

sys.path[:0] = ['src', 'third_party/usami']

from rift.verl import checkpoints


class VerifyCacheTests(unittest.TestCase):
    def test_a_second_store_from_the_manifest_does_not_rehash(self):
        """VERL builds a fresh store per sample; the process must hash the shards once."""
        calls = {'n': 0}
        original = checkpoints._checkpoint_content_digests

        def counting(*args, **kwargs):
            calls['n'] += 1
            return original(*args, **kwargs)

        checkpoints._checkpoint_content_digests = counting
        try:
            with tempfile.TemporaryDirectory() as root:
                directory = Path(root) / 'ckpt'
                directory.mkdir()
                for kind in ('model', 'optim', 'extra_state'):
                    for rank in range(8):
                        (directory / f'{kind}_world_size_8_rank_{rank}.pt').write_bytes(
                            b'x' * 100 + bytes([rank]))
                store = checkpoints.FSDPCheckpointStore({}, world_size=8)
                checkpoint = store.publish('service', '0', 'tok', directory)
                store.save_manifest(Path(root) / 'm.json')
                import json   # an older manifest without stat digests forces the hashing path
                manifest = Path(root) / 'm.json'
                data = json.loads(manifest.read_text(encoding='utf-8'))
                for row in data['records']:
                    row.pop('stat_digest', None)
                manifest.write_text(json.dumps(data), encoding='utf-8')
                checkpoints._PROCESS_VERIFICATION_CACHE.clear()   # a fresh worker process
                before = calls['n']
                for _ in range(5):
                    fresh = checkpoints.FSDPCheckpointStore.load_manifest(Path(root) / 'm.json')
                    self.assertTrue(fresh.verify(checkpoint))
                self.assertEqual(calls['n'] - before, 1)
        finally:
            checkpoints._checkpoint_content_digests = original
            checkpoints._PROCESS_VERIFICATION_CACHE.clear()

    def _published(self, root, fill=b'w'):
        directory = Path(root) / 'ckpt'
        directory.mkdir()
        for kind in ('model', 'optim', 'extra_state'):
            for rank in range(8):
                (directory / f'{kind}_world_size_8_rank_{rank}.pt').write_bytes(fill * 100 + bytes([rank]))
        store = checkpoints.FSDPCheckpointStore({}, world_size=8)
        checkpoint = store.publish('service', '0', 'tok', directory)
        store.save_manifest(Path(root) / 'm.json')
        checkpoints._PROCESS_VERIFICATION_CACHE.clear()
        return directory, checkpoint

    def test_unchanged_files_load_without_reading_their_contents(self):
        calls = {'n': 0}
        original = checkpoints._checkpoint_content_digests
        def counting(*args, **kwargs):
            calls['n'] += 1
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as root:
            directory, checkpoint = self._published(root)
            checkpoints._checkpoint_content_digests = counting
            try:
                loaded = checkpoints.FSDPCheckpointStore.load_manifest(Path(root) / 'm.json')
                self.assertTrue(loaded.verify(checkpoint))
                self.assertEqual(calls['n'], 0)
            finally:
                checkpoints._checkpoint_content_digests = original
                checkpoints._PROCESS_VERIFICATION_CACHE.clear()

    def test_resized_files_fall_back_to_hashing_and_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            directory, checkpoint = self._published(root)
            (directory / 'model_world_size_8_rank_3.pt').write_bytes(b'z' * 250)
            with self.assertRaises(ValueError):
                checkpoints.FSDPCheckpointStore.load_manifest(Path(root) / 'm.json')
            checkpoints._PROCESS_VERIFICATION_CACHE.clear()

    def test_manifest_without_stat_digest_still_hashes(self):
        """Manifests written before this field existed keep the full check."""
        import json
        with tempfile.TemporaryDirectory() as root:
            directory, checkpoint = self._published(root)
            manifest = Path(root) / 'm.json'
            data = json.loads(manifest.read_text(encoding='utf-8'))
            for row in data['records']:
                row.pop('stat_digest')
            manifest.write_text(json.dumps(data), encoding='utf-8')
            self.assertTrue(checkpoints.FSDPCheckpointStore.load_manifest(manifest).verify(checkpoint))
            checkpoints._PROCESS_VERIFICATION_CACHE.clear()

    def test_a_changed_shard_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / 'ckpt'
            directory.mkdir()
            for kind in ('model', 'optim', 'extra_state'):
                for rank in range(8):
                    (directory / f'{kind}_world_size_8_rank_{rank}.pt').write_bytes(b'y' * 100 + bytes([rank]))
            store = checkpoints.FSDPCheckpointStore({}, world_size=8)
            checkpoint = store.publish('service', '0', 'tok', directory)
            store.save_manifest(Path(root) / 'm.json')
            checkpoints._PROCESS_VERIFICATION_CACHE.clear()
            fresh = checkpoints.FSDPCheckpointStore.load_manifest(Path(root) / 'm.json')
            self.assertTrue(fresh.verify(checkpoint))
            (directory / 'model_world_size_8_rank_3.pt').write_bytes(b'z' * 250)
            self.assertFalse(fresh.verify(checkpoint))
            checkpoints._PROCESS_VERIFICATION_CACHE.clear()


if __name__ == '__main__':
    unittest.main()
