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
                checkpoints._PROCESS_VERIFICATION_CACHE.clear()   # a fresh worker process
                before = calls['n']
                for _ in range(5):
                    fresh = checkpoints.FSDPCheckpointStore.load_manifest(Path(root) / 'm.json')
                    self.assertTrue(fresh.verify(checkpoint))
                self.assertEqual(calls['n'] - before, 1)
        finally:
            checkpoints._checkpoint_content_digests = original
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
