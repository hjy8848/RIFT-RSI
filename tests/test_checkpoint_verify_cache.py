import sys
import tempfile
import unittest
from pathlib import Path

sys.path[:0] = ['src', 'third_party/usami']

from rift.verl import checkpoints


class InMemoryVersionTests(unittest.TestCase):
    def _store(self, root):
        directory = Path(root) / 'ckpt'
        directory.mkdir()
        for kind in ('model', 'optim', 'extra_state'):
            for rank in range(8):
                (directory / f'{kind}_world_size_8_rank_{rank}.pt').write_bytes(b'b' * 64 + bytes([rank]))
        store = checkpoints.FSDPCheckpointStore({}, world_size=8)
        return store, store.publish('service', '0', 'tok', directory)

    def test_a_virtual_version_verifies_has_no_files_and_chains(self):
        with tempfile.TemporaryDirectory() as root:
            store, base = self._store(root)
            first = store.publish_virtual(base, '1', 'update-1')
            second = store.publish_virtual(first, '2', 'update-2')
            self.assertTrue(store.verify(first) and store.verify(second))
            self.assertNotEqual(first.weights_digest, second.weights_digest)
            self.assertEqual(first.policy, base.policy)
            with self.assertRaises(ValueError):
                store.resolve(first)
            with self.assertRaises(ValueError):
                store.publish_virtual(base, '1', 'update-1')          # same identity twice

    def test_virtual_versions_survive_a_manifest_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            store, base = self._store(root)
            first = store.publish_virtual(base, '1', 'u1')
            store.save_manifest(Path(root) / 'm.json')
            loaded = checkpoints.FSDPCheckpointStore.load_manifest(Path(root) / 'm.json')
            self.assertTrue(loaded.verify(first) and loaded.verify(base))
            self.assertEqual(loaded.artifact_digests[first], store.artifact_digests[first])

    def test_the_update_ledger_and_sync_receipts_accept_virtual_versions(self):
        from rift.schemas import Outcome, Tokens
        from rift.training.contracts import CompletedUpdate, UpdateRequest
        from rift.training.records import service_rollout
        from rift.verl.serving import VerlServingAttestor
        from rift.verl.update_ledger import UpdateLedger
        with tempfile.TemporaryDirectory() as root:
            store, base = self._store(root)
            first = store.publish_virtual(base, '1', 'u1')
            second = store.publish_virtual(first, '2', 'u2')
            ledger = UpdateLedger(Path(root) / 'updates', store)
            rows = tuple(service_rollout(Tokens((1,), (2, index + 3), (True, True)), first, 'task', 'ep%d' % index,
                                         Outcome.SUCCESS, 'group', response_id='ep%d' % index)
                         for index in range(2))
            request = UpdateRequest(first, 'pkg', rows, 'grpo')
            evidence = ledger.record_update('run-1', request, second, {'ok': 1})
            self.assertTrue(ledger.verify_update(CompletedUpdate(request, second, 'run-1', evidence)))
            attestor = VerlServingAttestor(Path(root) / 'serving', store)
            attestor.activate_after_sync(second, 2, store.artifact_digests[second])
            self.assertTrue(attestor.verify_serving(second))
            self.assertTrue(attestor.verify_generation(second, {'global_steps': 2, 'min_global_steps': 2,
                                                                'max_global_steps': 2}))


class ParallelHashTests(unittest.TestCase):
    def test_concurrent_hashing_gives_the_serial_result(self):
        import hashlib
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / 'ckpt'
            directory.mkdir()
            for kind in ('model', 'optim', 'extra_state'):
                for rank in range(8):
                    (directory / f'{kind}_world_size_8_rank_{rank}.pt').write_bytes(
                        bytes([rank]) * (9 * 1024 * 1024 + rank))   # more than one read chunk
            model, artifact = checkpoints._checkpoint_content_digests(directory, 8)
            digests = {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).digest()
                       for p in sorted(directory.rglob('*')) if p.is_file()}
            expected_model = hashlib.sha256()
            for rank in range(8):
                name = f'model_world_size_8_rank_{rank}.pt'
                expected_model.update(name.encode() + b'\0' + digests[name])
            expected_artifact = hashlib.sha256()
            for name, digest in sorted(digests.items()):
                encoded = name.encode()
                expected_artifact.update(len(encoded).to_bytes(8, 'big') + encoded + digest)
            self.assertEqual((model, artifact), (expected_model.hexdigest(), expected_artifact.hexdigest()))


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
