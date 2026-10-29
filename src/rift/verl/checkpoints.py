"""Integrity checks for VERL v0.9.1 FSDP checkpoint shards.

These checks prove file presence/content, not that a real trainer produced the
files. An update verifier must independently establish update provenance.
"""
import hashlib
import json
import stat
from dataclasses import asdict
from pathlib import Path
from rift.schemas import Checkpoint

_VERIFY_CACHE_VERSION = 1
_PROCESS_VERIFICATION_CACHE = {}


def _artifact_metadata_digest(directory):
    """Fingerprint checkpoint paths without rereading their contents."""
    root = Path(directory)
    if not root.is_dir() or root.is_symlink(): raise ValueError('invalid checkpoint directory')
    digest = hashlib.sha256()
    entries = [root, *root.rglob('*')]
    entries.sort(key=lambda path: '' if path == root else path.relative_to(root).as_posix())
    for path in entries:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode): raise ValueError('checkpoint symlinks are unsupported')
        if stat.S_ISDIR(metadata.st_mode): kind = 'directory'
        elif stat.S_ISREG(metadata.st_mode): kind = 'file'
        else: kind = 'other'
        name = '' if path == root else path.relative_to(root).as_posix()
        row = [name, kind, metadata.st_dev, metadata.st_ino, metadata.st_mode,
               metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns]
        encoded = json.dumps(row, separators=(',', ':')).encode('utf-8')
        digest.update(len(encoded).to_bytes(8, 'big') + encoded)
    return digest.hexdigest()


def model_digest(directory, world_size):
    root = Path(directory)
    if type(world_size) is not int or world_size < 1: raise ValueError('invalid world size')
    digest = hashlib.sha256()
    for rank in range(world_size):
        name = f'model_world_size_{world_size}_rank_{rank}.pt'
        path = root / name
        if not path.is_file() or path.is_symlink(): raise ValueError('missing model shard: ' + name)
        digest.update(name.encode('utf-8') + b'\0')
        content = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''): content.update(chunk)
        digest.update(content.digest())
    return digest.hexdigest()

def artifact_digest(directory):
    root = Path(directory)
    if not root.is_dir() or root.is_symlink(): raise ValueError('invalid checkpoint directory')
    digest = hashlib.sha256()
    files = sorted(root.rglob('*'), key=lambda p: p.relative_to(root).as_posix())
    for path in files:
        if path.is_symlink(): raise ValueError('checkpoint symlinks are unsupported')
        if not path.is_file(): continue
        name = path.relative_to(root).as_posix().encode('utf-8')
        content = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''): content.update(chunk)
        digest.update(len(name).to_bytes(8, 'big') + name + content.digest())
    return digest.hexdigest()


def _checkpoint_content_digests(directory, world_size):
    """Hash checkpoint files once to derive both model and artifact identities."""
    root = Path(directory)
    if type(world_size) is not int or world_size < 1:
        raise ValueError('invalid world size')
    if not root.is_dir() or root.is_symlink():
        raise ValueError('invalid checkpoint directory')
    content_digests = {}
    for path in sorted(root.rglob('*'), key=lambda p: p.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError('checkpoint symlinks are unsupported')
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError('checkpoint contains a non-regular file')
        content = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                content.update(chunk)
        content_digests[path.relative_to(root).as_posix()] = content.digest()

    for rank in range(world_size):
        for kind in ('model', 'optim', 'extra_state'):
            name = f'{kind}_world_size_{world_size}_rank_{rank}.pt'
            if name not in content_digests or (root / name).stat().st_size == 0:
                raise ValueError('missing or empty checkpoint shard: ' + name)

    model_hash = hashlib.sha256()
    for rank in range(world_size):
        name = f'model_world_size_{world_size}_rank_{rank}.pt'
        model_hash.update(name.encode('utf-8') + b'\0' + content_digests[name])
    artifact_hash = hashlib.sha256()
    for name, content_digest in sorted(content_digests.items()):
        encoded_name = name.encode('utf-8')
        artifact_hash.update(len(encoded_name).to_bytes(8, 'big')
                             + encoded_name + content_digest)
    return model_hash.hexdigest(), artifact_hash.hexdigest()


class FSDPCheckpointStore:
    """Verify published checkpoint contents once per unchanged file-tree fingerprint."""
    def __init__(self, paths, *, world_size, artifact_digests=None):
        if type(world_size) is not int or world_size < 1: raise ValueError('invalid world size')
        self.paths = dict(paths)
        self.world_size = world_size
        self.artifact_digests = dict(artifact_digests or {})
        self._verified_fingerprints = {}
        for checkpoint, path in self.paths.items():
            if checkpoint in self.artifact_digests:
                continue
            root = Path(path).resolve()
            cache_key = (checkpoint, str(root), self.world_size)
            cached = _PROCESS_VERIFICATION_CACHE.get(cache_key)
            if cached is not None:
                self.artifact_digests[checkpoint] = cached[-2]
                continue
            try:
                model_hash, artifact_hash = _checkpoint_content_digests(root, self.world_size)
            except (OSError, ValueError):
                artifact_hash = artifact_digest(root)
                model_hash = None
            self.artifact_digests[checkpoint] = artifact_hash
            if model_hash == checkpoint.weights_digest:
                self._remember_verified(checkpoint, root)
    def register(self, checkpoint, path):
        if checkpoint in self.paths: raise ValueError('checkpoint identity already registered')
        model_hash, digest = _checkpoint_content_digests(path, self.world_size)
        if model_hash != checkpoint.weights_digest:
            raise ValueError('checkpoint model digest does not match its identity')
        self.paths[checkpoint] = path
        self.artifact_digests[checkpoint] = digest
        self._remember_verified(checkpoint, path)
    def publish(self, policy, revision, tokenizer, path):
        """Register a just-saved checkpoint after one content-hash pass."""
        model_hash, artifact_hash = _checkpoint_content_digests(path, self.world_size)
        checkpoint = Checkpoint(policy, revision, tokenizer, model_hash)
        if checkpoint in self.paths:
            raise ValueError('checkpoint identity already registered')
        self.paths[checkpoint] = path
        self.artifact_digests[checkpoint] = artifact_hash
        self._remember_verified(checkpoint, path)
        return checkpoint
    def _remember_verified(self, checkpoint, path):
        root = Path(path).resolve()
        fingerprint = _artifact_metadata_digest(root)
        self._verified_fingerprints[checkpoint] = (
            _VERIFY_CACHE_VERSION, str(root), self.world_size,
            checkpoint.weights_digest, self.artifact_digests[checkpoint], fingerprint)
        cache_key = (checkpoint, str(root), self.world_size)
        _PROCESS_VERIFICATION_CACHE[cache_key] = self._verified_fingerprints[checkpoint]
    def resolve(self, checkpoint):
        if not isinstance(checkpoint, Checkpoint) or checkpoint not in self.paths:
            raise ValueError('checkpoint is not registered')
        return str(Path(self.paths[checkpoint]).resolve())
    def verify(self, checkpoint):
        try:
            root = Path(self.resolve(checkpoint))
            if not root.is_dir(): return False
            artifact_digest_expected = self.artifact_digests[checkpoint]
            fingerprint = _artifact_metadata_digest(root)
            cache_entry = (_VERIFY_CACHE_VERSION, str(root), self.world_size,
                           checkpoint.weights_digest, artifact_digest_expected, fingerprint)
            cache_key = (checkpoint, str(root), self.world_size)
            if (self._verified_fingerprints.get(checkpoint) == cache_entry
                    or _PROCESS_VERIFICATION_CACHE.get(cache_key) == cache_entry):
                self._verified_fingerprints[checkpoint] = cache_entry
                return True
            model_hash, artifact_hash = _checkpoint_content_digests(root, self.world_size)
            verified = (model_hash == checkpoint.weights_digest
                        and artifact_hash == artifact_digest_expected)
            if verified:
                self._verified_fingerprints[checkpoint] = cache_entry
            else:
                self._verified_fingerprints.pop(checkpoint, None)
                _PROCESS_VERIFICATION_CACHE.pop(cache_key, None)
            return verified
        except (OSError, ValueError, TypeError):
            return False
    def save_manifest(self, path):
        records = [{'checkpoint': asdict(cp), 'path': str(Path(location).resolve()),
                    'artifact_digest': self.artifact_digests[cp]}
                   for cp, location in self.paths.items()]
        with Path(path).open('x', encoding='utf-8') as stream:
            json.dump({'world_size': self.world_size, 'records': records}, stream, sort_keys=True)
    @classmethod
    def load_manifest(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        paths, digests = {}, {}
        for row in data['records']:
            cp = Checkpoint(**row['checkpoint'])
            if cp in paths: raise ValueError('duplicate checkpoint identity')
            paths[cp] = row['path']; digests[cp] = row['artifact_digest']
        store = cls(paths, world_size=data['world_size'], artifact_digests=digests)
        if any(not store.verify(cp) for cp in paths):
            raise ValueError('checkpoint manifest does not match stored files')
        return store
