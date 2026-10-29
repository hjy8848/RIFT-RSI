"""Integrity checks for VERL v0.9.1 FSDP checkpoint shards.

These checks prove file presence/content, not that a real trainer produced the
files. An update verifier must independently establish update provenance.
"""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from rift.schemas import Checkpoint


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


class FSDPCheckpointStore:
    def __init__(self, paths, *, world_size, artifact_digests=None):
        if type(world_size) is not int or world_size < 1: raise ValueError('invalid world size')
        self.paths = dict(paths)
        self.world_size = world_size
        self.artifact_digests = dict(artifact_digests or {})
        for checkpoint, path in self.paths.items():
            if checkpoint not in self.artifact_digests:
                self.artifact_digests[checkpoint] = artifact_digest(path)
    def register(self, checkpoint, path):
        if checkpoint in self.paths: raise ValueError('checkpoint identity already registered')
        digest = artifact_digest(path)
        self.paths[checkpoint] = path
        self.artifact_digests[checkpoint] = digest
    def resolve(self, checkpoint):
        if not isinstance(checkpoint, Checkpoint) or checkpoint not in self.paths:
            raise ValueError('checkpoint is not registered')
        return str(Path(self.paths[checkpoint]).resolve())
    def verify(self, checkpoint):
        try:
            root = Path(self.resolve(checkpoint))
            if not root.is_dir(): return False
            for rank in range(self.world_size):
                for kind in ('model', 'optim', 'extra_state'):
                    file = root / f'{kind}_world_size_{self.world_size}_rank_{rank}.pt'
                    if not file.is_file() or file.is_symlink() or file.stat().st_size == 0: return False
            return (model_digest(root, self.world_size) == checkpoint.weights_digest
                    and artifact_digest(root) == self.artifact_digests[checkpoint])
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
