"""Host-side split assignment based on stable upstream identities."""
import json
from dataclasses import asdict
from pathlib import Path
from rift.schemas import Identity, Split

class SplitRegistry:
    def __init__(self, identities=()):
        self._identities = []
        for identity in identities: self.register(identity)
    def register(self, identity):
        if not isinstance(identity, Identity): raise TypeError("Identity required")
        if any(identity.overlaps(row) and identity.split != row.split for row in self._identities):
            raise ValueError("cross-split world, task, family or lineage overlap")
        if identity not in self._identities: self._identities.append(identity)
    def contains(self, identity):
        return identity in self._identities
    def save(self, path):
        Path(path).write_text(json.dumps([asdict(i) for i in self._identities], sort_keys=True), encoding='utf-8')
    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(Identity(i['world'], i['task'], i['family'], i['lineage'], Split(i['split'])) for i in data)
