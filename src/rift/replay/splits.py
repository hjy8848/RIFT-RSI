"""Host-side split assignment under an explicit, versioned isolation policy."""
import json
from dataclasses import asdict
import os
from pathlib import Path
from rift.schemas import (DataPurpose, Identity, Split, STRICT_FAMILY_DISJOINT_V1,
                          SPLIT_POLICY_VERSIONS)

class SplitRegistry:
    def __init__(self, identities=(), *, policy_version=STRICT_FAMILY_DISJOINT_V1,
                 persistence_path=None):
        if policy_version not in SPLIT_POLICY_VERSIONS:
            raise ValueError("unknown split policy version")
        self.policy_version = policy_version
        self.persistence_path = None
        self._identities = []
        for identity in identities: self.register(identity)
        self.persistence_path = None if persistence_path is None else Path(persistence_path)
    @property
    def identities(self):
        return tuple(self._identities)
    def register(self, identity):
        if not isinstance(identity, Identity): raise TypeError("Identity required")
        for row in self._identities:
            if not identity.overlaps_under(row, self.policy_version):
                continue
            if identity.split != row.split:
                raise ValueError("cross-split identity overlap under " + self.policy_version)
        if identity not in self._identities:
            self._identities.append(identity)
            try:
                if self.persistence_path is not None:
                    self.save()
            except BaseException:
                self._identities.pop()
                raise
    def contains(self, identity):
        return identity in self._identities
    def save(self, path=None):
        payload = {"schema_version": 2, "split_policy_version": self.policy_version,
                   "identities": [asdict(i) for i in self._identities]}
        target = Path(path) if path is not None else self.persistence_path
        if target is None: raise ValueError("persistent split registry path required")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding='utf-8')
        os.replace(temporary, target)
    @classmethod
    def load(cls, path, *, policy_version=None, persistence_path=None):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if isinstance(data, list):
            # Legacy registry files were always strict; never silently relax.
            version, rows = STRICT_FAMILY_DISJOINT_V1, data
        elif isinstance(data, dict) and data.get("schema_version") == 2:
            version, rows = data.get("split_policy_version"), data.get("identities", ())
        else:
            raise ValueError("unsupported split registry schema")
        if policy_version is not None and policy_version != version:
            raise ValueError("split policy migration must be explicit")
        identities = (Identity(i['world'], i['task'], i['family'], i['lineage'],
                               Split(i['split']),
                               None if i.get('purpose') is None else DataPurpose(i['purpose']))
                      for i in rows)
        return cls(identities, policy_version=version, persistence_path=persistence_path)
