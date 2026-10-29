"""Host-side split assignment under an explicit, versioned isolation policy."""
import json
from dataclasses import asdict
from pathlib import Path
from rift.schemas import (DataPurpose, Identity, Split, STRICT_FAMILY_DISJOINT_V1,
                          SPLIT_POLICY_VERSIONS)

class SplitRegistry:
    def __init__(self, identities=(), *, policy_version=STRICT_FAMILY_DISJOINT_V1):
        if policy_version not in SPLIT_POLICY_VERSIONS:
            raise ValueError("unknown split policy version")
        self.policy_version = policy_version
        self._identities = []
        for identity in identities: self.register(identity)
    def register(self, identity):
        if not isinstance(identity, Identity): raise TypeError("Identity required")
        for row in self._identities:
            if not identity.overlaps_under(row, self.policy_version):
                continue
            if identity.split != row.split:
                raise ValueError("cross-split identity overlap under " + self.policy_version)
            if (identity.purpose is not None and row.purpose is not None
                    and identity.purpose != row.purpose):
                raise ValueError("cross-purpose identity overlap under " + self.policy_version)
        if identity not in self._identities: self._identities.append(identity)
    def contains(self, identity):
        return identity in self._identities
    def save(self, path):
        payload = {"schema_version": 2, "split_policy_version": self.policy_version,
                   "identities": [asdict(i) for i in self._identities]}
        Path(path).write_text(json.dumps(payload, sort_keys=True), encoding='utf-8')
    @classmethod
    def load(cls, path, *, policy_version=None):
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
        return cls(identities, policy_version=version)
