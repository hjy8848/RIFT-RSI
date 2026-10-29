import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from rift.schemas import (Checkpoint, DataPurpose, Identity, Outcome, Split, Tokens,
                          STRICT_FAMILY_DISJOINT_V1)
from rift.replay.trajectory import Demonstration

def _encoded(rows):
    return json.dumps([asdict(r) for r in rows], sort_keys=True, ensure_ascii=False, separators=(",", ":"))

def _snapshot_digest(rows, training, split_policy_version=STRICT_FAMILY_DISJOINT_V1):
    value = {"rows": json.loads(_encoded(rows)), "training": [asdict(i) for i in training],
             "split_policy_version": split_policy_version}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()

class ReplayBank:
    def __init__(self, rows=(), *, verifier):
        self._rows = []
        self._verifier = verifier
        for row in rows: self.add(row)
    def add(self, sample):
        if not isinstance(sample, Demonstration): raise TypeError("Demonstration required")
        if self._verifier.verify_demonstration(sample) is not True:
            raise ValueError("replay demonstration evidence was not verified")
        if any(sample.sample_id == r.sample_id for r in self._rows): raise ValueError("duplicate sample")
        # Multiple decision samples from the same episode are allowed; different
        # episodes may not share identity keys to prevent hidden duplication.
        if any(sample.identity.overlaps(r.identity) and sample.episode_id != r.episode_id for r in self._rows):
            raise ValueError("overlapping source episodes")
        if any(sample.episode_id == r.episode_id and sample.identity != r.identity for r in self._rows):
            raise ValueError("episode identity mismatch")
        self._rows.append(sample)
    def freeze(self, training, registry, *, min_samples=1):
        if type(min_samples) is not int or min_samples < 1 or len(self._rows) < min_samples:
            raise ValueError("insufficient replay samples")
        if any(not registry.contains(i) for i in list(training) + [r.identity for r in self._rows]): raise ValueError("all identities must be registered")
        if not training or any(i.split is not Split.TRAIN for i in training): raise ValueError("train identities required")
        if any(r.identity.overlaps_under(t, registry.policy_version)
               for r in self._rows for t in training): raise ValueError("replay overlaps training")
        rows = tuple(sorted(self._rows, key=lambda r: r.sample_id))
        return Snapshot(rows, tuple(training),
                        _snapshot_digest(rows, training, registry.policy_version),
                        registry.policy_version)
    def save(self, path):
        Path(path).write_text(_encoded(self._rows), encoding="utf-8")
    @classmethod
    def load(cls, path, *, verifier):
        return cls(_decode(json.loads(Path(path).read_text(encoding="utf-8"))), verifier=verifier)

def _decode(raw):
    rows = []
    for r in raw:
        i = r["identity"]; t = r["tokens"]
        identity = Identity(i["world"], i["task"], i["family"], i["lineage"],
                            Split(i["split"]),
                            None if i.get("purpose") is None else DataPurpose(i["purpose"]))
        rows.append(Demonstration(r["sample_id"], identity, r["episode_id"], Checkpoint(**r["source_checkpoint"]), Tokens(tuple(t["prompt"]), tuple(t["response"]), tuple(t["mask"])), Outcome(r["outcome"]), r["replay_verified"], r["evidence_kind"]))
    return rows

@dataclass(frozen=True)
class Snapshot:
    rows: tuple
    training: tuple
    digest: str
    split_policy_version: str = STRICT_FAMILY_DISJOINT_V1
    @property
    def replay_digest(self):
        return hashlib.sha256(_encoded(self.rows).encode()).hexdigest()
    def save(self, path):
        if not self.verify(): raise ValueError("invalid snapshot")
        Path(path).write_text(json.dumps({"schema_version": 2,
            "rows": json.loads(_encoded(self.rows)), "training": [asdict(i) for i in self.training],
            "digest": self.digest, "split_policy_version": self.split_policy_version}, sort_keys=True), encoding="utf-8")
    @classmethod
    def load(cls, path, *, verifier):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("schema_version") != 2 or not raw.get("split_policy_version"):
            raise ValueError("legacy replay snapshot needs explicit split-policy migration")
        training = tuple(Identity(i["world"], i["task"], i["family"], i["lineage"], Split(i["split"]),
            None if i.get("purpose") is None else DataPurpose(i["purpose"])) for i in raw["training"])
        snapshot = cls(tuple(ReplayBank(_decode(raw["rows"]), verifier=verifier)._rows), training, raw["digest"])
        snapshot = cls(snapshot.rows, snapshot.training, raw["digest"], raw["split_policy_version"])
        if not snapshot.verify(): raise ValueError("invalid snapshot")
        return snapshot
    def verify(self):
        return (bool(self.rows) and bool(self.training)
                and all(t.split is Split.TRAIN for t in self.training)
                and not any(r.identity.overlaps_under(t, self.split_policy_version)
                            for r in self.rows for t in self.training)
                and _snapshot_digest(self.rows, self.training, self.split_policy_version) == self.digest)
