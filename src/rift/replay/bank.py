import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from rift.schemas import Identity, Outcome, Split, Tokens
from rift.replay.trajectory import Demonstration

def _encoded(rows):
    return json.dumps([asdict(r) for r in rows], sort_keys=True, ensure_ascii=False, separators=(",", ":"))

def _snapshot_digest(rows, training):
    value = {"rows": json.loads(_encoded(rows)), "training": [asdict(i) for i in training]}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()

class ReplayBank:
    def __init__(self, rows=()):
        self._rows = []
        for row in rows: self.add(row)
    def add(self, sample):
        if not isinstance(sample, Demonstration): raise TypeError("Demonstration required")
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
        if any(r.identity.overlaps(t) for r in self._rows for t in training): raise ValueError("replay overlaps training")
        rows = tuple(sorted(self._rows, key=lambda r: r.sample_id))
        return Snapshot(rows, tuple(training), _snapshot_digest(rows, training))
    def save(self, path):
        Path(path).write_text(_encoded(self._rows), encoding="utf-8")
    @classmethod
    def load(cls, path):
        return cls(_decode(json.loads(Path(path).read_text(encoding="utf-8"))))

def _decode(raw):
    rows = []
    for r in raw:
        i = r["identity"]; t = r["tokens"]
        rows.append(Demonstration(r["sample_id"], Identity(i["world"], i["task"], i["family"], i["lineage"], Split(i["split"])), r["episode_id"], r["source_checkpoint"], Tokens(tuple(t["prompt"]), tuple(t["response"]), tuple(t["mask"])), Outcome(r["outcome"]), r["replay_verified"], r["evidence_kind"]))
    return rows

@dataclass(frozen=True)
class Snapshot:
    rows: tuple
    training: tuple
    digest: str
    @property
    def replay_digest(self):
        return hashlib.sha256(_encoded(self.rows).encode()).hexdigest()
    def save(self, path):
        if not self.verify(): raise ValueError("invalid snapshot")
        Path(path).write_text(json.dumps({"rows": json.loads(_encoded(self.rows)), "training": [asdict(i) for i in self.training], "digest": self.digest}, sort_keys=True), encoding="utf-8")
    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        training = tuple(Identity(i["world"], i["task"], i["family"], i["lineage"], Split(i["split"])) for i in raw["training"])
        snapshot = cls(tuple(ReplayBank(_decode(raw["rows"]))._rows), training, raw["digest"])
        if not snapshot.verify(): raise ValueError("invalid snapshot")
        return snapshot
    def verify(self):
        return bool(self.rows) and bool(self.training) and all(t.split is Split.TRAIN for t in self.training) and not any(r.identity.overlaps(t) for r in self.rows for t in self.training) and _snapshot_digest(self.rows, self.training) == self.digest
