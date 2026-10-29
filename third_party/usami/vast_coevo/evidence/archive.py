"""Append-only evidence linked to complete, hash-checked native traces."""
from __future__ import annotations
import json
import re
from pathlib import Path

from ..models import EpisodeEvidence
from ..schema import atomic_json, digest
from .signatures import failure_counts


class EvidenceArchive:
    def __init__(self, root):
        self.root = Path(root)

    @staticmethod
    def verify_artifact(ref):
        raw = json.loads(Path(ref.path).read_text(encoding="utf-8"))
        if digest(raw) != ref.sha256:
            raise ValueError("native artifact hash mismatch: " + ref.path)
        return raw

    def save(self, evidence):
        evidence = EpisodeEvidence.model_validate(evidence.model_dump(mode="json"))
        self.verify_artifact(evidence.trace)
        for failure in evidence.failures:
            self.verify_artifact(failure.trace)
        atomic_json(self.root / (evidence.evidence_hash + ".json"), evidence.model_dump(mode="json"), immutable=True)
        return evidence.evidence_hash

    def load(self, evidence_hash):
        if not re.fullmatch(r"[0-9a-f]{64}", evidence_hash):
            raise ValueError("invalid evidence hash")
        evidence = EpisodeEvidence.model_validate_json((self.root / (evidence_hash + ".json")).read_text())
        if evidence.evidence_hash != evidence_hash:
            raise ValueError("evidence filename/hash mismatch")
        self.verify_artifact(evidence.trace)
        return evidence

    def query(self, domain=None, rule=None, operation=None, generation=None):
        rows = []
        for path in sorted(self.root.glob("*.json")):
            item = self.load(path.stem)
            if domain is not None and item.config.domain != domain:
                continue
            if generation is not None and not item.episode_id.startswith("g" + str(generation) + "."):
                continue
            if rule is not None and not any(rule in f.reference_codes for f in item.failures):
                continue
            if operation is not None and not any(f.operation == operation for f in item.failures):
                continue
            rows.append(item)
        return tuple(rows)

    def counts(self, **filters):
        return failure_counts(self.query(**filters))
