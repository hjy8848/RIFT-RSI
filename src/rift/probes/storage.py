"""Immutable JSON persistence for Probe snapshots and trial evidence."""
from dataclasses import asdict
import json
from pathlib import Path

from rift.banks.records import EvidenceLevel, ObservedTrial
from rift.benchmark.public_feedback import PublicEvent, PublicSummary
from rift.probes.records import (ProbeResultSet, ProbeSnapshot, ProbeTaskResult,
    TaskProbe, result_set_digest)
from rift.schemas import Checkpoint, DataPurpose, Identity, Outcome, Split


def _identity(raw):
    return Identity(raw["world"], raw["task"], raw["family"], raw["lineage"],
        Split(raw["split"]), None if raw.get("purpose") is None else DataPurpose(raw["purpose"]))


def _trial(raw):
    summary = raw.get("public_summary")
    public = None if summary is None else PublicSummary(
        tuple(PublicEvent(**row) for row in summary["events"]), Outcome(summary["outcome"]))
    return ObservedTrial(raw["task_ref"], raw["trial_id"], Checkpoint(**raw["checkpoint"]),
        raw["seed"], Outcome(raw["outcome"]), raw["protocol_digest"],
        raw.get("evidence_ref"), raw["replay_verified"], EvidenceLevel(raw["evidence_level"]), public)


def _probe(raw):
    return TaskProbe(raw["task_ref"], _identity(raw["identity"]), DataPurpose(raw["purpose"]),
        Checkpoint(**raw["baseline_checkpoint"]), tuple(_trial(row) for row in raw["baseline_trials"]),
        raw["protocol_ref"], raw["protocol_digest"], raw.get("weight", 1.0))


def _snapshot(raw):
    return ProbeSnapshot(raw["snapshot_id"], DataPurpose(raw["purpose"]),
        tuple(_probe(row) for row in raw["probes"]), raw["random_seed"],
        raw["split_policy_version"], Checkpoint(**raw["baseline_checkpoint"]),
        raw["protocol_digest"], raw["candidate_count"],
        tuple((row[0], row[1]) for row in raw["exclusions"]),
        raw["frozen_at_ns"], raw["content_digest"],
        raw.get("target_ability_id"), tuple(raw.get("priority_task_refs", ())))


def _result_set(raw):
    results = tuple(ProbeTaskResult(row["task_ref"], _identity(row["identity"]),
        Checkpoint(**row["checkpoint"]), tuple(_trial(t) for t in row["trials"]),
        row["protocol_digest"], row["phase"]) for row in raw["results"])
    return ProbeResultSet(raw["snapshot_digest"], DataPurpose(raw["purpose"]),
        Checkpoint(**raw["checkpoint"]), raw["protocol_digest"], raw["phase"],
        results, raw["evidence_digest"])


class JsonProbeEvidenceStore:
    """Content-addressed, write-once evidence used by reward verification and resume."""
    def __init__(self, root):
        self.root = Path(root)

    def _write_once(self, directory, name, payload):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (name + ".json")
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if path.exists():
            if path.read_text(encoding="utf-8") != encoded:
                raise ValueError("immutable Probe evidence already exists with different content")
            return str(path.resolve())
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        return str(path.resolve())

    def save_snapshot(self, snapshot):
        if not snapshot.verify(): raise ValueError("invalid Probe snapshot")
        payload = {"schema_version": 1, **asdict(snapshot)}
        self._write_once(self.root / "snapshots", snapshot.content_digest, payload)
        return snapshot.content_digest

    def load_snapshot(self, digest):
        raw = json.loads((self.root / "snapshots" / (digest + ".json")).read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1: raise ValueError("unsupported Probe snapshot schema")
        snapshot = _snapshot(raw)
        if snapshot.content_digest != digest or not snapshot.verify():
            raise ValueError("Probe snapshot content digest mismatch")
        return snapshot

    def save_results(self, result_set):
        if result_set.evidence_digest != result_set_digest(result_set.snapshot_digest,
                result_set.purpose, result_set.checkpoint, result_set.protocol_digest,
                result_set.phase, result_set.results):
            raise ValueError("invalid Probe result evidence")
        payload = {"schema_version": 1, **asdict(result_set)}
        self._write_once(self.root / "results", result_set.evidence_digest, payload)
        return result_set.evidence_digest

    def load_results(self, evidence_digest):
        raw = json.loads((self.root / "results" / (evidence_digest + ".json")).read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1: raise ValueError("unsupported Probe result schema")
        result = _result_set(raw)
        if result.evidence_digest != evidence_digest:
            raise ValueError("Probe result evidence key mismatch")
        return result
