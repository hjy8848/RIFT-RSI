"""Small versioned JSON codecs for host-owned bank evidence."""
from dataclasses import asdict
from hashlib import sha256
import json

from rift.benchmark.public_feedback import PublicEvent, PublicSummary
from rift.banks.records import (BankMembershipEvent, CapabilityState,
    EvidenceLevel, ObservedTrial, TaskEvidenceRecord)
from rift.schemas import Checkpoint, DataPurpose, Identity, Outcome, Split


def _identity(value):
    return Identity(value["world"], value["task"], value["family"], value["lineage"],
        Split(value["split"]), None if value.get("purpose") is None
        else DataPurpose(value["purpose"]))


def record_from_payload(value):
    trials = []
    for row in value["trials"]:
        summary = row.get("public_summary")
        public = None if summary is None else PublicSummary(
            tuple(PublicEvent(**event) for event in summary["events"]),
            Outcome(summary["outcome"]))
        trials.append(ObservedTrial(row["task_ref"], row["trial_id"],
            Checkpoint(**row["checkpoint"]), row["seed"], Outcome(row["outcome"]),
            row["protocol_digest"], row.get("evidence_ref"), row["replay_verified"],
            EvidenceLevel(row["evidence_level"]), public))
    return TaskEvidenceRecord(value["task_ref"], _identity(value["identity"]),
        DataPurpose(value["purpose"]), Checkpoint(**value["checkpoint"]),
        value["protocol_digest"], tuple(trials), value["observed_at"])


def canonical_record(record):
    return json.dumps(asdict(record), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def encode_manager(manager):
    payload = {"schema_version": 1,
        "success_threshold": manager.success_bank.success_threshold,
        "failure_threshold": manager.failure_bank.failure_threshold,
        "records": [json.loads(canonical_record(row)) for row in manager.records],
        "events": [asdict(event) for event in manager.events]}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    payload["manifest_digest"] = sha256(encoded.encode("utf-8")).hexdigest()
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def decode_manager(raw):
    from rift.banks.core import BankManager
    value = json.loads(raw)
    if value.get("schema_version") != 1:
        raise ValueError("unsupported bank manifest schema")
    digest = value.pop("manifest_digest", None)
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if digest != sha256(encoded.encode("utf-8")).hexdigest():
        raise ValueError("bank manifest digest mismatch")
    manager = BankManager(success_threshold=value["success_threshold"],
                          failure_threshold=value["failure_threshold"])
    for index, (record, event) in enumerate(zip(value["records"], value["events"])):
        item = record_from_payload(record)
        if (event["task_ref"] != item.task_ref or event["checkpoint"] != asdict(item.checkpoint)
                or event["sequence"] != index):
            raise ValueError("bank membership event binding mismatch")
        if item.purpose in (DataPurpose.RETENTION_FEEDBACK, DataPurpose.REPLAY_LOGP_FEEDBACK) and item.state(success_threshold=manager.success_bank.success_threshold) is CapabilityState.MASTERED_CANDIDATE:
            manager.success_bank.add(item)
        if item.purpose in (DataPurpose.DIAGNOSIS, DataPurpose.RECOVERY_FEEDBACK) and item.state(failure_threshold=manager.failure_bank.failure_threshold) is CapabilityState.ACTIVE_WEAKNESS:
            manager.failure_bank.add(item)
        manager.records.append(item)
    manager.events = [BankMembershipEvent(e["event_id"], e["task_ref"], e["bank"],
        CapabilityState(e["state"]), Checkpoint(**e["checkpoint"]), e["sequence"], e["reason"])
        for e in value["events"]]
    if len(manager.records) != len(manager.events):
        raise ValueError("bank manifest is missing membership history")
    return manager
