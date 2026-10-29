"""Capability banks and append-only membership history."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from rift.schemas import DataPurpose
from rift.banks.records import (BankMembershipEvent, CapabilityState,
    DiagnosisExperience, TaskEvidenceRecord)
from rift.schemas import Outcome


def public_projection(record: TaskEvidenceRecord) -> DiagnosisExperience:
    """Return an allowlisted summary; opaque Host fields never get serialized."""
    if record.purpose is not DataPurpose.DIAGNOSIS:
        raise ValueError("only DIAGNOSIS records can be exposed to Designer")
    if record.state() is not CapabilityState.ACTIVE_WEAKNESS:
        raise ValueError("diagnosis record must be a verified weak candidate")
    summaries = [t.public_summary for t in record.trials
                 if t.outcome.value == "FAILURE" and t.valid and t.public_summary is not None]
    if not summaries:
        raise ValueError("verified public failure summary required")
    if summaries[-1].outcome is not Outcome.FAILURE:
        raise ValueError("diagnosis projection must retain the public business failure outcome")
    return DiagnosisExperience(summaries[-1])


class SuccessBank:
    def __init__(self, rows=(), *, success_threshold=0.75):
        self.success_threshold = success_threshold
        self._rows = {}
        for row in rows:
            self.add(row)

    def add(self, row):
        if not isinstance(row, TaskEvidenceRecord):
            raise TypeError("TaskEvidenceRecord required")
        if row.purpose not in (DataPurpose.RETENTION_FEEDBACK, DataPurpose.REPLAY_LOGP_FEEDBACK):
            raise ValueError("SuccessBank only accepts retention or LogP candidates")
        if row.state(success_threshold=self.success_threshold) is not CapabilityState.MASTERED_CANDIDATE:
            raise ValueError("task does not meet the configured success threshold")
        key = (row.task_ref, row.checkpoint)
        if key in self._rows:
            raise ValueError("duplicate task/checkpoint success evidence")
        self._rows[key] = row

    @property
    def records(self):
        return tuple(self._rows.values())


class FailureBank:
    def __init__(self, rows=(), *, failure_threshold=0.25):
        self.failure_threshold = failure_threshold
        self._rows = {}
        for row in rows:
            self.add(row)

    def add(self, row):
        if not isinstance(row, TaskEvidenceRecord):
            raise TypeError("TaskEvidenceRecord required")
        if row.purpose not in (DataPurpose.DIAGNOSIS, DataPurpose.RECOVERY_FEEDBACK):
            raise ValueError("FailureBank only accepts diagnosis or recovery candidates")
        if row.state(failure_threshold=self.failure_threshold) is not CapabilityState.ACTIVE_WEAKNESS:
            raise ValueError("task does not meet the configured failure threshold")
        key = (row.task_ref, row.checkpoint)
        if key in self._rows:
            raise ValueError("duplicate task/checkpoint failure evidence")
        self._rows[key] = row

    @property
    def records(self):
        return tuple(self._rows.values())

    def diagnosis_experiences(self, *, limit, seed):
        rows = [r for r in self.records if r.purpose is DataPurpose.DIAGNOSIS]
        if type(limit) is not int or limit < 1 or type(seed) is not int:
            raise ValueError("positive diagnosis limit and integer seed required")
        import random
        rng = random.Random(seed)
        chosen = rng.sample(rows, min(limit, len(rows)))
        # Projection is the only model-facing representation.
        return tuple(public_projection(row) for row in chosen)


class BankManager:
    """Keep each checkpoint-relative observation and every membership transition."""
    def __init__(self, *, success_threshold=0.75, failure_threshold=0.25):
        if not 0 <= failure_threshold < success_threshold <= 1:
            raise ValueError("bank thresholds must be ordered in [0, 1]")
        self.success_bank = SuccessBank(success_threshold=success_threshold)
        self.failure_bank = FailureBank(failure_threshold=failure_threshold)
        self.records = []
        self.events = []

    def observe(self, record, *, event_id):
        if not isinstance(record, TaskEvidenceRecord) or not event_id:
            raise ValueError("task evidence and membership event ID required")
        if any(r.task_ref == record.task_ref and r.checkpoint == record.checkpoint for r in self.records):
            raise ValueError("duplicate task observation at checkpoint")
        state = record.state(success_threshold=self.success_bank.success_threshold,
                             failure_threshold=self.failure_bank.failure_threshold)
        if state is CapabilityState.MASTERED_CANDIDATE and record.purpose in (
                DataPurpose.RETENTION_FEEDBACK, DataPurpose.REPLAY_LOGP_FEEDBACK):
            self.success_bank.add(record)
            bank = "success"
        elif state is CapabilityState.ACTIVE_WEAKNESS and record.purpose in (
                DataPurpose.DIAGNOSIS, DataPurpose.RECOVERY_FEEDBACK):
            self.failure_bank.add(record)
            bank = "failure"
        elif state is CapabilityState.MASTERED_CANDIDATE and record.purpose is DataPurpose.RECOVERY_FEEDBACK:
            # Keep this as a recovery history event; promotion into a new split
            # requires an explicit future manifest and cannot rewrite this use.
            state = CapabilityState.RECOVERED
            bank = "success"
        else:
            bank = "failure" if record.purpose in (DataPurpose.DIAGNOSIS, DataPurpose.RECOVERY_FEEDBACK) else "success"
        if state is CapabilityState.UNRESOLVED:
            state = CapabilityState.UNRESOLVED
        elif state is CapabilityState.UNCERTAIN:
            state = CapabilityState.UNCERTAIN
        self.records.append(record)
        self.events.append(BankMembershipEvent(event_id, record.task_ref, bank, state,
            record.checkpoint, len(self.events), "bootstrap_or_observation"))
        return state

    def save(self, path):
        """Append-only event snapshot; task/evidence references stay Host-private."""
        from rift.banks.storage import encode_manager
        Path(path).write_text(encode_manager(self), encoding="utf-8")

    @classmethod
    def load(cls, path):
        from rift.banks.storage import decode_manager
        return decode_manager(Path(path).read_text(encoding="utf-8"))


class JsonTaskEvidenceStore:
    """Immutable, content-addressed JSON evidence index for trusted runtime factories."""
    def __init__(self, root):
        self.root = Path(root)

    def _path(self, task_ref, checkpoint):
        raw = json.dumps([task_ref, checkpoint.policy, checkpoint.revision,
            checkpoint.tokenizer, checkpoint.weights_digest], separators=(",", ":"))
        return self.root / (sha256(raw.encode()).hexdigest() + ".json")

    def save(self, record):
        if not isinstance(record, TaskEvidenceRecord):
            raise TypeError("TaskEvidenceRecord required")
        from rift.banks.storage import canonical_record
        payload = canonical_record(record)
        digest = sha256(payload.encode("utf-8")).hexdigest()
        path = self._path(record.task_ref, record.checkpoint)
        self.root.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("digest") != digest:
                raise ValueError("immutable task evidence already exists with different content")
            return digest
        envelope = json.dumps({"schema_version": 1, "digest": digest,
                               "record": json.loads(payload)}, sort_keys=True, ensure_ascii=False)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(envelope)
        return digest

    def load(self, task_ref, checkpoint):
        from rift.banks.storage import canonical_record, record_from_payload
        raw = json.loads(self._path(task_ref, checkpoint).read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1:
            raise ValueError("unsupported task evidence schema")
        record = record_from_payload(raw["record"])
        canonical = canonical_record(record)
        if (record.task_ref != task_ref or record.checkpoint != checkpoint
                or sha256(canonical.encode("utf-8")).hexdigest() != raw.get("digest")):
            raise ValueError("task evidence digest or binding mismatch")
        return record
