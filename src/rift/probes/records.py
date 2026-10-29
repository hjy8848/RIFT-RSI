"""Frozen, host-private independent task probes and trial-level evaluations."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite

from rift.banks.records import ObservedTrial
from rift.schemas import Checkpoint, DataPurpose, Identity, Outcome, Split


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class TaskProbe:
    task_ref: str
    identity: Identity
    purpose: DataPurpose
    baseline_checkpoint: Checkpoint
    baseline_trials: tuple[ObservedTrial, ...]
    protocol_ref: str
    protocol_digest: str
    weight: float = 1.0

    def __post_init__(self):
        if (self.purpose not in (DataPurpose.RECOVERY_FEEDBACK,
                                 DataPurpose.RETENTION_FEEDBACK)
                or self.identity.purpose is not DataPurpose.BANK_CANDIDATE
                or self.identity.split is Split.TEST):
            raise ValueError("probe role must use a non-TEST Failure Bank task")
        if (not self.task_ref or not self.protocol_ref or not self.protocol_digest
                or not self.baseline_trials or not isfinite(self.weight) or self.weight <= 0):
            raise ValueError("complete probe baseline required")
        if any(t.task_ref != self.task_ref or t.checkpoint != self.baseline_checkpoint
               or t.protocol_digest != self.protocol_digest for t in self.baseline_trials):
            raise ValueError("probe baseline trials do not match task, checkpoint or protocol")
        if len({t.seed for t in self.baseline_trials}) != len(self.baseline_trials):
            raise ValueError("probe baseline seeds must be unique")
        if any(t.public_summary is not None for t in self.baseline_trials):
            raise ValueError("Designer-private probe cannot include public summaries")

    @property
    def seeds(self):
        return tuple(t.seed for t in self.baseline_trials)

    @property
    def successes(self):
        return sum(t.outcome is Outcome.SUCCESS and t.valid for t in self.baseline_trials)

    @property
    def valid_trials(self):
        return sum(t.valid for t in self.baseline_trials)

    @property
    def unknown_trials(self):
        return len(self.baseline_trials) - self.valid_trials

    @property
    def evidence_refs(self):
        return tuple(t.evidence_ref for t in self.baseline_trials if t.valid)


def _snapshot_payload(snapshot):
    payload = {"snapshot_id": snapshot.snapshot_id, "purpose": snapshot.purpose.value,
        "probes": [asdict(p) for p in snapshot.probes], "random_seed": snapshot.random_seed,
        "split_policy_version": snapshot.split_policy_version,
        "baseline_checkpoint": asdict(snapshot.baseline_checkpoint),
        "protocol_digest": snapshot.protocol_digest, "candidate_count": snapshot.candidate_count,
        "exclusions": snapshot.exclusions}
    if snapshot.target_ability_id is not None or snapshot.priority_task_refs:
        payload["target_ability_id"] = snapshot.target_ability_id
        payload["priority_task_refs"] = snapshot.priority_task_refs
    return payload


@dataclass(frozen=True)
class ProbeSnapshot:
    snapshot_id: str
    purpose: DataPurpose
    probes: tuple[TaskProbe, ...]
    random_seed: int
    split_policy_version: str
    baseline_checkpoint: Checkpoint
    protocol_digest: str
    candidate_count: int
    exclusions: tuple[tuple[str, int], ...]
    frozen_at_ns: int
    content_digest: str
    target_ability_id: str | None = None
    priority_task_refs: tuple[str, ...] = ()

    def __post_init__(self):
        if self.purpose not in (DataPurpose.RECOVERY_FEEDBACK, DataPurpose.RETENTION_FEEDBACK):
            raise ValueError("invalid probe snapshot purpose")
        if (not self.snapshot_id or not self.probes or type(self.random_seed) is not int
                or not self.split_policy_version or not self.protocol_digest
                or type(self.candidate_count) is not int or self.candidate_count < len(self.probes)
                or type(self.frozen_at_ns) is not int or self.frozen_at_ns < 0):
            raise ValueError("incomplete frozen probe snapshot")
        if any(p.purpose is not self.purpose or p.baseline_checkpoint != self.baseline_checkpoint
               or p.protocol_digest != self.protocol_digest for p in self.probes):
            raise ValueError("probe snapshot contains a mismatched baseline")
        if len({p.task_ref for p in self.probes}) != len(self.probes):
            raise ValueError("duplicate probe task")
        if ((self.target_ability_id is None and self.priority_task_refs)
                or (self.target_ability_id is not None and not self.target_ability_id)
                or (self.purpose is DataPurpose.RETENTION_FEEDBACK
                    and (self.target_ability_id is not None or self.priority_task_refs))
                or len(set(self.priority_task_refs)) != len(self.priority_task_refs)
                or not set(self.priority_task_refs).issubset({p.task_ref for p in self.probes})):
            raise ValueError("Recovery ability priority metadata must bind selected Probe tasks")
        expected = sha256(_canonical(_snapshot_payload(self)).encode("utf-8")).hexdigest()
        if self.content_digest != expected:
            raise ValueError("probe snapshot digest mismatch")

    @classmethod
    def create(cls, *, snapshot_id, purpose, probes, random_seed,
               split_policy_version, baseline_checkpoint, protocol_digest,
               candidate_count, exclusions=(), frozen_at_ns=0,
               target_ability_id=None, priority_task_refs=()):
        provisional = cls.__new__(cls)
        values = dict(snapshot_id=snapshot_id, purpose=purpose, probes=tuple(probes),
            random_seed=random_seed, split_policy_version=split_policy_version,
            baseline_checkpoint=baseline_checkpoint, protocol_digest=protocol_digest,
            candidate_count=candidate_count, exclusions=tuple(exclusions),
            frozen_at_ns=frozen_at_ns, content_digest="",
            target_ability_id=target_ability_id,
            priority_task_refs=tuple(priority_task_refs))
        for key, value in values.items(): object.__setattr__(provisional, key, value)
        digest = sha256(_canonical(_snapshot_payload(provisional)).encode("utf-8")).hexdigest()
        values["content_digest"] = digest
        return cls(**values)

    def verify(self):
        return sha256(_canonical(_snapshot_payload(self)).encode("utf-8")).hexdigest() == self.content_digest

    @property
    def target_ability_task_count(self):
        return len(self.priority_task_refs)

    @property
    def random_fallback_task_count(self):
        return len(self.probes) - self.target_ability_task_count


@dataclass(frozen=True)
class ProbeTaskResult:
    task_ref: str
    identity: Identity
    checkpoint: Checkpoint
    trials: tuple[ObservedTrial, ...]
    protocol_digest: str
    phase: str

    def __post_init__(self):
        if self.phase not in ("before", "after") or not self.trials:
            raise ValueError("probe trial results require a before/after phase")
        if any(t.task_ref != self.task_ref or t.checkpoint != self.checkpoint
               or t.protocol_digest != self.protocol_digest for t in self.trials):
            raise ValueError("probe result trials are not bound to task/checkpoint/protocol")
        if len({t.seed for t in self.trials}) != len(self.trials):
            raise ValueError("probe result seeds must be unique")
        if any(t.public_summary is not None for t in self.trials):
            raise ValueError("probe result must remain private")

    @property
    def successes(self):
        return sum(t.outcome is Outcome.SUCCESS and t.valid for t in self.trials)

    @property
    def valid_trials(self):
        return sum(t.valid for t in self.trials)

    @property
    def unknown_trials(self):
        return len(self.trials) - self.valid_trials

    @property
    def success_rate(self):
        return None if self.valid_trials == 0 else self.successes / self.valid_trials

    @property
    def evidence_refs(self):
        return tuple(t.evidence_ref for t in self.trials if t.valid)


@dataclass(frozen=True)
class ProbeResultSet:
    snapshot_digest: str
    purpose: DataPurpose
    checkpoint: Checkpoint
    protocol_digest: str
    phase: str
    results: tuple[ProbeTaskResult, ...]
    evidence_digest: str

    def __post_init__(self):
        if (not self.snapshot_digest or not self.protocol_digest or not self.results
                or self.phase not in ("before", "after")):
            raise ValueError("complete Probe result set required")
        if any(r.phase != self.phase or r.checkpoint != self.checkpoint
               or r.protocol_digest != self.protocol_digest for r in self.results):
            raise ValueError("Probe result set binding mismatch")
        if len({r.task_ref for r in self.results}) != len(self.results):
            raise ValueError("duplicate task result")
        expected = result_set_digest(self.snapshot_digest, self.purpose, self.checkpoint,
            self.protocol_digest, self.phase, self.results)
        if self.evidence_digest != expected:
            raise ValueError("Probe result evidence digest mismatch")

    @property
    def valid_fraction(self):
        total = sum(len(row.trials) for row in self.results)
        return sum(row.valid_trials for row in self.results) / total if total else 0.0

    @property
    def mean_success_rate(self):
        rates = [row.success_rate for row in self.results]
        if not rates or any(rate is None for rate in rates):
            return None
        return sum(rates) / len(rates)


def result_set_digest(snapshot_digest, purpose, checkpoint, protocol_digest, phase, results):
    payload = {"snapshot_digest": snapshot_digest, "purpose": purpose.value,
        "checkpoint": asdict(checkpoint), "protocol_digest": protocol_digest,
        "phase": phase, "results": [asdict(result) for result in results]}
    return sha256(_canonical(payload).encode("utf-8")).hexdigest()


def make_result_set(snapshot, checkpoint, phase, results):
    rows = tuple(results)
    return ProbeResultSet(snapshot.content_digest, snapshot.purpose, checkpoint,
        snapshot.protocol_digest, phase, rows,
        result_set_digest(snapshot.content_digest, snapshot.purpose, checkpoint,
            snapshot.protocol_digest, phase, rows))
