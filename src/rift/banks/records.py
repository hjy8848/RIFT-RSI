"""Host-private, provenance-preserving records for evolving capability banks."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from rift.benchmark.public_feedback import PublicSummary
from rift.schemas import Checkpoint, DataPurpose, Identity, Outcome


class EvidenceLevel(str, Enum):
    REAL_SERVICE = "REAL_SERVICE"
    SCRIPTED_USAMI = "SCRIPTED_USAMI"
    FAKE_SCORER = "FAKE_SCORER"
    MOCK_VERL = "MOCK_VERL"


class CapabilityState(str, Enum):
    MASTERED_CANDIDATE = "MASTERED_CANDIDATE"
    ACTIVE_WEAKNESS = "ACTIVE_WEAKNESS"
    UNCERTAIN = "UNCERTAIN"
    UNRESOLVED = "UNRESOLVED"
    RECOVERED = "RECOVERED"


@dataclass(frozen=True)
class ObservedTrial:
    """One complete Usami evaluation result; evidence stays on the Host."""
    task_ref: str
    trial_id: str
    checkpoint: Checkpoint
    seed: int
    outcome: Outcome
    protocol_digest: str
    evidence_ref: str | None
    replay_verified: bool
    evidence_level: EvidenceLevel
    public_summary: PublicSummary | None = None

    def __post_init__(self):
        if (not self.task_ref or not self.trial_id or type(self.seed) is not int or self.seed < 0
                or not self.protocol_digest or not isinstance(self.outcome, Outcome)
                or not isinstance(self.evidence_level, EvidenceLevel)):
            raise ValueError("complete task-trial provenance required")
        if self.outcome is not Outcome.UNKNOWN and (not self.evidence_ref or not self.replay_verified):
            raise ValueError("known business outcomes require replay-verified evidence")
        if self.public_summary is not None and not isinstance(self.public_summary, PublicSummary):
            raise TypeError("public summary projection required")

    @property
    def valid(self):
        return self.outcome is not Outcome.UNKNOWN and self.replay_verified and bool(self.evidence_ref)


@dataclass(frozen=True)
class TaskEvidenceRecord:
    """Private task replay binding and its checkpoint-relative capability estimate."""
    task_ref: str
    identity: Identity
    purpose: DataPurpose
    checkpoint: Checkpoint
    protocol_digest: str
    trials: tuple[ObservedTrial, ...]
    observed_at: str

    def __post_init__(self):
        if (not self.task_ref or not isinstance(self.identity, Identity)
                or not isinstance(self.purpose, DataPurpose) or not self.checkpoint
                or not self.protocol_digest or not self.trials or not self.observed_at):
            raise ValueError("complete host task evidence required")
        if self.identity.purpose is not None and self.identity.purpose is not self.purpose:
            raise ValueError("task identity purpose mismatch")
        if any(t.task_ref != self.task_ref or t.checkpoint != self.checkpoint
               or t.protocol_digest != self.protocol_digest for t in self.trials):
            raise ValueError("task trials do not share the bound task, checkpoint and protocol")
        if self.purpose is not DataPurpose.DIAGNOSIS and any(t.public_summary is not None for t in self.trials):
            raise ValueError("private feedback tasks cannot carry a Designer-visible summary")
        if len({t.trial_id for t in self.trials}) != len(self.trials):
            raise ValueError("duplicate task trial")
        if len({t.seed for t in self.trials}) != len(self.trials):
            raise ValueError("duplicate evaluation seed")

    @property
    def successes(self):
        return sum(t.outcome is Outcome.SUCCESS and t.valid for t in self.trials)

    @property
    def failures(self):
        return sum(t.outcome is Outcome.FAILURE and t.valid for t in self.trials)

    @property
    def unknown_trials(self):
        return sum(not t.valid for t in self.trials)

    @property
    def valid_trials(self):
        return self.successes + self.failures

    @property
    def success_rate(self) -> float | None:
        return None if not self.valid_trials else self.successes / self.valid_trials

    def state(self, *, success_threshold=0.75, failure_threshold=0.25):
        if not 0 <= failure_threshold < success_threshold <= 1:
            raise ValueError("failure and success thresholds must be ordered in [0, 1]")
        rate = self.success_rate
        if rate is None:
            return CapabilityState.UNRESOLVED
        if rate >= success_threshold:
            return CapabilityState.MASTERED_CANDIDATE
        if rate <= failure_threshold:
            return CapabilityState.ACTIVE_WEAKNESS
        return CapabilityState.UNCERTAIN

    @property
    def evidence_levels(self):
        return frozenset(t.evidence_level for t in self.trials)


@dataclass(frozen=True)
class DiagnosisExperience:
    """Explicit public allowlist: no TaskRef, identity, checkpoint hash or private evidence."""
    summary: PublicSummary

    def __post_init__(self):
        if not isinstance(self.summary, PublicSummary):
            raise TypeError("PublicSummary required")


@dataclass(frozen=True)
class BankMembershipEvent:
    event_id: str
    task_ref: str
    bank: str
    state: CapabilityState
    checkpoint: Checkpoint
    sequence: int
    reason: str

    def __post_init__(self):
        if (not self.event_id or not self.task_ref or self.bank not in ("success", "failure")
                or not isinstance(self.state, CapabilityState) or type(self.sequence) is not int
                or self.sequence < 0 or not self.reason):
            raise ValueError("invalid bank membership event")
