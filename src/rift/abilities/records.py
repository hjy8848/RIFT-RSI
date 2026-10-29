"""Versioned host-side records for uncertain, revisable ability hypotheses."""
from dataclasses import dataclass
from enum import Enum
from math import isfinite

from rift.schemas import Checkpoint, Identity


class AbilityStatus(str, Enum):
    PROVISIONAL = "provisional"
    PERSISTENT_WEAKNESS = "persistent_weakness"
    IMPROVING = "improving"
    STABLE = "stable"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class AbilityCandidate:
    """Only the non-sensitive description is passed to an attribution judge."""
    ability_id: str
    description: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class AbilityEvidenceLink:
    task_ref: str
    evidence_refs: tuple[str, ...]
    confidence: float
    round_index: int
    rationale: str

    def __post_init__(self):
        if (not self.task_ref or not self.evidence_refs or len(set(self.evidence_refs)) != len(self.evidence_refs)
                or not isfinite(self.confidence) or not 0 <= self.confidence <= 1
                or type(self.round_index) is not int or self.round_index < 0 or not self.rationale):
            raise ValueError("invalid ability-to-failure evidence link")


@dataclass(frozen=True)
class AbilityRevision:
    round_index: int
    previous_description: str
    revised_description: str
    rationale: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self):
        if (type(self.round_index) is not int or self.round_index < 0
                or not self.previous_description or not self.revised_description
                or self.previous_description == self.revised_description or not self.rationale):
            raise ValueError("invalid ability hypothesis revision")


@dataclass(frozen=True)
class AbilityAttributionRecord:
    hypothesis: str
    ability_id: str
    round_index: int
    confidence: float
    decision: str
    method: str
    rationale: str
    evidence_refs: tuple[str, ...]
    input_evidence_refs: tuple[str, ...] = ()

    def __post_init__(self):
        if (not self.hypothesis or not self.ability_id or type(self.round_index) is not int
                or self.round_index < 0 or not isfinite(self.confidence)
                or not 0 <= self.confidence <= 1
                or self.decision not in ("matched", "provisional")
                or not self.method or not self.rationale
                or len(set(self.evidence_refs)) != len(self.evidence_refs)
                or len(set(self.input_evidence_refs)) != len(self.input_evidence_refs)):
            raise ValueError("invalid ability attribution record")


@dataclass(frozen=True)
class AbilityTrainingEvent:
    round_index: int
    package_id: str
    service_update_id: str
    reward: float

    def __post_init__(self):
        if (type(self.round_index) is not int or self.round_index < 0 or not self.package_id
                or not self.service_update_id or not isfinite(self.reward)):
            raise ValueError("invalid ability training history event")


@dataclass(frozen=True)
class AbilityEvaluationEvent:
    round_index: int
    service_update_id: str
    checkpoint: Checkpoint
    recovery_before_sr: float
    recovery_after_sr: float
    retention_before_sr: float
    retention_after_sr: float
    recovery_task_refs: tuple[str, ...]
    recovery_evidence_refs: tuple[str, ...]
    checkpoint_before: Checkpoint | None = None
    retention_task_refs: tuple[str, ...] = ()
    retention_evidence_refs: tuple[str, ...] = ()
    target_ability_task_refs: tuple[str, ...] = ()
    fallback_task_refs: tuple[str, ...] = ()
    target_recovery_before_sr: float | None = None
    target_recovery_after_sr: float | None = None

    def __post_init__(self):
        rates = (self.recovery_before_sr, self.recovery_after_sr,
                 self.retention_before_sr, self.retention_after_sr)
        if (type(self.round_index) is not int or self.round_index < 0 or not self.service_update_id
                or any(not isfinite(value) or not 0 <= value <= 1 for value in rates)
                or not self.recovery_task_refs or not self.recovery_evidence_refs
                or (self.checkpoint_before is not None
                    and not isinstance(self.checkpoint_before, Checkpoint))
                or not set(self.target_ability_task_refs).issubset(self.recovery_task_refs)
                or not set(self.fallback_task_refs).issubset(self.recovery_task_refs)
                or (self.target_recovery_before_sr is not None
                    and (not isfinite(self.target_recovery_before_sr)
                         or not 0 <= self.target_recovery_before_sr <= 1))
                or (self.target_recovery_after_sr is not None
                    and (not isfinite(self.target_recovery_after_sr)
                         or not 0 <= self.target_recovery_after_sr <= 1))):
            raise ValueError("invalid ability evaluation history event")


@dataclass(frozen=True)
class AbilityRecord:
    ability_id: str
    description: str
    status: AbilityStatus
    created_round: int
    last_observed_round: int
    aliases: tuple[str, ...] = ()
    previous_descriptions: tuple[str, ...] = ()
    failure_task_refs: tuple[str, ...] = ()
    failure_links: tuple[AbilityEvidenceLink, ...] = ()
    training_history: tuple[AbilityTrainingEvent, ...] = ()
    evaluation_history: tuple[AbilityEvaluationEvent, ...] = ()
    revisions: tuple[AbilityRevision, ...] = ()

    def __post_init__(self):
        if (not self.ability_id or not self.description or not isinstance(self.status, AbilityStatus)
                or type(self.created_round) is not int or self.created_round < 0
                or type(self.last_observed_round) is not int
                or self.last_observed_round < self.created_round
                or len(set(self.failure_task_refs)) != len(self.failure_task_refs)
                or any(not ref for ref in self.aliases + self.previous_descriptions)):
            raise ValueError("invalid persistent Ability Registry record")
        linked = {row.task_ref for row in self.failure_links}
        if not linked.issubset(set(self.failure_task_refs)):
            raise ValueError("failure links must be indexed in failure_task_refs")


@dataclass(frozen=True)
class AbilityPrompt:
    """Public target context supplied to the Designer's prompt encoder."""
    ability_id: str
    description: str

    def __post_init__(self):
        if not self.ability_id or not self.description:
            raise ValueError("Designer target ability context required")
