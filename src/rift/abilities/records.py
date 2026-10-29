"""Small persistent records for explicit Designer ability choices."""
from dataclasses import dataclass
from enum import Enum
from math import isfinite

from rift.schemas import Checkpoint


class AbilityStatus(str, Enum):
    PROVISIONAL = "provisional"
    PERSISTENT_WEAKNESS = "persistent_weakness"
    IMPROVING = "improving"
    STABLE = "stable"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class AbilityRevision:
    round_index: int
    previous_description: str
    revised_description: str
    added_failure_task_refs: tuple[str, ...] = ()
    removed_failure_task_refs: tuple[str, ...] = ()

    def __post_init__(self):
        if (type(self.round_index) is not int or self.round_index < 0
                or not self.previous_description or not self.revised_description
                or (self.previous_description == self.revised_description
                    and not self.added_failure_task_refs and not self.removed_failure_task_refs)
                or len(set(self.added_failure_task_refs)) != len(self.added_failure_task_refs)
                or len(set(self.removed_failure_task_refs)) != len(self.removed_failure_task_refs)):
            raise ValueError("invalid explicit ability revision")


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
    failure_task_refs: tuple[str, ...] = ()
    creator_rollout_id: str | None = None
    training_history: tuple[AbilityTrainingEvent, ...] = ()
    evaluation_history: tuple[AbilityEvaluationEvent, ...] = ()
    revisions: tuple[AbilityRevision, ...] = ()

    def __post_init__(self):
        if (not self.ability_id or not self.description or not isinstance(self.status, AbilityStatus)
                or type(self.created_round) is not int or self.created_round < 0
                or type(self.last_observed_round) is not int
                or self.last_observed_round < self.created_round
                or (self.creator_rollout_id is not None and not self.creator_rollout_id)
                or len(set(self.failure_task_refs)) != len(self.failure_task_refs)
                or any(not ref for ref in self.failure_task_refs)):
            raise ValueError("invalid Ability record")


@dataclass(frozen=True)
class AbilityPrompt:
    """Allowlisted existing ability and its explicitly linked failures."""
    ability_id: str
    description: str
    failure_task_refs: tuple[str, ...] = ()

    def __post_init__(self):
        if (not self.ability_id or not self.description
                or len(set(self.failure_task_refs)) != len(self.failure_task_refs)):
            raise ValueError("valid ability option required")
