from dataclasses import dataclass
from enum import Enum

class Outcome(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    UNKNOWN = "UNKNOWN"

class Split(str, Enum):
    TRAIN = "train"
    REPLAY = "replay_feedback"
    VALIDATION = "validation"
    TEST = "test"

class DataPurpose(str, Enum):
    """Logical use of a task without changing the legacy coarse Split enum."""
    TRAIN = "train"
    DIAGNOSIS = "diagnosis"
    RECOVERY_FEEDBACK = "recovery_feedback"
    RETENTION_FEEDBACK = "retention_feedback"
    REPLAY_LOGP_FEEDBACK = "replay_logp_feedback"
    VALIDATION = "validation"
    TEST = "test"

STRICT_FAMILY_DISJOINT_V1 = "strict_family_disjoint_v1"
INSTANCE_DISJOINT_WITHIN_FAMILY_V1 = "instance_disjoint_within_family_v1"
SPLIT_POLICY_VERSIONS = frozenset((STRICT_FAMILY_DISJOINT_V1, INSTANCE_DISJOINT_WITHIN_FAMILY_V1))

@dataclass(frozen=True)
class Identity:
    world: str
    task: str
    family: str
    lineage: str
    split: Split
    purpose: DataPurpose | None = None
    def __post_init__(self):
        if (not all((self.world, self.task, self.family, self.lineage))
                or not isinstance(self.split, Split)
                or (self.purpose is not None and not isinstance(self.purpose, DataPurpose))):
            raise ValueError("complete split and provenance required")
    def overlaps(self, other):
        return any(getattr(self, k) == getattr(other, k) for k in ("world", "task", "family", "lineage"))
    def overlaps_under(self, other, policy_version=STRICT_FAMILY_DISJOINT_V1):
        if policy_version == STRICT_FAMILY_DISJOINT_V1:
            return self.overlaps(other)
        if policy_version == INSTANCE_DISJOINT_WITHIN_FAMILY_V1:
            # Family is a broad capability category and may be shared across
            # disjoint instances. World, task and source lineage stay isolated.
            return any(getattr(self, key) == getattr(other, key)
                       for key in ("world", "task", "lineage"))
        raise ValueError("unknown split policy version")

@dataclass(frozen=True)
class Checkpoint:
    policy: str
    revision: str
    tokenizer: str
    weights_digest: str
    def __post_init__(self):
        if not all((self.policy, self.revision, self.tokenizer, self.weights_digest)):
            raise ValueError("complete checkpoint identity required")

@dataclass(frozen=True)
class Tokens:
    prompt: tuple[int, ...]
    response: tuple[int, ...]
    mask: tuple[bool, ...]
    def __post_init__(self):
        if not self.prompt or not self.response or len(self.response) != len(self.mask):
            raise ValueError("prompt, response and aligned mask required")
        if any(type(x) is not int or x < 0 for x in self.prompt + self.response):
            raise ValueError("invalid token IDs")
        if any(type(x) is not bool for x in self.mask) or not any(self.mask):
            raise ValueError("mask must contain selected action tokens")
