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

@dataclass(frozen=True)
class Identity:
    world: str
    task: str
    family: str
    lineage: str
    split: Split
    def __post_init__(self):
        if not all((self.world, self.task, self.family, self.lineage)) or not isinstance(self.split, Split):
            raise ValueError("complete split and provenance required")
    def overlaps(self, other):
        return any(getattr(self, k) == getattr(other, k) for k in ("world", "task", "family", "lineage"))

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
