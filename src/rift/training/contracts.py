from dataclasses import dataclass
from math import isfinite
from typing import Protocol, Sequence
from rift.schemas import Checkpoint, Tokens

@dataclass(frozen=True)
class Rollout:
    tokens: Tokens
    policy: Checkpoint
    task_id: str
    episode_id: str
    reward: float
    reward_source: str
    group_id: str | None = None
    def __post_init__(self):
        if not all((self.task_id, self.episode_id, self.reward_source)) or not isfinite(self.reward):
            raise ValueError("rollout provenance and finite reward required")
        if self.group_id is not None and not self.group_id:
            raise ValueError("empty group ID")

@dataclass(frozen=True)
class UpdateRequest:
    policy_before: Checkpoint
    package_id: str
    rollouts: tuple[Rollout, ...]
    algorithm: str
    def __post_init__(self):
        if not self.package_id or not self.algorithm or not self.rollouts:
            raise ValueError("update requires package, algorithm and rollouts")
        if any(r.policy != self.policy_before for r in self.rollouts):
            raise ValueError("rollouts must match checkpoint")
        if self.algorithm.lower() == "grpo":
            groups = {}
            for r in self.rollouts:
                if not r.group_id: raise ValueError("GRPO requires explicit groups")
                groups.setdefault(r.group_id, []).append(r)
            if any(len({r.episode_id for r in g}) < 2 or len({r.task_id for r in g}) != 1 for g in groups.values()):
                raise ValueError("GRPO groups require multiple samples from the same task")

@dataclass(frozen=True)
class CompletedUpdate:
    request: UpdateRequest
    policy_after: Checkpoint
    run_id: str
    evidence_ref: str
    def __post_init__(self):
        before = self.request.policy_before
        after = self.policy_after
        if not self.run_id or not self.evidence_ref or before.policy != after.policy or before.tokenizer != after.tokenizer or before.weights_digest == after.weights_digest or before.revision == after.revision:
            raise ValueError("completed update requires distinct same-policy weights and evidence")

class ServiceRLBackend(Protocol):
    def train(self, request: UpdateRequest) -> CompletedUpdate: ...
class DesignerRLBackend(Protocol):
    def train(self, request: UpdateRequest) -> CompletedUpdate: ...
class PolicyScorer(Protocol):
    def score(self, checkpoint: Checkpoint, batch: Sequence[Tokens]) -> Sequence[Sequence[float]]: ...
class CheckpointStore(Protocol):
    def resolve(self, checkpoint: Checkpoint) -> str: ...
    def verify(self, checkpoint: Checkpoint) -> bool: ...
