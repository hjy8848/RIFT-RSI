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
    response_id: str | None = None
    reward_ref: str | None = None
    target_ability_id: str | None = None
    prompt_ability_id: str | None = None
    def __post_init__(self):
        if not all((self.task_id, self.episode_id, self.reward_source)) or not isfinite(self.reward):
            raise ValueError("rollout provenance and finite reward required")
        if self.group_id is not None and not self.group_id:
            raise ValueError("empty group ID")
        if self.response_id == '' or self.reward_ref == '':
            raise ValueError("empty provenance reference")
        if ((self.target_ability_id is not None and not self.target_ability_id)
                or (self.prompt_ability_id is not None and not self.prompt_ability_id)):
            raise ValueError("empty ability attribution reference")

@dataclass(frozen=True)
class UpdateRequest:
    policy_before: Checkpoint
    package_id: str
    rollouts: tuple[Rollout, ...]
    algorithm: str
    advantage_baseline: float | None = None
    reward_protocol: str | None = None
    credit_assignment_mode: str | None = None
    def __post_init__(self):
        if not self.package_id or not self.algorithm or not self.rollouts:
            raise ValueError("update requires package, algorithm and rollouts")
        if self.advantage_baseline is not None and not isfinite(self.advantage_baseline):
            raise ValueError("finite scalar advantage baseline required")
        if any(r.policy != self.policy_before for r in self.rollouts):
            raise ValueError("rollouts must match checkpoint")
        if self.algorithm.lower() == "reinforce_baseline":
            if (self.advantage_baseline is None or self.reward_protocol != "dual_bank_outcome_v1"
                    or self.credit_assignment_mode != "sequential_curriculum"
                    or any(r.group_id is not None for r in self.rollouts)):
                raise ValueError("sequential REINFORCE baseline needs explicit dual-bank protocol and no fake groups")
        if self.algorithm.lower() == "grpo":
            groups = {}
            for r in self.rollouts:
                if not r.group_id: raise ValueError("GRPO requires explicit groups")
                groups.setdefault(r.group_id, []).append(r)
            if any(len({r.episode_id for r in g}) < 2 or len({r.tokens.prompt for r in g}) != 1 for g in groups.values()):
                raise ValueError("GRPO groups require distinct rollouts from the same prompt")
            if any(len({r.reward_source for r in g}) != 1 for g in groups.values()):
                raise ValueError("GRPO group members need a common, comparable reward source")
            if any(len({r.policy for r in g}) != 1 for g in groups.values()):
                raise ValueError("GRPO group members must use one policy checkpoint")

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

class UpdateVerifier(Protocol):
    """Trusted backend evidence check, including the request, run and both weights.

    A checkpoint's existence does not establish that a training update produced
    it. No production verifier is provided until a trainer is integrated.
    """
    def verify_update(self, update: CompletedUpdate) -> bool: ...

class DesignerActionVerifier(Protocol):
    """Verify the package action tokens and generating policy against its rollout."""
    def verify_designer_action(self, tokens: Tokens, checkpoint: Checkpoint,
                               package_id: str, rollout_id: str, group_id: str | None,
                               target_ability_id: str | None = None) -> bool: ...
    def verify_designer_hypothesis(self, rollout_id: str, hypothesis: str) -> bool: ...
