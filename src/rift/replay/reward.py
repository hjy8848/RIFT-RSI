from dataclasses import dataclass
from math import isfinite, isclose
from typing import Protocol
from rift.training.contracts import CompletedUpdate

@dataclass(frozen=True)
class SampleScore:
    sample_id: str
    before: float
    after: float
    delta: float

@dataclass(frozen=True)
class ReplayReward:
    package_id: str
    update_run_id: str
    snapshot_digest: str
    binding_digest: str
    normalization: str
    samples: tuple[SampleScore, ...]
    mean: float
    def __post_init__(self):
        if not self.package_id or not self.update_run_id or not self.snapshot_digest or not self.binding_digest or self.normalization not in ("sum", "mean") or not self.samples: raise ValueError("complete replay reward required")
        if len({r.sample_id for r in self.samples}) != len(self.samples): raise ValueError("duplicate scored sample")
        if any(not all(isfinite(v) for v in (r.before, r.after, r.delta)) or not isclose(r.delta, r.after-r.before, abs_tol=1e-12) for r in self.samples): raise ValueError("invalid sample score")
        if not isfinite(self.mean) or not isclose(self.mean, sum(r.delta for r in self.samples)/len(self.samples), abs_tol=1e-12): raise ValueError("invalid aggregate reward")

class ReplayRewardVerifier(Protocol):
    """Verify trusted score provenance for an update and frozen replay binding."""
    def verify_reward(self, reward: ReplayReward, update: CompletedUpdate) -> bool: ...

def score_update(update, snapshot, scorer, checkpoint_store, *, update_verifier, replay_verifier, normalization="sum"):
    if not isinstance(update, CompletedUpdate): raise ValueError("completed Service update required")
    if update_verifier.verify_update(update) is not True:
        raise ValueError("Service update evidence was not verified")
    if not snapshot.verify(): raise ValueError("invalid replay snapshot")
    if any(replay_verifier.verify_demonstration(row) is not True for row in snapshot.rows):
        raise ValueError("replay demonstration evidence was not verified")
    if {r.task_id for r in update.request.rollouts} != {i.task for i in snapshot.training}:
        raise ValueError("training identities do not match the update tasks")
    if any(r.reward_source != "business_outcome_v2" or not r.response_id or r.reward_ref != r.episode_id for r in update.request.rollouts): raise ValueError("Service update must use linked business rewards and responses")
    if normalization not in ("sum", "mean"): raise ValueError("unknown normalization")
    before = update.request.policy_before; after = update.policy_after
    if any(r.source_checkpoint.policy != before.policy or r.source_checkpoint.tokenizer != before.tokenizer for r in snapshot.rows):
        raise ValueError("replay tokens require the same Service policy and tokenizer")
    if not checkpoint_store.verify(before) or not checkpoint_store.verify(after):
        raise ValueError("checkpoint verification failed")
    batch = [r.tokens for r in snapshot.rows]
    b = scorer.score(before, batch); a = scorer.score(after, batch)
    if len(b) != len(batch) or len(a) != len(batch): raise ValueError("score batch size differs")
    rows = []
    for sample, bs, ans in zip(snapshot.rows, b, a):
        n = len(sample.tokens.response)
        if len(bs) != n or len(ans) != n: raise ValueError("token scores not aligned")
        if any(not isfinite(v) or v > 0 for v in list(bs) + list(ans)):
            raise ValueError("invalid log probability")
        selected = sample.tokens.mask
        x = sum(v for v, keep in zip(bs, selected) if keep)
        y = sum(v for v, keep in zip(ans, selected) if keep)
        if normalization == "mean":
            x /= sum(selected); y /= sum(selected)
        rows.append(SampleScore(sample.sample_id, x, y, y-x))
    return ReplayReward(update.request.package_id, update.run_id, snapshot.replay_digest, snapshot.digest, normalization, tuple(rows), sum(r.delta for r in rows)/len(rows))
