from dataclasses import dataclass
from math import isfinite, isclose
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

def score_update(update, snapshot, scorer, checkpoint_store, *, normalization="sum"):
    if not isinstance(update, CompletedUpdate): raise ValueError("completed Service update required")
    if not snapshot.verify(): raise ValueError("invalid replay snapshot")
    if {r.task_id for r in update.request.rollouts} - {i.task for i in snapshot.training}: raise ValueError("training identities do not cover update")
    if any(r.reward_source != "business_outcome_v2" for r in update.request.rollouts): raise ValueError("Service update must use business rewards")
    if normalization not in ("sum", "mean"): raise ValueError("unknown normalization")
    before = update.request.policy_before; after = update.policy_after
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
