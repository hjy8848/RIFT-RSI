"""Revalidate a saved replay reward by re-scoring its immutable snapshot."""
from math import isclose
from rift.replay.reward import ReplayReward, score_update

class RecomputedReplayRewardVerifier:
    def __init__(self, snapshot_resolver, scorer, checkpoint_store, update_verifier, replay_verifier, *, rel_tol=1e-6, abs_tol=1e-7):
        if rel_tol < 0 or abs_tol < 0: raise ValueError('invalid tolerance')
        self.snapshot_resolver, self.scorer, self.checkpoint_store = snapshot_resolver, scorer, checkpoint_store
        self.update_verifier, self.replay_verifier = update_verifier, replay_verifier
        self.rel_tol, self.abs_tol = rel_tol, abs_tol
    def verify_reward(self, reward, update):
        if not isinstance(reward, ReplayReward): return False
        snapshot = self.snapshot_resolver(reward.binding_digest)
        if snapshot is None or snapshot.digest != reward.binding_digest or snapshot.replay_digest != reward.snapshot_digest:
            return False
        if reward.update_run_id != update.run_id or reward.package_id != update.request.package_id:
            return False
        actual = score_update(update, snapshot, self.scorer, self.checkpoint_store,
            update_verifier=self.update_verifier, replay_verifier=self.replay_verifier,
            normalization=reward.normalization)
        if len(actual.samples) != len(reward.samples): return False
        for expected, observed in zip(reward.samples, actual.samples):
            if expected.sample_id != observed.sample_id: return False
            if any(not isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol) for a, b in (
                (expected.before, observed.before), (expected.after, observed.after), (expected.delta, observed.delta))):
                return False
        return isclose(reward.mean, actual.mean, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
