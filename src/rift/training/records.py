"""Explicit conversion of public outcomes and update-bound Designer credit."""
from math import isfinite
from dataclasses import asdict
import hashlib
import json
from rift.benchmark.public_feedback import service_reward
from rift.schemas import Outcome
from rift.replay.reward import ReplayReward
from rift.training.contracts import CompletedUpdate, Rollout

def service_rollout(tokens, checkpoint, task_id, episode_id, outcome, group_id=None):
    reward = service_reward(outcome)
    if reward is None: raise ValueError("unknown business outcome cannot be trained")
    return Rollout(tokens, checkpoint, task_id, episode_id, reward, "business_outcome_v2", group_id)

def designer_rollout(tokens, designer_checkpoint, package_id, reward, service_update, group_id=None):
    if not isinstance(reward, ReplayReward) or not isinstance(service_update, CompletedUpdate):
        raise ValueError("Designer credit requires completed Service update and replay reward")
    if package_id != service_update.request.package_id or reward.package_id != package_id or reward.update_run_id != service_update.run_id:
        raise ValueError("reward and Service update differ")
    if not isfinite(reward.mean): raise ValueError("invalid reward")
    # Source records the exact Service baseline so caller can enforce matched comparisons.
    source = replay_source(service_update.request.policy_before, reward.snapshot_digest, reward.normalization)
    return Rollout(tokens, designer_checkpoint, package_id, service_update.run_id, reward.mean, source, group_id)

def replay_source(checkpoint, snapshot_digest, normalization):
    identity = json.dumps(asdict(checkpoint), sort_keys=True, separators=(",", ":"))
    return "replay:" + hashlib.sha256(identity.encode()).hexdigest() + ":" + snapshot_digest + ":" + normalization

def comparable_designer_rollouts(rollouts):
    if not rollouts or any(not r.reward_source.startswith("replay:") for r in rollouts):
        raise ValueError("replay credit required")
    if len({r.reward_source for r in rollouts}) != 1:
        raise ValueError("mixed Service baselines or replay snapshots require explicit treatment")
