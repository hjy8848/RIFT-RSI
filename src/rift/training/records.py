"""Explicit conversion of public outcomes and update-bound Designer credit."""
from math import isfinite
from dataclasses import asdict
import hashlib
import json
from rift.benchmark.public_feedback import service_reward
from rift.schemas import Outcome, Tokens
from rift.replay.reward import ReplayReward
from rift.training.contracts import CompletedUpdate, Rollout
from rift.training.contracts import UpdateRequest
from dataclasses import dataclass

def service_rollout(tokens, checkpoint, task_id, episode_id, outcome, group_id=None, *, response_id):
    reward = service_reward(outcome)
    if reward is None: raise ValueError("unknown business outcome cannot be trained")
    if not response_id: raise ValueError("recorded Service response required")
    return Rollout(tokens, checkpoint, task_id, episode_id, reward, "business_outcome_v2", group_id, response_id, episode_id)

def designer_rollout(tokens, designer_checkpoint, package_id, reward, service_update, group_id=None, *, designer_rollout_id, update_verifier, reward_verifier, action_verifier):
    if not isinstance(reward, ReplayReward) or not isinstance(service_update, CompletedUpdate):
        raise ValueError("Designer credit requires completed Service update and replay reward")
    if update_verifier.verify_update(service_update) is not True:
        raise ValueError("Service update evidence was not verified")
    if package_id != service_update.request.package_id or reward.package_id != package_id or reward.update_run_id != service_update.run_id:
        raise ValueError("reward and Service update differ")
    if reward_verifier.verify_reward(reward, service_update) is not True:
        raise ValueError("Replay reward evidence was not verified")
    if action_verifier.verify_designer_action(tokens, designer_checkpoint, package_id, designer_rollout_id) is not True:
        raise ValueError("Designer action evidence was not verified")
    if not isfinite(reward.mean): raise ValueError("invalid reward")
    # Source records the exact Service baseline so caller can enforce matched comparisons.
    source = replay_source(service_update.request.policy_before, reward.snapshot_digest, reward.normalization)
    return Rollout(tokens, designer_checkpoint, package_id, designer_rollout_id, reward.mean, source, group_id, designer_rollout_id, service_update.run_id)

def replay_source(checkpoint, snapshot_digest, normalization):
    identity = json.dumps(asdict(checkpoint), sort_keys=True, separators=(",", ":"))
    return "replay:" + hashlib.sha256(identity.encode()).hexdigest() + ":" + snapshot_digest + ":" + normalization

def comparable_designer_rollouts(rollouts):
    if not rollouts or any(not r.reward_source.startswith("replay:") for r in rollouts):
        raise ValueError("replay credit required")
    if len({r.reward_source for r in rollouts}) != 1:
        raise ValueError("mixed Service baselines or replay snapshots require explicit treatment")

@dataclass(frozen=True)
class DesignerFeedback:
    """One recorded package action and the Service update it caused."""
    tokens: Tokens
    update: CompletedUpdate
    reward: ReplayReward
    designer_rollout_id: str
    group_id: str | None = None

def prepare_designer_update(batch_id, designer_checkpoint, feedback, *, update_verifier, reward_verifier, action_verifier, algorithm='policy_gradient'):
    """Accumulate package-level credit only for comparable Service baselines."""
    if not feedback:
        raise ValueError('Designer update needs package feedback')
    if any(not isinstance(item, DesignerFeedback) for item in feedback):
        raise ValueError('DesignerFeedback required')
    packages = [item.update.request.package_id for item in feedback]
    if len(set(packages)) != len(packages):
        raise ValueError('duplicate package feedback')
    if len({item.designer_rollout_id for item in feedback}) != len(feedback):
        raise ValueError('duplicate Designer rollout')
    if len({item.update.run_id for item in feedback}) != len(feedback):
        raise ValueError('duplicate Service update')
    rows = tuple(designer_rollout(item.tokens, designer_checkpoint,
        item.update.request.package_id, item.reward, item.update, item.group_id,
        designer_rollout_id=item.designer_rollout_id, update_verifier=update_verifier,
        reward_verifier=reward_verifier, action_verifier=action_verifier) for item in feedback)
    comparable_designer_rollouts(rows)
    return UpdateRequest(designer_checkpoint, batch_id, rows, algorithm)
