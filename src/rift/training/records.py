"""Explicit conversion of public outcomes and update-bound Designer credit."""
from math import isfinite
from dataclasses import asdict
import hashlib
import json
from rift.benchmark.public_feedback import service_reward
from rift.schemas import Outcome, Tokens
from rift.replay.reward import ReplayReward
from rift.rewards.learning_value import DualBankOutcomeReward, RewardMode
from rift.training.contracts import CompletedUpdate, Rollout
from rift.training.contracts import UpdateRequest
from dataclasses import dataclass

def service_rollout(tokens, checkpoint, task_id, episode_id, outcome, group_id=None, *, response_id):
    reward = service_reward(outcome)
    if reward is None: raise ValueError("unknown business outcome cannot be trained")
    if not response_id: raise ValueError("recorded Service response required")
    return Rollout(tokens, checkpoint, task_id, episode_id, reward, "business_outcome_v2", group_id, response_id, episode_id)

def reward_scalar(reward):
    if isinstance(reward, ReplayReward): return reward.mean
    if isinstance(reward, DualBankOutcomeReward): return reward.value
    raise ValueError("unsupported Designer reward protocol")

def reward_protocol(reward):
    if isinstance(reward, ReplayReward): return RewardMode.REPLAY_LOGP_V1.value
    if isinstance(reward, DualBankOutcomeReward): return reward.reward_protocol
    raise ValueError("unsupported Designer reward protocol")

def designer_reward_source(reward, service_update):
    if isinstance(reward, ReplayReward):
        return replay_source(service_update.request.policy_before, reward.snapshot_digest, reward.normalization)
    if not isinstance(reward, DualBankOutcomeReward):
        raise ValueError("unsupported Designer reward protocol")
    if reward.credit_assignment_mode != "sequential_curriculum":
        raise ValueError("unsupported dual-bank credit assignment mode")
    return "dualbank:sequential:" + reward.reward_source_digest

def designer_rollout(tokens, designer_checkpoint, package_id, reward, service_update, group_id=None, *, designer_rollout_id, update_verifier, reward_verifier, action_verifier):
    if not isinstance(reward, (ReplayReward, DualBankOutcomeReward)) or not isinstance(service_update, CompletedUpdate):
        raise ValueError("Designer credit requires a completed Service update and tagged reward evidence")
    if update_verifier.verify_update(service_update) is not True:
        raise ValueError("Service update evidence was not verified")
    reward_update_id = (reward.update_run_id if isinstance(reward, ReplayReward)
                        else reward.service_update_run_id)
    if package_id != service_update.request.package_id or reward.package_id != package_id or reward_update_id != service_update.run_id:
        raise ValueError("reward and Service update differ")
    if reward_verifier.verify_reward(reward, service_update) is not True:
        raise ValueError("Replay reward evidence was not verified")
    if action_verifier.verify_designer_action(tokens, designer_checkpoint, package_id, designer_rollout_id, group_id) is not True:
        raise ValueError("Designer action evidence was not verified")
    value = reward_scalar(reward)
    if not isfinite(value): raise ValueError("invalid reward")
    source = designer_reward_source(reward, service_update)
    return Rollout(tokens, designer_checkpoint, package_id, designer_rollout_id, value, source, group_id, designer_rollout_id, service_update.run_id)

def replay_source(checkpoint, snapshot_digest, normalization):
    identity = json.dumps(asdict(checkpoint), sort_keys=True, separators=(",", ":"))
    return "replay:" + hashlib.sha256(identity.encode()).hexdigest() + ":" + snapshot_digest + ":" + normalization

def comparable_designer_rollouts(rollouts, *, algorithm=None, credit_assignment_mode=None):
    if not rollouts:
        raise ValueError("Designer reward batch cannot be empty")
    sources = {r.reward_source for r in rollouts}
    prefixes = {"replay" if r.reward_source.startswith("replay:") else
                "dualbank_matched" if r.reward_source.startswith("dualbank:matched:") else
                "dualbank_sequential" if r.reward_source.startswith("dualbank:sequential:") else "unknown"
                for r in rollouts}
    if len(prefixes) != 1 or "unknown" in prefixes:
        raise ValueError("mixed or unknown Designer reward protocols")
    mode = next(iter(prefixes))
    if mode == "replay":
        if len(sources) != 1:
            raise ValueError("mixed Replay LogP baselines or snapshots require explicit treatment")
        return
    if (mode != "dualbank_sequential" or credit_assignment_mode != "sequential_curriculum" or algorithm is None
            or algorithm.lower() != "reinforce_baseline"):
        raise ValueError("sequential dual-bank rewards require ungrouped REINFORCE with an explicit baseline")
    if any(r.group_id is not None for r in rollouts):
        raise ValueError("sequential nonstationary rewards cannot be formed into GRPO groups")

@dataclass(frozen=True)
class DesignerFeedback:
    """One recorded package action and the Service update it caused."""
    tokens: Tokens
    update: CompletedUpdate
    reward: ReplayReward | DualBankOutcomeReward
    designer_rollout_id: str
    group_id: str | None = None

def prepare_designer_update(batch_id, designer_checkpoint, feedback, *, update_verifier,
    reward_verifier, action_verifier, algorithm=None, credit_assignment_mode=None,
    reward_baseline=None):
    """Build a replay-matched or sequential dual-bank Designer policy-gradient batch."""
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
    replay_only = all(isinstance(item.reward, ReplayReward) for item in feedback)
    dual_only = all(isinstance(item.reward, DualBankOutcomeReward) for item in feedback)
    if not (replay_only or dual_only):
        raise ValueError('Replay LogP and dual-bank rewards cannot share a batch')
    if replay_only:
        algorithm = algorithm or 'reinforce_plus_plus'
        output_protocol = None
        output_credit_mode = None
    else:
        modes = {item.reward.credit_assignment_mode for item in feedback}
        if len(modes) != 1:
            raise ValueError('dual-bank feedback credit-assignment modes cannot be mixed')
        credit_assignment_mode = credit_assignment_mode or next(iter(modes))
        if credit_assignment_mode == 'sequential_curriculum':
            algorithm = algorithm or 'reinforce_baseline'
            if reward_baseline is None or not isfinite(reward_baseline):
                raise ValueError('sequential Designer update requires an explicit scalar baseline')
            if algorithm.lower() != 'reinforce_baseline':
                raise ValueError('sequential curriculum cannot fabricate comparable GRPO groups')
        else:
            raise ValueError('dual-bank v1 implements sequential_curriculum only')
    rows = tuple(designer_rollout(item.tokens, designer_checkpoint,
        item.update.request.package_id, item.reward, item.update, item.group_id,
        designer_rollout_id=item.designer_rollout_id, update_verifier=update_verifier,
        reward_verifier=reward_verifier, action_verifier=action_verifier) for item in feedback)
    comparable_designer_rollouts(rows, algorithm=algorithm,
        credit_assignment_mode=credit_assignment_mode)
    if dual_only:
        protocol = reward_protocol(feedback[0].reward)
        output_protocol = protocol
        output_credit_mode = credit_assignment_mode
    return UpdateRequest(designer_checkpoint, batch_id, rows, algorithm,
        reward_baseline if algorithm.lower() == 'reinforce_baseline' else None,
        output_protocol, output_credit_mode)
