"""Construct Service requests from native replay-verified episodes and a trusted recorder."""
from rift.schemas import Outcome, Tokens
from rift.benchmark.private_evaluation import public_business_outcome
from rift.training.contracts import UpdateRequest
from rift.training.records import service_rollout

def prepare_service_update(package_id, certified_tasks, episodes, checkpoint, recorder, *, algorithm='policy_gradient'):
    from usami_v2.runtime.runner import replay_episode
    if not certified_tasks: raise ValueError('certified tasks required')
    episodes = tuple(episodes)
    episode_ids = [episode.record_hash for episode in episodes]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError('duplicate Service episode')
    rows = []
    for episode in episodes:
        task_id = episode.task.record_hash
        if task_id not in certified_tasks: raise ValueError('episode is not from a certified package task')
        verified = replay_episode(episode).verified
        if not verified: raise ValueError('episode replay failed')
        outcome = public_business_outcome(episode, verified)
        if outcome is Outcome.UNKNOWN: continue
        calls = [c for c in episode.model_calls if c.role == 'service']
        if not calls or recorder.verify_source(episode, checkpoint) is not True:
            raise ValueError('real Service model provenance required')
        for call in calls:
            tokens = recorder.tokens_for_call(call, checkpoint)
            if not isinstance(tokens, Tokens): raise ValueError('verified tokens required')
            rows.append(service_rollout(tokens, checkpoint, task_id, episode.record_hash, outcome, response_id=call.record_hash))
    if not rows: raise ValueError('no usable verified Service rollouts')
    if {row.task_id for row in rows} != set(certified_tasks):
        raise ValueError('every certified task must have a usable Service rollout')
    return UpdateRequest(checkpoint, package_id, tuple(rows), algorithm)
