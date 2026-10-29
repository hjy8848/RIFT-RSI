"""On-policy evidence checks for captured Service and Designer rollouts."""
import json
from rift.schemas import Outcome
from rift.benchmark.public_feedback import service_reward
from rift.training.records import designer_reward_source, reward_scalar

# Actor scoring below uses the unmodified policy at temperature 1. Reject
# sampling transformations unless their behavior-policy correction is added.
_ALLOWED_SAMPLING = {'temperature','top_p','top_k','min_p','repetition_penalty',
    'presence_penalty','frequency_penalty','logprobs','seed','max_tokens','min_tokens',
    'stop','stop_token_ids','ignore_eos','skip_special_tokens'}

def unmodified_sampling(params):
    return (isinstance(params, dict) and not (set(params) - _ALLOWED_SAMPLING)
        and params.get('temperature') == 1.0 and params.get('top_p', 1.0) == 1.0
        and params.get('top_k', -1) == -1 and params.get('min_p', 0.0) == 0.0
        and params.get('repetition_penalty', 1.0) == 1.0
        and params.get('presence_penalty', 0.0) == 0.0
        and params.get('frequency_penalty', 0.0) == 0.0)

def _group_records(records):
    if len(records) < 2 or len({r['rollout_id'] for r in records}) != len(records): return False
    settings = [{k:v for k,v in r['sampling_params'].items() if k != 'seed'} for r in records]
    if any(s != settings[0] for s in settings[1:]): return False
    seeds = [r['sampling_params'].get('seed') for r in records]
    if any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        return False
    return True

class NativeServiceRolloutVerifier:
    def __init__(self, episode_resolver, capture_store):
        self.episode_resolver, self.capture_store = episode_resolver, capture_store
    def verify_rollout(self, row, checkpoint):
        from rift.benchmark.usami import public_episode
        episode = self.episode_resolver(row.episode_id)
        if episode is None or episode.record_hash != row.episode_id or row.policy != checkpoint: return False
        if episode.task.record_hash != row.task_id or row.response_id != episode.record_hash or row.reward_ref != episode.record_hash: return False
        public = public_episode(episode)
        if public.outcome is Outcome.UNKNOWN or service_reward(public.outcome) != row.reward: return False
        if not self.capture_store.verify_trajectory(episode, checkpoint): return False
        if self.capture_store.trajectory_for(episode, checkpoint) != row.tokens or self.capture_store.group_for(episode, checkpoint) != row.group_id: return False
        calls = [c for c in episode.model_calls if c.role == 'service']
        return bool(calls) and all(unmodified_sampling(c.request.get('sampling_params')) for c in calls)
    def verify_group(self, rows, checkpoint):
        records = []
        for row in rows:
            if not self.verify_rollout(row, checkpoint): return False
            episode = self.episode_resolver(row.episode_id)
            calls = [c for c in episode.model_calls if c.role == 'service']
            records.append(calls[0].request)
        sizes = {r.get('group_size') for r in records}
        return (len({r.task_id for r in rows}) == 1 and _group_records(records)
                and sizes == {len(rows)}
                and len({tuple(r['prompt_ids']) for r in records}) == 1)

class DesignerRolloutVerifier:
    def __init__(self, capture_store, feedback_resolver, update_verifier, reward_verifier):
        self.capture_store, self.feedback_resolver = capture_store, feedback_resolver
        self.update_verifier, self.reward_verifier = update_verifier, reward_verifier
    def _record(self, row):
        return json.loads(self.capture_store._path(row.episode_id).read_text(encoding='utf-8'))
    def verify_rollout(self, row, checkpoint):
        if row.policy != checkpoint or row.response_id != row.episode_id: return False
        if not self.capture_store.verify_designer_action(row.tokens, checkpoint, row.task_id, row.episode_id, row.group_id): return False
        try: record = self._record(row)
        except (OSError, ValueError, KeyError): return False
        if not unmodified_sampling(record['request'].get('sampling_params')): return False
        pair = self.feedback_resolver(row.reward_ref)
        if pair is None: return False
        update, reward = pair
        reward_update_id = getattr(reward, 'update_run_id', getattr(reward, 'service_update_run_id', None))
        if (update.run_id != row.reward_ref or update.request.package_id != row.task_id
                or reward.package_id != row.task_id or reward_update_id != update.run_id
                or row.reward != reward_scalar(reward)
                or row.reward_source != designer_reward_source(reward, update)):
            return False
        return self.update_verifier.verify_update(update) is True and self.reward_verifier.verify_reward(reward, update) is True
    def verify_group(self, rows, checkpoint):
        try:
            if any(not self.verify_rollout(row, checkpoint) for row in rows): return False
            requests = [self._record(row)['request'] for row in rows]
            sizes = {request.get('group_size') for request in requests}
            return (_group_records(requests) and sizes == {len(rows)}
                    and len({tuple(request['prompt_ids']) for request in requests}) == 1)
        except (OSError, ValueError, KeyError, TypeError):
            return False
