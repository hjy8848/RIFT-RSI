"""Persist and verify generation-time token evidence without storing Oracle data."""
from dataclasses import asdict
import json
from pathlib import Path
from rift.schemas import Checkpoint, Tokens
from rift.training.trace import ExactPolicyTrace, RecordedDecisionEvidenceVerifier, digest


def _tokens(value):
    return Tokens(tuple(value['prompt']), tuple(value['response']), tuple(value['mask']))

class NativeCaptureStore:
    def __init__(self, root, *, serialization, attestor):
        if not serialization: raise ValueError('serialization identity required')
        self.root = Path(root)
        self.serialization = serialization
        self.attestor = attestor
    def _path(self, episode):
        identifier = episode.record_hash
        if len(identifier) != 64 or any(c not in '0123456789abcdef' for c in identifier):
            raise ValueError('invalid episode identity')
        return self.root / (identifier + '.json')
    async def persist(self, episode, checkpoint, decisions, trajectory):
        calls = [call for call in episode.model_calls if call.role == 'service']
        if not calls or len(calls) != len(decisions): raise ValueError('decision count differs from native calls')
        traces, covered = [], set()
        previous_end = 0
        for index, (call, decision) in enumerate(zip(calls, decisions)):
            tokens = decision.tokens
            if decision.sequence != index or decision.checkpoint != checkpoint:
                raise ValueError('decision identity differs from capture')
            start = len(tokens.prompt) - len(trajectory.prompt)
            end = start + len(tokens.response)
            if (not all(tokens.mask) or start < previous_end or end > len(trajectory.response)
                    or tuple(trajectory.prompt + trajectory.response[:start]) != tokens.prompt
                    or tuple(trajectory.response[start:end]) != tokens.response
                    or not all(trajectory.mask[start:end])):
                raise ValueError('decision is not an exact trajectory segment')
            if (call.request.get('prompt_ids') != list(tokens.prompt)
                    or call.response.get('generated_token_ids') != list(tokens.response)
                    or call.response.get('generated_log_probs') != (None if decision.logprobs is None else list(decision.logprobs))):
                raise ValueError('native model call differs from captured tokens')
            covered.update(range(start, end)); previous_end = end
            trace = ExactPolicyTrace('service', episode.record_hash, call.record_hash, checkpoint, tokens,
                digest(call.request), digest(call.response), self.serialization, str(self._path(episode)))
            if self.attestor.verify_capture(trace, call.request, call.response) is not True:
                raise ValueError('generation-time provenance was not verified')
            trace_record = asdict(trace)
            trace_record['generated_log_probs'] = None if decision.logprobs is None else list(decision.logprobs)
            traces.append(trace_record)
        if covered != {i for i, selected in enumerate(trajectory.mask) if selected}:
            raise ValueError('trajectory action mask is not fully covered by generated decisions')
        groups = {call.request.get('group_id') for call in calls}
        rollouts = {call.request.get('rollout_id') for call in calls}
        if len(groups) != 1 or len(rollouts) != 1 or None in rollouts:
            raise ValueError('rollout and group provenance must remain consistent')
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self._path(episode)
        with path.open('x', encoding='utf-8') as stream:
            json.dump({'episode_id':episode.record_hash, 'checkpoint':asdict(checkpoint),
                'trajectory':asdict(trajectory), 'traces':traces,
                'group_id':next(iter(groups)), 'rollout_id':next(iter(rollouts))}, stream, sort_keys=True)
    def _load(self, episode, checkpoint):
        data = json.loads(self._path(episode).read_text(encoding='utf-8'))
        if data['episode_id'] != episode.record_hash or Checkpoint(**data['checkpoint']) != checkpoint:
            raise ValueError('capture checkpoint or episode differs')
        return data
    def verify_trajectory(self, episode, checkpoint):
        try:
            data = self._load(episode, checkpoint)
            calls = [call for call in episode.model_calls if call.role == 'service']
            if not calls or len(calls) != len(data['traces']): return False
            trajectory = _tokens(data['trajectory']); covered = set(); previous_end = 0
            for raw, call in zip(data['traces'], calls):
                trace = ExactPolicyTrace(raw['role'], raw['rollout_id'], raw['decision_id'],
                    Checkpoint(**raw['policy']), _tokens(raw['tokens']), raw['request_digest'],
                    raw['response_digest'], raw['serialization'], raw['evidence_ref'], raw.get('package_id'))
                start = len(trace.tokens.prompt) - len(trajectory.prompt); end = start + len(trace.tokens.response)
                if (trace.role != 'service' or trace.rollout_id != episode.record_hash
                        or trace.decision_id != call.record_hash or trace.policy != checkpoint
                        or trace.serialization != self.serialization or trace.evidence_ref != str(self._path(episode))
                        or not all(trace.tokens.mask) or start < previous_end or end > len(trajectory.response)
                        or trajectory.prompt + trajectory.response[:start] != trace.tokens.prompt
                        or trajectory.response[start:end] != trace.tokens.response
                        or not all(trajectory.mask[start:end])
                        or call.request.get('prompt_ids') != list(trace.tokens.prompt)
                        or call.response.get('generated_token_ids') != list(trace.tokens.response)
                        or call.response.get('generated_log_probs') != raw.get('generated_log_probs')
                        or trace.request_digest != digest(call.request) or trace.response_digest != digest(call.response)
                        or call.request.get('group_id') != data['group_id']
                        or call.request.get('rollout_id') != data['rollout_id']
                        or self.attestor.verify_capture(trace, call.request, call.response) is not True): return False
                covered.update(range(start,end)); previous_end = end
            return covered == {i for i, selected in enumerate(trajectory.mask) if selected}
        except (OSError, ValueError, TypeError, KeyError):
            return False
    def trajectory_for(self, episode, checkpoint):
        if not self.verify_trajectory(episode, checkpoint): raise ValueError('trajectory evidence failed verification')
        return _tokens(self._load(episode, checkpoint)['trajectory'])
    def group_for(self, episode, checkpoint):
        if not self.verify_trajectory(episode, checkpoint): raise ValueError('trajectory evidence failed verification')
        return self._load(episode, checkpoint)['group_id']
    def decision_verifier(self, episode, checkpoint):
        if not self.verify_trajectory(episode, checkpoint): raise ValueError('trajectory evidence failed verification')
        data = self._load(episode, checkpoint)
        traces = [ExactPolicyTrace(raw['role'], raw['rollout_id'], raw['decision_id'],
            Checkpoint(**raw['policy']), _tokens(raw['tokens']), raw['request_digest'],
            raw['response_digest'], raw['serialization'], raw['evidence_ref'], raw.get('package_id'))
            for raw in data['traces']]
        return RecordedDecisionEvidenceVerifier(traces, self.attestor)
