"""Capture on-policy Designer proposals before their delayed RIFT reward exists."""
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from uuid import uuid4
import json
from rift.benchmark.public_feedback import PublicSummary
from rift.schemas import Checkpoint, Tokens
from rift.tasks import Proposal
from rift.training.trace import ExactPolicyTrace, digest

@dataclass(frozen=True)
class PendingDesignerRollout:
    package_id: str
    rollout_id: str
    group_id: str | None
    checkpoint: Checkpoint
    proposals: tuple[Proposal, ...]
    trace: ExactPolicyTrace
    logprobs: tuple[float, ...] | None

class DesignerCaptureStore:
    def __init__(self, root, *, attestor):
        self.root, self.attestor = Path(root), attestor
    def _path(self, rollout_id):
        if not rollout_id or any(c not in '0123456789abcdef' for c in rollout_id):
            raise ValueError('invalid Designer rollout ID')
        return self.root / (rollout_id + '.json')
    def save(self, pending, request, response):
        trace = pending.trace
        if (trace.role != 'designer' or trace.rollout_id != pending.rollout_id
                or trace.package_id != pending.package_id or trace.policy != pending.checkpoint
                or trace.request_digest != digest(request) or trace.response_digest != digest(response)
                or request['prompt_ids'] != list(trace.tokens.prompt)
                or response['generated_token_ids'] != list(trace.tokens.response)
                or response['log_probs'] != (None if pending.logprobs is None else list(pending.logprobs))
                or self.attestor.verify_capture(trace, request, response) is not True):
            raise ValueError('Designer generation evidence failed verification')
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._path(pending.rollout_id).open('x', encoding='utf-8') as stream:
            json.dump({'trace':asdict(trace), 'request':request, 'response':response,
                'group_id':pending.group_id, 'proposals':[asdict(p) for p in pending.proposals]}, stream, sort_keys=True)
    def verify_designer_action(self, tokens, checkpoint, package_id, rollout_id, group_id=None):
        try:
            record = json.loads(self._path(rollout_id).read_text(encoding='utf-8'))
            raw = record['trace']; t = raw['tokens']
            trace = ExactPolicyTrace(raw['role'], raw['rollout_id'], raw['decision_id'], Checkpoint(**raw['policy']),
                Tokens(tuple(t['prompt']), tuple(t['response']), tuple(t['mask'])), raw['request_digest'],
                raw['response_digest'], raw['serialization'], raw['evidence_ref'], raw['package_id'])
            request, response = record['request'], record['response']
            return (trace.role == 'designer' and trace.rollout_id == rollout_id and trace.package_id == package_id
                and trace.policy == checkpoint and trace.tokens == tokens and all(tokens.mask)
                and trace.evidence_ref == str(self._path(rollout_id))
                and request.get('package_id') == package_id and request.get('rollout_id') == rollout_id
                and request.get('group_id') == record['group_id'] == group_id
                and response.get('parsed_proposals') == record['proposals']
                and request.get('prompt_ids') == list(tokens.prompt)
                and response.get('generated_token_ids') == list(tokens.response)
                and digest(request) == trace.request_digest and digest(response) == trace.response_digest
                and self.attestor.verify_capture(trace, request, response) is True)
        except (OSError, ValueError, TypeError, KeyError):
            return False

class DesignerSampler:
    def __init__(self, server, prompt_encoder, proposal_parser, checkpoint_verifier, capture_store, *, serialization):
        self.server, self.prompt_encoder, self.proposal_parser = server, prompt_encoder, proposal_parser
        self.checkpoint_verifier, self.capture_store, self.serialization = checkpoint_verifier, capture_store, serialization
    async def sample(self, summaries, checkpoint, package_id, sampling_params, *, group_id=None,
                     group_size=None, rollout_id=None):
        summaries = tuple(summaries)
        if not summaries or any(not isinstance(row, PublicSummary) for row in summaries):
            raise ValueError('Designer receives public summaries only')
        if not package_id or not self.serialization or self.checkpoint_verifier.verify(checkpoint) is not True:
            raise ValueError('package and verified Designer checkpoint required')
        if group_size is not None and (type(group_size) is not int or group_size < 2 or not group_id):
            raise ValueError('Designer GRPO needs a named group of at least two rollouts')
        rollout_id = rollout_id or uuid4().hex
        path = self.capture_store._path(rollout_id)
        if path.exists(): raise ValueError('Designer rollout ID already exists')
        prompt_ids = tuple(await self.prompt_encoder(summaries))
        if not prompt_ids: raise ValueError('empty Designer prompt')
        request = {'prompt_ids':list(prompt_ids), 'sampling_params':dict(sampling_params),
                   'package_id':package_id, 'rollout_id':rollout_id, 'group_id':group_id,
                   'group_size':group_size,
                   'generation_id':uuid4().hex}
        from rift.verl.sampling import rollout_seed
        request['sampling_params']['seed'] = rollout_seed(request['sampling_params'].get('seed', 0), rollout_id)
        if self.capture_store.attestor.verify_serving(checkpoint) is not True:
            raise ValueError('Designer serving weights were not verified')
        output = await self.server.generate(request_id=rollout_id, prompt_ids=list(prompt_ids),
                                            sampling_params=request['sampling_params'])
        serving = dict(getattr(output, 'extra_fields', {}) or {})
        serving_fields = {key: serving.get(key) for key in
                          ('global_steps', 'min_global_steps', 'max_global_steps')}
        if self.capture_store.attestor.verify_generation(checkpoint, serving_fields) is not True:
            raise ValueError('Designer vLLM response does not match its synchronized checkpoint revision')
        response_ids = tuple(output.token_ids)
        tokens = Tokens(prompt_ids, response_ids, (True,) * len(response_ids))
        logprobs = None if output.log_probs is None else tuple(output.log_probs)
        if logprobs is not None and (len(logprobs) != len(response_ids) or any(not isfinite(x) or x > 0 for x in logprobs)):
            raise ValueError('invalid Designer generation log probabilities')
        self.capture_store.attestor.record_generation(request['generation_id'], checkpoint, rollout_id, group_id,
            prompt_ids, request['sampling_params'], response_ids, logprobs, serving_fields)
        proposals = tuple(await self.proposal_parser(response_ids))
        if not proposals or any(not isinstance(p, Proposal) for p in proposals) or len({p.proposal_id for p in proposals}) != len(proposals):
            raise ValueError('valid distinct structured proposals required')
        response = {'generated_token_ids':list(response_ids), 'log_probs':None if logprobs is None else list(logprobs),
                    'parsed_proposals':[asdict(p) for p in proposals], 'serving_fields':serving_fields}
        trace = ExactPolicyTrace('designer', rollout_id, uuid4().hex, checkpoint, tokens,
            digest(request), digest(response), self.serialization, str(path), package_id)
        pending = PendingDesignerRollout(package_id, rollout_id, group_id, checkpoint, proposals, trace, logprobs)
        self.capture_store.save(pending, request, response)
        return pending
