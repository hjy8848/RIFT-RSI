"""VERL v0.9.1 AgentLoop extension; the host session stays outside model input."""
from dataclasses import asdict
import json
from uuid import uuid4
from rift.verl.capture import ContinuousCapture

def __getattr__(name):
    if name != 'UsamiServiceAgentLoop':
        raise AttributeError(name)
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput

    class UsamiServiceAgentLoop(AgentLoopBase):
        def __init__(self, *args, rift_host_provider=None,
                     rift_host_provider_factory=None, **kwargs):
            super().__init__(*args, **kwargs)
            if rift_host_provider is not None and rift_host_provider_factory is not None:
                raise ValueError('supply a worker-local Host provider or its factory, not both')
            if rift_host_provider_factory is not None:
                factory = rift_host_provider_factory
                if isinstance(factory, str):
                    if ':' not in factory:
                        raise ValueError('Host provider factory must use module:callable syntax')
                    module_name, name = factory.split(':', 1)
                    import importlib
                    factory = getattr(importlib.import_module(module_name), name, None)
                if not callable(factory):
                    raise ValueError('Host provider factory is not callable')
                rift_host_provider = factory(config=self.config, tokenizer=self.tokenizer,
                                             processor=self.processor)
            self.rift_host_provider = rift_host_provider

        async def run(self, sampling_params, **kwargs):
            # VERL v1 writes sample kwargs back to TransferQueue. Only opaque
            # identifiers may travel there; private session objects are resolved
            # inside the trusted worker through an injected host provider.
            if any(key in kwargs for key in ('rift_session', 'rift_checkpoint', 'rift_checkpoint_verifier', 'rift_capture_sink')):
                raise ValueError('private host objects cannot be passed in sample fields')
            task_ref, group_id = kwargs.get('rift_task_ref'), kwargs.get('uid')
            mode = kwargs.get('rift_mode', 'train')
            request_id = kwargs.get('rift_request_id')
            if self.rift_host_provider is None or not isinstance(task_ref, str) or not task_ref or not isinstance(group_id, str) or not group_id:
                raise ValueError('trusted Host provider and opaque task/group IDs are required')
            if mode not in ('train', 'probe') or not isinstance(request_id, str) or not request_id:
                raise ValueError('explicit request identity and train/probe mode required')
            session_id = kwargs.get('session_id')
            if hasattr(session_id, 'item'):
                session_id = session_id.item()
            if type(session_id) is not int or session_id < 0:
                raise ValueError('VERL sample session_id is required')
            if mode == 'train':
                configured_group_size = getattr(self.rollout_config, 'n', None)
                requested_group_size = kwargs.get('rift_group_size')
                if (type(configured_group_size) is not int or type(requested_group_size) is not int
                        or requested_group_size != configured_group_size):
                    raise ValueError('RIFT group size must exactly match VERL rollout.n')
                rollout_id = kwargs.get('rift_rollout_id')
                sampling_seed = kwargs.get('rift_sampling_seed')
                customer_seed = kwargs.get('rift_customer_seed')
                if (not isinstance(rollout_id, str) or type(sampling_seed) is not int
                        or type(customer_seed) is not int):
                    raise ValueError('Service rollout requires preassigned IDs and independent seeds')
                expected_checkpoint = kwargs.get('rift_expected_checkpoint')
                session, checkpoint, checkpoint_verifier, sink = await self.rift_host_provider.open_rollout(
                    task_ref, group_id, session_id, self.tokenizer,
                    sampling_seed=sampling_seed, customer_seed=customer_seed,
                    rollout_id=rollout_id, expected_checkpoint=expected_checkpoint,
                    group_size=requested_group_size)
                if session.group_id != group_id or session.expected_group_size != requested_group_size:
                    raise ValueError('Host session does not match the complete sampled GRPO group')
            else:
                checkpoint_data = kwargs.get('rift_checkpoint')
                seed = kwargs.get('rift_sampling_seed')
                protocol_ref = kwargs.get('rift_protocol_ref')
                protocol_digest = kwargs.get('rift_protocol_digest')
                if not isinstance(checkpoint_data, (dict, str)) or type(seed) is not int:
                    raise ValueError('Probe requires an explicit checkpoint and frozen seed')
                from rift.schemas import Checkpoint
                checkpoint_request = (Checkpoint(**checkpoint_data) if isinstance(checkpoint_data, dict)
                                      else Checkpoint(**json.loads(checkpoint_data)))
                session, checkpoint, checkpoint_verifier, sink = await self.rift_host_provider.open_probe(
                    task_ref, checkpoint_request, seed, protocol_ref, self.tokenizer,
                    protocol_digest=protocol_digest)
                if checkpoint != checkpoint_request:
                    raise ValueError('Probe Host returned a different checkpoint')
                if not protocol_digest:
                    raise ValueError('Probe requires the frozen protocol digest')
            if checkpoint_verifier.verify(checkpoint) is not True:
                raise ValueError('rollout checkpoint was not verified')
            if sink.attestor.verify_serving(checkpoint) is not True:
                raise ValueError('rollout serving weights were not verified')
            try:
                messages, tools = await session.start()
                messages = list(messages)
                initial = await self.ct_build_initial_tokens(messages, tools=tools)
                capture = ContinuousCapture(initial, checkpoint)
                masks, logprobs = [], None
                logprobs_present = None
                serving_steps = set()
                model_request_id = uuid4().hex
                while True:
                    prefix = list(capture.current)
                    generation_id = uuid4().hex
                    if sink.attestor.verify_serving(checkpoint) is not True:
                        raise ValueError('rollout serving weights changed')
                    turn_sampling_params = dict(sampling_params)
                    from rift.verl.sampling import rollout_seed
                    base_seed = getattr(session, 'sampling_seed', None)
                    seed_scope = getattr(session, 'sampling_scope', None)
                    turn_sampling_params['seed'] = rollout_seed(
                        turn_sampling_params.get('seed', 0) if base_seed is None else base_seed,
                        session.rollout_id if seed_scope is None else seed_scope,
                        len(capture.decisions))
                    output = await self.server_manager.generate(request_id=model_request_id, prompt_ids=prefix,
                                                                sampling_params=turn_sampling_params)
                    token_ids = list(output.token_ids)
                    token_logprobs = None if output.log_probs is None else list(output.log_probs)
                    serving = dict(getattr(output, 'extra_fields', {}) or {})
                    serving_fields = {key: serving.get(key) for key in
                                      ('global_steps', 'min_global_steps', 'max_global_steps')}
                    if sink.attestor.verify_generation(checkpoint, serving_fields) is not True:
                        raise ValueError('vLLM response does not match the synchronized checkpoint revision')
                    serving_steps.update(serving_fields.values())
                    sink.attestor.record_generation(generation_id, checkpoint, session.rollout_id,
                        session.group_id, prefix, turn_sampling_params, token_ids, token_logprobs, serving_fields)
                    if logprobs_present is not None and logprobs_present != (token_logprobs is not None):
                        raise ValueError('inconsistent generation log-probability capture')
                    logprobs_present = token_logprobs is not None
                    merged, masks, logprobs = await self.ct_merge_assistant_token(
                        prefix, token_ids, masks, logprobs, assistant_logprobs=token_logprobs)
                    capture.add_generation(prefix, token_ids, token_logprobs, merged.token_ids, masks)
                    # Host parses the actual model output, invokes Usami, and returns
                    # only the appended public conversation. No Oracle enters messages.
                    assistant_message, additions, done = await session.advance(token_ids, prefix,
                        turn_sampling_params, token_logprobs, generation_id, serving_fields=serving_fields)
                    messages.append(assistant_message)
                    if len(capture.mask) > self.rollout_config.response_length:
                        raise ValueError('rollout exceeds response budget; no truncation of evidence')
                    if done: break
                    previous = list(messages)
                    messages.extend(additions)
                    merged, masks, logprobs = await self.ct_merge_context_msg(
                        previous, messages, list(capture.current), masks, logprobs, tools=tools)
                    capture.add_context(merged.token_ids, masks)
                    if len(capture.mask) >= self.rollout_config.response_length:
                        raise ValueError('rollout exceeds response budget; no truncation of evidence')
                episode = await session.finish()
                tokens = capture.output()
                if logprobs is not None and len(logprobs) != len(tokens.response):
                    raise ValueError('rollout log probabilities are not aligned with response tokens')
                if len(serving_steps) != 1:
                    raise ValueError('Service episode crossed synchronized policy revisions')
                # Persist the native replay object and exact model tokens before
                # deriving the public outcome. UNKNOWN remains unevaluable.
                await sink.persist(episode, checkpoint, tuple(capture.decisions), tokens)
                from rift.benchmark.public_feedback import service_reward
                from rift.benchmark.usami import public_episode
                public = public_episode(episode)
                reward = service_reward(public.outcome) if mode == 'train' else None
                evidence_ref = sink.evidence_ref(episode)
                return AgentLoopOutput(prompt_ids=list(tokens.prompt), response_ids=list(tokens.response),
                    response_mask=[int(x) for x in tokens.mask], response_logprobs=logprobs,
                    reward_score=reward, num_turns=len(capture.decisions), metrics={},
                    extra_fields={'global_steps': next(iter(serving_steps)),
                                  'min_global_steps': next(iter(serving_steps)),
                                  'max_global_steps': next(iter(serving_steps)),
                                  'rift_request_id': request_id,
                                  'rift_task_ref': task_ref,
                                  'rift_mode': mode,
                                  'rift_episode_id': episode.record_hash,
                                  'rift_checkpoint': json.dumps(asdict(checkpoint), sort_keys=True),
                                  'rift_group_id': group_id if mode == 'train' else None,
                                  'rift_seed': (kwargs.get('rift_sampling_seed') if mode == 'probe'
                                                else kwargs.get('rift_sampling_seed')),
                                  'rift_protocol_ref': kwargs.get('rift_protocol_ref'),
                                  'rift_protocol_digest': kwargs.get('rift_protocol_digest'),
                                  'rift_outcome': public.outcome.value,
                                  'rift_replay_verified': getattr(episode, 'replay_verified', False) is True,
                                  'rift_evidence_ref': evidence_ref})
            except BaseException:
                if hasattr(session, 'abort'):
                    await session.abort()
                raise
    globals()[name] = UsamiServiceAgentLoop
    return UsamiServiceAgentLoop
