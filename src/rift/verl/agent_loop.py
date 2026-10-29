"""VERL v0.9.1 AgentLoop extension; the host session stays outside model input."""
from uuid import uuid4
from rift.verl.capture import ContinuousCapture
from rift.verl.reward import verified_business_reward

def __getattr__(name):
    if name != 'UsamiServiceAgentLoop':
        raise AttributeError(name)
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput

    class UsamiServiceAgentLoop(AgentLoopBase):
        def __init__(self, *args, rift_host_provider=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.rift_host_provider = rift_host_provider

        async def run(self, sampling_params, **kwargs):
            # VERL v1 writes sample kwargs back to TransferQueue. Only opaque
            # identifiers may travel there; private session objects are resolved
            # inside the trusted worker through an injected host provider.
            if any(key in kwargs for key in ('rift_session', 'rift_checkpoint', 'rift_checkpoint_verifier', 'rift_capture_sink')):
                raise ValueError('private host objects cannot be passed in sample fields')
            task_ref, group_id = kwargs.get('rift_task_ref'), kwargs.get('uid')
            if self.rift_host_provider is None or not isinstance(task_ref, str) or not task_ref or not isinstance(group_id, str) or not group_id:
                raise ValueError('trusted Host provider and opaque task/group IDs are required')
            session, checkpoint, checkpoint_verifier, sink = await self.rift_host_provider.open_rollout(
                task_ref, group_id, kwargs.get('session_id'), self.tokenizer)
            if session.group_id != group_id:
                raise ValueError('Host session does not match the sampled group')
            group_size = kwargs.get('rift_group_size', getattr(self.rollout_config, 'n', None))
            if group_size is not None and (type(group_size) is not int or group_size < 1):
                raise ValueError('VERL rollout group size must be a positive integer')
            if hasattr(session, 'expected_group_size'):
                session.expected_group_size = group_size
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
                request_id = uuid4().hex
                while True:
                    prefix = list(capture.current)
                    generation_id = uuid4().hex
                    if sink.attestor.verify_serving(checkpoint) is not True:
                        raise ValueError('rollout serving weights changed')
                    turn_sampling_params = dict(sampling_params)
                    from rift.verl.sampling import rollout_seed
                    turn_sampling_params['seed'] = rollout_seed(turn_sampling_params.get('seed', 0),
                        session.rollout_id, len(capture.decisions))
                    output = await self.server_manager.generate(request_id=request_id, prompt_ids=prefix,
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
                from rift.benchmark.usami import public_episode
                reward = verified_business_reward(episode, lambda record: public_episode(record).outcome)
                tokens = capture.output()
                if logprobs is not None and len(logprobs) != len(tokens.response):
                    raise ValueError('rollout log probabilities are not aligned with response tokens')
                if len(serving_steps) != 1:
                    raise ValueError('Service episode crossed synchronized policy revisions')
                # Evidence persistence is a host operation, never sent to the model.
                await sink.persist(episode, checkpoint, tuple(capture.decisions), tokens)
                return AgentLoopOutput(prompt_ids=list(tokens.prompt), response_ids=list(tokens.response),
                    response_mask=[int(x) for x in tokens.mask], response_logprobs=logprobs,
                    reward_score=reward, num_turns=len(capture.decisions), metrics={},
                    extra_fields={'global_steps': next(iter(serving_steps)),
                                  'min_global_steps': next(iter(serving_steps)),
                                  'max_global_steps': next(iter(serving_steps))})
            except BaseException:
                if hasattr(session, 'abort'):
                    await session.abort()
                raise
    globals()[name] = UsamiServiceAgentLoop
    return UsamiServiceAgentLoop
