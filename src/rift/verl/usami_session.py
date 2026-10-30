"""Stepwise bridge to the native synchronous Usami commerce runner.

The model is called by VERL, not by this module. The native runner owns all
business logic, Customer transitions, Engine execution and private evaluation.
"""
import asyncio
import json
from queue import Queue
from uuid import uuid4


def service_tool_protocol_reminder(tools):
    """Build public schema reminders without exposing private task data."""
    names = sorted(row["function"]["name"] for row in tools
                   if isinstance(row, dict) and isinstance(row.get("function"), dict))
    return (
        "RIFT Service turn protocol: emit exactly ONE complete tool call, then "
        "stop and let the environment return the tool result. Never output "
        "multiple <tool_call> blocks in the same assistant turn. Tool arguments "
        "must match the supplied function JSON schema exactly and are flat: pass "
        "the named arguments directly, never wrap them in a 'params' object and "
        "never add fields such as 'messages' or 'scope_id'. Available tools: "
        + ", ".join(names) + "."
    )


class _ParserExecutionFailure(Exception):
    def __init__(self, call, cause):
        super().__init__(str(cause))
        self.call = call
        self.__cause__ = cause


class NativeUsamiSession:
    def __init__(self, world, task, profile, customer_agent, parser, *, budget, parser_schemas=None,
                 mode='reference', external_events=(), rollout_id=None, group_id=None, group_size=None,
                 sampling_seed=None, sampling_scope=None, task_ref=None, rift_mode='train',
                 customer_seed=None, protocol_ref=None, protocol_digest=None):
        self.world, self.task, self.profile, self.customer_agent = world, task, profile, customer_agent
        self.parser, self.budget, self.parser_schemas = parser, budget, parser_schemas
        self.mode, self.external_events = mode, external_events
        self.rollout_id = rollout_id or uuid4().hex
        self.group_id = group_id
        self.expected_group_size = group_size
        if sampling_seed is not None and (type(sampling_seed) is not int or sampling_seed < 0
                                          or not sampling_scope):
            raise ValueError('probe sampling seed and stable scope must be supplied together')
        self.sampling_seed, self.sampling_scope = sampling_seed, sampling_scope
        self.task_ref, self.rift_mode = task_ref, rift_mode
        self.customer_seed = customer_seed
        self.protocol_ref, self.protocol_digest = protocol_ref, protocol_digest
        if rift_mode not in ('train', 'probe'):
            raise ValueError('RIFT episode mode must be train or probe')
        if rift_mode == 'probe' and (not task_ref or not protocol_ref or not protocol_digest):
            raise ValueError('Probe session must bind task and frozen evaluation protocol')
        self._parser_failure = None
        self._responses = Queue()
        self._finished = None
        self._started = False
        self._waiting = False
        self._sequence = 0

    @classmethod
    def with_verl_parser(cls, world, task, profile, customer_agent, tokenizer, parser_format, **kwargs):
        """Use the pinned VERL parser; no model or worker is created here."""
        from verl.experimental.agent_loop.tool_parser import ToolParser
        from verl.tools.schemas import OpenAIFunctionToolSchema
        from usami_v2.runtime.agents import service_tools
        schemas = [OpenAIFunctionToolSchema.model_validate(row) for row in service_tools()]
        return cls(world, task, profile, customer_agent,
            ToolParser.get_tool_parser(parser_format, tokenizer), parser_schemas=schemas, **kwargs)

    async def start(self):
        if self._started: raise ValueError('session has already started')
        self._started = True
        self._loop = asyncio.get_running_loop()
        self._events = asyncio.Queue()
        from usami_v2.runtime.runner import run_episode
        from usami_v2.runtime.agents import OpenAIServiceAgent, service_tools
        self._tools = service_tools()
        owner = self
        class BlockingService:
            uses_model = True
            last_call = None
            def decide(self, prompt):
                asyncio.run_coroutine_threadsafe(owner._events.put(('prompt', prompt)), owner._loop).result()
                result = owner._responses.get()
                if isinstance(result, _ParserExecutionFailure):
                    self.last_call = result.call
                    raise result
                if isinstance(result, BaseException): raise result
                return result
        def run_native():
            try:
                result = run_episode(self.world, self.task, self.profile, self.customer_agent, BlockingService(),
                    budget=self.budget, mode=self.mode, external_events=self.external_events,
                    cancel_check=lambda: owner._parser_failure is not None)
                asyncio.run_coroutine_threadsafe(self._events.put(('finished', result)), self._loop).result()
            except BaseException as exc:
                asyncio.run_coroutine_threadsafe(self._events.put(('error', exc)), self._loop).result()
        self._worker = asyncio.create_task(asyncio.to_thread(run_native))
        kind, value = await self._events.get()
        if kind != 'prompt':
            await self._worker
            raise ValueError('native episode ended before the first Service decision') from (value if kind == 'error' else None)
        self._prompt = value
        self._waiting = True
        messages = [{'role': 'system', 'content': OpenAIServiceAgent.SYSTEM + "\n\n"
                    + service_tool_protocol_reminder(self._tools)},
            {'role': 'user', 'content': json.dumps({'authenticated_customer_id': value.customer_id,
                'available_merchant_scope_ids': list(value.available_merchant_scope_ids),
                'customer_request': value.customer_request}, ensure_ascii=False)}]
        history = list(value.conversation)
        if history and history[0].role == 'user' and history[0].content == value.customer_request: history = history[1:]
        messages.extend(row.model_dump(mode='json', exclude_none=True) for row in history)
        return messages, self._tools

    async def advance(self, token_ids, prompt_ids, sampling_params, token_logprobs=None, generation_id=None,
                      serving_fields=None):
        if not self._waiting: raise ValueError('no pending Service decision')
        from usami.service.models import ServiceError
        from usami_v2.runtime.agents import AgentCall, ServiceAction
        response = {'choices': [{'message': {'role': 'assistant', 'content': ''},
                                 'finish_reason': 'stop'}],
                    'generated_token_ids': list(token_ids),
                    'generated_log_probs': None if token_logprobs is None else list(token_logprobs),
                    'serving_fields': dict(serving_fields or {})}
        call = AgentCall(role='service', request={'prompt_ids': list(prompt_ids),
            'sampling_params': dict(sampling_params), 'rollout_id': self.rollout_id,
            'group_id': self.group_id, 'group_size': self.expected_group_size,
            'generation_id': generation_id, 'rift_mode': self.rift_mode,
            'task_ref': self.task_ref, 'sampling_seed': self.sampling_seed,
            'customer_seed': self.customer_seed, 'protocol_ref': self.protocol_ref,
            'protocol_digest': self.protocol_digest}, response=response)
        try:
            content, calls = await self.parser.extract_tool_calls(list(token_ids), self.parser_schemas)
            if not isinstance(content, str) or not isinstance(calls, (list, tuple)):
                raise TypeError('VERL parser returned an invalid result')
            message = {'role': 'assistant', 'content': content or ''}
            if calls:
                message['tool_calls'] = [
                    {'id': item.tool_call_id or f'rift-call-{self._sequence + index}',
                     'type': 'function',
                     'function': {'name': item.name, 'arguments': item.arguments}}
                    for index, item in enumerate(calls)]
            response['choices'][0]['message'] = message
            response['choices'][0]['finish_reason'] = 'tool_calls' if calls else 'stop'
            call = AgentCall(role='service', request=call.request, response=response)
            if len(calls) > 1:
                action = ServiceAction(kind='invalid', call=call)
            elif calls:
                try:
                    parsed = ServiceAction.from_tool_message(
                        response, self._prompt.customer_id, self._prompt.observed_subject_ids,
                        self._prompt.available_merchant_scope_ids)
                except ServiceError as exc:
                    if getattr(exc, 'code', None) != 'MODEL_OUTPUT_INVALID':
                        raise
                    action = ServiceAction(kind='invalid', call=call)
                else:
                    action = parsed.model_copy(update={'call': call})
            else:
                action = (ServiceAction(kind='reply', message=content, call=call)
                          if content.strip() else ServiceAction(kind='invalid', call=call))
        except Exception as exc:
            # Parser/runtime failures are infrastructure uncertainty. Send the
            # exact generated call through Usami's cancellation path so its
            # native Episode and token evidence are retained with UNKNOWN reward.
            self._parser_failure = exc
            self._waiting = False
            self._responses.put(_ParserExecutionFailure(call, exc))
            kind, value = await self._events.get()
            if kind == 'finished':
                self._finished = value
                await self._worker
                return {'role': 'assistant', 'content': ''}, [], True
            await self._worker
            raise RuntimeError('native Usami failed while recording a parser execution error') from value
        self._sequence += 1
        before = [m.model_dump(mode='json', exclude_none=True) for m in self._prompt.conversation]
        self._waiting = False
        self._responses.put(action)
        kind, value = await self._events.get()
        if kind == 'error':
            await self._worker
            raise RuntimeError('native Usami episode failed') from value
        if kind == 'finished':
            self._finished = value
            await self._worker
            return message, [], True
        self._prompt = value
        self._waiting = True
        after = [m.model_dump(mode='json', exclude_none=True) for m in value.conversation]
        if after[:len(before)] != before or len(after) <= len(before) or after[len(before)].get('role') != 'assistant':
            await self.abort()
            raise ValueError('native public conversation does not extend the preceding context')
        appended = after[len(before):]
        return appended[0], appended[1:], False

    async def finish(self):
        if self._finished is None: raise ValueError('native episode has not finished')
        return self._finished

    async def cancel_for_budget(self):
        """Cancel a pending decision when its public context cannot fit."""
        if not self._waiting:
            raise ValueError('no pending Service decision to cancel')
        self._parser_failure = RuntimeError('policy context exceeds token budget')
        self._waiting = False
        self._responses.put(self._parser_failure)
        kind, value = await self._events.get()
        await self._worker
        if kind != 'finished':
            raise RuntimeError('native Usami failed while cancelling an over-budget episode') from value
        self._finished = value

    async def abort(self):
        worker = getattr(self, '_worker', None)
        if worker is not None and not worker.done():
            self._waiting = False
            self._responses.put(RuntimeError('RIFT rollout aborted'))
        if worker is not None:
            await asyncio.shield(worker)
