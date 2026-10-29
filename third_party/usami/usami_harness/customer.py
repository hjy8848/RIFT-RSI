"""Open discovery through native typed intention tools, never a fixed task feed."""
import copy
import json
from typing import Any, Dict, List, Optional
from pydantic import Field
from usami.service.models import LLMRecord, ModelSettings, ServiceError, strict_json
from usami.service.runner import completion_message
from usami.service.client import RecordedCompletionClient
from .models import Model, digest
from .service import Experiment, Execution, ServiceSession, replay_execution, VERSION
from .tools import customer_schemas


class CustomerAction(Model):
    kind: str
    message: str = ''
    intentions: List[Dict[str, Any]] = Field(default_factory=list)
    reason: str = ''


class Interaction(Model):
    version: str = VERSION
    experiment: Experiment
    customer_model: ModelSettings
    customer_method: str
    customer_llm_records: List[LLMRecord]
    customer_messages: List[Dict[str, Any]]
    actions: List[CustomerAction]
    replies: List[Dict[str, Any]]
    format_recoveries: int
    service: Execution
    status: str
    interaction_hash: str = ''

    def computed_hash(self):
        return digest(self.model_dump(mode='json', exclude={'interaction_hash'}))


def parse_action(adapter, calls, names, *, syntax_fingerprint=False):
    """Validate the entire batch before granting any intention.

    Fingerprinting can decode ONLY a JSON-string params object for a repair
    comparison. It never dispatches or grants this malformed representation.
    A retry must independently declare exactly these values and messages.
    """
    if not calls:
        raise ValueError('Customer must use intention, ask_service or stop_exploration tools')
    messages, operations = [], []
    for call in calls:
        name = call['function']['name']
        args = strict_json(call['function']['arguments'])
        if not isinstance(args, dict):
            raise ValueError('Customer tool arguments must be an object')
        if name == 'stop_exploration':
            if len(calls) != 1 or set(args) != {'reason'} or not isinstance(args['reason'], str) or not args['reason'].strip():
                raise ValueError('stop_exploration requires a reason and must be exclusive')
            return CustomerAction(kind='stop', reason=args['reason'])
        if name == 'ask_service':
            if set(args) != {'message'}:
                raise ValueError('ask_service grants no writes')
        elif name.startswith('intend_'):
            op_name = name[len('intend_'):]
            if op_name not in names or op_name not in adapter.PARAMETERS or op_name in adapter.Operation.queries:
                raise ValueError('Customer intention not in frozen write interface')
            if set(args) != {'scope_id', 'resource_id', 'params', 'message'}:
                raise ValueError('Customer intention requires exact operation fields and message')
            params = args['params']
            if syntax_fingerprint and isinstance(params, str):
                params = strict_json(params)
            op = adapter.Operation(name=op_name, scope_id=args['scope_id'], resource_id=args['resource_id'], params=params)
            operations.append(op.model_dump(mode='json'))
        else:
            raise ValueError('unknown Customer tool')
        if not isinstance(args['message'], str) or not args['message'].strip():
            raise ValueError('a natural-language Customer message is required')
        messages.append(args['message'])
    return CustomerAction(kind='request', message='\n'.join(messages), intentions=operations)


class CustomerSession:
    def __init__(self, adapter, experiment, customer_client, service_client, method='adaptive'):
        if method not in {'adaptive', 'careful'}:
            raise ValueError('unknown general exploration method')
        self.adapter, self.experiment = adapter, experiment.model_copy(deep=True)
        self.customer, self.customer_model = customer_client, customer_client.settings.model_copy(deep=True)
        # Native function calling is the frozen structured-output protocol. An
        # incompatible JSON-only mode must be removed BEFORE settings are frozen.
        if 'response_format' in self.customer_model.options:
            raise ValueError('native Customer tools require settings without response_format')
        self.service = ServiceSession(adapter, experiment, service_client)
        self.method, self.tools = method, customer_schemas(adapter, experiment.tools)
        self.records, self.actions, self.replies = [], [], []
        self.seen, self.recoveries = set(), 0
        style = ('Explore business questions adaptively and change direction based on actual replies.' if method == 'adaptive'
                 else 'Inspect actual state, reason about feasibility and compare outcomes carefully before each new request.')
        self.messages = [{'role': 'system', 'content':
            'You are an autonomous Customer exploring a simulated ' + adapter.name + ' business. '
            'No task, target, family, script or reference solution is assigned. Choose your own questions, '
            'requests and exploration strategy. ' + style + ' You see only your scoped initial view and '
            'actual Service replies, not hidden world state, Oracle results or internal rules. '
            'Use ask_service for discussion/query requests without write authority. Use intend_<operation> '
            'with typed params OBJECTS to independently declare exact intended writes before Service acts. '
            'These tools express your intentions, not a prescribed Service execution order. '
            'Batch only exact operations you intend to authorize; Service must choose how to act. '
            'Never impersonate another user or supply external simulator outcomes. '
            'Use stop_exploration with your reason when finished. '
            'Public policies: ' + json.dumps(adapter.POLICY, sort_keys=True) + '. '
            'Frozen budget: ' + experiment.budget.model_dump_json()},
            {'role': 'user', 'content': 'Your permission-filtered initial view: ' +
             json.dumps(adapter.observation(self.service.state, experiment.identity), sort_keys=True, ensure_ascii=False)}]

    def _completion(self):
        if len(self.records) >= self.experiment.budget.max_customer_llm_calls:
            return None, None, 'customer_llm_budget_exhausted'
        request = copy.deepcopy(self.customer_model.request(self.messages, self.tools))
        record = LLMRecord(request=request)
        self.records.append(record)
        try:
            record.response = self.customer.complete(copy.deepcopy(request))
        except KeyboardInterrupt:
            record.error = {'stage': 'transport', 'code': 'PROVIDER_INTERRUPTED', 'message': 'Interrupted; no retry'}
            return record, None, 'interrupted'
        except ServiceError as exc:
            record.response = exc.response
            record.error = {'stage': 'transport', 'code': exc.code, 'message': str(exc)}
            return record, None, 'customer_model_output_error' if exc.code == 'MODEL_OUTPUT_INVALID' else 'customer_provider_error'
        except Exception:
            record.error = {'stage': 'transport', 'code': 'PROVIDER_CLIENT_ERROR', 'message': 'Unexpected completion client error'}
            return record, None, 'customer_provider_error'
        return record, record.response, None

    def _action(self):
        record, raw, status = self._completion()
        if status:
            return None, status
        try:
            message, calls = completion_message(raw, set(self.seen))
            action = parse_action(self.adapter, calls, self.experiment.tools)
        except (ValueError, TypeError, ServiceError) as exc:
            record.error = {'stage': 'protocol', 'code': 'CUSTOMER_ACTION_INVALID', 'message': str(exc)}
            # Recovery is permitted only for already fully specified, schema-valid
            # business values. Missing, unknown or changed intentions cannot recover.
            try:
                message, calls = completion_message(raw, set(self.seen))
                original = parse_action(self.adapter, calls, self.experiment.tools, syntax_fingerprint=True)
            except (ValueError, TypeError, ServiceError):
                return None, 'customer_protocol_error'
            if self.recoveries >= self.experiment.budget.max_format_recoveries:
                return None, 'customer_protocol_error'
            self.messages.append({'role': 'assistant', 'content': 'Invalid previous function output: ' + json.dumps(calls, sort_keys=True)})
            self.messages.append({'role': 'user', 'content':
                'FORMAT ONLY recovery: independently re-emit precisely the same tool names, messages, '
                'scope/resource IDs and parameter values, with params as typed JSON objects. '
                'Do not add, delete, reorder or change any business intention. No authority has been granted.'})
            self.recoveries += 1
            retry, retry_raw, status = self._completion()
            if status:
                return None, status
            try:
                message, calls = completion_message(retry_raw, set(self.seen))
                action = parse_action(self.adapter, calls, self.experiment.tools)
                if action != original:
                    raise ValueError('format recovery changed Customer intention')
            except (ValueError, TypeError, ServiceError) as exc:
                retry.error = {'stage': 'protocol', 'code': 'CUSTOMER_RECOVERY_INVALID', 'message': str(exc)}
                return None, 'customer_protocol_error'
        self.seen.update(call['id'] for call in calls)
        self.messages.append(message)
        return (action, calls), None

    def run(self):
        status = 'customer_turn_budget_exhausted'
        for index in range(self.experiment.budget.max_customer_turns):
            result, error = self._action()
            if error:
                status = error
                break
            action, calls = result
            self.actions.append(action)
            if action.kind == 'stop':
                status = 'customer_stopped'
                for call in calls:
                    self.messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': '{"stopped":true}'})
                break
            grants = [self.adapter.Operation.model_validate(g) for g in action.intentions]
            reply = self.service.respond(action.message, grants, turn_id=f'customer_{index+1:03d}')
            self.replies.append(reply)
            # Service state hashes and private Oracle findings are not Customer observations.
            public_reply = {'status': reply['status'], 'final_reply': reply['final_reply']}
            for call in calls:
                self.messages.append({'role': 'tool', 'tool_call_id': call['id'],
                                      'content': json.dumps(public_reply, sort_keys=True, ensure_ascii=False)})
            if reply['status'] != 'completed':
                status = reply['status']
                break
        execution = self.service.finish(status)
        interaction = Interaction(experiment=self.experiment, customer_model=self.customer_model,
            customer_method=self.method, customer_llm_records=self.records, customer_messages=self.messages,
            actions=self.actions, replies=self.replies, format_recoveries=self.recoveries,
            service=execution, status=status)
        interaction.interaction_hash = interaction.computed_hash()
        return interaction


def replay_interaction(adapter, interaction, experiment=None):
    if interaction.version != VERSION or interaction.interaction_hash != interaction.computed_hash():
        raise ValueError('interaction integrity or version mismatch')
    if experiment is not None and interaction.experiment != experiment:
        raise ValueError('externally frozen Discovery experiment differs')
    replay_execution(adapter, interaction.service, interaction.experiment)
    customer = RecordedCompletionClient(interaction.customer_model, interaction.customer_llm_records)
    service = RecordedCompletionClient(interaction.service.model, interaction.service.llm_records)
    rebuilt = CustomerSession(adapter, interaction.experiment, customer, service, interaction.customer_method).run()
    if (rebuilt != interaction or customer.index != len(interaction.customer_llm_records)
            or service.index != len(interaction.service.llm_records)):
        raise ValueError('Customer messages, independent intentions or episode replay differs')
    return rebuilt
