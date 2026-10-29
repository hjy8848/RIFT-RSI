"""Versioned read intentions over the unchanged native CloudOps interaction loop."""
import copy
import json
from typing import List, Literal
from pydantic import model_validator
from usami.service.models import LLMRecord, ServiceError, strict_json
from usami.service.runner import completion_message
from usami.service.client import RecordedCompletionClient
from usami_cloudops.models import Model, Operation, WRITES, PARAMETERS
from usami_cloudops.observations import observation
from usami_cloudops.service import ServiceSession, replay_execution
from usami_cloudops import customer as legacy

Action = legacy.Action

READ_PURPOSES = {'get_customer_context': {'catalog'}, 'get_project': {'state', 'quota'},
    'get_project_members': {'permissions'}, 'list_instances': {'state', 'dependencies'},
    'get_instance': {'state', 'dependencies'}, 'get_invoice': {'billing'}}


class ReadProbe(Model):
    name: Literal['get_customer_context', 'get_project', 'get_project_members', 'list_instances', 'get_instance', 'get_invoice']
    project_id: str = ''
    resource_id: str = ''
    purpose: Literal['catalog', 'state', 'permissions', 'quota', 'dependencies', 'billing']

    @model_validator(mode='after')
    def read_only(self):
        if self.purpose not in READ_PURPOSES[self.name]: raise ValueError('read purpose does not match query')
        if self.name == 'get_customer_context':
            if self.project_id or self.resource_id: raise ValueError('context read takes no object IDs')
        elif not self.project_id or not self.resource_id: raise ValueError('scoped read needs explicit objects')
        return self

    def operation(self):
        return None if self.name == 'get_customer_context' else Operation(name=self.name,
            project_id=self.project_id, resource_id=self.resource_id, params={})


class ProbeAction(Action):
    read_probes: List[ReadProbe]

    @model_validator(mode='after')
    def declared_probes(self):
        if self.type == 'stop' and self.read_probes: raise ValueError('stop cannot contain reads')
        if self.type == 'send_message' and not (self.read_probes or self.requested_operations):
            raise ValueError('probe protocol requires explicit read or write intentions')
        return self


def public_action_schema(tools, read_protocol=False):
    """Expose complete typed write inputs without private rules or task data."""
    schema = (ProbeAction if read_protocol else Action).model_json_schema()
    operation = schema['$defs']['Operation']
    writes = [name for name in tools if name in WRITES]
    operation['properties']['name']['enum'] = writes
    # Operation's runtime validator selects each parameter model. Publish that
    # selection explicitly: a generic dict would omit payment/quota/etc. inputs.
    operation['allOf'] = [
        {'if': {'properties': {'name': {'const': name}}, 'required': ['name']},
         'then': {'properties': {'params': PARAMETERS[name].model_json_schema()}, 'required': ['params']}}
        for name in writes]
    schema['allOf'] = [
        {'if': {'properties': {'type': {'const': 'send_message'}}, 'required': ['type']},
         'then': {'properties': {'message': {'minLength': 1}, 'reason': {'const': ''}}, 'required': ['message']}},
        {'if': {'properties': {'type': {'const': 'stop'}}, 'required': ['type']},
         'then': {'properties': {'reason': {'minLength': 1}, 'message': {'const': ''},
                                 'requested_operations': {'maxItems': 0}}, 'required': ['reason']}}]
    if read_protocol:
        schema['$defs']['ReadProbe']['properties']['name']['enum'] = [name for name in tools
            if name in {'get_customer_context', 'get_project', 'get_project_members', 'list_instances', 'get_instance', 'get_invoice'}]
        schema['$defs']['ReadProbe']['allOf'] = [
            {'if': {'properties': {'name': {'const': name}}, 'required': ['name']},
             'then': {'properties': {'purpose': {'enum': sorted(purposes)},
                'project_id': {'const': ''} if name == 'get_customer_context' else {'minLength': 1},
                'resource_id': {'const': ''} if name == 'get_customer_context' else {'minLength': 1}},
                'required': ['purpose'] + ([] if name == 'get_customer_context' else ['project_id', 'resource_id'])}}
            for name, purposes in READ_PURPOSES.items()]
        schema['allOf'][0]['then']['anyOf'] = [
            {'properties': {'read_probes': {'minItems': 1}}, 'required': ['read_probes']},
            {'properties': {'requested_operations': {'minItems': 1}}, 'required': ['requested_operations']}]
        schema['allOf'][1]['then']['properties']['read_probes'] = {'maxItems': 0}
    return schema


class CustomerAgent(legacy.CustomerAgent):
    def __init__(self, view, client, tools, strategy=None):
        super().__init__(view, client, tools, strategy)
        self.read_protocol = True
        old = json.dumps(legacy.public_action_schema(tools), sort_keys=True)
        self.prompt = self.prompt.replace(old, json.dumps(public_action_schema(tools, True), sort_keys=True))
        self.prompt += (' Read probe protocol v1: independently declare every information request in read_probes, '
            'with exact observed IDs and a typed information purpose. read_probes=[] for write-only requests and stop. '
            'Reads never grant writes. Do not infer your intention from Service tools or invent evaluator answers.')
        self.reset()
    def next_action(self, remaining):
        view = {'initial_observation': self.view} if not self.records else {'last_exchange':self.feedback}
        view['remaining_budget'] = remaining
        self.messages.append({'role':'user','content':json.dumps(view,sort_keys=True,ensure_ascii=False)})
        request = copy.deepcopy(self.model.request(self.messages,[]))
        # Customer emits JSON actions, not function calls. Some compatible
        # providers reject an empty tools array; record the actual wire shape.
        request.pop('tools'); request.pop('tool_choice')
        record = LLMRecord(request=request); self.records.append(record)
        try: record.response = self.client.complete(copy.deepcopy(record.request))
        except KeyboardInterrupt:
            record.error={'stage':'transport','code':'PROVIDER_INTERRUPTED','message':'Customer interrupted; no retry'}
            raise ServiceError('PROVIDER_INTERRUPTED','Customer interrupted') from None
        except ServiceError as exc:
            record.response = exc.response; record.error = {'stage':'transport','code':exc.code,'message':str(exc)}; raise
        except Exception:
            record.error = {'stage':'transport','code':'PROVIDER_CLIENT_ERROR','message':'Unexpected Customer transport error'}
            raise ServiceError('PROVIDER_CLIENT_ERROR','Customer request failed') from None
        try:
            message, calls = completion_message(record.response,set())
            if calls: raise ValueError('Customer cannot call tools')
            action = (ProbeAction if self.read_protocol else Action).model_validate(strict_json(message['content']))
        except (ValueError,TypeError,ServiceError):
            record.error = {'stage':'protocol','code':'CUSTOMER_OUTPUT_INVALID','message':'Expected CloudOps Customer action'}
            raise ServiceError('CUSTOMER_OUTPUT_INVALID','Expected CloudOps Customer action') from None
        self.messages.append(message); return action


class IntentAdapter(legacy.IntentAdapter):
    def prepare(self, action, world, identity, tools):
        message, grants, dispositions = super().prepare(action, world, identity, tools)
        if action.read_probes:
            message += '\nCustomer explicit read probes (no write authorization):\n' + json.dumps(
                [probe.model_dump(mode='json') for probe in action.read_probes], sort_keys=True)
        return message, grants, dispositions


class InteractionConfig(legacy.InteractionConfig):
    read_probe_protocol: Literal['cloudops-read-probes-v1'] = 'cloudops-read-probes-v1'


class Interaction(legacy.Interaction):
    version: Literal['cloudops-interaction-0.2.0'] = 'cloudops-interaction-0.2.0'
    config: InteractionConfig
    actions: List[ProbeAction]


class InteractionSession(legacy.InteractionSession):
    def __init__(self, config, customer_client, service_client):
        if service_client.settings != config.service_model or customer_client.settings != config.customer_model:
            raise ValueError('clients differ from frozen model settings')
        self.config = config.model_copy(deep=True)
        self.service = ServiceSession(config.experiment, service_client)
        self.view = observation(self.service.state, config.experiment.identity)
        self.customer = CustomerAgent(self.view, customer_client, config.experiment.tools, config.strategy)
        self.reset()

    def reset(self):
        state = super().reset()
        self.adapter = IntentAdapter(self.view)
        return state

    def run(self):
        # The existing native loop owns turns, authority, budgets and Engine.
        result = super().run()
        raw = result.model_dump(mode='json', exclude={'version', 'config', 'actions', 'interaction_hash'})
        trace = Interaction(config=self.config, actions=self.actions, **raw)
        trace.interaction_hash = trace.computed_hash()
        return trace


def replay_interaction(trace, config=None):
    if trace.version != 'cloudops-interaction-0.2.0' or trace.interaction_hash != trace.computed_hash():
        raise ValueError('CloudOps read-probe interaction integrity mismatch')
    if config is not None and trace.config != config: raise ValueError('frozen read protocol differs')
    replay_execution(trace.service_execution, trace.config.experiment)
    customer = RecordedCompletionClient(trace.config.customer_model, trace.customer_records)
    service = RecordedCompletionClient(trace.config.service_model, trace.service_execution.llm_records)
    regenerated = InteractionSession(trace.config, customer, service).run()
    if regenerated != trace or customer.index != len(trace.customer_records) or service.index != len(trace.service_execution.llm_records):
        raise ValueError('CloudOps read declarations, observation, authority or conversation replay differs')
    return regenerated
