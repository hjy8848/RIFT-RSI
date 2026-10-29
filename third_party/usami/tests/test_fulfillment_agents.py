"""Native protocol, independent authority, persistence and exact model replay."""
import copy
import json
import pytest
from usami.service.models import ModelSettings, ServiceError
from usami_harness.models import Budget
from usami_fulfillment import generate_world, Operation
from usami_fulfillment.models import TOOLS
from usami_fulfillment.service import Experiment, ServiceSession, replay_execution
from usami_fulfillment.customer import CustomerSession, replay_interaction
from usami_fulfillment.tools import schemas


class Client:
    settings = ModelSettings(model='offline-fixture', base_url='https://fixture.invalid/v1')
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []
    def complete(self, request):
        self.requests.append(copy.deepcopy(request))
        result = next(self.replies)
        if isinstance(result, Exception):
            raise result
        return result


def reply(text=None, calls=()):
    return {'choices': [{'finish_reason': 'tool_calls' if calls else 'stop',
                         'message': {'role': 'assistant', 'content': text,
                                     'tool_calls': list(calls)}}]}


def call(identifier, name, arguments):
    return {'id': identifier, 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments)}}


def op(name='capture_payment', **params):
    return Operation(name=name, scope_id='w0', resource_id='order0', params=params)


def args(operation):
    return operation.model_dump(mode='json', exclude={'name'})


def experiment(**limits):
    return Experiment(domain='fulfillment', world=generate_world().model_dump(mode='json'),
                      identity='u0', tools=list(TOOLS), budget=Budget(**limits))


def charge():
    return op(amount_cents=5300, key='paid', token='card_approved')


def test_service_independently_queries_and_executes_a_multistep_turn():
    payment, confirm = charge(), op('confirm_order')
    client = Client([reply(calls=[call('context', 'get_customer_context', {})]),
                     reply(calls=[call('pay', payment.name, args(payment))]),
                     reply(calls=[call('confirm', confirm.name, args(confirm))]), reply('Confirmed.')])
    session = ServiceSession(experiment(), client)
    assert session.respond('Pay and confirm my order.', [confirm, payment])['status'] == 'completed'
    execution = session.finish()
    assert execution.world_trace.final_state['orders']['order0']['status'] == 'confirmed'
    assert execution.oracle['status'] == 'pass'
    assert replay_execution(execution, experiment()) == execution
    assert session.reset() == generate_world()


def test_service_cannot_grant_itself_authority_and_grants_expire():
    payment = charge()
    client = Client([reply('Will handle later.'), reply(calls=[call('late', payment.name, args(payment))]), reply('Denied.')])
    session = ServiceSession(experiment(), client)
    session.respond('You may charge now.', [payment])
    session.respond('Just checking status.')
    result = session.finish()
    assert result.tool_records[0].response['error_code'] == 'UNAUTHORIZED_OPERATION'
    assert result.world_trace.final_state['payments'] == {}
    assert replay_execution(result) == result


def test_single_use_grants_do_not_authorize_duplicate_tool_calls():
    payment = charge()
    session = ServiceSession(experiment(), Client([reply(calls=[call('one', payment.name, args(payment)),
                                                               call('two', payment.name, args(payment))]), reply('Done.')]))
    session.respond('Pay once.', [payment])
    result = session.finish()
    assert result.tool_records[0].response['success']
    assert result.tool_records[1].response['error_code'] == 'UNAUTHORIZED_OPERATION'
    assert len(result.world_trace.final_state['payments']) == 1


@pytest.mark.parametrize('bad_args', [
    {'scope_id': 'w0', 'resource_id': 'order0', 'params': '{}'},
    {'scope_id': 'w0', 'resource_id': 'order0', 'params': {'quantity': 4}},
    {'scope_id': 'w0', 'resource_id': 'order0', 'params': {}, 'actor': 'u5'},
])
def test_bad_service_arguments_are_recorded_and_cannot_mutate(bad_args):
    session = ServiceSession(experiment(), Client([reply(calls=[call('bad', 'confirm_order', bad_args)]), reply('Error.')]))
    session.respond('Confirm.')
    result = session.finish()
    assert result.tool_records[0].response['error_code'] == 'TOOL_ARGUMENT_INVALID'
    assert not result.tool_records[0].dispatched
    assert result.world_trace.initial_state == result.world_trace.final_state
    assert replay_execution(result) == result


@pytest.mark.parametrize('output', [{}, reply(calls=[call('same', 'get_customer_context', {}), call('same', 'get_customer_context', {})])])
def test_model_batch_validation_precedes_dispatch(output):
    session = ServiceSession(experiment(), Client([output]))
    response = session.respond('Inspect my context.')
    assert response['status'] == 'model_output_error'
    result = session.finish(response['status'])
    assert not result.tool_records
    assert replay_execution(result) == result


def test_provider_failure_replays_without_network():
    session = ServiceSession(experiment(), Client([ServiceError('PROVIDER_TIMEOUT', 'Timeout')]))
    response = session.respond('Hello.')
    assert response['status'] == 'provider_error'
    result = session.finish(response['status'])
    assert replay_execution(result) == result


def test_service_tool_budget_counts_invalid_arguments_too():
    session = ServiceSession(experiment(max_service_tool_calls=1), Client([reply(calls=[
        call('bad', 'confirm_order', {'x': 1}), call('good', 'get_customer_context', {})])]))
    response = session.respond('Inspect.')
    assert response['status'] == 'tool_budget_exhausted'
    result = session.finish(response['status'])
    assert len(result.tool_records) == 1
    assert replay_execution(result) == result


def test_llm_budget_stops_even_when_model_keeps_querying():
    session = ServiceSession(experiment(max_service_llm_calls=1), Client([reply(calls=[call('context', 'get_customer_context', {})])]))
    response = session.respond('Inspect.')
    assert response['status'] == 'llm_budget_exhausted'
    assert replay_execution(session.finish(response['status'])).status == response['status']


def test_customer_native_intention_persists_state_across_turns():
    payment, confirm = charge(), op('confirm_order')
    customer = Client([
        reply(calls=[call('intent_pay', 'intend_capture_payment', {**args(payment), 'message': 'Pay my order.'})]),
        reply(calls=[call('intent_confirm', 'intend_confirm_order', {**args(confirm), 'message': 'Now confirm it.'})]),
        reply(calls=[call('stop', 'stop_exploration', {'reason': 'Finished my chosen experiment.'})])])
    service = Client([reply(calls=[call('pay', payment.name, args(payment))]), reply('Paid.'),
                      reply(calls=[call('confirm', confirm.name, args(confirm))]), reply('Confirmed.')])
    interaction = CustomerSession(experiment(), customer, service).run()
    assert interaction.status == 'customer_stopped'
    assert len(interaction.replies) == 2 and len(interaction.service.inputs) == 2
    assert interaction.service.world_trace.final_state['orders']['order0']['status'] == 'confirmed'
    assert interaction.service.oracle['status'] == 'pass'
    assert replay_interaction(interaction, experiment()) == interaction
    assert all('oracle' not in m['content'] for m in interaction.customer_messages if m['role'] == 'tool')


def test_customer_batch_invalid_later_intention_grants_nothing():
    customer = Client([reply(calls=[call('good', 'intend_capture_payment', {**args(charge()), 'message': 'Pay.'}),
                                  call('bad', 'intend_confirm_order', {'params': {}})])])
    interaction = CustomerSession(experiment(), customer, Client([])).run()
    assert interaction.status == 'customer_protocol_error'
    assert not interaction.service.inputs and not interaction.service.trusted_requests
    assert replay_interaction(interaction) == interaction


@pytest.mark.parametrize('change', [False, True])
def test_format_recovery_preserves_exact_intention_and_actual_budget(change):
    payment = charge()
    malformed = {**args(payment), 'message': 'Pay exactly this amount.'}
    malformed['params'] = json.dumps(malformed['params'])
    repaired = {**args(payment), 'message': 'Pay exactly this amount.'}
    if change:
        repaired['params']['amount_cents'] = 5200
    customer = Client([reply(calls=[call('bad', 'intend_capture_payment', malformed)]),
                       reply(calls=[call('retry', 'intend_capture_payment', repaired)]),
                       reply(calls=[call('stop', 'stop_exploration', {'reason': 'Done.'})])])
    service = Client([reply(calls=[call('pay', payment.name, args(payment))]), reply('Paid.')])
    interaction = CustomerSession(experiment(), customer, service).run()
    assert interaction.format_recoveries == 1
    assert interaction.customer_llm_records[0].error['code'] == 'CUSTOMER_ACTION_INVALID'
    if change:
        assert interaction.status == 'customer_protocol_error' and not interaction.service.inputs
        assert len(interaction.customer_llm_records) == 2
    else:
        assert interaction.status == 'customer_stopped' and len(interaction.customer_llm_records) == 3
        assert interaction.service.world_trace.final_state['payments']
    assert replay_interaction(interaction) == interaction


def test_recovery_cannot_exceed_customer_llm_budget():
    malformed = {**args(charge()), 'message': 'Pay.'}
    malformed['params'] = json.dumps(malformed['params'])
    interaction = CustomerSession(experiment(max_customer_llm_calls=1), Client([
        reply(calls=[call('bad', 'intend_capture_payment', malformed)])]), Client([])).run()
    assert interaction.status == 'customer_llm_budget_exhausted'
    assert len(interaction.customer_llm_records) == 1 and not interaction.service.inputs
    assert replay_interaction(interaction) == interaction


def test_customer_queries_do_not_create_write_grants():
    customer = Client([reply(calls=[call('ask', 'ask_service', {'message': 'Please tell me whether my order can confirm.'})]),
                       reply(calls=[call('stop', 'stop_exploration', {'reason': 'Enough.'})])])
    service = Client([reply(calls=[call('write', 'confirm_order', args(op('confirm_order')))]), reply('Not authorized.')])
    interaction = CustomerSession(experiment(), customer, service).run()
    assert interaction.service.tool_records[0].response['error_code'] == 'UNAUTHORIZED_OPERATION'
    assert not interaction.actions[0].intentions
    assert replay_interaction(interaction) == interaction


def test_schema_definitions_are_inlined_and_inputs_are_operation_specific():
    tools = {t['function']['name']: t['function']['parameters'] for t in schemas(TOOLS)}
    create = tools['create_order']['properties']['params']
    assert '$ref' not in json.dumps(create) and '$defs' not in json.dumps(create)
    assert create['properties']['lines']['items']['properties']['quantity']['type'] == 'integer'
    assert 'accepted' not in json.dumps(tools['inspect_return'])
    assert tools['confirm_order']['properties']['params']['properties'] == {}


def test_external_budget_or_identity_tamper_is_rejected():
    session = ServiceSession(experiment(), Client([reply('Hello.')]))
    session.respond('Hello.')
    result = session.finish()
    other = experiment(max_service_tool_calls=1)
    with pytest.raises(ValueError):
        replay_execution(result, other)
    from usami_harness.models import Request
    result.trusted_requests['fabricated'] = Request(id='fabricated',identity='u5',operation=charge().model_dump(mode='json'))
    with pytest.raises(ValueError):
        replay_execution(result)
