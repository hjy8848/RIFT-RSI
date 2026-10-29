"""Hotel adapter uses native intentions, exact actor identity and persistent state."""
import pytest
from usami_harness.models import Budget
from usami_stayops import generate_world,Operation
from usami_stayops.models import TOOLS
from usami_stayops.service import Experiment,ServiceSession,replay_execution
from usami_stayops.customer import CustomerSession,replay_interaction
from test_fulfillment_agents import Client,call,reply


def experiment(**limits):
    return Experiment(domain='stayops',world=generate_world().model_dump(mode='json'),identity='u0',tools=list(TOOLS),budget=Budget(**limits))


def hold():
    return Operation(name='create_hold',scope_id='p0',resource_id='p0',params={'reservation_id':'b0','guest_id':'g0',
        'room_type_id':'p0_standard','rate_plan_id':'p0_flex','arrival':'2026-01-15','departure':'2026-01-17','occupants':2,'hold_minutes':30})


def args(op):return op.model_dump(mode='json',exclude={'name'})


def test_multistep_native_customer_intention_and_hotel_state_persistence():
    create=hold();deposit=Operation(name='capture_payment',scope_id='p0',resource_id='b0',params={'amount_cents':2400,'key':'deposit','token':'card_approved'})
    confirm=Operation(name='confirm_reservation',scope_id='p0',resource_id='b0')
    customer=Client([reply(calls=[call('create_intent','intend_create_hold',{**args(create),'message':'Hold a standard room as b0 for my guest.'})]),
                     reply(calls=[call('pay_intent','intend_capture_payment',{**args(deposit),'message':'Pay its deposit.'}),
                                  call('confirm_intent','intend_confirm_reservation',{**args(confirm),'message':'Confirm the booking when eligible.'})]),
                     reply(calls=[call('stop','stop_exploration',{'reason':'My chosen booking is complete.'})])])
    service=Client([reply(calls=[call('ctx','get_customer_context',{})]),reply(calls=[call('create',create.name,args(create))]),reply('Held.'),
                    reply(calls=[call('pay',deposit.name,args(deposit))]),reply(calls=[call('confirm',confirm.name,args(confirm))]),reply('Confirmed.')])
    interaction=CustomerSession(experiment(),customer,service).run()
    assert interaction.status=='customer_stopped' and interaction.service.world_trace.final_state['reservations']['b0']['status']=='confirmed'
    assert interaction.service.oracle['status']=='pass' and replay_interaction(interaction,experiment())==interaction
    assert interaction.actions[0].intentions==[create.model_dump(mode='json')]


def test_service_cannot_infer_booking_grant_from_natural_language_or_change_domain():
    create=hold();session=ServiceSession(experiment(),Client([reply(calls=[call('unauthorized',create.name,args(create))]),reply('Denied.')]))
    session.respond('Can you book me a room?')
    result=session.finish();assert result.tool_records[0].response['error_code']=='UNAUTHORIZED_OPERATION'
    assert not result.world_trace.final_state['reservations'] and replay_execution(result)==result
    from usami_fulfillment.service import replay_execution as stock_replay
    with pytest.raises(ValueError):stock_replay(result)


def test_malformed_dates_and_external_identity_claims_are_recorded_errors():
    bad=args(hold());bad['params']['arrival']='2026-01-15T00:00Z'
    session=ServiceSession(experiment(),Client([reply(calls=[call('bad',hold().name,bad),
        call('fake','verify_guest',{'scope_id':'p0','resource_id':'b0','params':{'outcome':'verified'}})]),reply('Invalid parameters.')]))
    session.respond('Book and verify.')
    result=session.finish();assert len(result.tool_records)==2
    assert all(r.response['error_code']=='TOOL_ARGUMENT_INVALID' for r in result.tool_records)
    assert not result.trusted_requests and replay_execution(result)==result


def test_customer_error_batch_is_atomic_and_budget_is_real():
    create=hold()
    customer=Client([reply(calls=[call('valid','intend_create_hold',{**args(create),'message':'Hold.'}),
                                 call('bad','intend_check_in',{'params':'{}','message':'Check in.'})])])
    interaction=CustomerSession(experiment(max_customer_llm_calls=1),customer,Client([])).run()
    assert interaction.status=='customer_protocol_error' and not interaction.service.inputs
    assert len(interaction.customer_llm_records)==1 and replay_interaction(interaction)==interaction


def test_repeated_questions_end_at_explicit_customer_turn_budget():
    customer=Client([reply(calls=[call('q1','ask_service',{'message':'What properties can I inspect?'})]),
                     reply(calls=[call('q2','ask_service',{'message':'What room types are available?'})])])
    interaction=CustomerSession(experiment(max_customer_turns=2),customer,Client([reply('Public property directory.'),reply('Standard and suite.')])).run()
    assert interaction.status=='customer_turn_budget_exhausted' and len(interaction.replies)==2
    assert replay_interaction(interaction)==interaction


def test_provider_failure_keeps_raw_record_and_replays_without_network():
    from usami.service.models import ServiceError
    interaction=CustomerSession(experiment(),Client([ServiceError('PROVIDER_HTTP_ERROR','HTTP500')]),Client([])).run()
    assert interaction.status=='customer_provider_error' and len(interaction.customer_llm_records)==1
    assert replay_interaction(interaction)==interaction
