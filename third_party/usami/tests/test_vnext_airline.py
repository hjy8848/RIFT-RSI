"""Airline-specific inventory, timetable, fare and passenger process boundaries."""
import json
import pytest
from usami_airline import adapter
from usami_airline.rules import itinerary_code,net_paid,occupancy
from usami_airline.tasks import certified_tasks
from usami_dialogue.engine import DialogueSession,ExecutionTrace
from usami_dialogue.models import CustomerIntent
from usami_dialogue.oracle import evaluate
from usami_dialogue.replay import replay
from usami_dialogue.dependencies import certify_dependencies,verify_dependency
from usami_dialogue.tasks import verify_certificate


PROFILES={'refund':'profile_u0_booking1','rebook':'profile_u0_booking0','baggage':'profile_u0_booking3'}


def op(s,name,**params):
    b=s.state.bookings[s.private.facts['booking_id']]
    return adapter.Operation(name=name,scope_id=b.carrier_id,resource_id=b.id,params=params)


def declared(s,*operations,**kwargs):
    return s.customer(CustomerIntent(message='My actual travel request.',requested_operations=[o.model_dump(mode='json') for o in operations],**kwargs))


def verified(world,pid,service,route=None):
    s=DialogueSession(adapter,world,pid);b=world.bookings[s.private.facts['booking_id']]
    verification=op(s,'verify_passenger',document_token=s.private.facts['document_token'])
    declared(s,verification,disclosures=s.private.facts,preferences={'service':service,'itinerary':route or b.flights})
    assert s.act(verification)['success']
    return s


def refundable_quote():
    w=adapter.generate_world(0);s=verified(w,PROFILES['refund'],'refund')
    quote=op(s,'quote_refund');declared(s,quote);result=s.act(quote);assert result['success']
    return s,result['data']['quote']


@pytest.mark.parametrize('family',['refund','rebook','baggage'])
def test_native_certification_three_distinct_business_families(family):
    w=adapter.generate_world(0);initial=w.state_hash
    task,result,rows=certified_tasks(w,PROFILES[family],family=family,mixed=family!='rebook',max_nodes=256,max_paths=1)
    assert rows and result['nodes']<=256
    row=rows[0];assert row['certificate'].evaluation['success']
    assert verify_certificate(adapter,w,json.loads(row['certificate'].model_dump_json()),ExecutionTrace.model_validate_json(row['trace'].model_dump_json()))
    final=adapter.World.model_validate(row['trace'].final_world);b=final.bookings[task.goals[0].entity]
    assert net_paid(final,b)==b.fare_value_cents+b.retained_fees_cents+b.baggage_paid_cents
    assert not adapter.invariants(final) and w.state_hash==initial


def test_passenger_identity_and_actual_disclosure_preconditions():
    w=adapter.generate_world(0);s=DialogueSession(adapter,w,PROFILES['refund'])
    quote=op(s,'quote_refund')
    assert adapter.decide(s.state,s.private.identity,quote).error_code=='REQUIRED_INFORMATION_MISSING'
    declared(s,quote,disclosures={'booking_id':s.private.facts['booking_id']},preferences={'service':'refund'})
    assert adapter.decide(s.state,s.private.identity,quote).error_code=='PASSENGER_VERIFICATION_REQUIRED'
    view=adapter.observation(w,s.private.identity)
    assert 'document_token' not in json.dumps(view) and 'simulated_identity_valid' not in json.dumps(view)
    b=w.bookings[s.private.facts['booking_id']];other=next(u for u in w.users if u!=b.passenger_id)
    read=op(s,'get_booking')
    assert adapter.decide(s.state,other,read).error_code=='BOOKING_ACCESS_DENIED'


def test_no_consent_and_revised_preference_cannot_commit_old_quote():
    s,q=refundable_quote();commit=op(s,'commit_quote',quote_id=q['id'])
    assert adapter.decide(s.state,s.private.identity,commit).error_code=='EXPLICIT_CONSENT_REQUIRED'
    s.reply('Actual refund offer.',commit.model_dump(mode='json'))
    declared(s,preferences={'service':'baggage'})
    assert adapter.decide(s.state,s.private.identity,commit).error_code=='STALE_CUSTOMER_CHOICE'
    assert not s.state.ledger and replay(adapter,s.finish())['verified']


def test_premature_sandbox_ticket_refund_is_a_true_successful_violation():
    s,q=refundable_quote();commit=op(s,'commit_quote',quote_id=q['id'])
    assert s.act(commit,'sandbox')['success']
    trace=s.finish();report=evaluate(adapter,trace)
    assert report['metrics']['successful_business_violation']==1
    assert report['operations'][-1]['reference_error_code']=='EXPLICIT_CONSENT_REQUIRED'
    assert report['operations'][-1]['authorized'] is False
    assert replay(adapter,trace)['verified']


def test_connection_minimum_is_a_real_timetable_boundary():
    w=adapter.generate_world(0);found=False
    for p in w.profiles.values():
        b=w.bookings[p.facts['booking_id']]
        for route in p.preference_space['itinerary']:
            if len(route)!=2:continue
            a,z=[w.flights[f] for f in route]
            from datetime import timedelta
            if (a.status!='cancelled' and z.status!='cancelled' and z.departure<a.arrival+timedelta(minutes=w.airports[a.destination].min_connection_minutes)):
                assert itinerary_code(w,b,route,'economy',check_capacity=False)=='CONNECTION_INFEASIBLE';found=True;break
        if found:break
    assert found


def test_voluntary_forbidden_refund_becomes_allowed_after_actual_airline_cancellation():
    selected=None
    for seed in range(20):
        w=adapter.generate_world(seed)
        for pid,p in w.profiles.items():
            b=w.bookings[p.facts['booking_id']];rule=w.fare_rules[b.fare_rule_id]
            events=[e for e in w.external_events.values() if e.flight_id in b.flights and e.status=='cancelled']
            if w.users[p.identity].simulated_identity_valid and not rule.refundable and not any(w.flights[f].status=='cancelled' for f in b.flights) and events:selected=(w,pid,events[0]);break
        if selected:break
    assert selected
    w,pid,event=selected;s=verified(w,pid,'refund');quote=op(s,'quote_refund')
    assert adapter.decide(s.state,s.private.identity,quote).error_code=='FARE_REFUND_FORBIDDEN'
    s.external(event.id)
    d=adapter.decide(s.state,s.private.identity,quote)
    assert d.allowed and d.data['quote']['fee_cents']==0
    b=s.state.bookings[s.private.facts['booking_id']]
    assert d.data['quote']['refund_cents']==b.fare_value_cents
    trace=s.finish();proofs=certify_dependencies(adapter,trace)
    proof=next(p for p in proofs if p.proof.kind=='state_update' and p.proof.operation['name']=='quote_refund')
    assert proof.proof.expected_control_code=='FARE_REFUND_FORBIDDEN' and proof.proof.changed_allowed
    assert verify_dependency(adapter,trace,proof)


def test_sandbox_rebooking_over_capacity_is_not_hidden_by_engine():
    selected=None
    for seed in range(10):
        w=adapter.generate_world(seed)
        for pid,p in w.profiles.items():
            b=w.bookings[p.facts['booking_id']]
            if not w.users[p.identity].simulated_identity_valid:continue
            for route in p.preference_space['itinerary']:
                if route==b.flights or any(w.flights[f].status=='cancelled' for f in route):continue
                if any(occupancy(w,f,'economy',b.id)>=w.flights[f].seats['economy'] for f in route):selected=(w,pid,route);break
            if selected:break
        if selected:break
    assert selected
    w,pid,route=selected;s=verified(w,pid,'rebook',route)
    quote=op(s,'quote_change',flights=route,cabin='economy');declared(s,quote)
    assert adapter.decide(s.state,s.private.identity,quote).error_code=='ITINERARY_UNAVAILABLE'
    response=s.act(quote,'sandbox');assert response['success']
    commit=op(s,'commit_quote',quote_id=response['data']['quote']['id'])
    assert s.act(commit,'sandbox')['success']
    assert any(e['code']=='SEAT_OVERBOOKED' for e in adapter.invariants(s.state))
    trace=s.finish();report=evaluate(adapter,trace)
    assert report['metrics']['successful_business_violation']==2
    assert replay(adapter,trace)['verified']


def test_standalone_policy_refusal_is_certified_without_attempting_forbidden_quote():
    chosen=None
    for seed in range(10):
        world=adapter.generate_world(seed)
        for pid,p in world.profiles.items():
            b=world.bookings[p.facts['booking_id']]
            if world.users[p.identity].simulated_identity_valid and not world.fare_rules[b.fare_rule_id].refundable and not any(world.flights[f].status=='cancelled' for f in b.flights):chosen=(world,pid,b);break
        if chosen:break
    assert chosen
    world,pid,b=chosen
    probe=adapter.Operation(name='quote_refund',scope_id=b.carrier_id,resource_id=b.id,params={})
    task,result,rows=certified_tasks(world,pid,family='refuse_refund',refusal_operation=probe.model_dump(mode='json'),refusal_code='FARE_REFUND_FORBIDDEN',max_nodes=128,max_paths=1)
    assert rows and not task.goals and rows[0]['certificate'].evaluation['refusal_subgoals']==[True]
    report=rows[0]['certificate'].oracle
    assert report['metrics']['correct_refusal']==1 and report['metrics']['rejected_invalid_attempt']==0
    assert [r['operation'] for r in report['operations']]==['verify_passenger']
    assert not rows[0]['trace'].final_world['ledger'] and not rows[0]['trace'].final_world['quotes']


def test_new_quote_after_external_update_uses_current_terms_without_customer_revision():
    chosen=None
    for seed in range(10):
        w=adapter.generate_world(seed)
        for pid,p in w.profiles.items():
            b=w.bookings[p.facts['booking_id']];rule=w.fare_rules[b.fare_rule_id]
            events=[e for e in w.external_events.values() if e.flight_id in b.flights and e.status=='cancelled']
            if w.users[p.identity].simulated_identity_valid and rule.refundable and rule.refund_fee_cents and not any(w.flights[f].status=='cancelled' for f in b.flights) and events:chosen=(w,pid,events[0]);break
        if chosen:break
    assert chosen
    w,pid,event=chosen;s=verified(w,pid,'refund');quote=op(s,'quote_refund');declared(s,quote)
    old=s.act(quote)['data']['quote'];revision=s.state.interaction.revision
    s.external(event.id);declared(s,quote);new=s.act(quote)['data']['quote']
    assert new['id']!=old['id'] and new['fee_cents']==0 and new['refund_cents']>old['refund_cents']
    assert s.state.interaction.revision==revision
    assert evaluate(adapter,s.finish())['status']=='pass'


def test_service_can_query_own_payment_capacity_without_hidden_identity_facts():
    w=adapter.generate_world(0);s=DialogueSession(adapter,w,PROFILES['rebook'])
    query=op(s,'get_booking');d=adapter.decide(s.state,s.private.identity,query)
    assert d.allowed and d.data['remaining_payment_capacity_cents']==w.users[s.private.identity].payment_limit_cents
    assert 'document_token' not in json.dumps(d.data) and 'simulated_identity_valid' not in json.dumps(d.data)
    actual=s.act(query);assert actual['success'] and actual['data']==d.data
    assert evaluate(adapter,s.finish())['status']=='pass'


def test_refusal_scoring_checks_current_reference_reason_and_keeps_coverage_labels_separate():
    from usami_dialogue.tasks import score
    selected=None
    for seed in range(5):
        w=adapter.generate_world(seed)
        for pid,p in w.profiles.items():
            b=w.bookings[p.facts['booking_id']]
            if w.users[p.identity].simulated_identity_valid and not w.fare_rules[b.fare_rule_id].refundable and not any(w.flights[f].status=='cancelled' for f in b.flights):selected=(w,pid,b);break
        if selected:break
    assert selected
    w,pid,b=selected;probe=adapter.Operation(name='quote_refund',scope_id=b.carrier_id,resource_id=b.id,params={})
    task,_,rows=certified_tasks(w,pid,family='refuse_refund',refusal_operation=probe.model_dump(mode='json'),refusal_code='FARE_REFUND_FORBIDDEN',max_nodes=64,max_paths=1)
    assert rows
    s=verified(w,pid,'baggage')
    # An actual subsequent Customer request conflicts with their current choice.
    # Refusing that request for the correct current reason is not a Service error.
    declared(s,probe)
    d=adapter.decide(s.state,s.private.identity,probe);assert d.error_code=='PREFERENCE_RESOLUTION_REQUIRED'
    s.reply('Please resolve which service you want.',refused_operation=probe.model_dump(mode='json'),refusal_code=d.error_code)
    result=score(adapter,w,task,s.finish())
    assert result['protocol']=='dialogue-task-score-v2' and result['success']
    assert result['intended_refusal_boundaries']==['FARE_REFUND_FORBIDDEN']
    assert result['observed_refusal_reasons']==[['PREFERENCE_RESOLUTION_REQUIRED']]
