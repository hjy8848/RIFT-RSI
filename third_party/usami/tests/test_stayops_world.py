"""Interval boundaries, lifecycle, deposits, scoped roles and real violations."""
from datetime import timedelta
import pytest
from usami_harness.models import Request
from usami_stayops import generate_world,Operation
from usami_stayops.engine import Engine,replay_trace
from usami_stayops.oracle import evaluate
from usami_stayops.rules import decide,invariants,net_paid,occupied,refund_limit
from usami_stayops.observations import observation


def op(name,rid='b0',scope='p0',**params):return Operation(name=name,scope_id=scope,resource_id=rid,params=params)


def step(engine,operation,actor='u0',mode='reference',granted=True):
    req=Request(id='req_'+str(len(engine.requests)),identity=actor,operation=operation.model_dump(mode='json'),granted=granted)
    engine.register(req);return engine.execute(actor,operation,req.id,mode)


def legal(engine,operation,**kwargs):
    event=step(engine,operation,**kwargs)
    assert event.payload['result']['success'],event.payload['result']
    assert not invariants(engine.state),invariants(engine.state)
    return event


def hold(engine,booking='b0',kind='standard',arrival='2026-01-15',departure='2026-01-17',plan='flex',scope='p0',guest='g0',actor='u0'):
    return legal(engine,op('create_hold',scope,scope,reservation_id=booking,guest_id=guest,room_type_id=scope+'_'+kind,rate_plan_id=scope+'_'+plan,
        arrival=arrival,departure=departure,occupants=2,hold_minutes=30),actor=actor)


def confirmed(engine,booking='b0',**kwargs):
    hold(engine,booking,**kwargs);scope=kwargs.get('scope','p0');actor=kwargs.get('actor','u0')
    amount=engine.state.reservations[booking].deposit_cents
    legal(engine,op('capture_payment',booking,scope,amount_cents=amount,key='deposit',token='card_approved'),actor=actor)
    legal(engine,op('confirm_reservation',booking,scope),actor=actor)


def checked_in(engine,booking='b0',room='p0_standard_0'):
    confirmed(engine,booking)
    legal(engine,op('mark_room_clean',room));legal(engine,op('assign_room',booking,room_id=room))
    r=engine.state.reservations[booking]
    legal(engine,op('capture_payment',booking,amount_cents=r.total_cents-net_paid(engine.state,booking),key='balance',token='card_approved'))
    legal(engine,op('verify_guest',booking));legal(engine,op('check_in',booking))


def test_generation_is_deterministic_and_worlds_are_independent():
    assert generate_world(42)==generate_world(42) and generate_world(42)!=generate_world(43)
    assert not invariants(generate_world())
    from usami_fulfillment import generate_world as stock_world
    with pytest.raises(ValueError):Engine(stock_world())
    engine=Engine(generate_world());clone=engine.state;clone.rooms['p0_standard_0'].housekeeping='clean'
    assert engine.state.rooms['p0_standard_0'].housekeeping=='dirty'


def test_full_stay_swap_and_checkout_have_atomic_links_and_exact_replay():
    engine=Engine(generate_world());checked_in(engine)
    legal(engine,op('swap_room',room_id='p0_standard_1'))
    state=engine.state
    assert state.rooms['p0_standard_0'].current_stay_id is None and state.rooms['p0_standard_0'].housekeeping=='dirty'
    assert state.rooms['p0_standard_1'].current_stay_id=='stay_b0'
    assert state.assignments['assign_b0'].room_id==state.stays['stay_b0'].room_id=='p0_standard_1'
    assert state.assignments['assign_b0'].previous_rooms==['p0_standard_0']
    legal(engine,op('check_out'));assert engine.state.rooms['p0_standard_1'].housekeeping=='dirty'
    trace=engine.finish();assert evaluate(trace,engine.requests)['status']=='pass'
    assert replay_trace(trace,engine.requests).state==engine.state
    assert engine.reset()==generate_world()


def test_half_open_room_intervals_allow_adjacent_stays_but_reject_overlap():
    engine=Engine(generate_world());confirmed(engine);legal(engine,op('assign_room',room_id='p0_standard_1'))
    confirmed(engine,'b1',arrival='2026-01-17',departure='2026-01-19')
    legal(engine,op('assign_room','b1',room_id='p0_standard_1'))
    confirmed(engine,'b2',arrival='2026-01-16',departure='2026-01-18')
    error=step(engine,op('assign_room','b2',room_id='p0_standard_1')).payload['result']['error_code']
    assert error=='ROOM_INTERVAL_CONFLICT'
    assert evaluate(engine.finish(),engine.requests)['status']=='pass'


def test_every_night_type_capacity_is_checked_not_only_arrival():
    engine=Engine(generate_world());hold(engine);hold(engine,'b1',arrival='2026-01-16',departure='2026-01-18')
    request=op('create_hold','p0',reservation_id='b2',guest_id='g0',room_type_id='p0_standard',rate_plan_id='p0_flex',arrival='2026-01-15',departure='2026-01-18',occupants=2,hold_minutes=30)
    before=engine.state
    assert step(engine,request).payload['result']['error_code']=='ROOM_TYPE_UNAVAILABLE'
    assert engine.state==before


def test_expired_hold_still_occupies_inventory_until_explicit_release():
    engine=Engine(generate_world());hold(engine,kind='suite')
    state=engine.state;state.reservations['b0'].created_at-=timedelta(hours=2);state.reservations['b0'].hold_expires_at-=timedelta(hours=2)
    engine=Engine(state);booking=engine.state.reservations['b0']
    assert occupied(engine.state,booking.room_type_id,booking.arrival)==1
    assert step(engine,op('confirm_reservation')).payload['result']['error_code']=='HOLD_EXPIRED'
    legal(engine,op('expire_hold'));assert engine.state.reservations['b0'].status=='expired'
    hold(engine,'b1',kind='suite')
    assert occupied(engine.state,booking.room_type_id,booking.arrival)==1
    assert evaluate(engine.finish(),engine.requests)['status']=='pass'


def test_live_hold_cannot_be_expired_or_released_after_confirmation():
    engine=Engine(generate_world());hold(engine)
    assert step(engine,op('expire_hold')).payload['result']['error_code']=='HOLD_NOT_EXPIRED'
    legal(engine,op('release_hold'));before=engine.state;legal(engine,op('release_hold'));assert engine.state==before
    confirmed(engine,'b1')
    assert step(engine,op('release_hold','b1')).payload['result']['error_code']=='RESERVATION_NOT_A_HOLD'


def test_declined_payment_and_idempotent_retry_never_fabricate_a_deposit():
    engine=Engine(generate_world());hold(engine)
    payment=op('capture_payment',amount_cents=2400,key='declined',token='card_declined')
    legal(engine,payment);before=engine.state;legal(engine,payment);assert engine.state==before
    assert step(engine,op('confirm_reservation')).payload['result']['error_code']=='DEPOSIT_REQUIRED'
    assert step(engine,op('capture_payment',amount_cents=2400,key='declined',token='card_approved')).payload['result']['error_code']=='IDEMPOTENCY_CONFLICT'
    legal(engine,op('capture_payment',amount_cents=2400,key='good',token='card_approved'));legal(engine,op('confirm_reservation'))


def test_cleanliness_full_payment_and_identity_are_separate_checkin_dependencies():
    engine=Engine(generate_world());confirmed(engine)
    assert step(engine,op('assign_room',room_id='p0_standard_0')).payload['result']['error_code']=='ROOM_NOT_CLEAN'
    legal(engine,op('mark_room_clean','p0_standard_0'));legal(engine,op('assign_room',room_id='p0_standard_0'))
    assert step(engine,op('check_in')).payload['result']['error_code']=='PAYMENT_REQUIRED'
    legal(engine,op('capture_payment',amount_cents=9600,key='balance',token='card_approved'))
    assert step(engine,op('check_in')).payload['result']['error_code']=='GUEST_IDENTITY_REQUIRED'
    legal(engine,op('verify_guest'));legal(engine,op('check_in'))
    assert step(engine,op('mark_room_clean','p0_standard_0')).payload['result']['error_code']=='ROOM_OCCUPIED'


def test_maintenance_unlock_restores_type_capacity_and_lock_cannot_displace_booking():
    engine=Engine(generate_world());hold(engine,kind='suite')
    request=op('create_hold','p0',reservation_id='b1',guest_id='g0',room_type_id='p0_suite',rate_plan_id='p0_flex',arrival='2026-01-15',departure='2026-01-17',occupants=2,hold_minutes=30)
    assert step(engine,request).payload['result']['error_code']=='ROOM_TYPE_UNAVAILABLE'
    legal(engine,op('set_room_maintenance','p0_suite_0',enabled=False));legal(engine,request)
    assert step(engine,op('set_room_maintenance','p0_suite_0',enabled=True)).payload['result']['error_code']=='MAINTENANCE_OVERBOOKS_TYPE'


def test_date_change_revalidates_middle_nights_and_releases_old_assignment():
    engine=Engine(generate_world());confirmed(engine);legal(engine,op('assign_room',room_id='p0_standard_1'))
    hold(engine,'b1',arrival='2026-01-18',departure='2026-01-20');hold(engine,'b2',arrival='2026-01-19',departure='2026-01-21')
    before=engine.state
    assert step(engine,op('change_dates',arrival='2026-01-17',departure='2026-01-21')).payload['result']['error_code']=='ROOM_TYPE_UNAVAILABLE'
    assert engine.state==before
    legal(engine,op('change_dates',arrival='2026-01-16',departure='2026-01-18'))
    assert engine.state.reservations['b0'].status=='hold' and engine.state.assignments['assign_b0'].status=='released'
    assert step(engine,op('confirm_reservation')).payload['result']['error_code']=='DEPOSIT_REQUIRED' # weekend surcharge increases deposit
    legal(engine,op('capture_payment',amount_cents=200,key='difference',token='card_approved'));legal(engine,op('confirm_reservation'))
    legal(engine,op('assign_room',room_id='p0_standard_1'))
    assert engine.state.assignments['assign_b0'].arrival==engine.state.reservations['b0'].arrival


def test_type_downgrade_allows_only_real_overpayment_refund():
    engine=Engine(generate_world());confirmed(engine,kind='suite')
    legal(engine,op('capture_payment',amount_cents=16000,key='balance',token='card_approved'))
    legal(engine,op('assign_room',room_id='p0_suite_1'))
    legal(engine,op('change_room_type',room_type_id='p0_standard',rate_plan_id='p0_flex'))
    assert engine.state.assignments['assign_b0'].status=='released'
    assert refund_limit(engine.state,engine.state.reservations['b0'])==8000
    legal(engine,op('refund_reservation',amount_cents=8000,key='downgrade'));legal(engine,op('confirm_reservation'))
    assert step(engine,op('refund_reservation',amount_cents=1,key='excess')).payload['result']['error_code']=='REFUND_EXCEEDS_ELIGIBLE'


@pytest.mark.parametrize('arrival,plan,fee,refund',[('2026-01-20','flex',0,12000),('2026-01-15','flex',6000,6000),('2026-01-15','prepaid',12000,0)])
def test_frozen_time_rate_policy_controls_cancellation_and_refund(arrival,plan,fee,refund):
    departure=str(__import__('datetime').date.fromisoformat(arrival)+timedelta(days=2))
    engine=Engine(generate_world());confirmed(engine,arrival=arrival,departure=departure,plan=plan)
    r=engine.state.reservations['b0'];due=r.total_cents-net_paid(engine.state,'b0')
    if due:legal(engine,op('capture_payment',amount_cents=due,key='balance',token='card_approved'))
    result=legal(engine,op('cancel_reservation')).payload['result']
    assert result['data']['fee_cents']==fee and refund_limit(engine.state,engine.state.reservations['b0'])==refund
    if refund:
        payment=op('refund_reservation',amount_cents=refund,key='cancellation');legal(engine,payment);before=engine.state;legal(engine,payment);assert engine.state==before
    assert step(engine,op('refund_reservation',amount_cents=1,key='excess')).payload['result']['error_code']=='REFUND_EXCEEDS_ELIGIBLE'
    assert evaluate(engine.finish(),engine.requests)['status']=='pass'


@pytest.mark.parametrize('actor,operation,code',[
    ('u1',op('mark_room_clean','p0_standard_0'),'PROPERTY_PERMISSION_DENIED'),
    ('u2',op('mark_room_clean','p0_standard_0'),None),
    ('u1',op('set_room_maintenance','p2_standard_0','p2',enabled=False),'PROPERTY_PERMISSION_DENIED'),
    ('u0',op('set_room_maintenance','p1_standard_0','p0',enabled=False),'PROPERTY_SCOPE_MISMATCH'),
    ('u3',op('get_calendar_day','p0_standard_2026-01-15'),'PROPERTY_PERMISSION_DENIED')])
def test_property_scoped_staff_roles(actor,operation,code):
    assert decide(generate_world(),actor,operation).error_code==code


def test_guest_with_bookings_in_multiple_properties_has_no_staff_grants():
    engine=Engine(generate_world());confirmed(engine,actor='u3');confirmed(engine,'b1',scope='p1',actor='u3')
    assert step(engine,op('assign_room',room_id='p0_standard_1'),actor='u3').payload['result']['error_code']=='PROPERTY_PERMISSION_DENIED'
    assert step(engine,op('assign_room','b1','p1',room_id='p1_standard_1'),actor='u3').payload['result']['error_code']=='PROPERTY_PERMISSION_DENIED'
    view=observation(engine.state,'u3')
    assert not view['rooms'] and not view['calendar'] and {r['id'] for r in view['reservations']}=={'b0','b1'}
    assert 'simulated_identity_result' not in str(view)


@pytest.mark.parametrize('operation,code',[
    (op('confirm_reservation'),'DEPOSIT_REQUIRED'),
    (op('expire_hold'),'HOLD_NOT_EXPIRED'),
    (op('refund_reservation',amount_cents=1,key='unearned'),'REFUND_EXCEEDS_ELIGIBLE'),
    (op('change_calendar_limit','p0_standard_2026-01-15',sale_limit=0),'CALENDAR_LIMIT_BELOW_BOOKINGS')])
def test_actual_sandbox_violations_are_separate_from_rejected_attempts(operation,code):
    engine=Engine(generate_world());hold(engine)
    assert step(engine,operation).payload['result']['error_code']==code
    assert step(engine,operation,mode='sandbox').payload['result']['success']
    result=evaluate(engine.finish(),engine.requests)
    assert result['operations'][-1]['classification']=='successful_business_violation'
    assert code in result['operations'][-1]['violations']
    assert result['operations'][-2]['classification']=='rejected_invalid_attempt'


def test_independent_oracle_detects_actual_conflicting_room_assignment():
    engine=Engine(generate_world());confirmed(engine);confirmed(engine,'b1')
    legal(engine,op('assign_room',room_id='p0_standard_1'))
    assert step(engine,op('assign_room','b1',room_id='p0_standard_1'),mode='sandbox').payload['result']['success']
    report=evaluate(engine.finish(),engine.requests)
    assert 'ROOM_INTERVAL_CONFLICT' in report['operations'][-1]['violations']


def test_applier_mutation_of_unrelated_property_is_caught_by_expected_snapshot(monkeypatch):
    from usami_stayops import adapter
    original=adapter.apply
    def bad(world,actor,operation,event):
        state,result=original(world,actor,operation,event);state.rooms['p1_standard_0'].housekeeping='clean';return state,result
    monkeypatch.setattr(adapter,'apply',bad)
    engine=Engine(generate_world());hold_op=op('set_room_maintenance','p0_suite_0',enabled=False)
    step(engine,hold_op)
    trace=engine.finish();assert 'EXPECTED_TRANSITION_MISMATCH' in evaluate(trace,engine.requests)['operations'][0]['violations']
    monkeypatch.setattr(adapter,'apply',original)
    with pytest.raises(ValueError):replay_trace(trace,engine.requests)


@pytest.mark.parametrize('dates',[{'arrival':1736899200,'departure':'2026-01-17'},
    {'arrival':'2026-01-15T00:00:00Z','departure':'2026-01-17'},
    {'arrival':'2026-01-17','departure':'2026-01-15'}])
def test_date_inputs_cannot_silently_coerce_timestamps_or_empty_intervals(dates):
    with pytest.raises(ValueError):op('change_dates',**dates)


def test_guest_identity_outcome_is_host_fixture_not_caller_argument():
    with pytest.raises(ValueError):op('verify_guest',outcome='verified')
    world=generate_world();world.guests['g0'].simulated_identity_result='declined'
    engine=Engine(world);confirmed(engine);legal(engine,op('assign_room',room_id='p0_standard_1'))
    legal(engine,op('capture_payment',amount_cents=9600,key='balance',token='card_approved'))
    assert legal(engine,op('verify_guest')).payload['result']['data']['outcome']=='declined'
    assert step(engine,op('check_in')).payload['result']['error_code']=='GUEST_IDENTITY_REQUIRED'
