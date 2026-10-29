"""Independent lifecycle, conservation and adversarial boundary tests."""
import pytest
from usami_harness.models import Request
from usami_fulfillment import generate_world, Operation
from usami_fulfillment.engine import Engine, replay_trace
from usami_fulfillment.oracle import evaluate
from usami_fulfillment.rules import decide, invariants, physical_volume
from usami_fulfillment.observations import observation


def op(name, resource='order0', scope='w0', **params):
    return Operation(name=name,scope_id=scope,resource_id=resource,params=params)


def step(engine, operation, actor='u0', mode='reference', granted=True):
    request=Request(id=f'req_{len(engine.requests)}',identity=actor,operation=operation.model_dump(mode='json'),granted=granted)
    engine.register(request)
    return engine.execute(actor,operation,request.id,mode)


def legal(engine, operation, **kwargs):
    event=step(engine,operation,**kwargs)
    assert event.payload['result']['success'],event.payload['result']
    assert not invariants(engine.state)
    return event


def ready(engine, order='order0', scope='w0', amount=5300):
    legal(engine,op('capture_payment',order,scope,amount_cents=amount,key='pay',token='card_approved'))
    legal(engine,op('confirm_order',order,scope))


def allocate(engine, order='order0', scope='w0', reservation='r0', lot='lot0', quantity=4):
    legal(engine,op('reserve_inventory',order,scope,reservation_id=reservation,line_id='line0',lot_id=lot,quantity=quantity))


def dispatch(engine, quantity=4):
    ready(engine);allocate(engine,quantity=quantity)
    legal(engine,op('create_shipment',shipment_id='s0',reservation_ids=['r0'],key='ship'))
    legal(engine,op('pick_reservation','r0'));legal(engine,op('pack_reservation','r0'))
    legal(engine,op('dispatch_shipment','s0'))


def test_deterministic_generation_and_domain_isolation():
    assert generate_world(42)==generate_world(42)
    assert generate_world(42)!=generate_world(43)
    assert not invariants(generate_world())
    from usami_cloudops import generate_world as cloud
    with pytest.raises(ValueError):Engine(cloud())
    engine=Engine(generate_world());state=engine.state;state.lots['lot0'].on_hand+=100
    assert engine.state.lots['lot0'].on_hand==8


def test_full_lifecycle_return_partial_refund_and_exact_replay():
    engine=Engine(generate_world());dispatch(engine)
    assert engine.state.lots['lot0'].on_hand==4 and engine.state.lots['lot0'].picked==0
    assert engine.state.orders['order0'].status=='shipped'
    legal(engine,op('confirm_delivery','s0'))
    legal(engine,op('request_return','s0',return_id='ret0',reservation_id='r0',quantity=2,reason='customer_remorse'))
    legal(engine,op('receive_return','ret0'));legal(engine,op('inspect_return','ret0'))
    volume=physical_volume(engine.state,'w0')
    legal(engine,op('restock_return','ret0'))
    assert physical_volume(engine.state,'w0')==volume
    legal(engine,op('refund_order',amount_cents=2400,key='refund'))
    assert engine.state.lots['lot0'].on_hand==6 and len(engine.state.payments)==2
    trace=engine.finish();assert evaluate(trace,engine.requests)['status']=='pass'
    assert replay_trace(trace,engine.requests).state==engine.state
    assert engine.reset()==generate_world()


@pytest.mark.parametrize('actor,operation,code',[
    ('u1',op('receive_inventory','inbound0',lot_id='incoming',key='receipt'),'WAREHOUSE_PERMISSION_DENIED'),
    ('u2',op('receive_inventory','inbound0',lot_id='incoming',key='receipt'),'WAREHOUSE_PERMISSION_DENIED'),
    ('u2',op('receive_inventory','inbound1','w1',lot_id='incoming',key='receipt'),None),
    ('u3',op('capture_payment',amount_cents=5300,key='buyer',token='card_approved'),None),
    ('u3',op('get_inventory','w0'),'WAREHOUSE_PERMISSION_DENIED'),
    ('u4',op('capture_payment',amount_cents=5300,key='finance',token='card_approved'),None),
    ('u1',op('get_payment_ledger'),'FINANCIAL_PERMISSION_DENIED'),
    ('u0',op('capture_payment','order2','w2',amount_cents=3700,key='foreign',token='card_approved'),'WAREHOUSE_PERMISSION_DENIED'),
    ('u0',op('capture_payment','order1','w0',amount_cents=5300,key='wrong-scope',token='card_approved'),'WAREHOUSE_SCOPE_MISMATCH')])
def test_scoped_roles_and_customer_ownership(actor,operation,code):
    decision=decide(generate_world(),actor,operation)
    assert decision.error_code==code and decision.allowed==(code is None)


def test_stock_cannot_be_double_allocated_and_all_buckets_are_distinct():
    engine=Engine(generate_world());ready(engine);allocate(engine)
    lot=engine.state.lots['lot0'];assert (lot.on_hand,lot.reserved,lot.picked)==(8,4,0)
    legal(engine,op('create_order','w0',order_id='second',customer_id='u3',lines=[{'id':'line0','sku_id':'sku0','quantity':4}]))
    ready(engine,'second');allocate(engine,'second',reservation='r1')
    assert engine.state.lots['lot0'].reserved==8
    legal(engine,op('create_order','w0',order_id='third',customer_id='u3',lines=[{'id':'line0','sku_id':'sku0','quantity':4}]))
    ready(engine,'third')
    denied=step(engine,op('reserve_inventory','third',reservation_id='r2',line_id='line0',lot_id='lot0',quantity=4))
    assert denied.payload['result']['error_code']=='STOCK_INSUFFICIENT'
    legal(engine,op('pick_reservation','r0'))
    lot=engine.state.lots['lot0'];assert (lot.on_hand,lot.reserved,lot.picked)==(4,4,4)


def test_payment_failure_and_immutable_idempotency():
    engine=Engine(generate_world())
    bad=op('capture_payment',amount_cents=5300,key='declined',token='card_declined')
    legal(engine,bad);before=engine.state;legal(engine,bad);assert engine.state==before
    assert step(engine,op('confirm_order')).payload['result']['error_code']=='PAYMENT_REQUIRED'
    assert step(engine,op('capture_payment',amount_cents=5300,key='declined',token='card_approved')).payload['result']['error_code']=='IDEMPOTENCY_CONFLICT'
    ready(engine)
    assert len(engine.state.payments)==2 and evaluate(engine.finish(),engine.requests)['status']=='pass'


def test_receipt_and_shipping_retries_do_not_repeat_physical_effects():
    engine=Engine(generate_world())
    receive=op('receive_inventory','inbound0',lot_id='incoming',key='rkey')
    legal(engine,receive);before=engine.state;legal(engine,receive);assert engine.state==before
    assert step(engine,op('receive_inventory','inbound0',lot_id='incoming',key='new-key')).payload['result']['error_code']=='IDEMPOTENCY_CONFLICT'
    dispatch(engine);before=engine.state;legal(engine,op('dispatch_shipment','s0'));assert engine.state==before
    assert len(engine.state.shipments)==1


def test_cancel_releases_unpicked_stock_and_voids_labels():
    engine=Engine(generate_world());ready(engine);allocate(engine)
    legal(engine,op('create_shipment',shipment_id='s0',reservation_ids=['r0'],key='ship'))
    legal(engine,op('cancel_order'))
    assert engine.state.reservations['r0'].status=='released' and engine.state.lots['lot0'].reserved==0
    assert engine.state.shipments['s0'].status=='canceled'
    legal(engine,op('refund_order',amount_cents=5300,key='full-cancel'))
    assert evaluate(engine.finish(),engine.requests)['status']=='pass'


def test_partial_cancellation_shipment_return_and_refund():
    engine=Engine(generate_world());ready(engine);allocate(engine)
    legal(engine,op('cancel_order_line',line_id='line0',quantity=2))
    assert engine.state.reservations['r0'].quantity==2 and engine.state.reservations['r0'].released_quantity==2
    legal(engine,op('pick_reservation','r0'));legal(engine,op('pack_reservation','r0'))
    legal(engine,op('create_shipment',shipment_id='s0',reservation_ids=['r0'],key='ship'))
    legal(engine,op('dispatch_shipment','s0'))
    assert step(engine,op('cancel_order')).payload['result']['error_code']=='ORDER_ALREADY_SHIPPED'
    legal(engine,op('refund_order',amount_cents=2400,key='partial-cancel'))
    legal(engine,op('confirm_delivery','s0'))
    legal(engine,op('request_return','s0',return_id='ret0',reservation_id='r0',quantity=1,reason='customer_remorse'))
    assert step(engine,op('request_return','s0',return_id='excess',reservation_id='r0',quantity=2,reason='customer_remorse')).payload['result']['error_code']=='RETURN_EXCEEDS_DELIVERY'
    legal(engine,op('receive_return','ret0'));legal(engine,op('inspect_return','ret0'))
    legal(engine,op('refund_order',amount_cents=1200,key='partial-return'))
    assert step(engine,op('refund_order',amount_cents=1,key='excess-refund')).payload['result']['error_code']=='REFUND_EXCEEDS_ELIGIBLE'
    assert evaluate(engine.finish(),engine.requests)['status']=='pass'


def test_packed_cancellation_requires_real_reversal_chain():
    engine=Engine(generate_world());ready(engine);allocate(engine)
    legal(engine,op('pick_reservation','r0'));legal(engine,op('pack_reservation','r0'))
    legal(engine,op('create_shipment',shipment_id='s0',reservation_ids=['r0'],key='ship'))
    assert step(engine,op('cancel_order')).payload['result']['error_code']=='PICKED_CANCEL_REQUIRES_REVERSAL'
    assert step(engine,op('unpack_reservation','r0')).payload['result']['error_code']=='SHIPMENT_STILL_ASSIGNED'
    legal(engine,op('cancel_shipment','s0'));legal(engine,op('unpack_reservation','r0'));legal(engine,op('unpick_reservation','r0'));legal(engine,op('cancel_order'))
    assert engine.state.lots['lot0'].on_hand==8 and engine.state.lots['lot0'].reserved==0


def test_transfer_and_destination_admission_preserve_inventory():
    engine=Engine(generate_world())
    transfer=op('dispatch_transfer','lot0',transfer_id='t0',target_warehouse_id='w1',target_lot_id='arrival',quantity=4,key='move')
    legal(engine,transfer);assert engine.state.lots['lot0'].on_hand==4 and 'arrival' not in engine.state.lots
    legal(engine,op('receive_transfer','t0','w1'));before=engine.state
    legal(engine,op('receive_transfer','t0','w1'));assert engine.state==before
    ready(engine,'order1','w1');allocate(engine,'order1','w1','regional','arrival')
    assert engine.state.lots['arrival'].reserved==4
    assert evaluate(engine.finish(),engine.requests)['status']=='pass'


def test_receiving_capacity_and_quality_hold_are_independent_prerequisites():
    world=generate_world();world.warehouses['w0'].receiving_enabled=False;world.warehouses['w0'].capacity_volume=28
    engine=Engine(world)
    receive=op('receive_inventory','inbound2',lot_id='quality',key='qa')
    assert step(engine,receive).payload['result']['error_code']=='RECEIVING_DISABLED'
    legal(engine,op('set_warehouse_receiving','w0',enabled=True))
    assert step(engine,receive).payload['result']['error_code']=='CAPACITY_EXCEEDED'
    legal(engine,op('change_warehouse_capacity','w0',capacity_volume=44));legal(engine,receive)
    legal(engine,op('create_order','w0',order_id='glass',customer_id='u3',lines=[{'id':'line0','sku_id':'sku1','quantity':4}]))
    ready(engine,'glass',amount=9700)
    allocate_op=op('reserve_inventory','glass',reservation_id='glass-r',line_id='line0',lot_id='quality',quantity=4)
    assert step(engine,allocate_op).payload['result']['error_code']=='QUALITY_HOLD'
    legal(engine,op('inspect_lot','quality'));legal(engine,allocate_op)


@pytest.mark.parametrize('operation,code',[
    (op('confirm_order'),'PAYMENT_REQUIRED'),
    (op('change_warehouse_capacity','w0',capacity_volume=0),'CAPACITY_BELOW_USAGE'),
    (op('refund_order',amount_cents=1,key='unpaid'),'REFUND_EXCEEDS_ELIGIBLE'),
    (op('receive_inventory','inbound1','w0',lot_id='stolen',key='cross'),'WAREHOUSE_SCOPE_MISMATCH')])
def test_actual_sandbox_business_violations_are_independently_detected(operation,code):
    engine=Engine(generate_world());event=step(engine,operation,mode='sandbox');assert event.payload['result']['success']
    row=evaluate(engine.finish(),engine.requests)['operations'][0]
    assert row['classification']=='successful_business_violation' and code in row['violations']


def test_ship_without_pack_and_cancel_after_ship_are_real_violations():
    engine=Engine(generate_world());ready(engine);allocate(engine)
    legal(engine,op('create_shipment',shipment_id='s0',reservation_ids=['r0'],key='ship'))
    assert step(engine,op('dispatch_shipment','s0')).payload['result']['error_code']=='SHIPMENT_NOT_PACKED'
    event=step(engine,op('dispatch_shipment','s0'),mode='sandbox');assert event.payload['result']['success']
    report=evaluate(engine.finish(),engine.requests)
    assert report['operations'][-1]['classification']=='successful_business_violation'
    assert {'PICK_EVIDENCE_MISSING','PACK_EVIDENCE_MISSING'}<=set(report['operations'][-1]['violations'])
    engine=Engine(generate_world());dispatch(engine);step(engine,op('cancel_order'),mode='sandbox')
    assert 'SHIPPED_ORDER_CANCELED' in evaluate(engine.finish(),engine.requests)['operations'][-1]['violations']


def test_customer_authority_is_not_inferred_from_tool_arguments():
    engine=Engine(generate_world())
    operation=op('capture_payment',amount_cents=5300,key='unauthorized',token='card_approved')
    event=step(engine,operation,granted=False);assert event.payload['result']['error_code']=='UNAUTHORIZED_OPERATION'
    engine=Engine(generate_world());step(engine,operation,mode='sandbox',granted=False)
    row=evaluate(engine.finish(),engine.requests)['operations'][0]
    assert row['classification']=='authorization_only' and not row['operation_correct']


def test_scoped_observations_hide_other_tenants_and_inspection_ground_truth():
    owner=observation(generate_world(),'u0');buyer=observation(generate_world(),'u3')
    assert {w['id'] for w in owner['warehouses']}=={'w0','w1'}
    assert not buyer['lots'] and not buyer['inbound_receipts']
    assert all('simulated_return_condition' not in s for s in owner['catalog'])
    assert {o['id'] for o in buyer['orders']}=={'order0','order1'}


def test_oracle_does_not_trust_actual_applier_and_replay_requires_authority(monkeypatch):
    from usami_fulfillment import adapter
    actual=adapter.apply
    def broken(world,actor,operation,event):
        state,result=actual(world,actor,operation,event);state.warehouses['w2'].capacity_volume+=1;return state,result
    monkeypatch.setattr(adapter,'apply',broken)
    engine=Engine(generate_world());step(engine,op('capture_payment',amount_cents=5300,key='pay',token='card_approved'))
    trace=engine.finish();assert 'EXPECTED_TRANSITION_MISMATCH' in evaluate(trace,engine.requests)['operations'][0]['violations']
    monkeypatch.setattr(adapter,'apply',actual)
    with pytest.raises(ValueError):replay_trace(trace,engine.requests)


@pytest.mark.parametrize('params',[{'quantity':True},{'quantity':0},{'quantity':-1},{'quantity':'4'}])
def test_strict_integer_input_has_no_implicit_stock_conversion(params):
    with pytest.raises(ValueError):op('reserve_inventory',reservation_id='r0',line_id='line0',lot_id='lot0',**params)
