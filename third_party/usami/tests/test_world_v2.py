from datetime import timedelta
import pytest
from usami import generate_world, make_request, Operation, WorldEngine, evaluate_trace, replay_trace
from usami.benchmark.release_v2 import configuration
from usami.world.generator_v2 import WORKFLOWS
from usami.world.rules import ReferenceRules, check_invariants
from usami.world.rules_v2 import member_id, renewal_id, renewal_period
from usami.world.models import digest


def world(family='seat_activate', seed=42): return generate_world(seed, configuration(family))
def op(name, resource='sub_000', org='org_000', **params): return Operation(name=name, org_id=org, resource_id=resource, params=params)
def run(engine, operation, actor='user_000', mode='reference'):
    req = make_request(f"r_{len(engine.trace.events)}", actor, operation)
    engine.register_request(req)
    return engine.execute(actor, operation, req.request_id, mode).payload['result']
def verified(engine):
    trace = engine.finish()
    assert evaluate_trace(trace).status == 'pass'
    assert replay_trace(trace).state == engine.state
    assert not check_invariants(engine.state)


@pytest.mark.parametrize('family', WORKFLOWS)
@pytest.mark.parametrize('seed', [0, 1, 42])
def test_generator_consistent(family, seed):
    a = world(family, seed)
    assert a == world(family, seed) and not check_invariants(a)
    assert a.generator_version.endswith('0.2.0')


@pytest.mark.parametrize('name,resource,params,expected', [
    ('invite_member','org_000',{'user_id':'missing'},'USER_NOT_FOUND'),
    ('invite_member','org_000',{'user_id':'user_001'},'DUPLICATE_INVITATION'),
    ('activate_member',member_id('org_000','user_006'),{},'NO_AVAILABLE_SEATS'),
    ('activate_member',member_id('org_000','user_001'),{},'MEMBERSHIP_ALREADY_ACTIVE'),
    ('deactivate_member',member_id('org_000','user_006'),{},'MEMBERSHIP_NOT_ACTIVE'),
    ('remove_member',member_id('org_000','user_001'),{},'MEMBERSHIP_STILL_ACTIVE'),
    ('activate_member',member_id('org_000','user_000'),{},'OWNER_IMMUTABLE'),
    ('deactivate_member',member_id('org_000','user_000'),{},'OWNER_IMMUTABLE'),
    ('remove_member',member_id('org_000','user_000'),{},'OWNER_IMMUTABLE'),
    ('expire_subscription','sub_000',{},'PERIOD_NOT_ENDED'),
    ('reactivate_subscription','sub_000',{'invoice_id':'missing'},'SUBSCRIPTION_STATUS_MISMATCH'),
    ('pay_invoice','invoice_000_recent',{'payment_method':'card_visa','idempotency_key':'new'},'INVOICE_ALREADY_PAID'),
    ('refund_invoice','invoice_000_recent',{'amount_cents':0,'payment_method':'card_visa','idempotency_key':'a'},'REFUND_AMOUNT_EXCEEDED'),
    ('refund_invoice','invoice_000_recent',{'amount_cents':1,'payment_method':'card_visa'},'REFUND_KEY_REQUIRED'),
])
def test_negative_branches(name, resource, params, expected):
    engine = WorldEngine(world()); before = engine.state
    result = run(engine, op(name, resource, **params))
    assert result['error_code'] == expected and not result['success']
    assert engine.state == before
    verified(engine)


@pytest.mark.parametrize('name,resource,params', [
    ('invite_member','org_000',{'user_id':'user_005'}),
    ('activate_member',member_id('org_000','user_006'),{}),
    ('deactivate_member',member_id('org_000','user_001'),{}),
    ('remove_member',member_id('org_000','user_007'),{}),
    ('pay_invoice','invoice_000_unpaid',{'payment_method':'card_visa','idempotency_key':'settlement'}),
    ('issue_renewal','sub_000',{}),
    ('renew_subscription','sub_000',{'invoice_id':'missing'}),
    ('reactivate_subscription','sub_000',{'invoice_id':'missing'}),
    ('expire_subscription','sub_000',{}),
])
def test_member_permission(name, resource, params):
    engine = WorldEngine(world())
    assert run(engine, op(name, resource, **params), actor='user_001')['error_code'] == 'PERMISSION_DENIED'
    assert run(engine, op(name, resource, **params), actor='user_007')['error_code'] == 'CROSS_ORG_ACCESS'
    verified(engine)


@pytest.mark.parametrize('name,resource,params', [
    ('activate_member',member_id('org_000','user_006'),{}),
    ('deactivate_member',member_id('org_000','user_001'),{}),
    ('remove_member',member_id('org_000','user_007'),{}),
    ('pay_invoice','invoice_000_unpaid',{'payment_method':'card_visa','idempotency_key':'x'}),
    ('issue_renewal','sub_000',{}),
    ('renew_subscription','sub_000',{'invoice_id':'missing'}),
    ('refund_invoice','invoice_000_recent',{'payment_method':'card_visa','amount_cents':1,'idempotency_key':'x'}),
])
def test_tenant_substitution(name, resource, params):
    engine = WorldEngine(world()); before = engine.state
    other_owner = before.organizations['org_001'].owner_id
    result = run(engine, op(name,resource,org='org_001',**params), actor=other_owner)
    assert result['error_code'] == 'CROSS_ORG_RESOURCE' and engine.state == before
    verified(engine)


def test_membership_lifecycle_capacity_and_owner():
    engine = WorldEngine(world('invite_activate'))
    invite = op('invite_member','org_000',user_id='user_005',role='member')
    first = run(engine, invite); after = engine.state
    assert run(engine, invite) == first and engine.state == after
    mid = first['data']['membership_id']
    assert run(engine, op('activate_member',mid))['success']
    assert run(engine, op('change_seats',seats=3))['error_code'] == 'SEATS_BELOW_MEMBERS'
    assert run(engine, op('deactivate_member',mid))['success']
    assert run(engine, op('remove_member',mid))['success']
    assert mid not in engine.state.memberships
    assert run(engine, invite)['success']
    verified(engine)


def test_inactive_activation_equal_capacity():
    engine = WorldEngine(world())
    assert run(engine, op('change_seats',seats=4))['success']
    assert run(engine, op('activate_member',member_id('org_000','user_007')))['success']
    assert run(engine, op('activate_member',member_id('org_000','user_006')))['error_code'] == 'NO_AVAILABLE_SEATS'
    verified(engine)


def test_failed_payment_exact_retry_conflict_recovery_and_charge_conservation():
    engine = WorldEngine(world('unpaid_restore'))
    start, _ = renewal_period(engine.state, engine.state.subscriptions['sub_000']); iid = renewal_id(engine.state.subscriptions['sub_000'],start)
    failed = op('pay_invoice',iid,payment_method='card_declined',idempotency_key='first')
    first = run(engine,failed); state = engine.state
    assert first['success'] and first['data']['outcome'] == 'failed'
    assert run(engine, failed) == first and engine.state == state
    assert run(engine,op('pay_invoice',iid,payment_method='card_visa',idempotency_key='first'))['error_code'] == 'IDEMPOTENCY_CONFLICT'
    successful = op('pay_invoice',iid,payment_method='card_visa',idempotency_key='second')
    success = run(engine,successful); after = engine.state
    assert run(engine,successful) == success and engine.state == after
    assert run(engine,op('pay_invoice',iid,payment_method='card_visa',idempotency_key='third'))['error_code'] == 'INVOICE_ALREADY_PAID'
    assert run(engine,op('reactivate_subscription',invoice_id=iid))['success']
    assert run(engine,op('reactivate_subscription',invoice_id=iid))['error_code'] == 'SUBSCRIPTION_STATUS_MISMATCH'
    verified(engine)


def test_renewal_creation_payment_period_and_duplicate_extension():
    engine = WorldEngine(world('renew_cycle'))
    issued = run(engine,op('issue_renewal')); after = engine.state
    assert run(engine,op('issue_renewal')) == issued and engine.state == after
    iid = issued['data']['invoice_id']
    assert run(engine,op('renew_subscription',invoice_id=iid))['error_code'] == 'INVOICE_NOT_PAID'
    assert run(engine,op('renew_subscription',invoice_id='invoice_000_recent'))['error_code'] == 'BILLING_PERIOD_MISMATCH'
    assert run(engine,op('renew_subscription',invoice_id='invoice_001_recent'))['error_code'] == 'CROSS_ORG_RESOURCE'
    assert run(engine,op('pay_invoice',iid,payment_method='bank_transfer',idempotency_key='settlement'))['success']
    old_end = engine.state.subscriptions['sub_000'].period_end
    assert run(engine,op('renew_subscription',invoice_id=iid))['success']
    assert engine.state.subscriptions['sub_000'].period_end == old_end + timedelta(days=30)
    assert run(engine,op('renew_subscription',invoice_id=iid))['error_code'] == 'BILLING_PERIOD_MISMATCH'
    verified(engine)


def test_expired_activation_denied_expiration_idempotent():
    engine = WorldEngine(world('unpaid_restore'))
    assert run(engine,op('activate_member',member_id('org_000','user_006')))['error_code'] == 'SUBSCRIPTION_NOT_ACTIVE'
    assert run(engine,op('change_seats',seats=4))['error_code'] == 'SUBSCRIPTION_NOT_ACTIVE'
    assert run(engine,op('expire_subscription'))['success']
    assert run(engine,op('expire_subscription'))['success']
    verified(engine)


@pytest.mark.parametrize('age,allowed', [(14,True),(15,False)])
def test_partial_refund_window(age, allowed):
    w = world(); inv = w.invoices['invoice_000_recent']; inv.paid_at = w.clock - timedelta(days=age); inv.payments[0].attempted_at = inv.paid_at
    engine = WorldEngine(w)
    result = run(engine,op('refund_invoice',inv.id,amount_cents=1,payment_method=inv.payment_method,idempotency_key='boundary'))
    assert result['success'] is allowed
    verified(engine)


def test_partial_refund_cumulative_limits_retry_and_method():
    w = world(); inv = w.invoices['invoice_000_recent']; engine = WorldEngine(w)
    partial = op('refund_invoice',inv.id,amount_cents=inv.amount_cents//2,payment_method=inv.payment_method,idempotency_key='first')
    first = run(engine,partial); after = engine.state
    assert run(engine,partial) == first and engine.state == after
    assert run(engine,op('refund_invoice',inv.id,amount_cents=1,payment_method=inv.payment_method,idempotency_key='first'))['error_code'] == 'IDEMPOTENCY_CONFLICT'
    remain = engine.state.invoices[inv.id].refundable_amount_cents
    assert run(engine,op('refund_invoice',inv.id,amount_cents=remain+1,payment_method=inv.payment_method,idempotency_key='over'))['error_code'] == 'REFUND_AMOUNT_EXCEEDED'
    assert run(engine,op('refund_invoice',inv.id,amount_cents=remain,payment_method='wrong',idempotency_key='wrong'))['error_code'] == 'REFUND_METHOD_MISMATCH'
    final = op('refund_invoice',inv.id,amount_cents=remain,payment_method=inv.payment_method,idempotency_key='last')
    assert run(engine,final)['success'] and engine.state.invoices[inv.id].status == 'refunded'
    assert run(engine,final)['success']
    assert run(engine,op('refund_invoice',inv.id,amount_cents=1,payment_method=inv.payment_method,idempotency_key='extra'))['error_code'] == 'INVOICE_NOT_PAID'
    verified(engine)


@pytest.mark.parametrize('attack', ['capacity','duplicate_payment','over_refund','owner'])
def test_sandbox_real_violations_and_replay(attack):
    w = world(); engine = WorldEngine(w)
    if attack == 'capacity': operation = op('activate_member',member_id('org_000','user_006'))
    elif attack == 'duplicate_payment': operation = op('pay_invoice','invoice_000_recent',payment_method='card_visa',idempotency_key='second-charge')
    elif attack == 'over_refund':
        inv = w.invoices['invoice_000_recent']; operation = op('refund_invoice',inv.id,amount_cents=inv.amount_cents+1,payment_method=inv.payment_method,idempotency_key='over')
    else: operation = op('deactivate_member',member_id('org_000','user_000'))
    assert run(engine,operation,mode='sandbox')['success']
    trace = engine.finish()
    assert evaluate_trace(trace).status == 'fail'
    assert replay_trace(trace).state == engine.state
    assert check_invariants(engine.state)


def test_engine_transition_independent_oracle_detects_broken_implementation(monkeypatch):
    import usami.world.transitions_v2 as actual
    original = actual.apply_v2
    def broken(w, actor, operation, event):
        state, result = original(w,actor,operation,event)
        state.organizations['org_001'].name = 'unrelated mutation'
        return state, result
    monkeypatch.setattr(actual,'apply_v2',broken)
    engine = WorldEngine(world()); run(engine,op('change_seats',seats=4))
    assert evaluate_trace(engine.finish()).status == 'fail'


def test_v1_hash_and_explicit_version_boundary():
    w = generate_world()
    assert w.state_hash == 'cbdc43984d682555281707a7431c7ed702f0c9bc4a3ebb3346b614f10936e0ec'
    dec = ReferenceRules().decide(w,'user_000',op('issue_renewal'))
    assert dec.error_code == 'UNSUPPORTED_OPERATION_VERSION'


def test_elapsed_period_requires_expiration_and_blocks_stale_access():
    engine=WorldEngine(world('expire_cycle'))
    assert run(engine,op('change_seats',seats=4))['error_code']=='SUBSCRIPTION_NOT_ACTIVE'
    assert run(engine,op('activate_member',member_id('org_000','user_006')))['error_code']=='SUBSCRIPTION_NOT_ACTIVE'
    assert run(engine,op('renew_subscription',invoice_id='invoice_000_recent'))['error_code']=='PERIOD_ENDED'
    assert run(engine,op('expire_subscription'))['success']
    assert engine.state.subscriptions['sub_000'].status=='expired'
    assert engine.trace.schema_version=='usami-trace-0.2.0'
    verified(engine)


def test_billing_admin_cannot_manage_members_even_if_capacity_is_available():
    engine=WorldEngine(world('invite_activate'))
    assert run(engine,op('activate_member',member_id('org_000','user_006')),actor='user_002')['error_code']=='PERMISSION_DENIED'
    assert run(engine,op('invite_member','org_000',user_id='user_005'),actor='user_002')['error_code']=='PERMISSION_DENIED'
    verified(engine)


def test_optional_invite_role_normalizes_before_authorization():
    implicit=op('invite_member','org_000',user_id='user_005')
    explicit=op('invite_member','org_000',user_id='user_005',role='member')
    assert implicit==explicit
    assert make_request('invite','user_000',explicit).authorizes('user_000',implicit)


def test_legacy_rules_reject_new_state_fields():
    w=generate_world();nonowner=next(m for m in w.memberships.values() if m.role!='owner')
    nonowner.status='pending'
    assert any(v.code=='UNSUPPORTED_STATE_VERSION' for v in check_invariants(w))
    w=generate_world();next(iter(w.invoices.values())).payments=[]
    assert any(v.code=='UNSUPPORTED_STATE_VERSION' for v in check_invariants(w))


def test_query_projections_are_reference_checked_and_private_emails_scoped():
    w=world();engine=WorldEngine(w)
    data=run(engine,op('list_members','org_000'),actor='user_001')['data']
    assert set(data['directory_users'][0])=={'id','name'}
    assert {u['id'] for u in data['users']}=={m.user_id for m in w.memberships.values() if m.org_id=='org_000'}
    assert run(engine,op('get_invoice','invoice_000_recent'),actor='user_001')['error_code']=='PERMISSION_DENIED'
    info=run(engine,op('get_organization','org_000'))['data']
    assert all(w.invoices[i].org_id=='org_000' for i in info['invoice_ids'])
    verified(engine)
