"""Differential Engine/Reference/Oracle/Replay checks on generated small Worlds."""
from datetime import timedelta
from types import SimpleNamespace

import pytest
from hypothesis import given, settings, strategies as st

from usami_harness.engine import Engine, replay_trace
from usami_harness.models import Request, Result, digest
from usami_harness.oracle import evaluate
from usami_v2.commerce import adapter
from usami_v2.commerce.models import Delegation, Operation, PREPARATIONS
from usami_v2.scenarios.builder import build_world
from usami_v2.scenarios.schema import (IntegerRange, LineVariable, OrderVariable,
    ScenarioSpec, SKUVariable, WarehouseVariable)


def fixed(value):
    return IntegerRange(low=value, high=value)


def make_world(seed, quantity, amount, age, approval, status):
    spec = ScenarioSpec(scenario_id='differential', customer_aliases=('customer',),
        skus=(SKUVariable(alias='item', family='device', price_cents=fixed(1000)),
              SKUVariable(alias='alternative', family='device', price_cents=fixed(1200))),
        warehouses=(WarehouseVariable(alias='warehouse', delivery_days=fixed(1), delivery_fee_cents=fixed(0)),),
        orders=(OrderVariable(alias='order', customer_alias='customer', delivered_days_ago=fixed(age),
            fulfillment_status=status, lines=(LineVariable(alias='line', original_sku='item',
                quantity=fixed(quantity), paid_cents=fixed(amount)),)),),
        default_stock=fixed(10), refund_approval_limit_cents=approval, provenance_kind='offline_fixture')
    built = build_world(spec, seed)
    assert built.status == 'SAT'
    world, aliases = built.world, dict(built.private_aliases)
    world.delegations['test-delegation'] = Delegation(id='test-delegation',
        customer_id=aliases['customer'], line_ids=(aliases['line'],),
        allowed_preparations=tuple(sorted(PREPARATIONS)), expires_at=world.clock + timedelta(days=1),
        source_customer_event_hash=digest('test customer delegation'))
    assert adapter.invariants(world) == []
    return world, aliases


def verify_sequence(world, actor, operations, *, current_adapter=adapter):
    engine = Engine(current_adapter, world, operation_budget=len(operations))
    requests = {}
    for index, operation in enumerate(operations):
        request = Request(id=f'request-{index}', identity=actor,
                          operation=operation.model_dump(mode='json'))
        requests[request.id] = request
        before = engine.state
        decision = current_adapter.decide(before, actor, operation)
        projected, direct = current_adapter.apply(before, actor, operation, 0)
        engine.register(request)
        event = engine.execute(actor, operation, request.id, mode='reference')
        result = event.payload['result']
        assert result['success'] == decision.allowed
        assert direct.success == decision.allowed
        if decision.allowed:
            assert engine.state == decision.expected_state == projected
            assert result['data'] == decision.data == direct.data
        else:
            assert result['error_code'] == decision.error_code == direct.error_code
            assert engine.state == before == projected
        assert not current_adapter.invariants(engine.state)
    trace = engine.finish()
    oracle = evaluate(current_adapter, trace, requests)
    assert oracle['trace_valid'] and oracle['status'] == 'pass'
    assert replay_trace(current_adapter, trace, requests).state == engine.state


@settings(max_examples=2000, deadline=None, database=None, derandomize=True)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1),
       quantity=st.integers(min_value=1, max_value=4), amount=st.integers(min_value=1, max_value=20000),
       age=st.integers(min_value=0, max_value=60), approval=st.integers(min_value=0, max_value=20000),
       status=st.sampled_from(['delivered', 'shipped', 'unshipped']),
       actions=st.lists(st.sampled_from(['orders', 'details', 'refund', 'cancel', 'reship', 'bad_refund']),
                        min_size=1, max_size=4))
def test_engine_reference_oracle_replay_generated_scenarios(seed, quantity, amount, age, approval, status, actions):
    world, ids = make_world(seed, quantity, amount, age, approval, status)
    merchant, line, order = ids['merchant'], ids['line'], ids['order']
    def operation(action):
        name, resource, params = {
            'orders': ('list_customer_orders', merchant, {}),
            'details': ('get_order_details', order, {}),
            'refund': ('return_delivered_order_items', line, {'quantity': 1}),
            'bad_refund': ('return_delivered_order_items', line, {'quantity': quantity + 1}),
            'cancel': ('cancel_order', order, {}),
            'reship': ('reship_order_items', line, {'quantity': 1}),
        }[action]
        return Operation(name=name, scope_id=merchant, resource_id=resource, params=params)
    verify_sequence(world, ids['customer'], tuple(operation(action) for action in actions))


def test_differential_check_detects_injected_wrong_query_data():
    world, ids = make_world(5, 1, 1000, 1, 2000, 'delivered')
    def mutant_apply(state, actor, operation, index):
        new, result = adapter.apply(state, actor, operation, index)
        if result.success:
            result = Result(success=True, data={'wrong': True})
        return new, result
    mutant = SimpleNamespace(name=adapter.name, World=adapter.World, Operation=adapter.Operation,
        invariants=adapter.invariants, decide=adapter.decide, apply=mutant_apply)
    operation = Operation(name='get_order_details', scope_id=ids['merchant'], resource_id=ids['order'])
    with pytest.raises(AssertionError):
        verify_sequence(world, ids['customer'], (operation,), current_adapter=mutant)
