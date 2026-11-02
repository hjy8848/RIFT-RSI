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


def make_world(seed, quantity, amount, age, approval, status, *, line_count=1, order_count=1,
               stock=10, return_required=False):
    if type(line_count) is not int or line_count < 1 or type(order_count) is not int or not 1 <= order_count <= line_count:
        raise ValueError('positive line count and valid order count required')
    lines_by_order = [[] for _ in range(order_count)]
    skus = [SKUVariable(alias=f'item_{index}', family='device', price_cents=fixed(1000),
                        return_required=return_required) for index in range(line_count)]
    for index in range(line_count):
        lines_by_order[index % order_count].append(LineVariable(alias=f'line_{index}',
            original_sku=f'item_{index}', quantity=fixed(quantity), paid_cents=fixed(amount)))
    spec = ScenarioSpec(scenario_id='differential', customer_aliases=('customer',),
        skus=tuple(skus) + (SKUVariable(alias='alternative', family='device', price_cents=fixed(1200)),),
        warehouses=(WarehouseVariable(alias='warehouse', delivery_days=fixed(1), delivery_fee_cents=fixed(0)),),
        orders=tuple(OrderVariable(alias=f'order_{index}', customer_alias='customer',
            delivered_days_ago=fixed(age), fulfillment_status=status, lines=tuple(lines))
            for index, lines in enumerate(lines_by_order)),
        default_stock=fixed(stock), refund_approval_limit_cents=approval, provenance_kind='offline_fixture')
    built = build_world(spec, seed)
    assert built.status == 'SAT'
    world, aliases = built.world, dict(built.private_aliases)
    line_ids = tuple(aliases[f'line_{index}'] for index in range(line_count))
    world.delegations['test-delegation'] = Delegation(id='test-delegation',
        customer_id=aliases['customer'], line_ids=line_ids,
        allowed_preparations=tuple(sorted(PREPARATIONS)), expires_at=world.clock + timedelta(days=1),
        source_customer_event_hash=digest('test customer delegation'))
    assert adapter.invariants(world) == []
    aliases['line'], aliases['order'] = aliases['line_0'], aliases['order_0']
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
       actions=st.lists(st.sampled_from(['orders', 'details', 'refund', 'cancel', 'reship',
                                        'bad_refund', 'exchange', 'bad_exchange']),
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
            'exchange': ('exchange_delivered_order_items', line,
                         {'quantity': 1, 'target_sku_id': ids['alternative']}),
            'bad_exchange': ('exchange_delivered_order_items', line,
                             {'quantity': quantity + 1, 'target_sku_id': ids['alternative']}),
            'cancel': ('cancel_order', order, {}),
            'reship': ('reship_order_items', line, {'quantity': 1}),
        }[action]
        return Operation(name=name, scope_id=merchant, resource_id=resource, params=params)
    verify_sequence(world, ids['customer'], tuple(operation(action) for action in actions))


def test_differential_multiple_lines_orders_and_tight_exchange_stock():
    world, ids = make_world(71, 1, 1000, 1, 20000, 'delivered', line_count=3,
                            order_count=2, stock=1)
    first = Operation(name='exchange_delivered_order_items', scope_id=ids['merchant'],
        resource_id=ids['line_0'], params={'quantity': 1, 'target_sku_id': ids['alternative']})
    second = Operation(name='exchange_delivered_order_items', scope_id=ids['merchant'],
        resource_id=ids['line_2'], params={'quantity': 1, 'target_sku_id': ids['alternative']})
    assert world.lines[ids['line_0']].order_id != world.lines[ids['line_1']].order_id
    verify_sequence(world, ids['customer'], (first, second))


def test_differential_physical_return_requires_matching_receipt_event():
    from usami_v2.commerce.external import apply_event, reference_event
    from usami_v2.commerce.models import ExternalEvent

    world, ids = make_world(72, 1, 1000, 1, 20000, 'delivered', return_required=True)
    request_return = Operation(name='request_item_return', scope_id=ids['merchant'],
        resource_id=ids['line'], params={'quantity': 1})
    verify_sequence(world, ids['customer'], (request_return,))
    pending, result = adapter.apply(world, ids['customer'], request_return, 0)
    assert result.success is True and len(pending.returns) == 1
    case = next(iter(pending.cases.values()))
    event = ExternalEvent(id='return-receipt-test', kind='return_received', target_id=case.id,
                          at=pending.clock + timedelta(days=1))
    pending.external_events[event.id] = event
    expected = reference_event(pending, event.id)
    received = apply_event(pending, event.id)
    assert received == expected
    assert not adapter.invariants(received)
    refund = Operation(name='return_delivered_order_items', scope_id=ids['merchant'],
        resource_id=ids['line'], params={'quantity': 1})
    verify_sequence(received, ids['customer'], (refund,))


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


def test_differential_check_detects_injected_wrong_refund_amount():
    world, ids = make_world(73, 1, 1000, 1, 20000, 'delivered')
    def mutant_apply(state, actor, operation, index):
        new, result = adapter.apply(state, actor, operation, index)
        if result.success and operation.name == 'return_delivered_order_items':
            row_id, row = next((key, item) for key, item in new.ledger.items()
                               if item.kind == 'refund')
            new.ledger[row_id] = row.model_copy(update={'amount_cents': row.amount_cents + 1})
        return new, result
    mutant = SimpleNamespace(name=adapter.name, World=adapter.World, Operation=adapter.Operation,
        invariants=adapter.invariants, decide=adapter.decide, apply=mutant_apply)
    operation = Operation(name='return_delivered_order_items', scope_id=ids['merchant'],
                          resource_id=ids['line'], params={'quantity': 1})
    with pytest.raises(AssertionError):
        verify_sequence(world, ids['customer'], (operation,), current_adapter=mutant)


def test_differential_check_detects_a_cancellation_that_leaks_reserved_inventory():
    world, ids = make_world(74, 1, 1000, 1, 20000, 'unshipped')
    line = world.lines[ids['line']]
    inventory_id = line.reserved_inventory_id
    assert inventory_id is not None
    def mutant_apply(state, actor, operation, index):
        new, result = adapter.apply(state, actor, operation, index)
        if result.success and operation.name == 'cancel_order':
            new.inventory[inventory_id] = state.inventory[inventory_id]
        return new, result
    mutant = SimpleNamespace(name=adapter.name, World=adapter.World, Operation=adapter.Operation,
        invariants=adapter.invariants, decide=adapter.decide, apply=mutant_apply)
    operation = Operation(name='cancel_order', scope_id=ids['merchant'], resource_id=ids['order'])
    with pytest.raises(AssertionError):
        verify_sequence(world, ids['customer'], (operation,), current_adapter=mutant)


def test_differential_check_detects_return_window_boundary_mutation():
    world, ids = make_world(75, 1, 1000, 30, 20000, 'delivered')
    assert adapter.decide(world, ids['customer'], Operation(name='return_delivered_order_items',
        scope_id=ids['merchant'], resource_id=ids['line'], params={'quantity': 1})).allowed
    def mutant_apply(state, actor, operation, index):
        if operation.name == 'return_delivered_order_items':
            return state.clone(), Result(success=False, error_code='RETURN_WINDOW_EXPIRED')
        return adapter.apply(state, actor, operation, index)
    mutant = SimpleNamespace(name=adapter.name, World=adapter.World, Operation=adapter.Operation,
        invariants=adapter.invariants, decide=adapter.decide, apply=mutant_apply)
    operation = Operation(name='return_delivered_order_items', scope_id=ids['merchant'],
                          resource_id=ids['line'], params={'quantity': 1})
    with pytest.raises(AssertionError):
        verify_sequence(world, ids['customer'], (operation,), current_adapter=mutant)
