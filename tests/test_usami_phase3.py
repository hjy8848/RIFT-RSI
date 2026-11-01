"""Phase 3 tests: order cancellation, exchange/reship and physical returns.

Every business call runs through the real Engine, independent Reference,
Oracle and trace Replay; only Service/Customer decisions are scripted.
"""
from datetime import timedelta

import pytest

from usami_harness.engine import Engine, replay_trace
from usami_harness.models import Request, digest
from usami_harness.oracle import evaluate
from usami_v2.commerce import adapter
from usami_v2.commerce.invariants import invariants
from usami_v2.commerce.models import Delegation, Operation, World
from usami_v2.commerce.reference import decide
from usami_v2.commerce.transitions import apply
from usami_v2.scenarios.builder import build_world
from usami_v2.scenarios.schema import (
    Attribute, BuildBudget, IntegerRange, LineVariable, OrderVariable, ScenarioSpec,
    SKUVariable, StockConstraint, WarehouseVariable,
)


def _fixed(value):
    return IntegerRange(low=value, high=value)


def _sku(alias, family, price_cents, *, return_required=False, **attributes):
    return SKUVariable(
        alias=alias, family=family, price_cents=_fixed(price_cents), return_required=return_required,
        attributes=tuple(Attribute(name=name, value=value) for name, value in attributes.items()))


def base_spec(**order_updates):
    """One customer with a delivered order and a second order the tests reshape."""
    return ScenarioSpec(
        scenario_id="phase3-business", provenance_kind="offline_fixture",
        customer_aliases=("customer", "stranger"),
        skus=(
            _sku("laptop_black", "laptop", 8000, color="black", product="laptop"),
            _sku("laptop_silver", "laptop", 9000, color="silver", product="laptop"),
            _sku("laptop_white", "laptop", 7000, color="white", product="laptop"),
            _sku("mouse", "mouse", 1200, product="mouse"),
            _sku("cable", "cable", 1000, product="cable"),
            _sku("monitor", "monitor", 30000, return_required=True, product="monitor"),
        ),
        warehouses=(
            WarehouseVariable(alias="fast", delivery_days=_fixed(2), delivery_fee_cents=_fixed(0)),
            WarehouseVariable(alias="slow", delivery_days=_fixed(6), delivery_fee_cents=_fixed(0)),
        ),
        orders=(
            OrderVariable(alias="delivered", customer_alias="customer", delivered_days_ago=_fixed(2),
                          lines=(LineVariable(alias="laptop_line", original_sku="laptop_black",
                                              quantity=_fixed(2), paid_cents=_fixed(16000)),
                                 LineVariable(alias="monitor_line", original_sku="monitor",
                                              quantity=_fixed(1), paid_cents=_fixed(30000)),
                                 LineVariable(alias="cable_line", original_sku="cable",
                                              quantity=_fixed(1), paid_cents=_fixed(1000),
                                              evidence_valid=False))),
            OrderVariable(alias="pending", customer_alias="customer", delivered_days_ago=_fixed(3),
                          lines=(LineVariable(alias="mouse_line", original_sku="mouse",
                                              quantity=_fixed(2), paid_cents=_fixed(2400)),
                                 LineVariable(alias="pending_cable", original_sku="cable",
                                              quantity=_fixed(1), paid_cents=_fixed(1000)))),
            OrderVariable(alias="foreign", customer_alias="stranger", delivered_days_ago=_fixed(3),
                          lines=(LineVariable(alias="foreign_mouse", original_sku="mouse",
                                              quantity=_fixed(1), paid_cents=_fixed(1200)),)),
        ),
        default_stock=_fixed(0),
        stock=(
            StockConstraint(sku_alias="laptop_silver", warehouse_alias="slow", quantity=_fixed(3)),
            StockConstraint(sku_alias="laptop_silver", warehouse_alias="fast", quantity=_fixed(1)),
            StockConstraint(sku_alias="laptop_white", warehouse_alias="fast", quantity=_fixed(3)),
            StockConstraint(sku_alias="laptop_black", warehouse_alias="fast", quantity=_fixed(1)),
            StockConstraint(sku_alias="mouse", warehouse_alias="fast", quantity=_fixed(4)),
            StockConstraint(sku_alias="cable", warehouse_alias="fast", quantity=_fixed(2)),
        ),
        **order_updates)


def built(seed=71):
    result = build_world(base_spec(), seed, BuildBudget(max_nodes=64))
    assert result.status == "SAT"
    return result.world, dict(result.private_aliases)


def reshape_order(world, order_id, status, *, reserve=()):
    """Turn a delivered fixture order into an unshipped/shipped one.

    ``reserve`` names the lines that actually hold stock; their quantity is
    added to the reserved (and opening) balance so inventory stays conserved.
    """
    data = world.model_dump(mode="json")
    data["orders"][order_id]["fulfillment_status"] = status
    data["orders"][order_id]["delivered_at"] = (world.clock + timedelta(days=3)).isoformat()
    for line_id in reserve:
        line = world.lines[line_id]
        inv = next(row for row in sorted(world.inventory.values(), key=lambda x: x.id)
                   if row.sku_id == line.sku_id and row.available > 0)
        data["lines"][line_id]["reserved_inventory_id"] = inv.id
        balance = data["inventory"][inv.id]
        balance["reserved"] += line.quantity
        balance["opening_quantity"] += line.quantity
    new = World.model_validate(data)
    assert not invariants(new), invariants(new)
    return new


def delegate(world, customer_id, line_ids):
    new = world.clone()
    key = "delegation_" + digest([customer_id, list(line_ids)])[:20]
    new.delegations[key] = Delegation(
        id=key, customer_id=customer_id, line_ids=tuple(line_ids), allowed_preparations=(),
        expires_at=world.clock + timedelta(days=30), source_customer_event_hash=digest(["test", key]))
    assert not invariants(new)
    return new


def execute(world, actor, name, resource, params=None, mode="reference"):
    """One native Engine call with its Oracle verdict and an exact trace Replay."""
    operation = Operation(name=name, scope_id=next(iter(world.merchants)), resource_id=resource,
                          params=params or {})
    request = Request(id="phase3-request", identity=actor, operation=operation.model_dump(mode="json"),
                      granted=True)
    engine = Engine(adapter, world, operation_budget=1)
    engine.register(request)
    event = engine.execute(actor, operation, request.id, mode=mode)
    trace = engine.finish()
    oracle = evaluate(adapter, trace, {request.id: request})
    replayed = replay_trace(adapter, trace, {request.id: request})
    assert replayed.state == engine.state
    return engine.state, event.payload["result"], oracle


def engine_matches_reference(world, actor, name, resource, params=None):
    """The actual transition and the independent Reference agree on state and data."""
    operation = Operation(name=name, scope_id=next(iter(world.merchants)), resource_id=resource,
                          params=params or {})
    decision = decide(world, actor, operation)
    actual, result = apply(world, actor, operation, 0)
    if decision.allowed:
        assert result.success and actual == decision.expected_state and result.data == decision.data
    else:
        assert actual == world
    return decision, result


def lines_of(world, order_id):
    return sorted(row.id for row in world.lines.values() if row.order_id == order_id)


# ---------------------------------------------------------------------------
# P3.1 Atomic order cancellation
# ---------------------------------------------------------------------------

def cancellable_world():
    world, aliases = built()
    world = reshape_order(world, aliases["pending"], "unshipped", reserve=(aliases["mouse_line"],))
    customer = aliases["customer"]
    return delegate(world, customer, lines_of(world, aliases["pending"])), aliases


def test_cancel_unshipped_order_refunds_every_line_and_releases_reserved_stock():
    world, aliases = cancellable_world()
    order_id, customer = aliases["pending"], aliases["customer"]
    mouse_inv = world.lines[aliases["mouse_line"]].reserved_inventory_id
    cable_balances = {k: v for k, v in world.inventory.items() if v.sku_id == world.lines[aliases["pending_cable"]].sku_id}

    after, result, oracle = execute(world, customer, "cancel_order", order_id)

    assert result["success"] and oracle["status"] == "pass" and not invariants(after)
    assert result["data"] == {"status": "order_cancelled", "order_id": order_id, "refund_cents": 3400,
                              "payment_channel": "original-card"}
    new_settlements = [row for k, row in after.settlements.items() if k not in world.settlements]
    assert sorted(row.line_id for row in new_settlements) == lines_of(world, order_id)
    assert {row.remedy for row in new_settlements} == {"cancel"}
    refunds = {row.line_id: row for row in after.ledger.values() if row.kind == "refund"}
    assert refunds[aliases["mouse_line"]].amount_cents == 2400
    assert refunds[aliases["pending_cable"]].amount_cents == 1000
    assert {row.channel for row in refunds.values()} == {"original-card"}
    # Only the reservation that actually existed is released.
    assert after.inventory[mouse_inv].reserved == world.inventory[mouse_inv].reserved - 2
    assert after.inventory[mouse_inv].available == world.inventory[mouse_inv].available + 2
    assert {k: after.inventory[k] for k in cable_balances} == cable_balances
    assert after.orders == world.orders and after.lines == world.lines
    assert not after.fulfillments


def test_cancel_engine_and_reference_agree():
    world, aliases = cancellable_world()
    decision, _ = engine_matches_reference(world, aliases["customer"], "cancel_order", aliases["pending"])
    assert decision.allowed


@pytest.mark.parametrize("status", ["shipped", "delivered"])
def test_cancel_rejects_orders_that_already_shipped(status):
    world, aliases = built()
    order_id = aliases["pending"] if status == "shipped" else aliases["delivered"]
    if status == "shipped":
        world = reshape_order(world, order_id, "shipped")
    world = delegate(world, aliases["customer"], lines_of(world, order_id))

    after, result, oracle = execute(world, aliases["customer"], "cancel_order", order_id)

    assert result["error_code"] == "ORDER_NOT_CANCELLABLE"
    assert after == world and oracle["status"] == "pass"
    engine_matches_reference(world, aliases["customer"], "cancel_order", order_id)


def test_cancel_another_customers_order_is_denied():
    world, aliases = cancellable_world()
    world = reshape_order(world, aliases["foreign"], "unshipped")
    after, result, oracle = execute(world, aliases["customer"], "cancel_order", aliases["foreign"])
    assert result["error_code"] == "OBJECT_ACCESS_DENIED" and after == world
    assert oracle["status"] == "pass"


def test_cancel_requires_delegation_for_every_order_line():
    world, aliases = built()
    world = reshape_order(world, aliases["pending"], "unshipped")
    world = delegate(world, aliases["customer"], [aliases["mouse_line"]])
    after, result, _ = execute(world, aliases["customer"], "cancel_order", aliases["pending"])
    assert result["error_code"] == "BUSINESS_DELEGATION_REQUIRED" and after == world


def test_cancel_twice_is_rejected_without_a_second_refund():
    world, aliases = cancellable_world()
    once, first, _ = execute(world, aliases["customer"], "cancel_order", aliases["pending"])
    twice, second, oracle = execute(once, aliases["customer"], "cancel_order", aliases["pending"])
    assert first["success"] and second["error_code"] == "ORDER_ALREADY_CANCELLED"
    assert twice == once and oracle["status"] == "pass"
    engine_matches_reference(once, aliases["customer"], "cancel_order", aliases["pending"])


def test_cancellation_is_not_a_delivered_item_refund():
    world, aliases = cancellable_world()
    after, result, _ = execute(world, aliases["customer"], "return_delivered_order_items",
                               aliases["mouse_line"], {"quantity": 1, "payment_method": "original"})
    assert result["error_code"] == "ORDER_NOT_DELIVERED" and after == world
    cancelled, _, _ = execute(world, aliases["customer"], "cancel_order", aliases["pending"])
    after, result, _ = execute(cancelled, aliases["customer"], "return_delivered_order_items",
                               aliases["mouse_line"], {"quantity": 1, "payment_method": "original"})
    assert result["error_code"] == "ORDER_NOT_DELIVERED" and after == cancelled


def test_cancel_in_sandbox_mode_still_never_creates_stock():
    world, aliases = cancellable_world()
    after, result, oracle = execute(world, aliases["customer"], "cancel_order", aliases["pending"],
                                    mode="sandbox")
    total_before = sum(row.opening_quantity + row.adjustments for row in world.inventory.values())
    total_after = sum(row.opening_quantity + row.adjustments for row in after.inventory.values())
    assert result["success"] and oracle["status"] == "pass" and total_before == total_after


def test_new_order_fields_are_omitted_at_their_defaults():
    world, aliases = built()
    data = world.model_dump(mode="json")
    assert all("fulfillment_status" not in row for row in data["orders"].values())
    assert all("reserved_inventory_id" not in row for row in data["lines"].values())
    reshaped = reshape_order(world, aliases["pending"], "unshipped", reserve=(aliases["mouse_line"],))
    dumped = reshaped.model_dump(mode="json")
    assert dumped["orders"][aliases["pending"]]["fulfillment_status"] == "unshipped"
    assert World.model_validate(dumped) == reshaped


def test_reservation_invariant_rejects_stock_held_for_a_delivered_order():
    world, aliases = built()
    data = world.model_dump(mode="json")
    line = world.lines[aliases["laptop_line"]]
    inv = next(row for row in world.inventory.values() if row.sku_id == line.sku_id)
    data["lines"][line.id]["reserved_inventory_id"] = inv.id
    codes = {row["code"] for row in invariants(World.model_validate(data))}
    assert "LINE_RESERVATION_INVALID" in codes


# ---------------------------------------------------------------------------
# P3.2 Atomic exchange and reship
# ---------------------------------------------------------------------------

def replacement_world(**merchant_updates):
    world, aliases = built()
    world = delegate(world, aliases["customer"],
                     [aliases["laptop_line"], aliases["monitor_line"], aliases["cable_line"]])
    if merchant_updates:
        world = world.clone()
        for merchant in world.merchants.values():
            for name, value in merchant_updates.items():
                setattr(merchant, name, value)
    return world, aliases


def sku_of(world, aliases, alias):
    return next(row.id for row in world.skus.values()
                if row.attributes.get("color") == alias or row.attributes.get("product") == alias)


def stock(world, sku_id):
    return {row.warehouse_id: (row.available, row.shipped) for row in world.inventory.values()
            if row.sku_id == sku_id}


def test_list_exchange_options_shows_only_compatible_stocked_skus():
    world, aliases = replacement_world()
    after, result, oracle = execute(world, aliases["customer"], "list_exchange_options", aliases["laptop_line"])
    assert result["success"] and oracle["status"] == "pass" and after == world
    options = {row["sku_id"]: row for row in result["data"]["options"]}
    silver, white = sku_of(world, aliases, "silver"), sku_of(world, aliases, "white")
    assert set(options) == {silver, white}
    assert options[silver]["available_quantity"] == 4 and options[silver]["fastest_delivery_days"] == 2
    assert options[white]["price_cents"] == 7000 and options[white]["attributes"]["color"] == "white"
    assert "inventory" not in str(result["data"]) and "warehouse" not in str(result["data"])
    engine_matches_reference(world, aliases["customer"], "list_exchange_options", aliases["laptop_line"])


def test_exchange_for_a_dearer_variant_ships_from_fastest_stock_and_charges_the_difference():
    world, aliases = replacement_world()
    silver = sku_of(world, aliases, "silver")
    before = stock(world, silver)

    after, result, oracle = execute(world, aliases["customer"], "exchange_delivered_order_items",
                                    aliases["laptop_line"], {"quantity": 1, "target_sku_id": silver})

    assert result["success"] and oracle["status"] == "pass" and not invariants(after)
    data = result["data"]
    assert data["status"] == "exchange_shipped" and data["extra_charge_cents"] == 1000
    assert data["refund_cents"] == 0 and data["promised_arrival"].startswith("2026-01-17")
    charges = [row for row in after.ledger.values() if row.kind == "extra_charge"]
    assert len(charges) == 1 and charges[0].amount_cents == 1000 and charges[0].channel == "original-card"
    assert charges[0].debit_account == aliases["customer"]
    (fulfillment,) = after.fulfillments.values()
    assert fulfillment.sku_id == silver and fulfillment.quantity == 1
    changed = {k: v for k, v in stock(after, silver).items() if v != before[k]}
    assert list(changed.values()) == [(0, 1)]
    engine_matches_reference(world, aliases["customer"], "exchange_delivered_order_items",
                             aliases["laptop_line"], {"quantity": 1, "target_sku_id": silver})


def test_exchange_of_two_units_falls_back_to_the_warehouse_that_can_cover_both():
    world, aliases = replacement_world()
    silver = sku_of(world, aliases, "silver")
    after, result, oracle = execute(world, aliases["customer"], "exchange_delivered_order_items",
                                    aliases["laptop_line"], {"quantity": 2, "target_sku_id": silver})
    assert result["success"] and oracle["status"] == "pass"
    assert result["data"]["extra_charge_cents"] == 2000
    assert result["data"]["promised_arrival"].startswith("2026-01-21")
    assert sorted(stock(after, silver).values()) == [(1, 0), (1, 2)]


def test_exchange_for_a_cheaper_variant_refunds_the_difference_to_the_original_channel():
    world, aliases = replacement_world()
    white = sku_of(world, aliases, "white")
    after, result, oracle = execute(world, aliases["customer"], "exchange_delivered_order_items",
                                    aliases["laptop_line"], {"quantity": 1, "target_sku_id": white})
    assert result["success"] and oracle["status"] == "pass" and not invariants(after)
    assert result["data"]["refund_cents"] == 1000 and result["data"]["extra_charge_cents"] == 0
    (refund,) = [row for row in after.ledger.values() if row.kind == "refund"]
    assert refund.amount_cents == 1000 and refund.credit_account == aliases["customer"]


def test_exchange_refund_difference_respects_the_approval_limit():
    world, aliases = replacement_world(refund_approval_limit_cents=500)
    white = sku_of(world, aliases, "white")
    after, result, oracle = execute(world, aliases["customer"], "exchange_delivered_order_items",
                                    aliases["laptop_line"], {"quantity": 1, "target_sku_id": white})
    assert result["error_code"] == "APPROVAL_REQUIRED" and after == world and oracle["status"] == "pass"


@pytest.mark.parametrize("target, code", [
    ("mouse", "REPLACEMENT_FAMILY_MISMATCH"),
    ("black", "EXCHANGE_TARGET_SAME_AS_ORIGINAL"),
    ("sku_does_not_exist", "EXCHANGE_TARGET_NOT_FOUND"),
])
def test_exchange_target_must_be_a_different_compatible_sku(target, code):
    world, aliases = replacement_world()
    target_id = target if target.startswith("sku_does") else sku_of(world, aliases, target)
    params = {"quantity": 1, "target_sku_id": target_id}
    after, result, oracle = execute(world, aliases["customer"], "exchange_delivered_order_items",
                                    aliases["laptop_line"], params)
    assert result["error_code"] == code and after == world and oracle["status"] == "pass"
    engine_matches_reference(world, aliases["customer"], "exchange_delivered_order_items",
                             aliases["laptop_line"], params)


def test_reship_sends_the_same_sku_at_no_charge():
    world, aliases = replacement_world()
    black = sku_of(world, aliases, "black")
    after, result, oracle = execute(world, aliases["customer"], "reship_order_items",
                                    aliases["laptop_line"], {"quantity": 1})
    assert result["success"] and oracle["status"] == "pass" and not invariants(after)
    assert result["data"]["status"] == "replacement_shipped" and result["data"]["target_sku_id"] == black
    assert not after.ledger
    (fulfillment,) = after.fulfillments.values()
    assert fulfillment.sku_id == black
    engine_matches_reference(world, aliases["customer"], "reship_order_items",
                             aliases["laptop_line"], {"quantity": 1})


def test_reship_without_stock_is_rejected_instead_of_inventing_inventory():
    world, aliases = replacement_world()
    after, result, oracle = execute(world, aliases["customer"], "reship_order_items",
                                    aliases["laptop_line"], {"quantity": 2})
    assert result["error_code"] == "INSUFFICIENT_STOCK" and after == world and oracle["status"] == "pass"
    engine_matches_reference(world, aliases["customer"], "reship_order_items",
                             aliases["laptop_line"], {"quantity": 2})


def test_reship_needs_a_verified_delivery_problem():
    world, aliases = replacement_world()
    after, result, _ = execute(world, aliases["customer"], "reship_order_items",
                               aliases["cable_line"], {"quantity": 1})
    assert result["error_code"] == "RESHIP_NOT_ELIGIBLE" and after == world


def test_replacements_respect_physical_return_window_and_delivery_rules():
    world, aliases = replacement_world()
    silver = sku_of(world, aliases, "silver")
    _, result, _ = execute(world, aliases["customer"], "reship_order_items",
                           aliases["monitor_line"], {"quantity": 1})
    assert result["error_code"] == "RETURN_REQUIRED"
    late = world.clone()
    late.clock = world.clock + timedelta(days=40)
    late = delegate(late, aliases["customer"], [aliases["laptop_line"]])
    _, result, _ = execute(late, aliases["customer"], "exchange_delivered_order_items",
                           aliases["laptop_line"], {"quantity": 1, "target_sku_id": silver})
    assert result["error_code"] == "RETURN_WINDOW_EXPIRED"
    pending = reshape_order(world, aliases["pending"], "shipped")
    pending = delegate(pending, aliases["customer"], [aliases["mouse_line"]])
    _, result, _ = execute(pending, aliases["customer"], "reship_order_items",
                           aliases["mouse_line"], {"quantity": 1})
    assert result["error_code"] == "ORDER_NOT_DELIVERED"


def test_replacement_quantities_cannot_exceed_or_repeat_purchased_units():
    world, aliases = replacement_world()
    white = sku_of(world, aliases, "white")
    _, result, _ = execute(world, aliases["customer"], "exchange_delivered_order_items",
                           aliases["laptop_line"], {"quantity": 3, "target_sku_id": white})
    assert result["error_code"] == "PURCHASED_QUANTITY_UNAVAILABLE"
    refunded, result, _ = execute(world, aliases["customer"], "return_delivered_order_items",
                                  aliases["laptop_line"], {"quantity": 1, "payment_method": "original"})
    exchanged, result, oracle = execute(refunded, aliases["customer"], "exchange_delivered_order_items",
                                        aliases["laptop_line"], {"quantity": 1, "target_sku_id": white})
    assert result["success"] and oracle["status"] == "pass" and not invariants(exchanged)
    for name, params in (("exchange_delivered_order_items", {"quantity": 1, "target_sku_id": white}),
                         ("reship_order_items", {"quantity": 1})):
        after, result, _ = execute(exchanged, aliases["customer"], name, aliases["laptop_line"], params)
        assert result["error_code"] == "ALREADY_SETTLED" and after == exchanged
    after, result, _ = execute(exchanged, aliases["customer"], "return_delivered_order_items",
                               aliases["laptop_line"], {"quantity": 1, "payment_method": "original"})
    assert result["error_code"] == "ALREADY_REFUNDED"


def test_replacement_for_another_customers_line_is_denied():
    world, aliases = replacement_world()
    _, result, oracle = execute(world, aliases["customer"], "reship_order_items",
                                aliases["foreign_mouse"], {"quantity": 1})
    assert result["error_code"] == "OBJECT_ACCESS_DENIED" and oracle["status"] == "pass"


# ---------------------------------------------------------------------------
# P3.3 Physical returns: accepted -> received (trusted event) -> refunded
# ---------------------------------------------------------------------------

from usami_v2.commerce.external import apply_event, reference_event
from usami_v2.commerce.models import ExternalEvent, case_id as native_case_id


def return_world(**merchant_updates):
    return replacement_world(**merchant_updates)


def receive(world, line_id, units=(0,), *, days=1, event_id="return-receipt-1"):
    """A trusted carrier scan; the Engine and Reference projections must agree."""
    staged = world.clone()
    staged.external_events[event_id] = ExternalEvent(
        id=event_id, kind="return_received", target_id=native_case_id(line_id, tuple(units)),
        at=world.clock + timedelta(days=days))
    expected, actual = reference_event(staged, event_id), apply_event(staged, event_id)
    assert expected == actual and not invariants(actual)
    return actual


def refund_params(quantity=1):
    return {"quantity": quantity, "payment_method": "original"}


def test_return_request_records_acceptance_only():
    world, aliases = return_world()
    after, result, oracle = execute(world, aliases["customer"], "request_item_return",
                                    aliases["monitor_line"], {"quantity": 1})
    assert result["success"] and oracle["status"] == "pass" and not invariants(after)
    assert result["data"] == {"status": "return_requested", "line_id": aliases["monitor_line"], "quantity": 1,
                              "refund_cents": 30000, "payment_channel": "original-card",
                              "refund_status": "awaiting_return_receipt"}
    (ret,) = after.returns.values()
    assert ret.status == "scheduled" and ret.received_event_id is None
    assert not after.ledger and not after.settlements
    assert [row.status for row in after.cases.values()] == ["awaiting_return"]
    engine_matches_reference(world, aliases["customer"], "request_item_return",
                             aliases["monitor_line"], {"quantity": 1})


def test_refund_waits_for_the_trusted_receipt_event():
    world, aliases = return_world()
    line = aliases["monitor_line"]
    requested, _, _ = execute(world, aliases["customer"], "request_item_return", line, {"quantity": 1})

    early, result, oracle = execute(requested, aliases["customer"], "return_delivered_order_items",
                                    line, refund_params())
    assert result["error_code"] == "RETURN_NOT_RECEIVED" and early == requested
    assert oracle["status"] == "pass"
    engine_matches_reference(requested, aliases["customer"], "return_delivered_order_items", line, refund_params())

    received = receive(requested, line)
    assert next(iter(received.returns.values())).status == "received"
    assert not received.ledger
    refunded, result, oracle = execute(received, aliases["customer"], "return_delivered_order_items",
                                       line, refund_params())
    assert result["success"] and oracle["status"] == "pass" and not invariants(refunded)
    assert result["data"]["status"] == "refund_issued" and result["data"]["refund_cents"] == 30000
    (refund,) = refunded.ledger.values()
    assert refund.amount_cents == 30000 and refund.channel == "original-card"
    assert [row.status for row in refunded.cases.values()] == ["resolved"]
    engine_matches_reference(received, aliases["customer"], "return_delivered_order_items", line, refund_params())


def test_no_agent_operation_can_mark_goods_received():
    world, aliases = return_world()
    line = aliases["monitor_line"]
    requested, _, _ = execute(world, aliases["customer"], "request_item_return", line, {"quantity": 1})
    for name, params in (("request_item_return", {"quantity": 1}),
                         ("return_delivered_order_items", refund_params()),
                         ("list_exchange_options", {}), ("get_order_details", {})):
        resource = requested.lines[line].order_id if name == "get_order_details" else line
        after, result, _ = execute(requested, aliases["customer"], name, resource, params)
        assert next(iter(after.returns.values())).status == "scheduled"
        assert not after.ledger
    _, again, _ = execute(requested, aliases["customer"], "request_item_return", line, {"quantity": 1})
    assert again["error_code"] == "RETURN_ALREADY_REQUESTED"


def test_receipt_after_the_window_still_allows_the_accepted_return_refund():
    world, aliases = return_world()
    line = aliases["monitor_line"]
    requested, _, _ = execute(world, aliases["customer"], "request_item_return", line, {"quantity": 1})
    received = receive(requested, line, days=40)
    received = delegate(received, aliases["customer"], [line])
    refunded, result, oracle = execute(received, aliases["customer"], "return_delivered_order_items",
                                       line, refund_params())
    assert result["success"] and oracle["status"] == "pass"


def test_return_request_rules_and_duplicates():
    world, aliases = return_world()
    _, result, _ = execute(world, aliases["customer"], "request_item_return",
                           aliases["laptop_line"], {"quantity": 1})
    assert result["error_code"] == "RETURN_NOT_REQUIRED"
    _, result, _ = execute(world, aliases["customer"], "return_delivered_order_items",
                           aliases["monitor_line"], refund_params())
    assert result["error_code"] == "RETURN_REQUIRED"
    _, result, _ = execute(world, aliases["customer"], "request_item_return",
                           aliases["monitor_line"], {"quantity": 2})
    assert result["error_code"] == "PURCHASED_QUANTITY_UNAVAILABLE"
    capped, _ = return_world(refund_approval_limit_cents=10000)
    _, result, _ = execute(capped, aliases["customer"], "request_item_return",
                           aliases["monitor_line"], {"quantity": 1})
    assert result["error_code"] == "APPROVAL_REQUIRED"
    _, result, _ = execute(world, aliases["customer"], "request_item_return",
                           aliases["foreign_mouse"], {"quantity": 1})
    assert result["error_code"] == "OBJECT_ACCESS_DENIED"


def test_returned_refund_cannot_be_repeated_or_resized():
    world, aliases = return_world()
    line = aliases["monitor_line"]
    requested, _, _ = execute(world, aliases["customer"], "request_item_return", line, {"quantity": 1})
    received = receive(requested, line)
    _, result, _ = execute(received, aliases["customer"], "return_delivered_order_items", line, refund_params(2))
    assert result["error_code"] == "RETURNED_QUANTITY_MISMATCH"
    refunded, _, _ = execute(received, aliases["customer"], "return_delivered_order_items", line, refund_params())
    after, result, oracle = execute(refunded, aliases["customer"], "return_delivered_order_items",
                                    line, refund_params())
    assert result["error_code"] == "RETURN_REQUIRED" and after == refunded and oracle["status"] == "pass"
    _, result, _ = execute(refunded, aliases["customer"], "request_item_return", line, {"quantity": 1})
    assert result["error_code"] == "ALREADY_REFUNDED"


# ---------------------------------------------------------------------------
# P3.4 Dynamic tasks, certification, episodes, Replay and Reward
# ---------------------------------------------------------------------------
import json

from rift.benchmark.private_evaluation import public_business_outcome
from rift.benchmark.public_feedback import service_reward
from rift.benchmark.usami import public_episode
from rift.schemas import Outcome
from usami_v2.runtime.agents import ServiceAction
from usami_v2.runtime.customer import CustomerDecision, CustomerProfile, bind_customer_profile
from usami_v2.runtime.runner import EpisodeBudget, replay_episode, run_episode
from usami_v2.scenarios.task_certification import (
    CustomerScenario, ProofBudget, certification_cost, certify_task, verify_task_certificate,
)
from usami_v2.scenarios.tasks import (
    CustomerRequestBundle, OrderCancellationGoal, RemedyGoal, TaskSpecV3, evaluate_state_goals,
    evaluate_task_compliance, render_customer_request, request_intent_hash, delegated_line_ids,
)


def v3_task(world, customer_id, remedies=(), cancellations=()):
    remedies, cancellations = tuple(remedies), tuple(cancellations)
    request = render_customer_request(world, customer_id, remedies, (), cancellations)
    bundle = CustomerRequestBundle(
        customer_id=customer_id, goal_ids=tuple(g.goal_id for g in remedies + cancellations),
        intent_hash=request_intent_hash(customer_id, remedies, (), cancellations),
        request_hash=digest(request))
    return TaskSpecV3(
        task_id="phase3-task", world_hash=world.state_hash, customer_id=customer_id,
        customer_request=request, remedy_goals=remedies, cancellation_goals=cancellations,
        partition="train", source_distribution_hash=digest("phase3 tests"), request_bundle=bundle)


def cancel_goal(world, order_id, goal_id="cancel-goal"):
    return OrderCancellationGoal(goal_id=goal_id, order_id=order_id, line_ids=tuple(lines_of(world, order_id)))


def remedy(line_id, remedies, *, goal_id, quantity=1, extra=None):
    return RemedyGoal(goal_id=goal_id, line_id=line_id, quantity=quantity, allowed_remedies=remedies,
                      maximum_extra_charge_cents=extra)


def scenario_for(task, **updates):
    allowed = tuple(sorted({r for g in task.remedy_goals for r in g.allowed_remedies})) or ("refund",)
    data = {"customer_id": task.customer_id, "delegated_line_ids": delegated_line_ids(task),
            "accepted_remedies": allowed}
    data.update(updates)
    return CustomerScenario(**data)


def profile_for(task):
    return bind_customer_profile(CustomerProfile(customer_id=task.customer_id), task)


def ops(certificate):
    witness = next(row for row in certificate.attempts if row.status == "WITNESS")
    return [step.evidence["request"]["operation"]["name"] for step in witness.steps if step.kind == "operation"]


def pending_world():
    world, aliases = built()
    world = reshape_order(world, aliases["pending"], "unshipped", reserve=(aliases["mouse_line"],))
    return world, aliases


def monitor_event(world, aliases, days=0):
    return ExternalEvent(id="receipt-monitor", kind="return_received",
                         target_id=native_case_id(aliases["monitor_line"], (0,)),
                         at=world.clock + timedelta(days=days))


class Ctx:
    def __init__(self):
        self.prompt = None

    @property
    def scope(self):
        return self.prompt.available_merchant_scope_ids[0]

    def op(self, name, resource=None, **params):
        return ServiceAction(kind="operation", operation=Operation(
            name=name, scope_id=self.scope, resource_id=resource or self.scope, params=params))

    def say(self, text):
        return ServiceAction(kind="reply", message=text)

    def data(self):
        rows = [json.loads(m.content) for m in self.prompt.conversation if m.role == "tool"]
        return next(row for row in reversed(rows) if "data" in row)["data"]

    def notices(self):
        return [json.loads(m.content) for m in self.prompt.conversation
                if m.role == "tool" and "external_event" in m.content]


class ScriptedService:
    uses_model = False
    last_call = None

    def __init__(self, script):
        self.script, self.gen, self.ctx = script, None, Ctx()

    def decide(self, prompt):
        self.ctx.prompt = prompt
        self.gen = self.gen or self.script(self.ctx)
        try:
            return next(self.gen)
        except StopIteration:
            return self.ctx.say("Everything is done.")


class YesCustomer:
    uses_model = False
    last_call = None

    def __init__(self, yes):
        self.yes, self.turns = yes, 0

    def decide(self, prompt, profile):
        self.turns += 1
        if self.turns <= self.yes:
            return CustomerDecision(message="Yes, please go ahead.")
        return CustomerDecision(message="Thanks, that is all.", finish=True)


def gather(ctx):
    """Public discovery: list orders, read every order, expose all lines as ctx.lines."""
    yield ctx.op("list_customer_orders")
    ctx.orders = ctx.data()["orders"]
    ctx.lines = []
    for order in ctx.orders:
        yield ctx.op("get_order_details", order["id"])
        ctx.lines += ctx.data()["lines"]


def product(ctx, name):
    return next(r for r in ctx.lines if r["sku"]["attributes"].get("product") == name)


def status(order):
    return order.get("fulfillment_status", "delivered")


def run(world, task, service, customer, events=()):
    return run_episode(world, task, profile_for(task), customer, service,
                       budget=EpisodeBudget(max_customer_turns=10, max_service_decisions=40,
                                            max_service_tool_calls=24, max_model_calls=0),
                       external_events=tuple(events))


def verified_reward(record):
    replay = replay_episode(record)
    assert replay.verified, replay.errors
    return service_reward(public_business_outcome(record, replay.verified))


# ---- certification -------------------------------------------------------

def test_cancel_goal_certifies_with_the_public_cancel_path_and_cost():
    world, aliases = pending_world()
    task = v3_task(world, aliases["customer"], cancellations=(cancel_goal(world, aliases["pending"]),))
    scenario = scenario_for(task)
    certificate = certify_task(world, task, scenario, ProofBudget())
    assert certificate.status == "CERTIFIED", certificate.blockers
    assert ops(certificate) == ["list_customer_orders", "cancel_order"]
    assert certification_cost(certificate) == {
        "tool_calls": 2, "service_decisions": 4, "customer_turns": 2, "model_calls": 6}
    assert verify_task_certificate(world, task, scenario, certificate)["verified"]


def test_cancelling_a_shipped_order_is_unsat_and_a_tiny_budget_is_unknown():
    world, aliases = built()
    shipped = reshape_order(world, aliases["pending"], "shipped")
    task = v3_task(shipped, aliases["customer"], cancellations=(cancel_goal(shipped, aliases["pending"]),))
    certificate = certify_task(shipped, task, scenario_for(task), ProofBudget())
    assert certificate.status == "UNSAT" and certificate.search_complete
    pending, _ = pending_world()
    task = v3_task(pending, aliases["customer"], cancellations=(cancel_goal(pending, aliases["pending"]),))
    short = certify_task(pending, task, scenario_for(task), ProofBudget(max_operation_steps=1))
    assert short.status == "UNKNOWN" and "proof_operation_budget_exhausted" in short.blockers


def test_exchange_certification_respects_the_customer_extra_charge_limit():
    world, aliases = built()
    goal = remedy(aliases["laptop_line"], ("exchange",), goal_id="swap", extra=0)
    task = v3_task(world, aliases["customer"], remedies=(goal,))
    certificate = certify_task(world, task, scenario_for(task, maximum_extra_charge_cents=0), ProofBudget())
    assert certificate.status == "CERTIFIED", certificate.blockers
    rejected = [row for row in certificate.attempts if row.status == "REJECTED"]
    assert [row.reason for row in rejected] == ["customer_extra_charge_limit"]
    assert ops(certificate) == ["list_customer_orders", "get_order_details",
                                "list_exchange_options", "exchange_delivered_order_items"]


def test_reship_and_unreachable_replacement_goals():
    world, aliases = built()
    task = v3_task(world, aliases["customer"], remedies=(remedy(aliases["laptop_line"], ("reship",), goal_id="again"),))
    assert certify_task(world, task, scenario_for(task), ProofBudget()).status == "CERTIFIED"
    # No cable stock reaches a warehouse for two units, and the cable has no valid delivery problem.
    task = v3_task(world, aliases["customer"], remedies=(remedy(aliases["cable_line"], ("reship",), goal_id="cable"),))
    assert certify_task(world, task, scenario_for(task), ProofBudget()).status == "UNSAT"


def test_physical_return_is_certified_only_with_a_frozen_receipt_event():
    world, aliases = built()
    goal = remedy(aliases["monitor_line"], ("refund",), goal_id="send-back")
    task = v3_task(world, aliases["customer"], remedies=(goal,))
    unknown = certify_task(world, task, scenario_for(task), ProofBudget())
    assert unknown.status == "UNKNOWN" and "return_receipt_event_not_configured" in unknown.blockers
    scenario = scenario_for(task, return_received_after_days=1)
    certificate = certify_task(world, task, scenario, ProofBudget())
    assert certificate.status == "CERTIFIED", certificate.blockers
    assert ops(certificate) == ["list_customer_orders", "get_order_details", "request_item_return",
                                "return_delivered_order_items"]
    # one confirmation covers the accepted return and the refund that follows receipt
    assert certification_cost(certificate) == {
        "tool_calls": 4, "service_decisions": 6, "customer_turns": 2, "model_calls": 8}
    assert verify_task_certificate(world, task, scenario, certificate)["verified"]


def test_mixed_task_certifies_every_business_type_and_costs_each_confirmation():
    world, aliases = pending_world()
    task = v3_task(world, aliases["customer"], remedies=(
        remedy(aliases["laptop_line"], ("exchange",), goal_id="swap", extra=2000),
        remedy(aliases["monitor_line"], ("refund",), goal_id="send-back")),
        cancellations=(cancel_goal(world, aliases["pending"]),))
    certificate = certify_task(world, task, scenario_for(task, return_received_after_days=1), ProofBudget())
    assert certificate.status == "CERTIFIED", certificate.blockers
    witness = next(row for row in certificate.attempts if row.status == "WITNESS")
    assert [g for g, _ in witness.goal_results] == ["swap", "send-back", "cancel-goal"]
    assert {s for _, s in witness.goal_results} == {"PASS"}
    cost = certification_cost(certificate)
    assert cost["customer_turns"] == 4 and cost["tool_calls"] == len(ops(certificate))


# ---- end-to-end episodes: Replay, goals and Reward ------------------------

def test_cancel_episode_succeeds_replays_and_earns_full_reward():
    world, aliases = pending_world()
    task = v3_task(world, aliases["customer"], cancellations=(cancel_goal(world, aliases["pending"]),))

    def script(ctx):
        yield from gather(ctx)
        order = next(row for row in ctx.orders if status(row) == "unshipped")
        yield ctx.say("I can cancel this unshipped order and refund 3400 to your original card. Proceed?")
        yield ctx.op("cancel_order", order["id"])
        yield ctx.say("The order is cancelled and refunded to the original card.")

    record = run(world, task, ScriptedService(script), YesCustomer(1))
    assert record.status == "customer_finished" and record.outcome.business_goal_success is True
    assert [(g.goal_id, g.status) for g in record.outcome.state_goals] == [("cancel-goal", "PASS")]
    assert verified_reward(record) == 1.0
    assert public_episode(record).outcome is Outcome.SUCCESS
    assert sum(e.amount_cents for e in record.final_world.ledger.values() if e.kind == "refund") == 3400


def test_cancel_episode_fails_for_a_delivered_order_without_changing_state():
    world, aliases = built()
    task = v3_task(world, aliases["customer"],
                   cancellations=(cancel_goal(world, aliases["delivered"], "cancel-delivered"),))

    def script(ctx):
        yield ctx.op("list_customer_orders")
        yield ctx.say("I will cancel the order. Proceed?")
        yield ctx.op("cancel_order", aliases["delivered"])
        yield ctx.say("Sorry, that order cannot be cancelled.")

    record = run(world, task, ScriptedService(script), YesCustomer(1))
    assert next(t for t in record.tool_executions if t.operation["name"] == "cancel_order").result[
        "error_code"] == "ORDER_NOT_CANCELLABLE"
    assert record.final_world.ledger == world.ledger and record.outcome.business_goal_success is False
    assert verified_reward(record) == 0.0


def test_exchange_episode_uses_listed_stock_and_posts_the_price_difference():
    world, aliases = built()
    task = v3_task(world, aliases["customer"],
                   remedies=(remedy(aliases["laptop_line"], ("exchange",), goal_id="swap", extra=2000),))

    def script(ctx):
        yield from gather(ctx)
        line = product(ctx, "laptop")
        yield ctx.op("list_exchange_options", line["id"])
        option = next(r for r in ctx.data()["options"] if r["attributes"].get("color") == "silver")
        yield ctx.say("I can exchange one laptop for the silver one, with a 10.00 extra charge. Proceed?")
        yield ctx.op("exchange_delivered_order_items", line["id"], quantity=1, target_sku_id=option["sku_id"])
        yield ctx.say("Your silver laptop ships in 2 days; the 10.00 difference is charged to the original card.")

    record = run(world, task, ScriptedService(script), YesCustomer(1))
    assert record.outcome.business_goal_success is True and verified_reward(record) == 1.0
    (settlement,) = record.final_world.settlements.values()
    assert settlement.remedy == "exchange"
    assert [e.amount_cents for e in record.final_world.ledger.values() if e.kind == "extra_charge"] == [1000]
    assert len(record.final_world.fulfillments) == 1


def test_exchange_over_the_goal_extra_charge_cap_fails_the_goal():
    world, aliases = built()
    task = v3_task(world, aliases["customer"],
                   remedies=(remedy(aliases["laptop_line"], ("exchange",), goal_id="swap", extra=0),))

    def script(ctx):
        yield from gather(ctx)
        line = product(ctx, "laptop")
        yield ctx.op("list_exchange_options", line["id"])
        option = next(r for r in ctx.data()["options"] if r["attributes"].get("color") == "silver")
        yield ctx.say("Exchange for silver with an extra charge?")
        yield ctx.op("exchange_delivered_order_items", line["id"], quantity=1, target_sku_id=option["sku_id"])
        yield ctx.say("Done.")

    record = run(world, task, ScriptedService(script), YesCustomer(1))
    assert [(g.goal_id, g.status) for g in record.outcome.state_goals] == [("swap", "FAIL")]
    assert verified_reward(record) == 0.0


def test_reship_episode_ships_the_same_item_for_free():
    world, aliases = built()
    task = v3_task(world, aliases["customer"],
                   remedies=(remedy(aliases["laptop_line"], ("reship",), goal_id="again"),))

    def script(ctx):
        yield from gather(ctx)
        line = product(ctx, "laptop")
        yield ctx.say("I can ship one replacement laptop at no charge. Proceed?")
        yield ctx.op("reship_order_items", line["id"], quantity=1)
        yield ctx.say("The replacement is on its way.")

    record = run(world, task, ScriptedService(script), YesCustomer(1))
    assert record.outcome.business_goal_success is True and verified_reward(record) == 1.0
    assert not record.final_world.ledger and len(record.final_world.fulfillments) == 1


def return_script(ctx, line_pick, *, refund_early=False):
    yield from gather(ctx)
    line = product(ctx, line_pick)
    yield ctx.say("This monitor must be sent back before the refund of 300.00. Shall I accept the return?")
    yield ctx.op("request_item_return", line["id"], quantity=1)
    if refund_early:
        yield ctx.op("return_delivered_order_items", line["id"], quantity=1, payment_method="original")
        yield ctx.say("I could not refund yet because the goods have not arrived.")
        return
    notice = ctx.notices()[-1]
    assert notice["line_id"] == line["id"] and "target_id" not in notice  # no internal case id
    yield ctx.op("return_delivered_order_items", notice["line_id"], quantity=notice["quantity"],
                 payment_method="original")
    yield ctx.say("The goods arrived and 300.00 is refunded to the original card.")


def test_return_episode_refunds_only_after_the_trusted_receipt_event():
    world, aliases = built()
    task = v3_task(world, aliases["customer"], remedies=(remedy(aliases["monitor_line"], ("refund",), goal_id="send-back"),))
    record = run(world, task, ScriptedService(lambda c: return_script(c, "monitor")), YesCustomer(1),
                 events=(monitor_event(world, aliases),))
    assert record.outcome.business_goal_success is True and verified_reward(record) == 1.0
    kinds = [e.kind for e in record.events]
    assert kinds.index("external_event") < max(i for i, k in enumerate(kinds) if k == "service_tool")
    refund = [t for t in record.tool_executions if t.operation["name"] == "return_delivered_order_items"]
    assert len(refund) == 1 and refund[0].result["success"]
    public = public_episode(record)
    receipts = [json.loads(e.content) for e in public.events
                if e.kind == "tool_result" and "external_event" in e.content]
    assert receipts == [{"external_event": "return_received", "status": "applied",
                         "line_id": aliases["monitor_line"], "quantity": 1}]


def test_refund_before_receipt_is_rejected_and_the_goal_fails():
    world, aliases = built()
    task = v3_task(world, aliases["customer"], remedies=(remedy(aliases["monitor_line"], ("refund",), goal_id="send-back"),))
    # The carrier scan is scheduled for later but never becomes ready before the Service stops.
    record = run(world, task, ScriptedService(lambda c: return_script(c, "monitor", refund_early=True)),
                 YesCustomer(1), events=())
    early = [t for t in record.tool_executions if t.operation["name"] == "return_delivered_order_items"]
    assert early[0].result["error_code"] == "RETURN_NOT_RECEIVED"
    assert not record.final_world.ledger and record.outcome.business_goal_success is False
    assert verified_reward(record) == 0.0


def mixed_script(ctx, *, skip_return=False, try_foreign_cancel=False):
    yield from gather(ctx)
    orders = ctx.orders
    by_product = {row["sku"]["attributes"].get("product"): row for row in ctx.lines}
    yield ctx.say("First I will cancel your unshipped order (mouse and cable). Proceed?")
    yield ctx.op("cancel_order", by_product["mouse"]["order_id"])
    if try_foreign_cancel:
        # An order the Customer never asked about must not be touched.
        yield ctx.op("cancel_order", next(o["id"] for o in orders if status(o) == "delivered"))
    yield ctx.op("list_exchange_options", by_product["laptop"]["id"])
    silver = next(r for r in ctx.data()["options"] if r["attributes"].get("color") == "silver")
    yield ctx.say("Next, exchange one laptop for the silver variant with a 10.00 extra charge. Proceed?")
    yield ctx.op("exchange_delivered_order_items", by_product["laptop"]["id"], quantity=1,
                 target_sku_id=silver["sku_id"])
    if not skip_return:
        yield ctx.say("Last, the monitor must be returned before its 300.00 refund. Accept the return?")
        yield ctx.op("request_item_return", by_product["monitor"]["id"], quantity=1)
        yield ctx.op("return_delivered_order_items", by_product["monitor"]["id"], quantity=1,
                     payment_method="original")
    yield ctx.say("All requested actions are complete.")


def mixed_case():
    world, aliases = pending_world()
    task = v3_task(world, aliases["customer"], remedies=(
        remedy(aliases["laptop_line"], ("exchange",), goal_id="swap", extra=2000),
        remedy(aliases["monitor_line"], ("refund",), goal_id="send-back")),
        cancellations=(cancel_goal(world, aliases["pending"]),))
    return world, aliases, task


def test_mixed_multi_business_episode_passes_every_goal_independently():
    world, aliases, task = mixed_case()
    record = run(world, task, ScriptedService(mixed_script), YesCustomer(3),
                 events=(monitor_event(world, aliases),))
    statuses = {g.goal_id: g.status for g in record.outcome.state_goals}
    assert statuses == {"swap": "PASS", "send-back": "PASS", "cancel-goal": "PASS"}
    assert record.outcome.business_goal_success is True and verified_reward(record) == 1.0
    assert [r.status for r in evaluate_task_compliance(world, record.final_world, task)] == ["PASS"]
    assert not invariants(record.final_world)


def test_missing_one_business_goal_fails_the_whole_task():
    world, aliases, task = mixed_case()
    record = run(world, task, ScriptedService(lambda c: mixed_script(c, skip_return=True)), YesCustomer(2),
                 events=(monitor_event(world, aliases),))
    statuses = {g.goal_id: g.status for g in record.outcome.state_goals}
    assert statuses == {"swap": "PASS", "send-back": "FAIL", "cancel-goal": "PASS"}
    assert record.outcome.business_goal_success is False and verified_reward(record) == 0.0


def test_actions_outside_the_requested_scope_are_refused_without_state_change():
    world, aliases, task = mixed_case()
    record = run(world, task, ScriptedService(lambda c: mixed_script(c, try_foreign_cancel=True)),
                 YesCustomer(3), events=(monitor_event(world, aliases),))
    foreign = [t for t in record.tool_executions if t.operation["name"] == "cancel_order"
               and t.operation["resource_id"] == aliases["delivered"]]
    assert foreign and foreign[0].result["error_code"] == "BUSINESS_DELEGATION_REQUIRED"
    assert foreign[0].before_state_hash == foreign[0].after_state_hash
    assert record.outcome.business_goal_success is True
    # the refused attempt is still a recorded process error, so strict task success is withheld
    assert record.outcome.strict_task_success is False


def test_replay_detects_a_tampered_business_result():
    world, aliases = pending_world()
    task = v3_task(world, aliases["customer"], cancellations=(cancel_goal(world, aliases["pending"]),))

    def script(ctx):
        yield ctx.op("list_customer_orders")
        yield ctx.say("Cancel the unshipped order?")
        yield ctx.op("cancel_order", aliases["pending"])
        yield ctx.say("Cancelled.")

    record = run(world, task, ScriptedService(script), YesCustomer(1))
    assert replay_episode(record).verified
    forged = json.loads(record.model_dump_json())
    forged["final_world"]["ledger"] = {}
    from usami_v2.runtime.runner import EpisodeRecord
    # The sealed record hash rejects the edit outright ...
    with pytest.raises(ValueError):
        EpisodeRecord.model_validate(forged)
    # ... and even a re-sealed forgery cannot get past Replay of the real Engine trace.
    forged.pop("record_hash")
    forged["outcome"].pop("record_hash")
    resealed = EpisodeRecord.model_validate(forged)
    assert not replay_episode(resealed).verified


def test_independent_goal_evaluation_ignores_what_the_service_claims():
    world, aliases = pending_world()
    task = v3_task(world, aliases["customer"], cancellations=(cancel_goal(world, aliases["pending"]),))
    assert [(g.goal_id, g.status) for g in evaluate_state_goals(world, world, task)] == [("cancel-goal", "FAIL")]
    done, _, _ = execute(delegate(world, aliases["customer"], lines_of(world, aliases["pending"])),
                         aliases["customer"], "cancel_order", aliases["pending"])
    assert [(g.goal_id, g.status) for g in evaluate_state_goals(world, done, task)] == [("cancel-goal", "PASS")]



# ---- Designer -> grounding -> Host certification -> rollout -----------------

from rift.benchmark.usami import CommerceHost, ground, compile_candidate
from rift.replay.splits import SplitRegistry
from rift.schemas import Identity, Split
from rift.tasks import CertificationStatus, Proposal
from usami_v2.scenarios.designer import ScenarioDraft, TaskCapabilityUnsupported, TaskDraft, compile_task
from usami_v2.scenarios.schema import OrderVariable
from rift_usami_fixtures import distribution_and_slot


def designer_spec(distribution):
    spec = base_spec()
    orders = []
    for order in spec.orders:
        data = order.model_dump(mode="json", exclude={"record_hash"})
        if order.alias == "pending":
            data["fulfillment_status"] = "unshipped"
        orders.append(OrderVariable.model_validate(data))
    draft = ScenarioDraft(
        scenario_id="phase3-designer", customer_aliases=spec.customer_aliases, skus=spec.skus,
        warehouses=spec.warehouses, orders=tuple(orders), stock=spec.stock, default_stock=spec.default_stock)
    return draft.compile(distribution)


def host_for(distribution, build, spec, customer, **kwargs):
    return CommerceHost(
        [("world", spec, build)], distribution, None, None, customer, registry=SplitRegistry(),
        identity_factory=lambda task, s, b: Identity(b.world_hash, task.record_hash, "family", "lineage", Split.TRAIN),
        proof_budget=ProofBudget(),
        episode_budget=EpisodeBudget(max_customer_turns=10, max_service_decisions=40,
                                     max_service_tool_calls=24, max_model_calls=0), **kwargs)


def challenge(data):
    base = {"task_id": "phase3-challenge", "customer_alias": "customer", "hypothesis": "phase3"}
    return Proposal("p", "phase3", "request", json.dumps({**base, **data}))


def test_designer_world_reserves_real_stock_for_an_unshipped_order():
    distribution, _ = distribution_and_slot(13)
    spec = designer_spec(distribution)
    build = build_world(spec, 13, BuildBudget(max_nodes=64))
    assert build.status == "SAT"
    world, aliases = build.world, dict(build.private_aliases)
    order = world.orders[aliases["pending"]]
    assert order.fulfillment_status == "unshipped" and order.delivered_at > world.clock
    line = world.lines[aliases["mouse_line"]]
    held = world.inventory[line.reserved_inventory_id]
    assert held.reserved == line.quantity and held.opening_quantity == 4 + line.quantity
    assert not invariants(world)
    delivered = world.orders[aliases["delivered"]]
    assert delivered.fulfillment_status == "delivered"


def test_cancellation_challenge_grounds_compiles_certifies_and_rolls_out():
    distribution, _ = distribution_and_slot(13)
    spec = designer_spec(distribution)
    build = build_world(spec, 13, BuildBudget(max_nodes=64))
    data = {"cancellation_goals": [{"goal_id": "c", "line_alias": "item"}],
            "line_selectors": [{"alias": "item", "fulfillment_status": "unshipped"}]}
    host = host_for(distribution, build, spec, YesCustomer(1))
    proposal = challenge(data)
    assert ground(proposal, [("world", spec, build)]).status == "GROUNDED"
    cert, task = host.certify_proposal(proposal)
    assert cert.status is CertificationStatus.CERTIFIED and cert.task_ref == task.record_hash
    (goal,) = task.cancellation_goals
    assert goal.order_id == build.world.lines[goal.line_ids[0]].order_id
    assert "cancel my order" in task.customer_request

    def script(ctx):
        yield from gather(ctx)
        order = next(row for row in ctx.orders if status(row) == "unshipped")
        yield ctx.say("I can cancel your unshipped order and refund 34.00. Shall I?")
        yield ctx.op("cancel_order", order["id"])
        yield ctx.say("Cancelled and refunded to the original card.")

    record = host.rollout(task, ScriptedService(script))
    assert public_episode(record).outcome is Outcome.SUCCESS and verified_reward(record) == 1.0


def test_exchange_and_reship_challenges_ground_and_certify_through_the_host():
    distribution, _ = distribution_and_slot(13)
    spec = designer_spec(distribution)
    build = build_world(spec, 13, BuildBudget(max_nodes=64))
    for remedies, extra in ((["exchange"], 2000), (["reship"], None)):
        goal = {"goal_id": "g", "line_alias": "item", "quantity": 1, "allowed_remedies": remedies}
        if extra is not None:
            goal["maximum_extra_charge_cents"] = extra
        data = {"remedy_goals": [goal],
                "line_selectors": [{"alias": "item", "family": "laptop", "issue": "damaged"}]}
        cert, task = host_for(distribution, build, spec, YesCustomer(1)).certify_proposal(challenge(data))
        assert cert.status is CertificationStatus.CERTIFIED, remedies


def test_physical_return_challenge_needs_a_configured_receipt_event():
    distribution, _ = distribution_and_slot(13)
    spec = designer_spec(distribution)
    build = build_world(spec, 13, BuildBudget(max_nodes=64))
    data = {"remedy_goals": [{"goal_id": "g", "line_alias": "item", "quantity": 1,
                              "allowed_remedies": ["refund"]}],
            "line_selectors": [{"alias": "item", "return_required": True}]}
    cert, task = host_for(distribution, build, spec, YesCustomer(1)).certify_proposal(challenge(data))
    assert cert.status is CertificationStatus.UNKNOWN and task is None
    host = host_for(distribution, build, spec, YesCustomer(1), return_receipt_after_days=3)
    cert, task = host.certify_proposal(challenge(data))
    assert cert.status is CertificationStatus.CERTIFIED
    record = host.task_registry.resolve_record(task.record_hash)
    (event,) = record.external_events
    assert event.kind == "return_received" and event.at == build.world.clock + timedelta(days=3)

    def script(ctx):
        yield from gather(ctx)
        line = product(ctx, "monitor")
        yield ctx.say("The monitor has to be sent back before the 300.00 refund. Accept the return?")
        yield ctx.op("request_item_return", line["id"], quantity=1)
        notice = ctx.notices()[-1]
        yield ctx.op("return_delivered_order_items", notice["line_id"], quantity=notice["quantity"],
                     payment_method="original")
        yield ctx.say("Received and refunded to the original card.")

    episode = host.rollout(task, ScriptedService(script))
    assert episode.outcome.business_goal_success is True and verified_reward(episode) == 1.0
    assert episode.final_world.clock == build.world.clock + timedelta(days=3)


def test_task_designer_cannot_register_unsupported_return_capability():
    distribution, _ = distribution_and_slot(13)
    spec = designer_spec(distribution)
    build = build_world(spec, 13, BuildBudget(max_nodes=64))
    aliases = dict(build.private_aliases)
    draft = TaskDraft(task_id="return-draft", customer_alias="customer", remedy_goals=(
        {"goal_id": "g", "line_alias": "monitor_line", "quantity": 1, "allowed_remedies": ["refund"]},))
    with pytest.raises(TaskCapabilityUnsupported) as excinfo:
        compile_task(draft, spec, build, distribution)
    assert excinfo.value.code == "RETURN_RECEIPT_UNAVAILABLE"
    from usami_v2.scenarios.designer import task_design_capabilities
    capabilities = task_design_capabilities(spec, build, return_receipt_event_configured=True)
    assert compile_task(draft, spec, build, distribution, capabilities=capabilities).remedy_goals


# ---- historical compatibility ----------------------------------------------

def test_pre_phase3_records_keep_their_exact_shape_and_hashes():
    from usami_v2.scenarios.tasks import parse_task_spec
    world, aliases = built()
    dumped = json.loads(world.model_dump_json())
    assert all("fulfillment_status" not in row for row in dumped["orders"].values())
    assert all("reserved_inventory_id" not in row for row in dumped["lines"].values())
    task = v3_task(world, aliases["customer"], remedies=(remedy(aliases["laptop_line"], ("reship",), goal_id="g"),))
    raw = json.loads(task.model_dump_json())
    assert "cancellation_goals" not in raw
    assert parse_task_spec(raw) == task and parse_task_spec(raw).record_hash == task.record_hash
    # a Phase 3 intent hash differs only when a cancellation goal is actually present
    goals = task.remedy_goals
    assert request_intent_hash(task.customer_id, goals, ()) == request_intent_hash(task.customer_id, goals, (), ())
    assert request_intent_hash(task.customer_id, goals, ()) != request_intent_hash(
        task.customer_id, goals, (), (cancel_goal(world, aliases["delivered"]),))


def test_cancellation_goals_require_the_structured_v3_protocol():
    from usami_v2.scenarios.tasks import TaskSpec
    world, aliases = built()
    with pytest.raises(ValueError):
        TaskSpec(task_id="old", world_hash=world.state_hash, customer_id=aliases["customer"],
                 customer_request="Please cancel.", partition="train",
                 source_distribution_hash=digest("x"),
                 cancellation_goals=(cancel_goal(world, aliases["pending"]),))
