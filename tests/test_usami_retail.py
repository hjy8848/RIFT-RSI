"""Retail extension: addresses, payment methods and one-shot item changes on unshipped orders.

Every call runs the real Engine, the independent Reference, the Oracle and a trace Replay; the actual
transition and the Reference must agree on state and result.
"""
from datetime import datetime, timedelta, timezone

import pytest

from usami_harness.engine import Engine, replay_trace
from usami_harness.models import Request, digest
from usami_harness.oracle import evaluate
from usami_v2.commerce import adapter
from usami_v2.commerce.invariants import invariants
from usami_v2.commerce.models import (
    Address, Customer, Delegation, DeliveryRoute, Inventory, Merchant, Operation, Order, OrderLine,
    PaymentMethod, SKU, Warehouse, World,
)
from usami_v2.commerce.reference import decide
from usami_v2.commerce.transitions import apply

CLOCK = datetime(2026, 1, 15, 12, tzinfo=timezone.utc)
HOME = {"street": "1 Home Rd", "city": "Denver", "state": "CO", "postal_code": "80201"}
AWAY = {"street": "9 Away St", "city": "Austin", "state": "TX", "postal_code": "73301"}


def gift(balance):
    return PaymentMethod(id="gc", customer_id="buyer", kind="gift_card", balance_cents=balance,
                         opening_balance_cents=balance)


def world(balance=5000):
    skus = {
        "shoe_a": SKU(id="shoe_a", merchant_id="shop", product_family="shoe", attributes={"color": "red"}, price_cents=5000),
        "shoe_b": SKU(id="shoe_b", merchant_id="shop", product_family="shoe", attributes={"color": "blue"}, price_cents=6000),
        "shoe_c": SKU(id="shoe_c", merchant_id="shop", product_family="shoe", attributes={"color": "green"}, price_cents=4000),
        "shoe_d": SKU(id="shoe_d", merchant_id="shop", product_family="shoe", attributes={"color": "black"}, price_cents=9000),
        "hat": SKU(id="hat", merchant_id="shop", product_family="hat", attributes={"color": "red"}, price_cents=2000),
        "hat_b": SKU(id="hat_b", merchant_id="shop", product_family="hat", attributes={"color": "blue"}, price_cents=2500),
    }
    stock = {"shoe_a": 5, "shoe_b": 5, "shoe_c": 5, "shoe_d": 0, "hat": 5, "hat_b": 5}
    inventory = {key + "_w1": Inventory(id=key + "_w1", sku_id=key, warehouse_id="w1",
                                        opening_quantity=amount, available=amount)
                 for key, amount in stock.items()}
    for sku, qty in (("shoe_a", 1), ("hat", 1)):                       # lines of the unshipped order hold stock
        inventory[sku + "_w1"].reserved += qty
        inventory[sku + "_w1"].opening_quantity += qty
    result = World(
        seed=1, clock=CLOCK,
        customers={"buyer": Customer(id="buyer", name="Buyer", default_address=Address(**HOME)),
                   "other": Customer(id="other", name="Other")},
        merchants={"shop": Merchant(id="shop", refund_approval_limit_cents=50000)},
        skus=skus, warehouses={"w1": Warehouse(id="w1", merchant_id="shop")}, inventory=inventory,
        routes={"route_w1": DeliveryRoute(id="route_w1", warehouse_id="w1", days=2)},
        orders={
            "pending": Order(id="pending", merchant_id="shop", customer_id="buyer", paid_cents=7000,
                             payment_channel="cc", payment_method_id="cc", delivered_at=CLOCK + timedelta(days=3),
                             fulfillment_status="unshipped", shipping_address=Address(**HOME)),
            "done": Order(id="done", merchant_id="shop", customer_id="buyer", paid_cents=5000,
                          payment_channel="gc", payment_method_id="gc", delivered_at=CLOCK - timedelta(days=2),
                          shipping_address=Address(**AWAY)),
            "foreign": Order(id="foreign", merchant_id="shop", customer_id="other", paid_cents=2000,
                             payment_channel="original-card", delivered_at=CLOCK + timedelta(days=3),
                             fulfillment_status="unshipped"),
        },
        lines={
            "L1": OrderLine(id="L1", order_id="pending", sku_id="shoe_a", quantity=1, allocated_paid_cents=5000,
                            reserved_inventory_id="shoe_a_w1"),
            "L2": OrderLine(id="L2", order_id="pending", sku_id="hat", quantity=1, allocated_paid_cents=2000,
                            reserved_inventory_id="hat_w1"),
            "L3": OrderLine(id="L3", order_id="done", sku_id="shoe_a", quantity=1, allocated_paid_cents=5000),
            "L4": OrderLine(id="L4", order_id="foreign", sku_id="hat", quantity=1, allocated_paid_cents=2000),
        },
        payment_methods={"cc": PaymentMethod(id="cc", customer_id="buyer", kind="credit_card"),
                         "pp": PaymentMethod(id="pp", customer_id="buyer", kind="paypal"),
                         "gc": gift(balance)},
        delegations={"d": Delegation(id="d", customer_id="buyer", line_ids=("L1", "L2", "L3"),
                                     allowed_preparations=(), expires_at=CLOCK + timedelta(days=1),
                                     source_customer_event_hash=digest("retail fixture"))})
    # the reserved stock above belongs to the two unshipped lines; the foreign line holds none
    assert invariants(result) == [], invariants(result)
    return result


def run(value, name, resource, params=None, actor="buyer"):
    """Engine + Oracle + Replay, and the actual transition must equal the Reference decision."""
    operation = Operation(name=name, scope_id="shop", resource_id=resource, params=params or {})
    decision = decide(value, actor, operation)
    actual, outcome = apply(value, actor, operation, 0)
    if decision.allowed:
        assert outcome.success and actual == decision.expected_state and outcome.data == decision.data
    elif decision.error_code != "BUSINESS_DELEGATION_REQUIRED":        # authority is the Reference's job alone
        assert not outcome.success and outcome.error_code == decision.error_code
        assert actual == value
    request = Request(id="retail-request", identity=actor, operation=operation.model_dump(mode="json"), granted=True)
    engine = Engine(adapter, value, operation_budget=1)
    engine.register(request)
    event = engine.execute(actor, operation, request.id, "reference")
    trace = engine.finish()
    report = evaluate(adapter, trace, {request.id: request})
    assert report["trace_valid"]
    assert replay_trace(adapter, trace, {request.id: request}).state == engine.state
    assert invariants(engine.state) == [], invariants(engine.state)
    return engine.state, event.payload["result"]


def test_legacy_worlds_keep_their_hash_when_no_retail_field_is_used():
    legacy = World(seed=1, clock=CLOCK, customers={"c": Customer(id="c", name="C")},
                   merchants={}, skus={}, warehouses={}, inventory={}, routes={}, orders={}, lines={})
    dump = legacy.model_dump(mode="json")
    assert not {"payment_methods", "modifications"} & set(dump)
    assert dump["customers"]["c"] == {"id": "c", "name": "C"}
    order = Order(id="o", merchant_id="m", customer_id="c", paid_cents=1, payment_channel="x", delivered_at=CLOCK)
    assert set(order.model_dump(mode="json")) == {"id", "merchant_id", "customer_id", "paid_cents",
                                                  "payment_channel", "delivered_at"}


# --------------------------------------------------------------------------- addresses

def test_unshipped_order_address_can_change_and_shows_in_order_details():
    after, result = run(world(), "modify_order_address", "pending", AWAY)
    assert result["success"] and after.orders["pending"].shipping_address == Address(**AWAY)
    _, details = run(after, "get_order_details", "pending")
    assert details["data"]["order"]["shipping_address"] == AWAY


@pytest.mark.parametrize("order", ["done"])
def test_delivered_order_address_is_refused(order):
    value, result = run(world(), "modify_order_address", order, AWAY)
    assert result["error_code"] == "ORDER_NOT_MODIFIABLE" and value == world()


def test_foreign_order_and_missing_delegation_are_refused():
    _, result = run(world(), "modify_order_address", "foreign", AWAY)
    assert result["error_code"] == "OBJECT_ACCESS_DENIED"
    undelegated = world()
    undelegated.delegations.clear()
    _, result = run(undelegated, "modify_order_address", "pending", AWAY)
    assert result["error_code"] == "BUSINESS_DELEGATION_REQUIRED"


def test_cancelled_order_address_is_refused():
    cancelled, result = run(world(), "cancel_order", "pending")
    assert result["success"]
    _, result = run(cancelled, "modify_order_address", "pending", AWAY)
    assert result["error_code"] == "ORDER_ALREADY_CANCELLED"


def test_default_address_updates_and_profile_lists_methods_with_balances():
    after, result = run(world(), "update_default_address", "shop", AWAY)
    assert result["success"] and after.customers["buyer"].default_address == Address(**AWAY)
    _, profile = run(after, "get_customer_profile", "shop")
    assert profile["data"]["customer"]["default_address"] == AWAY
    assert profile["data"]["payment_methods"] == [{"id": "cc", "kind": "credit_card"},
                                                  {"id": "gc", "kind": "gift_card", "balance_cents": 5000},
                                                  {"id": "pp", "kind": "paypal"}]
    other, _ = run(world(), "get_customer_profile", "shop", actor="other")
    assert other == world()


# --------------------------------------------------------------------------- payment methods

def test_refund_to_a_gift_card_is_immediate_and_credits_the_balance():
    after, result = run(world(), "return_delivered_order_items", "L3", {"quantity": 1, "payment_method": "gc"})
    assert result["success"] and result["data"]["refund_timing"] == "immediate"
    assert after.payment_methods["gc"].balance_cents == 10000


def test_refund_to_another_card_or_paypal_or_unknown_method_is_refused():
    for method, code in (("pp", "REFUND_DESTINATION_NOT_ALLOWED"), ("cc", "REFUND_DESTINATION_NOT_ALLOWED"),
                         ("nope", "PAYMENT_METHOD_NOT_FOUND")):
        _, result = run(world(), "return_delivered_order_items", "L3", {"quantity": 1, "payment_method": method})
        assert result["error_code"] == code, (method, result)


def test_original_card_refund_takes_days_and_leaves_gift_card_alone():
    after, result = run(world(), "cancel_order", "pending")
    assert result["data"]["refund_timing"] == "5-7 business days"
    assert after.payment_methods["gc"].balance_cents == 5000


def test_exchange_charge_needs_a_gift_card_that_can_cover_it():
    _, poor = run(world(balance=500), "exchange_delivered_order_items", "L3",
                  {"quantity": 1, "target_sku_id": "shoe_b", "payment_method": "gc"})
    assert poor["error_code"] == "INSUFFICIENT_BALANCE"
    after, rich = run(world(), "exchange_delivered_order_items", "L3",
                      {"quantity": 1, "target_sku_id": "shoe_b", "payment_method": "pp"})
    assert rich["success"] and rich["data"]["payment_channel"] == "pp" and after.payment_methods["gc"].balance_cents == 5000
    paid, ok = run(world(), "exchange_delivered_order_items", "L3",
                   {"quantity": 1, "target_sku_id": "shoe_b", "payment_method": "gc"})
    assert ok["success"] and paid.payment_methods["gc"].balance_cents == 4000


def test_a_gift_card_that_paid_the_order_must_also_cover_its_own_original_charge():
    _, result = run(world(balance=100), "exchange_delivered_order_items", "L3",
                    {"quantity": 1, "target_sku_id": "shoe_b"})
    assert result["error_code"] == "INSUFFICIENT_BALANCE"


def test_worlds_without_payment_methods_still_reject_other_methods():
    legacy = world()
    legacy.payment_methods.clear()
    for order in legacy.orders.values():
        order.payment_method_id = None
    _, result = run(legacy, "return_delivered_order_items", "L3", {"quantity": 1, "payment_method": "gc"})
    assert result["error_code"] == "PAYMENT_METHOD_UNSUPPORTED"


# --------------------------------------------------------------------------- one-shot item changes

def change(*pairs, method="original"):
    return {"changes": [{"line_id": line, "target_sku_id": sku} for line, sku in pairs], "payment_method": method}


def test_item_change_swaps_stock_settles_the_difference_and_locks_the_order():
    after, result = run(world(), "modify_order_items", "pending", change(("L1", "shoe_b"), ("L2", "hat_b")))
    assert result["success"], result
    assert result["data"]["extra_charge_cents"] == 1500 and result["data"]["new_total_cents"] == 8500
    assert after.orders["pending"].items_locked and after.orders["pending"].paid_cents == 8500
    assert after.inventory["shoe_a_w1"].reserved == 0 and after.inventory["shoe_b_w1"].reserved == 1
    assert after.lines["L1"].reserved_inventory_id == "shoe_b_w1" and after.lines["L1"].sku_id == "shoe_b"


def test_second_change_and_cancellation_of_a_modified_order_are_refused():
    once, result = run(world(), "modify_order_items", "pending", change(("L1", "shoe_b")))
    assert result["success"]
    _, again = run(once, "modify_order_items", "pending", change(("L2", "hat_b")))
    assert again["error_code"] == "ORDER_LOCKED"
    _, cancel = run(once, "cancel_order", "pending")
    assert cancel["error_code"] == "ORDER_LOCKED"


def test_cheaper_item_refunds_to_the_gift_card_immediately():
    after, result = run(world(), "modify_order_items", "pending", change(("L1", "shoe_c"), method="gc"))
    assert result["data"]["refund_cents"] == 1000 and result["data"]["refund_timing"] == "immediate"
    assert after.payment_methods["gc"].balance_cents == 6000 and after.orders["pending"].paid_cents == 6000


def test_difference_can_be_charged_to_a_gift_card_only_when_it_covers_it():
    _, poor = run(world(balance=500), "modify_order_items", "pending", change(("L1", "shoe_b"), method="gc"))
    assert poor["error_code"] == "INSUFFICIENT_BALANCE"
    after, rich = run(world(), "modify_order_items", "pending", change(("L1", "shoe_b"), method="gc"))
    assert rich["success"] and after.payment_methods["gc"].balance_cents == 4000


@pytest.mark.parametrize("order,changes,code", [
    ("done", change(("L3", "shoe_b")), "ORDER_NOT_MODIFIABLE"),
    ("pending", change(("L1", "hat_b")), "REPLACEMENT_FAMILY_MISMATCH"),
    ("pending", change(("L1", "shoe_a")), "EXCHANGE_TARGET_SAME_AS_ORIGINAL"),
    ("pending", change(("L1", "shoe_d")), "INSUFFICIENT_STOCK"),
    ("pending", change(("L1", "ghost")), "EXCHANGE_TARGET_NOT_FOUND"),
    ("pending", change(("L1", "shoe_b"), ("L1", "shoe_c")), "DUPLICATE_LINE_CHANGE"),
    ("pending", change(("L3", "shoe_b")), "LINE_NOT_IN_ORDER"),
    ("pending", change(("L1", "shoe_b"), method="ghost"), "PAYMENT_METHOD_NOT_FOUND"),
    ("pending", change(("L1", "shoe_c"), method="pp"), "REFUND_DESTINATION_NOT_ALLOWED"),
])
def test_item_change_refusals(order, changes, code):
    value, result = run(world(), "modify_order_items", order, changes)
    assert result["error_code"] == code, result
    assert value == world()


def test_two_lines_competing_for_the_last_unit_are_refused():
    tight = world()
    tight.inventory["shoe_b_w1"].available = 1
    tight.inventory["shoe_b_w1"].opening_quantity = 1
    tight.lines["L2"].sku_id = "shoe_a"
    tight.lines["L2"].reserved_inventory_id = "shoe_a_w1"
    tight.inventory["hat_w1"].reserved -= 1
    tight.inventory["hat_w1"].opening_quantity -= 1
    tight.inventory["shoe_a_w1"].reserved += 1
    tight.inventory["shoe_a_w1"].opening_quantity += 1
    assert invariants(tight) == []
    _, result = run(tight, "modify_order_items", "pending", change(("L1", "shoe_b"), ("L2", "shoe_b")))
    assert result["error_code"] == "INSUFFICIENT_STOCK"


def test_ordinary_tools_are_untouched_by_the_extension():
    after, result = run(world(), "return_delivered_order_items", "L3", {"quantity": 1, "payment_method": "original"})
    assert result["success"] and result["data"]["payment_channel"] == "gc"
    assert after.payment_methods["gc"].balance_cents == 10000
