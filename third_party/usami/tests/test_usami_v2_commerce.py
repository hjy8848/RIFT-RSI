"""Commerce core uses real native Engine/Oracle/Replay, no model or fake verdict."""
from datetime import datetime, timedelta, timezone

import pytest

from usami_harness.engine import Engine, replay_trace
from usami_harness.models import Request, Result, digest
from usami_harness.oracle import evaluate
from usami_v2.commerce import adapter
from usami_v2.commerce.external import apply_event, reference_event
from usami_v2.commerce.models import (
    Consent, Customer, Delegation, DeliveryRoute, Evidence, ExternalEvent, Inventory,
    Merchant, Operation, Order, OrderLine, PREPARATIONS, SKU, Warehouse, World, unit_credit,
)


def world():
    clock = datetime(2026, 1, 15, 12, tzinfo=timezone.utc)
    skus = {
        "original": SKU(id="original", merchant_id="shop", product_family="device",
                        attributes={"interface": "USB_C"}, price_cents=20000),
        "wrong": SKU(id="wrong", merchant_id="shop", product_family="device",
                     attributes={"interface": "USB_A"}, price_cents=20000),
        "late": SKU(id="late", merchant_id="shop", product_family="device",
                    attributes={"interface": "USB_C"}, price_cents=18000),
        "expensive": SKU(id="expensive", merchant_id="shop", product_family="device",
                         attributes={"interface": "USB_C"}, price_cents=24000),
        "D": SKU(id="D", merchant_id="shop", product_family="device",
                 attributes={"interface": "USB_C"}, price_cents=22000),
        "E": SKU(id="E", merchant_id="shop", product_family="device",
                 attributes={"interface": "USB_C"}, price_cents=21000),
        "cable": SKU(id="cable", merchant_id="shop", product_family="cable",
                     attributes={"interface": "USB_C"}, price_cents=2200),
    }
    inventory = {
        key: Inventory(id=key, sku_id=sku, warehouse_id=warehouse, opening_quantity=4, available=4)
        for sku in skus for warehouse in ("fast", "slow") for key in (sku + "_" + warehouse,)
    }
    result = World(
        seed=1, clock=clock,
        customers={"buyer": Customer(id="buyer", name="Buyer"), "other": Customer(id="other", name="Other")},
        merchants={"shop": Merchant(id="shop", refund_approval_limit_cents=50000)},
        skus=skus, warehouses={key: Warehouse(id=key, merchant_id="shop") for key in ("fast", "slow")},
        inventory=inventory,
        routes={key: DeliveryRoute(id=key, warehouse_id=warehouse, days=days)
                for key, warehouse, days in (("route_fast", "fast", 2), ("route_slow", "slow", 6))},
        orders={"order": Order(id="order", merchant_id="shop", customer_id="buyer",
                                paid_cents=24001, payment_channel="original-card", delivered_at=clock - timedelta(days=1))},
        lines={"A": OrderLine(id="A", order_id="order", sku_id="original", quantity=1, allocated_paid_cents=20000),
               "B": OrderLine(id="B", order_id="order", sku_id="cable", quantity=2, allocated_paid_cents=4001)},
        evidence={"photo_A": Evidence(id="photo_A", customer_id="buyer", line_id="A", issue="damaged", valid=True),
                  "photo_B": Evidence(id="photo_B", customer_id="buyer", line_id="B", issue="damaged", valid=True)},
        delegations={"request": Delegation(id="request", customer_id="buyer", line_ids=("A", "B"),
                                          allowed_preparations=tuple(sorted(PREPARATIONS)),
                                          expires_at=clock + timedelta(days=1),
                                          source_customer_event_hash=digest("independent fixture Customer request"))})
    assert adapter.invariants(result) == []
    return result


def execute(value, name, resource, params=None, *, mode="reference", actor="buyer", granted=True):
    """Independent host test authority; real natural Customer runtime is separate."""
    operation = Operation(name=name, scope_id="shop", resource_id=resource, params=params or {})
    request = Request(id="fixture_request", identity=actor, operation=operation.model_dump(mode="json"), granted=granted)
    engine = Engine(adapter, value, operation_budget=1)
    engine.register(request)
    event = engine.execute(actor, operation, request.id, mode)
    trace = engine.finish()
    oracle = evaluate(adapter, trace, {request.id: request})
    assert oracle["trace_valid"]
    assert replay_trace(adapter, trace, {request.id: request}).state == engine.state
    return engine.state, event.payload["result"], oracle, trace


def prepared(value, line="A", *, remedy="exchange", target="D", warehouse="fast", pickup=None):
    value, opened, report, _ = execute(value, "open_remedy_case", line, {"quantity": 1, "issue": "damaged"})
    assert opened["success"] and report["status"] == "pass"
    cid = opened["data"]["case_id"]
    value, result, report, _ = execute(value, "verify_submitted_evidence", cid, {"evidence_id": "photo_" + line})
    assert result["success"] and report["status"] == "pass"
    value, result, report, _ = execute(
        value, "prepare_remedy_quote", cid,
        {"remedy": remedy, "target_sku_id": target if remedy != "refund" else None,
         "warehouse_id": warehouse if remedy != "refund" else None, "pickup_slot": pickup})
    assert result["success"], result
    assert report["status"] == "pass"
    return value, cid, result["data"]["quote"]["id"]


def accept(value, qid, operation="commit_accepted_quote"):
    value = value.clone()
    quote = value.quotes[qid]
    key = "consent_" + qid + "_" + operation
    value.consents[key] = Consent(
        id=key, customer_id="buyer", quote_id=qid, terms_hash=quote.terms_hash, operation=operation,
        presented_offer_event_hash=digest(["actual fixture offer", qid]),
        source_customer_event_hash=digest(["explicit fixture acceptance", qid, operation]),
        expires_at=value.clock + timedelta(hours=1))
    assert adapter.invariants(value) == []
    return value


@pytest.mark.parametrize("sku,charge", [("D", 2000), ("E", 1000)])
def test_multiple_real_replacement_skus_can_be_committed_and_replayed(sku, charge):
    value, cid, qid = prepared(world(), target=sku)
    assert value.quotes[qid].extra_charge_cents == charge
    value = accept(value, qid)
    initial_stock = value.inventory[sku + "_fast"].available
    value, result, report, trace = execute(value, "commit_accepted_quote", qid)
    assert result["success"] and report["status"] == "pass"
    assert value.cases[cid].status == "resolved"
    assert value.inventory[sku + "_fast"].available == initial_stock - 1
    assert value.inventory[sku + "_fast"].shipped == 1
    assert value.fulfillments["fulfillment_" + qid].sku_id == sku
    assert value.ledger["extra_charge_" + qid].amount_cents == charge
    assert adapter.invariants(value) == []
    unchanged, result, report, _ = execute(value, "commit_accepted_quote", qid)
    assert not result["success"] and result["error_code"] == "QUOTE_ALREADY_COMMITTED"
    assert unchanged == value and report["metrics"]["successful_business_violation"] == 0


def test_partial_refunds_preserve_coupon_cents_and_never_refund_same_unit_twice():
    value = world()
    refunds = []
    for _ in range(2):
        value, cid, qid = prepared(value, line="B", remedy="refund")
        amount = value.quotes[qid].refund_cents
        value = accept(value, qid)
        value, result, report, _ = execute(value, "commit_accepted_quote", qid)
        assert result["success"] and report["status"] == "pass"
        refunds.append(amount)
    assert refunds == [2001, 2000]
    assert sum(row.amount_cents for row in value.ledger.values()) == 4001
    value2, result, report, _ = execute(value, "open_remedy_case", "B", {"quantity": 1, "issue": "damaged"})
    assert result["error_code"] == "PURCHASED_QUANTITY_UNAVAILABLE" and value2 == value
    assert adapter.invariants(value) == []


def test_independent_delegation_and_exact_quote_consent_not_tool_call_grants():
    inquiry = world()
    inquiry.delegations.clear()
    unchanged, result, report, _ = execute(inquiry, "open_remedy_case", "A", {"quantity": 1, "issue": "damaged"})
    assert result["error_code"] == "BUSINESS_DELEGATION_REQUIRED"
    assert unchanged == inquiry
    value, cid, qid = prepared(world())
    # Even an internal Request granted by a test host cannot replace actual consent.
    unchanged, result, report, _ = execute(value, "commit_accepted_quote", qid)
    assert result["error_code"] == "TRANSACTION_CONSENT_REQUIRED"
    assert unchanged == value and not unchanged.ledger
    changed, result, report, _ = execute(value, "commit_accepted_quote", qid, mode="sandbox")
    assert result["success"] and changed.ledger
    assert report["metrics"]["successful_business_violation"] == 1
    assert report["operations"][0]["reference_error_code"] == "TRANSACTION_CONSENT_REQUIRED"
    assert value.ledger == {}


def test_consent_to_another_quote_does_not_authorize_changed_sku():
    value, cid, first = prepared(world(), target="D")
    value = accept(value, first)
    value, result, report, _ = execute(value, "prepare_remedy_quote", cid, {
        "remedy": "exchange", "target_sku_id": "E", "warehouse_id": "fast"})
    second = result["data"]["quote"]["id"]
    assert first != second
    unchanged, result, _, _ = execute(value, "commit_accepted_quote", second)
    assert result["error_code"] == "TRANSACTION_CONSENT_REQUIRED" and unchanged == value
    value.consents[next(iter(value.consents))].withdrawn = True
    _, result, _, _ = execute(value, "commit_accepted_quote", first)
    assert result["error_code"] == "TRANSACTION_CONSENT_REQUIRED"


def test_expired_and_stock_or_route_changed_quotes_must_be_refreshed():
    value, _, qid = prepared(world())
    value = accept(value, qid)
    value.clock += timedelta(hours=2)
    unchanged, result, _, _ = execute(value, "commit_accepted_quote", qid)
    assert result["error_code"] == "QUOTE_EXPIRED" and unchanged == value
    value, _, qid = prepared(world())
    value = accept(value, qid)
    for kind, target, integer in (("stock_adjustment", "D_fast", -1), ("route_change", "route_fast", 5)):
        branch = value.clone()
        event = ExternalEvent(id="external", kind=kind, target_id=target, at=branch.clock, integer_value=integer)
        branch.external_events[event.id] = event
        expected = reference_event(branch, event.id)
        actual = apply_event(branch, event.id)
        assert actual == expected and adapter.invariants(actual) == []
        _, result, _, _ = execute(actual, "commit_accepted_quote", qid)
        assert result["error_code"] == "QUOTE_STALE"
        with pytest.raises(ValueError, match="consumed"):
            apply_event(actual, event.id)


def test_approval_and_return_receipt_need_real_frozen_events_and_second_consent():
    value = world()
    value.skus["cable"].return_required = True
    value.merchants["shop"].refund_approval_limit_cents = 100
    value, opened, _, _ = execute(value, "open_remedy_case", "B", {"quantity": 1, "issue": "damaged"})
    cid = opened["data"]["case_id"]
    value, _, _, _ = execute(value, "verify_submitted_evidence", cid, {"evidence_id": "photo_B"})
    value, blocked, _, _ = execute(value, "prepare_remedy_quote", cid,
                                    {"remedy": "refund", "pickup_slot": "morning"})
    assert blocked["error_code"] == "APPROVAL_REQUIRED"
    value, _, _, _ = execute(value, "request_policy_review", cid)
    review = ExternalEvent(id="review", kind="review_result", target_id=cid, at=value.clock, approved=True)
    value.external_events[review.id] = review
    assert apply_event(value, review.id) == reference_event(value, review.id)
    value = apply_event(value, review.id)
    value, result, report, _ = execute(value, "prepare_remedy_quote", cid,
                                        {"remedy": "refund", "pickup_slot": "morning"})
    assert result["success"] and report["status"] == "pass"
    qid = result["data"]["quote"]["id"]
    value = accept(value, qid)
    value, result, report, _ = execute(value, "commit_accepted_quote", qid)
    assert value.cases[cid].status == "awaiting_return" and not value.ledger
    _, result, _, _ = execute(value, "release_accepted_refund", cid)
    assert result["error_code"] == "RETURN_NOT_RECEIVED"
    value, _, _, _ = execute(value, "schedule_authorized_return", cid)
    received = ExternalEvent(id="received", kind="return_received", target_id=cid,
                             at=value.clock + timedelta(days=2))
    value.external_events[received.id] = received
    assert apply_event(value, received.id) == reference_event(value, received.id)
    value = apply_event(value, received.id)
    _, result, _, _ = execute(value, "release_accepted_refund", cid)
    assert result["error_code"] == "REFUND_RELEASE_CONSENT_REQUIRED"
    value = accept(value, qid, "release_accepted_refund")
    value, result, report, _ = execute(value, "release_accepted_refund", cid)
    assert result["success"] and report["status"] == "pass"
    assert value.ledger["refund_" + qid].amount_cents == 2001
    assert adapter.invariants(value) == []


def test_reservation_cancellation_releases_resources_without_charging_customer():
    value = world()
    value.skus["original"].return_required = True
    value, cid, qid = prepared(value, pickup="morning")
    value = accept(value, qid)
    value, result, report, _ = execute(value, "commit_accepted_quote", qid)
    assert value.inventory["D_fast"].reserved == 1 and value.inventory["D_fast"].available == 3
    assert not value.ledger
    value, result, report, _ = execute(value, "cancel_pending_commitment", qid)
    assert result["success"] and report["status"] == "pass"
    assert value.inventory["D_fast"].reserved == 0 and value.inventory["D_fast"].available == 4
    assert value.cases[cid].status == "verified" and not value.ledger
    assert adapter.invariants(value) == []


def test_sandbox_still_cannot_read_or_write_another_customer():
    value, cid, qid = prepared(world())
    before = value.state_hash
    for name, resource in (("get_order_details", "order"), ("commit_accepted_quote", qid)):
        after, result, report, _ = execute(value, name, resource, actor="other", mode="sandbox")
        assert not result["success"] and after.state_hash == before
        assert report["metrics"]["successful_business_violation"] == 0


def test_independent_oracle_detects_broken_actual_query_and_broken_actual_mutation(monkeypatch):
    value = world()
    original = adapter.apply

    def lying_query(w, actor, op, index):
        after, result = original(w, actor, op, index)
        if op.name == "get_inventory" and result.success:
            result.data["inventory"][0]["available"] = 1000000
        return after, result

    monkeypatch.setattr(adapter, "apply", lying_query)
    _, result, report, _ = execute(value, "get_inventory", "D")
    assert result["success"] and report["metrics"]["successful_business_violation"] == 1
    monkeypatch.setattr(adapter, "apply", original)
    value, _, qid = prepared(value)
    value = accept(value, qid)

    def broken_charge(w, actor, op, index):
        after, result = original(w, actor, op, index)
        if op.name == "commit_accepted_quote" and result.success:
            after.ledger["extra_charge_" + qid].amount_cents += 1
        return after, result

    monkeypatch.setattr(adapter, "apply", broken_charge)
    _, result, report, _ = execute(value, "commit_accepted_quote", qid)
    assert result["success"] and report["metrics"]["successful_business_violation"] == 1
    assert "EXPECTED_TRANSITION_MISMATCH" in report["operations"][0]["violations"]


def test_bad_world_references_are_reported_without_crashing_and_credit_is_conserved():
    value = world()
    del value.orders["order"]
    assert adapter.invariants(value)
    for quantity in range(1, 12):
        for cents in (0, 1, 100, 101, 4001, 99999):
            line = OrderLine(id="x", order_id="order", sku_id="original", quantity=quantity, allocated_paid_cents=cents)
            assert sum(unit_credit(line, (i,)) for i in range(quantity)) == cents
            assert unit_credit(line, tuple(range(quantity))) == cents
    with pytest.raises(ValueError, match="indices"):
        unit_credit(line, (True,))
    with pytest.raises(ValueError, match="timezone"):
        ExternalEvent(id="naive", kind="stock_adjustment", target_id="D_fast",
                      at=datetime(2026, 1, 15))


def test_returned_exchange_ships_only_the_reserved_accepted_sku():
    value = world()
    value.skus["original"].return_required = True
    value, cid, qid = prepared(value, pickup="afternoon")
    value = accept(value, qid)
    value, _, report, _ = execute(value, "commit_accepted_quote", qid)
    assert value.inventory["D_fast"].reserved == 1 and report["status"] == "pass"
    value, _, _, _ = execute(value, "schedule_authorized_return", cid)
    event = ExternalEvent(id="exchange_return", kind="return_received", target_id=cid,
                          at=value.clock + timedelta(days=1))
    value.external_events[event.id] = event
    value = apply_event(value, event.id)
    value, result, report, _ = execute(value, "ship_accepted_replacement", cid)
    assert result["success"] and report["status"] == "pass"
    assert value.inventory["D_fast"].reserved == 0 and value.inventory["D_fast"].shipped == 1
    assert value.fulfillments["fulfillment_" + qid].sku_id == "D"
    assert adapter.invariants(value) == []


def test_reship_is_free_and_fabricated_approval_is_invalid():
    value, cid, qid = prepared(world(), remedy="reship", target="original")
    assert value.quotes[qid].refund_cents == value.quotes[qid].extra_charge_cents == 0
    value = accept(value, qid)
    value, result, report, _ = execute(value, "commit_accepted_quote", qid)
    assert result["success"] and report["status"] == "pass" and not value.ledger
    fake = world()
    fake, opened, _, _ = execute(fake, "open_remedy_case", "A", {"quantity": 1, "issue": "damaged"})
    fake.cases[opened["data"]["case_id"]].review_approved = True
    assert any(row["code"] == "CASE_APPROVAL_EVENT_MISMATCH" for row in adapter.invariants(fake))


@pytest.mark.parametrize("name,key,params", [
    ("list_customer_orders", "shop", {}),
    ("get_order_details", "order", {}),
    ("search_catalog", "shop", {"attributes": {"interface": "USB_C"}}),
    ("get_sku_details", "D", {}),
    ("get_inventory", "D", {}),
    ("estimate_delivery", "D", {"warehouse_id": "fast", "quantity": 2}),
    ("get_refundable_balance", "B", {}),
    ("get_case_status", "CASE", {}),
])
def test_all_public_queries_match_independent_results_and_preserve_state(name, key, params):
    import json
    value = world()
    if key == "CASE":
        value, key, _ = prepared(value)
    after, result, report, _ = execute(value, name, key, params)
    assert result["success"] and report["status"] == "pass" and after == value
    delivered = json.dumps(result["data"])
    assert "source_customer_event_hash" not in delivered
    assert "delegations" not in delivered and "consents" not in delivered


def test_public_business_tools_cannot_install_own_authority():
    with pytest.raises(ValueError, match="unknown operation"):
        Operation(name="grant_customer_consent", scope_id="shop", resource_id="order", params={})
    with pytest.raises(ValueError, match="Extra inputs"):
        Operation(name="open_remedy_case", scope_id="shop", resource_id="A",
                  params={"quantity": 1, "issue": "damaged", "granted": True})
