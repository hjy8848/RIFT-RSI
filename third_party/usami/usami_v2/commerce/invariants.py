"""Structural accounting and referential invariants, separate from consent policy."""
from collections import Counter

from .models import Quote, unit_credit
from usami_v2.contracts import aware


def invariants(world):
    errors = []

    def bad(code, path):
        errors.append({"code": code, "path": path})

    collections = ("customers", "merchants", "skus", "warehouses", "inventory", "routes",
                   "orders", "lines", "evidence", "cases", "quotes", "delegations", "consents",
                   "returns", "ledger", "settlements", "fulfillments", "external_events")
    for collection in collections:
        for key, row in getattr(world, collection).items():
            if key != row.id:
                bad("ENTITY_KEY_MISMATCH", collection + "/" + key)
    for row in world.skus.values():
        if row.merchant_id not in world.merchants:
            bad("SKU_MERCHANT_MISSING", row.id)
    for row in world.warehouses.values():
        if row.merchant_id not in world.merchants:
            bad("WAREHOUSE_MERCHANT_MISSING", row.id)
    for row in world.routes.values():
        if row.warehouse_id not in world.warehouses:
            bad("ROUTE_WAREHOUSE_MISSING", row.id)
    for row in world.inventory.values():
        sku, warehouse = world.skus.get(row.sku_id), world.warehouses.get(row.warehouse_id)
        if not sku or not warehouse or sku.merchant_id != warehouse.merchant_id:
            bad("INVENTORY_SCOPE_INVALID", row.id)
        if row.opening_quantity + row.adjustments != row.available + row.reserved + row.shipped:
            bad("INVENTORY_QUANTITY_NOT_CONSERVED", row.id)
    pairs = [(row.sku_id, row.warehouse_id) for row in world.inventory.values()]
    if len(set(pairs)) != len(pairs):
        bad("DUPLICATE_INVENTORY_BALANCE", "inventory")
    if len({row.warehouse_id for row in world.routes.values()}) != len(world.routes):
        bad("AMBIGUOUS_DEFAULT_DELIVERY_ROUTE", "routes")
    for order in world.orders.values():
        if order.customer_id not in world.customers or order.merchant_id not in world.merchants:
            bad("ORDER_SCOPE_INVALID", order.id)
        if sum(row.allocated_paid_cents for row in world.lines.values() if row.order_id == order.id) != order.paid_cents:
            bad("ORDER_PAID_ALLOCATION_MISMATCH", order.id)
    for line in world.lines.values():
        order, sku = world.orders.get(line.order_id), world.skus.get(line.sku_id)
        if not order or not sku or sku.merchant_id != order.merchant_id:
            bad("LINE_SCOPE_INVALID", line.id)
    for evidence in world.evidence.values():
        line = world.lines.get(evidence.line_id)
        order = world.orders.get(line.order_id) if line else None
        if not order or order.customer_id != evidence.customer_id:
            bad("EVIDENCE_SCOPE_INVALID", evidence.id)
    used_units, active_units = {}, {}
    for case in world.cases.values():
        line = world.lines.get(case.line_id)
        try:
            if not line or not case.unit_indices:
                raise ValueError("missing case line or units")
            unit_credit(line, case.unit_indices)
        except ValueError:
            bad("CASE_UNITS_INVALID", case.id)
            continue
        if case.evidence_id is not None:
            evidence = world.evidence.get(case.evidence_id)
            if not evidence or evidence.line_id != case.line_id:
                bad("CASE_EVIDENCE_INVALID", case.id)
        approvals = [world.external_events[eid] for eid in world.applied_events
                     if eid in world.external_events and world.external_events[eid].kind == "review_result"
                     and world.external_events[eid].target_id == case.id]
        if case.review_approved != (bool(approvals) and approvals[-1].approved):
            bad("CASE_APPROVAL_EVENT_MISMATCH", case.id)
        if case.status not in ("resolved", "cancelled"):
            for index in case.unit_indices:
                key = (case.line_id, index)
                if key in active_units:
                    bad("PURCHASE_UNIT_IN_MULTIPLE_ACTIVE_CASES", case.id)
                active_units[key] = case.id
    for settlement in world.settlements.values():
        case, quote = world.cases.get(settlement.case_id), world.quotes.get(settlement.quote_id)
        if (not case or not quote or case.status != "resolved" or case.line_id != settlement.line_id
                or case.unit_indices != settlement.unit_indices or quote.case_id != case.id
                or quote.remedy != settlement.remedy or quote.status != "committed"):
            bad("SETTLEMENT_BINDING_INVALID", settlement.id)
        for index in settlement.unit_indices:
            key = (settlement.line_id, index)
            if key in used_units or key in active_units:
                bad("PURCHASE_UNIT_SETTLED_TWICE", settlement.id)
            used_units[key] = settlement.id
    for case in world.cases.values():
        matching = [row for row in world.settlements.values() if row.case_id == case.id]
        if (case.status == "resolved") != (len(matching) == 1):
            bad("CASE_SETTLEMENT_MISMATCH", case.id)
    for quote in world.quotes.values():
        case = world.cases.get(quote.case_id)
        line = world.lines.get(case.line_id) if case else None
        order = world.orders.get(line.order_id) if line else None
        if not case or not order or order.customer_id != quote.customer_id or len(case.unit_indices) != quote.quantity:
            bad("QUOTE_SCOPE_INVALID", quote.id)
        try:
            Quote.model_validate(quote.model_dump(mode="json"))
        except ValueError:
            bad("QUOTE_TERMS_HASH_INVALID", quote.id)
        if quote.refund_cents and quote.extra_charge_cents:
            bad("QUOTE_CONTRADICTORY_CASHFLOW", quote.id)
        if quote.remedy == "refund" and any((quote.target_sku_id, quote.inventory_id, quote.route_id, quote.promised_arrival)):
            bad("REFUND_HAS_REPLACEMENT_TERMS", quote.id)
        if quote.remedy != "refund":
            inv = world.inventory.get(quote.inventory_id)
            route = world.routes.get(quote.route_id)
            if (not inv or not route or inv.sku_id != quote.target_sku_id
                    or inv.warehouse_id != route.warehouse_id or quote.promised_arrival is None):
                bad("REPLACEMENT_TERMS_INVALID", quote.id)
            target = world.skus.get(quote.target_sku_id)
            if target and order and target.merchant_id != order.merchant_id:
                bad("REPLACEMENT_MERCHANT_MISMATCH", quote.id)
        if quote.status == "committed" and case and case.committed_quote_id != quote.id:
            bad("QUOTE_COMMITMENT_CASE_MISMATCH", quote.id)
    for delegation in world.delegations.values():
        try:
            aware(delegation.expires_at)
        except ValueError:
            bad("DELEGATION_TIME_INVALID", delegation.id)
        for line_id in delegation.line_ids:
            line = world.lines.get(line_id)
            order = world.orders.get(line.order_id) if line else None
            if not order or order.customer_id != delegation.customer_id:
                bad("DELEGATION_SCOPE_INVALID", delegation.id)
        if not delegation.source_customer_event_hash:
            bad("DELEGATION_SOURCE_MISSING", delegation.id)
    for consent in world.consents.values():
        quote = world.quotes.get(consent.quote_id)
        try:
            aware(consent.expires_at)
        except ValueError:
            bad("CONSENT_TIME_INVALID", consent.id)
        if (not quote or quote.customer_id != consent.customer_id or
                quote.terms_hash != consent.terms_hash or not consent.presented_offer_event_hash
                or not consent.source_customer_event_hash):
            bad("CONSENT_BINDING_INVALID", consent.id)
    ledger_pairs = [(row.quote_id, row.kind) for row in world.ledger.values()]
    if len(ledger_pairs) != len(set(ledger_pairs)):
        bad("DUPLICATE_TRANSACTION_POSTING", "ledger")
    for entry in world.ledger.values():
        quote = world.quotes.get(entry.quote_id)
        case = world.cases.get(quote.case_id) if quote else None
        line = world.lines.get(entry.line_id)
        order = world.orders.get(line.order_id) if line else None
        if not quote or not case or not order or case.line_id != entry.line_id:
            bad("LEDGER_SCOPE_INVALID", entry.id)
            continue
        expected = quote.refund_cents if entry.kind == "refund" else quote.extra_charge_cents
        if entry.amount_cents != expected or entry.channel != order.payment_channel:
            bad("LEDGER_AMOUNT_OR_CHANNEL_INVALID", entry.id)
        debit, credit = (("merchant", order.customer_id) if entry.kind == "refund" else (order.customer_id, "merchant"))
        if (entry.debit_account, entry.credit_account) != (debit, credit):
            bad("LEDGER_DOUBLE_ENTRY_INVALID", entry.id)
        if not any(row.quote_id == quote.id for row in world.settlements.values()):
            bad("LEDGER_WITHOUT_SETTLEMENT", entry.id)
    for order in world.orders.values():
        refunded = sum(row.amount_cents for row in world.ledger.values()
                       if row.kind == "refund" and world.lines.get(row.line_id)
                       and world.lines[row.line_id].order_id == order.id)
        if refunded > order.paid_cents:
            bad("REFUNDS_EXCEED_ORIGINAL_PAYMENT", order.id)
    for line in world.lines.values():
        if sum(row.amount_cents for row in world.ledger.values()
               if row.line_id == line.id and row.kind == "refund") > line.allocated_paid_cents:
            bad("LINE_REFUNDS_EXCEED_ORIGINAL_CREDIT", line.id)
    for settlement in world.settlements.values():
        quote = world.quotes.get(settlement.quote_id)
        if not quote:
            continue
        for kind, amount in (("refund", quote.refund_cents), ("extra_charge", quote.extra_charge_cents)):
            entries = [row for row in world.ledger.values() if row.quote_id == quote.id and row.kind == kind]
            if bool(amount) != bool(entries):
                bad("SETTLEMENT_LEDGER_MISSING", settlement.id)
    expected_reserved, expected_shipped = Counter(), Counter()
    for quote in world.quotes.values():
        case = world.cases.get(quote.case_id)
        if quote.status != "committed" or not quote.inventory_id or not case:
            continue
        if case.status == "resolved":
            expected_shipped[quote.inventory_id] += quote.quantity
        else:
            expected_reserved[quote.inventory_id] += quote.quantity
    for inv in world.inventory.values():
        if (inv.reserved, inv.shipped) != (expected_reserved[inv.id], expected_shipped[inv.id]):
            bad("INVENTORY_COMMITMENT_MISMATCH", inv.id)
    for fulfillment in world.fulfillments.values():
        quote = world.quotes.get(fulfillment.quote_id)
        if (not quote or fulfillment.sku_id != quote.target_sku_id or fulfillment.quantity != quote.quantity
                or fulfillment.inventory_id != quote.inventory_id or fulfillment.promised_arrival != quote.promised_arrival
                or not any(row.quote_id == quote.id for row in world.settlements.values())):
            bad("FULFILLMENT_TERMS_MISMATCH", fulfillment.id)
    if len({row.quote_id for row in world.fulfillments.values()}) != len(world.fulfillments):
        bad("DUPLICATE_FULFILLMENT", "fulfillments")
    for settlement in world.settlements.values():
        quote = world.quotes.get(settlement.quote_id)
        if quote and (quote.remedy != "refund") != any(
                row.quote_id == quote.id for row in world.fulfillments.values()):
            bad("SETTLEMENT_FULFILLMENT_MISSING", settlement.id)
    for ret in world.returns.values():
        case = world.cases.get(ret.case_id)
        if not case or not case.committed_quote_id:
            bad("RETURN_SCOPE_INVALID", ret.id)
        if ret.status == "received" and ret.received_event_id not in world.applied_events:
            bad("RETURN_RECEIPT_EVENT_MISSING", ret.id)
        if ret.status == "received":
            event = world.external_events.get(ret.received_event_id)
            if not event or event.kind != "return_received" or event.target_id != ret.case_id:
                bad("RETURN_RECEIPT_BINDING_INVALID", ret.id)
    if len(set(world.applied_events)) != len(world.applied_events) or any(
            event not in world.external_events for event in world.applied_events):
        bad("EXTERNAL_EVENT_IDENTITY_INVALID", "applied_events")
    return errors
