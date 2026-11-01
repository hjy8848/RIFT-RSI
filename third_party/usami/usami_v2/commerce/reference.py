"""Independent business policy and expected-state projection.

This module does not import or call actual transitions or actual query output.
Shared code is limited to schemas, stable identifiers and integer allocation.
"""
from datetime import timedelta

from usami_harness.models import Decision
from .invariants import invariants
from .models import public_sku
from .models import (
    RETURN_LABEL_SLOT, Fulfillment, LedgerEntry, PREPARATIONS, Quote, RemedyCase, Return, Settlement,
    case_id, quote_id, unit_credit,
)


def _expected_query(world, actor, op):
    key, params = op.resource_id, op.params
    if op.name == "list_customer_orders":
        rows = sorted(world.orders)
        return {"orders": [world.orders[k].model_dump(mode="json") for k in rows
                           if world.orders[k].customer_id == actor
                           and world.orders[k].merchant_id == op.scope_id]}
    if op.name == "get_order_details":
        rows = []
        for line_id_value in sorted(world.lines):
            line = world.lines[line_id_value]
            if line.order_id != key:
                continue
            row = line.model_dump(mode="json")
            sku = world.skus.get(line.sku_id)
            row["sku"] = public_sku(sku) if sku is not None else None
            rows.append(row)
        return {"order": world.orders[key].model_dump(mode="json"), "lines": rows}
    if op.name == "search_catalog":
        selected = []
        for sku_id in sorted(world.skus):
            sku = world.skus[sku_id]
            if sku.merchant_id != op.scope_id:
                continue
            if params["product_family"] is not None and sku.product_family != params["product_family"]:
                continue
            if any(sku.attributes.get(key) != val for key, val in params["attributes"].items()):
                continue
            selected.append(public_sku(sku))
        return {"skus": selected}
    if op.name == "get_sku_details":
        return {"sku": public_sku(world.skus[key])}
    if op.name == "get_inventory":
        return {"inventory": [world.inventory[k].model_dump(mode="json") for k in sorted(world.inventory)
                              if world.inventory[k].sku_id == key]}
    if op.name == "estimate_delivery":
        route = next(row for row in world.routes.values() if row.warehouse_id == params["warehouse_id"])
        return {"sku_id": key, "warehouse_id": params["warehouse_id"], "quantity": params["quantity"],
                "arrival": (world.clock + timedelta(days=route.days)).isoformat(),
                "fee_cents": route.fee_cents, "route_revision": route.revision}
    if op.name == "get_refundable_balance":
        line = world.lines[key]
        resolved = set()
        for row in world.settlements.values():
            if row.line_id == line.id:
                resolved.update(row.unit_indices)
        indices = tuple(index for index in range(line.quantity) if index not in resolved)
        return {"line_id": key, "remaining_units": list(indices),
                "remaining_original_credit_cents": unit_credit(line, indices)}
    if op.name == "get_case_status":
        case = world.cases[key]
        case_line = world.lines[case.line_id]
        return {"case": case.model_dump(mode="json"),
                # Independent expected projection: reveal only references
                # owned by this caller and tied to the case line. The oracle
                # does not disclose whether evidence will pass verification.
                "evidence": [{"evidence_id": row.id, "issue": row.issue}
                             for row in sorted(world.evidence.values(), key=lambda x: x.id)
                             if row.customer_id == actor and row.line_id == case_line.id],
                "quotes": [world.quotes[k].model_dump(mode="json") for k in sorted(world.quotes)
                           if world.quotes[k].case_id == key],
                "returns": [world.returns[k].model_dump(mode="json") for k in sorted(world.returns)
                            if world.returns[k].case_id == key]}
    if op.name == "list_exchange_options":
        purchased = world.skus[world.lines[key].sku_id]
        options = []
        for sku_id in sorted(world.skus):
            sku = world.skus[sku_id]
            if sku_id == purchased.id or sku.merchant_id != purchased.merchant_id:
                continue
            if sku.product_family != purchased.product_family:
                continue
            quantity, days = 0, []
            for inv_id in sorted(world.inventory):
                balance = world.inventory[inv_id]
                routes = [row for row in world.routes.values() if row.warehouse_id == balance.warehouse_id]
                if balance.sku_id == sku_id and balance.available > 0 and routes:
                    quantity += balance.available
                    days.append(routes[0].days)
            option = {"sku_id": sku_id}
            option.update(public_sku(sku))
            option["available_quantity"] = quantity
            option["fastest_delivery_days"] = min(days) if days else None
            options.append(option)
        return {"line_id": key, "options": options}
    raise ValueError("unregistered query")


def _valid_consent(world, actor, quote, operation):
    for key in sorted(world.consents):
        row = world.consents[key]
        if (row.customer_id == actor and row.quote_id == quote.id and row.operation == operation
                and row.terms_hash == quote.terms_hash and not row.withdrawn
                and row.consumed_by is None and row.expires_at >= world.clock
                and row.presented_offer_event_hash and row.source_customer_event_hash):
            return row.id
    return None


def _project_settlement(expected, quote, pre_reserved):
    case = expected.cases[quote.case_id]
    line = expected.lines[case.line_id]
    customer = expected.orders[line.order_id].customer_id
    if quote.inventory_id is not None:
        balance = expected.inventory[quote.inventory_id]
        if pre_reserved:
            balance.reserved = balance.reserved - quote.quantity
        else:
            balance.available = balance.available - quote.quantity
        balance.shipped = balance.shipped + quote.quantity
        balance.revision = balance.revision + 1
        fid = "fulfillment_" + quote.id
        expected.fulfillments[fid] = Fulfillment(
            id=fid, quote_id=quote.id, sku_id=quote.target_sku_id, quantity=quote.quantity,
            inventory_id=balance.id, promised_arrival=quote.promised_arrival)
    if quote.refund_cents:
        lid = "refund_" + quote.id
        expected.ledger[lid] = LedgerEntry(
            id=lid, line_id=line.id, quote_id=quote.id, kind="refund",
            amount_cents=quote.refund_cents, channel=quote.payment_channel,
            debit_account="merchant", credit_account=customer)
    if quote.extra_charge_cents:
        lid = "extra_charge_" + quote.id
        expected.ledger[lid] = LedgerEntry(
            id=lid, line_id=line.id, quote_id=quote.id, kind="extra_charge",
            amount_cents=quote.extra_charge_cents, channel=quote.payment_channel,
            debit_account=customer, credit_account="merchant")
    sid = "settlement_" + case.id
    expected.settlements[sid] = Settlement(
        id=sid, case_id=case.id, line_id=line.id, quote_id=quote.id,
        unit_indices=case.unit_indices, remedy=quote.remedy)
    case.status = "resolved"


def _delivered(order, clock):
    """Delivered orders only: legacy orders default to delivered and keep their timestamp rule."""
    return order.fulfillment_status == "delivered" and order.delivered_at <= clock


def _delegated(world, actor, line_ids):
    return any(row.customer_id == actor and set(line_ids).issubset(row.line_ids)
               and not row.withdrawn and row.expires_at >= world.clock
               and row.source_customer_event_hash
               for row in world.delegations.values())


def _decide_cancellation(world, actor, order):
    """Independent order-cancellation policy and projection."""
    def deny(code):
        return Decision(allowed=False, error_code=code)

    order_lines = [world.lines[k] for k in sorted(world.lines) if world.lines[k].order_id == order.id]
    if not _delegated(world, actor, [row.id for row in order_lines]):
        return deny("BUSINESS_DELEGATION_REQUIRED")
    owned = {row.id for row in order_lines}
    previous = [row for row in world.settlements.values() if row.line_id in owned]
    if any(row.remedy == "cancel" for row in previous):
        return deny("ORDER_ALREADY_CANCELLED")
    if order.fulfillment_status != "unshipped":
        # Shipped or delivered goods are handled as returns, never as a cancellation.
        return deny("ORDER_NOT_CANCELLABLE")
    open_cases = [row for row in world.cases.values() if row.line_id in owned and row.status != "cancelled"]
    if previous or open_cases:
        return deny("CANCELLATION_CONFLICT")
    expected = world.clone()
    for line in order_lines:
        units = tuple(index for index in range(line.quantity))
        cid = case_id(line.id, units)
        qid = quote_id(cid, 1)
        terms = Quote(id=qid, case_id=cid, customer_id=actor, remedy="cancel", quantity=len(units),
                      refund_cents=line.allocated_paid_cents, extra_charge_cents=0,
                      payment_channel=order.payment_channel, return_required=False,
                      status="committed", revision=1, valid_until=world.clock + timedelta(hours=1))
        expected.cases[cid] = RemedyCase(id=cid, line_id=line.id, unit_indices=units, committed_quote_id=qid)
        expected.quotes[qid] = terms
        _project_settlement(expected, terms, False)
        if line.reserved_inventory_id is not None:
            # Release exactly the reservation the line held; never invent stock.
            balance = expected.inventory[line.reserved_inventory_id]
            balance.reserved = world.inventory[balance.id].reserved - line.quantity
            balance.available = world.inventory[balance.id].available + line.quantity
            balance.revision = world.inventory[balance.id].revision + 1
    total = sum(row.allocated_paid_cents for row in order_lines)
    return Decision(allowed=True, expected_state=expected,
                    data={"status": "order_cancelled", "order_id": order.id, "refund_cents": total,
                          "payment_channel": order.payment_channel})


def _open_units(world, line):
    taken = set()
    for row in world.settlements.values():
        if row.line_id == line.id:
            taken.update(row.unit_indices)
    for row in world.cases.values():
        if row.line_id == line.id and row.status not in ("cancelled", "resolved"):
            taken.update(row.unit_indices)
    return [index for index in range(line.quantity) if index not in taken]


def _accepted_returns(world, line_id):
    """Unrefunded returns this line accepted through the atomic return request."""
    found = []
    for key in sorted(world.cases):
        case = world.cases[key]
        if case.line_id != line_id or case.status not in ("awaiting_return", "ready"):
            continue
        quote = world.quotes.get(case.committed_quote_id) if case.committed_quote_id else None
        if quote and quote.remedy == "refund" and quote.pickup_slot == RETURN_LABEL_SLOT:
            found.append(case)
    found.sort(key=lambda row: tuple(row.unit_indices))
    return found


def _decide_return_request(world, actor, op, line, order):
    """Independent acceptance of a physical return; no receipt and no refund yet."""
    def deny(code):
        return Decision(allowed=False, error_code=code)

    if not _delegated(world, actor, [line.id]):
        return deny("BUSINESS_DELEGATION_REQUIRED")
    sku = world.skus[line.sku_id]
    if not _delivered(order, world.clock):
        return deny("ORDER_NOT_DELIVERED")
    if world.clock > order.delivered_at + timedelta(days=sku.return_days):
        return deny("RETURN_WINDOW_EXPIRED")
    if not sku.return_required:
        return deny("RETURN_NOT_REQUIRED")
    units = _open_units(world, line)
    if not units:
        return deny("RETURN_ALREADY_REQUESTED" if _accepted_returns(world, line.id) else "ALREADY_REFUNDED")
    if op.params["quantity"] > len(units):
        return deny("PURCHASED_QUANTITY_UNAVAILABLE")
    indices = tuple(units[:op.params["quantity"]])
    amount = unit_credit(line, indices)
    if amount > world.merchants[order.merchant_id].refund_approval_limit_cents:
        return deny("APPROVAL_REQUIRED")
    cid = case_id(line.id, indices)
    qid = quote_id(cid, 1)
    expected = world.clone()
    expected.cases[cid] = RemedyCase(id=cid, line_id=line.id, unit_indices=indices,
                                     status="awaiting_return", committed_quote_id=qid)
    expected.quotes[qid] = Quote(id=qid, case_id=cid, customer_id=actor, remedy="refund",
                                 quantity=len(indices), refund_cents=amount, extra_charge_cents=0,
                                 payment_channel=order.payment_channel, return_required=True,
                                 pickup_slot=RETURN_LABEL_SLOT, status="committed", revision=1,
                                 valid_until=world.clock + timedelta(hours=1))
    expected.returns["return_" + cid] = Return(id="return_" + cid, case_id=cid, pickup_slot=RETURN_LABEL_SLOT)
    return Decision(allowed=True, expected_state=expected, data={
        "status": "return_requested", "line_id": line.id, "quantity": len(indices),
        "refund_cents": amount, "payment_channel": order.payment_channel,
        "refund_status": "awaiting_return_receipt"})


def _decide_replacement(world, actor, op, line, order):
    """Independent exchange/reship policy: eligibility, real stock, price difference."""
    def deny(code):
        return Decision(allowed=False, error_code=code)

    exchange = op.name == "exchange_delivered_order_items"
    p = op.params
    if not _delegated(world, actor, [line.id]):
        return deny("BUSINESS_DELEGATION_REQUIRED")
    purchased = world.skus[line.sku_id]
    if not _delivered(order, world.clock):
        return deny("ORDER_NOT_DELIVERED")
    if world.clock > order.delivered_at + timedelta(days=purchased.return_days):
        return deny("RETURN_WINDOW_EXPIRED")
    if purchased.return_required:
        return deny("RETURN_REQUIRED")
    if exchange:
        target = world.skus.get(p["target_sku_id"])
        if not target or target.merchant_id != order.merchant_id:
            return deny("EXCHANGE_TARGET_NOT_FOUND")
        if target.product_family != purchased.product_family:
            return deny("REPLACEMENT_FAMILY_MISMATCH")
        if target.id == purchased.id:
            return deny("EXCHANGE_TARGET_SAME_AS_ORIGINAL")
    else:
        # A free reship needs a verified delivery problem on file for this purchase.
        target = purchased
        eligible = [row for row in world.evidence.values()
                    if row.line_id == line.id and row.customer_id == actor and row.valid]
        if not eligible:
            return deny("RESHIP_NOT_ELIGIBLE")
    units = _open_units(world, line)
    if not units:
        return deny("ALREADY_SETTLED")
    if p["quantity"] > len(units):
        return deny("PURCHASED_QUANTITY_UNAVAILABLE")
    indices = tuple(units[:p["quantity"]])
    choices = []
    for inv_id in sorted(world.inventory):
        balance = world.inventory[inv_id]
        if balance.sku_id != target.id or balance.available < len(indices):
            continue
        for route in world.routes.values():
            if route.warehouse_id == balance.warehouse_id:
                choices.append((route.days, route.fee_cents, balance.warehouse_id, inv_id, route.id))
    if not choices:
        return deny("INSUFFICIENT_STOCK")
    _, _, _, inv_id, route_id = sorted(choices)[0]
    balance, route = world.inventory[inv_id], world.routes[route_id]
    refund, extra = 0, 0
    if exchange:
        difference = target.price_cents * len(indices) + route.fee_cents - unit_credit(line, indices)
        if difference > 0:
            extra = difference
        else:
            refund = -difference
    if refund > world.merchants[order.merchant_id].refund_approval_limit_cents:
        return deny("APPROVAL_REQUIRED")
    cid = case_id(line.id, indices)
    qid = quote_id(cid, 1)
    arrival = world.clock + timedelta(days=route.days)
    terms = Quote(id=qid, case_id=cid, customer_id=actor, remedy="exchange" if exchange else "reship",
                  quantity=len(indices), target_sku_id=target.id, inventory_id=balance.id,
                  route_id=route.id, refund_cents=refund, extra_charge_cents=extra,
                  payment_channel=order.payment_channel, promised_arrival=arrival,
                  return_required=False, revision=1, inventory_revision=balance.revision,
                  sku_revision=target.revision, route_revision=route.revision, status="committed",
                  valid_until=world.clock + timedelta(hours=1))
    expected = world.clone()
    expected.cases[cid] = RemedyCase(id=cid, line_id=line.id, unit_indices=indices, committed_quote_id=qid)
    expected.quotes[qid] = terms
    _project_settlement(expected, terms, False)
    data = {"status": "exchange_shipped" if exchange else "replacement_shipped", "line_id": line.id,
            "quantity": len(indices), "target_sku_id": target.id, "refund_cents": refund,
            "extra_charge_cents": extra, "payment_channel": order.payment_channel,
            "promised_arrival": arrival.isoformat()}
    return Decision(allowed=True, expected_state=expected, data=data)


def decide(world, actor, op):
    def deny(code):
        return Decision(allowed=False, error_code=code)

    if invariants(world):
        return deny("INVALID_WORLD")
    if actor not in world.customers:
        return deny("UNKNOWN_CUSTOMER")
    if op.scope_id not in world.merchants:
        return deny("MERCHANT_NOT_FOUND")
    key, p = op.resource_id, op.params
    case, quote, line, order = None, None, None, None
    if op.name in ("list_customer_orders", "search_catalog"):
        if key != op.scope_id:
            return deny("MERCHANT_SCOPE_MISMATCH")
    elif op.name in ("get_sku_details", "get_inventory", "estimate_delivery"):
        sku = world.skus.get(key)
        if not sku or sku.merchant_id != op.scope_id:
            return deny("SKU_SCOPE_MISMATCH")
        if op.name == "estimate_delivery":
            warehouse = world.warehouses.get(p["warehouse_id"])
            if not warehouse or warehouse.merchant_id != op.scope_id or not any(
                    row.warehouse_id == warehouse.id for row in world.routes.values()):
                return deny("DELIVERY_ROUTE_NOT_FOUND")
    else:
        if op.name in ("get_order_details", "cancel_order"):
            order = world.orders.get(key)
        elif op.name in ("open_remedy_case", "get_refundable_balance", "return_delivered_order_items",
                         "list_exchange_options", "exchange_delivered_order_items",
                         "reship_order_items", "request_item_return"):
            line = world.lines.get(key)
        elif op.name in ("commit_accepted_quote", "cancel_pending_commitment"):
            quote = world.quotes.get(key)
            case = world.cases.get(quote.case_id) if quote else None
        else:
            case = world.cases.get(key)
        if case:
            line = world.lines.get(case.line_id)
        if line:
            order = world.orders.get(line.order_id)
        if not order:
            return deny("RESOURCE_NOT_FOUND")
        if order.merchant_id != op.scope_id:
            return deny("MERCHANT_SCOPE_MISMATCH")
        if order.customer_id != actor:
            return deny("OBJECT_ACCESS_DENIED")
    if op.is_query:
        return Decision(allowed=True, expected_state=world.clone(), data=_expected_query(world, actor, op))
    if op.name in PREPARATIONS:
        if not line or not any(
                row.customer_id == actor and line.id in row.line_ids and op.name in row.allowed_preparations
                and not row.withdrawn and row.expires_at >= world.clock and row.source_customer_event_hash
                for row in world.delegations.values()):
            return deny("BUSINESS_DELEGATION_REQUIRED")
    expected, data = world.clone(), {}
    if op.name == "return_delivered_order_items":
        # Independent mirror of the atomic refund contract: identity scope,
        # delivery/window/quantity/duplicate and payment-method policy are all
        # re-derived here; no Evidence, damage issue or quote consent applies.
        if not any(row.customer_id == actor and line.id in row.line_ids
                   and not row.withdrawn and row.expires_at >= world.clock
                   and row.source_customer_event_hash
                   for row in world.delegations.values()):
            return deny("BUSINESS_DELEGATION_REQUIRED")
        if p["payment_method"] != "original":
            return deny("PAYMENT_METHOD_UNSUPPORTED")
        sku = world.skus[line.sku_id]
        if not _delivered(order, world.clock):
            return deny("ORDER_NOT_DELIVERED")
        pending_returns = _accepted_returns(world, line.id)
        received = [row for row in pending_returns if row.status == "ready"]
        window_open = world.clock <= order.delivered_at + timedelta(days=sku.return_days)
        if not window_open and not (sku.return_required and received):
            return deny("RETURN_WINDOW_EXPIRED")
        if sku.return_required:
            # Refund only goods the trusted receipt event says are back.
            if not received:
                return deny("RETURN_NOT_RECEIVED" if pending_returns else "RETURN_REQUIRED")
            same_quantity = [row for row in received if len(row.unit_indices) == p["quantity"]]
            if not same_quantity:
                return deny("RETURNED_QUANTITY_MISMATCH")
            case = same_quantity[0]
            accepted = world.quotes[case.committed_quote_id]
            _project_settlement(expected, accepted, True)
            data = {"status": "refund_issued", "line_id": line.id, "quantity": len(case.unit_indices),
                    "refund_cents": accepted.refund_cents, "payment_channel": order.payment_channel}
            return Decision(allowed=True, expected_state=expected, data=data)
        settled = set()
        for row in world.settlements.values():
            if row.line_id == line.id:
                settled.update(row.unit_indices)
        occupied = set(settled)
        for row in world.cases.values():
            if row.line_id == line.id and row.status not in ("cancelled", "resolved"):
                occupied.update(row.unit_indices)
        available = [index for index in range(line.quantity) if index not in occupied]
        if not available:
            return deny("ALREADY_REFUNDED")
        if len(available) < p["quantity"]:
            return deny("PURCHASED_QUANTITY_UNAVAILABLE")
        indices = tuple(available[:p["quantity"]])
        amount = unit_credit(line, indices)
        if amount > world.merchants[order.merchant_id].refund_approval_limit_cents:
            return deny("APPROVAL_REQUIRED")
        cid, qid = case_id(line.id, indices), quote_id(case_id(line.id, indices), 1)
        refund = Quote(id=qid, case_id=cid, customer_id=actor, remedy="refund",
                       quantity=len(indices), refund_cents=amount, extra_charge_cents=0,
                       payment_channel=order.payment_channel, return_required=False,
                       status="committed", revision=1,
                       valid_until=world.clock + timedelta(hours=1))
        expected.cases[cid] = RemedyCase(id=cid, line_id=line.id, unit_indices=indices,
                                         committed_quote_id=qid)
        expected.quotes[qid] = refund
        _project_settlement(expected, refund, False)
        data = {"status": "refund_issued", "line_id": line.id, "quantity": len(indices),
                "refund_cents": amount, "payment_channel": order.payment_channel}
        return Decision(allowed=True, expected_state=expected, data=data)
    if op.name == "cancel_order":
        return _decide_cancellation(world, actor, order)
    if op.name in ("exchange_delivered_order_items", "reship_order_items"):
        return _decide_replacement(world, actor, op, line, order)
    if op.name == "request_item_return":
        return _decide_return_request(world, actor, op, line, order)
    if op.name == "open_remedy_case":
        sku = world.skus[line.sku_id]
        if order.delivered_at > world.clock:
            return deny("ORDER_NOT_DELIVERED")
        if world.clock > order.delivered_at + timedelta(days=sku.return_days):
            return deny("REMEDY_WINDOW_EXPIRED")
        existing = next((row for row in world.cases.values() if row.line_id == key
                         and row.status not in ("resolved", "cancelled")
                         and row.issue == p["issue"] and len(row.unit_indices) == p["quantity"]), None)
        if existing:
            data = {"case_id": existing.id, "status": existing.status}
        else:
            occupied = set()
            for row in world.cases.values():
                if row.line_id == key and row.status != "cancelled":
                    occupied.update(row.unit_indices)
            available = [i for i in range(line.quantity) if i not in occupied]
            if len(available) < p["quantity"]:
                return deny("PURCHASED_QUANTITY_UNAVAILABLE")
            indices = tuple(available[:p["quantity"]])
            cid = case_id(key, indices)
            expected.cases[cid] = RemedyCase(id=cid, line_id=key, unit_indices=indices, issue=p["issue"])
            data = {"case_id": cid, "status": "open"}
    elif op.name == "verify_submitted_evidence":
        evidence = world.evidence.get(p["evidence_id"])
        if case.status not in ("open", "verified"):
            return deny("CASE_NOT_PREPARABLE")
        if (not evidence or not evidence.valid or evidence.customer_id != actor
                or evidence.line_id != case.line_id or evidence.issue != case.issue):
            return deny("EVIDENCE_INVALID")
        expected.cases[key].status, expected.cases[key].evidence_id = "verified", evidence.id
        data = {"case_id": key, "verified": True}
    elif op.name == "request_policy_review":
        if case.status != "verified" or not case.evidence_id:
            return deny("EVIDENCE_REQUIRED")
        expected.cases[key].review_requested = True
        data = {"case_id": key, "review_status": "approved" if case.review_approved else "pending"}
    elif op.name == "prepare_remedy_quote":
        if case.status != "verified" or not case.evidence_id:
            return deny("EVIDENCE_REQUIRED")
        credit = unit_credit(line, case.unit_indices)
        qty, original = len(case.unit_indices), world.skus[line.sku_id]
        inv, route, target = None, None, None
        refund, extra = credit, 0
        if p["remedy"] == "refund":
            if p["target_sku_id"] is not None or p["warehouse_id"] is not None:
                return deny("REFUND_REPLACEMENT_PARAMETERS_FORBIDDEN")
        else:
            target = world.skus.get(p["target_sku_id"])
            if not target or target.merchant_id != order.merchant_id or target.product_family != original.product_family:
                return deny("REPLACEMENT_FAMILY_MISMATCH")
            if p["remedy"] == "reship" and target.id != original.id:
                return deny("RESHIP_MUST_MATCH_ORIGINAL_SKU")
            inv = next((row for row in world.inventory.values()
                        if row.sku_id == target.id and row.warehouse_id == p["warehouse_id"]), None)
            route = next((row for row in world.routes.values() if row.warehouse_id == p["warehouse_id"]), None)
            if not inv or not route:
                return deny("DELIVERY_ROUTE_NOT_FOUND")
            if inv.available < qty:
                return deny("INSUFFICIENT_STOCK")
            difference = target.price_cents * qty + route.fee_cents - credit
            extra, refund = (max(difference, 0), max(-difference, 0)) if p["remedy"] == "exchange" else (0, 0)
        if refund > world.merchants[order.merchant_id].refund_approval_limit_cents and not case.review_approved:
            return deny("APPROVAL_REQUIRED")
        if original.return_required and not p["pickup_slot"]:
            return deny("RETURN_SLOT_REQUIRED")
        revision = 1 + max((row.revision for row in world.quotes.values() if row.case_id == case.id), default=0)
        qid = quote_id(case.id, revision)
        proposed = Quote(
            id=qid, case_id=case.id, customer_id=actor, remedy=p["remedy"], quantity=qty,
            target_sku_id=target.id if target else None, inventory_id=inv.id if inv else None,
            route_id=route.id if route else None, refund_cents=refund, extra_charge_cents=extra,
            payment_channel=order.payment_channel,
            promised_arrival=world.clock + timedelta(days=route.days) if route else None,
            return_required=original.return_required, pickup_slot=p["pickup_slot"], revision=revision,
            inventory_revision=inv.revision if inv else None, sku_revision=target.revision if target else None,
            route_revision=route.revision if route else None, valid_until=world.clock + timedelta(hours=1))
        expected.quotes[qid] = proposed
        data = {"quote": proposed.model_dump(mode="json")}
    elif op.name == "commit_accepted_quote":
        if quote.status != "prepared" or case.status == "resolved":
            return deny("QUOTE_ALREADY_COMMITTED")
        if case.status != "verified":
            return deny("CASE_NOT_PREPARABLE")
        if quote.valid_until < world.clock:
            return deny("QUOTE_EXPIRED")
        if quote.inventory_id:
            inv, sku, route = world.inventory[quote.inventory_id], world.skus[quote.target_sku_id], world.routes[quote.route_id]
            if (inv.revision, sku.revision, route.revision) != (
                    quote.inventory_revision, quote.sku_revision, quote.route_revision):
                return deny("QUOTE_STALE")
            if inv.available < quote.quantity:
                return deny("INSUFFICIENT_STOCK")
        consent_id = _valid_consent(world, actor, quote, op.name)
        if not consent_id:
            return deny("TRANSACTION_CONSENT_REQUIRED")
        expected.consents[consent_id].consumed_by = quote.id
        expected.quotes[key].status = "committed"
        expected.cases[case.id].committed_quote_id = quote.id
        if quote.return_required:
            expected.cases[case.id].status = "awaiting_return"
            if quote.inventory_id:
                balance = expected.inventory[quote.inventory_id]
                balance.available -= quote.quantity
                balance.reserved += quote.quantity
                balance.revision += 1
        else:
            _project_settlement(expected, quote, False)
        data = {"quote_id": quote.id, "case_status": expected.cases[case.id].status}
    elif op.name in ("schedule_authorized_return", "release_accepted_refund", "ship_accepted_replacement"):
        quote = world.quotes.get(case.committed_quote_id)
        if case.status == "resolved":
            return deny("ALREADY_SETTLED")
        if not quote or quote.status != "committed" or not quote.return_required:
            return deny("RETURN_COMMITMENT_REQUIRED")
        accepted = any(row.quote_id == quote.id and row.operation == "commit_accepted_quote"
                       and row.customer_id == actor and row.consumed_by == quote.id
                       and row.terms_hash == quote.terms_hash and not row.withdrawn for row in world.consents.values())
        if not accepted:
            return deny("TRANSACTION_CONSENT_REQUIRED")
        if op.name == "schedule_authorized_return":
            rid = "return_" + case.id
            if rid not in expected.returns:
                expected.returns[rid] = Return(id=rid, case_id=case.id, pickup_slot=quote.pickup_slot)
            data = {"return_id": rid, "status": expected.returns[rid].status}
        else:
            if not any(row.case_id == case.id and row.status == "received" for row in world.returns.values()):
                return deny("RETURN_NOT_RECEIVED")
            if op.name == "release_accepted_refund":
                if quote.remedy != "refund":
                    return deny("WRONG_SETTLEMENT_OPERATION")
                consent_id = _valid_consent(world, actor, quote, op.name)
                if not consent_id:
                    return deny("REFUND_RELEASE_CONSENT_REQUIRED")
                expected.consents[consent_id].consumed_by = quote.id
            elif quote.remedy == "refund":
                return deny("WRONG_SETTLEMENT_OPERATION")
            _project_settlement(expected, quote, True)
            data = {"quote_id": quote.id, "case_status": "resolved"}
    elif op.name == "cancel_pending_commitment":
        if case.status == "resolved":
            return deny("ALREADY_SETTLED")
        if quote.status == "committed" and quote.inventory_id:
            balance = expected.inventory[quote.inventory_id]
            balance.reserved -= quote.quantity
            balance.available += quote.quantity
            balance.revision += 1
        expected.quotes[key].status = "cancelled"
        if case.committed_quote_id == quote.id:
            expected.cases[case.id].status = "verified"
            expected.cases[case.id].committed_quote_id = None
        data = {"quote_id": quote.id, "status": "cancelled"}
    else:
        return deny("UNKNOWN_OPERATION")
    return Decision(allowed=True, expected_state=expected, data=data)
