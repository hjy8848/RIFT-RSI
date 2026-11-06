"""Actual state mutations. Deliberately never imports the Reference."""
from datetime import timedelta

from usami_harness.models import Result, digest
from .models import (
    CUSTOMER_SCOPED, RETURN_LABEL_SLOT, Address, Fulfillment, LedgerEntry, Modification, ModificationChange,
    Quote, RemedyCase, Return, Settlement, case_id, quote_id, unit_credit,
)


def _channel(world, order, requested, *, refund, charge):
    """Resolve the payment channel of a cash movement: (channel, error code).

    Refunds go back to the original channel or to a gift card the Customer owns; a charge may use any
    owned instrument but a gift card must cover it. Worlds without payment methods only know "original".
    """
    if requested == "original":
        channel = order.payment_channel
    else:
        method = world.payment_methods.get(requested)
        if method is None or method.customer_id != order.customer_id:
            return None, "PAYMENT_METHOD_NOT_FOUND" if world.payment_methods else "PAYMENT_METHOD_UNSUPPORTED"
        channel = method.id
        if refund and method.kind != "gift_card" and channel != order.payment_channel:
            return None, "REFUND_DESTINATION_NOT_ALLOWED"
    method = world.payment_methods.get(channel)
    if charge and method is not None and method.kind == "gift_card" and method.balance_cents < charge:
        return None, "INSUFFICIENT_BALANCE"
    return channel, None


def _timing(world, channel):
    method = world.payment_methods.get(channel)
    return "immediate" if method is not None and method.kind == "gift_card" else "5-7 business days"


def _cash(world, channel, refund, charge):
    """Move gift-card balance for a settled refund or charge; other instruments keep no balance."""
    method = world.payment_methods.get(channel)
    if method is not None and method.kind == "gift_card":
        method.balance_cents = method.balance_cents + refund - charge


def _cancelled(world, order_id):
    owned = {row.id for row in world.lines.values() if row.order_id == order_id}
    return any(row.line_id in owned and row.remedy == "cancel" for row in world.settlements.values())


def _hard_scope(world, actor, op):
    if actor not in world.customers or op.scope_id not in world.merchants:
        return False
    if op.name in CUSTOMER_SCOPED:
        return op.resource_id == op.scope_id
    if op.name in ("get_sku_details", "get_inventory", "estimate_delivery"):
        sku = world.skus.get(op.resource_id)
        return sku is not None and sku.merchant_id == op.scope_id
    if op.name == "prepare_remedy_quote":
        target = world.skus.get(op.params.get("target_sku_id"))
        warehouse = world.warehouses.get(op.params.get("warehouse_id"))
        if target and target.merchant_id != op.scope_id or warehouse and warehouse.merchant_id != op.scope_id:
            return False
    key = op.resource_id
    if op.name in ("get_order_details", "cancel_order", "modify_order_address", "modify_order_items"):
        order = world.orders.get(key)
    elif op.name in ("open_remedy_case", "get_refundable_balance", "return_delivered_order_items",
                     "list_exchange_options", "exchange_delivered_order_items",
                     "reship_order_items", "request_item_return"):
        line = world.lines.get(key)
        order = world.orders.get(line.order_id) if line else None
    else:
        case = world.cases.get(key)
        quote = world.quotes.get(key)
        if case is None and quote is not None:
            case = world.cases.get(quote.case_id)
        line = world.lines.get(case.line_id) if case else None
        order = world.orders.get(line.order_id) if line else None
    return order is not None and order.customer_id == actor and order.merchant_id == op.scope_id


def _consume(world, actor, quote, operation):
    matches = sorted((row for row in world.consents.values()
                      if row.customer_id == actor and row.quote_id == quote.id
                      and row.operation == operation and row.terms_hash == quote.terms_hash
                      and not row.withdrawn and row.consumed_by is None and row.expires_at >= world.clock),
                     key=lambda row: row.id)
    # Sandbox can execute without consent; it must never fabricate a consent.
    if matches:
        matches[0].consumed_by = quote.id


def _settle(world, quote, *, inventory_pre_reserved):
    case = world.cases[quote.case_id]
    line = world.lines[case.line_id]
    order = world.orders[line.order_id]
    if any(row.case_id == case.id for row in world.settlements.values()):
        raise ValueError("transaction already settled")
    if quote.inventory_id:
        inv = world.inventory[quote.inventory_id]
        if inventory_pre_reserved:
            inv.reserved -= quote.quantity
        else:
            inv.available -= quote.quantity
        inv.shipped += quote.quantity
        inv.revision += 1
        fid = "fulfillment_" + quote.id
        world.fulfillments[fid] = Fulfillment(
            id=fid, quote_id=quote.id, sku_id=quote.target_sku_id,
            quantity=quote.quantity, inventory_id=inv.id, promised_arrival=quote.promised_arrival)
    for kind, amount in (("refund", quote.refund_cents), ("extra_charge", quote.extra_charge_cents)):
        if amount:
            key = kind + "_" + quote.id
            debit, credit = (("merchant", order.customer_id) if kind == "refund"
                             else (order.customer_id, "merchant"))
            world.ledger[key] = LedgerEntry(
                id=key, line_id=line.id, quote_id=quote.id, kind=kind, amount_cents=amount,
                channel=quote.payment_channel, debit_account=debit, credit_account=credit)
            _cash(world, quote.payment_channel, amount if kind == "refund" else 0,
                  amount if kind == "extra_charge" else 0)
    sid = "settlement_" + case.id
    world.settlements[sid] = Settlement(id=sid, case_id=case.id, line_id=line.id,
                                       quote_id=quote.id, unit_indices=case.unit_indices, remedy=quote.remedy)
    case.status = "resolved"


def _free_units(world, line):
    """Purchased units of a line that are neither settled nor held by an open case."""
    held = {i for row in world.settlements.values() if row.line_id == line.id for i in row.unit_indices}
    held.update(i for row in world.cases.values() if row.line_id == line.id
                and row.status not in ("cancelled", "resolved") for i in row.unit_indices)
    return [i for i in range(line.quantity) if i not in held]


def _label_return_cases(world, line_id):
    """Open returns accepted by request_item_return for this line, oldest units first."""
    rows = []
    for case in world.cases.values():
        quote = world.quotes.get(case.committed_quote_id or "")
        if (case.line_id == line_id and case.status in ("awaiting_return", "ready") and quote is not None
                and quote.remedy == "refund" and quote.pickup_slot == RETURN_LABEL_SLOT):
            rows.append(case)
    return sorted(rows, key=lambda row: row.unit_indices)


def _replacement_source(world, sku_id, quantity):
    """Fastest stocked warehouse with a delivery route for a replacement shipment."""
    rows = []
    for inv in world.inventory.values():
        route = next((row for row in world.routes.values() if row.warehouse_id == inv.warehouse_id), None)
        if inv.sku_id == sku_id and inv.available >= quantity and route is not None:
            rows.append(((route.days, route.fee_cents, inv.warehouse_id), inv, route))
    if not rows:
        return None, None
    _, inv, route = min(rows, key=lambda row: row[0])
    return inv, route


def apply(world, actor, op, event_index):
    if not _hard_scope(world, actor, op):
        return world.clone(), Result(success=False, error_code="OBJECT_ACCESS_DENIED")
    if op.is_query:
        from .observations import query_data
        try:
            return world.clone(), Result(success=True, data=query_data(world, actor, op))
        except (KeyError, ValueError, StopIteration):
            return world.clone(), Result(success=False, error_code="QUERY_NOT_REPRESENTABLE")
    new, p, key = world.clone(), op.params, op.resource_id
    try:
        if op.name == "return_delivered_order_items":
            # One Agent-facing business action: validate the delivered purchase
            # and issue the refund to the original payment channel. Internal
            # case/quote/settlement rows stay Agent-invisible and are never
            # fabricated: no Evidence, no damage issue, no consent lifecycle.
            line = new.lines[key]
            order = new.orders[line.order_id]
            sku = new.skus[line.sku_id]
            channel = order.payment_channel
            if p["payment_method"] != "original":
                channel, error = _channel(new, order, p["payment_method"], refund=True, charge=0)
                if error is None and sku.return_required:
                    error = "PAYMENT_METHOD_UNSUPPORTED"
                if error:
                    return world.clone(), Result(success=False, error_code=error)
            if order.fulfillment_status != "delivered" or order.delivered_at > new.clock:
                return world.clone(), Result(success=False, error_code="ORDER_NOT_DELIVERED")
            label_cases = _label_return_cases(new, key)
            received = [row for row in label_cases if row.status == "ready"]
            # A return accepted inside the window may be refunded after the
            # trusted receipt event, even if that event lands after the window.
            if not (sku.return_required and received) and \
                    new.clock > order.delivered_at + timedelta(days=sku.return_days):
                return world.clone(), Result(success=False, error_code="RETURN_WINDOW_EXPIRED")
            if sku.return_required:
                matching = [row for row in received if len(row.unit_indices) == p["quantity"]]
                if not matching:
                    code = ("RETURNED_QUANTITY_MISMATCH" if received
                            else "RETURN_NOT_RECEIVED" if label_cases else "RETURN_REQUIRED")
                    return world.clone(), Result(success=False, error_code=code)
                case = matching[0]
                accepted = new.quotes[case.committed_quote_id]
                _settle(new, accepted, inventory_pre_reserved=True)
                data = {"status": "refund_issued", "line_id": key, "quantity": accepted.quantity,
                        "refund_cents": accepted.refund_cents, "payment_channel": accepted.payment_channel}
                return new, Result(success=True, data=data)
            refunded = {i for row in new.settlements.values()
                        if row.line_id == key for i in row.unit_indices}
            in_progress = {i for row in new.cases.values() if row.line_id == key
                           and row.status not in ("cancelled", "resolved")
                           for i in row.unit_indices}
            available = [i for i in range(line.quantity) if i not in refunded and i not in in_progress]
            if not available:
                return world.clone(), Result(success=False, error_code="ALREADY_REFUNDED")
            if len(available) < p["quantity"]:
                return world.clone(), Result(success=False, error_code="PURCHASED_QUANTITY_UNAVAILABLE")
            indices = tuple(available[:p["quantity"]])
            amount = unit_credit(line, indices)
            if amount > new.merchants[order.merchant_id].refund_approval_limit_cents:
                return world.clone(), Result(success=False, error_code="APPROVAL_REQUIRED")
            cid, qid = case_id(key, indices), quote_id(case_id(key, indices), 1)
            new.cases[cid] = RemedyCase(id=cid, line_id=key, unit_indices=indices,
                                        committed_quote_id=qid)
            refund = Quote(id=qid, case_id=cid, customer_id=actor, remedy="refund",
                           quantity=len(indices), refund_cents=amount, extra_charge_cents=0,
                           payment_channel=channel, return_required=False,
                           status="committed", revision=1, valid_until=new.clock + timedelta(hours=1))
            new.quotes[qid] = refund
            _settle(new, refund, inventory_pre_reserved=False)
            data = {"status": "refund_issued", "line_id": key, "quantity": len(indices),
                    "refund_cents": amount, "payment_channel": channel}
            if new.payment_methods:
                data["refund_timing"] = _timing(new, channel)
        elif op.name == "cancel_order":
            # One business action: an order that has not shipped is cancelled as a
            # whole, every line's payment returns to the original channel and only
            # stock that was actually reserved for a line goes back to inventory.
            order = new.orders[key]
            lines = sorted((row for row in new.lines.values() if row.order_id == key), key=lambda x: x.id)
            line_ids = {row.id for row in lines}
            if any(row.line_id in line_ids and row.remedy == "cancel" for row in new.settlements.values()):
                return world.clone(), Result(success=False, error_code="ORDER_ALREADY_CANCELLED")
            if order.fulfillment_status != "unshipped":
                return world.clone(), Result(success=False, error_code="ORDER_NOT_CANCELLABLE")
            if order.items_locked:
                return world.clone(), Result(success=False, error_code="ORDER_LOCKED")
            if (any(row.line_id in line_ids for row in new.settlements.values())
                    or any(row.line_id in line_ids and row.status != "cancelled" for row in new.cases.values())):
                return world.clone(), Result(success=False, error_code="CANCELLATION_CONFLICT")
            for line in lines:
                indices = tuple(range(line.quantity))
                cid = case_id(line.id, indices)
                qid = quote_id(cid, 1)
                new.cases[cid] = RemedyCase(id=cid, line_id=line.id, unit_indices=indices,
                                            committed_quote_id=qid)
                cancellation = Quote(id=qid, case_id=cid, customer_id=actor, remedy="cancel",
                                     quantity=line.quantity, refund_cents=line.allocated_paid_cents,
                                     extra_charge_cents=0, payment_channel=order.payment_channel,
                                     return_required=False, status="committed", revision=1,
                                     valid_until=new.clock + timedelta(hours=1))
                new.quotes[qid] = cancellation
                _settle(new, cancellation, inventory_pre_reserved=False)
                if line.reserved_inventory_id is not None:
                    inv = new.inventory[line.reserved_inventory_id]
                    inv.reserved -= line.quantity
                    inv.available += line.quantity
                    inv.revision += 1
            data = {"status": "order_cancelled", "order_id": key, "refund_cents": order.paid_cents,
                    "payment_channel": order.payment_channel}
            if new.payment_methods:
                data["refund_timing"] = _timing(new, order.payment_channel)
        elif op.name == "request_item_return":
            # Accept a physical return. Receipt is a later trusted external
            # event; no refund is posted and nothing is marked received here.
            line = new.lines[key]
            order = new.orders[line.order_id]
            sku = new.skus[line.sku_id]
            if order.fulfillment_status != "delivered" or order.delivered_at > new.clock:
                return world.clone(), Result(success=False, error_code="ORDER_NOT_DELIVERED")
            if new.clock > order.delivered_at + timedelta(days=sku.return_days):
                return world.clone(), Result(success=False, error_code="RETURN_WINDOW_EXPIRED")
            if not sku.return_required:
                return world.clone(), Result(success=False, error_code="RETURN_NOT_REQUIRED")
            available = _free_units(new, line)
            if not available:
                code = "RETURN_ALREADY_REQUESTED" if _label_return_cases(new, key) else "ALREADY_REFUNDED"
                return world.clone(), Result(success=False, error_code=code)
            if len(available) < p["quantity"]:
                return world.clone(), Result(success=False, error_code="PURCHASED_QUANTITY_UNAVAILABLE")
            indices = tuple(available[:p["quantity"]])
            amount = unit_credit(line, indices)
            if amount > new.merchants[order.merchant_id].refund_approval_limit_cents:
                return world.clone(), Result(success=False, error_code="APPROVAL_REQUIRED")
            cid = case_id(key, indices)
            qid = quote_id(cid, 1)
            new.cases[cid] = RemedyCase(id=cid, line_id=key, unit_indices=indices,
                                        status="awaiting_return", committed_quote_id=qid)
            new.quotes[qid] = Quote(id=qid, case_id=cid, customer_id=actor, remedy="refund",
                                    quantity=len(indices), refund_cents=amount, extra_charge_cents=0,
                                    payment_channel=order.payment_channel, return_required=True,
                                    pickup_slot=RETURN_LABEL_SLOT, status="committed", revision=1,
                                    valid_until=new.clock + timedelta(hours=1))
            rid = "return_" + cid
            new.returns[rid] = Return(id=rid, case_id=cid, pickup_slot=RETURN_LABEL_SLOT)
            data = {"status": "return_requested", "line_id": key, "quantity": len(indices),
                    "refund_cents": amount, "payment_channel": order.payment_channel,
                    "refund_status": "awaiting_return_receipt"}
        elif op.name in ("exchange_delivered_order_items", "reship_order_items"):
            # One business action ships the replacement from real stock and posts
            # any price difference; internal case/quote/settlement stay hidden.
            exchange = op.name == "exchange_delivered_order_items"
            line = new.lines[key]
            order = new.orders[line.order_id]
            sku = new.skus[line.sku_id]
            if order.fulfillment_status != "delivered" or order.delivered_at > new.clock:
                return world.clone(), Result(success=False, error_code="ORDER_NOT_DELIVERED")
            if new.clock > order.delivered_at + timedelta(days=sku.return_days):
                return world.clone(), Result(success=False, error_code="RETURN_WINDOW_EXPIRED")
            if sku.return_required:
                return world.clone(), Result(success=False, error_code="RETURN_REQUIRED")
            if exchange:
                target = new.skus.get(p["target_sku_id"])
                if target is None or target.merchant_id != order.merchant_id:
                    return world.clone(), Result(success=False, error_code="EXCHANGE_TARGET_NOT_FOUND")
                if target.product_family != sku.product_family:
                    return world.clone(), Result(success=False, error_code="REPLACEMENT_FAMILY_MISMATCH")
                if target.id == sku.id:
                    return world.clone(), Result(success=False, error_code="EXCHANGE_TARGET_SAME_AS_ORIGINAL")
            else:
                target = sku
                if not any(row.line_id == key and row.customer_id == actor and row.valid
                           for row in new.evidence.values()):
                    return world.clone(), Result(success=False, error_code="RESHIP_NOT_ELIGIBLE")
            available = _free_units(new, line)
            if not available:
                return world.clone(), Result(success=False, error_code="ALREADY_SETTLED")
            if len(available) < p["quantity"]:
                return world.clone(), Result(success=False, error_code="PURCHASED_QUANTITY_UNAVAILABLE")
            indices = tuple(available[:p["quantity"]])
            inv, route = _replacement_source(new, target.id, len(indices))
            if inv is None:
                return world.clone(), Result(success=False, error_code="INSUFFICIENT_STOCK")
            delta = (target.price_cents * len(indices) + route.fee_cents - unit_credit(line, indices)
                     if exchange else 0)
            refund, extra = max(-delta, 0), max(delta, 0)
            if refund > new.merchants[order.merchant_id].refund_approval_limit_cents:
                return world.clone(), Result(success=False, error_code="APPROVAL_REQUIRED")
            channel = order.payment_channel
            if exchange and p.get("payment_method", "original") != "original":
                channel, error = _channel(new, order, p["payment_method"], refund=refund > 0, charge=extra)
                if error:
                    return world.clone(), Result(success=False, error_code=error)
            elif extra and new.payment_methods:
                _, error = _channel(new, order, "original", refund=False, charge=extra)
                if error:
                    return world.clone(), Result(success=False, error_code=error)
            cid = case_id(key, indices)
            qid = quote_id(cid, 1)
            new.cases[cid] = RemedyCase(id=cid, line_id=key, unit_indices=indices, committed_quote_id=qid)
            replacement = Quote(
                id=qid, case_id=cid, customer_id=actor, remedy="exchange" if exchange else "reship",
                quantity=len(indices), target_sku_id=target.id, inventory_id=inv.id, route_id=route.id,
                refund_cents=refund, extra_charge_cents=extra, payment_channel=channel,
                promised_arrival=new.clock + timedelta(days=route.days), return_required=False,
                revision=1, inventory_revision=inv.revision, sku_revision=target.revision,
                route_revision=route.revision, status="committed",
                valid_until=new.clock + timedelta(hours=1))
            new.quotes[qid] = replacement
            _settle(new, replacement, inventory_pre_reserved=False)
            data = {"status": "exchange_shipped" if exchange else "replacement_shipped",
                    "line_id": key, "quantity": len(indices), "target_sku_id": target.id,
                    "refund_cents": refund, "extra_charge_cents": extra,
                    "payment_channel": channel,
                    "promised_arrival": replacement.promised_arrival.isoformat()}
            if new.payment_methods and refund:
                data["refund_timing"] = _timing(new, channel)
        elif op.name == "open_remedy_case":
            line = new.lines[key]
            occupied = {i for row in new.cases.values() if row.line_id == key and row.status != "cancelled"
                        for i in row.unit_indices}
            existing = next((row for row in new.cases.values() if row.line_id == key
                             and row.status not in ("cancelled", "resolved")
                             and row.issue == p["issue"] and len(row.unit_indices) == p["quantity"]), None)
            if existing:
                return new, Result(success=True, data={"case_id": existing.id, "status": existing.status})
            indices = tuple(i for i in range(line.quantity) if i not in occupied)[:p["quantity"]]
            if len(indices) != p["quantity"]:
                raise ValueError("purchased units unavailable")
            cid = case_id(key, indices)
            new.cases[cid] = RemedyCase(id=cid, line_id=key, unit_indices=indices, issue=p["issue"])
            data = {"case_id": cid, "status": "open"}
        elif op.name == "verify_submitted_evidence":
            case, evidence = new.cases[key], new.evidence[p["evidence_id"]]
            if not evidence.valid or evidence.line_id != case.line_id or evidence.customer_id != actor or evidence.issue != case.issue:
                return world.clone(), Result(success=False, error_code="EVIDENCE_INVALID")
            case.evidence_id, case.status = evidence.id, "verified"
            data = {"case_id": key, "verified": True}
        elif op.name == "request_policy_review":
            case = new.cases[key]
            case.review_requested = True
            data = {"case_id": key, "review_status": "approved" if case.review_approved else "pending"}
        elif op.name == "prepare_remedy_quote":
            case = new.cases[key]
            line, quantity = new.lines[case.line_id], len(case.unit_indices)
            order = new.orders[line.order_id]
            original, target = new.skus[line.sku_id], new.skus.get(p["target_sku_id"])
            amount = unit_credit(line, case.unit_indices)
            inv = next((row for row in new.inventory.values()
                        if row.sku_id == p["target_sku_id"] and row.warehouse_id == p["warehouse_id"]), None)
            route = next((row for row in new.routes.values() if row.warehouse_id == p["warehouse_id"]), None)
            if p["remedy"] != "refund" and (not target or not inv or not route):
                raise ValueError("replacement terms not representable")
            delta = target.price_cents * quantity + route.fee_cents - amount if target and route else -amount
            if p["remedy"] == "reship":
                delta = 0
            revision = max((row.revision for row in new.quotes.values() if row.case_id == key), default=0) + 1
            qid = quote_id(key, revision)
            q = Quote(
                id=qid, case_id=key, customer_id=actor, remedy=p["remedy"], quantity=quantity,
                target_sku_id=target.id if target and p["remedy"] != "refund" else None,
                inventory_id=inv.id if inv and p["remedy"] != "refund" else None,
                route_id=route.id if route and p["remedy"] != "refund" else None,
                refund_cents=max(-delta, 0), extra_charge_cents=max(delta, 0),
                payment_channel=order.payment_channel,
                promised_arrival=new.clock + timedelta(days=route.days) if route and p["remedy"] != "refund" else None,
                return_required=original.return_required, pickup_slot=p["pickup_slot"],
                revision=revision, inventory_revision=inv.revision if inv and p["remedy"] != "refund" else None,
                sku_revision=target.revision if target and p["remedy"] != "refund" else None,
                route_revision=route.revision if route and p["remedy"] != "refund" else None,
                valid_until=new.clock + timedelta(hours=1))
            new.quotes[qid] = q
            data = {"quote": q.model_dump(mode="json")}
        elif op.name == "commit_accepted_quote":
            q = new.quotes[key]
            case = new.cases[q.case_id]
            if q.status != "prepared" or case.status == "resolved":
                return world.clone(), Result(success=False, error_code="QUOTE_ALREADY_COMMITTED")
            q.status = "committed"
            case.committed_quote_id = q.id
            _consume(new, actor, q, op.name)
            if q.return_required:
                case.status = "awaiting_return"
                if q.inventory_id:
                    inv = new.inventory[q.inventory_id]
                    inv.available -= q.quantity
                    inv.reserved += q.quantity
                    inv.revision += 1
            else:
                _settle(new, q, inventory_pre_reserved=False)
            data = {"quote_id": q.id, "case_status": case.status}
        elif op.name == "schedule_authorized_return":
            case = new.cases[key]
            quote = new.quotes[case.committed_quote_id]
            rid = "return_" + key
            if rid not in new.returns:
                new.returns[rid] = Return(id=rid, case_id=key, pickup_slot=quote.pickup_slot)
            data = {"return_id": rid, "status": new.returns[rid].status}
        elif op.name in ("release_accepted_refund", "ship_accepted_replacement"):
            case = new.cases[key]
            q = new.quotes[case.committed_quote_id]
            if case.status == "resolved":
                return world.clone(), Result(success=False, error_code="ALREADY_SETTLED")
            if op.name == "release_accepted_refund":
                _consume(new, actor, q, op.name)
            _settle(new, q, inventory_pre_reserved=True)
            data = {"quote_id": q.id, "case_status": case.status}
        elif op.name == "cancel_pending_commitment":
            q = new.quotes[key]
            case = new.cases[q.case_id]
            if case.status == "resolved":
                return world.clone(), Result(success=False, error_code="ALREADY_SETTLED")
            if q.status == "committed" and q.inventory_id:
                inv = new.inventory[q.inventory_id]
                inv.reserved -= q.quantity
                inv.available += q.quantity
                inv.revision += 1
            q.status = "cancelled"
            if case.committed_quote_id == q.id:
                case.status, case.committed_quote_id = "verified", None
            data = {"quote_id": q.id, "status": q.status}
        elif op.name == "modify_order_address":
            order = new.orders[key]
            if _cancelled(new, key):
                return world.clone(), Result(success=False, error_code="ORDER_ALREADY_CANCELLED")
            if order.fulfillment_status != "unshipped":
                return world.clone(), Result(success=False, error_code="ORDER_NOT_MODIFIABLE")
            order.shipping_address = Address(**p)
            data = {"status": "address_updated", "order_id": key, "shipping_address": dict(p)}
        elif op.name == "update_default_address":
            new.customers[actor].default_address = Address(**p)
            data = {"status": "default_address_updated", "address": dict(p)}
        elif op.name == "modify_order_items":
            order = new.orders[key]
            if _cancelled(new, key):
                return world.clone(), Result(success=False, error_code="ORDER_ALREADY_CANCELLED")
            if order.fulfillment_status != "unshipped":
                return world.clone(), Result(success=False, error_code="ORDER_NOT_MODIFIABLE")
            if order.items_locked:
                return world.clone(), Result(success=False, error_code="ORDER_LOCKED")
            changes = sorted(p["changes"], key=lambda row: row["line_id"])
            if len({row["line_id"] for row in changes}) != len(changes):
                return world.clone(), Result(success=False, error_code="DUPLICATE_LINE_CHANGE")
            if any(row["line_id"] not in new.lines or new.lines[row["line_id"]].order_id != key for row in changes):
                return world.clone(), Result(success=False, error_code="LINE_NOT_IN_ORDER")
            old_total = new_total = 0
            moves = []
            for row in changes:
                line = new.lines[row["line_id"]]
                original = new.skus[line.sku_id]
                if (any(item.line_id == line.id for item in new.settlements.values())
                        or any(item.line_id == line.id and item.status != "cancelled" for item in new.cases.values())):
                    return world.clone(), Result(success=False, error_code="MODIFICATION_CONFLICT")
                target = new.skus.get(row["target_sku_id"])
                if target is None or target.merchant_id != order.merchant_id:
                    return world.clone(), Result(success=False, error_code="EXCHANGE_TARGET_NOT_FOUND")
                if target.product_family != original.product_family:
                    return world.clone(), Result(success=False, error_code="REPLACEMENT_FAMILY_MISMATCH")
                if target.id == original.id:
                    return world.clone(), Result(success=False, error_code="EXCHANGE_TARGET_SAME_AS_ORIGINAL")
                held = new.inventory.get(line.reserved_inventory_id or "")
                if held is None:
                    return world.clone(), Result(success=False, error_code="MODIFICATION_NOT_REPRESENTABLE")
                stock = next((item for item in new.inventory.values()
                              if item.sku_id == target.id and item.warehouse_id == held.warehouse_id), None)
                promised = sum(item[3] for item in moves if item[2].id == stock.id) if stock else 0
                if stock is None or stock.available - promised < line.quantity:
                    return world.clone(), Result(success=False, error_code="INSUFFICIENT_STOCK")
                moves.append((line, target, stock, line.quantity))
                old_total += line.allocated_paid_cents
                new_total += target.price_cents * line.quantity
            delta = new_total - old_total
            refund, extra = max(-delta, 0), max(delta, 0)
            if refund > new.merchants[order.merchant_id].refund_approval_limit_cents:
                return world.clone(), Result(success=False, error_code="APPROVAL_REQUIRED")
            channel, error = _channel(new, order, p["payment_method"], refund=refund > 0, charge=extra)
            if error:
                return world.clone(), Result(success=False, error_code=error)
            records = []
            for line, target, stock, quantity in moves:
                released = new.inventory[line.reserved_inventory_id]
                released.reserved -= quantity
                released.available += quantity
                released.revision += 1
                stock.available -= quantity
                stock.reserved += quantity
                stock.revision += 1
                records.append(ModificationChange(line_id=line.id, from_sku_id=line.sku_id,
                                                  to_sku_id=target.id, inventory_id=stock.id))
                line.sku_id, line.reserved_inventory_id = target.id, stock.id
                line.allocated_paid_cents = target.price_cents * quantity
            order.paid_cents = sum(row.allocated_paid_cents for row in new.lines.values() if row.order_id == key)
            order.items_locked = True
            mid = "modification_" + digest([key, [row.line_id for row in records]])[:20]
            new.modifications[mid] = Modification(
                id=mid, order_id=key, customer_id=actor, changes=tuple(records), old_paid_cents=order.paid_cents - delta,
                new_paid_cents=order.paid_cents, refund_cents=refund, extra_charge_cents=extra, channel=channel)
            _cash(new, channel, refund, extra)
            data = {"status": "order_items_modified", "order_id": key, "refund_cents": refund,
                    "extra_charge_cents": extra, "new_total_cents": order.paid_cents, "payment_channel": channel}
            if new.payment_methods and refund:
                data["refund_timing"] = _timing(new, channel)
        else:
            raise ValueError("unknown commerce action")
        return new, Result(success=True, data=data)
    except (KeyError, ValueError, TypeError, AttributeError, StopIteration):
        return world.clone(), Result(success=False, error_code="MUTATION_NOT_REPRESENTABLE")
