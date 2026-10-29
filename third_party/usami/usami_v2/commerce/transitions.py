"""Actual state mutations. Deliberately never imports the Reference."""
from datetime import timedelta

from usami_harness.models import Result
from .models import (
    Fulfillment, LedgerEntry, Quote, RemedyCase, Return, Settlement,
    case_id, quote_id, unit_credit,
)


def _hard_scope(world, actor, op):
    if actor not in world.customers or op.scope_id not in world.merchants:
        return False
    if op.name in ("list_customer_orders", "search_catalog"):
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
    if op.name in ("get_order_details",):
        order = world.orders.get(key)
    elif op.name in ("open_remedy_case", "get_refundable_balance"):
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
    sid = "settlement_" + case.id
    world.settlements[sid] = Settlement(id=sid, case_id=case.id, line_id=line.id,
                                       quote_id=quote.id, unit_indices=case.unit_indices, remedy=quote.remedy)
    case.status = "resolved"


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
        if op.name == "open_remedy_case":
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
        else:
            raise ValueError("unknown commerce action")
        return new, Result(success=True, data=data)
    except (KeyError, ValueError, TypeError, AttributeError, StopIteration):
        return world.clone(), Result(success=False, error_code="MUTATION_NOT_REPRESENTABLE")
