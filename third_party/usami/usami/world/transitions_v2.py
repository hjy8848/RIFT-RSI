"""Actual v0.2 Engine mutations; deliberately independent of reference decisions.

Reference mode checks policy first. Sandbox uses existence/schema checks only;
the Oracle, never this module, determines whether a successful action was legal.
"""
from datetime import timedelta
from usami.protocol import ExecutionResult
from usami.world.models import Invoice, Membership, Payment, Refund, digest


def apply_v2(world, actor, op, event_id):
    state = world.clone()
    def reject(code): return state, ExecutionResult(success=False, error_code=code)
    if actor not in state.users: return reject("UNKNOWN_ACTOR")
    if op.org_id not in state.organizations: return reject("ORG_NOT_FOUND")
    name, rid, p = op.name, op.resource_id, op.params
    if name in {"invite_member", "get_organization", "list_members"}: objects = state.organizations
    elif name in {"refund_invoice", "pay_invoice", "get_invoice"}: objects = state.invoices
    elif name in {"activate_member", "deactivate_member", "remove_member", "change_member_role"}: objects = state.memberships
    else: objects = state.subscriptions
    obj = objects.get(rid)
    if obj is None: return reject("RESOURCE_NOT_FOUND")
    data = {}
    if name == "invite_member":
        if p["user_id"] not in state.users: return reject("USER_NOT_FOUND")
        existing = next((m for m in state.memberships.values() if m.org_id == op.org_id and m.user_id == p["user_id"]), None)
        mid = existing.id if existing else f"membership_{op.org_id}_{p['user_id']}"
        if not existing:
            state.memberships[mid] = Membership(id=mid, org_id=op.org_id, user_id=p["user_id"], role=p.get("role", "member"), status="pending")
        data = {"membership_id": mid, "status": state.memberships[mid].status}
    elif name == "activate_member": obj.status = "active"
    elif name == "deactivate_member": obj.status = "inactive"
    elif name == "remove_member": del state.memberships[rid]
    elif name == "change_member_role": obj.role = p["role"]
    elif name == "change_plan": obj.plan_id = p["plan_id"]
    elif name == "change_seats": obj.seats = p["seats"]
    elif name == "cancel_renewal": obj.auto_renew = False
    elif name == "expire_subscription": obj.status = "expired"; obj.auto_renew = False
    elif name == "issue_renewal":
        start = obj.period_end if obj.status == "active" and obj.period_end > state.clock else state.clock
        end = start + timedelta(days=30)
        iid = "renewal_" + digest({"subscription": obj.id, "start": start.isoformat()})[:24]
        if iid not in state.invoices:
            amount = (1000 + state.rules.plans[obj.plan_id].rank * 500) * obj.seats
            state.invoices[iid] = Invoice(id=iid, org_id=op.org_id, subscription_id=rid, period_start=start, period_end=end, purpose="renewal", payments=[], amount_cents=amount, refundable_amount_cents=amount, status="unpaid")
        data = {"invoice_id": iid, "period_start": start.isoformat(), "period_end": end.isoformat()}
    elif name == "pay_invoice":
        if obj.payments is None: return reject("VERSION_MISMATCH")
        prior = next((a for a in obj.payments if a.key == p["idempotency_key"]), None)
        if prior:
            data = {"payment_id": prior.id, "outcome": prior.outcome}
        else:
            outcome = "failed" if p["payment_method"] == "card_declined" else "succeeded"
            item = Payment(id="payment_" + digest([rid, p["idempotency_key"]])[:24], key=p["idempotency_key"], method=p["payment_method"], amount_cents=obj.amount_cents, outcome=outcome, attempted_at=state.clock)
            obj.payments.append(item)
            if outcome == "succeeded": obj.status = "paid"; obj.paid_at = state.clock; obj.payment_method = item.method
            else: obj.status = "failed"
            data = {"payment_id": item.id, "outcome": outcome}
    elif name in {"renew_subscription", "reactivate_subscription"}:
        invoice = state.invoices.get(p["invoice_id"])
        if invoice is None: return reject("INVOICE_NOT_FOUND")
        if name == "reactivate_subscription": obj.period_start = invoice.period_start
        obj.period_end = invoice.period_end; obj.status = "active"; obj.service_invoice_id = invoice.id
    elif name == "refund_invoice":
        if not p.get("idempotency_key"): return reject("REFUND_KEY_REQUIRED")
        prior = next((r for r in obj.refunds if r.key == p["idempotency_key"]), None)
        if prior: data = {"refund_id": prior.id, "amount_cents": prior.amount_cents}
        else:
            item = Refund(id="refund_" + digest([rid, p["idempotency_key"]])[:24], key=p["idempotency_key"], amount_cents=p["amount_cents"], payment_method=p["payment_method"], refunded_at=state.clock)
            obj.refunds.append(item)
            # Sandbox must represent an over-refund without crashing schema validation.
            obj.refundable_amount_cents = max(0, obj.amount_cents - sum(r.amount_cents for r in obj.refunds))
            obj.status = "refunded" if obj.refundable_amount_cents == 0 else "partially_refunded"
            data = {"refund_id": item.id, "amount_cents": item.amount_cents}
    elif name == "list_members":
        records = [m for m in state.memberships.values() if m.org_id == rid]
        data = {"memberships": [m.model_dump(mode="json") for m in records],
                "users": [state.users[m.user_id].model_dump(mode="json") for m in records],
                "directory_users": [{"id": u.id, "name": u.name} for u in state.users.values()]}
    else: data = obj.model_dump(mode="json")
    if name == "get_organization": data["invoice_ids"] = [inv.id for inv in state.invoices.values() if inv.org_id == op.org_id]
    return state, ExecutionResult(success=True, data=data)
