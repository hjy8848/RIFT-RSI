"""v0.2 independent business specification. Never imports Engine transitions."""
from datetime import timedelta

from usami.protocol import RuleDecision
from usami.world.models import Invoice, Membership, Payment, Refund, StateViolation, digest

MEMBER_WRITES = {"invite_member", "activate_member", "deactivate_member", "remove_member", "change_member_role"}
INVOICE_OPS = {"pay_invoice", "refund_invoice", "get_invoice"}
ORG_OPS = {"invite_member", "get_organization", "list_members"}
METHODS = {"card_visa", "card_mastercard", "bank_transfer", "card_declined"}


def renewal_period(world, sub):
    start = sub.period_end if sub.status == "active" and sub.period_end > world.clock else world.clock
    return start, start + timedelta(days=30)


def renewal_id(sub, start):
    return "renewal_" + digest({"subscription": sub.id, "start": start.isoformat()})[:24]


def member_id(org, user):
    return f"membership_{org}_{user}"


def accounting_id(kind, invoice, key):
    return kind + "_" + digest([invoice, key])[:24]


def invariants_v2(world):
    errors = []
    def bad(code, entity, path, detail):
        errors.append(StateViolation(code=code, entity_id=entity, state_path=path, detail=detail))
    if world.clock.tzinfo is None:
        bad("INVALID_CLOCK", world.world_id, "clock", "timezone required")
    for kind in ("users", "organizations", "memberships", "subscriptions", "invoices"):
        for key, obj in getattr(world, kind).items():
            if key != obj.id:
                bad("ENTITY_ID_MISMATCH", key, kind, "key differs from ID")
    for key, plan in world.rules.plans.items():
        if key != plan.id:
            bad("INVALID_PLAN_CONFIG", key, "rules.plans", "plan ID mismatch")
    pairs = set()
    for m in world.memberships.values():
        if m.org_id not in world.organizations or m.user_id not in world.users:
            bad("BROKEN_RELATIONSHIP", m.id, "memberships", "missing principal or tenant")
        if (m.org_id, m.user_id) in pairs:
            bad("DUPLICATE_MEMBERSHIP", m.id, "memberships", "unique tenant/user required")
        pairs.add((m.org_id, m.user_id))
        if m.role == "owner" and (m.status != "active" or m.org_id not in world.organizations or world.organizations[m.org_id].owner_id != m.user_id):
            bad("OWNER_INCONSISTENT", m.id, "memberships", "active designated owner required")
    for org in world.organizations.values():
        owners = [m for m in world.memberships.values() if m.org_id == org.id and m.role == "owner" and m.status == "active"]
        subs = [s for s in world.subscriptions.values() if s.org_id == org.id]
        if len(owners) != 1 or owners[0].user_id != org.owner_id or org.owner_id not in world.users:
            bad("OWNER_INCONSISTENT", org.id, "organizations", "exactly one designated owner")
        if len(subs) != 1 or subs[0].id != org.subscription_id:
            bad("SUBSCRIPTION_INCONSISTENT", org.id, "organizations", "exactly one linked subscription")
    for s in world.subscriptions.values():
        active = sum(m.org_id == s.org_id and m.status == "active" for m in world.memberships.values())
        if s.org_id not in world.organizations:
            bad("BROKEN_RELATIONSHIP", s.id, "subscriptions", "missing tenant")
        if s.plan_id not in world.rules.plans:
            bad("INVALID_PLAN", s.id, "subscriptions.plan_id", "unknown plan")
        elif s.seats > world.rules.plans[s.plan_id].max_seats:
            bad("SEAT_LIMIT_EXCEEDED", s.id, "subscriptions.seats", "above plan cap")
        if s.seats < max(1, active):
            bad("SEATS_BELOW_MEMBERS", s.id, "subscriptions.seats", "below active count")
        if s.period_start.tzinfo is None or s.period_end.tzinfo is None:
            bad("INVALID_CLOCK", s.id, "subscriptions", "timezone required")
        elif not s.period_start < s.period_end or s.period_start > world.clock or (s.status == "expired" and (s.period_end > world.clock or s.auto_renew)):
            bad("SUBSCRIPTION_INCONSISTENT", s.id, "subscriptions", "period/status disagree with frozen clock")
        if s.service_invoice_id:
            inv = world.invoices.get(s.service_invoice_id)
            if not inv or inv.subscription_id != s.id or inv.org_id != s.org_id or inv.purpose != "renewal" or inv.period_end != s.period_end or inv.status not in {"paid", "partially_refunded", "refunded"}:
                bad("BILLING_LINK_INCONSISTENT", s.id, "subscriptions.service_invoice_id", "funding invoice must match the service period")
    periods, accounting_ids = set(), set()
    for inv in world.invoices.values():
        sub = world.subscriptions.get(inv.subscription_id)
        if not sub or inv.org_id != sub.org_id or inv.org_id not in world.organizations:
            bad("BROKEN_RELATIONSHIP", inv.id, "invoices.subscription_id", "invoice and subscription must share tenant")
        if inv.period_start is None or inv.period_end is None or inv.period_start.tzinfo is None or inv.period_end.tzinfo is None or inv.period_end - inv.period_start != timedelta(days=30):
            bad("BILLING_PERIOD_INVALID", inv.id, "invoices", "explicit timezone-aware 30-day period required")
        if inv.purpose not in {"historical", "renewal"} or inv.payments is None:
            bad("BILLING_LINK_INCONSISTENT", inv.id, "invoices", "invoice purpose and payment ledger required")
        if inv.purpose == "renewal":
            p = (inv.subscription_id, inv.period_start, inv.purpose)
            if p in periods:
                bad("DUPLICATE_BILLING_PERIOD", inv.id, "invoices", "one renewal invoice per period")
            periods.add(p)
        payments = inv.payments or []
        keys = set()
        successes = [p for p in payments if p.outcome == "succeeded"]
        for p in payments:
            if p.key in keys or p.id in accounting_ids:
                bad("DUPLICATE_PAYMENT", inv.id, "invoices.payments", "unique idempotency keys and payment IDs")
            keys.add(p.key); accounting_ids.add(p.id)
            if p.amount_cents != inv.amount_cents or p.method not in METHODS or (p.outcome == "failed") != (p.method == "card_declined") or p.attempted_at.tzinfo is None or p.attempted_at > world.clock:
                bad("PAYMENT_INCONSISTENT", inv.id, "invoices.payments", "full payment and simulated outcome/time must agree")
        if len(successes) > 1:
            bad("DUPLICATE_PAYMENT", inv.id, "invoices.payments", "at most one charge per invoice")
        refunded = sum(r.amount_cents for r in inv.refunds)
        paid = inv.status in {"paid", "partially_refunded", "refunded"}
        if paid != (len(successes) == 1) or inv.refundable_amount_cents != inv.amount_cents - refunded or refunded > inv.amount_cents:
            bad("AMOUNT_CONSERVATION", inv.id, "invoices", "one charge = remaining refundable + cumulative refunds")
        expected_status = ("refunded" if refunded == inv.amount_cents else "partially_refunded" if refunded else "paid") if successes else "failed" if payments else "unpaid"
        if inv.status != expected_status or (not successes and (inv.paid_at is not None or inv.payment_method is not None or inv.refunds)):
            bad("INVOICE_INCONSISTENT", inv.id, "invoices", "status and accounting ledger disagree")
        if successes and (inv.paid_at != successes[0].attempted_at or inv.payment_method != successes[0].method):
            bad("INVOICE_INCONSISTENT", inv.id, "invoices", "original charge metadata must agree")
        rkeys = set()
        for r in inv.refunds:
            if not r.key or r.key in rkeys or r.id in accounting_ids:
                bad("DUPLICATE_REFUND", inv.id, "invoices.refunds", "unique scoped refund key required")
            rkeys.add(r.key); accounting_ids.add(r.id)
            if r.amount_cents <= 0 or r.payment_method != inv.payment_method:
                bad("REFUND_INCONSISTENT", inv.id, "invoices.refunds", "positive refund to original method")
            if not inv.paid_at or r.refunded_at.tzinfo is None or not inv.paid_at <= r.refunded_at <= world.clock or r.refunded_at - inv.paid_at > timedelta(days=world.rules.refund_window_days):
                bad("REFUND_WINDOW_EXPIRED", inv.id, "invoices.refunds", "refund time outside payment window")
    return errors


def decide_v2(world, actor, op, event_id):
    def deny(code):
        return RuleDecision(allowed=False, error_code=code)
    if invariants_v2(world): return deny("INVALID_WORLD")
    if actor not in world.users: return deny("UNKNOWN_ACTOR")
    if op.org_id not in world.organizations: return deny("ORG_NOT_FOUND")
    member = next((m for m in world.memberships.values() if m.org_id == op.org_id and m.user_id == actor and m.status == "active"), None)
    if not member: return deny("CROSS_ORG_ACCESS")
    name, rid, args = op.name, op.resource_id, op.params
    collection = world.organizations if name in ORG_OPS else world.invoices if name in INVOICE_OPS else world.memberships if name in MEMBER_WRITES else world.subscriptions
    obj = collection.get(rid)
    if not obj: return deny("RESOURCE_NOT_FOUND")
    scope = obj.id if name in ORG_OPS else obj.org_id
    if scope != op.org_id: return deny("CROSS_ORG_RESOURCE")
    if name in MEMBER_WRITES and member.role != "owner": return deny("PERMISSION_DENIED")
    if (not op.is_query or name == "get_invoice") and member.role == "member": return deny("PERMISSION_DENIED")
    new, data = world.clone(), {}
    sub = world.subscriptions[world.organizations[op.org_id].subscription_id]
    active = sum(m.org_id == op.org_id and m.status == "active" for m in world.memberships.values())
    if name == "invite_member":
        if args["user_id"] not in world.users: return deny("USER_NOT_FOUND")
        found = next((m for m in world.memberships.values() if m.org_id == op.org_id and m.user_id == args["user_id"]), None)
        if found:
            if found.status != "pending" or found.role != args.get("role", "member"): return deny("DUPLICATE_INVITATION")
            data = {"membership_id": found.id, "status": "pending"}
        else:
            mid = member_id(op.org_id, args["user_id"])
            new.memberships[mid] = Membership(id=mid, org_id=op.org_id, user_id=args["user_id"], role=args.get("role", "member"), status="pending")
            data = {"membership_id": mid, "status": "pending"}
    elif name in {"activate_member", "deactivate_member", "remove_member", "change_member_role"}:
        if obj.role == "owner" or (name == "change_member_role" and args["role"] == "owner"): return deny("OWNER_IMMUTABLE")
        if name == "activate_member":
            if obj.status == "active": return deny("MEMBERSHIP_ALREADY_ACTIVE")
            if sub.status != "active" or sub.period_end <= world.clock: return deny("SUBSCRIPTION_NOT_ACTIVE")
            if active >= sub.seats: return deny("NO_AVAILABLE_SEATS")
            new.memberships[rid].status = "active"
        elif name == "deactivate_member":
            if obj.status != "active": return deny("MEMBERSHIP_NOT_ACTIVE")
            new.memberships[rid].status = "inactive"
        elif name == "remove_member":
            if obj.status == "active": return deny("MEMBERSHIP_STILL_ACTIVE")
            del new.memberships[rid]
        else:
            if obj.status != "active": return deny("MEMBERSHIP_NOT_ACTIVE")
            new.memberships[rid].role = args["role"]
    elif name in {"change_plan", "change_seats", "cancel_renewal"}:
        if obj.status != "active" or obj.period_end <= world.clock: return deny("SUBSCRIPTION_NOT_ACTIVE")
        if name == "change_plan":
            target = world.rules.plans.get(args["plan_id"])
            if not target: return deny("INVALID_PLAN")
            if obj.seats > target.max_seats: return deny("SEAT_LIMIT_EXCEEDED")
            new.subscriptions[rid].plan_id = target.id
        elif name == "change_seats":
            if args["seats"] < max(1, active): return deny("SEATS_BELOW_MEMBERS")
            if args["seats"] > world.rules.plans[obj.plan_id].max_seats: return deny("SEAT_LIMIT_EXCEEDED")
            new.subscriptions[rid].seats = args["seats"]
        else: new.subscriptions[rid].auto_renew = False
    elif name == "issue_renewal":
        start, end = renewal_period(world, obj)
        iid = renewal_id(obj, start)
        if iid not in new.invoices:
            amount = (1000 + world.rules.plans[obj.plan_id].rank * 500) * obj.seats
            new.invoices[iid] = Invoice(id=iid, org_id=op.org_id, subscription_id=rid, period_start=start, period_end=end, purpose="renewal", payments=[], amount_cents=amount, refundable_amount_cents=amount, status="unpaid")
        data = {"invoice_id": iid, "period_start": start.isoformat(), "period_end": end.isoformat()}
    elif name == "pay_invoice":
        key, method = args["idempotency_key"], args["payment_method"]
        prior = next((p for p in obj.payments if p.key == key), None)
        if prior:
            if prior.method != method: return deny("IDEMPOTENCY_CONFLICT")
            data = {"payment_id": prior.id, "outcome": prior.outcome}
        else:
            if obj.status not in {"unpaid", "failed"}: return deny("INVOICE_ALREADY_PAID")
            outcome = "failed" if method == "card_declined" else "succeeded"
            p = Payment(id=accounting_id("payment", rid, key), key=key, method=method, amount_cents=obj.amount_cents, outcome=outcome, attempted_at=world.clock)
            inv = new.invoices[rid]; inv.payments.append(p)
            if outcome == "succeeded":
                inv.status = "paid"; inv.paid_at = world.clock; inv.payment_method = method
            else: inv.status = "failed"
            data = {"payment_id": p.id, "outcome": outcome}
    elif name in {"renew_subscription", "reactivate_subscription"}:
        if (name == "renew_subscription" and obj.status != "active") or (name == "reactivate_subscription" and obj.status != "expired"): return deny("SUBSCRIPTION_STATUS_MISMATCH")
        if name == "renew_subscription" and obj.period_end <= world.clock: return deny("PERIOD_ENDED")
        inv = world.invoices.get(args["invoice_id"])
        if not inv: return deny("INVOICE_NOT_FOUND")
        if inv.org_id != op.org_id or inv.subscription_id != rid: return deny("CROSS_ORG_RESOURCE")
        start, end = renewal_period(world, obj)
        if inv.purpose != "renewal" or inv.period_start != start or inv.period_end != end: return deny("BILLING_PERIOD_MISMATCH")
        if inv.status != "paid" or inv.refunds: return deny("INVOICE_NOT_PAID")
        s = new.subscriptions[rid]
        if name == "reactivate_subscription": s.period_start = start
        s.period_end = end; s.status = "active"; s.service_invoice_id = inv.id
    elif name == "expire_subscription":
        if obj.period_end > world.clock: return deny("PERIOD_NOT_ENDED")
        s = new.subscriptions[rid]; s.status = "expired"; s.auto_renew = False
    elif name == "refund_invoice":
        key, amount, method = args.get("idempotency_key"), args["amount_cents"], args["payment_method"]
        if not key: return deny("REFUND_KEY_REQUIRED")
        prior = next((r for r in obj.refunds if r.key == key), None)
        if prior:
            if prior.amount_cents != amount or prior.payment_method != method: return deny("IDEMPOTENCY_CONFLICT")
            data = {"refund_id": prior.id, "amount_cents": prior.amount_cents}
        else:
            if obj.status not in {"paid", "partially_refunded"}: return deny("INVOICE_NOT_PAID")
            if world.clock - obj.paid_at > timedelta(days=world.rules.refund_window_days): return deny("REFUND_WINDOW_EXPIRED")
            if amount <= 0 or amount > obj.refundable_amount_cents: return deny("REFUND_AMOUNT_EXCEEDED")
            if method != obj.payment_method: return deny("REFUND_METHOD_MISMATCH")
            r = Refund(id=accounting_id("refund", rid, key), key=key, amount_cents=amount, payment_method=method, refunded_at=world.clock)
            inv = new.invoices[rid]; inv.refunds.append(r); inv.refundable_amount_cents -= amount
            inv.status = "refunded" if inv.refundable_amount_cents == 0 else "partially_refunded"
            data = {"refund_id": r.id, "amount_cents": amount}
    elif name == "list_members":
        roster = [m for m in world.memberships.values() if m.org_id == rid]
        data = {"memberships": [m.model_dump(mode="json") for m in roster],
                "users": [world.users[m.user_id].model_dump(mode="json") for m in roster],
                "directory_users": [{"id": u.id, "name": u.name} for u in world.users.values()]}
    else: data = obj.model_dump(mode="json")
    if name == "get_organization": data["invoice_ids"] = [inv.id for inv in world.invoices.values() if inv.org_id == op.org_id]
    return RuleDecision(allowed=True, expected_state=new, expected_data=data)


def public_policy_catalog():
    """Same public codebook for every world/task; never a hidden task certificate."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(decide_v2))
    codes = sorted({node.args[0].value for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'deny' and node.args and isinstance(node.args[0], ast.Constant)})
    return {"error_codes": codes, "policy": {
        "PUBLIC_DIRECTORY": "Registered user IDs and display names form a public directory. Emails are visible only in your own profile or an accessible tenant roster. Tenant memberships, subscriptions and invoice details remain scoped.",
        "OWNER_IMMUTABLE": "The sole owner cannot be suspended, removed, assigned or transferred by member tools.",
        "PERMISSION_DENIED": "Only owners manage memberships. Owners/billing administrators manage invoices and subscriptions; ordinary members can query nonfinancial tenant information.",
        "CROSS_ORG_ACCESS": "Actor needs an active membership in the requested organization.",
        "CROSS_ORG_RESOURCE": "Every target and funding invoice must belong to the requested organization/subscription.",
        "NO_AVAILABLE_SEATS": "Activation requires unused seats and a live service period.",
        "SEATS_BELOW_MEMBERS": "Seats cannot be reduced below active membership count.",
        "SEAT_LIMIT_EXCEEDED": "Seats must fit the current or proposed plan.",
        "REFUND_AMOUNT_EXCEEDED": "Refund must be positive and no greater than the remaining paid refundable balance.",
        "REFUND_WINDOW_EXPIRED": "Refunds are limited to the configured window after payment.",
        "IDEMPOTENCY_CONFLICT": "A scoped payment/refund key cannot be reused with different parameters.",
        "BILLING_PERIOD_MISMATCH": "Renewal/restoration must use the linked next 30-day service period, paid without refunds.",
        "SUBSCRIPTION_NOT_ACTIVE": "Capacity and activation writes require active status and period_end after frozen time.",
        "PERIOD_ENDED": "An elapsed service period must be expired and reactivated, not renewed as a live contract."
    }}
