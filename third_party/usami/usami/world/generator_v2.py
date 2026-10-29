"""Seeded, versioned workflow-conditioned initial worlds, not solution scripts."""
from datetime import timedelta
from usami.world.models import Invoice, Membership, Payment, Refund, RulesConfig, digest

WORKFLOWS = (
    "seat_activate", "deactivate_reduce", "seat_downgrade", "plan_expand",
    "invite_activate", "owner_refusal", "failed_restore", "unpaid_restore",
    "renew_cycle", "payment_retry", "partial_refund", "excessive_refund",
    "two_org_budget", "restore_onboard", "offboard_downgrade", "upgrade_onboard",
    "invitation_resolve", "remove_departed", "restore_upgrade_onboard", "offboard_refund",
    "tenant_refusal",
    "expire_cycle",
)


def generate_v2(seed, config):
    from usami.world.generator import GeneratorConfig, generate_world
    from usami.world.rules import check_invariants
    from usami.world.rules_v2 import accounting_id, renewal_id
    if config.user_count < 8: raise ValueError("v0.2 requires at least eight directory users")
    if config.workflow is not None and config.workflow not in WORKFLOWS: raise ValueError("unknown workflow")
    legacy = GeneratorConfig(user_count=config.user_count, organization_count=config.organization_count,
        clock=config.clock, profile=config.profile, rules=RulesConfig(plans=config.rules.plans, refund_window_days=config.rules.refund_window_days))
    world = generate_world(seed, legacy)
    world.rules = config.rules.model_copy(deep=True)
    world.generator_version = config.version; world.config_hash = digest(config)
    world.world_id = "world_" + digest({"seed": seed, "config": world.config_hash})[:20]
    for inv in world.invoices.values():
        sub = world.subscriptions[world.organizations[inv.org_id].subscription_id]
        inv.subscription_id = sub.id; inv.period_start = sub.period_start - timedelta(days=60)
        inv.period_end = inv.period_start + timedelta(days=30); inv.purpose = "historical"; inv.payments = []
        if inv.paid_at:
            inv.payments = [Payment(id=accounting_id("payment", inv.id, "original"), key="original", amount_cents=inv.amount_cents, method=inv.payment_method, outcome="succeeded", attempted_at=inv.paid_at)]
        if inv.refunds:
            inv.refunds[0].key = "original"; inv.refunds[0].id = accounting_id("refund", inv.id, "original")
            inv.refundable_amount_cents = 0
    if config.workflow:
        family = config.workflow; oid = "org_000"; sub = world.subscriptions["sub_000"]
        for mid in [m.id for m in world.memberships.values() if m.org_id == oid]: del world.memberships[mid]
        count = 6 if family in {"offboard_downgrade", "offboard_refund"} else 3 + seed % 2
        for n in range(count):
            uid = f"user_{n:03d}"; mid = f"membership_{oid}_{uid}"
            world.memberships[mid] = Membership(id=mid, org_id=oid, user_id=uid, role="owner" if n == 0 else "billing_admin" if n == 2 else "member", status="active")
        for n, status in ((6, "pending"), (7, "inactive")):
            uid = f"user_{n:03d}"; mid = f"membership_{oid}_{uid}"
            world.memberships[mid] = Membership(id=mid, org_id=oid, user_id=uid, role="member", status=status)
        if family in {"seat_activate", "restore_onboard", "upgrade_onboard", "restore_upgrade_onboard"} and seed % 3 == 1:
            world.memberships[f"membership_{oid}_user_006"].status = "inactive"
        if family in {"partial_refund", "excessive_refund", "offboard_refund"} and seed % 3 == 1:
            original = world.invoices["invoice_000_recent"]
            amount = max(1, original.amount_cents // 5)
            original.refunds.append(Refund(id=accounting_id("refund", original.id, "previous-adjustment"), key="previous-adjustment", amount_cents=amount, payment_method=original.payment_method, refunded_at=world.clock))
            original.refundable_amount_cents -= amount; original.status = "partially_refunded"
        sub.plan_id = "team" if family in {"seat_downgrade", "offboard_downgrade", "offboard_refund"} else "starter"
        sub.seats = 8 if sub.plan_id == "team" else count
        if family in {"invite_activate", "invitation_resolve"}: sub.seats = 5
        if family in {"upgrade_onboard", "restore_upgrade_onboard", "plan_expand"}:
            # Owner + four existing members fills Starter, forcing plan -> seats.
            for n in range(5):
                uid = f"user_{n:03d}"; mid = f"membership_{oid}_{uid}"
                world.memberships[mid] = Membership(id=mid, org_id=oid, user_id=uid, role="owner" if n == 0 else "billing_admin" if n == 2 else "member", status="active")
            sub.seats = 5
        expired = family in {"failed_restore", "unpaid_restore", "restore_onboard", "restore_upgrade_onboard"}
        sub.status = "expired" if expired else "active"; sub.auto_renew = False
        sub.period_end = world.clock - timedelta(days=1 + seed % 5) if expired or family == "expire_cycle" else world.clock + timedelta(days=1 + seed % 20)
        sub.period_start = sub.period_end - timedelta(days=30)
        start = world.clock if expired or family == "expire_cycle" else sub.period_end
        iid = renewal_id(sub, start)
        if family not in {"renew_cycle"}:
            amount = (1000 + world.rules.plans[sub.plan_id].rank * 500) * sub.seats
            inv = Invoice(id=iid, org_id=oid, subscription_id=sub.id, purpose="renewal", period_start=start, period_end=start + timedelta(days=30), amount_cents=amount, refundable_amount_cents=amount, status="unpaid", payments=[])
            if family == "failed_restore":
                inv.payments.append(Payment(id=accounting_id("payment", iid, "declined"), key="declined", amount_cents=amount, method="card_declined", outcome="failed", attempted_at=world.clock)); inv.status = "failed"
            if family == "payment_retry":
                inv.payments.append(Payment(id=accounting_id("payment", iid, "confirmed"), key="confirmed", amount_cents=amount, method="card_visa", outcome="succeeded", attempted_at=world.clock)); inv.status = "paid"; inv.paid_at = world.clock; inv.payment_method = "card_visa"
            world.invoices[iid] = inv
        if family == "two_org_budget":
            sub.seats += 1
            org = world.organizations["org_001"]
            existing = next((m for m in world.memberships.values() if m.org_id == org.id and m.user_id == "user_000"), None)
            if existing: existing.role = "billing_admin"; existing.status = "active"
            else:
                mid = f"membership_{org.id}_user_000"; world.memberships[mid] = Membership(id=mid, org_id=org.id, user_id="user_000", role="billing_admin")
            s = world.subscriptions[org.subscription_id]; s.plan_id = "business"; s.seats = 20; s.status = "active"; s.period_start = world.clock - timedelta(days=10); s.period_end = world.clock + timedelta(days=20); s.auto_renew = False
    else:
        from usami.randomness import StableRandom
        rng = StableRandom(seed + 90210)
        for member in world.memberships.values():
            if member.role != "owner" and not (config.profile == "test_composed" and member.user_id == "user_002" and member.org_id in {"org_000", "org_001"}) and rng.random() < 0.2:
                member.status = "pending"
        for inv in world.invoices.values():
            if inv.status == "unpaid" and rng.random() < 0.5:
                inv.status = "failed"
                inv.payments.append(Payment(id=accounting_id("payment", inv.id, "declined"), key="declined", amount_cents=inv.amount_cents, method="card_declined", outcome="failed", attempted_at=world.clock))
            elif inv.status == "paid" and world.clock - inv.paid_at <= timedelta(days=world.rules.refund_window_days) and rng.random() < 0.4:
                amount = max(1, inv.amount_cents // 4)
                inv.refunds.append(Refund(id=accounting_id("refund", inv.id, "adjustment"), key="adjustment", amount_cents=amount, payment_method=inv.payment_method, refunded_at=world.clock))
                inv.refundable_amount_cents -= amount; inv.status = "partially_refunded"
    errors = check_invariants(world)
    if errors: raise ValueError(f"invalid v0.2 generated world: {errors}")
    return world
