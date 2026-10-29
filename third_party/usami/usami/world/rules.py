"""Pure reference specification. This module never imports the sandbox engine."""
from __future__ import annotations

from datetime import timedelta
from typing import List, Optional

from usami.protocol import Operation, RuleDecision
from usami.world.models import Membership, Refund, RULES_VERSION, StateViolation, WorldState


def membership_for(world: WorldState, user_id: str, org_id: str) -> Optional[Membership]:
    return next((m for m in world.memberships.values()
                 if m.user_id == user_id and m.org_id == org_id and m.status == "active"), None)


def check_invariants(world: WorldState) -> List[StateViolation]:
    if world.rules.version == "saas-rules-0.2.0":
        from usami.world.rules_v2 import invariants_v2
        return invariants_v2(world)
    errors: List[StateViolation] = []

    def bad(code: str, entity: str, path: str, detail: str) -> None:
        errors.append(StateViolation(code=code, entity_id=entity, state_path=path, detail=detail))

    for collection in ("users", "organizations", "memberships", "subscriptions", "invoices"):
        for key, entity in getattr(world, collection).items():
            if key != entity.id:
                bad("ENTITY_ID_MISMATCH", key, f"{collection}.{key}.id", "key and entity ID differ")
    for key, plan in world.rules.plans.items():
        if key != plan.id:
            bad("INVALID_PLAN_CONFIG", key, f"rules.plans.{key}", "plan ID must match key")
    seen = set()
    for m in world.memberships.values():
        path = f"memberships.{m.id}"
        if m.status not in {"active", "inactive"}:
            bad("UNSUPPORTED_STATE_VERSION", m.id, path, "pending memberships require v0.2 rules")
        if m.user_id not in world.users or m.org_id not in world.organizations:
            bad("BROKEN_RELATIONSHIP", m.id, path, "membership refers to missing user or org")
        if m.role == "owner" and (m.status != "active" or m.org_id not in world.organizations
                                 or world.organizations[m.org_id].owner_id != m.user_id):
            bad("OWNER_INCONSISTENT", m.id, path, "owner membership must be active and match the organization owner")
        pair = (m.user_id, m.org_id)
        if pair in seen:
            bad("DUPLICATE_MEMBERSHIP", m.id, path, "one membership per user and organization")
        seen.add(pair)
    for org in world.organizations.values():
        owners = [m for m in world.memberships.values()
                  if m.org_id == org.id and m.role == "owner" and m.status == "active"]
        if len(owners) != 1 or owners[0].user_id != org.owner_id or org.owner_id not in world.users:
            bad("OWNER_INCONSISTENT", org.id, f"organizations.{org.id}.owner_id", "exactly one active owner")
        subs = [s for s in world.subscriptions.values() if s.org_id == org.id]
        if len(subs) != 1 or subs[0].id != org.subscription_id:
            bad("SUBSCRIPTION_INCONSISTENT", org.id, f"organizations.{org.id}.subscription_id", "exactly one linked subscription")
    for s in world.subscriptions.values():
        path = f"subscriptions.{s.id}"
        if s.service_invoice_id is not None:
            bad("UNSUPPORTED_STATE_VERSION", s.id, path, "funding linkage requires v0.2 rules")
        if s.org_id not in world.organizations:
            bad("BROKEN_RELATIONSHIP", s.id, path, "missing organization")
        plan = world.rules.plans.get(s.plan_id)
        if plan is None:
            bad("INVALID_PLAN", s.id, path + ".plan_id", "plan not configured")
        elif s.seats > plan.max_seats:
            bad("SEAT_LIMIT_EXCEEDED", s.id, path + ".seats", "above plan limit")
        members = sum(m.org_id == s.org_id and m.status == "active" for m in world.memberships.values())
        if s.seats < max(1, members):
            bad("SEATS_BELOW_MEMBERS", s.id, path + ".seats", "insufficient seats for active members")
        if s.period_start.tzinfo is None or s.period_end.tzinfo is None:
            bad("INVALID_CLOCK", s.id, path, "period must include timezones")
        elif not s.period_start < s.period_end or s.period_start > world.clock:
            bad("SUBSCRIPTION_INCONSISTENT", s.id, path, "invalid cycle")
        elif ((s.status == "active" and world.clock >= s.period_end)
              or (s.status == "expired" and (world.clock < s.period_end or s.auto_renew))):
            bad("SUBSCRIPTION_INCONSISTENT", s.id, path, "status/renewal disagrees with frozen clock")
    refund_ids = set()
    for inv in world.invoices.values():
        path = f"invoices.{inv.id}"
        if any(getattr(inv, field) is not None for field in ("subscription_id", "period_start", "period_end", "purpose", "payments")) or any(r.key is not None for r in inv.refunds):
            bad("UNSUPPORTED_STATE_VERSION", inv.id, path, "linked accounting requires v0.2 rules")
        if inv.org_id not in world.organizations:
            bad("BROKEN_RELATIONSHIP", inv.id, path, "missing organization")
        if inv.refundable_amount_cents != inv.amount_cents:
            bad("INVALID_REFUNDABLE_AMOUNT", inv.id, path, "v0.1 permits full invoice refunds only")
        if inv.status == "unpaid":
            if inv.paid_at is not None or inv.payment_method is not None or inv.refunds:
                bad("INVOICE_INCONSISTENT", inv.id, path, "unpaid invoice has payment or refund data")
        elif inv.paid_at is None or inv.payment_method is None:
            bad("INVOICE_INCONSISTENT", inv.id, path, "paid invoice lacks original payment data")
        elif inv.paid_at.tzinfo is None or inv.paid_at > world.clock:
            bad("INVALID_PAYMENT_TIME", inv.id, path + ".paid_at", "payment must be timezone aware and not future")
        if (inv.status == "refunded") != bool(inv.refunds):
            bad("INVOICE_INCONSISTENT", inv.id, path, "status disagrees with refund history")
        if len(inv.refunds) > 1:
            bad("DUPLICATE_REFUND", inv.id, path + ".refunds", "invoice refunded more than once")
        for refund in inv.refunds:
            rp = path + ".refunds." + refund.id
            if refund.id in refund_ids:
                bad("DUPLICATE_REFUND_ID", inv.id, rp, "refund ID is not unique")
            refund_ids.add(refund.id)
            if refund.amount_cents != inv.refundable_amount_cents:
                bad("REFUND_AMOUNT_MISMATCH", inv.id, rp, "must refund full available amount")
            if refund.payment_method != inv.payment_method:
                bad("REFUND_METHOD_MISMATCH", inv.id, rp, "must return to original payment method")
            if refund.refunded_at.tzinfo is None:
                bad("INVALID_CLOCK", inv.id, rp, "refund time must include timezone")
            elif (inv.paid_at is None or inv.paid_at.tzinfo is None
                  or not inv.paid_at <= refund.refunded_at <= world.clock
                  or refund.refunded_at - inv.paid_at > timedelta(days=world.rules.refund_window_days)):
                bad("REFUND_WINDOW_EXPIRED", inv.id, rp, "refund outside allowed payment window")
    return errors


class ReferenceRules:
    def decide(self, world: WorldState, actor_id: str, operation: Operation,
               event_id: str = "reference") -> RuleDecision:
        if world.rules.version == "saas-rules-0.2.0":
            from usami.world.rules_v2 import decide_v2
            return decide_v2(world, actor_id, operation, event_id)
        def reject(code: str, detail: str = "") -> RuleDecision:
            return RuleDecision(allowed=False, error_code=code, detail=detail)

        if world.rules.version != RULES_VERSION:
            return reject("UNSUPPORTED_RULES_VERSION")
        if operation.name not in {"change_plan", "change_seats", "cancel_renewal", "refund_invoice", "change_member_role", "get_organization", "get_subscription", "get_invoice", "list_members"} or operation.params.get("idempotency_key") is not None:
            return reject("UNSUPPORTED_OPERATION_VERSION")
        if check_invariants(world):
            return reject("INVALID_WORLD")
        if actor_id not in world.users:
            return reject("UNKNOWN_ACTOR")
        if operation.org_id not in world.organizations:
            return reject("ORG_NOT_FOUND")
        member = membership_for(world, actor_id, operation.org_id)
        if member is None:
            return reject("CROSS_ORG_ACCESS")
        name = operation.name
        if name in {"get_organization", "list_members"}:
            resource = world.organizations.get(operation.resource_id)
        elif name in {"refund_invoice", "get_invoice"}:
            resource = world.invoices.get(operation.resource_id)
        elif name == "change_member_role":
            resource = world.memberships.get(operation.resource_id)
        else:
            resource = world.subscriptions.get(operation.resource_id)
        if resource is None:
            return reject("RESOURCE_NOT_FOUND")
        resource_org = resource.id if name in {"get_organization", "list_members"} else resource.org_id
        if resource_org != operation.org_id:
            return reject("CROSS_ORG_RESOURCE")
        if name == "change_member_role":
            if member.role != "owner":
                return reject("PERMISSION_DENIED")
        elif (not operation.is_query or name == "get_invoice") and member.role == "member":
            return reject("PERMISSION_DENIED")

        new = world.clone()
        data = {}
        if name == "change_plan":
            target = world.rules.plans.get(operation.params["plan_id"])
            if target is None:
                return reject("INVALID_PLAN")
            if resource.status != "active":
                return reject("SUBSCRIPTION_NOT_ACTIVE")
            if resource.seats > target.max_seats:
                return reject("SEAT_LIMIT_EXCEEDED")
            new.subscriptions[resource.id].plan_id = target.id
        elif name == "change_seats":
            if resource.status != "active":
                return reject("SUBSCRIPTION_NOT_ACTIVE")
            seats = operation.params["seats"]
            active_count = sum(m.org_id == operation.org_id and m.status == "active"
                               for m in world.memberships.values())
            if seats < max(1, active_count):
                return reject("SEATS_BELOW_MEMBERS")
            if seats > world.rules.plans[resource.plan_id].max_seats:
                return reject("SEAT_LIMIT_EXCEEDED")
            new.subscriptions[resource.id].seats = seats
        elif name == "cancel_renewal":
            if resource.status != "active":
                return reject("SUBSCRIPTION_NOT_ACTIVE")
            new.subscriptions[resource.id].auto_renew = False
        elif name == "refund_invoice":
            if resource.status == "refunded" or resource.refunds:
                return reject("DUPLICATE_REFUND")
            if resource.status != "paid":
                return reject("INVOICE_NOT_PAID")
            if world.clock - resource.paid_at > timedelta(days=world.rules.refund_window_days):
                return reject("REFUND_WINDOW_EXPIRED")
            if operation.params["amount_cents"] != resource.refundable_amount_cents:
                return reject("REFUND_AMOUNT_MISMATCH")
            if operation.params["payment_method"] != resource.payment_method:
                return reject("REFUND_METHOD_MISMATCH")
            inv = new.invoices[resource.id]
            inv.status = "refunded"
            inv.refunds.append(Refund(id=f"refund_{event_id}", amount_cents=inv.refundable_amount_cents,
                                      payment_method=inv.payment_method, refunded_at=world.clock))
        elif name == "change_member_role":
            if resource.role == "owner" or operation.params["role"] == "owner":
                return reject("OWNER_IMMUTABLE")
            if resource.status != "active":
                return reject("MEMBERSHIP_NOT_ACTIVE")
            new.memberships[resource.id].role = operation.params["role"]
        elif name == "get_organization":
            data = resource.model_dump(mode="json")
        elif name == "get_subscription":
            data = resource.model_dump(mode="json")
        elif name == "get_invoice":
            data = resource.model_dump(mode="json")
        elif name == "list_members":
            data = {"memberships": [m.model_dump(mode="json") for m in world.memberships.values()
                                     if m.org_id == operation.org_id]}
        return RuleDecision(allowed=True, expected_state=new, expected_data=data)
