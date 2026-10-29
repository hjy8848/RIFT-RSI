"""Structured situations derived from worlds and labeled only by reference rules."""
from __future__ import annotations

from typing import List, Literal, Optional, Tuple

from pydantic import Field

from usami.events import Trace
from usami.protocol import Operation, RequestContext, make_request
from usami.world.engine import WorldEngine
from usami.world.models import Model, WorldState, digest
from usami.world.rules import ReferenceRules, check_invariants

SCENARIO_VERSION = "saas-scenarios-0.1.0"


class ScenarioStep(Model):
    request: RequestContext
    expected_allowed: bool
    expected_error_code: Optional[str] = None
    expected_state_hash: str


class Scenario(Model):
    version: str = SCENARIO_VERSION
    scenario_id: str
    category: str
    world_id: str
    initial_state_hash: str
    rules_hash: str
    operation_budget: int = Field(default=20, ge=1)
    steps: List[ScenarioStep]


def derive_scenario(world: WorldState, label: str, operations: List[Tuple[str, Operation]],
                    category: str, authorized: bool = True, operation_budget: int = 20) -> Scenario:
    if check_invariants(world):
        raise ValueError("scenario generation needs a legal world")
    if len(operations) > operation_budget:
        raise ValueError("scenario exceeds operation budget")
    state, steps = world.clone(), []
    for n, (actor, operation) in enumerate(operations):
        request = make_request(f"{label}_request_{n:03d}", actor, operation, authorized)
        decision = ReferenceRules().decide(state, actor, operation, f"event_{2 * n + 2:06d}")
        if decision.error_code == "UNSUPPORTED_RULES_VERSION":
            raise ValueError("cannot label scenarios under unsupported rules")
        allowed = request.authorizes(actor, operation) and decision.allowed
        if allowed:
            state = decision.expected_state
        steps.append(ScenarioStep(request=request, expected_allowed=allowed,
                                  expected_error_code=None if allowed else
                                  decision.error_code if authorized else "UNAUTHORIZED_OPERATION",
                                  expected_state_hash=state.state_hash))
    scenario_id = "scenario_" + digest({"world": world.state_hash, "label": label, "steps": steps_as_json(steps)})[:20]
    return Scenario(scenario_id=scenario_id, category=category, world_id=world.world_id,
                    initial_state_hash=world.state_hash, rules_hash=digest(world.rules),
                    operation_budget=operation_budget, steps=steps)


def steps_as_json(steps: List[ScenarioStep]):
    return [s.model_dump(mode="json") for s in steps]


def generate_scenarios(world: WorldState) -> List[Scenario]:
    scenarios = []

    def op(name: str, org: str, resource: str, **params) -> Operation:
        return Operation(name=name, org_id=org, resource_id=resource, params=params)

    def one(label, actor, operation, category="single", authorized=True):
        scenarios.append(derive_scenario(world, label, [(actor, operation)], category, authorized))

    plans = sorted(world.rules.plans.values(), key=lambda p: (p.max_seats, p.id))
    for org in world.organizations.values():
        sub, owner = world.subscriptions[org.subscription_id], org.owner_id
        active = [m for m in world.memberships.values() if m.org_id == org.id and m.status == "active"]
        prefix = org.id
        one(prefix + "_query", owner, op("get_subscription", org.id, sub.id), "query")
        one(prefix + "_cancel", owner, op("cancel_renewal", org.id, sub.id))
        one(prefix + "_seats_min", owner, op("change_seats", org.id, sub.id, seats=len(active)))
        one(prefix + "_seats_over", owner, op("change_seats", org.id, sub.id,
                                            seats=world.rules.plans[sub.plan_id].max_seats + 1), "constraint")
        one(prefix + "_seats_under", owner, op("change_seats", org.id, sub.id, seats=len(active) - 1), "constraint")
        one(prefix + "_plan", owner, op("change_plan", org.id, sub.id, plan_id=plans[0].id), "constraint")
        one(prefix + "_no_authorization", owner, op("cancel_renewal", org.id, sub.id), "authorization", False)
        for member in active:
            if member.role == "member":
                one(prefix + "_member_denied_" + member.id, member.user_id,
                    op("cancel_renewal", org.id, sub.id), "permission")
                one(prefix + "_member_query_" + member.id, member.user_id,
                    op("get_subscription", org.id, sub.id), "query")
        for inv in world.invoices.values():
            if inv.org_id != org.id:
                continue
            refund = op("refund_invoice", org.id, inv.id, amount_cents=inv.refundable_amount_cents,
                        payment_method=inv.payment_method or "card_visa")
            one(inv.id + "_refund", owner, refund, "refund")
            if inv.status == "paid":
                one(inv.id + "_wrong_method", owner,
                    op("refund_invoice", org.id, inv.id, amount_cents=inv.refundable_amount_cents,
                       payment_method="different_channel"), "refund_constraint")
                one(inv.id + "_wrong_amount", owner,
                    op("refund_invoice", org.id, inv.id, amount_cents=inv.refundable_amount_cents - 1,
                       payment_method=inv.payment_method), "refund_constraint")
                scenarios.append(derive_scenario(world, inv.id + "_repeat", [(owner, refund), (owner, refund)], "multi_step_refund"))
        for target in active:
            one(prefix + "_role_" + target.id, owner,
                op("change_member_role", org.id, target.id, role="billing_admin"), "permission")
            if target.role != "owner" and sub.status == "active":
                recent = next((i for i in world.invoices.values() if i.org_id == org.id and
                               ReferenceRules().decide(world, target.user_id if target.role == "billing_admin" else owner,
                                   op("refund_invoice", org.id, i.id, amount_cents=i.refundable_amount_cents,
                                      payment_method=i.payment_method or "card_visa")).allowed), None)
                sequence = [(owner, op("change_member_role", org.id, target.id, role="member")),
                            (target.user_id, op("cancel_renewal", org.id, sub.id)),
                            (owner, op("change_member_role", org.id, target.id, role="billing_admin")),
                            (target.user_id, op("cancel_renewal", org.id, sub.id))]
                if recent is not None:
                    sequence.append((target.user_id, op("refund_invoice", org.id, recent.id,
                                         amount_cents=recent.refundable_amount_cents, payment_method=recent.payment_method)))
                scenarios.append(derive_scenario(world, prefix + "_role_then_billing_" + target.id, sequence, "multi_entity"))
        if sub.status == "active" and len(active) <= plans[0].max_seats < plans[-1].max_seats:
            sequence = [(owner, op("change_plan", org.id, sub.id, plan_id=plans[-1].id)),
                        (owner, op("change_seats", org.id, sub.id, seats=plans[0].max_seats + 1)),
                        (owner, op("change_plan", org.id, sub.id, plan_id=plans[0].id)),
                        (owner, op("change_seats", org.id, sub.id, seats=len(active))),
                        (owner, op("change_plan", org.id, sub.id, plan_id=plans[0].id))]
            scenarios.append(derive_scenario(world, prefix + "_downgrade_dependencies", sequence, "multi_step_plan"))
    orgs = list(world.organizations.values())
    for a, b in zip(orgs, orgs[1:] + orgs[:1]):
        one(a.id + "_foreign_resource_" + b.id, a.owner_id,
            op("cancel_renewal", a.id, b.subscription_id), "cross_organization")
        outsiders = [u for u in world.users if not any(m.org_id == b.id and m.user_id == u and m.status == "active"
                                                       for m in world.memberships.values())]
        if outsiders:
            one(a.id + "_outsider_" + b.id, outsiders[0],
                op("get_subscription", b.id, b.subscription_id), "cross_organization")
    return scenarios


def validate_scenario(world: WorldState, scenario: Scenario) -> None:
    if (scenario.version != SCENARIO_VERSION or scenario.world_id != world.world_id
            or scenario.initial_state_hash != world.state_hash or scenario.rules_hash != digest(world.rules)):
        raise ValueError("scenario world, version or rules mismatch")
    if not scenario.steps or len(scenario.steps) > scenario.operation_budget:
        raise ValueError("empty or over-budget scenario")
    state = world.clone()
    request_ids = set()
    for n, step in enumerate(scenario.steps):
        req = step.request
        if req.request_id in request_ids:
            raise ValueError("duplicate scenario request ID")
        request_ids.add(req.request_id)
        decision = ReferenceRules().decide(state, req.requester_id, req.operation, f"event_{2 * n + 2:06d}")
        authorized = req.authorizes(req.requester_id, req.operation)
        allowed = decision.allowed and authorized
        error = None if allowed else decision.error_code if authorized else "UNAUTHORIZED_OPERATION"
        if allowed:
            state = decision.expected_state
        if (allowed != step.expected_allowed or error != step.expected_error_code
                or state.state_hash != step.expected_state_hash):
            raise ValueError("scenario expected label/state was not derived from reference rules")


def simulate_scenario(world: WorldState, scenario: Scenario,
                      mode: Literal["reference", "sandbox", "refuse"] = "reference") -> Trace:
    validate_scenario(world, scenario)
    engine = WorldEngine(world, scenario.operation_budget)
    for step in scenario.steps:
        req = step.request
        engine.register_request(req)
        engine.execute(req.requester_id, req.operation, req.request_id, mode)
    return engine.finish()
