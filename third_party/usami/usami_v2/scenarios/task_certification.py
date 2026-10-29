"""Bounded Reference search for frozen commerce tasks.

The private CustomerScenario is a deterministic certification fixture, not a
natural-language Customer runtime. Every Service operation still goes through
the existing Engine, Reference, Oracle and Replay. Customer authority events
are separately content-bound and replayed by this module.
"""
from datetime import datetime, timedelta
from itertools import product
from typing import Dict, Literal, Optional, Tuple

from pydantic import Field, model_validator

from usami_harness.engine import Engine, replay_trace
from usami_harness.models import Request, digest
from usami_harness.oracle import evaluate
from usami_v2.contracts import Sealed, aware
from usami_v2.commerce import adapter
from usami_v2.commerce.external import apply_event, reference_event
from usami_v2.commerce.models import (
    Consent, Delegation, ExternalEvent, Operation, PREPARATIONS, World, unit_credit,
)
from .tasks import RemedyGoal, TaskSpec, check_task_scope, evaluate_state_goals


class ReviewPlan(Sealed):
    line_id: str
    approved: bool
    result_after_days: int = Field(default=1, ge=0, le=365, strict=True)


class CustomerScenario(Sealed):
    """Private, fixed Customer behavior used only by the proof search."""

    protocol: Literal["commerce-customer-scenario-v1"] = "commerce-customer-scenario-v1"
    customer_id: str
    delegated_line_ids: Tuple[str, ...] = ()
    allowed_preparations: Tuple[str, ...] = tuple(sorted(PREPARATIONS - {"cancel_pending_commitment"}))
    accepted_remedies: Tuple[Literal["refund", "exchange", "reship"], ...] = (
        "refund", "exchange", "reship")
    maximum_extra_charge_cents: int = Field(default=100000000, ge=0, strict=True)
    latest_arrival: Optional[datetime] = None
    return_pickup_slot: Optional[str] = None
    return_received_after_days: Optional[int] = Field(default=None, ge=0, le=365, strict=True)
    second_refund_consent: bool = True
    reviews: Tuple[ReviewPlan, ...] = ()

    @model_validator(mode="after")
    def unique_policy(self):
        if len(set(self.delegated_line_ids)) != len(self.delegated_line_ids):
            raise ValueError("CustomerScenario repeats delegated lines")
        if len(set(self.allowed_preparations)) != len(self.allowed_preparations):
            raise ValueError("CustomerScenario repeats preparations")
        if set(self.allowed_preparations) - PREPARATIONS:
            raise ValueError("CustomerScenario contains an unregistered preparation")
        if len(set(self.accepted_remedies)) != len(self.accepted_remedies):
            raise ValueError("CustomerScenario repeats remedies")
        if len({row.line_id for row in self.reviews}) != len(self.reviews):
            raise ValueError("CustomerScenario has multiple review outcomes for one line")
        if self.latest_arrival is not None:
            aware(self.latest_arrival)
        return self


class CustomerTaskScope(Sealed):
    """Public aliases a Task Designer may select for one frozen Customer."""

    customer_alias: str
    order_aliases: Tuple[str, ...] = ()
    line_aliases: Tuple[str, ...] = ()


class TaskDesignCapabilities(Sealed):
    """Versioned declaration of what this run's certifier can actually prove.

    This is supplied to both the Task Designer and compiler. It describes the
    native proof boundary; it never contains a Reference witness or answer key.
    """

    protocol: Literal["commerce-task-design-capabilities-v1"] = "commerce-task-design-capabilities-v1"
    certifier_protocol: Literal["commerce-task-certification-v1"] = "commerce-task-certification-v1"
    scenario_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    world_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    native_reference_certifies: Tuple[Literal["settled_remedy_goals"], ...] = ("settled_remedy_goals",)
    supported_remedies: Tuple[Literal["refund", "exchange", "reship"], ...] = (
        "refund", "exchange", "reship")
    supported_target_stages: Tuple[Literal["settled", "delivered"], ...] = ("settled",)
    interaction_certifier_available: bool = False
    information_goals_certifiable: bool = False
    delivery_confirmation_available: bool = False
    return_receipt_event_configured: bool = False
    policy_review_event_configured: bool = False
    customer_scopes: Tuple[CustomerTaskScope, ...] = ()
    grounding_protocol: Literal[
        "public-intent-grounding-v1", "structured-intent-grounding-v2"
    ] = "structured-intent-grounding-v2"

    @model_validator(mode="after")
    def capability_consistency(self):
        if self.delivery_confirmation_available != ("delivered" in self.supported_target_stages):
            raise ValueError("delivery capability and supported target stages disagree")
        if self.information_goals_certifiable and not self.interaction_certifier_available:
            raise ValueError("information certification requires an interaction certifier")
        if self.delivery_confirmation_available and not self.interaction_certifier_available:
            raise ValueError("delivery confirmation requires an interaction certifier")
        if len({row.customer_alias for row in self.customer_scopes}) != len(self.customer_scopes):
            raise ValueError("Task capability manifest repeats a Customer alias")
        for row in self.customer_scopes:
            if len(set(row.order_aliases)) != len(row.order_aliases):
                raise ValueError("Task capability manifest repeats an order alias")
            if len(set(row.line_aliases)) != len(row.line_aliases):
                raise ValueError("Task capability manifest repeats a line alias")
        return self


class ProofBudget(Sealed):
    max_plan_candidates: int = Field(default=256, ge=1, le=10000, strict=True)
    max_operation_steps: int = Field(default=64, ge=1, le=2000, strict=True)
    max_witnesses: int = Field(default=2, ge=1, le=32, strict=True)


class ProofStep(Sealed):
    kind: Literal["customer_delegation", "customer_consent", "external_event", "operation"]
    before_state_hash: str
    after_state_hash: str
    evidence: Dict[str, object] = Field(default_factory=dict)


class PlanCandidate(Sealed):
    goal_id: str
    remedy: Literal["refund", "exchange", "reship"]
    target_sku_id: Optional[str] = None
    warehouse_id: Optional[str] = None


class ProofAttempt(Sealed):
    candidate_hash: str
    candidates: Tuple[PlanCandidate, ...]
    status: Literal["WITNESS", "REJECTED"]
    reason: str
    steps: Tuple[ProofStep, ...]
    final_state_hash: Optional[str] = None
    goal_results: Tuple[Tuple[str, str], ...] = ()


class TaskCertification(Sealed):
    protocol: Literal["commerce-task-certification-v1"] = "commerce-task-certification-v1"
    status: Literal["CERTIFIED", "UNSAT", "UNKNOWN"]
    world_hash: str
    task_hash: str
    customer_scenario_hash: str
    proof_budget: ProofBudget
    budget_hash: str
    candidates_examined: int = Field(ge=0, strict=True)
    search_complete: bool
    blockers: Tuple[str, ...] = ()
    attempts: Tuple[ProofAttempt, ...] = ()
    witness_hashes: Tuple[str, ...] = ()
    minimum_witnesses: int = Field(ge=1, strict=True)
    minimum_witnesses_met: bool
    business_reachability_certified: bool
    full_task_reachability_certified: bool
    interaction_certified: bool = False

    @model_validator(mode="after")
    def budget_binding(self):
        if self.proof_budget.record_hash != self.budget_hash:
            raise ValueError("task certificate budget binding differs")
        return self


def _bind_scenario(world: World, task: TaskSpec, scenario: CustomerScenario):
    task = check_task_scope(world, task)
    scenario = scenario.checked()
    if scenario.customer_id != task.customer_id:
        raise ValueError("CustomerScenario identity differs from task")
    goal_lines = {row.line_id for row in task.remedy_goals}
    if not goal_lines.issubset(set(scenario.delegated_line_ids)):
        raise ValueError("CustomerScenario does not delegate every requested remedy line")
    if any(row.line_id not in goal_lines for row in scenario.reviews):
        raise ValueError("review event plan targets a line outside the task")
    if any(getattr(world, name) for name in ("cases", "quotes", "delegations", "consents", "returns",
                                               "ledger", "settlements", "fulfillments", "external_events",
                                               "applied_events")):
        raise ValueError("task proof requires a frozen world without prior transactions or authority")
    return task, scenario


def _base_candidate_options(world: World, goal: RemedyGoal, scenario: CustomerScenario, blockers):
    line = world.lines[goal.line_id]
    original = world.skus[line.sku_id]
    indices = tuple(range(goal.quantity))
    credit = unit_credit(line, indices)
    evidence = [row for row in world.evidence.values()
                if row.line_id == line.id and row.customer_id == scenario.customer_id and row.valid]
    if not evidence:
        return ()
    options = []
    for remedy in goal.allowed_remedies:
        if remedy not in scenario.accepted_remedies:
            continue
        if remedy == "refund":
            if goal.required_attributes or goal.latest_arrival is not None:
                continue
            if original.return_required and (not scenario.return_pickup_slot
                                             or scenario.return_received_after_days is None):
                blockers.append("return_receipt_event_not_registered_in_frozen_customer_scenario")
                continue
            review_plan = next((row for row in scenario.reviews if row.line_id == line.id), None)
            if (credit > world.merchants[world.orders[line.order_id].merchant_id].refund_approval_limit_cents
                    and review_plan is None):
                blockers.append("policy_review_event_not_registered_in_frozen_customer_scenario")
                continue
            options.append(PlanCandidate(goal_id=goal.goal_id, remedy=remedy))
            continue
        targets = sorted(world.skus.values(), key=lambda row: row.id)
        warehouses = sorted(world.warehouses.values(), key=lambda row: row.id)
        for target in targets:
            if target.merchant_id != original.merchant_id or target.product_family != original.product_family:
                continue
            if remedy == "reship" and target.id != original.id:
                continue
            if any(target.attributes.get(row.name) != row.value for row in goal.required_attributes):
                continue
            for warehouse in warehouses:
                inv = next((row for row in world.inventory.values()
                            if row.sku_id == target.id and row.warehouse_id == warehouse.id), None)
                route = next((row for row in world.routes.values()
                              if row.warehouse_id == warehouse.id), None)
                if not inv or not route or inv.available < goal.quantity:
                    continue
                if original.return_required and (not scenario.return_pickup_slot
                                                 or scenario.return_received_after_days is None):
                    blockers.append("return_receipt_event_not_registered_in_frozen_customer_scenario")
                    continue
                delta = target.price_cents * goal.quantity + route.fee_cents - credit
                extra = max(delta, 0) if remedy == "exchange" else 0
                arrival = world.clock + timedelta(days=route.days)
                if extra > scenario.maximum_extra_charge_cents:
                    continue
                if goal.maximum_extra_charge_cents is not None and extra > goal.maximum_extra_charge_cents:
                    continue
                if goal.latest_arrival is not None and arrival > goal.latest_arrival:
                    continue
                if scenario.latest_arrival is not None and arrival > scenario.latest_arrival:
                    continue
                if goal.target_stage == "delivered":
                    blockers.append("delivery_confirmation_event_not_registered_in_commerce_v1")
                    continue
                options.append(PlanCandidate(goal_id=goal.goal_id, remedy=remedy,
                                             target_sku_id=target.id, warehouse_id=warehouse.id))
    return tuple(options)


def diagnose_candidate_alternatives(world: World, task: TaskSpec, scenario: CustomerScenario):
    """Count rule-level outcomes for plausible alternatives without changing proof search.

    This is a researcher-only diagnostic. It does not create candidates for the
    certifier, claim an execution witness, or affect certification status.
    Counts are non-exclusive: one alternative can violate more than one stated
    constraint, which is useful when explaining why it is unsuitable.
    """
    task = check_task_scope(world, task)
    scenario = scenario.checked()
    if scenario.customer_id != task.customer_id:
        raise ValueError("diagnostic Customer identity differs from task")
    results = []
    for goal in task.remedy_goals:
        line = world.lines[goal.line_id]
        order = world.orders[line.order_id]
        original = world.skus[line.sku_id]
        evidence = any(row.line_id == line.id and row.customer_id == scenario.customer_id and row.valid
                       for row in world.evidence.values())
        counts = {}
        examined = feasible = 0

        def excluded(*labels):
            for label in set(labels):
                counts[label] = counts.get(label, 0) + 1

        if not evidence:
            excluded("no_valid_evidence")
            results.append({"goal_id": goal.goal_id, "alternatives_examined": 0,
                            "feasible_alternatives": 0,
                            "excluded_condition_counts": tuple(sorted(counts.items()))})
            continue

        for remedy in goal.allowed_remedies:
            if remedy not in scenario.accepted_remedies:
                excluded("customer_policy_mismatch")
                continue
            if remedy == "refund":
                examined += 1
                failures = []
                if goal.required_attributes or goal.latest_arrival is not None:
                    failures.append("refund_cannot_satisfy_product_or_deadline_constraint")
                if original.return_required and (not scenario.return_pickup_slot
                                                 or scenario.return_received_after_days is None):
                    failures.append("return_receipt_unconfigured")
                credit = unit_credit(line, tuple(range(goal.quantity)))
                limit = world.merchants[order.merchant_id].refund_approval_limit_cents
                review = next((row for row in scenario.reviews if row.line_id == line.id), None)
                if credit > limit and review is None:
                    failures.append("policy_review_unconfigured")
                elif credit > limit and review is not None and not review.approved:
                    failures.append("policy_review_denied")
                if failures:
                    excluded(*failures)
                else:
                    feasible += 1
                continue

            for target in sorted(world.skus.values(), key=lambda row: row.id):
                for warehouse in sorted(world.warehouses.values(), key=lambda row: row.id):
                    examined += 1
                    failures = []
                    if target.merchant_id != original.merchant_id or target.product_family != original.product_family:
                        failures.append("incompatible_replacement_family")
                    if remedy == "reship" and target.id != original.id:
                        failures.append("reship_requires_original_sku")
                    if any(target.attributes.get(row.name) != row.value for row in goal.required_attributes):
                        failures.append("specification_mismatch")
                    inventory = next((row for row in world.inventory.values()
                                      if row.sku_id == target.id and row.warehouse_id == warehouse.id), None)
                    route = next((row for row in world.routes.values()
                                  if row.warehouse_id == warehouse.id), None)
                    if inventory is None or route is None:
                        failures.append("route_or_inventory_missing")
                    elif inventory.available < goal.quantity:
                        failures.append("inventory_insufficient")
                    if original.return_required and (not scenario.return_pickup_slot
                                                     or scenario.return_received_after_days is None):
                        failures.append("return_receipt_unconfigured")
                    if inventory is not None and route is not None:
                        credit = unit_credit(line, tuple(range(goal.quantity)))
                        delta = target.price_cents * goal.quantity + route.fee_cents - credit
                        extra = max(delta, 0) if remedy == "exchange" else 0
                        if (extra > scenario.maximum_extra_charge_cents
                                or (goal.maximum_extra_charge_cents is not None
                                    and extra > goal.maximum_extra_charge_cents)):
                            failures.append("budget_exceeded")
                        arrival = world.clock + timedelta(days=route.days)
                        if ((goal.latest_arrival is not None and arrival > goal.latest_arrival)
                                or (scenario.latest_arrival is not None and arrival > scenario.latest_arrival)):
                            failures.append("deadline_missed")
                    if failures:
                        excluded(*failures)
                    else:
                        feasible += 1
        results.append({"goal_id": goal.goal_id, "alternatives_examined": examined,
                        "feasible_alternatives": feasible,
                        "excluded_condition_counts": tuple(sorted(counts.items()))})
    return tuple(results)


def _step(kind, before, after, **evidence):
    return ProofStep(kind=kind, before_state_hash=before.state_hash,
                     after_state_hash=after.state_hash, evidence=evidence)


def _customer_delegation(world, task, scenario):
    if not task.remedy_goals:
        return world, None
    before = world
    new = world.clone()
    delegation_id = "delegation_" + digest([task.record_hash, scenario.record_hash])[:20]
    source_hash = digest({"customer_request": task.customer_request, "scenario": scenario.record_hash,
                         "delegated_line_ids": scenario.delegated_line_ids,
                         "allowed_preparations": scenario.allowed_preparations})
    new.delegations[delegation_id] = Delegation(
        id=delegation_id, customer_id=scenario.customer_id,
        line_ids=scenario.delegated_line_ids,
        allowed_preparations=scenario.allowed_preparations,
        expires_at=world.clock + timedelta(days=365), source_customer_event_hash=source_hash)
    if adapter.invariants(new):
        raise ValueError("derived Customer delegation violates native invariants")
    step = _step("customer_delegation", before, new, delegation=new.delegations[delegation_id].model_dump(mode="json"),
                 source_hash=source_hash, task_request_hash=digest(task.customer_request))
    return new, step


def _execute(world, actor, name, resource, params, attempt_id, step_no):
    operation = Operation(name=name, scope_id=next(iter(world.merchants)), resource_id=resource, params=params)
    request = Request(id="proof-{}-{:03d}".format(attempt_id[:12], step_no), identity=actor,
                      operation=operation.model_dump(mode="json"), granted=True)
    engine = Engine(adapter, world, operation_budget=1)
    engine.register(request)
    event = engine.execute(actor, operation, request.id, mode="reference")
    trace = engine.finish()
    oracle = evaluate(adapter, trace, {request.id: request})
    replayed = replay_trace(adapter, trace, {request.id: request})
    result = event.payload["result"]
    after = engine.state
    step = _step("operation", world, after, trace=trace.model_dump(mode="json"),
                 trace_hash=trace.trace_hash, oracle=oracle, request=request.model_dump(mode="json"),
                 result=result, replay_state_hash=replayed.state.state_hash)
    if not oracle.get("trace_valid") or oracle.get("status") != "pass" or not result.get("success"):
        return after, step, result.get("error_code") or "reference_or_oracle_rejected_operation"
    return after, step, None


def _customer_consent(world, quote_id_value, operation, task, scenario):
    quote = world.quotes[quote_id_value]
    if quote.remedy not in scenario.accepted_remedies:
        return world, None, "customer_does_not_accept_remedy"
    if quote.extra_charge_cents > scenario.maximum_extra_charge_cents:
        return world, None, "customer_extra_charge_limit"
    if scenario.latest_arrival is not None and quote.promised_arrival is not None \
            and quote.promised_arrival > scenario.latest_arrival:
        return world, None, "customer_delivery_deadline"
    if operation == "release_accepted_refund" and not scenario.second_refund_consent:
        return world, None, "customer_does_not_consent_to_refund_release"
    before, new = world, world.clone()
    key = "consent_" + digest([task.record_hash, scenario.record_hash, quote.id, operation])[:20]
    presented = digest(["customer_scenario_offer_presented", scenario.record_hash, quote.id, quote.terms_hash])
    response = digest(["customer_scenario_explicit_consent", scenario.record_hash, quote.id,
                       quote.terms_hash, operation])
    new.consents[key] = Consent(
        id=key, customer_id=scenario.customer_id, quote_id=quote.id, terms_hash=quote.terms_hash,
        operation=operation, presented_offer_event_hash=presented,
        source_customer_event_hash=response, expires_at=new.clock + timedelta(days=365))
    if adapter.invariants(new):
        raise ValueError("derived Customer consent violates native invariants")
    return new, _step("customer_consent", before, new,
                      consent=new.consents[key].model_dump(mode="json"), quote_terms_hash=quote.terms_hash), None


def _apply_event(world, event):
    before, registered = world, world.clone()
    registered.external_events[event.id] = event
    if adapter.invariants(registered):
        raise ValueError("frozen event registration violates native invariants")
    register_step = _step("external_event", before, registered,
                          action="register", event=event.model_dump(mode="json"))
    expected = reference_event(registered, event.id)
    actual = apply_event(registered, event.id)
    if expected != actual or adapter.invariants(actual):
        raise ValueError("actual and independent external event projections differ")
    apply_step = _step("external_event", registered, actual,
                       action="apply", event_id=event.id,
                       reference_state_hash=expected.state_hash, actual_state_hash=actual.state_hash)
    return actual, (register_step, apply_step)


def _run_candidate(world, task, scenario, candidates, attempt_id, max_steps):
    state = world.clone()
    steps = []
    if not task.remedy_goals:
        return ProofAttempt(candidate_hash=digest([row.model_dump(mode="json") for row in candidates]),
                            candidates=candidates, status="REJECTED",
                            reason="no_business_state_goal_for_reference_search", steps=())
    state, delegation_step = _customer_delegation(state, task, scenario)
    if delegation_step:
        steps.append(delegation_step)
    goals = {row.goal_id: row for row in task.remedy_goals}
    attempts_step_no = 0

    def act(name, resource, params):
        nonlocal state, attempts_step_no
        if attempts_step_no >= max_steps:
            return "proof_operation_budget_exhausted"
        attempts_step_no += 1
        state, step, error = _execute(state, scenario.customer_id, name, resource, params,
                                      attempt_id, attempts_step_no)
        steps.append(step)
        return error

    for candidate in candidates:
        goal = goals[candidate.goal_id]
        line = state.lines[goal.line_id]
        evidence = next((row for row in sorted(state.evidence.values(), key=lambda row: row.id)
                         if row.line_id == line.id and row.customer_id == scenario.customer_id and row.valid), None)
        if evidence is None:
            reason = "no_valid_customer_evidence"
            break
        error = act("open_remedy_case", line.id, {"quantity": goal.quantity, "issue": evidence.issue})
        if error:
            reason = error
            break
        opened = steps[-1].evidence["result"]["data"]
        case_id = opened["case_id"]
        error = act("verify_submitted_evidence", case_id, {"evidence_id": evidence.id})
        if error:
            reason = error
            break
        original = state.skus[line.sku_id]
        credit = unit_credit(line, tuple(range(goal.quantity)))
        review_plan = next((row for row in scenario.reviews if row.line_id == line.id), None)
        if candidate.remedy == "refund" and credit > state.merchants[state.orders[line.order_id].merchant_id].refund_approval_limit_cents:
            if review_plan is None:
                reason = "refund_requires_unregistered_policy_review_event"
                break
            error = act("request_policy_review", case_id, {})
            if error:
                reason = error
                break
            event_id = "review_" + digest([scenario.record_hash, case_id])[:20]
            event = ExternalEvent(id=event_id, kind="review_result", target_id=case_id,
                                  at=state.clock + timedelta(days=review_plan.result_after_days),
                                  approved=review_plan.approved)
            state, event_steps = _apply_event(state, event)
            steps.extend(event_steps)
            if not review_plan.approved:
                reason = "policy_review_denied_in_frozen_customer_scenario"
                break
        pickup = scenario.return_pickup_slot if original.return_required else None
        params = {"remedy": candidate.remedy,
                  "target_sku_id": candidate.target_sku_id,
                  "warehouse_id": candidate.warehouse_id,
                  "pickup_slot": pickup}
        error = act("prepare_remedy_quote", case_id, params)
        if error:
            reason = error
            break
        prepared = steps[-1].evidence["result"]["data"]["quote"]["id"]
        state, consent_step, error = _customer_consent(state, prepared, "commit_accepted_quote", task, scenario)
        if error:
            reason = error
            break
        steps.append(consent_step)
        error = act("commit_accepted_quote", prepared, {})
        if error:
            reason = error
            break
        quote = state.quotes[prepared]
        if quote.return_required:
            if scenario.return_received_after_days is None:
                reason = "return_receipt_absent_from_frozen_event_plan"
                break
            case = state.cases[quote.case_id]
            error = act("schedule_authorized_return", case.id, {})
            if error:
                reason = error
                break
            event_id = "return_" + digest([scenario.record_hash, case.id])[:20]
            event = ExternalEvent(id=event_id, kind="return_received", target_id=case.id,
                                  at=state.clock + timedelta(days=scenario.return_received_after_days))
            state, event_steps = _apply_event(state, event)
            steps.extend(event_steps)
            if quote.remedy == "refund":
                state, consent_step, error = _customer_consent(
                    state, prepared, "release_accepted_refund", task, scenario)
                if error:
                    reason = error
                    break
                steps.append(consent_step)
                error = act("release_accepted_refund", case.id, {})
            else:
                error = act("ship_accepted_replacement", case.id, {})
            if error:
                reason = error
                break
    else:
        score = evaluate_state_goals(world, state, task)
        result_map = tuple((row.goal_id, row.status) for row in score)
        if all(row.status == "PASS" for row in score):
            return ProofAttempt(candidate_hash=digest([row.model_dump(mode="json") for row in candidates]),
                                candidates=candidates, status="WITNESS",
                                reason="all_registered_state_goals_passed_by_reference_execution",
                                steps=tuple(steps), final_state_hash=state.state_hash, goal_results=result_map)
        reason = "registered_state_goal_not_reached"
    return ProofAttempt(candidate_hash=digest([row.model_dump(mode="json") for row in candidates]),
                        candidates=candidates, status="REJECTED",
                        reason=reason, steps=tuple(steps), final_state_hash=state.state_hash,
                        goal_results=tuple((row.goal_id, row.status)
                                           for row in evaluate_state_goals(world, state, task)))


def certify_task(world: World, task: TaskSpec, customer_scenario: CustomerScenario,
                 budget: Optional[ProofBudget] = None, *, minimum_witnesses: int = 1):
    """Search bounded remedy plans; never infer UNSAT from an incomplete search."""
    task, customer_scenario = _bind_scenario(world, task, customer_scenario)
    budget = (budget or ProofBudget()).checked()
    if type(minimum_witnesses) is not int or minimum_witnesses < 1:
        raise ValueError("minimum_witnesses must be a positive integer")
    blockers = []
    if task.information_goals:
        blockers.append("communication_objectives_require_natural_runtime_p4")
    if not task.remedy_goals:
        return TaskCertification(
            status="UNKNOWN", world_hash=world.state_hash, task_hash=task.record_hash,
            customer_scenario_hash=customer_scenario.record_hash, proof_budget=budget,
            budget_hash=budget.record_hash,
            candidates_examined=0, search_complete=True, blockers=tuple(blockers or ["no_remedy_goals"]),
            minimum_witnesses=minimum_witnesses, minimum_witnesses_met=False,
            business_reachability_certified=False, full_task_reachability_certified=False)
    option_sets = []
    for goal in task.remedy_goals:
        option_sets.append(_base_candidate_options(world, goal, customer_scenario, blockers))
    if any(not rows for rows in option_sets):
        # The enumerator's candidate grammar is complete for the registered no-return,
        # settled/committed task subset. Explicit unsupported features remain UNKNOWN.
        unsupported = bool(blockers)
        return TaskCertification(
            status="UNKNOWN" if unsupported else "UNSAT", world_hash=world.state_hash,
            task_hash=task.record_hash, customer_scenario_hash=customer_scenario.record_hash,
            proof_budget=budget, budget_hash=budget.record_hash,
            candidates_examined=0, search_complete=True,
            blockers=tuple(sorted(set(blockers + ["one_or_more_goals_have_no_supported_candidate"]))),
            minimum_witnesses=minimum_witnesses, minimum_witnesses_met=False,
            business_reachability_certified=False, full_task_reachability_certified=False)
    attempts = []
    examined, exhausted = 0, False
    for combination in product(*option_sets):
        if examined >= budget.max_plan_candidates:
            exhausted = True
            break
        examined += 1
        candidate_id = digest([task.record_hash, [row.model_dump(mode="json") for row in combination]])
        attempt = _run_candidate(world, task, customer_scenario, tuple(combination),
                                 candidate_id, budget.max_operation_steps)
        attempts.append(attempt)
        if attempt.reason == "proof_operation_budget_exhausted":
            blockers.append("proof_operation_budget_exhausted")
        if attempt.status == "WITNESS" and sum(row.status == "WITNESS" for row in attempts) >= budget.max_witnesses:
            # Search is incomplete unless the last candidate was the final Cartesian item.
            exhausted = examined < _product_size(option_sets)
            break
    witnesses = tuple(row for row in attempts if row.status == "WITNESS")
    search_complete = not exhausted
    if witnesses and not blockers:
        status = "CERTIFIED"
    elif witnesses:
        status = "UNKNOWN"
    elif search_complete and not blockers:
        status = "UNSAT"
    else:
        status = "UNKNOWN"
    full = bool(witnesses) and not task.information_goals
    return TaskCertification(
        status=status, world_hash=world.state_hash, task_hash=task.record_hash,
        customer_scenario_hash=customer_scenario.record_hash, proof_budget=budget,
        budget_hash=budget.record_hash,
        candidates_examined=examined, search_complete=search_complete,
        blockers=tuple(sorted(set(blockers))), attempts=tuple(attempts),
        witness_hashes=tuple(row.record_hash for row in witnesses),
        minimum_witnesses=minimum_witnesses,
        minimum_witnesses_met=len(witnesses) >= minimum_witnesses,
        business_reachability_certified=bool(witnesses),
        full_task_reachability_certified=full, interaction_certified=False)


def _product_size(option_sets):
    total = 1
    for rows in option_sets:
        total *= len(rows)
    return total


def verify_task_certificate(world, task, customer_scenario, certification):
    """Recompute the finite search, Reference executions, Oracle checks and trace replays."""
    certification = certification.checked()
    if certification.proof_budget.record_hash != certification.budget_hash:
        raise ValueError("proof budget hash differs")
    rebuilt = certify_task(world, task, customer_scenario, certification.proof_budget,
                           minimum_witnesses=certification.minimum_witnesses)
    if rebuilt != certification:
        raise ValueError("task reachability certificate or private witness differs")
    return {"verified": True, "status": certification.status,
            "business_reachability_certified": certification.business_reachability_certified,
            "full_task_reachability_certified": certification.full_task_reachability_certified,
            "interaction_certified": certification.interaction_certified,
            "witness_count": len(certification.witness_hashes),
            "candidate_count": certification.candidates_examined,
            "certificate_hash": certification.record_hash}
