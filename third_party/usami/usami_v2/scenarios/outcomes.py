"""Derive candidate outcomes by executing the native bounded Reference search.

These results are labels, not certificates. Callers must independently certify
the resulting task before admitting it into a training or evaluation set.
"""
from datetime import timedelta

from usami_harness.models import digest
from usami_v2.commerce import adapter
from .refusal_certification import _operations
from .task_certification import CustomerScenario, _available_goal_units, _customer_delegation, certify_task
from .tasks import (CustomerRequestBundle, OrderCancellationGoal, RefusalGoal, RemedyGoal,
                    TaskSpecV3, delegated_line_ids, render_customer_request, request_intent_hash)


SUPPORTED = frozenset(('RETURN_WINDOW_EXPIRED', 'ORDER_NOT_CANCELLABLE', 'APPROVAL_REQUIRED',
                       'INSUFFICIENT_STOCK', 'ORDER_NOT_DELIVERED', 'CUSTOMER_DECLINED',
                       'RESHIP_NOT_ELIGIBLE', 'PURCHASED_QUANTITY_UNAVAILABLE'))


class OutcomeUnsupported(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _temporary_task(world, customer_id, goal, *, refusal=False):
    remedies = (goal,) if isinstance(goal, RemedyGoal) else ()
    cancellations = (goal,) if isinstance(goal, OrderCancellationGoal) else ()
    refusals = (goal,) if refusal else ()
    text = render_customer_request(world, customer_id, remedies, (), cancellations, refusals)
    bundle = CustomerRequestBundle(customer_id=customer_id, goal_ids=(goal.goal_id,),
        intent_hash=request_intent_hash(customer_id, remedies, (), cancellations, refusals),
        request_hash=digest(text))
    return TaskSpecV3(task_id='outcome-probe', world_hash=world.state_hash, customer_id=customer_id,
        customer_request=text, remedy_goals=remedies, cancellation_goals=cancellations,
        refusal_goals=refusals, partition='train', source_distribution_hash=digest('outcome-probe'),
        request_bundle=bundle)


def ordered_contexts(world, customer_id, requests, *, return_receipt_event_configured=False):
    """Recompute each ordered request's input state using certified witnesses."""
    from usami_v2.commerce.models import World
    current, contexts = world.clone(), []
    for request in requests:
        goal = derive_outcomes(current, customer_id, (request,),
            return_receipt_event_configured=return_receipt_event_configured)[0]
        contexts.append((goal, current))
        if isinstance(goal, RefusalGoal):
            continue
        task = _temporary_task(current, customer_id, goal)
        scenario = CustomerScenario(customer_id=customer_id, delegated_line_ids=delegated_line_ids(task),
            accepted_remedies=(request.action,) if request.action != 'cancel' else (),
            maximum_extra_charge_cents=(request.maximum_extra_charge_cents
                if request.maximum_extra_charge_cents is not None else 10**18),
            latest_arrival=goal.latest_arrival if isinstance(goal, RemedyGoal) else None,
            return_received_after_days=0 if return_receipt_event_configured else None)
        proof = certify_task(current, task, scenario)
        witness = next((row for row in proof.attempts if row.status == 'WITNESS'), None)
        if witness is None:
            raise OutcomeUnsupported('GOALS_CONFLICT')
        operations = [step for step in witness.steps if step.kind == 'operation']
        if not operations:
            raise OutcomeUnsupported('GOALS_CONFLICT')
        current = World.model_validate(operations[-1].evidence['trace']['final_state'])
        current.delegations.clear()
        current.consents.clear()
    return tuple(contexts)


def derive_outcomes(world, customer_id, requests, *, return_receipt_event_configured=False, sequential=False):
    """Return native goals in request order, or an explicit unsupported reason.

    ``requests`` are CustomerRequestIntent objects whose line IDs already refer
    to this frozen world. No decision is made from a Designer-provided label.
    """
    if sequential:
        return tuple(goal for goal, _ in ordered_contexts(world, customer_id, requests,
            return_receipt_event_configured=return_receipt_event_configured))
    outcomes = []
    for request in requests:
        request = request.checked()
        line = world.lines.get(request.line_id)
        if line is None or world.orders[line.order_id].customer_id != customer_id:
            raise ValueError('request line does not belong to customer')
        order = world.orders[line.order_id]
        line_ids = (tuple(sorted(row.id for row in world.lines.values() if row.order_id == order.id))
                    if request.action == 'cancel' else (line.id,))
        if request.action == 'cancel':
            if request.quantity != line.quantity:
                raise OutcomeUnsupported('UNSUPPORTED_OUTCOME:PARTIAL_ORDER_CANCELLATION')
            goal = OrderCancellationGoal(goal_id=request.goal_id, order_id=order.id, line_ids=line_ids)
        else:
            goal = RemedyGoal(goal_id=request.goal_id, line_id=line.id, quantity=request.quantity,
                allowed_remedies=(request.action,), required_attributes=request.required_attributes,
                maximum_extra_charge_cents=request.maximum_extra_charge_cents,
                latest_arrival=(world.clock + timedelta(days=request.latest_arrival_within_days)
                                if request.latest_arrival_within_days is not None else None))
        task = _temporary_task(world, customer_id, goal)
        scenario = CustomerScenario(customer_id=customer_id, delegated_line_ids=delegated_line_ids(task),
            accepted_remedies=(request.action,) if request.action != 'cancel' else (),
            maximum_extra_charge_cents=(request.maximum_extra_charge_cents
                if request.maximum_extra_charge_cents is not None else 10**18),
            latest_arrival=goal.latest_arrival if isinstance(goal, RemedyGoal) else None,
            return_received_after_days=0 if return_receipt_event_configured else None)
        proof = None
        if request.action == 'cancel' or request.quantity <= len(_available_goal_units(world, goal)):
            proof = certify_task(world, task, scenario)
            if proof.status == 'CERTIFIED' and proof.business_reachability_certified:
                outcomes.append(goal)
                continue
        refusal = RefusalGoal(goal_id=request.goal_id, line_ids=line_ids,
            reason_code='CUSTOMER_DECLINED', requested_action=request.action,
            maximum_extra_charge_cents=request.maximum_extra_charge_cents)
        refusal_task = _temporary_task(world, customer_id, refusal, refusal=True)
        state, _ = _customer_delegation(world, refusal_task, scenario)
        operations = _operations(state, refusal, quantity=request.quantity,
                                 required_attributes=request.required_attributes)
        decisions = tuple(adapter.decide(state, customer_id, operation) for operation in operations)
        codes = {row.error_code for row in decisions if not row.allowed}
        allowed = any(row.allowed for row in decisions)
        if allowed and proof is not None and proof.search_complete:
            reasons = {attempt.reason for attempt in proof.attempts}
            decline_reasons = {'customer_extra_charge_limit', 'customer_delivery_deadline'}
            if (not proof.blockers and reasons.intersection(decline_reasons)
                    and reasons <= decline_reasons | codes):
                outcomes.append(refusal)
                continue
        if decisions and not allowed and len(codes) == 1:
            code = next(iter(codes))
            if code in SUPPORTED:
                outcomes.append(RefusalGoal(goal_id=request.goal_id, line_ids=line_ids,
                    reason_code=code, requested_action=request.action,
                    maximum_extra_charge_cents=request.maximum_extra_charge_cents))
                continue
            raise OutcomeUnsupported('UNSUPPORTED_OUTCOME:' + str(code))
        reason = ('EXCHANGE_TARGET_NOT_FOUND' if not decisions and request.action == 'exchange'
                  else 'NO_CANDIDATES' if not decisions
                  else 'MIXED_OUTCOMES' if allowed or len(codes) > 1 else 'UNKNOWN')
        raise OutcomeUnsupported('UNSUPPORTED_OUTCOME:' + reason)
    return tuple(outcomes)
