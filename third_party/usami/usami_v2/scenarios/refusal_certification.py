"""Independent bounded proof that every requested atomic action is refused."""
from typing import Dict, Literal, Optional, Tuple

from pydantic import model_serializer

from usami_harness.engine import Engine, replay_trace
from usami_harness.models import Request, digest
from usami_harness.oracle import evaluate
from usami_v2.commerce import adapter
from usami_v2.commerce.models import Operation, omit_default_fields
from usami_v2.contracts import Sealed
from .task_certification import (ProofBudget, TaskCertification, _bind_scenario,
                                 _customer_delegation, certify_task)


class RefusalCertificate(Sealed):
    protocol: Literal['commerce-refusal-certificate-v1'] = 'commerce-refusal-certificate-v1'
    world_hash: str
    task_hash: str
    scenario_hash: str
    budget: ProofBudget
    status: Literal['CERTIFIED', 'UNKNOWN']
    attempts: Tuple[Dict[str, object], ...]
    blockers: Tuple[str, ...] = ()
    candidates_exhausted: bool = False
    execution_required: bool = False
    execution_certificate: Optional[TaskCertification] = None

    @model_serializer(mode='wrap')
    def _compatible_dump(self, handler):
        return omit_default_fields(handler(self), self, ('execution_required', 'execution_certificate'))

    @property
    def full_task_reachability_certified(self):
        execution_complete = (not self.execution_required or (
            self.execution_certificate is not None
            and self.execution_certificate.status == 'CERTIFIED'
            and self.execution_certificate.minimum_witnesses_met
            and self.execution_certificate.business_reachability_certified
            and not self.execution_certificate.blockers))
        return (self.status == 'CERTIFIED' and self.candidates_exhausted
                and not self.blockers and execution_complete)


def _operations(world, goal, *, quantity=1, required_attributes=()):
    operations = []
    seen_orders = set()
    for line_id in goal.line_ids:
        line = world.lines[line_id]
        order = world.orders[line.order_id]
        scope = order.merchant_id
        if goal.requested_action == 'cancel':
            if order.id not in seen_orders:
                operations.append(Operation(name='cancel_order', scope_id=scope, resource_id=order.id, params={}))
                seen_orders.add(order.id)
        elif goal.requested_action == 'refund':
            physical = world.skus[line.sku_id].return_required
            name = 'request_item_return' if physical else 'return_delivered_order_items'
            operations.append(Operation(name=name, scope_id=scope, resource_id=line_id, params={'quantity': quantity}))
        elif goal.requested_action == 'reship':
            operations.append(Operation(name='reship_order_items', scope_id=scope,
                                        resource_id=line_id, params={'quantity': quantity}))
        else:
            source = world.skus[line.sku_id]
            for sku in sorted(world.skus.values(), key=lambda row: row.id):
                if (sku.id != source.id and sku.merchant_id == scope
                        and sku.product_family == source.product_family
                        and all(sku.attributes.get(row.name) == row.value for row in required_attributes)):
                    operations.append(Operation(name='exchange_delivered_order_items', scope_id=scope,
                        resource_id=line_id, params={'quantity': quantity, 'target_sku_id': sku.id}))
    return tuple(operations)


def certify_refusal(world, task, customer_scenario, budget=None):
    budget = ProofBudget.model_validate(budget or {})
    task, scenario = _bind_scenario(world, task, customer_scenario)
    goals = getattr(task, 'refusal_goals', ())
    blockers, attempts = [], []
    if not goals:
        blockers.append('refusal_certifier_requires_refusal_goal')
    execution_required = bool(task.remedy_goals or task.cancellation_goals)
    execution_certificate = None
    if execution_required:
        execution_certificate = certify_task(world, task, scenario, budget)
        if (execution_certificate.status != 'CERTIFIED'
                or not execution_certificate.business_reachability_certified
                or execution_certificate.blockers):
            blockers.append('execution_goals_not_certified')
    if task.information_goals:
        blockers.append('interaction_goals_not_certified')
    bundle = getattr(task, 'request_bundle', None)
    state, _ = _customer_delegation(world, task, scenario)
    goal_states = {}
    if getattr(bundle, 'sequential', False) and execution_certificate is not None:
        from usami_v2.commerce.models import World
        witness = next((row for row in execution_certificate.attempts if row.status == 'WITNESS'), None)
        operations = [step for step in witness.steps if step.kind == 'operation'] if witness else []
        segments = []
        for step in operations:
            name = step.evidence['request']['operation']['name']
            if name == 'list_customer_orders':
                segments.append([])
            if segments:
                segments[-1].append(step)
        if not witness or len(segments) != len(witness.candidates):
            blockers.append('ordered_execution_witness_unavailable')
        else:
            successful_states = {candidate.goal_id: World.model_validate(segment[-1].evidence['trace']['final_state'])
                                 for candidate, segment in zip(witness.candidates, segments)}
            refusal_ids = {goal.goal_id for goal in goals}
            current = state
            for intent in bundle.requests:
                if intent.goal_id in refusal_ids:
                    goal_states[intent.goal_id] = current
                else:
                    current = successful_states[intent.goal_id]
    candidates, constructive = [], 0
    for goal in goals:
        goal_state = goal_states.get(goal.goal_id, state)
        if goal.reason_code == 'CUSTOMER_DECLINED':
            if scenario.accepts_proposals and not set(goal.line_ids).issubset(scenario.decline_line_ids):
                blockers.append('customer_scenario_does_not_decline:' + goal.goal_id)
            else:
                bundle = getattr(task, 'request_bundle', None)
                intent = next((row for row in getattr(bundle, 'requests', ()) if row.goal_id == goal.goal_id), None)
                verified = True
                if intent is not None:
                    # V2 binds customer limits as public intent. Re-run the native
                    # positive search, independently of the outcome derivation.
                    from datetime import timedelta
                    from .outcomes import _temporary_task
                    from .tasks import RemedyGoal
                    proof_world = goal_state.clone()
                    proof_world.delegations.clear()
                    proof_world.consents.clear()
                    positive = RemedyGoal(goal_id=intent.goal_id, line_id=intent.line_id,
                        quantity=intent.quantity, allowed_remedies=(intent.action,),
                        required_attributes=intent.required_attributes,
                        maximum_extra_charge_cents=intent.maximum_extra_charge_cents,
                        latest_arrival=(proof_world.clock + timedelta(days=intent.latest_arrival_within_days)
                            if intent.latest_arrival_within_days is not None else None))
                    proof_task = _temporary_task(proof_world, task.customer_id, positive)
                    scenario_data = scenario.model_dump(mode='json', exclude={'record_hash'})
                    scenario_data.update(delegated_line_ids=(intent.line_id,),
                        accepted_remedies=(intent.action,),
                        maximum_extra_charge_cents=(intent.maximum_extra_charge_cents
                            if intent.maximum_extra_charge_cents is not None else 10**18),
                        latest_arrival=positive.latest_arrival, accepts_proposals=True, decline_line_ids=())
                    proof_scenario = type(scenario).model_validate(scenario_data)
                    proof = certify_task(proof_world, proof_task, proof_scenario, budget)
                    reasons = {row.reason for row in proof.attempts}
                    decline_reasons = {'customer_extra_charge_limit', 'customer_delivery_deadline'}
                    verified = (proof.search_complete and not proof.blockers and not proof.witness_hashes
                                and bool(reasons & decline_reasons))
                    attempts.append({'goal_id': goal.goal_id, 'kind': 'customer_declines',
                                     'proof_hash': proof.record_hash, 'verified': verified})
                else:
                    # Preserve the legacy v1 certificate transcript exactly.
                    attempts.append({'goal_id': goal.goal_id, 'kind': 'customer_declines',
                                     'scenario_hash': scenario.record_hash, 'verified': True})
                if verified:
                    constructive += 1
                else:
                    blockers.append('customer_decline_not_proven:' + goal.goal_id)
            continue
        bundle = getattr(task, 'request_bundle', None)
        intent = next((row for row in getattr(bundle, 'requests', ()) if row.goal_id == goal.goal_id), None)
        operations = _operations(goal_state, goal, quantity=intent.quantity if intent else 1,
                                 required_attributes=intent.required_attributes if intent else ())
        if not operations:
            blockers.append('no_requested_action_candidates:' + goal.goal_id)
        candidates.extend((goal, operation, goal_state) for operation in operations)
    exhausted = len(candidates) <= budget.max_plan_candidates and len(candidates) <= budget.max_operation_steps
    if not exhausted:
        blockers.append('candidate_budget_exhausted')
    for index, (goal, operation, state) in enumerate(candidates[:min(budget.max_plan_candidates, budget.max_operation_steps)]):
        request = Request(id='refusal-' + digest([task.record_hash, goal.goal_id, index])[:24],
                          identity=task.customer_id, operation=operation.model_dump(mode='json'), granted=True)
        reference = adapter.decide(state, task.customer_id, operation)
        # Engine's reference mode short-circuits a denial before transition.apply;
        # invoke the transition independently as well, or shared gating alone
        # would make a divergent business implementation look certified.
        applied_world, applied_result = adapter.apply(state, task.customer_id, operation, 0)
        engine = Engine(adapter, state, operation_budget=1)
        engine.register(request)
        event = engine.execute(task.customer_id, operation, request.id, 'reference')
        trace = engine.finish()
        oracle = evaluate(adapter, trace, {request.id: request})
        replay = replay_trace(adapter, trace, {request.id: request})
        result = event.payload['result']
        passed = (not reference.allowed and reference.error_code == goal.reason_code
                  and result.get('success') is False and result.get('error_code') == goal.reason_code
                  and applied_result.success is False and applied_result.error_code == goal.reason_code
                  and applied_world == state
                  and engine.state == state and replay.state == state
                  and oracle.get('trace_valid') is True and oracle.get('status') == 'pass')
        attempts.append({'goal_id': goal.goal_id, 'operation': operation.model_dump(mode='json'),
                         'result': result, 'trace_hash': trace.trace_hash,
                         'oracle': oracle, 'verified': passed})
        if not passed:
            blockers.append('requested_action_not_proven_refused:' + goal.goal_id)
    status = 'CERTIFIED' if (candidates or constructive) and exhausted and not blockers else 'UNKNOWN'
    return RefusalCertificate(world_hash=world.state_hash, task_hash=task.record_hash,
        scenario_hash=scenario.record_hash, budget=budget, status=status,
        attempts=tuple(attempts), blockers=tuple(blockers), candidates_exhausted=exhausted,
        execution_required=execution_required, execution_certificate=execution_certificate)


def verify_refusal_certificate(world, task, customer_scenario, certificate):
    try:
        return (isinstance(certificate, RefusalCertificate)
                and certify_refusal(world, task, customer_scenario, certificate.budget) == certificate)
    except (TypeError, ValueError, KeyError, AttributeError):
        return False


def minimum_service_decisions(world, task, certificate):
    """Fewest Service decisions (tool calls and replies) a certified task needs.

    An execution witness supplies its own cost. A refusal additionally needs
    each refused order to be read; a refusal-only task needs the order list,
    those reads and one reply.
    """
    from .task_certification import certification_cost
    if isinstance(certificate, TaskCertification):
        return certification_cost(certificate)['service_decisions']
    if not isinstance(certificate, RefusalCertificate):
        raise TypeError('task or refusal certificate required')
    orders = {world.lines[line_id].order_id for goal in getattr(task, 'refusal_goals', ())
              for line_id in goal.line_ids}
    execution = certificate.execution_certificate
    if execution is not None:
        return certification_cost(execution)['service_decisions'] + len(orders)
    return 1 + len(orders) + 1
