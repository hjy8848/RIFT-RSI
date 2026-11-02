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


def _operations(world, goal):
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
            operations.append(Operation(name=name, scope_id=scope, resource_id=line_id, params={'quantity': 1}))
        elif goal.requested_action == 'reship':
            operations.append(Operation(name='reship_order_items', scope_id=scope,
                                        resource_id=line_id, params={'quantity': 1}))
        else:
            source = world.skus[line.sku_id]
            for sku in sorted(world.skus.values(), key=lambda row: row.id):
                if (sku.id != source.id and sku.merchant_id == scope
                        and sku.product_family == source.product_family):
                    operations.append(Operation(name='exchange_delivered_order_items', scope_id=scope,
                        resource_id=line_id, params={'quantity': 1, 'target_sku_id': sku.id}))
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
    state, _ = _customer_delegation(world, task, scenario)
    candidates, constructive = [], 0
    for goal in goals:
        if goal.reason_code == 'CUSTOMER_DECLINED':
            if scenario.accepts_proposals:
                blockers.append('customer_scenario_does_not_decline:' + goal.goal_id)
            else:
                # The bound Customer runtime enforces this policy. The actual
                # Episode must additionally contain a structured decline turn.
                attempts.append({'goal_id': goal.goal_id, 'kind': 'customer_declines',
                                 'scenario_hash': scenario.record_hash, 'verified': True})
                constructive += 1
            continue
        operations = _operations(state, goal)
        if not operations:
            blockers.append('no_requested_action_candidates:' + goal.goal_id)
        candidates.extend((goal, operation) for operation in operations)
    exhausted = len(candidates) <= budget.max_plan_candidates and len(candidates) <= budget.max_operation_steps
    if not exhausted:
        blockers.append('candidate_budget_exhausted')
    for index, (goal, operation) in enumerate(candidates[:min(budget.max_plan_candidates, budget.max_operation_steps)]):
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
