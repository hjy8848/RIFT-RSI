"""ID-independent signatures of native commerce decisions.

Perturbations are counterfactual probes, not new certified worlds or labels.
An unsupported counterfactual is ignored; it is never treated as a flip.
"""
from datetime import datetime, timedelta
from typing import Tuple

from usami_v2.commerce.invariants import invariants
from usami_v2.commerce import adapter
from usami_v2.contracts import Sealed
from .builder import _construct
from .outcomes import OutcomeUnsupported, _temporary_task, derive_outcomes, ordered_contexts
from .task_certification import CustomerScenario, ProofBudget, _customer_delegation, certify_task
from .refusal_certification import _operations
from .tasks import (CustomerRequestIntent, OrderCancellationGoal, RefusalGoal, RemedyGoal,
                    delegated_line_ids)


class GoalDecisionSignature(Sealed):
    action: str
    outcome: str
    tool_sequence: Tuple[str, ...]
    critical_facts: Tuple[str, ...]
    reference: str = 'attributes'
    feasible_candidates: int = 0


class DecisionSignature(Sealed):
    goals: Tuple[GoalDecisionSignature, ...]
    goal_count: int
    shared_line: bool = False
    shared_inventory: bool = False
    near_distractors: int = 0


def _intents(world, task):
    bundle = getattr(task, 'request_bundle', None)
    if bundle is not None and getattr(bundle, 'requests', ()):
        return bundle.requests
    rows = []
    for goal in task.remedy_goals:
        if len(goal.allowed_remedies) != 1:
            raise ValueError('decision signatures require one requested action per goal')
        days = None if goal.latest_arrival is None else int((goal.latest_arrival - world.clock).total_seconds() // 86400)
        rows.append(CustomerRequestIntent(goal_id=goal.goal_id, line_id=goal.line_id,
            action=goal.allowed_remedies[0], quantity=goal.quantity, required_attributes=goal.required_attributes,
            maximum_extra_charge_cents=goal.maximum_extra_charge_cents, latest_arrival_within_days=days))
    for goal in task.cancellation_goals:
        line = world.lines[goal.line_ids[0]]
        rows.append(CustomerRequestIntent(goal_id=goal.goal_id, line_id=line.id, action='cancel', quantity=line.quantity))
    for goal in getattr(task, 'refusal_goals', ()):
        line = world.lines[goal.line_ids[0]]
        rows.append(CustomerRequestIntent(goal_id=goal.goal_id, line_id=line.id, action=goal.requested_action,
            quantity=1, maximum_extra_charge_cents=goal.maximum_extra_charge_cents))
    if task.information_goals:
        raise ValueError('information objectives have no atomic decision signature')
    return tuple(rows)


def _outcome(goal):
    return goal.reason_code if isinstance(goal, RefusalGoal) else 'success'


def _probe(world, customer, intent, receipt):
    line = world.lines[intent.line_id]
    # Existing returns require a multi-step witness. For a new physical return,
    # the accepted request is sufficient for a perturbation only when the
    # configured receipt event can complete it.
    if not world.cases:
        if (intent.action == 'reship' and any(world.skus[line.sku_id].attributes.get(row.name) != row.value
                                              for row in intent.required_attributes)):
            return None
        lines = (tuple(sorted(row.id for row in world.lines.values() if row.order_id == line.order_id))
                 if intent.action == 'cancel' else (line.id,))
        refusal = RefusalGoal(goal_id=intent.goal_id, line_ids=lines, reason_code='CUSTOMER_DECLINED',
                              requested_action=intent.action,
                              maximum_extra_charge_cents=intent.maximum_extra_charge_cents)
        task = _temporary_task(world, customer, refusal, refusal=True)
        scenario = CustomerScenario(customer_id=customer, delegated_line_ids=delegated_line_ids(task))
        state, _ = _customer_delegation(world, task, scenario)
        operations = _operations(state, refusal, quantity=intent.quantity,
                                 required_attributes=intent.required_attributes)
        decisions = [adapter.decide(state, customer, operation) for operation in operations]
        declined = False
        codes = set()
        for decision in decisions:
            if not decision.allowed:
                codes.add(decision.error_code)
                continue
            if world.skus[line.sku_id].return_required and intent.action == 'refund' and not receipt:
                return None
            data = decision.data or {}
            amount = data.get('extra_charge_cents', 0)
            arrival = data.get('promised_arrival')
            too_expensive = (intent.maximum_extra_charge_cents is not None
                             and amount > intent.maximum_extra_charge_cents)
            too_late = (intent.latest_arrival_within_days is not None and arrival is not None
                        and datetime.fromisoformat(arrival) > world.clock + timedelta(days=intent.latest_arrival_within_days))
            if not too_expensive and not too_late:
                return 'success'
            declined = True
        if declined:
            return 'CUSTOMER_DECLINED'
        if len(codes) == 1:
            code = next(iter(codes))
            from .outcomes import SUPPORTED
            return code if code in SUPPORTED else None
        return None
    try:
        return _outcome(derive_outcomes(world, customer, (intent,),
            return_receipt_event_configured=receipt)[0])
    except (OutcomeUnsupported, ValueError):
        return None


def _paths(world, customer, intent, receipt):
    line = world.lines[intent.line_id]
    if intent.action == 'cancel':
        order = world.orders[line.order_id]
        lines = tuple(sorted(row.id for row in world.lines.values() if row.order_id == order.id))
        goal = OrderCancellationGoal(goal_id=intent.goal_id, order_id=order.id, line_ids=lines)
    else:
        goal = RemedyGoal(goal_id=intent.goal_id, line_id=line.id, quantity=intent.quantity,
            allowed_remedies=(intent.action,), required_attributes=intent.required_attributes,
            maximum_extra_charge_cents=intent.maximum_extra_charge_cents,
            latest_arrival=(world.clock + timedelta(days=intent.latest_arrival_within_days)
                            if intent.latest_arrival_within_days is not None else None))
    task = _temporary_task(world, customer, goal)
    scenario = CustomerScenario(customer_id=customer, delegated_line_ids=delegated_line_ids(task),
        accepted_remedies=(intent.action,) if intent.action != 'cancel' else (),
        maximum_extra_charge_cents=(intent.maximum_extra_charge_cents
            if intent.maximum_extra_charge_cents is not None else 10**18),
        latest_arrival=goal.latest_arrival if isinstance(goal, RemedyGoal) else None,
        return_received_after_days=0 if receipt else None)
    proof = certify_task(world, task, scenario, ProofBudget(max_witnesses=32))
    if not proof.search_complete:
        raise ValueError('candidate search budget is insufficient for a decision signature')
    witnesses = [row for row in proof.attempts if row.status == 'WITNESS']
    selected = witnesses[0] if witnesses else (proof.attempts[0] if proof.attempts else None)
    path = []
    if selected:
        for step in selected.steps:
            if step.kind == 'operation':
                path.append(step.evidence['request']['operation']['name'])
            elif step.kind == 'external_event' and step.evidence.get('action') == 'apply':
                path.append('return_received')
    return tuple(path), len(witnesses) if intent.action == 'exchange' else 0


def _values(value, thresholds):
    result = {0, value - 1, value + 1, value // 2, value * 2}
    if thresholds:
        nearest = min(thresholds, key=lambda item: (abs(item - value), item))
        result.update((nearest - 1, nearest, nearest + 1))
    return sorted(result - {value})


def decision_signature(spec, build, task, *, return_receipt_event_configured=False):
    """Measure outcomes, Reference paths and outcome-changing numeric facts."""
    if build.status != 'SAT' or build.world_hash != task.world_hash:
        raise ValueError('signature requires the task and its SAT world')
    world = build.world
    intents = _intents(world, task)
    sequential = bool(getattr(getattr(task, 'request_bundle', None), 'sequential', False))
    contexts = (ordered_contexts(world, task.customer_id, intents,
                    return_receipt_event_configured=return_receipt_event_configured)
                if sequential else tuple((None, world) for _ in intents))
    registered = {row.goal_id: _outcome(row) for row in task.remedy_goals + task.cancellation_goals
                  + getattr(task, 'refusal_goals', ())}
    values = {(a, b, c): value for a, b, c, value in build.assignment}
    receipt = return_receipt_event_configured
    signatures = []
    for index, intent in enumerate(intents):
        context = contexts[index][1]
        baseline = _outcome(contexts[index][0]) if sequential else _probe(world, task.customer_id, intent, receipt)
        if baseline is None:
            raise ValueError('initial request outcome cannot be independently derived')
        if baseline != registered[intent.goal_id]:
            raise ValueError('derived outcome differs from the registered task goal')
        path, count = _paths(context, task.customer_id, intent, receipt)
        line = world.lines[intent.line_id]
        sku = world.skus[line.sku_id]
        order = world.orders[line.order_id]
        merchant = world.merchants[order.merchant_id]
        critical = set()
        thresholds = [intent.quantity, sku.return_days, merchant.refund_approval_limit_cents,
                      line.allocated_paid_cents, sku.price_cents]
        if intent.maximum_extra_charge_cents is not None:
            thresholds.append(intent.maximum_extra_charge_cents)
        if intent.latest_arrival_within_days is not None:
            thresholds.append(intent.latest_arrival_within_days)
        labels = {'delivered_days_ago': 'delivery_age vs return_days',
                  'paid_cents': 'credit vs approval_or_charge_limit',
                  'price_cents': 'replacement_price vs charge_limit',
                  'quantity': 'quantity vs purchased_or_stock',
                  'stock': 'stock vs quantity', 'delivery_days': 'delivery_days vs deadline',
                  'delivery_fee_cents': 'delivery_fee vs charge_limit'}
        def perturbed_world(changed_spec, assignment):
            candidate_world, aliases = _construct(changed_spec, build.seed, assignment)
            if changed_spec.history:
                from .history import apply_history
                candidate_world = apply_history(candidate_world, dict(aliases), changed_spec.history)
            return candidate_world

        def perturbed_outcome(candidate_world, candidate_intent):
            if not sequential:
                return _probe(candidate_world, task.customer_id, candidate_intent, receipt)
            altered = list(intents[:index + 1])
            altered[-1] = candidate_intent
            try:
                return _outcome(derive_outcomes(candidate_world, task.customer_id, tuple(altered),
                    return_receipt_event_configured=receipt, sequential=True)[-1])
            except (OutcomeUnsupported, ValueError):
                return None
        for key, value in sorted(values.items()):
            minimum = 1 if key[1] in ('price_cents', 'quantity') else 0
            maximum = 1000 if key[1] == 'quantity' else 10**9
            for candidate in _values(value, thresholds):
                if not minimum <= candidate <= maximum:
                    continue
                changed = dict(values)
                changed[key] = candidate
                try:
                    changed_world = perturbed_world(spec, changed)
                except ValueError:
                    continue
                if invariants(changed_world):
                    continue
                result = perturbed_outcome(changed_world, intent)
                if result is not None and result != baseline:
                    critical.add(labels[key[1]])
                    break
        for kind, current, thresholds_for_kind in (
            ('approval_limit', spec.refund_approval_limit_cents, [line.allocated_paid_cents]),
            ('return_days', sku.return_days, [max(0, (world.clock - order.delivered_at).days)])):
            for candidate in _values(current, thresholds_for_kind):
                if candidate < 0:
                    continue
                if kind == 'approval_limit':
                    changed_spec = spec.model_copy(update={'refund_approval_limit_cents': candidate})
                else:
                    aliases = dict(build.private_aliases)
                    skus = tuple(row.model_copy(update={'return_days': candidate}) if aliases[row.alias] == sku.id else row
                                 for row in spec.skus)
                    changed_spec = spec.model_copy(update={'skus': skus})
                try:
                    changed_world = perturbed_world(changed_spec, values)
                except ValueError:
                    continue
                if invariants(changed_world):
                    continue
                result = perturbed_outcome(changed_world, intent)
                if result is not None and result != baseline:
                    critical.add('credit vs approval_limit' if kind == 'approval_limit' else 'delivery_age vs return_days')
                    break
        for field, label in (('maximum_extra_charge_cents', 'extra_charge vs customer_limit'),
                             ('latest_arrival_within_days', 'delivery_days vs deadline')):
            current = getattr(intent, field)
            if current is None:
                continue
            for candidate in _values(current, thresholds):
                if candidate < 0:
                    continue
                changed = intent.model_dump(mode='json', exclude={'record_hash'})
                changed[field] = candidate
                result = perturbed_outcome(world, CustomerRequestIntent.model_validate(changed))
                if result is not None and result != baseline:
                    critical.add(label)
                    break
        signatures.append(GoalDecisionSignature(action=intent.action, outcome=baseline, tool_sequence=path,
            critical_facts=tuple(sorted(critical)), reference=intent.reference, feasible_candidates=count))
    signatures.sort(key=lambda row: (row.action, row.outcome, row.tool_sequence, row.critical_facts,
                                     row.reference, row.feasible_candidates))
    target_ids = {row.line_id for row in intents}
    families = {world.skus[world.lines[line_id].sku_id].product_family for line_id in target_ids}
    nearby = sum(1 for row in world.lines.values() if row.id not in target_ids
                 and world.orders[row.order_id].customer_id == task.customer_id
                 and world.skus[row.sku_id].product_family in families)
    exchange_families = [world.skus[world.lines[row.line_id].sku_id].product_family
                         for row in intents if row.action in ('exchange', 'reship')]
    return DecisionSignature(goals=tuple(signatures), goal_count=len(intents),
        shared_line=len(target_ids) < len(intents),
        shared_inventory=len(set(exchange_families)) < len(exchange_families), near_distractors=nearby)
