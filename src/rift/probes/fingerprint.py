"""Content fingerprints for optional near-duplicate Probe filtering.

Bins are deliberately explicit; the filtering threshold remains disabled until
it has been calibrated against same-ability sibling instances.
"""
from dataclasses import dataclass
from math import inf


@dataclass(frozen=True)
class TaskFingerprint:
    discrete: tuple
    numeric: tuple


def task_fingerprint(world, task):
    entries = []
    targets = [(goal.line_id, tuple(sorted(goal.allowed_remedies)), goal.quantity)
               for goal in task.remedy_goals]
    for goal in task.cancellation_goals:
        targets.extend((line_id, ('cancel',), world.lines[line_id].quantity)
                       for line_id in goal.line_ids)
    for goal in getattr(task, 'refusal_goals', ()):
        targets.extend((line_id, ('refusal', goal.reason_code), world.lines[line_id].quantity)
                       for line_id in goal.line_ids)
    for line_id, remedies, quantity in targets:
        line = world.lines[line_id]
        sku, order = world.skus[line.sku_id], world.orders[line.order_id]
        age = max(0, (world.clock - order.delivered_at).days)
        discrete = (remedies, sku.product_family, _fingerprint_attributes(sku.attributes),
                    order.fulfillment_status, sku.return_required)
        numeric = (quantity, line.allocated_paid_cents // 5000, age // 7)
        entries.append((discrete, numeric))
    entries.sort()
    information = tuple(sorted((goal.fact_kind, 'line' if goal.subject_id in world.lines else 'order')
                               for goal in task.information_goals))
    return TaskFingerprint(tuple(item[0] for item in entries) + (('information', information), ('goal_count',
        len(task.remedy_goals) + len(task.cancellation_goals) + len(task.information_goals)
        + len(getattr(task, 'refusal_goals', ()))),),
        tuple(item[1] for item in entries))


def _fingerprint_attributes(attributes):
    """Drop per-instance labels while retaining product characteristics.

    The family generator seeds ``model`` names and uses ``instance`` solely to
    keep sibling products addressable. Those values must not make a copied task
    look like a different capability.
    """
    normalized = []
    for name, value in attributes.items():
        if name == 'instance':
            continue
        if name == 'model':
            value = '<instance-model>'
        normalized.append((name, value))
    return tuple(sorted(normalized))


def fingerprint_distance(left, right):
    """Chebyshev distance in numeric bins, or infinity for different categories."""
    if not isinstance(left, TaskFingerprint) or not isinstance(right, TaskFingerprint):
        raise TypeError('TaskFingerprint required')
    if left.discrete != right.discrete or len(left.numeric) != len(right.numeric):
        return inf
    return max((abs(a - b) for x, y in zip(left.numeric, right.numeric)
                for a, b in zip(x, y)), default=0)


def decision_fingerprint(spec, build, task, *, return_receipt_event_configured=False):
    """Exact judgement signature; the existing binned fingerprint stays unchanged."""
    from usami_v2.scenarios.signature import decision_signature
    return decision_signature(spec, build, task,
                              return_receipt_event_configured=return_receipt_event_configured)


def same_decision(left, right):
    """Use only fully derived signatures for exact decision deduplication."""
    from usami_v2.scenarios.signature import DecisionSignature
    if not isinstance(left, DecisionSignature) or not isinstance(right, DecisionSignature):
        raise TypeError('DecisionSignature required')
    return left.checked() == right.checked()
