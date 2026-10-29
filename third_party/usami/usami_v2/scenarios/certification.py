"""Independent inspection of actual constructed business data.

No constructor, mutation function or Designer-supplied witness is called here.
Task reachability/interaction certification is a subsequent, separate layer.
"""
from datetime import datetime, timezone
from typing import Tuple

from usami_v2.contracts import Sealed
from usami_v2.commerce.invariants import invariants


class WorldCertification(Sealed):
    scenario_hash: str
    world_hash: str
    build_hash: str
    actual_assignment: Tuple[Tuple[str, str, str, int], ...]
    constraint_ids: Tuple[str, ...]
    evidence_kind: str = "business_world_constraints"
    task_reachability_certified: bool = False
    interaction_certified: bool = False


def verify_world_constraints(spec, result):
    spec, result = spec.checked(), result.checked()
    if result.status != "SAT" or result.world is None or result.scenario_hash != spec.record_hash:
        raise ValueError("actual SAT world matching the scenario required")
    world = result.world
    if invariants(world):
        raise ValueError("constructed world violates native business invariants")
    expected_aliases = {spec.merchant_alias, *spec.customer_aliases}
    expected_aliases.update(row.alias for row in spec.skus)
    expected_aliases.update(row.alias for row in spec.warehouses)
    expected_aliases.update(row.alias for row in spec.orders)
    expected_aliases.update(line.alias for order in spec.orders for line in order.lines)
    aliases = dict(result.private_aliases)
    if (set(aliases) != expected_aliases or len(aliases) != len(result.private_aliases)
            or len(set(aliases.values())) != len(aliases)):
        raise ValueError("scene aliases are missing, duplicated or not one-to-one")
    if world.seed != result.seed or world.clock != datetime(2026, 1, 15, 12, tzinfo=timezone.utc):
        raise ValueError("initial world seed or virtual business time differs")
    merchant = aliases[spec.merchant_alias]
    if (set(world.merchants) != {merchant}
            or world.merchants[merchant].refund_approval_limit_cents != spec.refund_approval_limit_cents):
        raise ValueError("merchant policy differs from registered scenario")
    if set(world.customers) != {aliases[name] for name in spec.customer_aliases}:
        raise ValueError("Customer population differs")
    if (set(world.skus) != {aliases[row.alias] for row in spec.skus}
            or set(world.warehouses) != {aliases[row.alias] for row in spec.warehouses}
            or set(world.orders) != {aliases[row.alias] for row in spec.orders}
            or set(world.lines) != {aliases[row.alias] for order in spec.orders for row in order.lines}):
        raise ValueError("business entity population differs")
    if any(getattr(world, key) for key in ("cases", "quotes", "delegations", "consents", "returns",
                                         "ledger", "settlements", "fulfillments", "external_events", "applied_events")):
        raise ValueError("initial scene contains unregistered transactions, authority or events")
    actual, bounds = {}, {}

    def capture(key, value, interval):
        if not interval.low <= value <= interval.high:
            raise ValueError("constructed value outside scenario interval: " + str(key))
        actual[key], bounds[key] = value, interval

    for spec_sku in spec.skus:
        row = world.skus[aliases[spec_sku.alias]]
        if (row.merchant_id != merchant or row.product_family != spec_sku.family
                or row.attributes != {item.name: item.value for item in spec_sku.attributes}
                or row.return_days != spec_sku.return_days or row.return_required != spec_sku.return_required
                or row.revision != 0):
            raise ValueError("SKU business attributes differ")
        capture((spec_sku.alias, "price_cents", ""), row.price_cents, spec_sku.price_cents)
    if len(world.routes) != len(spec.warehouses) or len(world.inventory) != len(spec.skus) * len(spec.warehouses):
        raise ValueError("route/inventory population differs")
    stock_specs = {(row.sku_alias, row.warehouse_alias): row.quantity for row in spec.stock}
    for warehouse_spec in spec.warehouses:
        wid = aliases[warehouse_spec.alias]
        if world.warehouses[wid].merchant_id != merchant:
            raise ValueError("warehouse merchant differs")
        route = next(row for row in world.routes.values() if row.warehouse_id == wid)
        if route.revision != 0:
            raise ValueError("unregistered route revision")
        capture((warehouse_spec.alias, "delivery_days", ""), route.days, warehouse_spec.delivery_days)
        capture((warehouse_spec.alias, "delivery_fee_cents", ""), route.fee_cents, warehouse_spec.delivery_fee_cents)
        for sku_spec in spec.skus:
            stock = next(row for row in world.inventory.values() if row.sku_id == aliases[sku_spec.alias]
                         and row.warehouse_id == wid)
            if stock.reserved or stock.shipped or stock.adjustments or stock.revision or stock.opening_quantity != stock.available:
                raise ValueError("initial inventory contains unregistered changes")
            capture((sku_spec.alias, "stock", warehouse_spec.alias), stock.available,
                    stock_specs.get((sku_spec.alias, warehouse_spec.alias), spec.default_stock))
    if len(world.evidence) != sum(len(order.lines) for order in spec.orders):
        raise ValueError("initial evidence population differs")
    for order_spec in spec.orders:
        order = world.orders[aliases[order_spec.alias]]
        if order.customer_id != aliases[order_spec.customer_alias] or order.merchant_id != merchant or order.payment_channel != "original-card":
            raise ValueError("order Customer, merchant or payment binding differs")
        seconds = (world.clock - order.delivered_at).total_seconds()
        if seconds % 86400:
            raise ValueError("delivery age does not match the registered day units")
        capture((order_spec.alias, "delivered_days_ago", ""), int(seconds // 86400), order_spec.delivered_days_ago)
        for line_spec in order_spec.lines:
            line = world.lines[aliases[line_spec.alias]]
            if line.order_id != order.id or line.sku_id != aliases[line_spec.original_sku]:
                raise ValueError("order-line binding differs")
            capture((line_spec.alias, "quantity", ""), line.quantity, line_spec.quantity)
            capture((line_spec.alias, "paid_cents", ""), line.allocated_paid_cents, line_spec.paid_cents)
            evidence = [row for row in world.evidence.values() if row.line_id == line.id]
            if (len(evidence) != 1 or evidence[0].customer_id != order.customer_id
                    or evidence[0].issue != line_spec.issue or evidence[0].valid != line_spec.evidence_valid):
                raise ValueError("Customer evidence differs from the scene")
    # Calculate predicates from actual world values, not the Builder's proof.
    def value(expr):
        return expr.constant + sum(term.coefficient * actual[term.key] for term in expr.terms)

    expected_proofs = []
    for relation in spec.constraints:
        left, right = value(relation.left), value(relation.right)
        allowed = {"eq": left == right, "ne": left != right, "lt": left < right,
                   "le": left <= right, "gt": left > right, "ge": left >= right}[relation.operator]
        if not allowed:
            raise ValueError("constructed world violates scene constraint: " + relation.constraint_id)
        expected_proofs.append((relation.constraint_id, left, relation.operator, right))
    supplied = [(row.constraint_id, row.left, row.operator, row.right) for row in result.proofs]
    if supplied != expected_proofs:
        raise ValueError("scene predicate proof differs from actual business values")
    assignment = tuple((*key, val) for key, val in sorted(actual.items()))
    if assignment != result.assignment:
        raise ValueError("Builder assignment differs from actual business data")
    return WorldCertification(
        scenario_hash=spec.record_hash, world_hash=world.state_hash, build_hash=result.record_hash,
        actual_assignment=assignment, constraint_ids=tuple(row.constraint_id for row in spec.constraints))
