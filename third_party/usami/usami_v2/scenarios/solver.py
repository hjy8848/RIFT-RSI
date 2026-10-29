"""Bounded deterministic integer search with sound UNSAT/UNKNOWN separation."""
from itertools import product
import random


def variable_ranges(spec):
    result = {}
    for sku in spec.skus:
        result[(sku.alias, "price_cents", "")] = sku.price_cents
    for warehouse in spec.warehouses:
        result[(warehouse.alias, "delivery_days", "")] = warehouse.delivery_days
        result[(warehouse.alias, "delivery_fee_cents", "")] = warehouse.delivery_fee_cents
    for order in spec.orders:
        result[(order.alias, "delivered_days_ago", "")] = order.delivered_days_ago
        for line in order.lines:
            result[(line.alias, "quantity", "")] = line.quantity
            result[(line.alias, "paid_cents", "")] = line.paid_cents
    explicit = {(row.sku_alias, row.warehouse_alias): row.quantity for row in spec.stock}
    for sku in spec.skus:
        for warehouse in spec.warehouses:
            result[(sku.alias, "stock", warehouse.alias)] = explicit.get(
                (sku.alias, warehouse.alias), spec.default_stock)
    return result


def expression_value(expression, assignment):
    return expression.constant + sum(term.coefficient * assignment[term.key] for term in expression.terms)


def relation_satisfied(relation, assignment):
    left, right = expression_value(relation.left, assignment), expression_value(relation.right, assignment)
    return {"eq": left == right, "ne": left != right, "lt": left < right,
            "le": left <= right, "gt": left > right, "ge": left >= right}[relation.operator]


def impossible_by_interval(relation, ranges):
    coefficients = {}
    for sign, expression in ((1, relation.left), (-1, relation.right)):
        for term in expression.terms:
            coefficients[term.key] = coefficients.get(term.key, 0) + sign * term.coefficient
    low = high = relation.left.constant - relation.right.constant
    for key, coefficient in coefficients.items():
        interval = ranges[key]
        low += coefficient * (interval.low if coefficient >= 0 else interval.high)
        high += coefficient * (interval.high if coefficient >= 0 else interval.low)
    return {"eq": low > 0 or high < 0, "ne": low == high == 0,
            "lt": low >= 0, "le": low > 0, "gt": high <= 0, "ge": high < 0}[relation.operator]


def assignments(ranges, seed, max_nodes):
    """Enumerate complete finite spaces when affordable; otherwise sample boundedly."""
    keys = sorted(ranges)
    size = 1
    for key in keys:
        size *= ranges[key].high - ranges[key].low + 1
        if size > max_nodes:
            break
    if size <= max_nodes:
        for values in product(*(range(ranges[key].low, ranges[key].high + 1) for key in keys)):
            yield dict(zip(keys, values)), True
    else:
        rng = random.Random(seed)
        for _ in range(max_nodes):
            yield {key: rng.randint(ranges[key].low, ranges[key].high) for key in keys}, False
