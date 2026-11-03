"""Actual public tool results; no private task, witness or preference labels."""
from datetime import timedelta

from .models import public_sku, unit_credit


def _public_line_rows(world, order_key):
    """Order lines with the public catalog descriptor attached.

    A Service can only match the customer's words ("damaged USB C device") to a
    line if the order details carry the SKU family and attributes. The nested
    descriptor keeps the raw line record intact for existing consumers.
    """
    rows = []
    for line in sorted(world.lines.values(), key=lambda x: x.id):
        if line.order_id != order_key:
            continue
        row = line.model_dump(mode="json")
        sku = world.skus.get(line.sku_id)
        row["sku"] = public_sku(sku) if sku is not None else None
        settled = {unit for settlement in world.settlements.values() if settlement.line_id == line.id
                   for unit in settlement.unit_indices}
        if world.observation_protocol == 'commerce-observation-v2' and settled:
            row['settled_quantity'] = len(settled)
        case_ids = {case.id for case in world.cases.values() if case.line_id == line.id}
        returns = sorted({ret.status for ret in world.returns.values() if ret.case_id in case_ids})
        if world.observation_protocol == 'commerce-observation-v2' and returns:
            row['return_status'] = returns
        rows.append(row)
    return rows


def query_data(world, actor, op):
    key, p = op.resource_id, op.params
    if op.name == "list_customer_orders":
        return {"orders": [row.model_dump(mode="json") for row in sorted(world.orders.values(), key=lambda x: x.id)
                           if row.customer_id == actor and row.merchant_id == op.scope_id]}
    if op.name == "get_order_details":
        return {"order": world.orders[key].model_dump(mode="json"),
                "lines": _public_line_rows(world, key)}
    if op.name == "search_catalog":
        return {"skus": [public_sku(row) for row in sorted(world.skus.values(), key=lambda x: x.id)
                         if row.merchant_id == op.scope_id
                         and (p["product_family"] is None or row.product_family == p["product_family"])
                         and all(row.attributes.get(k) == v for k, v in p["attributes"].items())]}
    if op.name == "get_sku_details":
        return {"sku": public_sku(world.skus[key])}
    if op.name == "get_inventory":
        return {"inventory": [row.model_dump(mode="json") for row in sorted(world.inventory.values(), key=lambda x: x.id)
                              if row.sku_id == key]}
    if op.name == "estimate_delivery":
        route = next(row for row in world.routes.values() if row.warehouse_id == p["warehouse_id"])
        return {"sku_id": key, "warehouse_id": p["warehouse_id"], "quantity": p["quantity"],
                "arrival": (world.clock + timedelta(days=route.days)).isoformat(),
                "fee_cents": route.fee_cents, "route_revision": route.revision}
    if op.name == "get_refundable_balance":
        line = world.lines[key]
        spent = {i for row in world.settlements.values() if row.line_id == key for i in row.unit_indices}
        remaining = tuple(i for i in range(line.quantity) if i not in spent)
        return {"line_id": key, "remaining_units": list(remaining), "remaining_original_credit_cents": unit_credit(line, remaining)}
    if op.name == "get_case_status":
        case = world.cases[key]
        line = world.lines[case.line_id]
        return {"case": world.cases[key].model_dump(mode="json"),
                # Expose only evidence references belonging to this caller and
                # case line. Verification remains a separate business action;
                # do not reveal the hidden validity label in a query response.
                "evidence": [{"evidence_id": row.id, "issue": row.issue}
                             for row in sorted(world.evidence.values(), key=lambda x: x.id)
                             if row.customer_id == actor and row.line_id == line.id],
                "quotes": [row.model_dump(mode="json") for row in sorted(world.quotes.values(), key=lambda x: x.id)
                           if row.case_id == key],
                "returns": [row.model_dump(mode="json") for row in sorted(world.returns.values(), key=lambda x: x.id)
                            if row.case_id == key]}
    if op.name == "list_exchange_options":
        line = world.lines[key]
        original = world.skus[line.sku_id]
        options = []
        for sku in sorted(world.skus.values(), key=lambda x: x.id):
            if (sku.id == original.id or sku.merchant_id != original.merchant_id
                    or sku.product_family != original.product_family):
                continue
            stocked = []
            for inv in world.inventory.values():
                route = next((r for r in world.routes.values() if r.warehouse_id == inv.warehouse_id), None)
                if inv.sku_id == sku.id and inv.available > 0 and route is not None:
                    stocked.append((inv.available, route.days))
            options.append({"sku_id": sku.id, **public_sku(sku),
                            "available_quantity": sum(row[0] for row in stocked),
                            "fastest_delivery_days": min((row[1] for row in stocked), default=None)})
        return {"line_id": key, "options": options}
    raise ValueError("unknown public query")
