"""Actual public tool results; no private task, witness or preference labels."""
from datetime import timedelta

from .models import unit_credit


def query_data(world, actor, op):
    key, p = op.resource_id, op.params
    if op.name == "list_customer_orders":
        return {"orders": [row.model_dump(mode="json") for row in sorted(world.orders.values(), key=lambda x: x.id)
                           if row.customer_id == actor and row.merchant_id == op.scope_id]}
    if op.name == "get_order_details":
        return {"order": world.orders[key].model_dump(mode="json"),
                "lines": [row.model_dump(mode="json") for row in sorted(world.lines.values(), key=lambda x: x.id)
                          if row.order_id == key]}
    if op.name == "search_catalog":
        return {"skus": [row.model_dump(mode="json") for row in sorted(world.skus.values(), key=lambda x: x.id)
                         if row.merchant_id == op.scope_id
                         and (p["product_family"] is None or row.product_family == p["product_family"])
                         and all(row.attributes.get(k) == v for k, v in p["attributes"].items())]}
    if op.name == "get_sku_details":
        return {"sku": world.skus[key].model_dump(mode="json")}
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
    raise ValueError("unknown public query")
