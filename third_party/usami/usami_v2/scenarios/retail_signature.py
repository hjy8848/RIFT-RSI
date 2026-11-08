"""Decision-structure signatures for certified retail tasks."""
from usami_harness.models import digest
from usami_v2.commerce.reference import decide
from .retail_tasks import _address_params, _operation, delegated, gold_operations, projection


def retail_signature(world, task):
    current = delegated(world, task.customer_id)
    naive = delegated(world, task.customer_id)
    rows = []
    for goal in task.goals:
        start = current
        isolated = task.model_copy(update={"world_hash": current.state_hash, "goals": (goal,), "ask_total": False})
        operations, current = gold_operations(current, isolated)
        row = {"action": goal.action,
               "order_status": start.orders[goal.order_id].fulfillment_status if goal.order_id else None,
               "address_source": goal.address_source if goal.action != "change_items" else None,
               "changed": projection(start) != projection(current), "fallback": False}
        if goal.action == "change_items":
            chosen = next((op for op in operations if op.name == "modify_order_items"), None)
            row["choices"] = []
            for wish in goal.wishes:
                actual = next((c["target_sku_id"] for c in chosen.params["changes"]
                               if c["line_id"] == wish.line_id), None) if chosen else None
                row["choices"].append(wish.targets.index(actual) if actual in wish.targets else None)
            row["fallback"] = any(op.name == "cancel_order" for op in operations)
            if chosen:
                delta = sum(start.skus[c["target_sku_id"]].price_cents -
                            start.lines[c["line_id"]].allocated_paid_cents for c in chosen.params["changes"])
                payment = chosen.params["payment_method"]
                resolved = start.orders[goal.order_id].payment_method_id if payment == "original" else payment
                method = start.payment_methods.get(resolved)
                card = next((p for p in start.payment_methods.values() if p.customer_id == task.customer_id
                             and p.kind == "gift_card"), None)
                row.update(direction="more" if delta > 0 else "less" if delta < 0 else "zero",
                           payment_index=goal.payment_preferences.index(payment),
                           payment_kind=method.kind if method else None,
                           gift_card_sufficient=bool(card and card.balance_cents >= max(delta, 0)))
            else:
                row.update(direction=None, payment_index=None, payment_kind=None, gift_card_sufficient=None)
            op = _operation(naive, "modify_order_items", goal.order_id,
                {"changes": [{"line_id": w.line_id, "target_sku_id": w.targets[0]} for w in goal.wishes],
                 "payment_method": goal.payment_preferences[0]})
        elif goal.action == "set_order_address":
            op = _operation(naive, "modify_order_address", goal.order_id, _address_params(goal.address))
        else:
            op = _operation(naive, "update_default_address", next(iter(naive.merchants)), _address_params(goal.address))
        decision = decide(naive, task.customer_id, op)
        row["naive_refusal"] = [] if decision.allowed else [decision.error_code]
        if decision.allowed:
            naive = decision.expected_state
        rows.append(row)
    return digest({"goals": rows, "ask_total": task.ask_total})
