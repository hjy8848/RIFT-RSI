"""Retail Designer instructions and generator-derived examples."""
import json
import random

from .prompt import failure_text, plan_numbers
from .retail_format import plan_from_scene

SYSTEM = """You design customer-service training tasks in an online store.
Write the decision variables as one JSON object. The environment generates all IDs,
attributes, addresses, customers and inventory and derives the correct answer.
Do not label requests as successful or refused; every task must contain at least one
request that changes the world or it will be discarded.

Rules: only unshipped orders can change their shipping address. A default address can
always be changed. An unshipped order can have its items modified only once, in one
call; afterward it cannot be changed or cancelled. Replacements must be in-stock
variants of the same product. Only unlocked unshipped orders can be cancelled, and
cancellation is for the entire order. Shipped or delivered orders cannot have items
or shipping addresses changed. A positive price difference can be paid with any of
the customer's methods; a gift card must have sufficient balance. Refunds of negative
differences can only go to the original payment method or a gift card.

Write hypothesis, gift_card_balance_cents, orders, variants and tasks. Orders have a
unique ref, status, pays_with and one to three items (ref, lowercase product name,
price_cents). Variants have a unique ref, the item ref in 'of', price_cents and
in_stock. Use one to four orders, zero to eight variants and one to three tasks.
A task has one to three requests and optional ask_total. Each request has a type:
set_order_address (order, to), set_default_address (to), or change_items (order,
changes [{item, want: [variant refs in preference order]}], pay_with and otherwise).
ask_total may be true only in a task that contains a change_items request; otherwise leave
it false. In change_items, every variant listed in a change's want must be a variant whose
'of' is exactly that change's item (a variant of another item is rejected), and the item must
be one of the items of the request's order.
Addresses are exactly one of: new, profile, or order:<order ref> (write the ref itself,
for example order:B; never profile:something). A default address can be new or order:<ref>,
never profile. An order's address source must never be that same order.
Payment preferences are original, gift_card, credit_card or paypal; otherwise is
nothing or cancel_order. Amounts are integer cents from 100 to 1000000; gift card
balance is 0 to 1000000. Refs must be unique and refer to existing entities.

Useful tasks put balances close to the price difference, put an unavailable variant
first, mix feasible and refused requests, refer to a profile or another order address,
change multiple items at once, or ask to cancel if no option works. Do not copy the
examples or failures. Explain the targeted weakness in your own words. Output JSON only."""


def example_retail_plans(seed, count=2):
    from usami_v2.scenarios.retail_tasks import FAMILIES, family_scene
    rng = random.Random(seed)
    offset = rng.randrange(len(FAMILIES))
    plans = []
    for n in range(count):
        family = FAMILIES[(offset + n) % len(FAMILIES)]
        scene, task = family_scene(family, random.Random(rng.randrange(10**9)))
        plans.append(plan_from_scene(scene, [task], "what this task trains, in your own words"))
    return plans


def retail_copy_score(plan_value, examples):
    mine = plan_numbers(plan_value)
    total = sum(mine.values())
    if not total:
        return 0.0
    def request_structure(value):
        refs = {}
        def ref(value):
            if value not in refs:
                refs[value] = len(refs)
            return refs[value]
        def walk(value):
            if isinstance(value, list):
                return [walk(row) for row in value]
            if not isinstance(value, dict):
                return value
            result = {}
            for key, item in sorted(value.items()):
                if key in ("order", "item") and item is not None:
                    result[key] = ref(item)
                elif key == "want":
                    result[key] = [ref(row) for row in item]
                elif key == "to" and isinstance(item, str) and item.startswith("order:"):
                    result[key] = ["order", ref(item[6:])]
                else:
                    result[key] = walk(item)
            return result
        return walk(value.get("tasks", []))
    best = 0.0
    for example in examples:
        value = example.model_dump(mode="json")
        overlap = sum((mine & plan_numbers(value)).values()) / total
        if overlap >= 0.5 and request_structure(plan_value) == request_structure(value):
            return 1.0
        best = max(best, overlap)
    return round(best, 3)


def retail_designer_messages(failures, *, seed, examples=2, feedback=(), focus=False):
    parts = ["Training failures of the agent:"]
    parts += [f"[{i + 1}]\n" + failure_text(case) for i, case in enumerate(failures)] or ["(none yet)"]
    if feedback:
        parts += ["Earlier plans rejected by the environment:"] + [f"- {row}" for row in feedback]
    parts.append("Examples of valid plans:")
    parts += [json.dumps(plan.model_dump(mode="json", exclude_defaults=True), ensure_ascii=False)
              for plan in example_retail_plans(seed, examples)]
    if focus:
        parts.append("Vary the decision structure and target the observed failures.")
    parts.append("Write one new plan as a single JSON object.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]
