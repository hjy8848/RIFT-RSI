"""Retail tasks: addresses and one-shot item changes, judged by the final business state.

A task is structured customer intent (goals with preferences and fallbacks) bound to a frozen world. The
customer text is rendered from the goals. What counts as success is the *golden final state*: the ideal
Service follows the customer's preference order using the independent Reference, and an episode succeeds
when the business-relevant part of its final world equals that state (tau-bench's database-state check).
The engine, not a person, therefore derives the right answer for every branch of a preference chain.
"""
import random
from datetime import timedelta
from typing import Literal, Optional, Tuple

from pydantic import Field, model_validator

from usami_harness.models import digest
from usami_v2.commerce import adapter
from usami_v2.commerce.invariants import invariants
from usami_v2.commerce.models import Delegation, Operation, World
from usami_v2.commerce.reference import decide
from usami_v2.commerce.transitions import apply
from usami_v2.contracts import Sealed

PROTOCOL = "retail-task-v1"


class AddressFields(Sealed):
    street: str = Field(min_length=1)
    city: str = Field(min_length=1)
    state: str = Field(min_length=1)
    postal_code: str = Field(min_length=1)

    def text(self):
        return "{}, {}, {} {}".format(self.street, self.city, self.state, self.postal_code)


class ItemWish(Sealed):
    line_id: str
    targets: Tuple[str, ...] = Field(min_length=1)       # SKU ids, most wanted first


class RetailGoal(Sealed):
    goal_id: str
    action: Literal["set_order_address", "set_default_address", "change_items"]
    order_id: Optional[str] = None
    address: Optional[AddressFields] = None
    address_source: Literal["stated", "other_order", "profile"] = "stated"
    source_order_id: Optional[str] = None
    wishes: Tuple[ItemWish, ...] = ()
    payment_preferences: Tuple[str, ...] = ("original",)  # payment method ids or 'original', most wanted first
    fallback: Literal["nothing", "cancel_order"] = "nothing"

    @model_validator(mode="after")
    def coherent(self):
        if self.action == "change_items":
            if not self.order_id or not self.wishes or self.address is not None:
                raise ValueError("an item change names its order and wishes")
            if len({row.line_id for row in self.wishes}) != len(self.wishes):
                raise ValueError("one wish per line")
        else:
            if self.wishes or (self.action == "set_order_address") != bool(self.order_id):
                raise ValueError("an address goal names an order exactly when it changes an order address")
            if self.address is None:
                raise ValueError("an address goal carries its target address")
            if (self.address_source == "other_order") != bool(self.source_order_id):
                raise ValueError("the source order is named exactly for other_order addresses")
        return self


class RetailTask(Sealed):
    protocol: Literal["retail-task-v1"] = PROTOCOL
    task_id: str
    world_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    customer_id: str
    family: str
    goals: Tuple[RetailGoal, ...] = Field(min_length=1)
    ask_total: bool = False
    customer_request: str = Field(min_length=1)
    partition: Literal["train", "validation", "heldout"] = "train"

    @model_validator(mode="after")
    def unique_goals(self):
        if len({row.goal_id for row in self.goals}) != len(self.goals):
            raise ValueError("duplicate goal id")
        if self.ask_total and not any(row.action == "change_items" for row in self.goals):
            raise ValueError("a total can only be asked after an item change")
        return self


# --------------------------------------------------------------------------- public wording

def _name(value):
    return " ".join(str(value).replace("_", " ").split())


def _item(world, line_id):
    sku = world.skus[world.lines[line_id].sku_id]
    parts = [_name(value) for _, value in sorted(sku.attributes.items()) if value]
    return " ".join(parts + [_name(sku.product_family)])


def _order_phrase(world, customer_id, order_id):
    def phrase(key):
        lines = sorted(row.id for row in world.lines.values() if row.order_id == key)
        return "order containing the " + " and the ".join(_item(world, line) for line in lines)
    mine = [row.id for row in world.orders.values() if row.customer_id == customer_id]
    text = phrase(order_id)
    if sum(phrase(key) == text for key in mine) != 1:
        raise ValueError("AMBIGUOUS_TARGET: the order cannot be told apart by its items")
    return text


def _variant_phrase(world, original_sku_id, target_sku_id):
    original, target = world.skus[original_sku_id], world.skus[target_sku_id]
    names = sorted(name for name, value in target.attributes.items() if original.attributes.get(name) != value)
    rivals = [row for row in world.skus.values() if row.product_family == original.product_family
              and row.merchant_id == original.merchant_id and row.id != original.id]
    for size in range(1, len(names) + 1):
        for start in range(len(names) - size + 1):
            chosen = names[start:start + size]
            if [row.id for row in rivals if all(row.attributes.get(n) == target.attributes[n] for n in chosen)] == [target.id]:
                return " ".join(_name(target.attributes[n]) for n in chosen) + " " + _name(target.product_family)
    raise ValueError("AMBIGUOUS_TARGET: the variant cannot be told apart by its attributes")


def _payment_phrase(world, customer_id, method):
    if method == "original":
        return "my original payment method"
    held = world.payment_methods[method]
    same = [row for row in world.payment_methods.values() if row.customer_id == customer_id and row.kind == held.kind]
    if len(same) != 1:
        raise ValueError("AMBIGUOUS_TARGET: the payment method cannot be told apart by its kind")
    return {"gift_card": "my gift card", "credit_card": "my credit card", "paypal": "my PayPal account"}[held.kind]


def _address_phrase(world, task_customer, goal):
    if goal.address_source == "stated":
        return goal.address.text()
    if goal.address_source == "other_order":
        return "the same address as my " + _order_phrase(world, task_customer, goal.source_order_id)
    return "the default address on my profile"


def render_retail_request(world, task):
    clauses = []
    for goal in task.goals:
        if goal.action == "set_order_address":
            text = "Please change the shipping address of my {} to {}.".format(
                _order_phrase(world, task.customer_id, goal.order_id), _address_phrase(world, task.customer_id, goal))
        elif goal.action == "set_default_address":
            text = "Please update the default address on my profile to {}.".format(
                _address_phrase(world, task.customer_id, goal))
        else:
            changes = []
            for wish in goal.wishes:
                original = world.lines[wish.line_id].sku_id
                options = [_variant_phrase(world, original, target) for target in wish.targets]
                piece = "the {} to the {}".format(_item(world, wish.line_id), options[0])
                for fallback in options[1:]:
                    piece += ", or if that is not available the {}".format(fallback)
                changes.append(piece)
            text = "In my {}, please change {}.".format(_order_phrase(world, task.customer_id, goal.order_id),
                                                       "; and ".join(changes))
            if len(changes) > 1:
                text += " Please make all of these changes in one go."
            methods = [_payment_phrase(world, task.customer_id, row) for row in goal.payment_preferences]
            text += " Settle any price difference with {}".format(methods[0])
            for fallback in methods[1:]:
                text += ", or if that is not possible with {}".format(fallback)
            text += "."
            text += (" If none of that is possible, cancel the order instead." if goal.fallback == "cancel_order"
                     else " If it is not possible, leave the order as it is.")
        clauses.append("{}. {}".format(len(clauses) + 1, text))
    if task.ask_total:
        clauses.append("{}. Then tell me the new total of that order in CNY.".format(len(clauses) + 1))
    return "Please help me with the following request(s):\n" + "\n".join(clauses)


# --------------------------------------------------------------------------- golden state

def delegated(world, customer_id):
    """The customer authorizes every line of their own orders (what their request grants)."""
    new = world.clone()
    lines = tuple(sorted(row.id for row in world.lines.values()
                         if world.orders[row.order_id].customer_id == customer_id))
    key = "delegation_" + digest([customer_id, list(lines)])[:20]
    new.delegations[key] = Delegation(
        id=key, customer_id=customer_id, line_ids=lines, allowed_preparations=(),
        expires_at=world.clock + timedelta(days=30), source_customer_event_hash=digest(["retail task", key]))
    return new


def _operation(world, name, resource, params):
    return Operation(name=name, scope_id=next(iter(world.merchants)), resource_id=resource, params=params)


def _address_params(address):
    return {"street": address.street, "city": address.city, "state": address.state, "postal_code": address.postal_code}


def gold_operations(world, task):
    """(operations, final world) of the ideal Service; each goal follows its preference chain."""
    current, ops, actor = delegated(world, task.customer_id), [], task.customer_id

    def attempt(operation):
        nonlocal current
        decision = decide(current, actor, operation)
        if decision.allowed:
            current = decision.expected_state
            ops.append(operation)
        return decision.allowed

    for goal in task.goals:
        if goal.action in ("set_order_address", "set_default_address"):
            if goal.address_source == "other_order":
                found = current.orders[goal.source_order_id].shipping_address
            elif goal.address_source == "profile":
                found = current.customers[actor].default_address
            else:
                found = None
            fields = _address_params(found) if found is not None else _address_params(goal.address)
            if goal.action == "set_order_address":
                attempt(_operation(current, "modify_order_address", goal.order_id, fields))
            else:
                attempt(_operation(current, "update_default_address", next(iter(current.merchants)), fields))
            continue
        chosen = []
        for wish in goal.wishes:
            for target in wish.targets:
                trial = chosen + [{"line_id": wish.line_id, "target_sku_id": target}]
                if any(decide(current, actor, _operation(current, "modify_order_items", goal.order_id, {
                        "changes": trial, "payment_method": pay})).allowed for pay in goal.payment_preferences):
                    chosen = trial
                    break
        done = False
        if chosen:
            for pay in goal.payment_preferences:
                if attempt(_operation(current, "modify_order_items", goal.order_id,
                                      {"changes": chosen, "payment_method": pay})):
                    done = True
                    break
        if not done and goal.fallback == "cancel_order":
            attempt(_operation(current, "cancel_order", goal.order_id, {}))
    return ops, current


def projection(world):
    """The business facts a task can change; authority records and internal cases are not part of it."""
    return digest({
        "orders": {k: [o.shipping_address.model_dump(mode="json") if o.shipping_address else None,
                       o.paid_cents, o.items_locked] for k, o in sorted(world.orders.items())},
        "lines": {k: [l.sku_id, l.reserved_inventory_id, l.quantity, l.allocated_paid_cents]
                  for k, l in sorted(world.lines.items())},
        "customers": {k: c.default_address.model_dump(mode="json") if c.default_address else None
                      for k, c in sorted(world.customers.items())},
        "payments": {k: p.balance_cents for k, p in sorted(world.payment_methods.items())},
        "modifications": {k: m.model_dump(mode="json") for k, m in sorted(world.modifications.items())},
        "settlements": sorted([s.line_id, list(s.unit_indices), s.remedy] for s in world.settlements.values()),
        "inventory": {k: [i.available, i.reserved, i.shipped] for k, i in sorted(world.inventory.items())},
    })


def expected_total_text(world, task):
    """The new order total the customer asked for, in CNY with two decimals (after the golden changes)."""
    _, final = gold_operations(world, task)
    goal = next(row for row in task.goals if row.action == "change_items")
    return "{:.2f}".format(final.orders[goal.order_id].paid_cents / 100)


def mentions_amount(text, amount_text):
    """Does the message state this amount (in yuan) as a number, however it is formatted?

    "99", "99.00", "CNY99", "1,299.50" and "93 yuan" all count; 13,600 for a 136.00 total does not
    (cents mistaken for yuan is a wrong answer, not a formatting difference).
    """
    import re
    target = float(amount_text)
    for raw in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        try:
            if abs(float(raw.replace(",", "")) - target) < 0.005:
                return True
        except ValueError:
            continue
    return False


def evaluate_retail_task(initial, final, task, service_texts=()):
    """Business-state success (final state equals the golden state) plus the told-total check."""
    if initial.state_hash != task.world_hash:
        raise ValueError("task belongs to another world")
    _, gold = gold_operations(initial, task)
    state_ok = projection(final) == projection(gold) and not invariants(final)
    told = True
    if task.ask_total:
        told = any(mentions_amount(text, expected_total_text(initial, task)) for text in service_texts)
    return {"state_match": state_ok, "told_total": told, "success": state_ok and told}


# --------------------------------------------------------------------------- certification

def certify_retail_task(world, task):
    """Native evidence that the task is well formed and its golden path is real and reproducible."""
    task = task.checked()
    errors = []
    if world.state_hash != task.world_hash or invariants(world):
        raise ValueError("task world binding or native invariants differ")
    owned = {row.id for row in world.orders.values() if row.customer_id == task.customer_id}
    for goal in task.goals:
        if (goal.order_id and goal.order_id not in owned) or (goal.source_order_id and goal.source_order_id not in owned):
            raise ValueError("goal crosses the frozen Customer identity")
        for wish in goal.wishes:
            if wish.line_id not in world.lines or world.lines[wish.line_id].order_id != goal.order_id:
                raise ValueError("wish names a line outside its order")
        for method in goal.payment_preferences:
            if method != "original" and (method not in world.payment_methods
                                         or world.payment_methods[method].customer_id != task.customer_id):
                raise ValueError("payment preference names a method the Customer does not own")
    if render_retail_request(world, task) != task.customer_request:
        raise ValueError("public request was not rendered from its structured intent")
    ops, gold = gold_operations(world, task)
    start = delegated(world, task.customer_id)
    replay = start
    for op in ops:
        decision = decide(replay, task.customer_id, op)
        actual, result = apply(replay, task.customer_id, op, 0)
        if not (decision.allowed and result.success and actual == decision.expected_state):
            errors.append("engine and reference disagree on " + op.name)
        replay = decision.expected_state
    if invariants(gold):
        errors.append("golden state violates invariants")
    changes = projection(gold) != projection(world)
    # A trap: doing what the customer asked first, literally, without reading the situation, ends elsewhere.
    naive = delegated(world, task.customer_id)
    for goal in task.goals:
        if goal.action == "change_items":
            op = _operation(naive, "modify_order_items", goal.order_id, {
                "changes": [{"line_id": w.line_id, "target_sku_id": w.targets[0]} for w in goal.wishes],
                "payment_method": goal.payment_preferences[0]})
        elif goal.action == "set_order_address":
            op = _operation(naive, "modify_order_address", goal.order_id, _address_params(goal.address))
        else:
            op = _operation(naive, "update_default_address", next(iter(naive.merchants)), _address_params(goal.address))
        decision = decide(naive, task.customer_id, op)
        if decision.allowed:
            naive = decision.expected_state
    trap = projection(naive) != projection(gold)
    if errors:
        raise ValueError("; ".join(errors))
    return {"certified": True, "gold_operations": [op.model_dump(mode="json") for op in ops],
            "changes_state": changes, "has_trap": trap}


def retail_task(world, customer_id, family, goals, *, ask_total=False, task_id=None, partition="train"):
    draft = RetailTask(task_id=task_id or family, world_hash=world.state_hash, customer_id=customer_id,
                       family=family, goals=goals, ask_total=ask_total, customer_request="pending",
                       partition=partition)
    text = render_retail_request(world, draft)
    return RetailTask(task_id=draft.task_id, world_hash=draft.world_hash, customer_id=customer_id, family=family,
                      goals=goals, ask_total=ask_total, customer_request=text, partition=partition)


# --------------------------------------------------------------------------- generator

FAMILIES = ("address_stated", "address_from_order", "address_profile", "default_from_order", "address_delivered",
            "address_all_pending", "items_simple", "items_gift_card_fallback", "items_stock_fallback",
            "items_two_lines", "items_cheaper_to_gift_card", "items_fallback_cancel", "items_ask_total")

_PRODUCTS = ("sneaker", "jacket", "lamp", "kettle", "backpack", "speaker", "blender", "watch", "camera", "chair")
_COLORS = ("red", "blue", "green", "black", "white", "grey", "yellow", "purple")
_TIERS = ("basic", "plus", "pro")
_CITIES = (("Denver", "CO", "80201"), ("Austin", "TX", "73301"), ("Seattle", "WA", "98101"),
           ("Boston", "MA", "02108"), ("Phoenix", "AZ", "85001"), ("Chicago", "IL", "60601"),
           ("Miami", "FL", "33101"), ("Portland", "OR", "97201"))
_STREETS = ("Maple Drive", "Oak Street", "Pine Road", "Cedar Lane", "Elm Avenue", "Lake View")


def _address(rng, used):
    while True:
        city, state, postal = rng.choice(_CITIES)
        street = "{} {}".format(rng.randrange(10, 990), rng.choice(_STREETS))
        if (street, city) not in used:
            used.add((street, city))
            return AddressFields(street=street, city=city, state=state, postal_code=postal)


def generate_retail_task(seed, family):
    """(world, task, evidence) for one family; every number and name derives from the seed."""
    from .builder import build_world
    from .schema import (Attribute, AddressSpec, BuildBudget, CustomerAddress, IntegerRange, LineVariable,
                         OrderVariable, PaymentVariable, ScenarioSpec, SKUVariable, StockConstraint,
                         WarehouseVariable)
    if family not in FAMILIES:
        raise ValueError("unknown retail family")
    rng = random.Random("retail:{}:{}".format(family, seed))

    def fixed(value):
        return IntegerRange(low=value, high=value)

    names = rng.sample(_PRODUCTS, 4)
    base = {n: rng.randrange(30, 90) * 100 for n in names}
    up = {n: rng.randrange(8, 25) * 100 for n in names}
    down = {n: rng.randrange(5, 15) * 100 for n in names}
    colors = {n: rng.sample(_COLORS, 3) for n in names}
    price = {}
    skus, stock = [], []
    # which variants have no stock: decided by the family
    dead = {"items_stock_fallback": {(0, 1)}, "items_fallback_cancel": {(0, 1), (0, 2), (1, 1), (1, 2)}}.get(family, set())
    for fi, name in enumerate(names):
        for vi, delta in enumerate((0, up[name], -down[name])):
            alias = "{}{}".format("abcd"[fi], vi)
            price[alias] = base[name] + delta
            skus.append(SKUVariable(alias=alias, family=name, price_cents=fixed(price[alias]), attributes=(
                Attribute(name="color", value=colors[name][vi]), Attribute(name="tier", value=_TIERS[vi]))))
            if (fi, vi) in dead:
                stock.append(StockConstraint(sku_alias=alias, warehouse_alias="w", quantity=fixed(0)))
    extra_up = price["a1"] - price["a0"]
    if family == "items_gift_card_fallback":
        balance = rng.randrange(1, max(2, extra_up // 100)) * 100 - 100 if extra_up > 200 else 0
        balance = max(0, min(balance, extra_up - 100))
    elif family == "items_cheaper_to_gift_card":
        balance = rng.randrange(0, 40) * 100
    else:
        balance = rng.randrange(60, 200) * 100
    used = set()
    home, a1, a2, a3 = (_address(rng, used) for _ in range(4))

    def line(alias, sku, qty=1):
        return LineVariable(alias=alias, original_sku=sku, quantity=fixed(qty), paid_cents=fixed(price[sku] * qty))

    def addr(value):
        return AddressSpec(street=value.street, city=value.city, state=value.state, postal_code=value.postal_code)

    spec = ScenarioSpec(
        scenario_id="retail-" + family, provenance_kind="offline_fixture", customer_aliases=("me",),
        skus=tuple(skus),
        warehouses=(WarehouseVariable(alias="w", delivery_days=fixed(2), delivery_fee_cents=fixed(0)),),
        orders=(
            OrderVariable(alias="o1", customer_alias="me", delivered_days_ago=fixed(3), fulfillment_status="unshipped",
                          lines=(line("l1", "a0"), line("l2", "b0")), payment_alias="cc", shipping_address=addr(a1)),
            OrderVariable(alias="o2", customer_alias="me", delivered_days_ago=fixed(4), fulfillment_status="unshipped",
                          lines=(line("l3", "c0"),), payment_alias="pp", shipping_address=addr(a2)),
            OrderVariable(alias="o3", customer_alias="me", delivered_days_ago=fixed(5),
                          lines=(line("l4", "d0"),), payment_alias="gc", shipping_address=addr(a3)),
        ),
        payment_methods=(PaymentVariable(alias="gc", customer_alias="me", kind="gift_card", balance_cents=balance),
                         PaymentVariable(alias="cc", customer_alias="me", kind="credit_card"),
                         PaymentVariable(alias="pp", customer_alias="me", kind="paypal")),
        default_addresses=(CustomerAddress(customer_alias="me", address=addr(home)),),
        default_stock=fixed(3), stock=tuple(stock))
    built = build_world(spec, seed, BuildBudget(max_nodes=8))
    if built.status != "SAT":
        raise ValueError("retail scene is not constructible: " + built.reason)
    world, ids = built.world, dict(built.private_aliases)
    me = ids["me"]

    def goal(**kw):
        kw.setdefault("goal_id", "g{}".format(len(goals) + 1))
        return RetailGoal(**kw)

    goals = []
    target = _address(rng, used)
    ask = False
    if family == "address_stated":
        goals.append(goal(action="set_order_address", order_id=ids["o1"], address=target))
    elif family == "address_from_order":
        goals.append(goal(action="set_order_address", order_id=ids["o2"], address=a3, address_source="other_order",
                          source_order_id=ids["o3"]))
    elif family == "address_profile":
        goals.append(goal(action="set_order_address", order_id=ids["o1"], address=home, address_source="profile"))
    elif family == "default_from_order":
        goals.append(goal(action="set_default_address", address=a2, address_source="other_order",
                          source_order_id=ids["o2"]))
    elif family == "address_delivered":
        goals.append(goal(action="set_order_address", order_id=ids["o3"], address=target))
    elif family == "address_all_pending":
        goals.append(goal(action="set_order_address", order_id=ids["o1"], address=target))
        goals.append(goal(action="set_order_address", order_id=ids["o2"], address=target))
    elif family in ("items_simple", "items_ask_total"):
        goals.append(goal(action="change_items", order_id=ids["o1"],
                          wishes=(ItemWish(line_id=ids["l1"], targets=(ids["a1"],)),),
                          payment_preferences=("original",)))
        ask = family == "items_ask_total"
    elif family == "items_gift_card_fallback":
        goals.append(goal(action="change_items", order_id=ids["o1"],
                          wishes=(ItemWish(line_id=ids["l1"], targets=(ids["a1"],)),),
                          payment_preferences=(ids["gc"], ids["pp"])))
    elif family == "items_stock_fallback":
        goals.append(goal(action="change_items", order_id=ids["o1"],
                          wishes=(ItemWish(line_id=ids["l1"], targets=(ids["a1"], ids["a2"])),),
                          payment_preferences=("original",)))
    elif family == "items_two_lines":
        goals.append(goal(action="change_items", order_id=ids["o1"],
                          wishes=(ItemWish(line_id=ids["l1"], targets=(ids["a1"],)),
                                  ItemWish(line_id=ids["l2"], targets=(ids["b2"],))),
                          payment_preferences=("original",)))
    elif family == "items_cheaper_to_gift_card":
        goals.append(goal(action="change_items", order_id=ids["o1"],
                          wishes=(ItemWish(line_id=ids["l1"], targets=(ids["a2"],)),),
                          payment_preferences=(ids["gc"], "original")))
    else:                                                                    # items_fallback_cancel
        goals.append(goal(action="change_items", order_id=ids["o1"],
                          wishes=(ItemWish(line_id=ids["l1"], targets=(ids["a1"], ids["a2"])),
                                  ItemWish(line_id=ids["l2"], targets=(ids["b1"], ids["b2"]))),
                          payment_preferences=("original",), fallback="cancel_order"))
    task = retail_task(world, me, family, tuple(goals), ask_total=ask, task_id="{}-{}".format(family, seed))
    return world, task, certify_retail_task(world, task)
