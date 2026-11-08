"""Compact, seeded retail scene descriptions shared by task producers."""
import random
import re
from typing import Literal, Optional, Tuple

from pydantic import Field

from usami_v2.contracts import Sealed
from .builder import build_world
from .schema import (AddressSpec, Attribute, CustomerAddress, IntegerRange, LineVariable,
                     OrderVariable, PaymentVariable, ScenarioSpec, SKUVariable,
                     StockConstraint, WarehouseVariable)
from .retail_tasks import AddressFields, ItemWish, RetailGoal, _address, retail_task


class SceneError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(f"{code}: {message}")


class SceneIds(dict):
    """Ref mapping with internal world bindings kept outside the ref namespace."""
    def __init__(self, refs, *, customer_id, new_address, item_skus, payments):
        super().__init__(refs)
        self.customer_id = customer_id
        self.new_address = new_address
        self.item_skus = item_skus
        self.payments = payments


class SceneItem(Sealed):
    ref: str
    product: str
    price_cents: int


class SceneOrder(Sealed):
    ref: str
    status: Literal["unshipped", "shipped", "delivered"]
    pays_with: Literal["gift_card", "credit_card", "paypal"]
    items: Tuple[SceneItem, ...]


class SceneVariant(Sealed):
    ref: str
    of: str
    price_cents: int
    in_stock: bool


class SceneChange(Sealed):
    item: str
    want: Tuple[str, ...]


class SceneRequest(Sealed):
    type: Literal["set_order_address", "set_default_address", "change_items"]
    order: Optional[str] = None
    to: Optional[str] = None
    changes: Tuple[SceneChange, ...] = ()
    pay_with: Tuple[Literal["original", "gift_card", "credit_card", "paypal"], ...] = ("original",)
    otherwise: Literal["nothing", "cancel_order"] = "nothing"


class SceneTask(Sealed):
    requests: Tuple[SceneRequest, ...]
    ask_total: bool = False


class RetailScene(Sealed):
    gift_card_balance_cents: int
    orders: Tuple[SceneOrder, ...]
    variants: Tuple[SceneVariant, ...] = ()


def validate_scene(scene, tasks=()):
    """Validate references before compilation, with stable rejection codes."""
    items = [item for order in scene.orders for item in order.items]
    refs = [order.ref for order in scene.orders] + [item.ref for item in items] + [v.ref for v in scene.variants]
    if len(refs) != len(set(refs)):
        raise SceneError("DUPLICATE_REF", "references must be unique")
    if (not 1 <= len(scene.orders) <= 4 or len(scene.variants) > 8
            or not 0 <= scene.gift_card_balance_cents <= 1_000_000
            or any(not 1 <= len(o.items) <= 3 for o in scene.orders)
            or any(not 100 <= x.price_cents <= 1_000_000 for x in items + list(scene.variants))
            or any(not ref for ref in refs)
            or any(not re.fullmatch(r"[a-z]+(?: [a-z]+){0,2}", item.product) for item in items)):
        raise SceneError("LIMIT", "scene values are outside the supported limits")
    orders = {o.ref: o for o in scene.orders}
    item_by_ref = {item.ref: item for item in items}
    variants = {v.ref: v for v in scene.variants}
    if any(v.of not in item_by_ref for v in scene.variants):
        raise SceneError("UNKNOWN_REF", "variant references an unknown item")
    for task in tasks:
        if not 1 <= len(task.requests) <= 3:
            raise SceneError("LIMIT", "a task needs one to three requests")
        if task.ask_total and not any(request.type == "change_items" for request in task.requests):
            raise SceneError("ASK_TOTAL_INVALID", "a total can only be asked in a task that changes items")
        for request in task.requests:
            if request.type != "set_default_address" and request.order not in orders:
                raise SceneError("UNKNOWN_REF", "unknown order")
            if request.type == "set_default_address" and request.order is not None:
                raise SceneError("ADDRESS_SOURCE_INVALID", "default address has no target order")
            if request.type != "change_items":
                if request.changes or request.pay_with != ("original",) or request.otherwise != "nothing":
                    raise SceneError("EMPTY_REQUEST", "address requests cannot carry item-change preferences")
                source = request.to or ""
                if source.startswith("order:") and source[6:] not in orders:
                    raise SceneError("UNKNOWN_REF", "unknown source order")
                if (source not in ("new", "profile") and not source.startswith("order:")) or (
                    source == "profile" and request.type == "set_default_address") or (
                    source.startswith("order:") and source[6:] == request.order):
                    raise SceneError("ADDRESS_SOURCE_INVALID", "invalid address source")
            else:
                if not request.changes or any(not c.want for c in request.changes):
                    raise SceneError("EMPTY_REQUEST", "item changes and preferences are required")
                if not request.pay_with:
                    raise SceneError("EMPTY_REQUEST", "payment preferences are required")
                if request.to is not None or len({c.item for c in request.changes}) != len(request.changes):
                    raise SceneError("EMPTY_REQUEST", "item changes must be unique and cannot carry an address")
                for change in request.changes:
                    if change.item not in item_by_ref or any(v not in variants for v in change.want):
                        raise SceneError("UNKNOWN_REF", "unknown item or variant")
                    if change.item not in {i.ref for i in orders[request.order].items}:
                        raise SceneError("ITEM_NOT_IN_ORDER", "item belongs to another order")
                    if any(variants[v].of != change.item for v in change.want):
                        raise SceneError("VARIANT_OF_OTHER_ITEM", "variant belongs to another item")
                    if len(set(change.want)) != len(change.want):
                        raise SceneError("EMPTY_REQUEST", "a variant cannot be repeated in one preference chain")
    return scene


def compile_scene(scene, seed):
    validate_scene(scene)
    rng = random.Random(f"retail-scene:{seed}")
    fixed = lambda n: IntegerRange(low=n, high=n)
    used = set()
    home = _address(rng, used)
    addresses = {o.ref: _address(rng, used) for o in scene.orders}
    new_address = _address(rng, used)
    as_spec = lambda a: AddressSpec(**a.model_dump())
    items = [i for o in scene.orders for i in o.items]
    item_map = {i.ref: i for i in items}
    # Internal aliases are independent of external refs, avoiding collisions with
    # the builder's customer, merchant, warehouse and payment aliases.
    item_alias = {i.ref: f"item_{n}" for n, i in enumerate(items)}
    variant_alias = {v.ref: f"variant_{n}" for n, v in enumerate(scene.variants)}
    order_alias = {o.ref: f"order_{n}" for n, o in enumerate(scene.orders)}
    line_alias = {i.ref: f"line_{n}" for n, i in enumerate(items)}
    from .retail_tasks import _COLORS, _TIERS
    choices = [(c, t) for c in _COLORS for t in _TIERS]
    attrs = {}
    for product in sorted({i.product for i in items}):
        group = [i.ref for i in items if i.product == product]
        group += [v.ref for v in scene.variants if item_map[v.of].product == product]
        if len(group) > len(choices):
            raise SceneError("LIMIT", "too many distinct variants of one product")
        attrs.update(zip(group, rng.sample(choices, len(group))))
    def sku(alias, ref, product, price):
        color, tier = attrs[ref]
        return SKUVariable(alias=alias, family=product, price_cents=fixed(price),
                           attributes=(Attribute(name="color", value=color), Attribute(name="tier", value=tier)))
    skus = [sku(item_alias[i.ref], i.ref, i.product, i.price_cents) for i in items]
    skus += [sku(variant_alias[v.ref], v.ref, item_map[v.of].product, v.price_cents) for v in scene.variants]
    orders = []
    for o in scene.orders:
        lines = tuple(LineVariable(alias=line_alias[i.ref], original_sku=item_alias[i.ref],
                                   quantity=fixed(1), paid_cents=fixed(i.price_cents)) for i in o.items)
        orders.append(OrderVariable(alias=order_alias[o.ref], customer_alias="me", delivered_days_ago=fixed(3),
                                    fulfillment_status=o.status, lines=lines, payment_alias="pay_" + o.pays_with,
                                    shipping_address=as_spec(addresses[o.ref])))
    payments = tuple(PaymentVariable(alias="pay_" + kind, customer_alias="me", kind=kind,
                    balance_cents=scene.gift_card_balance_cents if kind == "gift_card" else None)
                     for kind in ("gift_card", "credit_card", "paypal"))
    spec = ScenarioSpec(scenario_id="retail-scene", provenance_kind="offline_fixture", customer_aliases=("me",),
                        skus=tuple(skus), warehouses=(WarehouseVariable(alias="w", delivery_days=fixed(2),
                        delivery_fee_cents=fixed(0)),), orders=tuple(orders), payment_methods=payments,
                        default_addresses=(CustomerAddress(customer_alias="me", address=as_spec(home)),),
                        default_stock=fixed(3), stock=tuple(StockConstraint(sku_alias=variant_alias[v.ref],
                        warehouse_alias="w", quantity=fixed(3 if v.in_stock else 0)) for v in scene.variants))
    result = build_world(spec, seed)
    if result.status != "SAT":
        raise SceneError("WORLD_BUILD_FAILED", result.reason)
    aliases = dict(result.private_aliases)
    ids = SceneIds({ref: aliases[alias] for ref, alias in order_alias.items()},
        customer_id=aliases["me"], new_address=new_address,
        item_skus={ref: aliases[alias] for ref, alias in item_alias.items()},
        payments={kind: aliases["pay_" + kind] for kind in ("gift_card", "credit_card", "paypal")})
    ids.update({ref: aliases[alias] for ref, alias in line_alias.items()})
    ids.update({ref: aliases[alias] for ref, alias in variant_alias.items()})
    return result.world, ids


def scene_task(world, ids, task, *, task_id, family="retail:designer", scene=None):
    if scene is not None:
        validate_scene(scene, (task,))
    goals = []
    for i, request in enumerate(task.requests):
        if request.type == "change_items":
            goals.append(RetailGoal(goal_id=f"g{i+1}", action=request.type, order_id=ids[request.order],
                wishes=tuple(ItemWish(line_id=ids[c.item], targets=tuple(ids[v] for v in c.want))
                             for c in request.changes),
                payment_preferences=tuple(p if p == "original" else ids.payments[p] for p in request.pay_with),
                fallback=request.otherwise))
        else:
            source = request.to
            if source == "new":
                address, address_source, source_id = ids.new_address, "stated", None
            elif source == "profile":
                address, address_source, source_id = world.customers[ids.customer_id].default_address, "profile", None
            else:
                source_id = ids[source[6:]]
                address, address_source = world.orders[source_id].shipping_address, "other_order"
            goals.append(RetailGoal(goal_id=f"g{i+1}", action=request.type,
                order_id=ids[request.order] if request.order else None,
                address=AddressFields(**address.model_dump()), address_source=address_source,
                source_order_id=source_id))
    return retail_task(world, ids.customer_id, family, tuple(goals), ask_total=task.ask_total, task_id=task_id)
