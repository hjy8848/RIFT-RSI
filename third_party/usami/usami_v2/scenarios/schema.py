"""Business-oriented scene constraints; no arbitrary Python or complete World JSON."""
from typing import Literal, Optional, Tuple

from pydantic import Field, StrictInt, model_serializer, model_validator

from usami_v2.contracts import Sealed


class IntegerRange(Sealed):
    low: int = Field(strict=True)
    high: int = Field(strict=True)

    @model_validator(mode="after")
    def ordered(self):
        if self.low > self.high:
            raise ValueError("range lower bound exceeds upper bound")
        return self


class Attribute(Sealed):
    name: str = Field(min_length=1)
    value: str = Field(min_length=1)


class SKUVariable(Sealed):
    alias: str = Field(min_length=1)
    family: str = Field(
        min_length=1,
        description=(
            "Narrow after-sales replacement-compatibility group, not a broad retail category. "
            "Use distinct families for products that must not replace one another (for example, "
            "'laptop' and 'mouse' may both be electronics but belong to different families; "
            "different laptop configurations may share a family)."
        ),
    )
    attributes: Tuple[Attribute, ...] = ()
    price_cents: IntegerRange
    return_days: int = Field(default=30, ge=0, strict=True)
    return_required: bool = False

    @model_validator(mode="after")
    def price_attributes(self):
        if self.price_cents.low < 1:
            raise ValueError("SKU prices must be positive")
        if len({row.name for row in self.attributes}) != len(self.attributes):
            raise ValueError("duplicate SKU attribute")
        return self


class WarehouseVariable(Sealed):
    alias: str = Field(min_length=1)
    delivery_days: IntegerRange
    delivery_fee_cents: IntegerRange

    @model_validator(mode="after")
    def nonnegative(self):
        if self.delivery_days.low < 0 or self.delivery_fee_cents.low < 0:
            raise ValueError("delivery days and fees cannot be negative")
        return self


class LineVariable(Sealed):
    alias: str = Field(min_length=1)
    original_sku: str
    quantity: IntegerRange
    paid_cents: IntegerRange
    issue: Literal["damaged", "wrong_item", "missing_parts"] = "damaged"
    evidence_valid: bool = True

    @model_validator(mode="after")
    def business_amounts(self):
        if self.quantity.low < 1 or self.quantity.high > 1000 or self.paid_cents.low < 0:
            raise ValueError("line quantity/paid ranges are outside the supported business contract")
        return self


class OrderVariable(Sealed):
    alias: str = Field(min_length=1)
    customer_alias: str
    delivered_days_ago: IntegerRange = Field(description=(
        "Days since delivery for a delivered order; for an unshipped or shipped order, "
        "days until its expected delivery."))
    lines: Tuple[LineVariable, ...] = Field(min_length=1, max_length=32)
    fulfillment_status: Literal["unshipped", "shipped", "delivered"] = Field(
        default="delivered",
        description="Only an unshipped order can be cancelled; its lines reserve stock in the first stocked warehouse.")
    ordered_days_ago: Optional[int] = Field(default=None, ge=0, strict=True)

    @model_serializer(mode="wrap")
    def _compatible_dump(self, handler):
        data = handler(self)
        if self.fulfillment_status == "delivered":
            data.pop("fulfillment_status", None)
        if self.ordered_days_ago is None:
            data.pop("ordered_days_ago", None)
        return data

    @model_validator(mode="after")
    def nonnegative_age(self):
        if self.delivered_days_ago.low < 0:
            raise ValueError("this initial-order contract requires already-delivered lines")
        return self


class StockConstraint(Sealed):
    sku_alias: str
    warehouse_alias: str
    quantity: IntegerRange

    @model_validator(mode="after")
    def nonnegative(self):
        if self.quantity.low < 0:
            raise ValueError("stock cannot be negative")
        return self


class LinearTerm(Sealed):
    variable: str
    field: Literal["price_cents", "delivery_days", "delivery_fee_cents",
                   "quantity", "paid_cents", "delivered_days_ago", "stock"]
    warehouse: Optional[str] = None
    coefficient: int = Field(default=1, strict=True)

    @property
    def key(self):
        return (self.variable, self.field, self.warehouse or "")

    @model_validator(mode="after")
    def stock_binding(self):
        if (self.field == "stock") != bool(self.warehouse):
            raise ValueError("stock terms require a warehouse; other terms cannot name one")
        if self.coefficient == 0:
            raise ValueError("zero coefficient is redundant")
        return self


class LinearExpression(Sealed):
    terms: Tuple[LinearTerm, ...] = ()
    constant: int = Field(default=0, strict=True)


class RelationConstraint(Sealed):
    constraint_id: str
    left: LinearExpression
    operator: Literal["eq", "ne", "lt", "le", "gt", "ge"]
    right: LinearExpression


class HistoryStep(Sealed):
    operation: Literal['return_delivered_order_items', 'cancel_order', 'exchange_delivered_order_items',
                       'reship_order_items', 'request_item_return', 'return_received']
    line_alias: Optional[str] = None
    order_alias: Optional[str] = None
    quantity: int = Field(default=1, ge=1, strict=True)
    target_sku_alias: Optional[str] = None

    @model_validator(mode='after')
    def valid_target(self):
        if self.operation == 'cancel_order':
            if not self.order_alias or self.line_alias:
                raise ValueError('cancellation history requires an order alias')
        elif not self.line_alias or self.order_alias:
            raise ValueError('history operation requires a line alias')
        if (self.operation == 'exchange_delivered_order_items') != bool(self.target_sku_alias):
            raise ValueError('exchange history alone requires a target SKU alias')
        return self


class ScenarioSpec(Sealed):
    protocol: Literal["commerce-scenario-v1"] = "commerce-scenario-v1"
    scenario_id: str
    merchant_alias: str = "merchant"
    customer_aliases: Tuple[str, ...] = Field(min_length=1, max_length=16)
    skus: Tuple[SKUVariable, ...] = Field(min_length=1, max_length=64)
    warehouses: Tuple[WarehouseVariable, ...] = Field(min_length=1, max_length=8)
    orders: Tuple[OrderVariable, ...] = Field(min_length=1, max_length=64)
    stock: Tuple[StockConstraint, ...] = ()
    default_stock: IntegerRange = IntegerRange(low=0, high=5)
    refund_approval_limit_cents: int = Field(default=50000, ge=0, strict=True)
    constraints: Tuple[RelationConstraint, ...] = ()
    provenance_kind: Literal["synthetic_reference", "designer_exploration", "offline_fixture"]
    distribution_hash: Optional[str] = None
    history: Tuple[HistoryStep, ...] = ()

    @model_serializer(mode='wrap')
    def _compatible_dump(self, handler):
        data = handler(self)
        if not self.history:
            data.pop('history', None)
        return data

    @model_validator(mode="after")
    def references_and_capabilities(self):
        from .solver import variable_ranges
        aliases = list(self.customer_aliases) + [self.merchant_alias]
        aliases += [row.alias for row in self.skus] + [row.alias for row in self.warehouses]
        aliases += [row.alias for row in self.orders]
        aliases += [line.alias for order in self.orders for line in order.lines]
        if len(aliases) != len(set(aliases)):
            raise ValueError("entity aliases must be globally unique")
        sku_ids, warehouse_ids = {row.alias for row in self.skus}, {row.alias for row in self.warehouses}
        if self.default_stock.low < 0:
            raise ValueError("default stock cannot be negative")
        if any(order.customer_alias not in self.customer_aliases
               or any(line.original_sku not in sku_ids for line in order.lines) for order in self.orders):
            raise ValueError("order references unknown Customer or SKU")
        if any(row.sku_alias not in sku_ids or row.warehouse_alias not in warehouse_ids for row in self.stock):
            raise ValueError("stock references unknown SKU or warehouse")
        if len({(row.sku_alias, row.warehouse_alias) for row in self.stock}) != len(self.stock):
            raise ValueError("duplicate stock constraint")
        ranges = variable_ranges(self)
        for relation in self.constraints:
            for term in relation.left.terms + relation.right.terms:
                if term.key not in ranges:
                    raise ValueError("constraint references an unavailable business variable/field")
        if len({row.constraint_id for row in self.constraints}) != len(self.constraints):
            raise ValueError("duplicate scene constraint IDs")
        line_ids = {line.alias for order in self.orders for line in order.lines}
        order_ids = {order.alias for order in self.orders}
        for step in self.history:
            if ((step.line_alias and step.line_alias not in line_ids)
                    or (step.order_alias and step.order_alias not in order_ids)
                    or (step.target_sku_alias and step.target_sku_alias not in sku_ids)):
                raise ValueError('history references an unknown scene alias')
        if self.provenance_kind != "offline_fixture" and not self.distribution_hash:
            raise ValueError("reference/exploration scenes require distribution provenance")
        return self


class BuildBudget(Sealed):
    max_nodes: int = Field(default=256, ge=1, le=100000, strict=True)
    strategy: Literal["sample", "propagate"] = "sample"

    @model_serializer(mode="wrap")
    def _compatible_dump(self, handler):
        data = handler(self)
        if self.strategy == "sample":
            data.pop("strategy", None)
        return data
