"""Native SKU, inventory, quantity-level settlement and transaction contracts."""
from datetime import datetime
from typing import Dict, Literal, Optional, Tuple

from pydantic import Field, StrictInt, model_validator

from usami_harness.models import Empty, Model, OperationBase, WorldBase, digest
from usami_v2.contracts import aware
from . import VERSION


class Customer(Model):
    id: str
    name: str


class Merchant(Model):
    id: str
    refund_approval_limit_cents: int = Field(ge=0, strict=True)


class SKU(Model):
    id: str
    merchant_id: str
    product_family: str
    attributes: Dict[str, str]
    price_cents: int = Field(gt=0, strict=True)
    revision: int = Field(default=0, ge=0, strict=True)
    return_days: int = Field(default=30, ge=0, strict=True)
    return_required: bool = False


class Warehouse(Model):
    id: str
    merchant_id: str


class Inventory(Model):
    id: str
    sku_id: str
    warehouse_id: str
    opening_quantity: int = Field(ge=0, strict=True)
    adjustments: int = Field(default=0, strict=True)
    available: int = Field(ge=0, strict=True)
    reserved: int = Field(default=0, ge=0, strict=True)
    shipped: int = Field(default=0, ge=0, strict=True)
    revision: int = Field(default=0, ge=0, strict=True)


class DeliveryRoute(Model):
    id: str
    warehouse_id: str
    days: int = Field(ge=0, strict=True)
    fee_cents: int = Field(default=0, ge=0, strict=True)
    revision: int = Field(default=0, ge=0, strict=True)


class Order(Model):
    id: str
    merchant_id: str
    customer_id: str
    paid_cents: int = Field(ge=0, strict=True)
    payment_channel: str
    delivered_at: datetime

    @model_validator(mode="after")
    def timezone(self):
        aware(self.delivered_at)
        return self


class OrderLine(Model):
    id: str
    order_id: str
    sku_id: str
    quantity: int = Field(gt=0, strict=True)
    allocated_paid_cents: int = Field(ge=0, strict=True)


class Evidence(Model):
    id: str
    customer_id: str
    line_id: str
    issue: Literal["damaged", "wrong_item", "missing_parts"]
    valid: bool


class RemedyCase(Model):
    id: str
    line_id: str
    unit_indices: Tuple[StrictInt, ...] = Field(min_length=1)
    issue: Literal["damaged", "wrong_item", "missing_parts"]
    status: Literal["open", "verified", "awaiting_return", "ready", "resolved", "cancelled"] = "open"
    evidence_id: Optional[str] = None
    review_requested: bool = False
    review_approved: bool = False
    committed_quote_id: Optional[str] = None


class Quote(Model):
    id: str
    case_id: str
    customer_id: str
    remedy: Literal["refund", "exchange", "reship"]
    quantity: int = Field(gt=0, strict=True)
    target_sku_id: Optional[str] = None
    inventory_id: Optional[str] = None
    route_id: Optional[str] = None
    refund_cents: int = Field(ge=0, strict=True)
    extra_charge_cents: int = Field(ge=0, strict=True)
    payment_channel: str
    promised_arrival: Optional[datetime] = None
    return_required: bool
    pickup_slot: Optional[str] = None
    revision: int = Field(ge=0, strict=True)
    inventory_revision: Optional[int] = None
    sku_revision: Optional[int] = None
    route_revision: Optional[int] = None
    valid_until: datetime
    status: Literal["prepared", "committed", "cancelled"] = "prepared"
    terms_hash: str = ""

    @model_validator(mode="after")
    def sealed_terms(self):
        aware(self.valid_until)
        if self.promised_arrival is not None:
            aware(self.promised_arrival)
        expected = digest(self.model_dump(mode="json", exclude={"terms_hash", "status"}))
        if self.terms_hash and self.terms_hash != expected:
            raise ValueError("quote terms hash differs")
        object.__setattr__(self, "terms_hash", expected)
        return self


class Delegation(Model):
    id: str
    customer_id: str
    line_ids: Tuple[str, ...]
    allowed_preparations: Tuple[str, ...]
    expires_at: datetime
    withdrawn: bool = False
    source_customer_event_hash: str = Field(min_length=1)

    @model_validator(mode="after")
    def timezone(self):
        aware(self.expires_at)
        return self


class Consent(Model):
    id: str
    customer_id: str
    quote_id: str
    terms_hash: str
    operation: Literal["commit_accepted_quote", "release_accepted_refund"]
    presented_offer_event_hash: str = Field(min_length=1)
    source_customer_event_hash: str = Field(min_length=1)
    expires_at: datetime
    consumed_by: Optional[str] = None
    withdrawn: bool = False

    @model_validator(mode="after")
    def timezone(self):
        aware(self.expires_at)
        return self


class Return(Model):
    id: str
    case_id: str
    pickup_slot: str
    status: Literal["scheduled", "received"] = "scheduled"
    received_event_id: Optional[str] = None


class LedgerEntry(Model):
    id: str
    line_id: str
    quote_id: str
    kind: Literal["refund", "extra_charge"]
    amount_cents: int = Field(gt=0, strict=True)
    channel: str
    debit_account: str
    credit_account: str


class Settlement(Model):
    id: str
    case_id: str
    line_id: str
    quote_id: str
    unit_indices: Tuple[StrictInt, ...] = Field(min_length=1)
    remedy: Literal["refund", "exchange", "reship"]


class Fulfillment(Model):
    id: str
    quote_id: str
    sku_id: str
    quantity: int = Field(gt=0, strict=True)
    inventory_id: str
    promised_arrival: datetime
    actual_arrival: Optional[datetime] = None

    @model_validator(mode="after")
    def timezone(self):
        aware(self.promised_arrival)
        if self.actual_arrival is not None:
            aware(self.actual_arrival)
        return self


class ExternalEvent(Model):
    id: str
    kind: Literal["review_result", "return_received", "stock_adjustment", "route_change"]
    target_id: str
    at: datetime
    integer_value: int = Field(default=0, strict=True)
    approved: bool = False

    @model_validator(mode="after")
    def timezone(self):
        aware(self.at)
        return self


class World(WorldBase):
    domain: Literal["commerce"] = "commerce"
    version: Literal["commerce-2.0.0"] = VERSION
    customers: Dict[str, Customer]
    merchants: Dict[str, Merchant]
    skus: Dict[str, SKU]
    warehouses: Dict[str, Warehouse]
    inventory: Dict[str, Inventory]
    routes: Dict[str, DeliveryRoute]
    orders: Dict[str, Order]
    lines: Dict[str, OrderLine]
    evidence: Dict[str, Evidence] = Field(default_factory=dict)
    cases: Dict[str, RemedyCase] = Field(default_factory=dict)
    quotes: Dict[str, Quote] = Field(default_factory=dict)
    delegations: Dict[str, Delegation] = Field(default_factory=dict)
    consents: Dict[str, Consent] = Field(default_factory=dict)
    returns: Dict[str, Return] = Field(default_factory=dict)
    ledger: Dict[str, LedgerEntry] = Field(default_factory=dict)
    settlements: Dict[str, Settlement] = Field(default_factory=dict)
    fulfillments: Dict[str, Fulfillment] = Field(default_factory=dict)
    external_events: Dict[str, ExternalEvent] = Field(default_factory=dict)
    applied_events: Tuple[str, ...] = ()


class OpenCaseInput(Model):
    quantity: int = Field(gt=0, strict=True)
    issue: Literal["damaged", "wrong_item", "missing_parts"]


class EvidenceInput(Model):
    evidence_id: str


class CatalogInput(Model):
    product_family: Optional[str] = None
    attributes: Dict[str, str] = Field(default_factory=dict)


class DeliveryInput(Model):
    warehouse_id: str
    quantity: int = Field(default=1, gt=0, strict=True)


class PrepareQuoteInput(Model):
    remedy: Literal["refund", "exchange", "reship"]
    target_sku_id: Optional[str] = None
    warehouse_id: Optional[str] = None
    pickup_slot: Optional[str] = None


PARAMETERS = {
    "list_customer_orders": Empty, "get_order_details": Empty,
    "search_catalog": CatalogInput, "get_sku_details": Empty,
    "get_inventory": Empty, "estimate_delivery": DeliveryInput,
    "get_refundable_balance": Empty, "get_case_status": Empty,
    "open_remedy_case": OpenCaseInput, "verify_submitted_evidence": EvidenceInput,
    "request_policy_review": Empty, "prepare_remedy_quote": PrepareQuoteInput,
    "commit_accepted_quote": Empty, "schedule_authorized_return": Empty,
    "release_accepted_refund": Empty, "ship_accepted_replacement": Empty,
    "cancel_pending_commitment": Empty,
}
QUERIES = frozenset(name for name in PARAMETERS if name.startswith(("get_", "list_", "search_", "estimate_")))
PREPARATIONS = frozenset({"open_remedy_case", "verify_submitted_evidence",
                         "request_policy_review", "prepare_remedy_quote", "cancel_pending_commitment"})


class Operation(OperationBase):
    parameters = PARAMETERS
    queries = QUERIES


def unit_credit(line: OrderLine, indices: Tuple[int, ...]) -> int:
    """Freeze coupon/proration cents per purchased unit; partial totals add exactly."""
    if len(set(indices)) != len(indices) or any(type(i) is not int or i < 0 or i >= line.quantity for i in indices):
        raise ValueError("invalid or duplicate purchased-unit indices")
    base, remainder = divmod(line.allocated_paid_cents, line.quantity)
    return sum(base + (index < remainder) for index in indices)


def case_id(line_id, indices):
    return "case_" + digest([line_id, list(indices)])[:20]


def quote_id(case_key, revision):
    return "quote_" + digest([case_key, revision])[:20]
