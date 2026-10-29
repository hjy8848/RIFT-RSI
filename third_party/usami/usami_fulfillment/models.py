from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import Field, model_validator
from usami_harness.models import Model, WorldBase, OperationBase, Empty

VERSION = 'fulfillment-0.1.0'


class User(Model):
    id: str
    name: str


class Organization(Model):
    id: str
    name: str
    owner_id: str


class Warehouse(Model):
    id: str
    org_id: str
    name: str
    receiving_enabled: bool = True
    capacity_volume: int = Field(default=1000, strict=True, ge=0, le=1000000)
    allowed_categories: List[str] = Field(default_factory=lambda: ['general', 'fragile'])
    shipping_cents: int = Field(default=500, strict=True, ge=0)


class Access(Model):
    id: str
    warehouse_id: str
    user_id: str
    role: Literal['manager', 'picker', 'receiver', 'finance', 'viewer']


class SKU(Model):
    id: str
    org_id: str
    name: str
    category: Literal['general', 'fragile', 'restricted'] = 'general'
    unit_price_cents: int = Field(strict=True, gt=0)
    unit_volume: int = Field(default=1, strict=True, gt=0)
    returnable: bool = True
    return_window_days: int = Field(default=30, strict=True, ge=0)
    simulated_return_condition: Literal['resalable', 'merchant_fault', 'customer_damage'] = 'resalable'


class Lot(Model):
    id: str
    warehouse_id: str
    sku_id: str
    on_hand: int = Field(strict=True, ge=0)
    reserved: int = Field(default=0, strict=True, ge=0)
    picked: int = Field(default=0, strict=True, ge=0)
    quality: Literal['available', 'quarantine'] = 'available'


class ReceiptNotice(Model):
    id: str
    warehouse_id: str
    sku_id: str
    quantity: int = Field(strict=True, gt=0)
    quality: Literal['available', 'quarantine'] = 'available'
    attestation: Dict[str, Any]


class Receipt(Model):
    id: str
    notice_id: str
    warehouse_id: str
    lot_id: str
    quantity: int = Field(strict=True, gt=0)
    key: str


class OrderLine(Model):
    id: str
    sku_id: str
    quantity: int = Field(strict=True, gt=0)
    unit_price_cents: int = Field(strict=True, gt=0)
    canceled: int = Field(default=0, strict=True, ge=0)


class Order(Model):
    id: str
    warehouse_id: str
    customer_id: str
    status: Literal['draft', 'confirmed', 'partially_shipped', 'shipped', 'partially_delivered', 'delivered', 'canceled'] = 'draft'
    lines: List[OrderLine] = Field(min_length=1)
    shipping_cents: int = Field(strict=True, ge=0)


class Reservation(Model):
    id: str
    warehouse_id: str
    order_id: str
    line_id: str
    lot_id: str
    quantity: int = Field(strict=True, gt=0)
    released_quantity: int = Field(default=0, strict=True, ge=0)
    status: Literal['reserved', 'picked', 'packed', 'shipped', 'released'] = 'reserved'


class Handling(Model):
    id: str
    reservation_id: str
    warehouse_id: str
    kind: Literal['pick', 'pack']
    quantity: int = Field(strict=True, gt=0)


class Shipment(Model):
    id: str
    warehouse_id: str
    order_id: str
    reservation_ids: List[str] = Field(min_length=1)
    status: Literal['label_created', 'shipped', 'delivered', 'canceled'] = 'label_created'
    key: str
    carrier_notice: Dict[str, Any] = Field(default_factory=dict)
    delivered_at: Optional[datetime] = None


class Transfer(Model):
    id: str
    source_warehouse_id: str
    target_warehouse_id: str
    source_lot_id: str
    target_lot_id: str
    quantity: int = Field(strict=True, gt=0)
    status: Literal['in_transit', 'received'] = 'in_transit'
    key: str
    receiving_notice: Dict[str, Any]


class Return(Model):
    id: str
    warehouse_id: str
    shipment_id: str
    reservation_id: str
    quantity: int = Field(strict=True, gt=0)
    reason: Literal['customer_remorse', 'merchant_damage']
    status: Literal['requested', 'received', 'inspected', 'restocked', 'canceled'] = 'requested'
    accepted: int = Field(default=0, strict=True, ge=0)
    scrapped: int = Field(default=0, strict=True, ge=0)
    refundable_units: int = Field(default=0, strict=True, ge=0)
    receipt_notice: Dict[str, Any]
    inspection_notice: Dict[str, Any] = Field(default_factory=dict)


class Payment(Model):
    id: str
    warehouse_id: str
    order_id: str
    kind: Literal['charge', 'refund']
    amount_cents: int = Field(strict=True, gt=0)
    key: str
    token: Literal['card_approved', 'card_declined'] = 'card_approved'
    outcome: Literal['succeeded', 'failed']
    gateway_notice: Dict[str, Any]


class World(WorldBase):
    domain: Literal['fulfillment'] = 'fulfillment'
    version: Literal['fulfillment-0.1.0'] = VERSION
    users: Dict[str, User]
    organizations: Dict[str, Organization]
    warehouses: Dict[str, Warehouse]
    access: Dict[str, Access]
    skus: Dict[str, SKU]
    lots: Dict[str, Lot]
    receipt_notices: Dict[str, ReceiptNotice]
    receipts: Dict[str, Receipt] = Field(default_factory=dict)
    lot_inspections: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    orders: Dict[str, Order]
    reservations: Dict[str, Reservation] = Field(default_factory=dict)
    handling: Dict[str, Handling] = Field(default_factory=dict)
    shipments: Dict[str, Shipment] = Field(default_factory=dict)
    transfers: Dict[str, Transfer] = Field(default_factory=dict)
    returns: Dict[str, Return] = Field(default_factory=dict)
    payments: Dict[str, Payment] = Field(default_factory=dict)


Quantity = lambda: Field(strict=True, gt=0, le=100000)
Key = lambda: Field(min_length=1, max_length=80)


class Receive(Model):
    lot_id: str = Field(min_length=1)
    key: str = Key()


class CreateLine(Model):
    id: str = Field(min_length=1)
    sku_id: str
    quantity: int = Quantity()


class CreateOrder(Model):
    order_id: str = Field(min_length=1)
    customer_id: str
    lines: List[CreateLine] = Field(min_length=1, max_length=10)

    @model_validator(mode='after')
    def unique_lines(self):
        if len({line.id for line in self.lines}) != len(self.lines):
            raise ValueError('duplicate order line IDs')
        return self


class ModifyLine(Model):
    line_id: str
    quantity: int = Quantity()


class Reserve(Model):
    reservation_id: str = Field(min_length=1)
    line_id: str
    lot_id: str
    quantity: int = Quantity()


class TransferInput(Model):
    transfer_id: str = Field(min_length=1)
    target_warehouse_id: str
    target_lot_id: str = Field(min_length=1)
    quantity: int = Quantity()
    key: str = Key()


class ShipmentInput(Model):
    shipment_id: str = Field(min_length=1)
    reservation_ids: List[str] = Field(min_length=1, max_length=20)
    key: str = Key()

    @model_validator(mode='after')
    def unique_allocations(self):
        if len(set(self.reservation_ids)) != len(self.reservation_ids):
            raise ValueError('duplicate shipment allocations')
        return self


class ReturnInput(Model):
    return_id: str = Field(min_length=1)
    reservation_id: str
    quantity: int = Quantity()
    reason: Literal['customer_remorse', 'merchant_damage']


class Charge(Model):
    amount_cents: int = Field(strict=True, gt=0, le=100000000)
    key: str = Key()
    token: Literal['card_approved', 'card_declined']


class Refund(Model):
    amount_cents: int = Field(strict=True, gt=0, le=100000000)
    key: str = Key()


class Capacity(Model):
    capacity_volume: int = Field(strict=True, ge=0, le=1000000)


class Receiving(Model):
    enabled: bool = Field(strict=True)


PARAMETERS = {
    'receive_inventory': Receive, 'inspect_lot': Empty, 'dispatch_transfer': TransferInput,
    'receive_transfer': Empty, 'create_order': CreateOrder, 'modify_order_line': ModifyLine,
    'capture_payment': Charge, 'confirm_order': Empty, 'reserve_inventory': Reserve,
    'release_reservation': Empty, 'pick_reservation': Empty, 'unpick_reservation': Empty,
    'pack_reservation': Empty, 'unpack_reservation': Empty, 'create_shipment': ShipmentInput,
    'cancel_shipment': Empty, 'dispatch_shipment': Empty, 'confirm_delivery': Empty,
    'cancel_order': Empty, 'cancel_order_line': ModifyLine, 'request_return': ReturnInput,
    'cancel_return': Empty, 'receive_return': Empty, 'inspect_return': Empty, 'restock_return': Empty,
    'refund_order': Refund, 'change_warehouse_capacity': Capacity, 'set_warehouse_receiving': Receiving,
    'get_warehouse': Empty, 'get_inventory': Empty, 'get_catalog': Empty,
    'get_order': Empty, 'get_shipment': Empty, 'get_returns': Empty, 'get_payment_ledger': Empty,
}
QUERIES = frozenset(name for name in PARAMETERS if name.startswith('get_'))
WRITES = tuple(name for name in PARAMETERS if name not in QUERIES)
TOOLS = tuple(PARAMETERS) + ('get_customer_context',)


class Operation(OperationBase):
    parameters = PARAMETERS
    queries = QUERIES
