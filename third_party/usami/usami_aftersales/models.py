"""Order-line disputes, verifiable evidence, replacement stock and refund ledgers."""
from datetime import datetime
from typing import Dict,Literal,Optional
from pydantic import Field
from usami_harness.models import Model,OperationBase,Empty
from usami_dialogue.models import DialogueWorldBase,CustomerPrivateState

VERSION='aftersales-0.1.0'


class User(Model):
    id:str
    name:str


class Merchant(Model):
    id:str
    name:str
    refund_approval_limit_cents:int=Field(strict=True,ge=0)


class Product(Model):
    id:str
    merchant_id:str
    name:str
    stock:int=Field(strict=True,ge=0)
    opening_stock:int=Field(strict=True,ge=0)
    returnable:bool=True
    return_days:int=Field(strict=True,ge=0)
    return_required:bool=True
    replacement_days:int=Field(strict=True,ge=1)


class Order(Model):
    id:str
    merchant_id:str
    customer_id:str
    status:Literal['processing','shipped','delivered']
    delivered_at:Optional[datetime]=None
    payment_channel:str
    paid_cents:int=Field(strict=True,ge=0)


class Line(Model):
    id:str
    order_id:str
    product_id:str
    quantity:int=Field(strict=True,gt=0)
    unit_price_cents:int=Field(strict=True,gt=0)
    status:Literal['unresolved','return_pending','refunded','exchanged','reshipped']='unresolved'


class Evidence(Model):
    id:str
    order_id:str
    line_id:str
    holder_id:str
    issue:Literal['damaged','wrong_item','missing_parts']
    simulated_valid:bool


class Case(Model):
    id:str
    order_id:str
    line_id:str
    issue:Literal['damaged','wrong_item','missing_parts']
    status:Literal['open','evidence_verified','return_pending','resolved']='open'
    evidence_id:Optional[str]=None
    evidence_notice:dict=Field(default_factory=dict)
    review_requested:bool=False
    agreement_quote:Optional[str]=None
    agreement_revision:Optional[int]=None


class Quote(Model):
    id:str
    order_id:str
    case_id:str
    option:Literal['refund','exchange','reship']
    amount_cents:int=Field(strict=True,ge=0)
    delivery_at:Optional[datetime]=None
    requires_return:bool
    revision:int=Field(strict=True,ge=0)
    valid_until:datetime
    status:Literal['offered','committed']='offered'
    quote_hash:str


class Return(Model):
    id:str
    case_id:str
    pickup_slot:str
    status:Literal['scheduled','received']='scheduled'
    receipt_notice:dict=Field(default_factory=dict)


class Refund(Model):
    id:str
    order_id:str
    line_id:str
    case_id:str
    amount_cents:int=Field(strict=True,gt=0)
    channel:str
    notice:dict


class Replacement(Model):
    id:str
    order_id:str
    line_id:str
    product_id:str
    quantity:int=Field(strict=True,gt=0)
    kind:Literal['exchange','reship']
    delivery_at:datetime


class ExternalEvent(Model):
    id:str
    kind:Literal['manager_review','return_received']
    line_id:str
    at:datetime
    outcome:Literal['approved','denied','received','missing']


class World(DialogueWorldBase):
    domain:Literal['aftersales']='aftersales'
    version:Literal['aftersales-0.1.0']=VERSION
    users:Dict[str,User]
    merchants:Dict[str,Merchant]
    products:Dict[str,Product]
    orders:Dict[str,Order]
    lines:Dict[str,Line]
    evidence:Dict[str,Evidence]
    profiles:Dict[str,CustomerPrivateState]
    cases:Dict[str,Case]=Field(default_factory=dict)
    quotes:Dict[str,Quote]=Field(default_factory=dict)
    returns:Dict[str,Return]=Field(default_factory=dict)
    refunds:Dict[str,Refund]=Field(default_factory=dict)
    replacements:Dict[str,Replacement]=Field(default_factory=dict)
    approvals:Dict[str,dict]=Field(default_factory=dict)
    external_events:Dict[str,ExternalEvent]=Field(default_factory=dict)
    applied_events:list[str]=Field(default_factory=list)


class OpenCase(Model):
    line_id:str
    issue:Literal['damaged','wrong_item','missing_parts']


class CaseInput(Model):
    case_id:str


class Verify(CaseInput):
    evidence_id:str


class QuoteInput(CaseInput):
    option:Literal['refund','exchange','reship']


class Commit(Model):
    quote_id:str
    destination_channel:str


class Pickup(CaseInput):
    pickup_slot:str=Field(min_length=1)


PARAMETERS={'get_orders':Empty,'get_order':Empty,'get_case':CaseInput,'get_options':CaseInput,
 'open_case':OpenCase,'verify_evidence':Verify,'request_manager_review':CaseInput,
 'quote_resolution':QuoteInput,'commit_resolution':Commit,'schedule_return':Pickup,'release_refund':CaseInput}
QUERIES=frozenset(n for n in PARAMETERS if n.startswith('get_'))
TOOLS=tuple(PARAMETERS)+('get_customer_context',)


class Operation(OperationBase):
    parameters=PARAMETERS
    queries=QUERIES
