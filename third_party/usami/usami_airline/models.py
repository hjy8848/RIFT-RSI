"""Airline inventory, ticket value, fare contracts and passenger authority."""
from datetime import datetime
from typing import Literal,Optional
from pydantic import Field
from usami_harness.models import Model,OperationBase,Empty
from usami_dialogue.models import DialogueWorldBase,CustomerPrivateState

VERSION='airline-0.1.0'
class User(Model):
    id:str
    name:str
    document_token:str
    simulated_identity_valid:bool=True
    payment_limit_cents:int=Field(ge=0,strict=True)
class Carrier(Model):
    id:str
    name:str
class Airport(Model):
    id:str
    min_connection_minutes:int=Field(ge=0,strict=True)
class Flight(Model):
    id:str
    carrier_id:str
    origin:str
    destination:str
    departure:datetime
    arrival:datetime
    status:Literal['scheduled','delayed','cancelled']='scheduled'
    seats:dict[str,int]
    fares:dict[str,int]
    max_baggage_kg:int=Field(ge=0,strict=True)
class FareRule(Model):
    id:str
    carrier_id:str
    changeable:bool
    refundable:bool
    change_fee_cents:int=Field(ge=0,strict=True)
    refund_fee_cents:int=Field(ge=0,strict=True)
    notice_hours:int=Field(ge=0,strict=True)
    baggage_included_kg:int=Field(ge=0,strict=True)
    baggage_price_per_kg_cents:int=Field(ge=0,strict=True)
class Booking(Model):
    id:str
    carrier_id:str
    passenger_id:str
    flights:list[str]=Field(min_length=1,max_length=3)
    cabin:Literal['economy','business']
    fare_rule_id:str
    status:Literal['active','cancelled']='active'
    initial_paid_cents:int=Field(ge=0,strict=True)
    fare_value_cents:int=Field(ge=0,strict=True)
    retained_fees_cents:int=Field(default=0,ge=0,strict=True)
    baggage_paid_cents:int=Field(default=0,ge=0,strict=True)
    extra_baggage_kg:int=Field(default=0,ge=0,strict=True)
    revision:int=Field(default=0,ge=0,strict=True)
class Quote(Model):
    id:str
    booking_id:str
    action:Literal['rebook','refund','baggage']
    flights:list[str]
    cabin:Literal['economy','business']
    charge_cents:int=Field(ge=0,strict=True)
    refund_cents:int=Field(ge=0,strict=True)
    fee_cents:int=Field(ge=0,strict=True)
    fare_value_cents:int=Field(ge=0,strict=True)
    baggage_kg:int=Field(ge=0,strict=True)
    revision:int=Field(ge=0,strict=True)
    booking_revision:int=Field(ge=0,strict=True)
    expires_at:datetime
    status:Literal['offered','committed']='offered'
    quote_hash:str
class LedgerEntry(Model):
    id:str
    booking_id:str
    kind:Literal['charge','refund']
    amount_cents:int=Field(gt=0,strict=True)
    quote_id:str
    notice:dict
class Verification(Model):
    id:str
    booking_id:str
    verified:bool
    notice:dict
class FlightEvent(Model):
    id:str
    flight_id:str
    at:datetime
    status:Literal['delayed','cancelled']
    delay_minutes:int=Field(default=0,ge=0,strict=True)
class World(DialogueWorldBase):
    domain:Literal['airline']='airline'
    version:Literal['airline-0.1.0']=VERSION
    users:dict[str,User]
    carriers:dict[str,Carrier]
    airports:dict[str,Airport]
    flights:dict[str,Flight]
    fare_rules:dict[str,FareRule]
    bookings:dict[str,Booking]
    profiles:dict[str,CustomerPrivateState]
    quotes:dict[str,Quote]=Field(default_factory=dict)
    ledger:dict[str,LedgerEntry]=Field(default_factory=dict)
    verifications:dict[str,Verification]=Field(default_factory=dict)
    external_events:dict[str,FlightEvent]=Field(default_factory=dict)
    applied_events:list[str]=Field(default_factory=list)
class Verify(Model):
    document_token:str
class FlightSearch(Model):
    origin:str
    destination:str
class Rebook(Model):
    flights:list[str]=Field(min_length=1,max_length=3)
    cabin:Literal['economy','business']
class Baggage(Model):
    extra_kg:int=Field(gt=0,strict=True)
class Commit(Model):
    quote_id:str
PARAMETERS={'get_bookings':Empty,'get_booking':Empty,'search_flights':FlightSearch,'get_quote':Commit,
 'verify_passenger':Verify,'quote_change':Rebook,'quote_refund':Empty,'quote_baggage':Baggage,'commit_quote':Commit}
QUERIES=frozenset(n for n in PARAMETERS if n.startswith('get_') or n=='search_flights')
TOOLS=tuple(PARAMETERS)+('get_customer_context',)
class Operation(OperationBase):
    parameters=PARAMETERS
    queries=QUERIES
