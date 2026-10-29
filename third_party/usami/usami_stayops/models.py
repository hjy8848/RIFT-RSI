from datetime import date,datetime
from typing import Any,Dict,List,Literal,Optional,Annotated
from pydantic import Field,BeforeValidator,model_validator
from usami_harness.models import Model,WorldBase,OperationBase,Empty

VERSION='stayops-0.1.0'


def exact_date(value):
    if isinstance(value,date) and not isinstance(value,datetime):return value
    if isinstance(value,str):
        result=date.fromisoformat(value)
        if result.isoformat()==value:return result
    raise ValueError('date must be an exact YYYY-MM-DD calendar date')


Date=Annotated[date,BeforeValidator(exact_date)]


class User(Model):
    id:str
    name:str


class Property(Model):
    id:str
    name:str
    owner_id:str
    timezone:Literal['UTC']='UTC'
    max_stay_nights:int=Field(default=14,strict=True,ge=1,le=30)


class Access(Model):
    id:str
    property_id:str
    user_id:str
    role:Literal['manager','frontdesk','housekeeping','finance','viewer']


class Guest(Model):
    id:str
    user_id:str
    name:str
    simulated_identity_result:Literal['verified','declined']='verified'


class RoomType(Model):
    id:str
    property_id:str
    name:str
    nightly_cents:int=Field(strict=True,gt=0)
    max_occupants:int=Field(default=2,strict=True,gt=0,le=6)


class Room(Model):
    id:str
    property_id:str
    room_type_id:str
    number:str
    housekeeping:Literal['clean','dirty']='clean'
    maintenance:bool=Field(default=False,strict=True)
    current_stay_id:Optional[str]=None


class CalendarDay(Model):
    id:str
    property_id:str
    room_type_id:str
    day:Date
    sale_limit:int=Field(strict=True,ge=0,le=20)
    closed:bool=Field(default=False,strict=True)


class RatePlan(Model):
    id:str
    property_id:str
    name:str
    room_type_ids:List[str]=Field(min_length=1)
    weekend_surcharge_cents:int=Field(default=1000,strict=True,ge=0)
    deposit_bps:int=Field(default=2000,strict=True,ge=0,le=10000)
    free_cancel_hours:int=Field(default=48,strict=True,ge=0,le=168)
    late_fee:Literal['first_night','full_stay']='first_night'


class Reservation(Model):
    id:str
    property_id:str
    guest_id:str
    room_type_id:str
    rate_plan_id:str
    arrival:Date
    departure:Date
    occupants:int=Field(strict=True,gt=0,le=6)
    status:Literal['hold','confirmed','checked_in','checked_out','canceled','released','expired']='hold'
    total_cents:int=Field(strict=True,gt=0)
    deposit_cents:int=Field(strict=True,ge=0)
    hold_expires_at:datetime
    created_at:datetime


class Assignment(Model):
    id:str
    property_id:str
    reservation_id:str
    room_id:str
    arrival:Date
    departure:Date
    status:Literal['active','released']='active'
    previous_rooms:List[str]=Field(default_factory=list)


class Payment(Model):
    id:str
    property_id:str
    reservation_id:str
    kind:Literal['charge','refund']
    amount_cents:int=Field(strict=True,gt=0)
    key:str
    token:Literal['card_approved','card_declined']='card_approved'
    outcome:Literal['succeeded','failed']
    gateway_notice:Dict[str,Any]


class Cancellation(Model):
    id:str
    property_id:str
    reservation_id:str
    canceled_at:datetime
    fee_cents:int=Field(strict=True,ge=0)


class IdentityCheck(Model):
    id:str
    property_id:str
    guest_id:str
    reservation_id:str
    outcome:Literal['verified','declined']
    notice:Dict[str,Any]


class Stay(Model):
    id:str
    property_id:str
    reservation_id:str
    room_id:str
    status:Literal['checked_in','checked_out']='checked_in'
    check_in_at:datetime
    check_out_at:Optional[datetime]=None


class World(WorldBase):
    domain:Literal['stayops']='stayops'
    version:Literal['stayops-0.1.0']=VERSION
    users:Dict[str,User]
    properties:Dict[str,Property]
    access:Dict[str,Access]
    guests:Dict[str,Guest]
    room_types:Dict[str,RoomType]
    rooms:Dict[str,Room]
    calendar:Dict[str,CalendarDay]
    rate_plans:Dict[str,RatePlan]
    reservations:Dict[str,Reservation]=Field(default_factory=dict)
    assignments:Dict[str,Assignment]=Field(default_factory=dict)
    payments:Dict[str,Payment]=Field(default_factory=dict)
    cancellations:Dict[str,Cancellation]=Field(default_factory=dict)
    identity_checks:Dict[str,IdentityCheck]=Field(default_factory=dict)
    stays:Dict[str,Stay]=Field(default_factory=dict)


class Dates(Model):
    arrival:Date
    departure:Date

    @model_validator(mode='after')
    def interval(self):
        if self.departure<=self.arrival or (self.departure-self.arrival).days>30:
            raise ValueError('require a positive half-open stay interval of at most 30 nights')
        return self


class Hold(Dates):
    reservation_id:str=Field(min_length=1)
    guest_id:str
    room_type_id:str
    rate_plan_id:str
    occupants:int=Field(strict=True,gt=0,le=6)
    hold_minutes:Literal[15,30,60]

    @model_validator(mode='before')
    @classmethod
    def strict_minutes(cls,value):
        if isinstance(value,dict) and 'hold_minutes' in value and type(value['hold_minutes']) is not int:
            raise ValueError('hold_minutes must be an integer, never a bool/string')
        return value


class Availability(Dates):
    room_type_id:str


class RoomTypeChange(Model):
    room_type_id:str
    rate_plan_id:str


class Charge(Model):
    amount_cents:int=Field(strict=True,gt=0,le=100000000)
    key:str=Field(min_length=1,max_length=80)
    token:Literal['card_approved','card_declined']


class Refund(Model):
    amount_cents:int=Field(strict=True,gt=0,le=100000000)
    key:str=Field(min_length=1,max_length=80)


class RoomChoice(Model):
    room_id:str


class Switch(Model):
    enabled:bool=Field(strict=True)


class CalendarLimit(Model):
    sale_limit:int=Field(strict=True,ge=0,le=20)


PARAMETERS={
    'create_hold':Hold,'capture_payment':Charge,'confirm_reservation':Empty,
    'release_hold':Empty,'expire_hold':Empty,'change_dates':Dates,'change_room_type':RoomTypeChange,
    'assign_room':RoomChoice,'unassign_room':Empty,'cancel_reservation':Empty,'refund_reservation':Refund,
    'mark_room_clean':Empty,'set_room_maintenance':Switch,'check_in':Empty,'swap_room':RoomChoice,'check_out':Empty,
    'change_calendar_limit':CalendarLimit,'set_calendar_closed':Switch,'verify_guest':Empty,
    'get_property':Empty,'get_availability':Availability,'get_room':Empty,'get_reservation':Empty,
    'get_guest':Empty,'get_payment_ledger':Empty,'get_calendar_day':Empty,'get_rate_plan':Empty,
}
QUERIES=frozenset(n for n in PARAMETERS if n.startswith('get_'))
WRITES=tuple(n for n in PARAMETERS if n not in QUERIES)
TOOLS=tuple(PARAMETERS)+('get_customer_context',)


class Operation(OperationBase):
    parameters=PARAMETERS
    queries=QUERIES
