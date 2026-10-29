"""Residential assets, responsibility contracts and scarce technician calendars."""
from datetime import datetime
from typing import Literal,Optional
from pydantic import Field,model_validator
from usami_harness.models import Model,OperationBase,Empty
from usami_dialogue.models import DialogueWorldBase,CustomerPrivateState
VERSION='property-0.1.0'
Severity=Literal['normal','urgent','emergency']
Symptom=Literal['leak','sparks','not_cooling','stuck','routine_check']
class User(Model):
    id:str
    name:str
    contact:str
class Property(Model):
    id:str
    name:str
class Building(Model):
    id:str
    property_id:str
class Unit(Model):
    id:str
    building_id:str
    property_id:str
    owner_id:str
    resident_ids:list[str]
class Contract(Model):
    id:str
    property_id:str
    covered_categories:list[str]
    coverage_cents:int=Field(ge=0,strict=True)
    resident_approval_limit_cents:int=Field(ge=0,strict=True)
    normal_start_hour:int=Field(ge=0,le=23,strict=True)
    normal_end_hour:int=Field(ge=1,le=24,strict=True)
    urgent_response_hours:int=Field(ge=1,strict=True)
    emergency_response_hours:int=Field(ge=1,strict=True)
class Asset(Model):
    id:str
    unit_id:str
    property_id:str
    category:Literal['plumbing','electrical','hvac','lock']
    condition:Literal['faulty','working']
    simulated_symptom:Symptom
    simulated_hazard:bool
    responsibility:Literal['owner','resident','property']
    repair_cost_cents:int=Field(ge=0,strict=True)
    duration_minutes:int=Field(gt=0,strict=True)
    revision:int=Field(default=0,ge=0,strict=True)
    inspection_notice:dict=Field(default_factory=dict)
class Technician(Model):
    id:str
    property_id:str
    name:str
    skills:list[str]
    on_call:bool
class Slot(Model):
    id:str
    technician_id:str
    property_id:str
    start:datetime
    end:datetime
class Busy(Model):
    id:str
    technician_id:str
    property_id:str
    start:datetime
    end:datetime
class Ticket(Model):
    id:str
    unit_id:str
    asset_id:str
    reporter_id:str
    symptom:Symptom
    created_at:datetime
    status:Literal['open','triaged','scheduled','work_verified','awaiting_repair','closed']='open'
    severity:Optional[Severity]=None
    assessment_version:int=Field(default=-1,ge=-1,strict=True)
    version:int=Field(default=0,ge=0,strict=True)
    approval_requested:bool=False
    inspection_requested:bool=False
    appointment_id:Optional[str]=None
    booked_slot_id:Optional[str]=None
    repair_notice:dict=Field(default_factory=dict)
    closure_notice:dict=Field(default_factory=dict)
class Plan(Model):
    id:str
    ticket_id:str
    technician_id:str
    slot_id:str
    requester_id:str
    cost_cents:int=Field(ge=0,strict=True)
    customer_charge_cents:int=Field(ge=0,strict=True)
    payer_id:str
    revision:int=Field(ge=0,strict=True)
    ticket_version:int=Field(ge=0,strict=True)
    asset_revision:int=Field(ge=0,strict=True)
    expires_at:datetime
    status:Literal['offered','committed']='offered'
    quote_hash:str
class Appointment(Model):
    id:str
    ticket_id:str
    plan_id:str
    technician_id:str
    slot_id:str
    status:Literal['scheduled','cancelled','completed','failed']='scheduled'
class Complaint(Model):
    id:str
    ticket_id:str
    reporter_id:str
    reason:Literal['unresolved_repair','missed_visit','service_quality']
    status:Literal['open','escalated','reviewed','dismissed']='open'
    review_notice:dict=Field(default_factory=dict)
class Charge(Model):
    id:str
    ticket_id:str
    plan_id:str
    payer_id:str
    amount_cents:int=Field(gt=0,strict=True)
    notice:dict
class ExternalEvent(Model):
    id:str
    asset_id:str
    kind:Literal['owner_review','inspection','technician_visit','complaint_review']
    timing:Literal['fixed','appointment_end']='fixed'
    at:Optional[datetime]=None
    outcome:Literal['approved','denied','hazard','safe','repaired','unresolved','no_show','reviewed','dismissed']
    @model_validator(mode='after')
    def native_event_contract(self):
        choices={'owner_review':{'approved','denied'},'inspection':{'hazard','safe'},'technician_visit':{'repaired','unresolved','no_show'},'complaint_review':{'reviewed','dismissed'}}
        if self.outcome not in choices[self.kind]:raise ValueError('event outcome does not belong to this actor')
        if self.kind=='technician_visit':
            if self.timing!='appointment_end' or self.at is not None:raise ValueError('physical work uses the actual booked appointment end')
        elif self.timing!='fixed' or self.at is None or self.at.tzinfo is None:raise ValueError('review/inspection requires a fixed aware timestamp')
        return self
class World(DialogueWorldBase):
    domain:Literal['property']='property'
    version:Literal['property-0.1.0']=VERSION
    users:dict[str,User]
    properties:dict[str,Property]
    buildings:dict[str,Building]
    units:dict[str,Unit]
    contracts:dict[str,Contract]
    assets:dict[str,Asset]
    technicians:dict[str,Technician]
    slots:dict[str,Slot]
    background_busy:dict[str,Busy]
    profiles:dict[str,CustomerPrivateState]
    tickets:dict[str,Ticket]=Field(default_factory=dict)
    plans:dict[str,Plan]=Field(default_factory=dict)
    appointments:dict[str,Appointment]=Field(default_factory=dict)
    complaints:dict[str,Complaint]=Field(default_factory=dict)
    charges:dict[str,Charge]=Field(default_factory=dict)
    approvals:dict[str,dict]=Field(default_factory=dict)
    external_events:dict[str,ExternalEvent]=Field(default_factory=dict)
    applied_events:list[str]=Field(default_factory=list)
class AssetInput(Model):
    asset_id:str
class Open(AssetInput):
    symptom:Symptom
class TicketInput(Model):
    ticket_id:str
class Triage(TicketInput):
    severity:Severity
class PlanInput(TicketInput):
    technician_id:str
    slot_id:str
class PlanCommit(Model):
    plan_id:str
class ComplaintInput(TicketInput):
    reason:Literal['unresolved_repair','missed_visit','service_quality']
class ComplaintId(Model):
    complaint_id:str
PARAMETERS={'get_units':Empty,'get_unit':Empty,'get_asset':AssetInput,'get_ticket':TicketInput,'get_technicians':Empty,'get_plan':PlanCommit,
 'open_ticket':Open,'triage_ticket':Triage,'request_owner_review':TicketInput,'request_inspection':TicketInput,'quote_visit':PlanInput,
 'book_visit':PlanCommit,'cancel_visit':TicketInput,'close_ticket':TicketInput,'file_complaint':ComplaintInput,'escalate_complaint':ComplaintId}
QUERIES=frozenset(n for n in PARAMETERS if n.startswith('get_'))
TOOLS=tuple(PARAMETERS)+('get_customer_context',)
class Operation(OperationBase):
    parameters=PARAMETERS
    queries=QUERIES
