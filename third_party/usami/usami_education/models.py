"""Study-credit contracts, cohort places and actual teaching/learning records."""
from datetime import datetime
from typing import Literal,Optional
from pydantic import Field,model_validator
from usami_harness.models import Model,OperationBase,Empty
from usami_dialogue.models import DialogueWorldBase,CustomerPrivateState
VERSION='education-0.1.0'
Reason=Literal['too_busy','illness','level_mismatch','teaching_quality']
class User(Model):
    id:str
    name:str
    contact:str
    levels:dict[str,int]
class School(Model):
    id:str
    name:str
class Course(Model):
    id:str
    school_id:str
    subject:Literal['math','language','coding']
    level:int=Field(ge=1,le=3,strict=True)
    unit_price_cents:int=Field(gt=0,strict=True)
    unit_minutes:int=Field(default=60,gt=0,strict=True)
class Package(Model):
    id:str
    school_id:str
    course_ids:list[str]=Field(min_length=1)
class Teacher(Model):
    id:str
    school_id:str
    name:str
    subjects:list[str]
    max_level:int=Field(ge=1,le=3,strict=True)
class Meeting(Model):
    id:str
    start:datetime
    end:datetime
class Cohort(Model):
    id:str
    school_id:str
    course_id:str
    teacher_id:str
    capacity:int=Field(ge=1,strict=True)
    meetings:list[Meeting]=Field(min_length=1)
class Contract(Model):
    id:str
    school_id:str
    package_id:str
    total_units:int=Field(gt=0,strict=True)
    refund_days:int=Field(ge=0,strict=True)
    refund_consumed_limit_bps:int=Field(ge=0,le=10000,strict=True)
    refund_fee_cents:int=Field(ge=0,strict=True)
    transferable:bool
    transfer_fee_cents:int=Field(ge=0,strict=True)
    max_extension_days:int=Field(ge=0,strict=True)
    self_service_extension_days:int=Field(ge=0,strict=True)
    extension_fee_per_day_cents:int=Field(ge=0,strict=True)
    extension_grace_days:int=Field(ge=0,strict=True)
    extension_reasons:list[Reason]
    no_show_consumes:bool
class Enrollment(Model):
    id:str
    school_id:str
    student_id:str
    course_id:str
    original_course_id:str
    class_id:str
    contract_id:str
    enrolled_at:datetime
    expires_at:datetime
    status:Literal['active','refunded']='active'
    initial_paid_cents:int=Field(ge=0,strict=True)
    remaining_credit_cents:int=Field(ge=0,strict=True)
    retained_fees_cents:int=Field(default=0,ge=0,strict=True)
    extended_days:int=Field(default=0,ge=0,strict=True)
    payment_channel:str
    version:int=Field(default=0,ge=0,strict=True)
class LearningRecord(Model):
    id:str
    enrollment_id:str
    course_id:str
    units:int=Field(gt=0,strict=True)
    unit_price_cents:int=Field(gt=0,strict=True)
    origin:Literal['opening','attendance']
    appointment_id:Optional[str]=None
    notice:dict
class Slot(Model):
    id:str
    school_id:str
    teacher_id:str
    start:datetime
    end:datetime
class Busy(Model):
    id:str
    kind:Literal['student','teacher']
    resource_id:str
    start:datetime
    end:datetime
class Case(Model):
    id:str
    enrollment_id:str
    request_id:str
    requester_id:str
    reason:Reason
    created_at:datetime
    pending_extension_days:Optional[int]=None
    review_version:Optional[int]=None
    review_notice:dict=Field(default_factory=dict)
    last_slot_id:Optional[str]=None
    last_appointment_id:Optional[str]=None
    last_lesson_status:Optional[str]=None
    attended_units:int=Field(default=0,ge=0,strict=True)
class Quote(Model):
    id:str
    enrollment_id:str
    case_id:str
    action:Literal['refund','transfer','extend','tutoring']
    target_class_id:Optional[str]=None
    slot_id:Optional[str]=None
    extension_days:int=Field(default=0,ge=0,strict=True)
    topup_units:int=Field(default=0,ge=0,strict=True)
    charge_cents:int=Field(default=0,ge=0,strict=True)
    refund_cents:int=Field(default=0,ge=0,strict=True)
    fee_cents:int=Field(default=0,ge=0,strict=True)
    resulting_credit_cents:int=Field(ge=0,strict=True)
    resulting_expiry:datetime
    revision:int=Field(ge=0,strict=True)
    enrollment_version:int=Field(ge=0,strict=True)
    expires_at:datetime
    status:Literal['offered','committed']='offered'
    quote_hash:str
class Appointment(Model):
    id:str
    enrollment_id:str
    case_id:str
    quote_id:str
    course_id:str
    slot_id:str
    unit_price_cents:int=Field(gt=0,strict=True)
    status:Literal['scheduled','attended','missed','cancelled']='scheduled'
    attendance_notice:dict=Field(default_factory=dict)
class Ledger(Model):
    id:str
    enrollment_id:str
    case_id:str
    quote_id:str
    kind:Literal['charge','refund']
    amount_cents:int=Field(gt=0,strict=True)
    channel:str
    notice:dict
class ExternalEvent(Model):
    id:str
    enrollment_id:str
    request_id:str
    kind:Literal['academic_review','attendance']
    outcome:Literal['approved','denied','attended','student_no_show','teacher_cancelled']
    at:Optional[datetime]=None
    grant_days:int=Field(default=0,ge=0,strict=True)
    @model_validator(mode='after')
    def actor_contract(self):
        if self.kind=='academic_review':
            if self.outcome not in {'approved','denied'} or self.at is None or self.at.tzinfo is None:raise ValueError('academic review requires fixed time and approval outcome')
        elif self.outcome not in {'attended','student_no_show','teacher_cancelled'} or self.at is not None:raise ValueError('attendance uses the actual selected lesson end')
        return self
class World(DialogueWorldBase):
    domain:Literal['education']='education'
    version:Literal['education-0.1.0']=VERSION
    users:dict[str,User]
    schools:dict[str,School]
    courses:dict[str,Course]
    packages:dict[str,Package]
    teachers:dict[str,Teacher]
    classes:dict[str,Cohort]
    contracts:dict[str,Contract]
    enrollments:dict[str,Enrollment]
    learning_records:dict[str,LearningRecord]
    slots:dict[str,Slot]
    background_busy:dict[str,Busy]
    profiles:dict[str,CustomerPrivateState]
    cases:dict[str,Case]=Field(default_factory=dict)
    quotes:dict[str,Quote]=Field(default_factory=dict)
    appointments:dict[str,Appointment]=Field(default_factory=dict)
    ledger:dict[str,Ledger]=Field(default_factory=dict)
    external_events:dict[str,ExternalEvent]=Field(default_factory=dict)
    applied_events:list[str]=Field(default_factory=list)
class Open(Model):
    request_id:str
    reason:Reason
class CaseInput(Model):
    case_id:str
class Transfer(CaseInput):
    target_class_id:str
    topup_units:int=Field(default=0,ge=0,strict=True)
class Extension(CaseInput):
    days:int=Field(gt=0,strict=True)
class Lesson(CaseInput):
    slot_id:str
class Commit(Model):
    quote_id:str
    payment_channel:str
class Cancel(Model):
    appointment_id:str
PARAMETERS={'get_enrollments':Empty,'get_enrollment':Empty,'get_progress':Empty,'get_catalogue':Empty,'get_lesson_slots':Empty,'get_case':CaseInput,
 'open_request':Open,'request_extension_review':Extension,'quote_refund':CaseInput,'quote_transfer':Transfer,'quote_extension':Extension,'quote_lesson':Lesson,'commit_quote':Commit,'cancel_lesson':Cancel}
QUERIES=frozenset(n for n in PARAMETERS if n.startswith('get_'))
TOOLS=tuple(PARAMETERS)+('get_customer_context',)
class Operation(OperationBase):
    parameters=PARAMETERS
    queries=QUERIES
