"""CloudOps domain types; no SaaS operation, world or goal semantics."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator

VERSION = 'cloudops-0.1.0'


class Model(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_assignment=True)


def digest(value):
    if isinstance(value, BaseModel): value = value.model_dump(mode='json')
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class Identity(Model):
    id: str
    name: str


class Organization(Model):
    id: str
    name: str
    owner_id: str


class Quota(Model):
    cpu: int = Field(strict=True, ge=0, le=64)
    memory: int = Field(strict=True, ge=0, le=256)
    instances: int = Field(strict=True, ge=0, le=16)


class Project(Model):
    id: str
    name: str
    org_id: str
    status: Literal['active', 'suspended']
    invoice_id: str
    quota: Quota


class Access(Model):
    id: str
    project_id: str
    user_id: str
    role: Literal['admin', 'operator', 'viewer']


class Instance(Model):
    id: str
    name: str
    project_id: str
    cpu: int = Field(strict=True, ge=1, le=64)
    memory: int = Field(strict=True, ge=1, le=256)
    status: Literal['stopped', 'running'] = 'stopped'
    dependencies: List[str] = Field(default_factory=list)


class Payment(Model):
    id: str
    key: str
    method: Literal['card_approved', 'card_declined']
    amount_cents: int = Field(strict=True, gt=0)
    outcome: Literal['succeeded', 'failed']
    at: datetime


class Invoice(Model):
    id: str
    project_id: str
    amount_cents: int = Field(strict=True, gt=0)
    status: Literal['unpaid', 'paid', 'failed']
    payments: List[Payment] = Field(default_factory=list)


class World(Model):
    domain: Literal['cloudops'] = 'cloudops'
    version: Literal['cloudops-0.1.0'] = VERSION
    seed: int = Field(strict=True)
    clock: datetime = datetime(2026, 1, 15, 12, tzinfo=timezone.utc)
    users: Dict[str, Identity]
    organizations: Dict[str, Organization]
    projects: Dict[str, Project]
    access: Dict[str, Access]
    instances: Dict[str, Instance]
    invoices: Dict[str, Invoice]

    @model_validator(mode='after')
    def utc_clock(self):
        if self.clock.tzinfo is None: raise ValueError('business clock must be timezone-aware')
        return self

    @property
    def state_hash(self): return digest(self)
    def clone(self): return self.model_copy(deep=True)


class Create(Model):
    instance_id: str = Field(pattern=r'^[A-Za-z][A-Za-z0-9_]{0,63}$')
    name: str = Field(min_length=1, max_length=80)
    cpu: int = Field(strict=True, ge=1, le=64)
    memory: int = Field(strict=True, ge=1, le=256)


class Dependency(Model):
    dependency_id: str


class Grant(Model):
    user_id: str
    role: Literal['admin', 'operator', 'viewer']


class Revoke(Model):
    user_id: str


class Pay(Model):
    payment_method: Literal['card_approved', 'card_declined']
    idempotency_key: str = Field(min_length=1, max_length=80)


class Resume(Model):
    invoice_id: str


class Empty(Model): pass

PARAMETERS = {'create_instance': Create, 'start_instance': Empty, 'stop_instance': Empty,
    'delete_instance': Empty, 'attach_dependency': Dependency, 'detach_dependency': Dependency,
    'change_project_quota': Quota, 'grant_project_access': Grant, 'revoke_project_access': Revoke,
    'pay_invoice': Pay, 'resume_project': Resume, 'get_project': Empty, 'list_instances': Empty,
    'get_instance': Empty, 'get_invoice': Empty, 'get_project_members': Empty}
QUERIES = frozenset({'get_project', 'list_instances', 'get_instance', 'get_invoice', 'get_project_members'})
WRITES = tuple(n for n in PARAMETERS if n not in QUERIES)
TOOLS = tuple(PARAMETERS) + ('get_customer_context',)


class Operation(Model):
    name: Literal['create_instance', 'start_instance', 'stop_instance', 'delete_instance',
        'attach_dependency', 'detach_dependency', 'change_project_quota', 'grant_project_access',
        'revoke_project_access', 'pay_invoice', 'resume_project', 'get_project', 'list_instances',
        'get_instance', 'get_invoice', 'get_project_members']
    project_id: str
    resource_id: str
    params: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def normalize(self):
        object.__setattr__(self, 'params', PARAMETERS[self.name].model_validate(self.params).model_dump(mode='json'))
        return self

    @property
    def is_query(self): return self.name in QUERIES


class Request(Model):
    id: str
    identity: str
    operation: Operation
    granted: bool = True

    def authorizes(self, actor, op): return self.granted and actor == self.identity and op == self.operation


class Result(Model):
    success: bool
    error_code: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)


class Decision(Model):
    allowed: bool
    error_code: Optional[str] = None
    expected_state: Optional[World] = None
    data: Dict[str, Any] = Field(default_factory=dict)
