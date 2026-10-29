from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

GENERATOR_VERSION = "saas-generator-0.1.0"
RULES_VERSION = "saas-rules-0.1.0"
GENERATOR_VERSION_V2 = "saas-generator-0.2.0"
RULES_VERSION_V2 = "saas-rules-0.2.0"


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def digest(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class Plan(Model):
    id: str
    name: str
    rank: int = Field(ge=0)
    max_seats: int = Field(ge=1)


class RulesConfig(Model):
    version: str = RULES_VERSION
    refund_window_days: int = Field(default=14, ge=0)
    plans: Dict[str, Plan] = Field(default_factory=lambda: {
        "starter": Plan(id="starter", name="Starter", rank=0, max_seats=5),
        "team": Plan(id="team", name="Team", rank=1, max_seats=20),
        "business": Plan(id="business", name="Business", rank=2, max_seats=100),
    })


class User(Model):
    id: str
    name: str
    email: str


class Organization(Model):
    id: str
    name: str
    owner_id: str
    subscription_id: str


class Membership(Model):
    id: str
    org_id: str
    user_id: str
    role: Literal["owner", "billing_admin", "member"]
    status: Literal["pending", "active", "inactive"] = "active"


class Subscription(Model):
    id: str
    org_id: str
    plan_id: str
    seats: int = Field(ge=0)
    auto_renew: bool
    period_start: datetime
    period_end: datetime
    status: Literal["active", "expired"]
    service_invoice_id: Optional[str] = Field(default=None, exclude_if=lambda v: v is None)


class Payment(Model):
    id: str
    key: str
    amount_cents: int = Field(gt=0)
    method: str
    outcome: Literal["succeeded", "failed"]
    attempted_at: datetime


class Refund(Model):
    id: str
    amount_cents: int = Field(ge=0)
    payment_method: str
    refunded_at: datetime
    key: Optional[str] = Field(default=None, exclude_if=lambda v: v is None)


class Invoice(Model):
    id: str
    org_id: str
    amount_cents: int = Field(gt=0)
    refundable_amount_cents: int = Field(ge=0)
    status: Literal["unpaid", "failed", "paid", "partially_refunded", "refunded"]
    paid_at: Optional[datetime] = None
    payment_method: Optional[str] = None
    refunds: List[Refund] = Field(default_factory=list)
    subscription_id: Optional[str] = Field(default=None, exclude_if=lambda v: v is None)
    period_start: Optional[datetime] = Field(default=None, exclude_if=lambda v: v is None)
    period_end: Optional[datetime] = Field(default=None, exclude_if=lambda v: v is None)
    purpose: Optional[Literal["historical", "renewal"]] = Field(default=None, exclude_if=lambda v: v is None)
    payments: Optional[List[Payment]] = Field(default=None, exclude_if=lambda v: v is None)


class WorldState(Model):
    world_id: str
    generator_version: str = GENERATOR_VERSION
    seed: int
    config_hash: str
    clock: datetime
    rules: RulesConfig = Field(default_factory=RulesConfig)
    users: Dict[str, User]
    organizations: Dict[str, Organization]
    memberships: Dict[str, Membership]
    subscriptions: Dict[str, Subscription]
    invoices: Dict[str, Invoice]

    @field_validator("clock")
    @classmethod
    def frozen_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("clock must include a timezone")
        return value.astimezone(timezone.utc)

    def clone(self) -> WorldState:
        return self.model_copy(deep=True)

    @property
    def state_hash(self) -> str:
        return digest(self)


class StateViolation(Model):
    code: str
    entity_id: str
    state_path: str
    detail: str
