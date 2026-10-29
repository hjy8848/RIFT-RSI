"""Immutable, content-addressed contracts; no provider or legacy runtime changes."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional, Tuple, Union

from pydantic import Field, StrictBool, StrictInt, StrictStr, model_validator

from vast_coevo.schema import FrozenModel, digest


Scalar = Union[StrictBool, StrictInt, StrictStr]
Role = Literal["designer", "customer", "service", "oracle", "learner"]
FactKind = Literal[
    "refund_amount", "extra_charge", "arrival_date", "refund_status",
    "shipment_status", "quantity", "currency",
]


class Sealed(FrozenModel):
    record_hash: str = ""

    @model_validator(mode="after")
    def seal(self):
        expected = digest(self.model_dump(mode="json", exclude={"record_hash"}))
        if self.record_hash and self.record_hash != expected:
            raise ValueError("content hash differs")
        object.__setattr__(self, "record_hash", expected)
        return self

    def checked(self):
        # Re-parse even when the caller passes a model_copy(update=...) instance.
        return type(self).model_validate(self.model_dump(mode="json"))


class SourceRef(Sealed):
    source_id: str = Field(min_length=1)
    source_kind: Literal[
        "customer_message", "public_policy", "public_tool", "transaction_receipt",
        "private_scenario", "task_spec", "oracle", "reference_witness", "heldout",
    ]
    source_hash: str = Field(min_length=1)
    event_index: int = Field(ge=0, strict=True)


class FactEvidence(Sealed):
    fact_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    kind: FactKind
    value: Scalar
    unit: str = ""
    valid_from: int = Field(ge=0, strict=True)
    valid_until: Optional[int] = Field(default=None, ge=0, strict=True)
    source: SourceRef
    visible_to: Tuple[Role, ...] = ("service", "oracle")

    @model_validator(mode="after")
    def typed_fact(self):
        if self.valid_until is not None and self.valid_until < self.valid_from:
            raise ValueError("fact validity interval is reversed")
        if self.source.event_index > self.valid_from:
            raise ValueError("fact cannot predate its source")
        if self.kind in ("refund_amount", "extra_charge", "quantity"):
            if type(self.value) is not int or self.value < 0:
                raise ValueError("amount/quantity facts require nonnegative integers")
        if self.kind in ("refund_amount", "extra_charge") and self.unit != "CNY_cents":
            raise ValueError("v2 commerce amounts use CNY integer cents")
        if self.kind == "arrival_date":
            if not isinstance(self.value, str):
                raise ValueError("arrival date must be ISO date")
            from datetime import date
            date.fromisoformat(self.value)
        return self


class PublicRequest(Sealed):
    request_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    fact_kind: FactKind
    source: SourceRef
    public_text: str = Field(min_length=1)

    @model_validator(mode="after")
    def independent_public_source(self):
        if self.source.source_kind not in ("customer_message", "public_policy"):
            raise ValueError("information obligation needs an actual public request/rule")
        return self


class CommunicationObligation(Sealed):
    obligation_id: str = Field(min_length=1)
    request: PublicRequest
    due_event_index: int = Field(ge=0, strict=True)
    required: bool = True

    @model_validator(mode="after")
    def deadline_follows_request(self):
        if self.due_event_index < self.request.source.event_index:
            raise ValueError("information due boundary precedes its trigger")
        return self


class ServiceMessage(Sealed):
    message_id: str = Field(min_length=1)
    event_index: int = Field(ge=0, strict=True)
    text: str = Field(min_length=1)
    # Resolved from public conversation, not hidden task labels.
    subject_ids: Tuple[str, ...]
    fact_refs: Tuple[str, ...] = ()

    @model_validator(mode="after")
    def unique_refs(self):
        if len(set(self.subject_ids)) != len(self.subject_ids):
            raise ValueError("duplicate public subject")
        if len(set(self.fact_refs)) != len(self.fact_refs):
            raise ValueError("duplicate fact reference")
        return self


class CounterfactualBudget(Sealed):
    max_nodes: int = Field(default=512, ge=1, le=100000, strict=True)
    max_depth: int = Field(default=12, ge=1, le=100, strict=True)
    max_alternatives: int = Field(default=16, ge=1, le=512, strict=True)


def aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timezone-aware virtual business time required")
    return value
