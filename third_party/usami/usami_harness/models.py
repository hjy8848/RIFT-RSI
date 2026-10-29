"""Wire types and strict primitives shared by independent business adapters."""
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_assignment=True)


def digest(value):
    if isinstance(value, BaseModel):
        value = value.model_dump(mode='json')
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class WorldBase(Model):
    seed: int = Field(strict=True)
    clock: datetime = datetime(2026, 1, 15, 12, tzinfo=timezone.utc)

    @model_validator(mode='after')
    def aware_clock(self):
        if self.clock.tzinfo is None:
            raise ValueError('a controlled timezone-aware business clock is required')
        return self

    def clone(self):
        return self.model_copy(deep=True)

    @property
    def state_hash(self):
        return digest(self)


class OperationBase(Model):
    name: str
    scope_id: str = Field(min_length=1)
    resource_id: str = Field(min_length=1)
    params: Dict[str, Any] = Field(default_factory=dict)
    parameters: ClassVar[Dict[str, type]] = {}
    queries: ClassVar[frozenset] = frozenset()

    @model_validator(mode='after')
    def typed_parameters(self):
        if self.name not in self.parameters:
            raise ValueError('unknown operation in this domain')
        value = self.parameters[self.name].model_validate(self.params).model_dump(mode='json')
        object.__setattr__(self, 'params', value)
        return self

    @property
    def is_query(self):
        return self.name in self.queries


class Request(Model):
    id: str = Field(min_length=1)
    identity: str = Field(min_length=1)
    operation: Dict[str, Any]
    granted: bool = Field(default=True, strict=True)

    def authorizes(self, actor, operation):
        return self.granted and self.identity == actor and self.operation == operation.model_dump(mode='json')


class Result(Model):
    success: bool
    error_code: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)


class Decision(Model):
    allowed: bool
    error_code: Optional[str] = None
    expected_state: Any = None
    data: Dict[str, Any] = Field(default_factory=dict)


class Empty(Model):
    pass


def signed_notice(domain, kind, binding, clock):
    """Deterministic trusted simulator attestation, not a production signature.

    Only host-side transitions issue notices; no LLM tool accepts source/outcome
    or a raw attestation. Oracle reconstructs the complete expected binding.
    """
    payload = {'domain': domain, 'source': kind + '-simulator-v0.1',
               'binding': binding, 'at': clock.isoformat()}
    return {**payload, 'signature': digest(payload)}


def notice_valid(notice):
    return notice.get('signature') == digest({k: v for k, v in notice.items() if k != 'signature'})


class Budget(Model):
    max_customer_turns: int = Field(default=8, strict=True, ge=1, le=100)
    max_customer_llm_calls: int = Field(default=12, strict=True, ge=0, le=200)
    max_service_llm_calls: int = Field(default=32, strict=True, ge=1, le=200)
    max_service_tool_calls: int = Field(default=64, strict=True, ge=1, le=2000)
    max_format_recoveries: int = Field(default=1, strict=True, ge=0, le=3)
