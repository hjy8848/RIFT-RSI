"""Harness-owned customer input and frozen experiment configuration."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from usami.protocol import Operation
from usami.world.generator import GeneratorConfig
from usami.world.models import Model, digest

SERVICE_VERSION = "usami-service-0.1.0"
TOOL_NAMES = ["get_customer_context", "get_organization", "list_members", "get_subscription",
              "get_invoice", "change_plan", "change_seats", "cancel_renewal", "refund_invoice",
              "change_member_role"]
TOOL_NAMES_V2 = TOOL_NAMES + ["invite_member", "activate_member", "deactivate_member", "remove_member",
    "pay_invoice", "issue_renewal", "renew_subscription", "reactivate_subscription", "expire_subscription"]


class CustomerInput(Model):
    """Supplied by a trusted customer adapter, before the Service is invoked.

    The message is model-visible. Exact single-use write authorizations are not.
    Their order does not prescribe tool execution order. Duplicate operations
    express separate customer requests, each with its own single-use grant.
    """
    identity: str = Field(min_length=1)
    message: str = Field(min_length=1)
    authorizations: List[Operation] = Field(default_factory=list)

    @model_validator(mode="after")
    def writes_only(self):
        if any(op.is_query for op in self.authorizations):
            raise ValueError("authorizations contain writes only; queries use the fixed read capability")
        return self


class Experiment(Model):
    seed: int = Field(default=42, strict=True)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    customer: CustomerInput
    mode: Literal["reference", "sandbox"] = "reference"
    tools: List[str] = Field(default_factory=lambda: list(TOOL_NAMES))
    max_llm_calls: int = Field(default=8, strict=True, ge=1, le=100)
    max_tool_calls: int = Field(default=16, strict=True, ge=1, le=1000)

    @model_validator(mode="after")
    def fixed_tools(self):
        supported = TOOL_NAMES_V2 if self.generator.rules.version == "saas-rules-0.2.0" else TOOL_NAMES
        if len(self.tools) != len(set(self.tools)) or set(self.tools) - set(supported):
            raise ValueError("tools must be distinct supported tool names")
        return self

    @property
    def experiment_hash(self):
        return digest(self)


class ModelSettings(Model):
    """Public, reproducible request settings; authentication is held separately."""
    model: str = Field(min_length=1)
    base_url: str
    timeout_seconds: float = Field(default=60, gt=0, le=600)
    options: Dict[str, Any] = Field(default_factory=lambda: {"max_tokens": 2048})
    trust_env: bool = False

    @model_validator(mode="after")
    def safe_settings(self):
        url = urlsplit(self.base_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("base_url must be an HTTP(S) API base without credentials, query or fragment")
        blocked = {"model", "messages", "tools", "tool_choice", "stream", "n", "api_key", "authorization", "headers"}
        if any(k.lower() in blocked for k in self.options):
            raise ValueError("options cannot override harness fields or contain authentication")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))
        return self

    def request(self, messages, tools):
        return {**self.options, "model": self.model, "messages": messages, "tools": tools,
                "tool_choice": "auto" if tools else "none", "stream": False, "n": 1}


class ToolRecord(Model):
    call_id: str
    name: str
    arguments: str
    dispatched: bool
    response: Dict[str, Any]
    event_id: str = ""
    before_hash: str
    after_hash: str


class LLMRecord(Model):
    request: Dict[str, Any]
    response: Any = None
    error: Dict[str, Any] = Field(default_factory=dict)


class ServiceTurn(Model):
    """A response boundary, not a finalized engine or episode evaluation."""
    turn_id: str
    customer: CustomerInput
    status: str
    final_reply: Optional[str] = None
    llm_start: int
    llm_end: int
    tool_start: int
    tool_end: int
    before_hash: str
    after_hash: str
    request_ids: List[str] = Field(default_factory=list)
    expired_request_ids: List[str] = Field(default_factory=list)
    duplicate_operations: List[Operation] = Field(default_factory=list)


class ServiceError(Exception):
    def __init__(self, code: str, message: str, response: Any = None):
        super().__init__(message)
        self.code, self.response = code, response


def strict_json(text):
    """Reject duplicate keys and non-JSON numeric values before tool validation."""
    import json

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("non-finite JSON number")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
