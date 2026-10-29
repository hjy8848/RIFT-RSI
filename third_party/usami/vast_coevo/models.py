"""Versioned, environment-independent experiment and evidence contracts."""
from __future__ import annotations

import copy
from typing import Dict, Literal, Optional, Tuple
from urllib.parse import parse_qsl, urlsplit
from pydantic import Field, JsonValue, model_validator

from .schema import FrozenModel, digest
from .skills.models import Hash, SkillRole
from .skills.retriever import SkillHit

Partition = Literal["development", "validation", "heldout"]
Classification = Literal["successful_business_violation", "rejected_invalid_attempt", "authorization_only", "false_rejection", "legal_execution", "unscored"]


class ModelSpec(FrozenModel):
    schema_version: Literal[1] = 1
    model: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    options: Dict[str, JsonValue] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=60, gt=0, strict=True, allow_inf_nan=False)
    trust_env: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def no_credentials(self):
        forbidden = {"api_key", "authorization", "headers", "extra_headers", "access_token", "model", "messages", "tools", "tool_choice"}
        if any(key.casefold().replace("-", "_") in forbidden for key in self.options):
            raise ValueError("model options cannot contain credentials or replace protocol fields")
        extra = self.options.get("extra_body", {})
        if isinstance(extra, dict) and any(key.casefold().replace("-", "_") in forbidden for key in extra):
            raise ValueError("extra_body cannot replace credentials or native protocol fields")
        endpoint = urlsplit(self.base_url)
        if endpoint.username is not None or endpoint.password is not None or any(
            key.casefold().replace("-", "_") in {"api_key", "key", "token", "access_token", "authorization"}
            for key, _ in parse_qsl(endpoint.query, keep_blank_values=True)
        ):
            raise ValueError("credentials must not be embedded in the API URL")
        object.__setattr__(self, "options", copy.deepcopy(self.options))
        return self


class Budget(FrozenModel):
    max_customer_turns: int = Field(default=3, ge=1, strict=True)
    max_customer_llm_calls: int = Field(default=4, ge=0, strict=True)
    max_service_llm_calls: int = Field(default=8, ge=1, strict=True)
    max_service_tool_calls: int = Field(default=12, ge=0, strict=True)
    max_format_recoveries: int = Field(default=0, ge=0, strict=True)


class EpisodeConfig(FrozenModel):
    schema_version: Literal[1] = 1
    id: str = Field(min_length=1)
    config_id: str = Field(min_length=1)
    environment: str = Field(default="usami", min_length=1)
    domain: str = Field(min_length=1)
    partition: Partition = "development"
    initial_world_hash: Hash
    identity: str = Field(min_length=1)
    mode: Literal["reference", "sandbox"] = "reference"
    customer_model: ModelSpec
    service_model: ModelSpec
    budget: Budget
    tools: Tuple[str, ...] = ()
    config_hash: str = ""

    @model_validator(mode="after")
    def seal(self):
        object.__setattr__(self, "customer_model", ModelSpec.model_validate(self.customer_model.model_dump(mode="json")))
        object.__setattr__(self, "service_model", ModelSpec.model_validate(self.service_model.model_dump(mode="json")))
        object.__setattr__(self, "budget", Budget.model_validate(self.budget.model_dump(mode="json")))
        if len(set(self.tools)) != len(self.tools):
            raise ValueError("duplicate configured tools")
        expected = digest(self.model_dump(mode="json", exclude={"config_hash"}))
        if self.config_hash and self.config_hash != expected:
            raise ValueError("episode configuration hash mismatch")
        object.__setattr__(self, "config_hash", expected)
        return self


class PublicContext(FrozenModel):
    domain: str
    public_text: str = ""
    intents: Tuple[str, ...] = ()
    available_tools: Tuple[str, ...] = ()


class ArtifactRef(FrozenModel):
    path: str = Field(min_length=1)
    sha256: Hash


class ReplayResult(FrozenModel):
    verified: bool
    reason: str = ""
    request_count: int = Field(default=0, ge=0, strict=True)
    native_trace_hash: Optional[Hash] = None


class ModelCall(FrozenModel):
    schema_version: Literal[1] = 1
    id: str = Field(min_length=1)
    role: Literal["customer", "service", "customer_reflector", "service_reflector", "designer", "designer_reflector"]
    model: ModelSpec
    base_request: Dict[str, JsonValue]
    request: Dict[str, JsonValue]
    response: Optional[JsonValue] = None
    error: Optional[Dict[str, JsonValue]] = None
    request_hash: str = ""
    response_hash: Optional[str] = None

    @model_validator(mode="after")
    def hashes(self):
        object.__setattr__(self, "model", ModelSpec.model_validate(self.model.model_dump(mode="json")))
        for name in ("base_request", "request", "response", "error"):
            object.__setattr__(self, name, copy.deepcopy(getattr(self, name)))
        expected = digest(self.request)
        if self.request_hash and self.request_hash != expected:
            raise ValueError("actual request hash mismatch")
        object.__setattr__(self, "request_hash", expected)
        rh = digest(self.response) if self.response is not None else None
        if self.response_hash and self.response_hash != rh:
            raise ValueError("response hash mismatch")
        object.__setattr__(self, "response_hash", rh)
        return self


class InjectionRecord(FrozenModel):
    call_id: str
    role: SkillRole
    snapshot_hash: Hash
    context_hash: Hash
    hits: Tuple[SkillHit, ...] = ()
    rendered: str = ""
    request_hash: Hash


class Consumption(FrozenModel):
    customer_llm_calls: int = Field(default=0, ge=0, strict=True)
    service_llm_calls: int = Field(default=0, ge=0, strict=True)
    reflection_calls: int = Field(default=0, ge=0, strict=True)
    tool_calls: int = Field(default=0, ge=0, strict=True)

    @property
    def llm_calls(self):
        return self.customer_llm_calls + self.service_llm_calls + self.reflection_calls


class VerifiedEvent(FrozenModel):
    event_id: str
    operation: str
    classification: Classification
    reference_codes: Tuple[str, ...] = ()
    successful_execution: bool
    authorized: Optional[bool] = None
    operation_correct: Optional[bool] = None
    before_hash: Hash
    after_hash: Hash


class VerifiedFailure(FrozenModel):
    schema_version: Literal[1] = 1
    episode_id: str
    event_id: str
    domain: str
    operation: str
    reference_codes: Tuple[str, ...]
    relevant_prestate: Dict[str, JsonValue]
    permission_scope: Dict[str, JsonValue]
    causal_prerequisites: Tuple[Dict[str, JsonValue], ...] = ()
    expected: Dict[str, JsonValue]
    actual: Dict[str, JsonValue]
    successful_business_violation: bool
    authorization_only: bool = False
    trace: ArtifactRef
    fingerprint: Hash
    normalization_scope: Literal["policy_relative_features", "pending_causal_normalization"]

    @model_validator(mode="after")
    def real_business_failure(self):
        if not self.successful_business_violation or self.authorization_only:
            raise ValueError("business failure archive requires actual successful business violation")
        for name in ("relevant_prestate", "permission_scope", "expected", "actual", "causal_prerequisites"):
            object.__setattr__(self, name, copy.deepcopy(getattr(self, name)))
        return self


class TaskDecision(FrozenModel):
    """Task-level decision proof, deliberately not an Engine operation event."""
    protocol: Literal['frozen-task-decision-v1']='frozen-task-decision-v1'
    kind: Literal['false_rejection']='false_rejection'
    operation: Dict[str, JsonValue]
    before_hash: Hash
    claimed_code: str
    model_call_id: str
    response_hash: Hash
    turn_id: str


class EpisodeEvidence(FrozenModel):
    schema_version: Literal[1] = 1
    episode_id: str
    config: EpisodeConfig
    track: Literal["discovery", "service_task", "failure_regression"] = "discovery"
    task_ref: Optional[str] = None
    # An authoring episode starts from a template, then freezes its execution scene.
    execution_initial_world_hash: Optional[Hash] = Field(default=None, exclude_if=lambda value: value is None)
    execution_identity: Optional[str] = Field(default=None, exclude_if=lambda value: value is None)
    customer_snapshot_hash: Hash
    service_snapshot_hash: Hash
    model_calls: Tuple[ModelCall, ...] = ()
    injections: Tuple[InjectionRecord, ...] = ()
    events: Tuple[VerifiedEvent, ...] = ()
    failures: Tuple[VerifiedFailure, ...] = ()
    # Replay-verified native assessments remain separate from confirmed business
    # violations. Omission preserves historical evidence hashes.
    native_assessment: Optional[Dict[str, JsonValue]] = Field(default=None, exclude_if=lambda value: value is None)
    task_decisions: Optional[Tuple[TaskDecision,...]] = Field(default=None, exclude_if=lambda v: v is None)
    consumption: Consumption
    termination: str
    trace: ArtifactRef
    native_trace_hash: Hash
    replay: ReplayResult
    task_success: Optional[bool] = None
    legal_task_success: Optional[bool] = None
    correct_refusal: Optional[bool] = None
    goal_satisfied: Optional[bool] = None
    frame_preserved: Optional[bool] = None
    evidence_hash: str = ""

    @model_validator(mode="after")
    def seal(self):
        object.__setattr__(self, "config", EpisodeConfig.model_validate(self.config.model_dump(mode="json")))
        object.__setattr__(self, "model_calls", tuple(ModelCall.model_validate(call.model_dump(mode="json")) for call in self.model_calls))
        object.__setattr__(self, "consumption", Consumption.model_validate(self.consumption.model_dump(mode="json")))
        if self.track == "service_task" and self.task_ref is None:
            raise ValueError("service task evidence requires its fixed task reference")
        if any(f.episode_id != self.episode_id or f.domain != self.config.domain for f in self.failures):
            raise ValueError("failure evidence belongs to another episode")
        expected = digest(self.model_dump(mode="json", exclude={"evidence_hash"}))
        if self.evidence_hash and self.evidence_hash != expected:
            raise ValueError("evidence content hash mismatch")
        object.__setattr__(self, "evidence_hash", expected)
        return self
