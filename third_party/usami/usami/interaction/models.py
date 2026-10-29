"""Pinned episode configuration and auditable interaction evidence."""
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import Field, model_validator

from usami.customer.models import CustomerAction
from usami.evaluation.oracle import RequestEvaluation
from usami.protocol import Operation
from usami.service import CustomerInput, ModelSettings, ServiceExecution, ServiceTurn
from usami.service.models import LLMRecord, TOOL_NAMES, TOOL_NAMES_V2
from usami.world.generator import GeneratorConfig
from usami.world.models import Model, digest

INTERACTION_VERSION = "usami-interaction-0.1.0"
CUSTOMER_VERSION = "usami-customer-0.1.0"


class InteractionBudget(Model):
    max_customer_turns: int = Field(default=6, strict=True, ge=1, le=100)
    max_customer_llm_calls: int = Field(default=8, strict=True, ge=1, le=100)
    max_service_llm_calls: int = Field(default=24, strict=True, ge=1, le=100)
    max_service_tool_calls: int = Field(default=48, strict=True, ge=1, le=1000)


class InteractionConfig(Model):
    seed: int = Field(default=42, strict=True)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    customer_identity: str = Field(min_length=1)
    exploration_strategy: Optional[str] = None
    mode: Literal["reference", "sandbox"] = "reference"
    tools: List[str] = Field(default_factory=lambda: list(TOOL_NAMES))
    budget: InteractionBudget = Field(default_factory=InteractionBudget)

    @model_validator(mode="after")
    def fixed_tools(self):
        supported = TOOL_NAMES_V2 if self.generator.rules.version == "saas-rules-0.2.0" else TOOL_NAMES
        if len(self.tools) != len(set(self.tools)) or set(self.tools) - set(supported):
            raise ValueError("tools must be distinct supported Service tool names")
        return self


class IntentRecord(Model):
    operation: Operation
    disposition: Literal["accepted", "duplicate", "unverified_object"]
    reason: str
    request_id: Optional[str] = None


class PreparedTurn(Model):
    turn_id: str
    action: CustomerAction
    customer: CustomerInput
    intents: List[IntentRecord]
    natural_language_status: Literal["canonical_requests", "unverified_free_text"]


class InteractionTurn(Model):
    prepared: PreparedTurn
    service: ServiceTurn


class OperationAudit(Model):
    customer_identity: str
    turn_id: str
    call_id: str
    event_id: str
    request_id: str
    operation: Operation
    reference_allowed: Optional[bool]
    reference_error_code: Optional[str] = None
    expected_state_hash: Optional[str] = None
    oracle: RequestEvaluation
    classification: Literal["legal_execution", "rejected_invalid_attempt", "successful_business_violation",
                          "authorization_only", "false_rejection", "unscored"]


class InteractionTrace(Model):
    schema_version: str = INTERACTION_VERSION
    versions: Dict[str, str]
    config: InteractionConfig
    customer_model: ModelSettings
    service_model: ModelSettings
    initial_observation: Dict[str, Any]
    customer_records: List[LLMRecord]
    customer_actions: List[CustomerAction]
    customer_messages: List[Dict[str, Any]]
    turns: List[InteractionTurn]
    service_execution: ServiceExecution
    operation_audits: List[OperationAudit]
    status: str
    termination_reason: str
    stop_reason: str = ""
    metrics: Dict[str, Any]
    interaction_hash: str = ""

    def computed_hash(self):
        return digest(self.model_dump(mode="json", exclude={"interaction_hash"}))

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]):
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
