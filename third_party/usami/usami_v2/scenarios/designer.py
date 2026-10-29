"""Two-stage, world-grounded Designer with independent business certification.

Stage one proposes only a constrained ScenarioSpec draft. Stage two sees the
constructed business world and proposes a TaskDraft using aliases. The compiler
resolves aliases and attaches immutable provenance; the Designer never supplies
an answer key, Reference implementation or proof witness.
"""
from datetime import timedelta
import json
from typing import Any, Callable, Dict, Literal, Optional, Protocol, Tuple, Union

from pydantic import Field, ValidationError, model_validator

from usami.service.client import CompletionClient
from usami.service.models import ModelSettings, ServiceError
from usami_harness.models import digest
from usami_v2.contracts import Sealed
from usami_v2.runtime.agents import AgentCall, _json_arguments, _one_tool_call, _usage
from usami_v2.runtime.runner import EpisodeRecord, replay_episode
from usami_v2.commerce.models import World, unit_credit
from usami_v2.scenarios.builder import BuildResult, build_world, verify_build
from usami_v2.scenarios.distribution import DistributionSpec, SamplingSlot
from usami_v2.scenarios.schema import (
    Attribute, IntegerRange, LineVariable, OrderVariable, RelationConstraint,
    SKUVariable, StockConstraint, WarehouseVariable, ScenarioSpec, BuildBudget,
)
from usami_v2.scenarios.task_certification import (
    CustomerScenario as ProofCustomerScenario, CustomerTaskScope, ProofBudget,
    TaskCertification, TaskDesignCapabilities, certify_task, diagnose_candidate_alternatives,
    verify_task_certificate,
)
from usami_v2.scenarios.tasks import (
    CustomerRequestBundle, InformationObjective, RemedyGoal, StructuredInformationObjective,
    TaskSpec, TaskSpecV2, TaskSpecV3, check_task_scope, render_customer_request,
    request_intent_hash,
)


class ScenarioDraft(Sealed):
    scenario_id: str
    merchant_alias: str = "merchant"
    customer_aliases: Tuple[str, ...]
    skus: Tuple[SKUVariable, ...]
    warehouses: Tuple[WarehouseVariable, ...]
    orders: Tuple[OrderVariable, ...]
    stock: Tuple[StockConstraint, ...] = ()
    default_stock: IntegerRange = IntegerRange(low=0, high=5)
    refund_approval_limit_cents: int = Field(default=50000, ge=0, strict=True)
    constraints: Tuple[RelationConstraint, ...] = ()

    def compile(self, distribution: DistributionSpec):
        return ScenarioSpec(
            scenario_id=self.scenario_id, merchant_alias=self.merchant_alias,
            customer_aliases=self.customer_aliases, skus=self.skus, warehouses=self.warehouses,
            orders=self.orders, stock=self.stock, default_stock=self.default_stock,
            refund_approval_limit_cents=self.refund_approval_limit_cents,
            constraints=self.constraints, provenance_kind="designer_exploration",
            distribution_hash=distribution.record_hash)


class RemedyDraft(Sealed):
    goal_id: str
    line_alias: str
    quantity: int = Field(ge=1, strict=True,
                          description="Number of purchased units covered by this structured Customer intent.")
    allowed_remedies: Tuple[Literal["refund", "exchange", "reship"], ...] = Field(
        min_length=1,
        description="Customer-authorized outcomes. The Host renders these structured outcomes into the request.",
    )
    required_attributes: Tuple[Attribute, ...] = Field(
        default=(), description="Required outcome attributes, checked against frozen SKU facts.")
    maximum_extra_charge_cents: Optional[int] = Field(
        default=None, ge=0, strict=True,
        description="Maximum authorized extra charge in cents; the Host renders the exact amount.")
    latest_arrival_within_days: Optional[int] = Field(
        default=None, ge=0, strict=True,
        description="Latest permitted arrival in whole days from frozen world time; rendered by the Host.")
    target_stage: Literal["settled", "delivered"] = Field(
        default="settled",
        description=("Use 'settled' for the currently supported Reference proof. 'delivered' means actual receipt, "
                     "which this run can certify only when a natural interaction certifier is explicitly available."),
    )


class InformationDraft(Sealed):
    goal_id: str
    subject_alias: str = Field(
        description=("Alias of an existing order or order line owned by the selected Customer, exactly as listed "
                     "in that Customer's scope. Never use a Customer, SKU, or another Customer's object."),
    )
    fact_kind: Literal["refund_amount", "extra_charge", "arrival_date", "refund_status",
                       "shipment_status", "quantity", "currency"] = Field(
        description="The structured business fact requested; the Host renders its public question.",
    )


class TaskDraft(Sealed):
    task_id: str
    customer_alias: str
    remedy_goals: Tuple[RemedyDraft, ...] = ()
    information_goals: Tuple[InformationDraft, ...] = Field(
        default=(),
        description=("Only use when task_certification_capabilities says a natural interaction certifier is available; "
                     "the bounded native Reference proof certifies remedy goals, not communication goals."),
    )
    parent_training_evidence_hashes: Tuple[str, ...] = ()

    @model_validator(mode="after")
    def unique_goal_ids(self):
        rows = self.remedy_goals + self.information_goals
        if not rows or len({row.goal_id for row in rows}) != len(rows):
            raise ValueError("TaskDraft needs unique business or information goals")
        if any(len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)
               for value in self.parent_training_evidence_hashes):
            raise ValueError("training evidence references must be SHA-256 hashes")
        return self


class TaskCapabilityUnsupported(ValueError):
    """The native proof contract cannot certify a structurally valid goal."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class DesignerTrainingEvidence(Sealed):
    """Curated training-only observations; no held-out or validator-only labels."""

    source_record_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    partition: Literal["train"]
    summary: str = Field(min_length=1)
    observed_events: Tuple[Dict[str, Any], ...] = ()


class DesignerCall(Sealed):
    phase: Literal["world", "task"]
    attempt: int = Field(ge=0, strict=True)
    call: AgentCall
    prompt_hash: str = ""
    submission_schema_hash: str = ""
    capability_manifest_hash: str = ""
    parsed_output_hash: str
    validation_feedback: str = ""


class TaskDesignAttempt(Sealed):
    attempt: int = Field(ge=0, strict=True)
    draft: TaskDraft
    task_hash: str = ""
    certification_hash: str = ""
    status: Literal["COMPILED", "CERTIFIED", "REJECTED", "UNKNOWN", "ERROR"]
    reason: str = ""


class WorldDesignAttempt(Sealed):
    attempt: int = Field(ge=0, strict=True)
    proposal_hash: str = ""
    build_result_hash: str = ""
    status: Literal["SAT", "UNSAT", "UNKNOWN", "ERROR"]
    reason: str = ""


class CandidateConsequenceDiagnostic(Sealed):
    """Researcher-only screen of alternatives excluded by business constraints."""

    protocol: Literal["commerce-candidate-consequence-diagnostic-v1"] = "commerce-candidate-consequence-diagnostic-v1"
    world_hash: str
    task_hash: str
    certificate_hash: str
    goal_id: str
    alternatives_examined: int = Field(ge=0, strict=True)
    feasible_alternatives: int = Field(ge=0, strict=True)
    excluded_condition_counts: Tuple[Tuple[str, int], ...] = ()
    method: Literal["pre_reference_rule_screen_not_execution"] = "pre_reference_rule_screen_not_execution"
    decision_power: Literal["diagnostic_only"] = "diagnostic_only"


class ChallengeQualityVector(Sealed):
    protocol: Literal["commerce-challenge-quality-vector-v2"] = "commerce-challenge-quality-vector-v2"
    world_hash: str
    task_hash: str
    business_certificate_hash: str
    business_valid: bool
    business_witnesses: int = Field(ge=0, strict=True)
    distinct_consequence_classes: int = Field(ge=0, strict=True)
    diagnosability: bool
    diagnostic_evidence_refs: Tuple[str, ...] = ()
    objective_kinds: Tuple[str, ...] = ()
    scenario_structure: str
    task_structure: str
    structural_signature: str
    novelty_status: Literal["EVALUATED_NOVEL", "EVALUATED_SEEN", "NOT_EVALUATED"]
    novel_against_history: Optional[bool] = None
    history_index_hash: str
    candidate_consequence_diagnostics: Tuple[CandidateConsequenceDiagnostic, ...] = ()
    authorized_training_evidence_hash: str
    interaction_certified: bool = False
    interaction_certificate_hash: str = ""
    decision_power: Literal["diagnostic_only"] = "diagnostic_only"
    quality_status: Literal["business_certified_interaction_pending", "fully_interaction_certified",
                            "task_certification_incomplete"]


class ChallengeQualityVectorV1(Sealed):
    """Read-only compatibility model for archived v1 Designer records."""

    protocol: Literal["commerce-challenge-quality-vector-v1"] = "commerce-challenge-quality-vector-v1"
    world_hash: str
    task_hash: str
    business_certificate_hash: str
    business_valid: bool
    business_witnesses: int = Field(ge=0, strict=True)
    distinct_consequence_classes: int = Field(ge=0, strict=True)
    diagnosability: bool
    diagnostic_evidence_refs: Tuple[str, ...] = ()
    objective_kinds: Tuple[str, ...] = ()
    scenario_structure: str
    task_structure: str
    structural_signature: str
    novel_against_history: bool
    authorized_training_evidence_hash: str
    interaction_certified: bool = False
    interaction_certificate_hash: str = ""
    decision_power: Literal["diagnostic_only"] = "diagnostic_only"
    quality_status: Literal["business_certified_interaction_pending", "fully_interaction_certified",
                            "task_certification_incomplete"]


class PostRunQualityVector(Sealed):
    protocol: Literal["commerce-post-run-quality-v1"] = "commerce-post-run-quality-v1"
    pre_run_quality_hash: str
    execution_hash: str
    replay_verified: bool
    strict_task_success: Optional[bool]
    final_goal_reached: Optional[bool]
    oracle_business_errors: int = Field(ge=0, strict=True)
    false_rejections: int = Field(ge=0, strict=True)
    protocol_failures: int = Field(ge=0, strict=True)
    customer_failures: int = Field(ge=0, strict=True)
    transport_failures: int = Field(ge=0, strict=True)
    correct_rejections: int = Field(ge=0, strict=True)
    blocked_illegal_attempts: int = Field(ge=0, strict=True)
    legal_operations: int = Field(ge=0, strict=True)
    distinct_error_classes: Tuple[str, ...] = ()
    model_calls: int = Field(ge=0, strict=True)
    known_tokens: Optional[int] = Field(default=None, ge=0, strict=True)
    decision_power: Literal["diagnostic_only"] = "diagnostic_only"


class DesignerRun(Sealed):
    protocol: Literal["commerce-two-stage-designer-v1"] = "commerce-two-stage-designer-v1"
    run_id: str
    status: Literal["CERTIFIED", "UNKNOWN", "BLOCKED"]
    sampling_slot: SamplingSlot
    distribution: DistributionSpec
    scenario_spec: Optional[ScenarioSpec] = None
    build_result: Optional[BuildResult] = None
    build_attempts: Tuple[BuildResult, ...] = ()
    world_attempts: Tuple[WorldDesignAttempt, ...] = ()
    task_draft: Optional[TaskDraft] = None
    task: Optional[Union[TaskSpecV3, TaskSpecV2, TaskSpec]] = None
    certification: Optional[TaskCertification] = None
    task_attempts: Tuple[TaskDesignAttempt, ...] = ()
    pre_run_quality: Optional[Union[ChallengeQualityVectorV1, ChallengeQualityVector]] = None
    interaction_episode: Optional[EpisodeRecord] = None
    calls: Tuple[DesignerCall, ...] = ()
    validation_errors: Tuple[str, ...] = ()


class DesignerProvider(Protocol):
    def submit(self, phase: str, messages, model_schema): ...


class ProposalError(ValueError):
    """A deterministic proposal/compiler/certifier rejection with a stable code."""

    def __init__(self, code, message):
        self.code = code
        super().__init__(str(message))


def _tool_schema(name, model):
    schema = model.model_json_schema()

    def strip_internal_metadata(node):
        if not isinstance(node, dict):
            return
        node.pop("title", None)
        properties = node.get("properties")
        if isinstance(properties, dict):
            properties.pop("record_hash", None)
        required = node.get("required")
        if isinstance(required, list):
            node["required"] = [name for name in required if name != "record_hash"]
        for value in node.values():
            if isinstance(value, dict):
                strip_internal_metadata(value)
            elif isinstance(value, list):
                for row in value:
                    strip_internal_metadata(row)

    strip_internal_metadata(schema)
    tool = {"type": "function", "function": {"name": name,
        "description": "Submit exactly one structured proposal for this design stage.",
        "parameters": schema}}
    return tool


def _call_provenance(phase, call):
    request = call.request if isinstance(call.request, dict) else {}
    messages = request.get("messages")
    tools = request.get("tools")
    capability_hash = ""
    if phase == "task" and isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                continue
            try:
                payload = json.loads(message["content"])
            except (TypeError, ValueError):
                continue
            context = payload.get("available_context") if isinstance(payload, dict) else None
            manifest = context.get("task_certification_capabilities") if isinstance(context, dict) else None
            if isinstance(manifest, dict):
                capability_hash = digest(manifest)
                break
    return {
        "prompt_hash": digest(messages) if isinstance(messages, list) else "",
        "submission_schema_hash": digest(tools) if isinstance(tools, list) else "",
        "capability_manifest_hash": capability_hash,
    }


def _feedback(exc):
    if isinstance(exc, ServiceError):
        code, message = exc.code, str(exc)
    elif isinstance(exc, ValidationError):
        code = "DESIGNER_SCHEMA_INVALID"
        message = "; ".join(
            "{}: {}".format(".".join(str(part) for part in error["loc"]), error["msg"])
            for error in exc.errors(include_input=False)[:6]
        )
    else:
        code = getattr(exc, "code", None) or "PROPOSAL_INVALID"
        message = str(exc)
    # Validation feedback is a repair cue, not a transcript dump. The full
    # model request/response remains preserved in DesignerCall for audit.
    message = " ".join(message.split())[:700]
    return "{}: {}".format(code, message)


def _is_provider_failure(exc):
    """Provider failures are not proposal feedback and must not be retried as corrections."""
    return isinstance(exc, ServiceError) and exc.code.startswith("PROVIDER_")


def _attempt_error(phase, exc):
    kind = "provider failed" if _is_provider_failure(exc) else "proposal rejected"
    return "{} {}: {}".format(phase, kind, _feedback(exc))


def _submit(provider, phase, messages, model, attempt, calls):
    if hasattr(provider, "last_call"):
        provider.last_call = None
    try:
        return provider.submit(phase, messages, model)
    except Exception as exc:
        call = getattr(exc, "agent_call", None) or getattr(provider, "last_call", None)
        if call is not None:
            calls.append(DesignerCall(phase=phase, attempt=attempt, call=call,
                                      **_call_provenance(phase, call),
                                      parsed_output_hash="",
                                      validation_feedback="" if _is_provider_failure(exc) else _feedback(exc)))
        raise


class OpenAIDesignerProvider:
    """OpenAI-compatible Function Calling adapter; call only when explicitly used."""

    def __init__(self, client: CompletionClient):
        self.client = client
        self.settings: ModelSettings = client.settings.model_copy(deep=True)
        self.last_call: Optional[AgentCall] = None

    def submit(self, phase, messages, model, *, schema_override=None):
        self.last_call = None
        name = "submit_world_constraints" if phase == "world" else "submit_customer_task"
        tool = _tool_schema(name, model)
        if schema_override is not None:
            tool["function"]["parameters"] = schema_override
        request = self.settings.request(messages, [tool])
        # Qwen3 thinking-mode endpoints reject both `required` and forced
        # function objects with HTTP 400. `auto` remains usable there; the
        # local parser below still requires exactly one complete call to this
        # phase's expected function, so provider compatibility does not relax
        # the Designer protocol.
        if "qwen3" in self.settings.model.casefold():
            request["tool_choice"] = "auto"
        else:
            request["tool_choice"] = {"type": "function", "function": {"name": name}}
        request["parallel_tool_calls"] = False
        try:
            raw = self.client.complete(request)
        except Exception as exc:
            code = getattr(exc, "code", None)
            error = {"type": type(exc).__name__,
                     "code": code if isinstance(code, str) else "DESIGNER_REQUEST_FAILED"}
            # Keep bounded provider diagnostics for offline audit. The native
            # OpenAI-compatible client redacts its credential in ServiceError
            # responses; only persist a response after applying that redactor.
            provider_response = getattr(exc, "response", None)
            redactor = getattr(self.client, "_redact", None)
            if isinstance(provider_response, dict) and callable(redactor):
                error["provider_response"] = redactor(provider_response)
            self.last_call = AgentCall(role="designer", request=request,
                                       response={"error": error})
            raise
        self.last_call = AgentCall(role="designer", request=request, response=raw,
                                   prompt_tokens=_usage(raw, "prompt_tokens"),
                                   completion_tokens=_usage(raw, "completion_tokens"))
        message = _one_tool_call(raw)
        call = message["tool_calls"][0]
        if call["function"]["name"] != name:
            raise ServiceError("MODEL_OUTPUT_INVALID", "Designer called the wrong phase submission tool")
        arguments = _json_arguments(call["function"]["arguments"])
        if not arguments:
            raise ServiceError(
                "EMPTY_TOOL_ARGUMENTS",
                "Designer submitted an empty object; provide the required fields for this proposal stage",
            )
        if schema_override is not None:
            from jsonschema import Draft202012Validator, ValidationError as SchemaValidationError
            try:
                Draft202012Validator(schema_override).validate(arguments)
            except SchemaValidationError as exc:
                path = ".".join(str(part) for part in exc.absolute_path)
                raise ServiceError("DESIGNER_SCHEMA_INVALID",
                    "Designer proposal violates this run's schema at {}: {}".format(path, exc.message)) from None
        try:
            output = model.model_validate(arguments)
        except ValidationError as exc:
            details = "; ".join(
                "{}: {}".format(".".join(str(part) for part in error["loc"]), error["msg"])
                for error in exc.errors(include_input=False)
            )
            raise ServiceError("DESIGNER_SCHEMA_INVALID", "Designer proposal does not match the {} schema: {}".format(
                model.__name__, details)) from None
        return output, self.last_call


def task_design_capabilities(spec: ScenarioSpec, build: BuildResult, *,
                             interaction_certifier_available=False,
                             information_goals_certifiable=False,
                             delivery_confirmation_available=False,
                             return_receipt_event_configured=False,
                             policy_review_event_configured=False):
    """Construct the exact capability manifest shared by prompt and compiler."""
    aliases = dict(build.private_aliases)
    scopes = []
    for customer_alias in spec.customer_aliases:
        orders = tuple(row.alias for row in spec.orders if row.customer_alias == customer_alias)
        lines = tuple(line.alias for row in spec.orders if row.customer_alias == customer_alias
                      for line in row.lines)
        scopes.append(CustomerTaskScope(customer_alias=customer_alias,
                                        order_aliases=orders, line_aliases=lines))
    required_aliases = {spec.merchant_alias, *spec.customer_aliases}
    required_aliases.update(row.alias for row in spec.orders)
    required_aliases.update(line.alias for row in spec.orders for line in row.lines)
    required_aliases.update(row.alias for row in spec.skus)
    required_aliases.update(row.alias for row in spec.warehouses)
    if not required_aliases.issubset(aliases):
        raise ValueError("the frozen world alias map is unavailable")
    interaction = bool(interaction_certifier_available)
    information = bool(information_goals_certifiable)
    delivery = bool(delivery_confirmation_available)
    return TaskDesignCapabilities(
        scenario_hash=spec.record_hash,
        world_hash=build.world_hash,
        supported_target_stages=("settled", "delivered") if delivery else ("settled",),
        interaction_certifier_available=interaction,
        information_goals_certifiable=information,
        delivery_confirmation_available=delivery,
        return_receipt_event_configured=bool(return_receipt_event_configured),
        policy_review_event_configured=bool(policy_review_event_configured),
        customer_scopes=tuple(scopes),
    )


def compile_task(draft: TaskDraft, spec: ScenarioSpec, build: BuildResult,
                 distribution: DistributionSpec, allowed_training_hashes=(), *,
                 interaction_certifier_available: Optional[bool] = None,
                 information_goals_certifiable: Optional[bool] = None,
                 delivery_confirmation_available: Optional[bool] = None,
                 capabilities: Optional[TaskDesignCapabilities] = None):
    draft, spec, build, distribution = draft.checked(), spec.checked(), build.checked(), distribution.checked()
    if build.status != "SAT" or not build.world:
        raise ValueError("a SAT frozen world is required before task design")
    aliases = dict(build.private_aliases)
    capabilities = (capabilities.checked() if capabilities is not None else
                    task_design_capabilities(spec, build,
                        interaction_certifier_available=bool(interaction_certifier_available),
                        information_goals_certifiable=information_goals_certifiable,
                        delivery_confirmation_available=bool(delivery_confirmation_available)))
    if capabilities.scenario_hash != spec.record_hash or capabilities.world_hash != build.world_hash:
        raise ValueError("Task capability manifest belongs to a different frozen world")
    if (interaction_certifier_available is not None
            and capabilities.interaction_certifier_available != bool(interaction_certifier_available)):
        raise ValueError("Task capability manifest differs from the configured interaction certifier")
    if (delivery_confirmation_available is not None
            and capabilities.delivery_confirmation_available != bool(delivery_confirmation_available)):
        raise ValueError("Task capability manifest differs from configured delivery confirmation")
    if (information_goals_certifiable is not None
            and capabilities.information_goals_certifiable != bool(information_goals_certifiable)):
        raise ValueError("Task capability manifest differs from configured information certification")
    scopes = {row.customer_alias: row for row in capabilities.customer_scopes}
    customer_scope = scopes.get(draft.customer_alias)
    if customer_scope is None:
        raise ValueError("TaskDraft references a Customer alias outside the declared capability scope")
    if draft.customer_alias not in aliases:
        raise ValueError("TaskDraft references a Customer alias outside the frozen world")
    if not set(draft.parent_training_evidence_hashes).issubset(set(allowed_training_hashes)):
        raise ValueError("TaskDraft references training evidence outside the authorized input set")
    if draft.information_goals and not capabilities.information_goals_certifiable:
        raise TaskCapabilityUnsupported("INFORMATION_CERTIFIER_UNAVAILABLE",
            "information goals require an explicitly enabled natural-runtime interaction certification capability"
        )
    if draft.remedy_goals and not capabilities.native_reference_certifies and not capabilities.interaction_certifier_available:
        raise TaskCapabilityUnsupported("TASK_CERTIFIER_UNAVAILABLE",
                                        "this run has no configured certifier for remedy goals")
    remedies, information = [], []
    for row in draft.remedy_goals:
        unsupported_remedies = set(row.allowed_remedies) - set(capabilities.supported_remedies)
        if unsupported_remedies:
            raise TaskCapabilityUnsupported(
                "REMEDY_UNSUPPORTED", "unsupported remedy method(s): {}".format(
                    ", ".join(sorted(unsupported_remedies))))
        if row.line_alias not in customer_scope.line_aliases:
            raise ValueError("TaskDraft references an unknown order-line alias")
        original_id = aliases[row.line_alias]
        if row.target_stage not in capabilities.supported_target_stages:
            raise TaskCapabilityUnsupported("TARGET_STAGE_UNSUPPORTED",
                "target_stage '{}' is outside this run's certified capabilities; supported stages are {}".format(
                    row.target_stage, ", ".join(capabilities.supported_target_stages))
            )
        line = build.world.lines[original_id]
        original_sku = build.world.skus[line.sku_id]
        if row.quantity > line.quantity:
            raise ValueError("TaskDraft quantity exceeds the purchased line quantity")
        if (original_sku.return_required and not capabilities.return_receipt_event_configured):
            raise TaskCapabilityUnsupported(
                "RETURN_RECEIPT_UNAVAILABLE",
                "this task requires a return-receipt event that is not configured for certification")
        if "refund" in row.allowed_remedies:
            credit = unit_credit(line, tuple(range(row.quantity)))
            merchant_id = build.world.orders[line.order_id].merchant_id
            approval_limit = build.world.merchants[merchant_id].refund_approval_limit_cents
            if credit > approval_limit and not capabilities.policy_review_event_configured:
                raise TaskCapabilityUnsupported(
                    "POLICY_REVIEW_UNAVAILABLE",
                    "this refund requires a policy-review event that is not configured for certification")
        arrival = (build.world.clock + timedelta(days=row.latest_arrival_within_days)
                   if row.latest_arrival_within_days is not None else None)
        compiled_goal = RemedyGoal(
            goal_id=row.goal_id, line_id=original_id, quantity=row.quantity,
            allowed_remedies=row.allowed_remedies, required_attributes=row.required_attributes,
            maximum_extra_charge_cents=row.maximum_extra_charge_cents,
            latest_arrival=arrival, target_stage=row.target_stage)
        remedies.append(compiled_goal)
    for row in draft.information_goals:
        if row.subject_alias not in aliases:
            raise ValueError("TaskDraft references an unknown public order or line alias")
        if row.subject_alias not in set(customer_scope.order_aliases) | set(customer_scope.line_aliases):
            raise ValueError(
                "information subject must be the selected Customer's owned order or order-line alias"
            )
        information.append(StructuredInformationObjective(
            goal_id=row.goal_id, subject_id=aliases[row.subject_alias], fact_kind=row.fact_kind))
    if any(len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)
           for value in draft.parent_training_evidence_hashes):
        raise ValueError("invalid training evidence hash")
    customer_id = aliases[draft.customer_alias]
    customer_request = render_customer_request(build.world, customer_id, tuple(remedies), tuple(information))
    request_bundle = CustomerRequestBundle(
        customer_id=customer_id,
        goal_ids=tuple(row.goal_id for row in tuple(remedies) + tuple(information)),
        intent_hash=request_intent_hash(customer_id, tuple(remedies), tuple(information)),
        request_hash=digest(customer_request))
    task = TaskSpecV3(
        task_id=draft.task_id, world_hash=build.world_hash, customer_id=aliases[draft.customer_alias],
        customer_request=customer_request, remedy_goals=tuple(remedies),
        information_goals=tuple(information), partition=distribution.split,
        source_distribution_hash=distribution.record_hash,
        parent_training_evidence_hashes=draft.parent_training_evidence_hashes,
        request_bundle=request_bundle)
    return check_task_scope(build.world, task)


def _structure_signatures(spec, world, task, certificate):
    aliases = {}
    aliases.update({row: "customer" for row in spec.customer_aliases})
    aliases.update({row.alias: "sku" for row in spec.skus})
    aliases.update({row.alias: "warehouse" for row in spec.warehouses})
    aliases.update({row.alias: "order" for row in spec.orders})
    aliases.update({line.alias: "line" for order in spec.orders for line in order.lines})

    def expression_shape(expression):
        return sorted((aliases.get(row.variable, "unknown"), row.field,
                       1 if row.coefficient > 0 else -1, abs(row.coefficient),
                       bool(row.warehouse)) for row in expression.terms)

    world_structure = {
        "customers": len(spec.customer_aliases), "skus": len(spec.skus),
        "families": sorted({row.family for row in spec.skus}),
        "warehouses": len(spec.warehouses), "orders": len(spec.orders),
        "lines": sum(len(row.lines) for row in spec.orders),
        "stock_constraints": len(spec.stock),
        "constraint_shapes": sorted((row.operator, expression_shape(row.left), expression_shape(row.right))
                                    for row in spec.constraints),
    }
    task_structure = {
        "remedy_goals": sorted((tuple(sorted(row.allowed_remedies)), row.quantity,
                                 tuple(sorted((a.name, a.value) for a in row.required_attributes)),
                                 row.maximum_extra_charge_cents is not None,
                                 row.latest_arrival is not None, row.target_stage)
                                for row in task.remedy_goals),
        "information_goals": sorted(row.fact_kind for row in task.information_goals),
        "distinct_target_lines": len({row.line_id for row in task.remedy_goals}),
        "mixed_objectives": bool(task.remedy_goals and task.information_goals),
    }
    def candidate_outcome(candidate):
        goal = next((row for row in task.remedy_goals if row.goal_id == candidate.goal_id), None)
        if goal is None:
            return {"remedy": candidate.remedy, "goal_kind": "unknown"}, 0, 0
        line = world.lines[goal.line_id]
        original = world.skus[line.sku_id]
        credit = unit_credit(line, tuple(range(goal.quantity)))
        if candidate.remedy == "refund":
            return {"remedy": "refund", "return_required": original.return_required}, -credit, 0
        target = world.skus[candidate.target_sku_id]
        warehouse = world.warehouses[candidate.warehouse_id]
        inventory = next(row for row in world.inventory.values()
                         if row.sku_id == target.id and row.warehouse_id == warehouse.id)
        route = next(row for row in world.routes.values() if row.warehouse_id == warehouse.id)
        delta = target.price_cents * goal.quantity + route.fee_cents - credit
        if candidate.remedy == "reship":
            delta = 0
        attrs_match = all(target.attributes.get(row.name) == row.value for row in goal.required_attributes)
        within_budget = (goal.maximum_extra_charge_cents is None
                         or max(delta, 0) <= goal.maximum_extra_charge_cents)
        within_deadline = (goal.latest_arrival is None
                           or world.clock + timedelta(days=route.days) <= goal.latest_arrival)
        return ({"remedy": candidate.remedy, "compatible": attrs_match,
                 "charge_relation": "within" if within_budget else "over",
                 "deadline_relation": "within" if within_deadline else "late",
                 "stock_relation": "sufficient" if inventory.available >= goal.quantity else "short",
                 "return_required": original.return_required}, delta, route.days * 86400)

    all_rows = []
    valid_rows = []
    for attempt in certificate.attempts:
        row_features = [candidate_outcome(candidate) for candidate in attempt.candidates]
        all_rows.append((attempt.status, tuple(sorted(item[0]["remedy"] for item in row_features)),
                         tuple(sorted(digest(item[0]) for item in row_features))))
        if attempt.status == "WITNESS":
            valid_rows.extend(row_features)
    costs = sorted({row[1] for row in all_rows_raw(certificate, candidate_outcome)})
    delays = sorted({row[2] for row in all_rows_raw(certificate, candidate_outcome)})
    consequence_classes = set()
    for consequence, money, delay in valid_rows:
        normalized = {**consequence, "relative_cost_rank": costs.index(money),
                      "relative_delay_rank": delays.index(delay)}
        consequence_classes.add(digest(normalized))
    world_structure["candidate_outcome_structure"] = sorted(all_rows)
    # IDs and absolute prices/times are deliberately absent from signatures.
    scenario_sig, task_sig = digest(world_structure), digest(task_structure)
    return (world_structure, task_structure, scenario_sig, task_sig,
            digest({"world": scenario_sig, "task": task_sig}), len(consequence_classes))


def all_rows_raw(certificate, candidate_outcome):
    for attempt in certificate.attempts:
        for candidate in attempt.candidates:
            yield candidate_outcome(candidate)


def _quality(spec, world, task, certificate, history_signatures, training_hashes,
             proof_customer, interaction_episode=None):
    ws, ts, world_sig, task_sig, combined, consequence_count = _structure_signatures(
        spec, world, task, certificate)
    witnesses = [row for row in certificate.attempts if row.status == "WITNESS"]
    interaction_ok = bool(interaction_episode and interaction_episode.replay_verified
                          and interaction_episode.outcome.strict_task_success is True)
    proof_refs = [certificate.record_hash] + [row.record_hash for row in witnesses]
    if interaction_episode:
        proof_refs.append(interaction_episode.record_hash)
    business_valid = certificate.business_reachability_certified or interaction_ok
    objectives = tuple(sorted(("remedy" for _ in task.remedy_goals),)) + tuple(
        sorted("information:" + row.fact_kind for row in task.information_goals))
    history_available = history_signatures is not None
    history = tuple(sorted(set(history_signatures or ())))
    novelty_status = ("NOT_EVALUATED" if not history_available else
                      "EVALUATED_NOVEL" if combined not in history else "EVALUATED_SEEN")
    candidate_diagnostics = tuple(CandidateConsequenceDiagnostic(
        world_hash=certificate.world_hash, task_hash=certificate.task_hash,
        certificate_hash=certificate.record_hash, **row)
        for row in diagnose_candidate_alternatives(world, task, proof_customer))
    return ChallengeQualityVector(
        world_hash=certificate.world_hash, task_hash=certificate.task_hash,
        business_certificate_hash=certificate.record_hash,
        business_valid=business_valid,
        business_witnesses=len(witnesses), distinct_consequence_classes=consequence_count,
        diagnosability=bool(witnesses or interaction_ok), diagnostic_evidence_refs=tuple(proof_refs),
        objective_kinds=objectives,
        scenario_structure=json.dumps(ws, sort_keys=True), task_structure=json.dumps(ts, sort_keys=True),
        structural_signature=combined, novelty_status=novelty_status,
        novel_against_history=(combined not in history) if history_available else None,
        history_index_hash=digest({"available": history_available, "signatures": history}),
        candidate_consequence_diagnostics=candidate_diagnostics,
        authorized_training_evidence_hash=digest(tuple(sorted(training_hashes))),
        interaction_certified=interaction_ok,
        interaction_certificate_hash=interaction_episode.record_hash if interaction_ok else "",
        quality_status=("fully_interaction_certified" if interaction_ok
                        else "business_certified_interaction_pending" if certificate.business_reachability_certified
                        else "task_certification_incomplete"))


def _chat(phase, content, distribution, slot, errors=()):
    active = next((row for row in distribution.strata if row.stratum_id == slot.stratum_id), None)
    if active is None:
        raise ValueError("sampling slot names a stratum outside its frozen distribution")
    active_stratum = {"stratum_id": active.stratum_id, "design_focus": list(active.contexts),
                      "dependencies": list(active.dependencies)}
    if phase == "task":
        system = (
            "You are the Usami Commerce Task Designer. The business world is already frozen; do not edit it. "
            "Choose a Customer and define one or more structured business intents over aliases in that Customer's "
            "published scope. Submit only aliases, outcome types and numeric/business constraints in TaskDraft; "
            "do not write customer_request, natural-language evidence spans, product mentions, internal IDs, "
            "answers or proof witnesses. The trusted Host renders the Customer request from these same structured "
            "intents and binds that text to the task record. Thus every scored goal is authorized by construction, "
            "without matching keywords or substrings. Follow the active stratum's focus, use only supported "
            "capabilities, and do not invent unavailable events. Information goals must reference an owned order or "
            "line alias. On correction, fix only the cited schema/contract issue. Submit one complete TaskDraft."
        )
    elif phase == "world":
        system = (
            "You are the Usami Commerce World Designer. Propose a ScenarioDraft for this pre-registered sampling "
            "slot and active business stratum. Create meaningful, observable business decision conditions while "
            "keeping values and relationships internally consistent; do not add unrelated entities just to enlarge "
            "the world. SKU family means a narrow after-sales replacement-compatibility group, not a broad retail "
            "category: laptop and mouse are different families even if both are electronics; compatible laptop "
            "configurations may share a family. Prices, paid amounts, quantities, stock, fees and durations use "
            "their declared integer units. Use registered fields and native business assumptions only. Do not create "
            "a Task, modify policy, state Oracle labels, or provide a Reference solution. Submit exactly one "
            "ScenarioDraft using the registered tool."
        )
    else:
        raise ValueError("unknown Designer phase")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps({
            "phase": phase,
            "distribution": {"id": distribution.distribution_id, "version": distribution.version,
                             "hash": distribution.record_hash, "split": distribution.split,
                             "population_scope": distribution.population_scope, "claim": distribution.claim},
            "active_stratum": active_stratum,
            "pre_registered_slot": slot.model_dump(mode="json"),
            "available_context": content,
            "validator_feedback": list(errors)}, ensure_ascii=False)},
    ]


class WorldDesignRun(Sealed):
    """The existing World phase, reusable without generating a TaskDraft."""
    status: Literal["SAT", "BLOCKED"]
    sampling_slot: SamplingSlot
    distribution: DistributionSpec
    scenario_spec: Optional[ScenarioSpec] = None
    build_result: Optional[BuildResult] = None
    build_attempts: Tuple[BuildResult, ...] = ()
    world_attempts: Tuple[WorldDesignAttempt, ...] = ()
    calls: Tuple[DesignerCall, ...] = ()
    validation_errors: Tuple[str, ...] = ()


def design_world_slot(provider: DesignerProvider, distribution: DistributionSpec, slot: SamplingSlot, *,
                      build_budget: Optional[BuildBudget] = None, max_attempts: int = 3) -> WorldDesignRun:
    """Generate one independently registered World using the native Designer/Builder."""
    distribution, slot = distribution.checked(), slot.checked()
    if slot.distribution_hash != distribution.record_hash:
        raise ValueError("pre-registered sampling slot belongs to another distribution")
    if not 1 <= max_attempts <= 3:
        raise ValueError("Designer correction budget must be between one and three attempts")
    build_budget = (build_budget or BuildBudget()).checked()
    calls, errors, scenario, build_attempts, world_attempts = [], [], None, [], []
    build = None
    for attempt in range(max_attempts):
        draft = None
        candidate_build = None
        content = {"required_provenance": "compiler-attached", "supported_fields": list(ScenarioDraft.model_fields.keys()),
                   "attempt_seed": slot.seed}
        try:
            draft, call = _submit(provider, "world", _chat("world", content, distribution, slot, errors),
                                  ScenarioDraft, attempt, calls)
            calls.append(DesignerCall(phase="world", attempt=attempt, call=call,
                                      **_call_provenance("world", call),
                                      parsed_output_hash=draft.record_hash))
            scenario = draft.compile(distribution)
            candidate_build = build_world(scenario, slot.seed, build_budget)
            build_attempts.append(candidate_build)
            if candidate_build.status != "SAT":
                feedback = "WORLD_BUILDER_{}: {}".format(candidate_build.status, candidate_build.reason)
                errors.append(feedback)
                world_attempts.append(WorldDesignAttempt(attempt=attempt, proposal_hash=draft.record_hash,
                    build_result_hash=candidate_build.record_hash, status=candidate_build.status, reason=feedback))
                continue
            verify_build(scenario, candidate_build)
            world_attempts.append(WorldDesignAttempt(attempt=attempt, proposal_hash=draft.record_hash,
                build_result_hash=candidate_build.record_hash, status="SAT", reason="native build certificate passed"))
            build = candidate_build
            break
        except Exception as exc:
            feedback = _attempt_error("world", exc)
            errors.append(feedback)
            world_attempts.append(WorldDesignAttempt(attempt=attempt,
                proposal_hash=draft.record_hash if draft else "",
                build_result_hash=candidate_build.record_hash if candidate_build else "",
                status="ERROR", reason=feedback))
            if calls and not _is_provider_failure(exc):
                last = calls[-1]
                if last.phase == "world" and last.attempt == attempt:
                    calls[-1] = DesignerCall(phase=last.phase, attempt=last.attempt, call=last.call,
                                             prompt_hash=last.prompt_hash,
                                             submission_schema_hash=last.submission_schema_hash,
                                             capability_manifest_hash=last.capability_manifest_hash,
                                             parsed_output_hash=last.parsed_output_hash,
                                             validation_feedback=feedback)
            if _is_provider_failure(exc):
                break
    return WorldDesignRun(
        status="SAT" if build is not None else "BLOCKED", sampling_slot=slot,
        distribution=distribution, scenario_spec=scenario,
        build_result=build if build is not None else candidate_build,
        build_attempts=tuple(build_attempts), world_attempts=tuple(world_attempts),
        calls=tuple(calls), validation_errors=tuple(errors))


def design_and_certify(provider: DesignerProvider, distribution: DistributionSpec, slot: SamplingSlot, *,
                       build_budget: Optional[BuildBudget] = None,
                       proof_budget: Optional[ProofBudget] = None,
                       allowed_training_hashes=(), history_signatures: Optional[Tuple[str, ...]] = None,
                       customer_scenario_factory: Optional[Callable[[Any, TaskSpec], ProofCustomerScenario]] = None,
                       training_evidence: Tuple[DesignerTrainingEvidence, ...] = (),
                       interaction_certifier: Optional[Callable[[World, TaskSpec, ProofCustomerScenario], EpisodeRecord]] = None,
                       information_goals_certifiable=False,
                       delivery_confirmation_available=False,
                       return_receipt_event_configured=False,
                       policy_review_event_configured=False,
                       max_attempts: int = 3) -> DesignerRun:
    """Run the two LLM phases with bounded compiler feedback and native proof.

    Reference witnesses, Oracle data and private customer proof policy are never
    added to a Designer prompt. Correction attempts receive rejection reasons only.
    """
    distribution, slot = distribution.checked(), slot.checked()
    if slot.distribution_hash != distribution.record_hash:
        raise ValueError("pre-registered sampling slot belongs to another distribution")
    if customer_scenario_factory is None:
        raise ValueError("a trusted, private CustomerScenario factory is required before any Designer model call")
    if max_attempts < 1 or max_attempts > 3:
        raise ValueError("Designer correction budget must be between one and three attempts")
    evidence = tuple(row.checked() for row in training_evidence)
    allowed_training_hashes = tuple(sorted(set(allowed_training_hashes)))
    if any(row.source_record_hash not in set(allowed_training_hashes) for row in evidence):
        raise ValueError("Designer context contains training evidence outside the authorized hash set")
    build_budget, proof_budget = (build_budget or BuildBudget()).checked(), (proof_budget or ProofBudget()).checked()
    world_run = design_world_slot(provider, distribution, slot,
                                  build_budget=build_budget, max_attempts=max_attempts)
    scenario, candidate_build = world_run.scenario_spec, world_run.build_result
    build = candidate_build if world_run.status == "SAT" else None
    calls, errors = list(world_run.calls), list(world_run.validation_errors)
    build_attempts, world_attempts = world_run.build_attempts, world_run.world_attempts
    if not scenario or not build or build.status != "SAT":
        # Preserve the last available attempt instead of manufacturing a world.
        return DesignerRun(
            run_id="designer-" + digest([slot.record_hash, tuple(x.record_hash for x in calls)])[:20],
            status="BLOCKED", sampling_slot=slot, distribution=distribution,
            scenario_spec=scenario, build_result=candidate_build, build_attempts=tuple(build_attempts),
            world_attempts=tuple(world_attempts),
            calls=tuple(calls), validation_errors=tuple(errors or ["no parseable world proposal"]))

    aliases = dict(build.private_aliases)
    capabilities = task_design_capabilities(
        scenario, build, interaction_certifier_available=interaction_certifier is not None,
        information_goals_certifiable=information_goals_certifiable,
        delivery_confirmation_available=delivery_confirmation_available,
        return_receipt_event_configured=return_receipt_event_configured,
        policy_review_event_configured=policy_review_event_configured)

    def public_product_description(sku):
        attributes = {row.name: row.value for row in sku.attributes}
        preferred = ("brand", "product", "product_name", "model", "name")
        details = [attributes[name] for name in preferred if attributes.get(name)]
        return " ".join(dict.fromkeys(details)) or sku.alias

    world_context = {
        "world_hash": build.world_hash,
        "customers": [{"alias": alias} for alias in scenario.customer_aliases],
        "orders": [{"alias": order.alias, "customer_alias": order.customer_alias,
                    "delivered_days_ago": (build.world.clock - build.world.orders[aliases[order.alias]].delivered_at).days,
                    "lines": [{"alias": line.alias,
                               "original_sku_alias": line.original_sku,
                               "quantity": build.world.lines[aliases[line.alias]].quantity,
                               "line_paid_cents": build.world.lines[aliases[line.alias]].allocated_paid_cents,
                               "issue": line.issue, "evidence_valid": line.evidence_valid}
                              for line in order.lines]} for order in scenario.orders],
        "skus": [{"alias": sku.alias, "product_description": public_product_description(sku),
                  "replacement_compatibility_family": sku.family,
                  "price_cents": build.world.skus[aliases[sku.alias]].price_cents,
                  "return_days": build.world.skus[aliases[sku.alias]].return_days,
                  "return_required": build.world.skus[aliases[sku.alias]].return_required,
                  "attributes": {row.name: row.value for row in sku.attributes}}
                 for sku in scenario.skus],
        "warehouses": [{"alias": row.alias,
                         "delivery_days": next(r.days for r in build.world.routes.values()
                                               if r.warehouse_id == aliases[row.alias]),
                         "delivery_fee_cents": next(r.fee_cents for r in build.world.routes.values()
                                                    if r.warehouse_id == aliases[row.alias])}
                        for row in scenario.warehouses],
        "inventory": [{"sku_alias": sku.alias, "warehouse_alias": wh.alias,
                       "available": next(inv.available for inv in build.world.inventory.values()
                                         if inv.sku_id == aliases[sku.alias]
                                         and inv.warehouse_id == aliases[wh.alias])}
                      for sku in scenario.skus for wh in scenario.warehouses],
        "clock": build.world.clock.isoformat(),
        "merchant_policy": {"refund_approval_limit_cents": scenario.refund_approval_limit_cents,
                            "currency_unit": "CNY_cents"},
        "task_certification_capabilities": capabilities.model_dump(mode="json"),
        "reference_policy": "independent; not included in this context",
    }
    errors, task_attempts = [], []
    evidence_context = [row.model_dump(mode="json") for row in evidence]
    task_context = {**world_context, "authorized_training_evidence": evidence_context}
    draft_task, task, certificate, interaction_episode = None, None, None, None
    for attempt in range(max_attempts):
        draft_task, task, certificate, interaction_episode = None, None, None, None
        try:
            draft_task, call = _submit(provider, "task", _chat("task", task_context, distribution, slot, errors),
                                       TaskDraft, attempt, calls)
            calls.append(DesignerCall(phase="task", attempt=attempt, call=call,
                                      **_call_provenance("task", call),
                                      parsed_output_hash=draft_task.record_hash))
            if not set(draft_task.parent_training_evidence_hashes).issubset(set(allowed_training_hashes)):
                raise ValueError("TaskDraft cites training evidence outside the authorized context")
            try:
                task = compile_task(draft_task, scenario, build, distribution, allowed_training_hashes,
                                    interaction_certifier_available=interaction_certifier is not None,
                                    information_goals_certifiable=capabilities.information_goals_certifiable,
                                    delivery_confirmation_available=capabilities.delivery_confirmation_available,
                                    capabilities=capabilities)
            except Exception as exc:
                raise ProposalError("TASK_COMPILE_INVALID", str(exc)) from None
            proof_customer = customer_scenario_factory(build.world, task)
            if proof_customer.customer_id != task.customer_id:
                raise ProposalError("CAPABILITY_MANIFEST_MISMATCH",
                                    "trusted Customer proof policy belongs to a different identity")
            for goal in task.remedy_goals:
                line = build.world.lines[goal.line_id]
                sku = build.world.skus[line.sku_id]
                if sku.return_required and capabilities.return_receipt_event_configured and (
                        not proof_customer.return_pickup_slot or proof_customer.return_received_after_days is None):
                    raise ProposalError("CAPABILITY_MANIFEST_MISMATCH",
                        "return-receipt capability was declared but the frozen Customer plan has no receipt event")
                merchant_id = build.world.orders[line.order_id].merchant_id
                credit = unit_credit(line, tuple(range(goal.quantity)))
                if ("refund" in goal.allowed_remedies
                        and credit > build.world.merchants[merchant_id].refund_approval_limit_cents
                        and capabilities.policy_review_event_configured
                        and not any(row.line_id == line.id for row in proof_customer.reviews)):
                    raise ProposalError("CAPABILITY_MANIFEST_MISMATCH",
                        "policy-review capability was declared but the frozen Customer plan has no review event")
            certificate = certify_task(build.world, task, proof_customer, proof_budget, minimum_witnesses=1)
            if certificate.status == "CERTIFIED":
                verify_task_certificate(build.world, task, proof_customer, certificate)
            if interaction_certifier is not None:
                interaction_episode = interaction_certifier(build.world.clone(), task, proof_customer)
                if not isinstance(interaction_episode, EpisodeRecord):
                    raise ValueError("interaction certifier must return a sealed EpisodeRecord")
                interaction_replay = replay_episode(interaction_episode)
                if (interaction_episode.initial_world.state_hash != build.world_hash
                        or interaction_episode.task.record_hash != task.record_hash
                        or not interaction_replay.verified):
                    raise ValueError("interaction Episode is not bound to the frozen task/world or fails Replay")
            interaction_ok = bool(interaction_episode and interaction_episode.outcome.strict_task_success is True
                                   and interaction_episode.replay_verified)
            if certificate.status == "CERTIFIED" or interaction_ok:
                task_attempts.append(TaskDesignAttempt(attempt=attempt, draft=draft_task,
                    task_hash=task.record_hash, certification_hash=certificate.record_hash,
                    status="CERTIFIED", reason=("native Reference certificate passed" if certificate.status == "CERTIFIED"
                                                  else "independent interaction Episode passed strict evaluation and Replay")))
                break
            task_attempts.append(TaskDesignAttempt(attempt=attempt, draft=draft_task,
                task_hash=task.record_hash, certification_hash=certificate.record_hash,
                status="UNKNOWN" if certificate.status == "UNKNOWN" else "REJECTED",
                reason=", ".join(certificate.blockers)))
            raise ProposalError("REFERENCE_{}".format(certificate.status),
                                ", ".join(certificate.blockers) or "no certified witness")
        except Exception as exc:
            provider_failure = _is_provider_failure(exc)
            feedback = (_attempt_error("task", exc) if provider_failure else
                        "task proposal rejected or uncertified: " + _feedback(exc))
            errors.append(feedback)
            if draft_task is not None and (not task_attempts or task_attempts[-1].draft.record_hash != draft_task.record_hash):
                task_attempts.append(TaskDesignAttempt(attempt=attempt, draft=draft_task,
                    task_hash=task.record_hash if task else "", certification_hash=certificate.record_hash if certificate else "",
                    status="ERROR" if provider_failure else "REJECTED", reason=feedback))
            if calls and calls[-1].phase == "task" and not provider_failure:
                last = calls[-1]
                if last.attempt == attempt:
                    calls[-1] = DesignerCall(phase=last.phase, attempt=last.attempt, call=last.call,
                                             prompt_hash=last.prompt_hash,
                                             submission_schema_hash=last.submission_schema_hash,
                                             capability_manifest_hash=last.capability_manifest_hash,
                                             parsed_output_hash=last.parsed_output_hash,
                                             validation_feedback=feedback)
            if provider_failure:
                break
            if isinstance(exc, ProposalError) and exc.code == "CAPABILITY_MANIFEST_MISMATCH":
                break
            if certificate is not None and certificate.status == "UNKNOWN":
                # UNKNOWN denotes an incomplete proof boundary or exhausted
                # deterministic budget; a replacement proposal cannot make
                # this configured certifier more decisive.
                break
    if task is None or certificate is None or certificate.status != "CERTIFIED":
        # A pure communication task can be fully proved by the independent
        # interaction Runner even though the bounded transaction certifier marks
        # it UNKNOWN. Preserve both records and the distinction.
        interaction_ok = bool(interaction_episode and interaction_episode.outcome.strict_task_success is True
                              and interaction_episode.replay_verified)
        if task is None or certificate is None or not interaction_ok:
            status = "UNKNOWN" if certificate is not None and certificate.status == "UNKNOWN" else "BLOCKED"
            return DesignerRun(run_id="designer-" + digest([slot.record_hash, tuple(x.record_hash for x in calls)])[:20],
                status=status, sampling_slot=slot, distribution=distribution, scenario_spec=scenario,
                build_result=build, build_attempts=tuple(build_attempts), world_attempts=tuple(world_attempts),
                task_draft=draft_task, task=task, certification=certificate, interaction_episode=interaction_episode,
                task_attempts=tuple(task_attempts), calls=tuple(calls), validation_errors=tuple(errors))
    quality = _quality(scenario, build.world, task, certificate, history_signatures,
                       draft_task.parent_training_evidence_hashes, proof_customer, interaction_episode)
    return DesignerRun(
        run_id="designer-" + digest([slot.record_hash, scenario.record_hash, task.record_hash])[:20],
        status="CERTIFIED", sampling_slot=slot, distribution=distribution, scenario_spec=scenario,
        build_result=build, build_attempts=tuple(build_attempts), world_attempts=tuple(world_attempts),
        task_draft=draft_task, task=task, certification=certificate, interaction_episode=interaction_episode,
        task_attempts=tuple(task_attempts), pre_run_quality=quality, calls=tuple(calls), validation_errors=tuple(errors))


def post_run_quality(pre_run: ChallengeQualityVector, episode):
    """Return a diagnostic vector only; it never accepts or rejects a Designer Skill."""
    violations = []
    false_rejections = 0
    correct_rejections = 0
    blocked_illegal = 0
    legal_operations = 0
    transport_failures = 0
    for tool in episode.tool_executions:
        for row in tool.oracle.get("operations", []):
            violations.extend(row.get("violations", []))
            false_rejections += row.get("classification") == "false_rejection"
            correct_rejections += (row.get("classification") == "rejected_invalid_attempt"
                                   and row.get("operation_correct") is True)
            blocked_illegal += row.get("classification") == "rejected_invalid_attempt"
            legal_operations += row.get("classification") == "legal_execution"
    for call in episode.model_calls:
        error = call.response.get("error", {}) if isinstance(call.response, dict) else {}
        transport_failures += int(isinstance(error, dict) and str(error.get("code", "")).startswith((
            "PROVIDER_", "API_", "HTTP_")))
    return PostRunQualityVector(
        pre_run_quality_hash=pre_run.record_hash, execution_hash=episode.record_hash,
        replay_verified=episode.replay_verified, strict_task_success=episode.outcome.strict_task_success,
        final_goal_reached=episode.outcome.final_goal_reached,
        oracle_business_errors=sum(bool(row.get("success") and row.get("classification") == "successful_business_violation")
                                   for tool in episode.tool_executions for row in tool.oracle.get("operations", [])),
        false_rejections=false_rejections,
        protocol_failures=int(episode.status in ("service_protocol_error", "customer_protocol_error")),
        customer_failures=int(episode.status == "customer_protocol_error"),
        transport_failures=transport_failures, correct_rejections=correct_rejections,
        blocked_illegal_attempts=blocked_illegal, legal_operations=legal_operations,
        distinct_error_classes=tuple(sorted(set(violations))),
        model_calls=len(episode.model_calls), known_tokens=episode.total_tokens)
