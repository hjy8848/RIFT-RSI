"""Optional OpenAI-compatible Customer/Service decision adapters.

No provider is contacted on import. Callers must explicitly construct these
adapters with an existing CompletionClient and invoke the episode Runner.
"""
import json
from typing import Any, Dict, List, Optional, Protocol, Tuple

from pydantic import Field

from usami.service.client import CompletionClient
from usami.service.models import ModelSettings, ServiceError, strict_json
from usami_v2.commerce.models import Operation, PARAMETERS
from usami_v2.runtime.customer import CustomerDecision, CustomerProfile, OfferPresentation
from usami_v2.contracts import Sealed


class AgentMessage(Sealed):
    role: str
    content: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Tuple[Dict[str, Any], ...] = ()


class CustomerPrompt(Sealed):
    public_request: str
    last_service_message: str = ""
    offer: Optional[OfferPresentation] = None
    conversation: Tuple[AgentMessage, ...] = ()


class ServicePrompt(Sealed):
    customer_id: str
    customer_request: str
    conversation: Tuple[AgentMessage, ...] = ()
    # Public operational context required to route the authenticated caller's
    # queries. These identifiers do not reveal hidden task goals or witnesses.
    available_merchant_scope_ids: Tuple[str, ...] = ()
    # These are public tool results already delivered to the Service.
    observed_subject_ids: Tuple[str, ...] = ()


class AgentCall(Sealed):
    role: str
    request: Dict[str, Any]
    response: Dict[str, Any]
    prompt_tokens: Optional[int] = Field(default=None, ge=0, strict=True)
    completion_tokens: Optional[int] = Field(default=None, ge=0, strict=True)


class ServiceAction(Sealed):
    kind: str
    operation: Optional[Operation] = None
    message: str = ""
    present_quote_id: Optional[str] = None
    present_for: str = "commit_accepted_quote"
    subject_ids: Tuple[str, ...] = ()
    call: Optional[AgentCall] = None
    call_id: str = ""

    @classmethod
    def from_tool_message(cls, raw, customer_id, observed_subject_ids=()):
        message = _one_tool_call(raw)
        call = message["tool_calls"][0]
        name, args = call["function"]["name"], _json_arguments(call["function"]["arguments"])
        completion = AgentCall(
            role="service", request={}, response=raw,
            prompt_tokens=_usage(raw, "prompt_tokens"), completion_tokens=_usage(raw, "completion_tokens"))
        if name == "respond_to_customer":
            if not isinstance(args.get("message"), str) or not args["message"].strip():
                raise ServiceError("MODEL_OUTPUT_INVALID", "Service reply must contain nonempty message text")
            subjects = tuple(args.get("subject_ids") or ())
            if not set(subjects).issubset(set(observed_subject_ids)):
                raise ServiceError("MODEL_OUTPUT_INVALID", "Service reply references an object it never observed")
            return cls(kind="reply", message=args["message"], present_quote_id=args.get("present_quote_id"),
                       present_for=args.get("present_for") or "commit_accepted_quote", subject_ids=subjects,
                       call=completion, call_id=call["id"])
        if name not in PARAMETERS:
            raise ServiceError("MODEL_OUTPUT_INVALID", "Service selected an unregistered commerce tool")
        try:
            operation = Operation(name=name, scope_id=args["scope_id"], resource_id=args["resource_id"],
                                  params=args.get("params", {}))
        except (KeyError, TypeError, ValueError):
            raise ServiceError("MODEL_OUTPUT_INVALID", "Service tool arguments do not match the native schema") from None
        # A Request is created only by the Runner for this exact tool call; the
        # Reference still independently checks identity, delegation and consent.
        return cls(kind="operation", operation=operation, call=completion, call_id=call["id"])


def _usage(raw, name):
    usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
    value = usage.get(name) if isinstance(usage, dict) else None
    return value if type(value) is int and value >= 0 else None


def _json_arguments(text):
    try:
        value = strict_json(text)
    except (TypeError, ValueError):
        raise ServiceError("MODEL_OUTPUT_INVALID", "Tool arguments are not valid JSON") from None
    if not isinstance(value, dict):
        raise ServiceError("MODEL_OUTPUT_INVALID", "Tool arguments must be a JSON object")
    return value


def _safe_failure(role, request, exc):
    """Keep the attempted request and a credential-safe failure code, never headers."""
    code = getattr(exc, "code", None)
    if not isinstance(code, str) or not code:
        code = "AGENT_REQUEST_FAILED"
    return AgentCall(role=role, request=request,
                     response={"error": {"type": type(exc).__name__, "code": code}})


def _one_tool_call(raw):
    try:
        choices = raw["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError
        message = choices[0]["message"]
        calls = message.get("tool_calls")
        if message.get("role") != "assistant" or not isinstance(calls, list) or len(calls) != 1:
            raise ValueError
        call = calls[0]
        if (call.get("type") != "function" or not isinstance(call.get("id"), str)
                or not call["id"] or not isinstance(call.get("function", {}).get("name"), str)
                or not isinstance(call["function"].get("arguments"), str)):
            raise ValueError
        if choices[0].get("finish_reason") != "tool_calls":
            raise ValueError
        return message
    except (KeyError, TypeError, ValueError, AttributeError):
        raise ServiceError("MODEL_OUTPUT_INVALID", "Expected exactly one complete structured function call") from None


def _schema_with_defs(model):
    schema = model.model_json_schema()
    definitions = schema.pop("$defs", {})
    schema.pop("title", None)
    return schema, definitions


def service_tools():
    descriptions = {
        "list_customer_orders": "List orders belonging to the authenticated customer in a merchant scope. Set scope_id to a merchant ID and resource_id to that same merchant ID; params is an empty object.",
        "get_order_details": "Read an order and its lines. resource_id is the order ID; scope_id is its merchant ID; params is empty.",
        "search_catalog": "Search a merchant catalog. Set scope_id and resource_id to the same merchant ID; use params to filter product_family and attributes.",
        "get_sku_details": "Read one SKU. resource_id is the SKU ID and scope_id is its merchant ID; params is empty.",
        "get_inventory": "Read inventory rows for one SKU. resource_id is the SKU ID and scope_id is its merchant ID; params is empty.",
        "estimate_delivery": "Estimate delivery for one SKU. resource_id is the SKU ID, scope_id is its merchant ID, and params supplies warehouse_id and optionally quantity.",
        "get_refundable_balance": "Read the unrefunded purchased units and original credit for one order line. resource_id is the order-line ID; scope_id is its merchant ID; params is empty.",
        "get_case_status": "Read one remedy case, its quotes and returns, and submitted evidence references for the case line. Evidence references are not pre-verified. resource_id is the case ID; scope_id is its merchant ID; params is empty.",
        "open_remedy_case": "Prepare a remedy case for one purchased order line after customer delegation. resource_id is the order-line ID; params supplies quantity and issue.",
        "verify_submitted_evidence": "Verify already submitted evidence for a remedy case. First use get_case_status to obtain an evidence_id for this case; resource_id is the case ID and params supplies that evidence_id. Querying does not verify the evidence.",
        "request_policy_review": "Request policy review for an existing remedy case. resource_id is the case ID; params is empty.",
        "prepare_remedy_quote": "Prepare refund, exchange, or reship terms for a verified remedy case. resource_id is the case ID; params supplies remedy and any replacement details.",
        "commit_accepted_quote": "Commit an existing quote only when the runtime has recorded exact customer consent. resource_id is the quote ID; params is empty.",
        "schedule_authorized_return": "Schedule the return required by a committed quote. resource_id is the case ID; params is empty.",
        "release_accepted_refund": "Release a refund for a committed case only when the runtime has recorded exact refund-release consent. resource_id is the case ID; params is empty.",
        "ship_accepted_replacement": "Ship the replacement for a committed case after required return conditions are met. resource_id is the case ID; params is empty.",
        "cancel_pending_commitment": "Cancel an existing pending quote commitment. resource_id is the quote ID; params is empty.",
    }
    tools = []
    for name, parameter_model in sorted(PARAMETERS.items()):
        schema, definitions = _schema_with_defs(parameter_model)
        tools.append({"type": "function", "function": {
            "name": name, "description": descriptions[name],
            "parameters": {"type": "object", "properties": {
                "scope_id": {"type": "string", "description": "Merchant ID defining the business scope."},
                "resource_id": {"type": "string", "description": descriptions[name]}, "params": schema,
            }, "required": ["scope_id", "resource_id", "params"], "additionalProperties": False,
                **({"$defs": definitions} if definitions else {})}}})
    tools.append({"type": "function", "function": {
        "name": "respond_to_customer", "description": (
            "Send the next customer-facing message. The runtime automatically includes any actual "
            "quote that is awaiting the Customer's decision."
        ),
        "parameters": {"type": "object", "properties": {
            "message": {"type": "string"},
            "subject_ids": {"type": "array", "items": {"type": "string"}},
        }, "required": ["message", "subject_ids"],
           "additionalProperties": False}}})
    return tools


CUSTOMER_TOOLS = [{"type": "function", "function": {
    "name": "customer_turn", "description": "Respond naturally and declare only Customer-side intent.",
    "parameters": {"type": "object", "properties": {
        "message": {"type": "string"},
        "delegate_line_ids": {"type": "array", "items": {"type": "string"}},
        "allowed_preparations": {"type": "array", "items": {"type": "string"}},
        "accept_quote_id": {"type": ["string", "null"]},
        "accept_operation": {"type": ["string", "null"], "enum": ["commit_accepted_quote", "release_accepted_refund", None]},
        "reject_quote_id": {"type": ["string", "null"]},
        "withdraw_delegation_ids": {"type": "array", "items": {"type": "string"}},
        "disclose_fact_ids": {"type": "array", "items": {"type": "string"}},
        "finish": {"type": "boolean"},
    }, "required": ["message", "delegate_line_ids", "allowed_preparations", "accept_quote_id",
                    "accept_operation", "reject_quote_id", "withdraw_delegation_ids", "disclose_fact_ids", "finish"],
       "additionalProperties": False}}}]


class OpenAICustomerAgent:
    """Customer-only model adapter; never pass its private prompt to Service."""

    uses_model = True

    def __init__(self, client: CompletionClient):
        self.client = client
        self.settings: ModelSettings = client.settings.model_copy(deep=True)
        self.last_call: Optional[AgentCall] = None

    def decide(self, prompt: CustomerPrompt, profile: CustomerProfile):
        self.last_call = None
        profile = profile.checked()
        system = (
            "Simulate the authenticated customer. You may disclose only facts in the private profile, "
            "delegate only the listed business lines/preparations, and accept only the exact quote shown. "
            "Do not claim an operation completed. A quote acceptance is transaction consent only for the "
            "named quote and operation. Keep natural-language preferences consistent across turns.\n"
            "PRIVATE CUSTOMER PROFILE: " + json.dumps(profile.model_dump(mode="json"), ensure_ascii=False)
        )
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps({
                        "public_request": prompt.public_request,
                        "last_service_message": prompt.last_service_message,
                        "presented_offer": prompt.offer.model_dump(mode="json") if prompt.offer else None,
                        "conversation": [row.model_dump(mode="json") for row in prompt.conversation],
                    }, ensure_ascii=False)}]
        request = self.settings.request(messages, CUSTOMER_TOOLS)
        request["tool_choice"] = {"type": "function", "function": {"name": "customer_turn"}}
        try:
            raw = self.client.complete(request)
        except Exception as exc:
            self.last_call = _safe_failure("customer", request, exc)
            raise
        self.last_call = AgentCall(role="customer", request=request, response=raw,
                                   prompt_tokens=_usage(raw, "prompt_tokens"),
                                   completion_tokens=_usage(raw, "completion_tokens"))
        call_message = _one_tool_call(raw)
        function = call_message["tool_calls"][0]["function"]
        if function["name"] != "customer_turn":
            raise ServiceError("MODEL_OUTPUT_INVALID", "Customer selected an unregistered action")
        try:
            decision = CustomerDecision.model_validate(_json_arguments(function["arguments"]))
        except ValueError:
            raise ServiceError("MODEL_OUTPUT_INVALID", "Customer decision violates its response contract") from None
        return decision


class OpenAIServiceAgent:
    """Function-calling Service; it receives public task text and tool outputs only."""

    uses_model = True

    SYSTEM = (
        "You are a commerce customer-service agent. Decide what to query and what action to take from "
        "the customer's public request and public tool results. You never receive hidden TaskSpec goals, "
        "Customer preferences, proof witnesses or Oracle decisions. Tool writes are not Customer consent. "
        "Never commit a quote unless the Customer runtime has separately recorded exact consent; never claim "
        "completion without a successful tool result. After you prepare a quote, the runtime automatically "
        "shows its exact terms to the Customer with your next customer-facing reply. Explain the quote and "
        "ask whether the Customer accepts; do not choose a quote-presentation action or invent quote terms. "
        "After a registered return receipt, the runtime likewise shows the committed refund terms before "
        "requesting any release consent. The caller is already authenticated; use the supplied customer ID and merchant "
        "scope IDs instead of asking the Customer for account or order identifiers that the tools can look up. "
        "Use one function call for a business operation. A normal assistant message is also allowed for a "
        "customer-facing clarification or status update, but it has no business side effect and must not claim "
        "that an operation was performed. Never make multiple function calls in one response."
    )

    def __init__(self, client: CompletionClient, *, skill_snapshot=None,
                 skill_domain: str = "aftersales"):
        self.client = client
        self.settings: ModelSettings = client.settings.model_copy(deep=True)
        self.last_call: Optional[AgentCall] = None
        # Skills are optional and are only used by the Research Studio's
        # isolated candidate runs. The default path keeps the historical S0
        # prompt byte-for-byte unchanged.
        self.skill_snapshot = skill_snapshot
        self.skill_domain = skill_domain
        self.last_skill_injection: Optional[Dict[str, Any]] = None
        if skill_snapshot is not None:
            from vast_coevo.skills.library import LibrarySnapshot
            checked = LibrarySnapshot.model_validate(skill_snapshot.model_dump(mode="json"))
            if checked.role != "service":
                raise ValueError("Service can only receive a Service Skill snapshot")
            self.skill_snapshot = checked

    def decide(self, prompt: ServicePrompt):
        self.last_call = None
        system_prompt = self.SYSTEM
        self.last_skill_injection = None
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps({
                        "authenticated_customer_id": prompt.customer_id,
                        "available_merchant_scope_ids": list(prompt.available_merchant_scope_ids),
                        "customer_request": prompt.customer_request,
                    }, ensure_ascii=False)}]
        history = list(prompt.conversation)
        # The initial request is already included with trusted routing context.
        # Do not repeat it as a second, context-free message on the first turn.
        if history and history[0].role == "user" and history[0].content == prompt.customer_request:
            history = history[1:]
        messages.extend(row.model_dump(mode="json", exclude_none=True) for row in history)
        if self.skill_snapshot is not None:
            from vast_coevo.schema import digest
            from vast_coevo.skills.renderer import render_skills
            from vast_coevo.skills.retriever import SkillContext, SkillRetriever
            context = SkillContext(
                domain=self.skill_domain,
                public_text=prompt.customer_request,
                available_tools=tuple(sorted({
                    row["function"]["name"] for row in service_tools()
                    if isinstance(row, dict) and isinstance(row.get("function"), dict)
                    and isinstance(row["function"].get("name"), str)
                })),
            )
            retrieval = SkillRetriever("service").retrieve(self.skill_snapshot, context)
            rendered = render_skills(retrieval.skills)
            if rendered:
                # A separate system message keeps the immutable base contract
                # untouched and makes the exact injected guidance auditable.
                messages.append({"role": "system", "content": rendered})
            self.last_skill_injection = {
                "snapshot_hash": self.skill_snapshot.snapshot_hash,
                "context_hash": digest(context),
                "hits": [hit.model_dump(mode="json") for hit in retrieval.hits],
                "rendered": rendered,
                "rendered_hash": digest(rendered),
            }
        request = self.settings.request(messages, service_tools())
        # Qwen thinking models reject `tool_choice=required`; keep automatic
        # selection and handle a safe, side-effect-free text reply explicitly.
        request["parallel_tool_calls"] = False
        if self.last_skill_injection is not None:
            from vast_coevo.schema import digest
            self.last_skill_injection["request_hash"] = digest(request)
        try:
            raw = self.client.complete(request)
        except Exception as exc:
            self.last_call = _safe_failure("service", request, exc)
            raise
        self.last_call = AgentCall(role="service", request=request, response=raw,
                                   prompt_tokens=_usage(raw, "prompt_tokens"),
                                   completion_tokens=_usage(raw, "completion_tokens"))
        # OpenAI-compatible providers may validly choose a plain assistant reply
        # when tool_choice=auto. Treat only complete, nonempty final text as a
        # user-facing response. It cannot execute a business operation; all
        # side effects still require a validated function call.
        try:
            choices = raw["choices"]
            message = choices[0]["message"] if isinstance(choices, list) and len(choices) == 1 else {}
            no_tool_calls = not message.get("tool_calls")
            content = message.get("content")
            if (message.get("role") == "assistant" and no_tool_calls
                    and choices[0].get("finish_reason") == "stop"
                    and isinstance(content, str) and content.strip()):
                return ServiceAction(kind="reply", message=content, call=self.last_call)
        except (KeyError, TypeError, IndexError, AttributeError):
            pass
        action = ServiceAction.from_tool_message(raw, prompt.customer_id, prompt.observed_subject_ids)
        data = action.model_dump(mode="json", exclude={"record_hash"})
        data["call"] = self.last_call.model_dump(mode="json")
        return ServiceAction.model_validate(data)
