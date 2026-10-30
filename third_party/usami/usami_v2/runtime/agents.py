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
    def from_tool_message(cls, raw, customer_id, observed_subject_ids=(), merchant_scope_ids=()):
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
        if name not in PUBLIC_PARAMETERS:
            raise ServiceError("MODEL_OUTPUT_INVALID", "Service selected an unregistered commerce tool")
        scope = _single_merchant_scope(merchant_scope_ids)
        if name == "list_customer_orders":
            resource, params = scope, {}
        elif name == "get_order_details":
            order_id = args.get("order_id")
            if not isinstance(order_id, str) or not order_id:
                raise ServiceError("MODEL_OUTPUT_INVALID", "get_order_details requires an order_id")
            resource, params = order_id, {}
        else:
            line_id = args.get("line_id")
            if not isinstance(line_id, str) or not line_id:
                raise ServiceError("MODEL_OUTPUT_INVALID", "return_delivered_order_items requires a line_id")
            resource = line_id
            params = {"quantity": args.get("quantity"),
                      "payment_method": args.get("payment_method") or "original"}
        try:
            operation = Operation(name=name, scope_id=scope, resource_id=resource, params=params)
        except (KeyError, TypeError, ValueError):
            raise ServiceError("MODEL_OUTPUT_INVALID", "Service tool arguments do not match the native schema") from None
        # A Request is created only by the Runner for this exact tool call; the
        # Reference still independently checks identity, delegation, ownership
        # and the refund business rules.
        return cls(kind="operation", operation=operation, call=completion, call_id=call["id"])


def _single_merchant_scope(merchant_scope_ids):
    """Phase 1 worlds expose one merchant scope; no model-supplied tenant ID."""
    scopes = tuple(merchant_scope_ids or ())
    if len(scopes) != 1:
        raise ServiceError("MODEL_OUTPUT_INVALID",
                           "this world exposes exactly one merchant scope for customer-scoped tools")
    return scopes[0]


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


PUBLIC_TOOLS = ("list_customer_orders", "get_order_details", "return_delivered_order_items")
PUBLIC_PARAMETERS = frozenset(PUBLIC_TOOLS)


def _public_tool(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": required, "additionalProperties": False}}}


def service_tools():
    """The small Agent-facing surface; internal case/quote/consent steps stay hidden."""
    return [
        _public_tool("list_customer_orders",
                     "List the orders belonging to the authenticated customer.", {}, []),
        _public_tool("get_order_details",
                     "Read one order and its purchased line items.",
                     {"order_id": {"type": "string",
                                   "description": "Order ID returned by list_customer_orders."}},
                     ["order_id"]),
        _public_tool("return_delivered_order_items",
                     "Return delivered items of one order line and refund the original payment "
                     "channel in a single step. Only items that were delivered, are inside the "
                     "merchant return window and have not already been refunded can be returned.",
                     {"line_id": {"type": "string",
                                  "description": "Order line ID from get_order_details."},
                      "quantity": {"type": "integer", "minimum": 1,
                                   "description": "Number of purchased units to refund."},
                      "payment_method": {"type": "string", "enum": ["original"],
                                         "description": "Refund target; Phase 1 supports only 'original'."}},
                     ["line_id", "quantity"]),
        _public_tool("respond_to_customer",
                     "Send the next customer-facing message.",
                     {"message": {"type": "string"}}, ["message"]),
    ]


CUSTOMER_TOOLS = [{"type": "function", "function": {
    "name": "customer_turn", "description": "Reply to the service agent as the customer, or finish.",
    "parameters": {"type": "object", "properties": {
        "message": {"type": "string"},
        "finish": {"type": "boolean"},
    }, "required": ["message", "finish"], "additionalProperties": False}}}]


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
            "Simulate the authenticated customer in a support conversation. Write short, natural replies "
            "consistent with your private profile. Disclose only what the conversation actually requires, "
            "and never invent order, item or payment details. Never claim that an operation was already "
            "completed. Set finish=true when your request is settled or you no longer want to continue.\n"
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
        "You are a commerce customer-service agent. Resolve the customer's request with the tools shown. "
        "You only see the public customer request and public tool results; never assume hidden task goals, "
        "private customer preferences, proof witnesses or oracle decisions. \n"
        "Typical flow: call list_customer_orders to find the authenticated customer's orders, "
        "get_order_details to read one order and its line items, then return_delivered_order_items with "
        "the line ID and the quantity of units to refund. That single call records the returned items and "
        "credits the original payment channel; physical return logistics happen outside this dialogue. \n"
        "The environment enforces ownership, delivery status, the return window, quantities, duplicate "
        "refunds and the refund method, and answers with an explicit error_code when a request is invalid. "
        "Do not ask the customer to upload evidence or to read internal case, quote, settlement or consent "
        "identifiers; those objects are environment internals and are never shown to you. \n"
        "Never claim that a refund or return happened without a successful tool result. Use "
        "respond_to_customer for every customer-facing reply. Never make multiple function calls in one response."
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
        action = ServiceAction.from_tool_message(raw, prompt.customer_id, prompt.observed_subject_ids,
                                                prompt.available_merchant_scope_ids)
        data = action.model_dump(mode="json", exclude={"record_hash"})
        data["call"] = self.last_call.model_dump(mode="json")
        return ServiceAction.model_validate(data)
