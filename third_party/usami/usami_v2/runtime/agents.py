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
    def from_tool_message(cls, raw, customer_id, observed_subject_ids=(), merchant_scope_ids=(), retail=False):
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
        if name not in PUBLIC_PARAMETERS or (name in RETAIL_TOOLS and not retail):
            raise ServiceError("MODEL_OUTPUT_INVALID", "Service selected an unregistered commerce tool")
        scope = _single_merchant_scope(merchant_scope_ids)
        payment = args.get("payment_method") or "original"
        if payment != "original" and not retail:
            raise ServiceError("MODEL_OUTPUT_INVALID", "this tool profile only supports the original payment method")
        if name in ("list_customer_orders", "get_customer_profile"):
            resource, params = scope, {}
        elif name == "update_default_address":
            resource, params = scope, {key: args.get(key) for key in ADDRESS_FIELDS}
        elif name in ("get_order_details", "cancel_order", "modify_order_address", "modify_order_items"):
            order_id = args.get("order_id")
            if not isinstance(order_id, str) or not order_id:
                raise ServiceError("MODEL_OUTPUT_INVALID", name + " requires an order_id")
            resource = order_id
            if name == "modify_order_address":
                params = {key: args.get(key) for key in ADDRESS_FIELDS}
            elif name == "modify_order_items":
                changes = args.get("changes")
                if not isinstance(changes, list) or any(not isinstance(row, dict) for row in changes):
                    raise ServiceError("MODEL_OUTPUT_INVALID", "modify_order_items requires a list of changes")
                params = {"changes": [{"line_id": row.get("line_id"), "target_sku_id": row.get("target_sku_id")}
                                      for row in changes], "payment_method": payment}
            else:
                params = {}
        else:
            line_id = args.get("line_id")
            if not isinstance(line_id, str) or not line_id:
                raise ServiceError("MODEL_OUTPUT_INVALID", name + " requires a line_id")
            resource = line_id
            if name == "return_delivered_order_items":
                params = {"quantity": args.get("quantity"), "payment_method": payment}
            elif name == "exchange_delivered_order_items":
                params = {"quantity": args.get("quantity"), "target_sku_id": args.get("target_sku_id")}
                if payment != "original":
                    params["payment_method"] = payment
            elif name == "list_exchange_options":
                params = {}
            else:
                params = {"quantity": args.get("quantity")}
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


PUBLIC_TOOLS = ("list_customer_orders", "get_order_details", "return_delivered_order_items",
                "cancel_order", "list_exchange_options", "exchange_delivered_order_items",
                "reship_order_items", "request_item_return")
# Retail extension: only the 'retail' tool profile exposes (and parses) these.
RETAIL_TOOLS = frozenset({"get_customer_profile", "modify_order_address", "update_default_address",
                          "modify_order_items"})
ADDRESS_FIELDS = ("street", "city", "state", "postal_code")
PUBLIC_PARAMETERS = frozenset(PUBLIC_TOOLS) | RETAIL_TOOLS


def _public_tool(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": required, "additionalProperties": False}}}


def service_tools(profile='full9'):
    """The small Agent-facing surface; internal case/quote/consent steps stay hidden."""
    line = {"type": "string", "description": "Order line ID from get_order_details."}
    quantity = {"type": "integer", "minimum": 1, "description": "Number of purchased units."}
    if profile not in ('core4', 'full9', 'retail'):
        raise ValueError('unknown Service tool profile')
    tools = [
        _public_tool("list_customer_orders",
                     "List the orders belonging to the authenticated customer. An order that has not been "
                     "delivered shows fulfillment_status 'unshipped' or 'shipped'; an order without that "
                     "field was delivered.", {}, []),
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
        _public_tool("cancel_order",
                     "Cancel a whole order that has not shipped yet and refund it to the original payment "
                     "channel. Shipped or delivered orders cannot be cancelled.",
                     {"order_id": {"type": "string",
                                   "description": "Order ID returned by list_customer_orders."}},
                     ["order_id"]),
        _public_tool("list_exchange_options",
                     "List the variants a delivered order line can be exchanged for, with price and stock.",
                     {"line_id": line}, ["line_id"]),
        _public_tool("exchange_delivered_order_items",
                     "Exchange delivered units of one line for a different variant in a single step. Ships "
                     "the variant from stock and charges or refunds any price difference on the original "
                     "payment channel.",
                     {"line_id": line, "quantity": quantity,
                      "target_sku_id": {"type": "string",
                                        "description": "sku_id returned by list_exchange_options."}},
                     ["line_id", "quantity", "target_sku_id"]),
        _public_tool("reship_order_items",
                     "Send a free replacement of the same item for delivered units with a verified "
                     "delivery problem.",
                     {"line_id": line, "quantity": quantity}, ["line_id", "quantity"]),
        _public_tool("request_item_return",
                     "Accept a physical return for items that must be sent back before a refund. The refund "
                     "is issued with return_delivered_order_items only after the environment reports that "
                     "the goods were received.",
                     {"line_id": line, "quantity": quantity}, ["line_id", "quantity"]),
        _public_tool("respond_to_customer",
                     "Send the next customer-facing message.",
                     {"message": {"type": "string"}}, ["message"]),
    ]
    if profile == 'core4':
        names = {'list_customer_orders', 'get_order_details',
                 'return_delivered_order_items', 'respond_to_customer'}
        return [tool for tool in tools if tool['function']['name'] in names]
    if profile == 'retail':
        return _retail_tools(tools, line)
    return tools


def _retail_tools(tools, line):
    """The full9 surface plus address, payment-method and pending-order-item tools."""
    from copy import deepcopy
    tools = deepcopy(tools)
    payment = {"type": "string", "description": "'original', or the id of a payment method from "
               "get_customer_profile. Refunds may go to the original method or a gift card; a gift card "
               "that pays a charge must hold enough balance."}
    for tool in tools:
        spec = tool["function"]
        if spec["name"] == "return_delivered_order_items":
            spec["parameters"]["properties"]["payment_method"] = payment
        elif spec["name"] == "exchange_delivered_order_items":
            spec["parameters"]["properties"]["payment_method"] = payment
        elif spec["name"] == "cancel_order":
            spec["description"] += " An order whose items were already modified cannot be cancelled."
    address = {key: {"type": "string"} for key in ADDRESS_FIELDS}
    order = {"type": "string", "description": "Order ID returned by list_customer_orders."}
    extra = [
        _public_tool("get_customer_profile",
                     "Read the authenticated customer's default address and payment methods (gift-card "
                     "balances included).", {}, []),
        _public_tool("modify_order_address",
                     "Change the shipping address of an order that has not shipped yet.",
                     {"order_id": order, **address}, ["order_id", *ADDRESS_FIELDS]),
        _public_tool("update_default_address",
                     "Change the customer's default address on file.", address, list(ADDRESS_FIELDS)),
        _public_tool("modify_order_items",
                     "Change items of an order that has not shipped yet to another variant of the same "
                     "product. This can be done ONLY ONCE per order: afterwards the order can be neither "
                     "modified nor cancelled, so collect every change first. Any price difference is paid or "
                     "refunded through payment_method.",
                     {"order_id": order,
                      "changes": {"type": "array", "minItems": 1, "items": {
                          "type": "object", "additionalProperties": False,
                          "properties": {"line_id": {"type": "string"},
                                         "target_sku_id": {"type": "string",
                                                           "description": "sku_id from list_exchange_options."}},
                          "required": ["line_id", "target_sku_id"]}},
                      "payment_method": payment},
                     ["order_id", "changes"]),
    ]
    respond = [tool for tool in tools if tool["function"]["name"] == "respond_to_customer"]
    rest = [tool for tool in tools if tool["function"]["name"] != "respond_to_customer"]
    return rest + extra + respond


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
            "completed. When the service asks you to confirm a refund, cancellation, exchange, replacement or "
            "return, answer yes or no in your own words. Set finish=true when your request is settled or you "
            "no longer want to continue.\n"
            "PRIVATE CUSTOMER PROFILE: " + json.dumps(profile.model_dump(mode="json"), ensure_ascii=False)
        )
        if not getattr(profile, 'accepts_proposals', True):
            system += ("\nYour private choice is to decline the proposed action. "
                       "Reply no and set declines_proposal=true in customer_turn.")
        elif getattr(profile, 'decline_line_ids', ()):
            system += ("\nOnly the listed decline_references are declined; other requests may be accepted. "
                       "When declining, set declines_proposal=true and declined_line_id to its matching ID. "
                       "If the proposed item is unclear, ask which item is being discussed.")
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps({
                        "public_request": prompt.public_request,
                        "last_service_message": prompt.last_service_message,
                        "presented_offer": prompt.offer.model_dump(mode="json") if prompt.offer else None,
                        "conversation": [row.model_dump(mode="json") for row in prompt.conversation],
                    }, ensure_ascii=False)}]
        customer_tools = CUSTOMER_TOOLS
        if not getattr(profile, 'accepts_proposals', True) or getattr(profile, 'decline_line_ids', ()):
            from copy import deepcopy
            customer_tools = deepcopy(CUSTOMER_TOOLS)
            properties = customer_tools[0]['function']['parameters']['properties']
            properties['declines_proposal'] = {'type': 'boolean'}
            customer_tools[0]['function']['parameters']['required'].append('declines_proposal')
            if getattr(profile, 'decline_line_ids', ()):
                properties['declined_line_id'] = {'anyOf': [{'type': 'string'}, {'type': 'null'}]}
                customer_tools[0]['function']['parameters']['required'].append('declined_line_id')
        request = self.settings.request(messages, customer_tools)
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
        "Follow these steps in order. 1. List the customer's orders with list_customer_orders. "
        "2. Read the relevant order details with get_order_details (and list_exchange_options before an "
        "exchange). 3. Explain the exact action, item, quantity, any price difference and the payment "
        "channel in a respond_to_customer message. 4. Ask the customer for confirmation and wait for their "
        "reply. 5. Only if the customer agrees, perform that one business action: return_delivered_order_items "
        "(refund), cancel_order (unshipped order), exchange_delivered_order_items (different variant), "
        "reship_order_items (same item again) or request_item_return (items that must be sent back first); "
        "never act when the customer declines. 6. Report the actual result. For items that must be returned, "
        "only refund with return_delivered_order_items after the environment reports the goods were received. \n"
        "The environment enforces ownership, fulfillment and delivery status, return windows, quantities, stock, "
        "duplicate actions and the payment method, and answers with an explicit error_code when a request is "
        "invalid. Do not ask the customer to upload evidence or to read internal case, quote, settlement or "
        "consent identifiers; those objects are environment internals and are never shown to you. \n"
        "Never claim that a refund, cancellation, shipment, receipt or return happened without a successful tool "
        "result or environment report. Use respond_to_customer for every customer-facing reply. Never make "
        "multiple function calls in one response."
    )

    CORE4_SYSTEM = (
        "You are a commerce customer-service agent. Use only the shown tools and public customer request. "
        "List the customer's orders, read the relevant order details, and explain the requested refund. "
        "Ask the customer for confirmation before returning delivered items and never act if they decline. "
        "Respect delivery status, return windows, quantities and the original payment method. "
        "Report only results supported by tool output. Use respond_to_customer for replies and make "
        "exactly one function call per response."
    )

    RETAIL_ADDENDUM = (
        "\nRetail rules. The customer may own several payment methods (gift card with a balance, credit card, "
        "PayPal); read them with get_customer_profile. A refund goes back to the original method or to a gift "
        "card of the customer. A price difference may be paid with any method of the customer, but a gift card "
        "must hold enough balance. Gift-card refunds arrive immediately, other refunds in 5 to 7 business days. "
        "Addresses can be changed (modify_order_address, update_default_address) only for orders that have not "
        "shipped; look up an address the customer does not state in their profile or other orders instead of "
        "guessing. modify_order_items works once per order and locks the order; before calling it, make sure the "
        "customer has named every item to change."
    )

    @classmethod
    def system_for_profile(cls, profile):
        if profile == 'full9':
            return cls.SYSTEM
        if profile == 'retail':
            return cls.SYSTEM + cls.RETAIL_ADDENDUM
        if profile == 'core4':
            return cls.CORE4_SYSTEM
        raise ValueError('unknown Service tool profile')

    def __init__(self, client: CompletionClient, *, skill_snapshot=None,
                 skill_domain: str = "aftersales", tool_profile='full9'):
        self.client = client
        self.tool_profile = tool_profile
        self.system_for_profile(tool_profile)
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
        system_prompt = self.system_for_profile(self.tool_profile)
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
                    row["function"]["name"] for row in service_tools(self.tool_profile)
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
        request = self.settings.request(messages, service_tools(self.tool_profile))
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
        message = _one_tool_call(raw)
        name = message['tool_calls'][0]['function']['name']
        if name not in {tool['function']['name'] for tool in service_tools(self.tool_profile)}:
            raise ServiceError('MODEL_OUTPUT_INVALID', 'Service selected a tool outside the active profile')
        action = ServiceAction.from_tool_message(raw, prompt.customer_id, prompt.observed_subject_ids,
                                                prompt.available_merchant_scope_ids,
                                                retail=self.tool_profile == 'retail')
        data = action.model_dump(mode="json", exclude={"record_hash"})
        data["call"] = self.last_call.model_dump(mode="json")
        return ServiceAction.model_validate(data)
