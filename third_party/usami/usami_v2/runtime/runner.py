"""Independent commerce episode Runner with native Engine/Oracle/Replay.

The Runner accepts any Customer/Service decision adapters. No model is called
unless the caller explicitly supplies an online adapter and invokes run_episode.
"""
from datetime import timedelta
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Dict, Literal, Optional, Protocol, Tuple, Union

from pydantic import Field

from usami_harness.engine import Engine, Trace, replay_trace
from usami_harness.models import Request, digest
from usami_harness.oracle import evaluate
from usami_v2.commerce import adapter
from usami_v2.commerce.external import apply_event, reference_event
from usami_v2.commerce.models import ExternalEvent, Operation, World
from usami_v2.contracts import FactEvidence, Sealed, ServiceMessage, SourceRef
from usami_v2.evaluation.outcomes import (
    Outcome, OutcomeV2, evaluate_execution_compliance, evaluate_outcome, evaluate_outcome_v2,
)
from usami_v2.evaluation.communication import CommunicationResult
from usami_v2.runtime.agents import (
    AgentCall, AgentMessage, CustomerPrompt, ServiceAction, ServicePrompt,
)
from usami_v2.runtime.customer import (
    CustomerDecision, CustomerProfile, CustomerProfileV2, CustomerProfileV3, HumanCustomerProfileV2,
    CustomerRuntime, OfferPresentation, render_offer,
)
from usami_v2.scenarios.tasks import (
    TaskSpec, TaskSpecV2, TaskSpecV3, check_task_scope, evaluate_state_goals, evaluate_task_compliance,
    evaluate_task_communication,
)


class EpisodeBudget(Sealed):
    max_customer_turns: int = Field(default=8, ge=1, le=100, strict=True)
    max_service_decisions: int = Field(default=64, ge=1, le=2000, strict=True)
    max_service_tool_calls: int = Field(default=64, ge=1, le=2000, strict=True)
    max_model_calls: int = Field(default=96, ge=0, le=5000, strict=True)


class EpisodeEvent(Sealed):
    index: int = Field(ge=0, strict=True)
    kind: Literal["customer_request", "customer_transition", "service_tool", "service_reply",
                  "offer_presented", "external_event", "agent_response", "agent_error", "termination"]
    payload: Dict[str, Any]
    before_state_hash: str
    after_state_hash: str
    previous_event_hash: str = ""


class ToolExecution(Sealed):
    request: Dict[str, Any]
    operation: Dict[str, Any]
    engine_trace: Dict[str, Any]
    oracle: Dict[str, Any]
    result: Dict[str, Any]
    before_state_hash: str
    after_state_hash: str


class EpisodeRecord(Sealed):
    protocol: Literal["commerce-episode-v1"] = "commerce-episode-v1"
    task: Union[TaskSpecV3, TaskSpecV2, TaskSpec]
    profile: Union[HumanCustomerProfileV2, CustomerProfileV3, CustomerProfileV2, CustomerProfile]
    budget: EpisodeBudget
    mode: Literal["reference", "sandbox"]
    initial_world: World
    scheduled_external_events: Tuple[ExternalEvent, ...] = ()
    runtime_initial_world: World
    final_world: World
    status: Literal["customer_finished", "budget_exhausted", "service_protocol_error",
                    "customer_protocol_error", "agent_error", "runner_error", "cancelled"]
    terminal_reason: str
    events: Tuple[EpisodeEvent, ...]
    tool_executions: Tuple[ToolExecution, ...]
    service_messages: Tuple[ServiceMessage, ...]
    facts: Tuple[FactEvidence, ...]
    model_calls: Tuple[AgentCall, ...]
    communication: CommunicationResult
    outcome: Union[Outcome, OutcomeV2]
    replay_verified: bool

    @property
    def total_tokens(self):
        vals = [row.prompt_tokens + row.completion_tokens for row in self.model_calls
                if row.prompt_tokens is not None and row.completion_tokens is not None]
        return sum(vals) if len(vals) == len(self.model_calls) else None

    @property
    def costs(self):
        return {
            "customer_calls": sum(row.role == "customer" for row in self.model_calls),
            "service_calls": sum(row.role == "service" for row in self.model_calls),
            "tool_calls": len(self.tool_executions), "known_total_tokens": self.total_tokens,
            "unknown_usage_calls": sum(row.prompt_tokens is None or row.completion_tokens is None
                                        for row in self.model_calls),
        }


class EpisodeReplay(Sealed):
    verified: bool
    event_count: int
    operation_count: int
    final_world_hash: str
    errors: Tuple[str, ...] = ()
    outcome_hash: str


class CustomerAgent(Protocol):
    uses_model: bool
    last_call: Optional[AgentCall]

    def decide(self, prompt: CustomerPrompt, profile: CustomerProfile) -> CustomerDecision: ...


class ServiceAgent(Protocol):
    uses_model: bool
    last_call: Optional[AgentCall]

    def decide(self, prompt: ServicePrompt) -> ServiceAction: ...


class EpisodeCancelled(Exception):
    """A caller stopped an interactive Episode; this is incomplete evidence, not a Service error."""


def _notify_progress(callback, kind, **payload):
    if callback is None:
        return
    try:
        callback({"kind": kind, **payload})
    except Exception:
        # UI observers are deliberately outside the evidence and execution path.
        pass


def _append(events, kind, payload, before, after):
    return events + [EpisodeEvent(index=len(events), kind=kind, payload=payload,
                                   before_state_hash=before.state_hash,
                                   after_state_hash=after.state_hash,
                                   previous_event_hash=events[-1].record_hash if events else "")]


def _public_ids(value):
    found = set()
    if isinstance(value, dict):
        for key, row in value.items():
            if (key.endswith("_id") or key == "id") and isinstance(row, str):
                found.add(row)
            found.update(_public_ids(row))
    elif isinstance(value, (list, tuple)):
        for row in value:
            found.update(_public_ids(row))
    return found


def _facts_for_result(world, operation, data, event_hash, event_index):
    """Extract only explicit typed facts from genuine public tool results."""
    subject, facts = operation.resource_id, []
    source = SourceRef(source_id="tool-" + event_hash[:20], source_kind="public_tool",
                       source_hash=event_hash, event_index=event_index)

    def add(kind, value, *, fact_subject=subject, unit=""):
        facts.append(FactEvidence(fact_id="fact-" + digest([event_hash, fact_subject, kind, value])[:24],
                                  subject_id=fact_subject, kind=kind, value=value, unit=unit,
                                  valid_from=event_index, source=source))

    if operation.name == "get_refundable_balance":
        amount = data.get("remaining_original_credit_cents")
        if type(amount) is int:
            add("refund_amount", amount, unit="CNY_cents")
        units = data.get("remaining_units")
        if isinstance(units, list):
            add("quantity", len(units))
    quote_data = data.get("quote") if isinstance(data, dict) else None
    if isinstance(quote_data, dict):
        line_id = world.cases.get(quote_data.get("case_id"))
        fact_subject = line_id.line_id if line_id else subject
        amount = quote_data.get("refund_cents")
        extra = quote_data.get("extra_charge_cents")
        quantity = quote_data.get("quantity")
        arrival = quote_data.get("promised_arrival")
        if type(amount) is int:
            add("refund_amount", amount, fact_subject=fact_subject, unit="CNY_cents")
        if type(extra) is int:
            add("extra_charge", extra, fact_subject=fact_subject, unit="CNY_cents")
        if type(quantity) is int:
            add("quantity", quantity, fact_subject=fact_subject)
        if isinstance(arrival, str):
            add("arrival_date", arrival[:10], fact_subject=fact_subject)
    if operation.name in ("commit_accepted_quote", "release_accepted_refund"):
        quote = world.quotes.get(data.get("quote_id", operation.resource_id)) if isinstance(data, dict) else None
        if quote and quote.remedy == "refund" and any(row.quote_id == quote.id for row in world.settlements.values()):
            case = world.cases.get(quote.case_id)
            if case:
                add("refund_status", "released", fact_subject=case.line_id)
    return tuple(facts)


def _execution(world, customer_id, operation, request_id, mode):
    request = Request(id=request_id, identity=customer_id,
                      operation=operation.model_dump(mode="json"), granted=True)
    engine = Engine(adapter, world, operation_budget=1)
    engine.register(request)
    event = engine.execute(customer_id, operation, request_id, mode=mode)
    trace = engine.finish()
    oracle = evaluate(adapter, trace, {request_id: request})
    replayed = replay_trace(adapter, trace, {request_id: request})
    if replayed.state.state_hash != engine.state.state_hash:
        raise ValueError("native Engine Replay differs from the actual operation state")
    row = ToolExecution(request=request.model_dump(mode="json"), operation=operation.model_dump(mode="json"),
                        engine_trace=trace.model_dump(mode="json"), oracle=oracle,
                        result=event.payload["result"], before_state_hash=world.state_hash,
                        after_state_hash=engine.state.state_hash)
    return engine.state, row


def _event_replay(initial_world, task, profile, record):
    errors, world, operation_count = [], initial_world.clone(), 0
    registered = dict(world.external_events)
    for external in record.scheduled_external_events:
        if external.id in registered:
            errors.append("scheduled external event ID is duplicated")
            break
        registered[external.id] = external.model_copy(deep=True)
    world.external_events = registered
    if world.state_hash != record.runtime_initial_world.state_hash:
        errors.append("runtime initial world is not derived from the frozen world and event schedule")
    if initial_world.state_hash != record.initial_world.state_hash:
        errors.append("Replay input differs from the Episode's frozen initial world")
    if task.record_hash != record.task.record_hash or profile.record_hash != record.profile.record_hash:
        errors.append("Replay task or Customer profile differs from the Episode")
    customer = CustomerRuntime(profile, allow_legacy=record.task.protocol == "commerce-task-v1",
                               task=record.task)
    tool_iter = iter(record.tool_executions)
    reconstructed_facts, reconstructed_messages, reconstructed_calls = [], [], []
    scheduled_ids = {row.id for row in record.scheduled_external_events} | set(initial_world.external_events)
    applied_ids = set()
    active_presentation = None
    for event in record.events:
        try:
            if event.index >= 0:
                expected_previous = record.events[event.index - 1].record_hash if event.index else ""
                if event.previous_event_hash != expected_previous:
                    raise ValueError("event hash-chain predecessor differs")
                if event.record_hash != EpisodeEvent.model_validate(event.model_dump()).record_hash:
                    raise ValueError("event content hash differs")
            if event.before_state_hash != world.state_hash:
                raise ValueError("event before-state hash differs")
            if event.kind == "customer_request":
                if event.payload.get("request") != task.customer_request:
                    raise ValueError("initial public request differs from TaskSpec")
            elif event.kind in ("agent_response", "agent_error"):
                call_hash = event.payload.get("agent_call_hash")
                role = event.payload.get("role")
                if role not in ("customer", "service"):
                    raise ValueError("agent event has an unknown role")
                if call_hash is None:
                    if event.kind == "agent_response":
                        # Deterministic test adapters legitimately have no online call.
                        pass
                else:
                    if len(reconstructed_calls) >= len(record.model_calls):
                        raise ValueError("agent event references a missing model request")
                    model_call = record.model_calls[len(reconstructed_calls)]
                    if model_call.record_hash != call_hash or model_call.role != role:
                        raise ValueError("model request/response order or role differs from event history")
                    reconstructed_calls.append(model_call)
            elif event.kind == "customer_transition":
                kind = event.payload.get("transition_kind")
                if kind == "initialize":
                    result = customer.initialize(world, task.customer_request, event_index=event.index)
                elif kind == "decision":
                    decision = CustomerDecision.model_validate(event.payload["decision"])
                    presentation = (OfferPresentation.model_validate(event.payload["presentation"])
                                    if event.payload.get("presentation") else None)
                    result = customer.apply(world, decision, presentation=presentation, event_index=event.index)
                else:
                    raise ValueError("unknown Customer transition")
                world = result.world
                if (result.event.record_hash != event.payload.get("customer_event_hash")
                        or result.event.model_dump(mode="json") != event.payload.get("customer_event")):
                    raise ValueError("Customer-originated authority proof changed")
                if [row.model_dump(mode="json") for row in result.authority_proofs] != event.payload.get("authority_proofs"):
                    raise ValueError("Customer authority proofs differ from the executed transition")
                if [row.model_dump(mode="json") for row in result.disclosed_facts] != event.payload.get("disclosed_facts", []):
                    raise ValueError("Customer disclosure facts differ from the executed transition")
                reconstructed_facts.extend(result.disclosed_facts)
                supplied = (OfferPresentation.model_validate(event.payload["presentation"])
                            if event.payload.get("presentation") else None)
                if supplied != active_presentation:
                    raise ValueError("Customer consent is not bound to the offer actually presented")
                active_presentation = None
            elif event.kind == "service_tool":
                try:
                    row = next(tool_iter)
                except StopIteration:
                    raise ValueError("operation event has no matching ToolExecution") from None
                if event.payload.get("tool_execution_hash") != row.record_hash:
                    raise ValueError("tool execution reference differs")
                trace = Trace.model_validate(row.engine_trace)
                request = Request.model_validate(row.request)
                operation = Operation.model_validate(row.operation)
                if (event.payload.get("request_id") != request.id
                        or event.payload.get("operation") != row.operation
                        or request.operation != row.operation
                        or request.identity != task.customer_id):
                    raise ValueError("tool event, operation and exact execution authority differ")
                if row.before_state_hash != world.state_hash:
                    raise ValueError("tool execution starts from another business state")
                replayed = replay_trace(adapter, trace, {request.id: request})
                if trace.initial_state != world.model_dump(mode="json"):
                    raise ValueError("tool trace starts from another world")
                recalculated = evaluate(adapter, trace, {request.id: request})
                if recalculated != row.oracle:
                    raise ValueError("independent Oracle result differs")
                operation_event = next((item for item in trace.events if item.kind == "operation"), None)
                if (operation_event is None or operation_event.payload.get("result") != row.result
                        or event.payload.get("result") != row.result
                        or row.after_state_hash != replayed.state.state_hash):
                    raise ValueError("tool result or resulting state differs from its native trace")
                if row.result.get("success"):
                    reconstructed_facts.extend(_facts_for_result(
                        replayed.state, operation, row.result.get("data", {}),
                        event.record_hash, event.index))
                world = replayed.state
                operation_count += 1
            elif event.kind == "external_event":
                event_id = event.payload["event_id"]
                if event_id not in scheduled_ids or event_id in applied_ids:
                    raise ValueError("external event is absent from the frozen schedule or is repeated")
                expected = reference_event(world, event_id)
                actual = apply_event(world, event_id)
                if expected.state_hash != actual.state_hash or event.payload.get("reference_state_hash") != expected.state_hash:
                    raise ValueError("external event projection differs from reference")
                world = actual
                applied_ids.add(event_id)
            elif event.kind == "offer_presented":
                presentation = OfferPresentation.model_validate(event.payload)
                quote = world.quotes.get(presentation.quote_id)
                allowed_status = "prepared" if presentation.operation == "commit_accepted_quote" else "committed"
                expected = render_offer(world, presentation.quote_id, presentation.operation,
                                        presentation.event_id, presentation.event_index,
                                        presentation.service_message,
                                        expected_customer_id=task.customer_id)
                if (quote is None or quote.customer_id != task.customer_id or quote.status != allowed_status
                        or digest(quote) != presentation.quote_hash or presentation != expected
                        or presentation.event_index != event.index
                        or presentation.event_id != "offer_" + str(event.index)):
                    raise ValueError("offer presentation is not bound to the current business quote")
                active_presentation = presentation
            elif event.kind == "service_reply":
                message = ServiceMessage.model_validate(event.payload["message"])
                if event.payload["message"] != next((row.model_dump(mode="json") for row in record.service_messages
                                                       if row.message_id == message.message_id), None):
                    raise ValueError("ServiceMessage artifact differs from the delivered event")
                reconstructed_messages.append(message)
                offered_hash = event.payload.get("presented_offer_hash")
                if offered_hash != (active_presentation.record_hash if active_presentation else None):
                    raise ValueError("Service reply and exact offer presentation are not bound")
                if active_presentation and active_presentation.service_message != message.text:
                    raise ValueError("Service prose differs from the prose bound to the presented quote")
            elif event.kind == "termination":
                if (event.payload.get("status") != record.status
                        or event.payload.get("reason") != record.terminal_reason
                        or event.index != len(record.events) - 1):
                    raise ValueError("termination event differs from the Episode result")
            else:
                raise ValueError("unknown event kind")
            if event.after_state_hash != world.state_hash:
                raise ValueError("event after-state hash differs")
        except Exception as exc:
            errors.append("event {}: {}".format(event.index, str(exc)))
            break
    if sum(1 for _ in record.tool_executions) != operation_count:
        errors.append("not every tool execution was replayed")
    if tuple(reconstructed_facts) != record.facts:
        errors.append("FactEvidence cannot be reconstructed from observed tools and Customer messages")
    if tuple(reconstructed_messages) != record.service_messages:
        errors.append("delivered Service messages cannot be reconstructed from the event log")
    if tuple(reconstructed_calls) != record.model_calls:
        errors.append("model request/response records cannot be reconstructed from agent events")
    if world.state_hash != record.final_world.state_hash:
        errors.append("final world differs")
    return EpisodeReplay(verified=not errors, event_count=len(record.events), operation_count=operation_count,
                         final_world_hash=world.state_hash, errors=tuple(errors),
                         outcome_hash=record.outcome.record_hash)


def replay_episode(record: EpisodeRecord):
    record = record.checked()
    replay = _event_replay(record.initial_world, record.task, record.profile, record)
    errors = list(replay.errors)
    try:
        goals = evaluate_state_goals(record.initial_world, record.final_world, record.task)
        compliance = (evaluate_task_compliance(record.initial_world, record.final_world, record.task)
                      + (evaluate_execution_compliance(record.tool_executions),))
        end_index = max((row.event_index for row in record.service_messages), default=0)
        communication = evaluate_task_communication(record.initial_world, record.task, facts=record.facts,
                                                     messages=record.service_messages, end_event_index=end_index)
        if communication != record.communication:
            errors.append("communication evaluation differs")
        if record.outcome.protocol == "commerce-outcome-v2":
            expected_outcome = evaluate_outcome_v2(
                goals, compliance, communication, record.tool_executions,
                replay_verified=replay.verified, episode_status=record.status)
        else:
            expected_outcome = evaluate_outcome(goals, compliance, communication,
                                                replay_verified=replay.verified,
                                                valid_end=record.status == "customer_finished")
        if expected_outcome != record.outcome:
            errors.append("task outcome evaluation differs")
    except Exception as exc:
        errors.append("outcome replay: {}".format(str(exc)))
    return EpisodeReplay(verified=not errors, event_count=replay.event_count,
                         operation_count=replay.operation_count, final_world_hash=replay.final_world_hash,
                         errors=tuple(errors), outcome_hash=record.outcome.record_hash)


def run_episode(initial_world: World, task: Union[TaskSpecV3, TaskSpecV2, TaskSpec],
                profile: Union[CustomerProfileV3, CustomerProfileV2, HumanCustomerProfileV2, CustomerProfile],
                customer_agent: CustomerAgent, service_agent: ServiceAgent, *,
                budget: Optional[EpisodeBudget] = None, mode: Literal["reference", "sandbox"] = "reference",
                external_events: Tuple[ExternalEvent, ...] = (),
                on_progress: Optional[Callable[[dict], None]] = None,
                cancel_check: Optional[Callable[[], bool]] = None) -> EpisodeRecord:
    """Run a complete independent episode; task targets never enter Service context."""
    initial_world, task, profile = initial_world.clone(), task.checked(), profile.checked()
    budget = (budget or EpisodeBudget()).checked()
    if mode not in ("reference", "sandbox"):
        raise ValueError("unknown episode mode")
    check_task_scope(initial_world, task)
    if task.protocol == "commerce-task-v1" and task.remedy_goals:
        raise ValueError(
            "fresh transaction Episodes require a public-request-grounded TaskSpecV2 or structured TaskSpecV3"
        )
    if profile.customer_id != task.customer_id:
        raise ValueError("Customer identity differs from the fixed TaskSpec identity")
    if profile.delegated_line_ids and profile.protocol not in (
            "commerce-customer-profile-v2", "commerce-customer-profile-v3"):
        raise ValueError("new Episodes require a versioned Customer intent binding")
    if task.protocol == "commerce-task-v2":
        if profile.protocol != "commerce-customer-profile-v2":
            raise ValueError("TaskSpecV2 requires its historical CustomerProfileV2 binding")
        expected_delegations = {goal.line_id for goal in task.remedy_goals}
        if set(profile.delegated_line_ids) != expected_delegations:
            raise ValueError("Customer delegations must match exactly the publicly requested transaction targets")
        for goal in task.remedy_goals:
            profile_span = profile.delegation_request_spans.get(goal.line_id)
            if (goal.line_id not in profile.delegated_line_ids
                    or profile_span not in (goal.public_request_span, goal.subject_mention)):
                raise ValueError("Customer delegation must match the goal's public request span or subject wording")
    elif task.protocol == "commerce-task-v3":
        if profile.protocol != "commerce-customer-profile-v3":
            raise ValueError("TaskSpecV3 requires a structured CustomerProfileV3 binding")
        expected_delegations = tuple(goal.line_id for goal in task.remedy_goals)
        if profile.delegated_line_ids != expected_delegations:
            raise ValueError("CustomerProfileV3 scope differs from structured remedy goals")
        if profile.intent_bundle_hash != task.request_bundle.record_hash:
            raise ValueError("CustomerProfileV3 is not bound to the task request bundle")
    base_world = initial_world.clone()
    runtime_initial_world = initial_world.clone()
    scheduled_ids = [event.id for event in external_events]
    if len(set(scheduled_ids)) != len(scheduled_ids):
        raise ValueError("external event schedule repeats an event ID")
    if any(event.id in runtime_initial_world.external_events for event in external_events):
        raise ValueError("external event IDs must be unique and pre-registered")
    for event in external_events:
        runtime_initial_world.external_events[event.id] = event.model_copy(deep=True)
    scheduled_external_events = tuple(event.model_copy(deep=True) for event in external_events)
    if adapter.invariants(runtime_initial_world):
        raise ValueError("initial world or frozen event schedule violates business invariants")

    world, events, tools, service_messages, facts, calls = runtime_initial_world.clone(), [], [], [], [], []
    conversation = [AgentMessage(role="user", content=task.customer_request)]
    events = _append(events, "customer_request", {"request": task.customer_request,
                         "customer_id": task.customer_id}, world, world)
    customer_runtime = CustomerRuntime(profile, allow_legacy=task.protocol == "commerce-task-v1", task=task)
    initialized = customer_runtime.initialize(world, task.customer_request, event_index=len(events))
    before = world
    world = initialized.world
    events = _append(events, "customer_transition", {"transition_kind": "initialize",
                         "customer_event_hash": initialized.event.record_hash,
                         "customer_event": initialized.event.model_dump(mode="json"),
                         "authority_proofs": [row.model_dump(mode="json") for row in initialized.authority_proofs]}, before, world)

    status, terminal_reason = "budget_exhausted", "service_decision_budget_exhausted"
    current_offer = None
    # Quotes awaiting Customer review are queued from native operation results,
    # never inferred from Service prose or selected by scanning World state.
    pending_customer_offers = []
    observed_ids = {task.customer_id}
    request_counter = customer_turns = service_decisions = tool_count = model_calls = 0
    service_uses_model = bool(getattr(service_agent, "uses_model", True))
    customer_uses_model = bool(getattr(customer_agent, "uses_model", True))
    while service_decisions < budget.max_service_decisions and tool_count < budget.max_service_tool_calls:
        if cancel_check and cancel_check():
            status, terminal_reason = "cancelled", "human_session_cancelled"
            break
        if service_uses_model and model_calls >= budget.max_model_calls:
            terminal_reason = "episode_model_call_budget_exhausted"
            break
        prompt = ServicePrompt(customer_id=task.customer_id, customer_request=task.customer_request,
                               conversation=tuple(conversation),
                               available_merchant_scope_ids=tuple(sorted(runtime_initial_world.merchants)),
                               observed_subject_ids=tuple(sorted(observed_ids)))
        _notify_progress(on_progress, "service_running", service_decisions=service_decisions,
                         tool_calls=tool_count, message="正在调用 Service 模型。")
        if hasattr(service_agent, "last_call"):
            service_agent.last_call = None
        try:
            action = service_agent.decide(prompt)
        except Exception as exc:
            failed_call = getattr(service_agent, "last_call", None)
            if failed_call is not None:
                before = world
                calls.append(failed_call)
                model_calls += 1
                events = _append(events, "agent_error", {"role": "service",
                                    "agent_call_hash": failed_call.record_hash,
                                    "error_type": type(exc).__name__}, before, world)
            else:
                events = _append(events, "agent_error", {"role": "service", "agent_call_hash": None,
                                    "error_type": type(exc).__name__}, world, world)
            if cancel_check and cancel_check():
                status, terminal_reason = "cancelled", "human_session_cancelled_during_service_call"
            else:
                status, terminal_reason = "agent_error", type(exc).__name__ + ": " + str(exc)
            break
        service_decisions += 1
        service_call = action.call or getattr(service_agent, "last_call", None)
        if service_uses_model and service_call is None:
            status, terminal_reason = "service_protocol_error", "model-backed Service did not record its request"
            break
        if action.call and getattr(service_agent, "last_call", None) not in (None, action.call):
            actual_call = getattr(service_agent, "last_call", None)
            if actual_call is not None:
                calls.append(actual_call)
                model_calls += 1
                events = _append(events, "agent_error", {
                    "role": "service", "agent_call_hash": actual_call.record_hash,
                    "claimed_action_call_hash": action.call.record_hash,
                    "error_type": "ServiceCallLogMismatch"}, world, world)
            status, terminal_reason = "service_protocol_error", "Service decision and request log disagree"
            break
        model_calls += int(service_call is not None)
        if service_call:
            calls.append(service_call)
        events = _append(events, "agent_response", {"role": "service",
                            "agent_call_hash": service_call.record_hash if service_call else None,
                            "decision_kind": action.kind}, world, world)
        if cancel_check and cancel_check():
            status, terminal_reason = "cancelled", "human_session_cancelled"
            break
        if action.kind == "operation":
            if action.operation is None:
                status, terminal_reason = "service_protocol_error", "operation action missing native operation"
                break
            if tool_count >= budget.max_service_tool_calls:
                terminal_reason = "service_tool_call_budget_exhausted"
                break
            before = world
            request_id = "episode_request_{:06d}".format(request_counter)
            request_counter += 1
            try:
                world, row = _execution(world, task.customer_id, action.operation, request_id, mode)
            except Exception as exc:
                status, terminal_reason = "runner_error", type(exc).__name__ + ": " + str(exc)
                break
            tools.append(row)
            tool_count += 1
            tool_event_index = len(events)
            operation_event_payload = {"request_id": request_id, "tool_execution_hash": row.record_hash,
                                       "operation": row.operation, "result": row.result}
            events = _append(events, "service_tool", operation_event_payload, before, world)
            _notify_progress(on_progress, "service_tool", operation=row.operation,
                             result=row.result, before_state_hash=row.before_state_hash,
                             after_state_hash=row.after_state_hash, tool_count=tool_count)
            observed_ids.update(_public_ids(row.result))
            if row.result.get("success"):
                # A successful tool result confirms the exact requested public
                # object ID was valid and observed; failed guesses do not enter
                # the Service's visible-object set.
                observed_ids.add(action.operation.resource_id)
                facts.extend(_facts_for_result(world, action.operation, row.result.get("data", {}),
                                               events[-1].record_hash, tool_event_index))
                if action.operation.name == "prepare_remedy_quote":
                    data = row.result.get("data")
                    quote_data = data.get("quote") if isinstance(data, dict) else None
                    quote_id_value = quote_data.get("id") if isinstance(quote_data, dict) else None
                    quote = world.quotes.get(quote_id_value) if isinstance(quote_id_value, str) else None
                    if quote is None or quote.model_dump(mode="json") != quote_data:
                        status, terminal_reason = "runner_error", (
                            "successful quote preparation did not return the quote in the resulting World")
                        break
                    pending_customer_offers.append((quote.id, "commit_accepted_quote"))
                elif action.operation.name == "cancel_pending_commitment":
                    pending_customer_offers[:] = [row for row in pending_customer_offers
                                                   if row[0] != action.operation.resource_id]
            # Mirror the exact one-tool-call exchange in the visible Service history.
            if action.call:
                raw_message = action.call.response.get("choices", [{}])[0].get("message", {})
                conversation.append(AgentMessage(role="assistant", content=raw_message.get("content"),
                                                 tool_calls=tuple(raw_message.get("tool_calls", []))))
            conversation.append(AgentMessage(role="tool", tool_call_id=action.call_id or request_id,
                                             content=json.dumps(row.result, ensure_ascii=False, sort_keys=True)))
            # Frozen review/return outcomes become visible only after their
            # registered prerequisite occurs. Both projections must agree.
            for ext in sorted(external_events, key=lambda item: (item.at, item.id)):
                current = world.external_events[ext.id]
                if ext.id in world.applied_events:
                    continue
                ready = ((ext.kind == "review_result" and ext.target_id in world.cases
                          and world.cases[ext.target_id].review_requested)
                         or (ext.kind == "return_received" and any(
                             row.case_id == ext.target_id and row.status == "scheduled"
                             for row in world.returns.values())))
                if not ready:
                    continue
                event_before = world
                try:
                    expected = reference_event(world, ext.id)
                    applied = apply_event(world, ext.id)
                    if expected.state_hash != applied.state_hash:
                        raise ValueError("external event projection differs from Reference")
                    world = applied
                    events = _append(events, "external_event", {"event_id": ext.id,
                                        "reference_state_hash": expected.state_hash}, event_before, world)
                    if ext.kind == "return_received":
                        returned_case = world.cases[ext.target_id]
                        returned_quote = world.quotes[returned_case.committed_quote_id]
                        if returned_quote.remedy == "refund":
                            pending_customer_offers.append((returned_quote.id, "release_accepted_refund"))
                except Exception as exc:
                    status, terminal_reason = "runner_error", type(exc).__name__ + ": " + str(exc)
                    break
                conversation.append(AgentMessage(role="tool", tool_call_id="external-" + ext.id,
                    content=json.dumps({"external_event": ext.kind, "status": "applied",
                                        "target_id": ext.target_id}, ensure_ascii=False)))
            if status == "runner_error":
                break
            continue

        if action.kind != "reply" or not action.message.strip():
            status, terminal_reason = "service_protocol_error", "Service must submit one operation or a nonempty reply"
            break
        if not set(action.subject_ids).issubset(observed_ids):
            status, terminal_reason = "service_protocol_error", "Service message references unobserved business objects"
            break
        event_index = len(events)
        if pending_customer_offers:
            # The Service's ordinary reply is enough to deliver the exact
            # quote created by the native Engine; no quote-ID tool choice needed.
            quote_id_value, present_for = pending_customer_offers[0]
            try:
                current_offer = render_offer(world, quote_id_value, present_for,
                                             "offer_" + str(event_index), event_index, action.message,
                                             expected_customer_id=task.customer_id)
            except ValueError as exc:
                status, terminal_reason = "service_protocol_error", str(exc)
                break
            pending_customer_offers.pop(0)
            events = _append(events, "offer_presented", current_offer.model_dump(mode="json"), world, world)
            customer_text = action.message + "\n" + current_offer.rendered_terms
        elif action.present_quote_id:
            # Retain the direct in-process adapter path for existing scripted
            # callers. Model-facing Service tools no longer expose this choice.
            if action.present_quote_id not in observed_ids:
                status, terminal_reason = "service_protocol_error", "Service can present only a previously observed quote"
                break
            try:
                current_offer = render_offer(world, action.present_quote_id, action.present_for,
                                             "offer_" + str(event_index), event_index, action.message,
                                             expected_customer_id=task.customer_id)
            except ValueError as exc:
                status, terminal_reason = "service_protocol_error", str(exc)
                break
            events = _append(events, "offer_presented", current_offer.model_dump(mode="json"), world, world)
            customer_text = action.message + "\n" + current_offer.rendered_terms
        else:
            current_offer = None
            customer_text = action.message
        service_message = ServiceMessage(message_id="service_message_{:06d}".format(event_index),
                                         event_index=event_index, text=action.message,
                                         subject_ids=action.subject_ids)
        service_messages.append(service_message)
        events = _append(events, "service_reply", {"message": service_message.model_dump(mode="json"),
                                                    "presented_offer_hash": current_offer.record_hash if current_offer else None}, world, world)
        _notify_progress(on_progress, "service_reply", message=service_message.model_dump(mode="json"),
                         offer=current_offer.model_dump(mode="json") if current_offer else None)
        conversation.append(AgentMessage(role="assistant", content=customer_text))

        if customer_turns >= budget.max_customer_turns or (
                customer_uses_model and model_calls >= budget.max_model_calls):
            terminal_reason = "customer_turn_or_model_call_budget_exhausted"
            break
        cp = CustomerPrompt(public_request=task.customer_request, last_service_message=customer_text,
                            offer=current_offer, conversation=tuple(conversation))
        if hasattr(customer_agent, "last_call"):
            customer_agent.last_call = None
        try:
            decision = customer_agent.decide(cp, profile)
        except EpisodeCancelled as exc:
            status, terminal_reason = "cancelled", str(exc) or "human_session_cancelled"
            break
        except Exception as exc:
            failed_call = getattr(customer_agent, "last_call", None)
            if failed_call is not None:
                calls.append(failed_call)
                model_calls += 1
                events = _append(events, "agent_error", {"role": "customer",
                                    "agent_call_hash": failed_call.record_hash,
                                    "error_type": type(exc).__name__}, world, world)
            else:
                events = _append(events, "agent_error", {"role": "customer", "agent_call_hash": None,
                                    "error_type": type(exc).__name__}, world, world)
            status, terminal_reason = "customer_protocol_error", type(exc).__name__ + ": " + str(exc)
            break
        customer_turns += 1
        last_customer_call = getattr(customer_agent, "last_call", None)
        if customer_uses_model and last_customer_call is None:
            status, terminal_reason = "customer_protocol_error", "model-backed Customer did not record its request"
            break
        if last_customer_call is not None:
            calls.append(last_customer_call)
            model_calls += 1
        events = _append(events, "agent_response", {"role": "customer",
                            "agent_call_hash": last_customer_call.record_hash if last_customer_call else None,
                            "decision_kind": "customer_turn"}, world, world)
        before = world
        try:
            transition = customer_runtime.apply(world, decision, presentation=current_offer, event_index=len(events))
        except Exception as exc:
            status, terminal_reason = "customer_protocol_error", type(exc).__name__ + ": " + str(exc)
            break
        world = transition.world
        event_payload = {"transition_kind": "decision", "decision": decision.model_dump(mode="json"),
                         "presentation": current_offer.model_dump(mode="json") if current_offer else None,
                         "customer_event_hash": transition.event.record_hash,
                         "customer_event": transition.event.model_dump(mode="json"),
                         "authority_proofs": [row.model_dump(mode="json") for row in transition.authority_proofs],
                         "disclosed_facts": [row.model_dump(mode="json") for row in transition.disclosed_facts]}
        events = _append(events, "customer_transition", event_payload, before, world)
        facts.extend(transition.disclosed_facts)
        if decision.message:
            conversation.append(AgentMessage(role="user", content=decision.message))
        # A consent made before finish must always be offered to Service. The
        # Customer transition marks finish_allowed=false in that case.
        if decision.finish and transition.event.finish_allowed:
            status, terminal_reason = "customer_finished", "customer_finished_after_final_service_reply"
            events = _append(events, "termination", {"status": status, "reason": terminal_reason}, world, world)
            break
        if transition.event.rejection_reasons:
            conversation.append(AgentMessage(role="user", content=(
                "A previously recorded Customer consent is still pending execution. Please continue the requested workflow.")))
        current_offer = None
    else:
        terminal_reason = "service_decision_or_tool_call_budget_exhausted"

    if status == "budget_exhausted" and tool_count >= budget.max_service_tool_calls:
        terminal_reason = "service_tool_call_budget_exhausted"
    if not events or events[-1].kind != "termination":
        events = _append(events, "termination", {"status": status, "reason": terminal_reason}, world, world)

    # Independent result evaluation; the Service never sees these targets/results.
    state_goals = evaluate_state_goals(base_world, world, task)
    compliance = evaluate_task_compliance(base_world, world, task) + (evaluate_execution_compliance(tools),)
    end_index = max((row.event_index for row in service_messages), default=0)
    communication = evaluate_task_communication(base_world, task, facts=facts, messages=service_messages,
                                                 end_event_index=end_index)
    # First build a provisional record, replay all event/operation evidence, then seal the result.
    placeholder = evaluate_outcome_v2(
        state_goals, compliance, communication, tools,
        replay_verified=False, episode_status=status)
    provisional = EpisodeRecord(task=task, profile=profile, budget=budget, mode=mode,
        initial_world=base_world, runtime_initial_world=runtime_initial_world,
        final_world=world, status=status, terminal_reason=terminal_reason,
        scheduled_external_events=scheduled_external_events,
        events=tuple(events), tool_executions=tuple(tools), service_messages=tuple(service_messages),
        facts=tuple(facts), model_calls=tuple(calls), communication=communication,
        outcome=placeholder, replay_verified=False)
    replay = _event_replay(initial_world, task, profile, provisional)
    outcome = evaluate_outcome_v2(
        state_goals, compliance, communication, tools,
        replay_verified=replay.verified, episode_status=status)
    final_data = provisional.model_dump(mode="json", exclude={"record_hash"})
    final_data.update({"outcome": outcome.model_dump(mode="json"), "replay_verified": replay.verified})
    return EpisodeRecord.model_validate(final_data)


def save_episode(record: EpisodeRecord, path):
    """Write a private, non-overwriting episode artifact; it includes Customer-private data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = (record.model_dump_json(indent=2) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError:
        raise
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return {"path": str(path), "record_hash": record.record_hash,
            "replay": replay_episode(record).model_dump(mode="json"), "costs": record.costs}


def verify_episode_artifact(path):
    """Load a private Episode JSON artifact and verify hashes, Engine, Oracle and outcomes offline."""
    record = EpisodeRecord.model_validate_json(Path(path).read_text(encoding="utf-8"))
    replay = replay_episode(record)
    return {"verified": replay.verified, "record_hash": record.record_hash,
            "replay": replay.model_dump(mode="json"), "costs": record.costs,
            "strict_task_success": record.outcome.strict_task_success,
            "final_goal_reached": record.outcome.final_goal_reached}
