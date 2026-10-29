"""Deterministic post-run evidence projection; never a semantic blame judge.

Native adapters add exact trusted inputs/tool results with artifact pointers.
Without that information the wire-only projection explicitly leaves grants,
turn linkage and responsibility unknown. Neither projection enters Agent runtime.
"""
from __future__ import annotations

import json
from typing import Dict, Literal, Optional, Tuple
from pydantic import Field, JsonValue, model_validator

from ..models import EpisodeEvidence
from ..schema import FrozenModel, canonical
from ..skills.models import Hash


class CustomerRequestEvidence(FrozenModel):
    turn_id: str
    message: str
    declared_write_intents: Tuple[Dict[str, JsonValue], ...] = ()
    accepted_write_grants: Optional[Tuple[Dict[str, JsonValue], ...]] = None
    authorization_source: Literal["native_session_input", "unknown"] = "unknown"
    source_refs: Tuple[str, ...]


class ServiceToolEvidence(FrozenModel):
    order: int = Field(ge=1)
    call_id: str
    name: str
    arguments_raw: str
    arguments: Optional[Dict[str, JsonValue]] = None
    result: Optional[JsonValue] = None
    turn_id: Optional[str] = None
    event_id: Optional[str] = None
    parameter_origin: Literal["service_tool_call", "matches_customer_write_intent", "unknown"] = "unknown"
    oracle_classification: Optional[str] = None
    oracle_operation_correct: Optional[bool] = None
    service_decision_correctness: Literal["unknown"] = "unknown"
    fault_attribution: Literal["unknown"] = "unknown"
    source_refs: Tuple[str, ...]

    @model_validator(mode="after")
    def parsed_arguments_match_raw(self):
        parsed = json_object(self.arguments_raw)
        if canonical(self.arguments) != canonical(parsed):
            raise ValueError("parsed arguments differ from the recorded Service arguments")
        return self


class InteractionEvidenceView(FrozenModel):
    schema_version: Literal[1] = 1
    episode_id: str
    evidence_hash: Hash
    trace_sha256: Hash
    source: Literal["verified_native", "wire_only"]
    customer_requests: Tuple[CustomerRequestEvidence, ...] = ()
    service_tools: Tuple[ServiceToolEvidence, ...] = ()
    service_replies: Tuple[Dict[str, JsonValue], ...] = ()
    customer_visible_context: Tuple[JsonValue, ...] = ()
    oracle_events: Tuple[Dict[str, JsonValue], ...] = ()
    confirmed_business_failures: Tuple[Dict[str, JsonValue], ...] = ()
    oracle_visibility: Literal["post_run_verifier_only"] = "post_run_verifier_only"
    interpretation: str = (
        "Tool errors are observations, not blame. Oracle operation correctness describes "
        "Engine policy compliance, not Service reasoning correctness. Responsibility and "
        "untested hypotheses remain unknown; Service replies are unverified claims."
    )


def json_object(raw):
    """Ambiguous/malformed arguments cannot support provenance assertions."""
    if not isinstance(raw, str):
        return None
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result
    def finite(value):
        raise ValueError("non-finite JSON")
    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=finite)
        return value if isinstance(value, dict) else None
    except (ValueError, TypeError):
        return None


def _message(call):
    response = call.response
    if not isinstance(response, dict):
        return {}
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return {}
    message = choices[0].get("message")
    return message if isinstance(message, dict) else {}


def wire_view(evidence: EpisodeEvidence) -> InteractionEvidenceView:
    requests, tools, replies, visible, results = [], [], [], [], {}
    for call in evidence.model_calls:
        for msg in call.base_request.get("messages", []):
            if isinstance(msg, dict) and msg.get("role") == "tool" and isinstance(msg.get("tool_call_id"), str):
                value = msg.get("content")
                parsed = json_object(value)
                results.setdefault(msg["tool_call_id"], parsed if parsed is not None else value)
    for index, call in enumerate(evidence.model_calls):
        message = _message(call)
        ref = "/model_calls/" + str(index)
        if call.role == "customer":
            if not visible:
                visible = [msg.get("content") for msg in call.base_request.get("messages", [])
                           if isinstance(msg, dict) and msg.get("role") == "user"]
            action = json_object(message.get("content"))
            if action and action.get("type") == "send_message" and isinstance(action.get("message"), str):
                declared = action.get("requested_operations", [])
                requests.append(CustomerRequestEvidence(turn_id=call.id, message=action["message"],
                    declared_write_intents=tuple(row for row in declared if isinstance(row, dict)) if isinstance(declared, list) else (),
                    source_refs=(ref + "/response",)))
        elif call.role == "service":
            if isinstance(message.get("content"), str) and message["content"]:
                replies.append(dict(model_call_id=call.id, content=message["content"], status="unverified_text", source_ref=ref + "/response"))
            calls = message.get("tool_calls", [])
            for number, tool in enumerate(calls if isinstance(calls, list) else []):
                if not isinstance(tool, dict) or not isinstance(tool.get("function"), dict):
                    continue
                function = tool["function"]
                if not isinstance(function.get("name"), str) or not isinstance(function.get("arguments"), str):
                    continue
                call_id = str(tool.get("id", "unknown"))
                tools.append(ServiceToolEvidence(order=len(tools)+1, call_id=call_id, name=function["name"],
                    arguments_raw=function["arguments"], arguments=json_object(function["arguments"]), result=results.get(call_id),
                    parameter_origin="service_tool_call", source_refs=(ref + "/response/choices/0/message/tool_calls/" + str(number),)))
    return InteractionEvidenceView(episode_id=evidence.episode_id, evidence_hash=evidence.evidence_hash,
        trace_sha256=evidence.trace.sha256, source="wire_only", customer_requests=tuple(requests), service_tools=tuple(tools),
        service_replies=tuple(replies), customer_visible_context=tuple(visible),
        oracle_events=tuple(row.model_dump(mode="json") for row in evidence.events),
        confirmed_business_failures=tuple(row.model_dump(mode="json") for row in evidence.failures))


def native_view(evidence, requests, tool_records, replies, visible_context, argument_style="nested"):
    """Project already replay-verified native facts, retaining exact source pointers."""
    customer = tuple(CustomerRequestEvidence.model_validate(row) for row in requests)
    events = {row.event_id: row for row in evidence.events}
    tools = []
    for number, record in enumerate(tool_records):
        args = json_object(record["arguments"])
        turn_id = record.get("turn_id")
        origin = "service_tool_call"
        if args is not None and turn_id is not None:
            for request in customer:
                if request.turn_id != turn_id:
                    continue
                for intent in request.declared_write_intents:
                    operation = dict(intent)
                    name = operation.pop("name", None)
                    if argument_style == "flattened":
                        operation.update(operation.pop("params", {}))
                    if name == record["name"] and canonical(operation) == canonical(args):
                        origin = "matches_customer_write_intent"
        event = events.get(record.get("event_id"))
        tools.append(ServiceToolEvidence(order=number+1, call_id=record["call_id"], name=record["name"],
            arguments_raw=record["arguments"], arguments=args, result=record["response"], turn_id=turn_id,
            event_id=record.get("event_id") or None, parameter_origin=origin,
            oracle_classification=event.classification if event else None,
            oracle_operation_correct=event.operation_correct if event else None,
            source_refs=tuple(record["source_refs"])))
    return InteractionEvidenceView(episode_id=evidence.episode_id, evidence_hash=evidence.evidence_hash,
        trace_sha256=evidence.trace.sha256, source="verified_native", customer_requests=customer, service_tools=tuple(tools),
        service_replies=tuple(replies), customer_visible_context=tuple(visible_context),
        oracle_events=tuple(row.model_dump(mode="json") for row in evidence.events),
        confirmed_business_failures=tuple(row.model_dump(mode="json") for row in evidence.failures))
