"""Audit actual events against trusted requests and the independent reference policy."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import Field, ValidationError

from usami.events import Event, Trace
from usami.protocol import ExecutionResult, Operation, RequestContext
from usami.world.models import Model, RULES_VERSION, WorldState, digest
from usami.world.rules import ReferenceRules, check_invariants


class Violation(Model):
    code: str
    entity_id: str
    event_id: Optional[str] = None
    state_hash: Optional[str] = None
    state_path: Optional[str] = None
    detail: str


class RequestEvaluation(Model):
    world_id: str
    request_id: str
    valid_request: Optional[bool]
    operation_correct: Optional[bool]
    violations: List[Violation] = Field(default_factory=list)
    evidence_event_ids: List[str] = Field(default_factory=list)


class EvaluationReport(Model):
    world_id: str
    trace_valid: bool
    status: str
    requests: List[RequestEvaluation]
    violations: List[Violation] = Field(default_factory=list)
    final_state_hash: str
    metrics: Dict[str, Union[int, float]] = Field(default_factory=dict)


def evaluate_trace(trace: Union[Trace, str, Path],
                   trusted_requests: Optional[Dict[str, RequestContext]] = None) -> EvaluationReport:
    """Offline trace assumes harness request events are trusted input.

    Use trusted_requests from an external harness/manifest when the executor may
    tamper with traces. Hashes are integrity checks, not authentication signatures.
    Unknown is represented by null booleans and report status='unknown'.
    """
    if not isinstance(trace, Trace):
        trace = Trace.load(trace)
    trace = trace.model_copy(deep=True)
    global_errors: List[Violation] = []
    evaluations: Dict[str, RequestEvaluation] = {}
    contexts: Dict[str, RequestContext] = {}
    request_events: Dict[str, str] = {}
    recorded_request_ids = set()
    counts: Dict[str, int] = {}
    state = trace.initial_state.clone()
    rules = ReferenceRules()
    expected_schema = "usami-trace-0.2.0" if state.rules.version == "saas-rules-0.2.0" else "usami-trace-0.1.0"
    unsupported = trace.initial_state.rules.version not in {RULES_VERSION, "saas-rules-0.2.0"} or trace.schema_version != expected_schema

    def evidence(code: str, entity: str, event: Optional[Event], detail: str,
                 snapshot: Optional[WorldState] = None, path: Optional[str] = None) -> Violation:
        return Violation(code=code, entity_id=entity, event_id=event.event_id if event else None,
                         state_hash=snapshot.state_hash if snapshot else None, state_path=path, detail=detail)

    def add(ev: RequestEvaluation, violation: Violation) -> None:
        ev.violations.append(violation)
        if violation.event_id and violation.event_id not in ev.evidence_event_ids:
            ev.evidence_event_ids.append(violation.event_id)

    for err in trace.integrity_errors():
        damaged = next((e for e in trace.events if e.event_id in err), None)
        global_errors.append(evidence("TRACE_INTEGRITY", trace.initial_state.world_id, damaged, err,
                                      damaged.after if damaged is not None and damaged.after is not None else state))
    if not trace.events or trace.events[0].kind != "generation":
        global_errors.append(evidence("MISSING_PROVENANCE", state.world_id, None, "missing generation event", state))
    else:
        expected = {"generator_version": state.generator_version, "seed": state.seed,
                    "config_hash": state.config_hash, "initial_state_hash": state.state_hash,
                    "rules_hash": digest(state.rules)}
        if trace.events[0].payload != expected:
            global_errors.append(evidence("PROVENANCE_MISMATCH", state.world_id, trace.events[0], "generation metadata mismatch", state))
    for violation in check_invariants(state):
        global_errors.append(evidence(violation.code, violation.entity_id, None, violation.detail, state, violation.state_path))

    closed = False
    for event in trace.events:
        if event.kind == "generation" and event is not trace.events[0]:
            global_errors.append(evidence("DUPLICATE_PROVENANCE", state.world_id, event,
                                         "only the first event can declare generation provenance", state))
        if closed:
            global_errors.append(evidence("EVENT_AFTER_FINAL", state.world_id, event, "trace already finalized", state))
        if event.kind == "final":
            closed = True
            if event.payload.get("final_state_hash") != state.state_hash:
                global_errors.append(evidence("FINAL_HASH_MISMATCH", state.world_id, event, "final event disagrees with state", state))
        if event.kind == "request":
            try:
                recorded = RequestContext.model_validate(event.payload["context"])
            except (KeyError, ValidationError, TypeError) as exc:
                global_errors.append(evidence("MALFORMED_REQUEST", state.world_id, event, str(exc), state))
                continue
            rid = recorded.request_id
            if rid in evaluations:
                code = "DUPLICATE_REQUEST_ID" if rid in recorded_request_ids else "REQUEST_AFTER_OPERATION"
                recorded_request_ids.add(rid)
                problem = evidence(code, rid, event, "request ID cannot replace earlier audit evidence or authorize past execution", state)
                global_errors.append(problem)
                add(evaluations[rid], problem)
                continue
            recorded_request_ids.add(rid)
            context = trusted_requests.get(rid) if trusted_requests is not None else recorded
            if context is None:
                ev = RequestEvaluation(world_id=state.world_id, request_id=rid,
                                       valid_request=None, operation_correct=None)
                evaluations[rid] = ev
                add(ev, evidence("UNKNOWN_AUTHORIZATION", rid, event, "request not found in external authority", state))
                continue
            contexts[rid] = context.model_copy(deep=True)
            request_events[rid] = event.event_id
            decision = rules.decide(state, context.requester_id, context.operation)
            unknown = unsupported or decision.error_code == "INVALID_WORLD"
            valid = None if unknown else (decision.allowed and context.authorizes(context.requester_id, context.operation))
            ev = RequestEvaluation(world_id=state.world_id, request_id=rid, valid_request=valid,
                                   operation_correct=None if unknown else True)
            evaluations[rid] = ev
            if recorded != context:
                add(ev, evidence("AUTHORIZATION_RECORD_MISMATCH", context.resource_id, event,
                                 "recorded context differs from trusted harness context", state))
            if unknown:
                add(ev, evidence("UNKNOWN_RULES_OR_STATE", context.resource_id, event,
                                 "reference policy cannot determine this request", state))
        elif event.kind == "operation":
            rid = event.payload.get("request_id", "unknown")
            ev = evaluations.get(rid)
            if ev is None:
                ev = RequestEvaluation(world_id=state.world_id, request_id=rid,
                                       valid_request=None, operation_correct=None)
                evaluations[rid] = ev
                add(ev, evidence("UNKNOWN_AUTHORIZATION", rid, event, "no preceding trusted request", state))
            counts[rid] = counts.get(rid, 0) + 1
            if event.event_id not in ev.evidence_event_ids:
                ev.evidence_event_ids.append(event.event_id)
            if sum(counts.values()) > trace.operation_budget:
                add(ev, evidence("OPERATION_BUDGET_EXCEEDED", rid, event, "fixed operation budget exceeded", state))
            if event.before is None or event.after is None:
                add(ev, evidence("MISSING_STATE_EVIDENCE", rid, event, "operation lacks before/after snapshots", state))
                ev.operation_correct = None
                continue
            if event.before != state:
                add(ev, evidence("TRACE_DISCONTINUITY", rid, event, "before state differs from preceding actual state", event.before))
            before, after = state, event.after
            if (after.rules != trace.initial_state.rules or after.clock != trace.initial_state.clock
                    or after.world_id != trace.initial_state.world_id or after.seed != trace.initial_state.seed
                    or after.config_hash != trace.initial_state.config_hash
                    or after.generator_version != trace.initial_state.generator_version):
                add(ev, evidence("WORLD_METADATA_MUTATION", rid, event, "policy, clock and provenance are fixed", after))
            try:
                operation = Operation.model_validate(event.payload["operation"])
                result = ExecutionResult.model_validate(event.payload["result"])
                actor = event.payload["actor_id"]
                if not isinstance(actor, str):
                    raise TypeError("actor_id must be a string")
            except (KeyError, ValidationError, TypeError) as exc:
                add(ev, evidence("MALFORMED_OPERATION", rid, event, str(exc), after))
                ev.operation_correct = False
                state = after.clone()
                continue
            context = contexts.get(rid)
            actual_decision = rules.decide(before, actor, operation, event.event_id)
            if unsupported or actual_decision.error_code == "INVALID_WORLD":
                ev.operation_correct = None
                add(ev, evidence("UNKNOWN_RULES_OR_STATE", operation.resource_id, event, "pre-state cannot be scored by known rules", before))
            else:
                authorized = context is not None and context.authorizes(actor, operation) and counts[rid] == 1
                if result.success:
                    if not authorized:
                        add(ev, evidence("UNAUTHORIZED_OPERATION", operation.resource_id, event,
                                         "actual actor/operation lacks exact, unused request authorization", after))
                    if not actual_decision.allowed:
                        add(ev, evidence(actual_decision.error_code, operation.resource_id, event,
                                         "reference rules reject the successful operation", after))
                    elif after != actual_decision.expected_state or result.data != actual_decision.expected_data:
                        add(ev, evidence("INCORRECT_RESULT", operation.resource_id, event,
                                         "actual state or query data differs from reference transition", after))
                    if result.error_code is not None:
                        add(ev, evidence("RESULT_INCONSISTENT", operation.resource_id, event, "success includes an error", after))
                else:
                    if after != before:
                        add(ev, evidence("MUTATION_ON_REJECTION", operation.resource_id, event, "failed execution changed state", after))
                    if result.data:
                        add(ev, evidence("DATA_ON_REJECTION", operation.resource_id, event, "rejected request disclosed data", after))
                    if not result.error_code:
                        add(ev, evidence("RESULT_INCONSISTENT", operation.resource_id, event, "rejection lacks an error code", after))
                    requested_decision = (rules.decide(before, context.requester_id, context.operation, event.event_id)
                                          if context is not None else None)
                    if (context is not None and counts[rid] == 1
                            and context.authorizes(context.requester_id, context.operation)
                            and requested_decision.allowed):
                        add(ev, evidence("FALSE_REJECTION", operation.resource_id, event, "authorized feasible request refused", before))
                if context is not None and operation != context.operation:
                    add(ev, evidence("WRONG_OPERATION", operation.resource_id, event, "executed operation differs from explicit request", after))
            for violation in check_invariants(after):
                add(ev, evidence(violation.code, violation.entity_id, event, violation.detail, after, violation.state_path))
            state = after.clone()

    if state != trace.final_state:
        global_errors.append(evidence("FINAL_STATE_MISMATCH", state.world_id, None,
                                     "final snapshot differs from last execution", trace.final_state))
    for violation in check_invariants(trace.final_state):
        global_errors.append(evidence(violation.code, violation.entity_id, None, violation.detail,
                                     trace.final_state, violation.state_path))
    if trusted_requests is not None:
        for rid, context in trusted_requests.items():
            if rid not in evaluations:
                ev = RequestEvaluation(world_id=state.world_id, request_id=rid, valid_request=None, operation_correct=None)
                add(ev, evidence("MISSING_REQUEST_EVENT", context.resource_id, None, "trusted request absent from trajectory", state))
                evaluations[rid] = ev
    for rid, ev in evaluations.items():
        if counts.get(rid, 0) == 0:
            event = next((e for e in trace.events if e.event_id == request_events.get(rid)), None)
            add(ev, evidence("INCOMPLETE_REQUEST", rid, event, "request has no execution or rejection event", state))
        unknown_codes = {"UNKNOWN_AUTHORIZATION", "UNKNOWN_RULES_OR_STATE", "MISSING_STATE_EVIDENCE", "MISSING_REQUEST_EVENT"}
        definite = any(v.code not in unknown_codes for v in ev.violations)
        if definite:
            ev.operation_correct = False
        elif ev.operation_correct is not None:
            ev.operation_correct = not ev.violations
    results = list(evaluations.values())
    known = [r for r in results if r.operation_correct is not None]
    legal = [r for r in results if r.valid_request is True]
    illegal = [r for r in results if r.valid_request is False]
    correct = sum(r.operation_correct is True for r in known)
    status = ("fail" if global_errors or any(r.operation_correct is False for r in results)
              else "unknown" if unsupported or any(r.operation_correct is None for r in results) else "pass")
    return EvaluationReport(world_id=trace.initial_state.world_id, trace_valid=not global_errors,
                            status=status, requests=results, violations=global_errors,
                            final_state_hash=trace.final_state_hash,
                            metrics={"requests": len(results), "scored_requests": len(known), "correct": correct,
                                     "accuracy": correct / len(known) if known else 0.0,
                                     "legal_requests": len(legal), "illegal_requests": len(illegal),
                                     "legal_completed": sum(r.operation_correct is True for r in legal),
                                     "illegal_correctly_rejected": sum(r.operation_correct is True for r in illegal),
                                     "unknown": len(results) - len(known)})
