"""Offline, non-gating diagnosis for After-sales protocol-v3 decisions.

This module summarizes decisions already recorded by the native Oracle and
trace. It never changes task scores, Engine behavior, Reference rules, or the
selected Skill snapshot. A mechanism label is an observed category, not a
counterfactually verified causal root.
"""
from __future__ import annotations

from collections import Counter

from usami_harness.models import digest
from .agents_v3 import AgentExecutionV3, replay_v3


PROTOCOL = "aftersales-decision-failure-audit-v1"


def _mechanism_ids(operation: dict, row: dict, post_invariants: list) -> list[str]:
    result = []
    if row.get("authorized") is False:
        result.append("aftersales/protocol/current_turn_authorization")
    code = row.get("reference_error_code")
    if row.get("reference_allowed") is False and isinstance(code, str) and code:
        result.append("aftersales/business/" + code.lower())
    if row.get("classification") == "false_rejection":
        result.append("aftersales/business/false_rejection")
    if post_invariants:
        result.append("aftersales/state/post_operation_invariant_failure")
    return sorted(set(result))


def _goal_state(adapter, execution):
    task = execution.config.task
    initial = adapter.World.model_validate(execution.trace.initial_world)
    final = adapter.World.model_validate(execution.trace.final_world)
    positive = [goal.satisfied(final) for goal in task.goals]
    positive.extend(goal.satisfied(initial, final) for goal in task.created_goals)
    return {
        "final_goal_reached": bool(positive) and all(positive),
        "completion_subgoals": positive,
        "strict_task_success": bool(execution.task_score.get("success")),
        "refusal_subgoals": list(execution.task_score.get("refusal_subgoals", [])),
    }


def _target_operation_names(adapter, execution):
    task = execution.config.task
    names = {adapter.Operation.model_validate(refusal.operation).name for refusal in task.refusals}
    if task.goals or task.created_goals:
        names.update({"open_case", "verify_evidence", "quote_resolution", "commit_resolution"})
        world = adapter.World.model_validate(execution.trace.initial_world)
        for requirement in task.composition.preference_requirements if task.composition else []:
            if requirement.preference != "resolution" or "refund" not in requirement.accepted_values:
                continue
            for profile_id in [requirement.profile_id]:
                private = adapter.private_state(world, profile_id)
                line = world.lines.get(private.facts.get("line_id"))
                product = world.products.get(line.product_id) if line else None
                if product and product.return_required:
                    names.update({"schedule_return", "release_refund"})
    return sorted(names)


def analyze_execution(adapter, execution, *, replay_result=None) -> dict:
    """Create a replay-bound view of attempted decisions and observed blockers."""
    row = AgentExecutionV3.model_validate(
        execution.model_dump(mode="json") if hasattr(execution, "model_dump") else execution)
    if row.execution_hash != row.computed_hash():
        raise ValueError("execution integrity differs; decision audit refuses unsealed input")
    if replay_result is None:
        try:
            replay_result = replay_v3(adapter, row)
        except Exception as exc:  # Audit the failure; never present it as verified.
            replay_result = {"verified": False, "reason": type(exc).__name__ + ": " + str(exc)}

    trace_events = {event.id: event for event in row.trace.events}
    event_positions = {event.id: index for index, event in enumerate(row.trace.events)}
    decisions = []
    for raw in row.oracle.get("operations", []):
        event_id = raw.get("dialogue_event_id")
        event = trace_events.get(event_id)
        operation = raw.get("operation")
        native = event.payload.get("trace", {}) if event is not None else {}
        if isinstance(operation, str):
            native_operations = [item for item in native.get("events", []) if item.get("kind") == "operation"]
            if len(native_operations) == 1:
                operation = native_operations[0].get("payload", {}).get("operation")
        if not isinstance(operation, dict):
            continue
        initial_raw = native.get("initial_state", {})
        final_raw = native.get("final_state", {})
        post_invariants = []
        if final_raw:
            try:
                post_invariants = adapter.invariants(adapter.World.model_validate(final_raw))
            except Exception:
                post_invariants = ["POST_STATE_UNREADABLE"]
        mechanisms = _mechanism_ids(operation, raw, post_invariants)
        failed_checks = []
        if raw.get("authorized") is False:
            failed_checks.append({"kind": "current_turn_authorization", "status": "missing_or_mismatched"})
        if raw.get("reference_allowed") is False:
            failed_checks.append({"kind": "native_reference_precondition", "status": "denied",
                                 "code": raw.get("reference_error_code")})
        if post_invariants:
            failed_checks.append({"kind": "post_operation_invariant", "status": "violated",
                                 "codes": post_invariants})
        if len(failed_checks) > 1:
            causal_status = "multiple_observed_constraints_causality_unresolved"
        elif failed_checks:
            causal_status = "single_observed_constraint_not_counterfactually_verified"
        else:
            causal_status = "no_observed_failure_constraint"
        decisions.append({
            "dialogue_event_id": event_id,
            "trace_position": event_positions.get(event_id),
            "operation": operation,
            "attempted_write": not adapter.Operation.model_validate(operation).is_query,
            "classification": raw.get("classification"),
            "execution_outcome": (
                "successful_business_violation" if raw.get("classification") == "successful_business_violation"
                else "blocked_unauthorized_attempt" if not raw.get("success") and raw.get("authorized") is False
                else "blocked_by_reference" if not raw.get("success") and raw.get("reference_allowed") is False
                else "false_rejection" if raw.get("classification") == "false_rejection"
                else "executed_correctly" if raw.get("operation_correct") is True
                else "outcome_uncertain"),
            "operation_correct": raw.get("operation_correct"),
            "reference_allowed": raw.get("reference_allowed"),
            "reference_error_code": raw.get("reference_error_code"),
            "current_turn_authorized": raw.get("authorized"),
            "engine_execution_succeeded": bool(raw.get("success")),
            "state_changed": bool(event and event.before_hash != event.after_hash),
            "prestate_hash": event.before_hash if event else None,
            "poststate_hash": event.after_hash if event else None,
            "failed_checks": failed_checks,
            "post_operation_invariants": post_invariants,
            "observed_mechanism_ids": mechanisms,
            "causal_root_verified": False,
            "causal_status": causal_status,
        })

    invalid_tools = []
    for index, tool in enumerate(row.tools):
        result = tool.get("result", {}) if isinstance(tool, dict) else {}
        if not isinstance(result, dict) or result.get("success") is not False:
            continue
        if result.get("error_code") == "TOOL_ARGUMENT_INVALID":
            invalid_tools.append({"tool_index": index, "name": tool.get("name"),
                                 "arguments": tool.get("arguments"),
                                 "error": result.get("message"),
                                 "mechanism_id": "aftersales/protocol/invalid_tool_arguments"})

    service_protocol_errors = [error for error in row.errors
        if error.get("role") == "service" and error.get("kind") == "protocol"]
    service_transport_errors = [error for error in row.errors
        if error.get("role") == "service" and error.get("kind") == "transport"]
    writes = [decision for decision in decisions if decision["attempted_write"]]
    invalid_decisions = [decision for decision in writes
        if decision["operation_correct"] is False or decision["reference_allowed"] is False
        or decision["current_turn_authorized"] is False]
    first_invalid = min(invalid_decisions, key=lambda item: item["trace_position"]
                        if item["trace_position"] is not None else 10**9) if invalid_decisions else None
    mechanism_counts = Counter(mechanism for decision in decisions
                               for mechanism in decision["observed_mechanism_ids"])
    operation_counts = Counter(decision["operation"].get("name") for decision in decisions)
    target_names = _target_operation_names(adapter, row)
    targets = [{"operation": name, "reached": operation_counts[name] > 0,
                "attempt_count": operation_counts[name]} for name in target_names]
    refusal_rows = []
    for raw in row.oracle.get("service_decisions", []):
        operation = raw.get("operation")
        if isinstance(operation, dict):
            refusal_rows.append({"dialogue_event_id": raw.get("event_id"),
                "operation": operation, "classification": raw.get("classification"),
                "reference_allowed": raw.get("reference_allowed"),
                "reference_error_code": raw.get("reference_code"),
                "causal_root_verified": False})
    for target in targets:
        refusals = [item for item in refusal_rows if item["operation"].get("name") == target["operation"]]
        target["declared_refusal_count"] = len(refusals)
        target["reached"] = target["reached"] or bool(refusals)
    goal_state = _goal_state(adapter, row)
    usage = row.cost.get("by_role", {})
    report = {
        "protocol": PROTOCOL,
        "execution_hash": row.execution_hash,
        "trace_hash": row.trace.trace_hash,
        "world_hash": row.config.initial_world_hash,
        "task_hash": row.config.task.task_hash,
        "mode": row.config.mode,
        "replay": replay_result,
        "goal_state": goal_state,
        "operations": decisions,
        "service_declared_refusals": refusal_rows,
        "target_decision_opportunities": targets,
        "invalid_tool_arguments": invalid_tools,
        "service_protocol_errors": service_protocol_errors,
        "service_transport_errors": service_transport_errors,
        "first_invalid_tool_proposal": first_invalid,
        "summary": {
            "service_operation_attempts": len(writes),
            "service_query_attempts": len(decisions) - len(writes),
            "invalid_write_proposals": len(invalid_decisions),
            "reference_rejected_attempts": sum(d["attempted_write"] and d["reference_allowed"] is False for d in decisions),
            "unauthorized_write_attempts": sum(d["attempted_write"] and d["current_turn_authorized"] is False for d in decisions),
            "reference_blocked_unauthorized_attempts": sum(d["execution_outcome"] == "blocked_unauthorized_attempt" for d in decisions),
            "successful_business_violations": sum(d["classification"] == "successful_business_violation" for d in decisions),
            "false_rejections": sum(d["classification"] == "false_rejection" for d in decisions),
            "invalid_tool_argument_count": len(invalid_tools),
            "service_protocol_error_count": len(service_protocol_errors),
            "observed_mechanism_counts": dict(sorted(mechanism_counts.items())),
            "service_cost": usage.get("service", {}),
            "customer_cost": usage.get("customer", {}),
            "total_cost": dict(row.cost),
        },
    }
    if report["first_invalid_tool_proposal"] is None and invalid_tools:
        report["first_invalid_tool_proposal"] = {"kind": "invalid_tool_arguments",
            "tool_index": invalid_tools[0]["tool_index"], "name": invalid_tools[0]["name"],
            "mechanism_id": invalid_tools[0]["mechanism_id"]}
    if report["first_invalid_tool_proposal"] is None and service_protocol_errors:
        report["first_invalid_tool_proposal"] = {"kind": "service_protocol_error",
            "error": service_protocol_errors[0],
            "mechanism_id": "aftersales/protocol/invalid_model_output"}
    report["summary"]["invalid_proposal_count_including_tool_arguments"] = len(invalid_decisions) + len(invalid_tools)
    report["audit_hash"] = digest(report)
    return report


def compare_matched_audits(baseline: dict, candidate: dict) -> dict:
    """Compare two already-sealed runs only when task and world match exactly.

    This is a descriptive paired diagnostic, not a Skill gate or statistical
    claim. A missing decision opportunity is reported as not reached rather
    than silently counted as a successful repair.
    """
    for audit in (baseline, candidate):
        supplied_hash = audit.get("audit_hash")
        payload = {key: value for key, value in audit.items() if key != "audit_hash"}
        if not supplied_hash or supplied_hash != digest(payload):
            raise ValueError("matched decision comparison requires intact audit hashes")
        if audit.get("replay", {}).get("verified") is not True:
            raise ValueError("matched decision comparison requires verified offline Replay")
    for field in ("task_hash", "world_hash"):
        if baseline.get(field) != candidate.get(field):
            raise ValueError("matched decision comparison requires identical " + field)
    target_names = sorted({row.get("operation") for row in baseline.get("target_decision_opportunities", [])} |
                          {row.get("operation") for row in candidate.get("target_decision_opportunities", [])})
    left_rows, right_rows = baseline.get("operations", []), candidate.get("operations", [])
    left_refusals = baseline.get("service_declared_refusals", [])
    right_refusals = candidate.get("service_declared_refusals", [])
    targets = []
    for name in target_names:
        left = [row for row in left_rows if row.get("operation", {}).get("name") == name]
        right = [row for row in right_rows if row.get("operation", {}).get("name") == name]
        left_declared = [row for row in left_refusals if row.get("operation", {}).get("name") == name]
        right_declared = [row for row in right_refusals if row.get("operation", {}).get("name") == name]
        targets.append({
            "operation": name,
            "baseline_opportunity_reached": bool(left or left_declared),
            "candidate_opportunity_reached": bool(right or right_declared),
            "baseline_attempts": len(left),
            "candidate_attempts": len(right),
            "baseline_declared_refusals": len(left_declared),
            "candidate_declared_refusals": len(right_declared),
            "baseline_invalid_proposals": sum(row.get("operation_correct") is False or
                row.get("reference_allowed") is False or row.get("current_turn_authorized") is False for row in left),
            "candidate_invalid_proposals": sum(row.get("operation_correct") is False or
                row.get("reference_allowed") is False or row.get("current_turn_authorized") is False for row in right),
            "baseline_successful_business_violations": sum(
                row.get("classification") == "successful_business_violation" for row in left),
            "candidate_successful_business_violations": sum(
                row.get("classification") == "successful_business_violation" for row in right),
            "comparison_status": "matched_opportunity" if (left or left_declared) and (right or right_declared) else "opportunity_not_reached_in_one_or_both_runs",
        })
    result = {
        "protocol": "aftersales-matched-decision-audit-v1",
        "task_hash": baseline.get("task_hash"),
        "world_hash": baseline.get("world_hash"),
        "baseline_execution_hash": baseline.get("execution_hash"),
        "candidate_execution_hash": candidate.get("execution_hash"),
        "baseline_strict_task_success": baseline.get("goal_state", {}).get("strict_task_success"),
        "candidate_strict_task_success": candidate.get("goal_state", {}).get("strict_task_success"),
        "baseline_final_goal_reached": baseline.get("goal_state", {}).get("final_goal_reached"),
        "candidate_final_goal_reached": candidate.get("goal_state", {}).get("final_goal_reached"),
        "target_decisions": targets,
        "claim_boundary": "Descriptive matched diagnostic only; no inference of causal repair or generalization.",
    }
    result["comparison_hash"] = digest(result)
    return result
