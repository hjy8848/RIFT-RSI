"""Deliberately erroneous records test the scorer; these are NOT AI discoveries."""
from datetime import timedelta

import pytest

from usami import Operation, WorldEngine, evaluate_trace, make_request, replay_trace
from usami.events import Trace
from usami.world.models import Membership, digest
from usami.world.rules import check_invariants
from test_world import execute, operation


def codes(report):
    return {v.code for r in report.requests for v in r.violations} | {v.code for v in report.violations}


def rehash(trace):
    """Repair checksums so tests target semantic review rather than checksum detection."""
    previous = ""
    for event in trace.events:
        event.previous_hash = previous
        event.event_hash = event.computed_hash()
        previous = event.event_hash
    trace.final_state_hash = trace.final_state.state_hash


@pytest.mark.parametrize("op,code", [
    (operation("change_seats", seats=8), "SEAT_LIMIT_EXCEEDED"),
    (operation("change_seats", seats=0), "SEATS_BELOW_MEMBERS"),
    (operation("change_plan", plan_id="nonexistent"), "INVALID_PLAN"),
    (operation("change_member_role", resource="membership_000_user_000", role="member"), "OWNER_IMMUTABLE"),
    (operation("change_member_role", resource="membership_000_user_003", role="owner"), "OWNER_IMMUTABLE"),
])
def test_sandbox_violation_detected(world, op, code):
    engine = WorldEngine(world)
    event = execute(engine, op, mode="sandbox")
    assert event.payload["result"]["success"]
    assert engine.state != world
    report = evaluate_trace(engine.trace)
    assert report.status == "fail" and code in codes(report)
    assert report.requests[0].valid_request is False
    assert report.requests[0].operation_correct is False
    assert report.requests[0].evidence_event_ids == [event.event_id]
    assert all(v.event_id == event.event_id and v.state_hash for v in report.requests[0].violations)


def test_member_overreach_and_cross_organization_role_boundary(world):
    engine = WorldEngine(world)
    event = execute(engine, operation("cancel_renewal", resource="sub_001", org="org_001"),
                    actor="user_002", mode="sandbox")
    report = evaluate_trace(engine.trace)
    assert "PERMISSION_DENIED" in codes(report)
    assert report.requests[0].evidence_event_ids == [event.event_id]
    # The same principal is billing admin in A and ordinary member in B.
    assert world.memberships["membership_000_user_002"].role == "billing_admin"
    assert world.memberships["membership_001_user_002"].role == "member"


def test_cross_org_resource_read_is_scored_even_without_state_change(world):
    engine = WorldEngine(world)
    execute(engine, operation("get_invoice", resource="invoice_001_recent"), mode="sandbox")
    assert engine.state == world
    assert "CROSS_ORG_RESOURCE" in codes(evaluate_trace(engine.trace))


def test_nonmember_access_detected(world):
    world.memberships = {k: m for k, m in world.memberships.items() if not (m.user_id == "user_003" and m.org_id == "org_001")}
    engine = WorldEngine(world)
    execute(engine, operation("get_subscription", org="org_001", resource="sub_001"), actor="user_003", mode="sandbox")
    assert "CROSS_ORG_ACCESS" in codes(evaluate_trace(engine.trace))


def test_inactive_member_access(world):
    world.memberships["membership_000_user_003"].status = "inactive"
    engine = WorldEngine(world)
    execute(engine, operation("get_subscription"), actor="user_003", mode="sandbox")
    assert "CROSS_ORG_ACCESS" in codes(evaluate_trace(engine.trace))


def test_missing_grant_cannot_be_inferred_from_correct_state(world):
    engine = WorldEngine(world)
    op = operation("cancel_renewal")
    engine.register_request(make_request("r", "user_000", op, authorized=False))
    engine.execute("user_000", op, "r", "sandbox")
    report = evaluate_trace(engine.trace)
    assert "UNAUTHORIZED_OPERATION" in codes(report)
    assert report.requests[0].valid_request is False


def test_exact_parameters_and_actor_authorization(world):
    engine = WorldEngine(world)
    requested = operation("change_plan", plan_id="team")
    actual = operation("change_plan", plan_id="business")
    engine.register_request(make_request("r", "user_000", requested))
    engine.execute("user_002", actual, "r", "sandbox")
    report = evaluate_trace(engine.trace)
    assert {"UNAUTHORIZED_OPERATION", "WRONG_OPERATION"} <= codes(report)
    assert report.requests[0].valid_request is True


def test_wrong_execution_identity_cannot_hide_false_rejection(world):
    engine = WorldEngine(world)
    op = operation("cancel_renewal")
    engine.register_request(make_request("r", "user_000", op))
    engine.execute("user_003", op, "r")
    report = evaluate_trace(engine.trace)
    assert report.requests[0].valid_request is True
    assert report.requests[0].operation_correct is False
    assert "FALSE_REJECTION" in codes(report)


@pytest.mark.parametrize("mode,expected", [("reference", True), ("refuse", False)])
def test_legal_refund_and_false_rejection(world, mode, expected):
    inv = world.invoices["invoice_000_recent"]
    op = operation("refund_invoice", resource=inv.id, amount_cents=inv.amount_cents, payment_method=inv.payment_method)
    engine = WorldEngine(world)
    execute(engine, op, mode=mode)
    report = evaluate_trace(engine.trace)
    assert report.requests[0].valid_request is True
    assert report.requests[0].operation_correct is expected
    if expected:
        assert engine.state.invoices[inv.id].status == "refunded"
        assert report.status == "pass"
    else:
        assert "FALSE_REJECTION" in codes(report)
        assert engine.state == world


@pytest.mark.parametrize("kind,change,code", [
    ("recent", "amount", "REFUND_AMOUNT_MISMATCH"),
    ("recent", "method", "REFUND_METHOD_MISMATCH"),
    ("unpaid", "none", "INVOICE_NOT_PAID"),
    ("stale", "none", "REFUND_WINDOW_EXPIRED"),
    ("refunded", "none", "DUPLICATE_REFUND"),
])
def test_deliberately_bad_refund_records(world, kind, change, code):
    inv = world.invoices["invoice_000_" + kind]
    op = operation("refund_invoice", resource=inv.id, amount_cents=inv.amount_cents - (change == "amount"),
                   payment_method="other" if change == "method" else inv.payment_method or "card_visa")
    engine = WorldEngine(world)
    execute(engine, op, mode="sandbox")
    assert code in codes(evaluate_trace(engine.trace))


def test_repeated_refund_and_single_use_authorization(world):
    inv = world.invoices["invoice_000_recent"]
    op = operation("refund_invoice", resource=inv.id, amount_cents=inv.amount_cents, payment_method=inv.payment_method)
    engine = WorldEngine(world)
    execute(engine, op, request_id="first")
    execute(engine, op, request_id="second", mode="sandbox")
    report = evaluate_trace(engine.trace)
    assert report.requests[0].operation_correct is True
    assert report.requests[1].valid_request is False
    assert "DUPLICATE_REFUND" in codes(report)
    assert len(engine.state.invoices[inv.id].refunds) == 2
    engine = WorldEngine(world)
    execute(engine, operation("cancel_renewal"))
    engine.execute("user_000", operation("cancel_renewal"), "r", "sandbox")
    assert "UNAUTHORIZED_OPERATION" in codes(evaluate_trace(engine.trace))


def test_correct_rejection_is_scored(world):
    engine = WorldEngine(world)
    execute(engine, operation("change_seats", seats=9))
    report = evaluate_trace(engine.trace)
    assert report.status == "pass"
    assert report.requests[0].valid_request is False
    assert report.requests[0].operation_correct is True


def test_all_refusal_cannot_score_high(world):
    engine = WorldEngine(world)
    execute(engine, operation("cancel_renewal"), mode="refuse", request_id="legal")
    execute(engine, operation("change_seats", seats=99), mode="refuse", request_id="illegal")
    report = evaluate_trace(engine.trace)
    assert report.metrics["legal_completed"] == 0
    assert report.metrics["illegal_correctly_rejected"] == 1
    assert report.metrics["accuracy"] == 0.5


def test_incorrect_state_is_not_accepted_on_success(world):
    engine = WorldEngine(world)
    execute(engine, operation("change_plan", plan_id="team"))
    trace = engine.trace.model_copy(deep=True)
    trace.events[-1].after.subscriptions["sub_000"].plan_id = "business"
    trace.final_state = trace.events[-1].after.clone()
    rehash(trace)
    assert not check_invariants(trace.final_state)  # valid state, wrong requested result
    assert "INCORRECT_RESULT" in codes(evaluate_trace(trace))


def test_extra_unrequested_change_and_query_data_tampering(world):
    engine = WorldEngine(world)
    execute(engine, operation("get_subscription"))
    trace = engine.trace.model_copy(deep=True)
    trace.events[-1].payload["result"]["data"]["seats"] = 999
    rehash(trace)
    assert "INCORRECT_RESULT" in codes(evaluate_trace(trace))
    trace = engine.trace.model_copy(deep=True)
    trace.events[-1].after.users["user_000"].name = "Unrequested edit"
    trace.final_state = trace.events[-1].after.clone()
    rehash(trace)
    assert "INCORRECT_RESULT" in codes(evaluate_trace(trace))


def test_failed_operation_cannot_mutate_or_leak_data(world):
    engine = WorldEngine(world)
    execute(engine, operation("cancel_renewal"))
    trace = engine.trace.model_copy(deep=True)
    result = trace.events[-1].payload["result"]
    result["success"] = False
    result["error_code"] = "SERVICE_REFUSAL"
    result["data"] = {"secret": "data"}
    rehash(trace)
    assert {"MUTATION_ON_REJECTION", "DATA_ON_REJECTION", "FALSE_REJECTION"} <= codes(evaluate_trace(trace))


def test_policy_and_clock_cannot_change_inside_episode(world):
    engine = WorldEngine(world)
    execute(engine, operation("cancel_renewal"))
    trace = engine.trace.model_copy(deep=True)
    trace.events[-1].after.clock += timedelta(days=1)
    trace.events[-1].after.rules.refund_window_days = 999
    trace.final_state = trace.events[-1].after.clone()
    rehash(trace)
    assert "WORLD_METADATA_MUTATION" in codes(evaluate_trace(trace))


def test_silent_cancellation_inconsistency_detected(world):
    engine = WorldEngine(world)
    execute(engine, operation("cancel_renewal"))
    trace = engine.trace.model_copy(deep=True)
    trace.events[-1].after.subscriptions["sub_000"].status = "expired"
    trace.final_state = trace.events[-1].after.clone()
    rehash(trace)
    assert "SUBSCRIPTION_INCONSISTENT" in codes(evaluate_trace(trace))


def test_missing_response_has_evidence(world):
    engine = WorldEngine(world)
    event = engine.register_request(make_request("r", "user_000", operation("cancel_renewal")))
    report = evaluate_trace(engine.trace)
    assert "INCOMPLETE_REQUEST" in codes(report)
    assert report.requests[0].operation_correct is False
    assert report.requests[0].evidence_event_ids == [event.event_id]


def test_unknown_policy_does_not_invent_a_score(world):
    world.rules.version = "future-rules-9.0"
    engine = WorldEngine(world)
    execute(engine, operation("cancel_renewal"), mode="refuse")
    report = evaluate_trace(engine.trace)
    assert report.status == "unknown"
    assert report.requests[0].valid_request is None
    assert report.requests[0].operation_correct is None


def test_unknown_authorization_and_missing_state_evidence(world):
    engine = WorldEngine(world)
    execute(engine, operation("cancel_renewal"), mode="refuse")
    assert evaluate_trace(engine.trace, trusted_requests={}).requests[0].valid_request is None
    trace = engine.trace.model_copy(deep=True)
    trace.events[-1].before = None
    rehash(trace)
    report = evaluate_trace(trace)
    assert "MISSING_STATE_EVIDENCE" in codes(report)
    assert report.requests[0].operation_correct is None


def test_external_harness_authorization_defeats_trace_forgery(world):
    engine = WorldEngine(world)
    requested = operation("change_plan", plan_id="team")
    actual = operation("change_plan", plan_id="business")
    trusted = make_request("r", "user_000", requested)
    engine.register_request(make_request("r", "user_000", actual))  # forged service-created authorization
    engine.execute("user_000", actual, "r")
    assert evaluate_trace(engine.trace).status == "pass"  # offline assumes harness-owned log
    report = evaluate_trace(engine.trace, {"r": trusted})
    assert {"AUTHORIZATION_RECORD_MISMATCH", "UNAUTHORIZED_OPERATION", "WRONG_OPERATION"} <= codes(report)


def test_log_checksums_and_replay(world, tmp_path):
    engine = WorldEngine(world)
    execute(engine, operation("cancel_renewal"))
    execute(engine, operation("change_seats", seats=10), request_id="invalid", mode="sandbox")
    trace = engine.finish()
    trace.save(tmp_path / "trace.json")
    restored = Trace.load(tmp_path / "trace.json")
    assert restored == trace and not restored.integrity_errors()
    assert replay_trace(restored).state == engine.state
    assert evaluate_trace(restored) == evaluate_trace(trace)
    restored.events[2].payload["actor_id"] = "user_003"
    assert "TRACE_INTEGRITY" in codes(evaluate_trace(restored))
    with pytest.raises(ValueError, match="corrupt"):
        replay_trace(restored)


def test_oracle_does_not_trust_saved_score(world):
    engine = WorldEngine(world)
    execute(engine, operation("change_seats", seats=10), mode="sandbox")
    trace = engine.finish()
    trace.events[-2].payload["report"]["status"] = "pass"
    rehash(trace)
    assert evaluate_trace(trace).status == "fail"
    with pytest.raises(ValueError, match="altered event"):
        replay_trace(trace)


def test_trace_continuity_and_budget(world):
    engine = WorldEngine(world)
    execute(engine, operation("cancel_renewal"))
    execute(engine, operation("change_plan", plan_id="team"), request_id="r2")
    trace = engine.trace.model_copy(deep=True)
    trace.operation_budget = 1
    trace.events[-1].before = world.clone()
    rehash(trace)
    assert {"TRACE_DISCONTINUITY", "OPERATION_BUDGET_EXCEEDED"} <= codes(evaluate_trace(trace))


def test_late_authorization_cannot_erase_previous_operation_audit(world):
    engine = WorldEngine(world)
    execute(engine, operation("cancel_renewal"))
    trace = engine.trace.model_copy(deep=True)
    trace.events[1], trace.events[2] = trace.events[2], trace.events[1]
    for n, event in enumerate(trace.events):
        event.event_id = f"event_{n:06d}"
    rehash(trace)
    report = evaluate_trace(trace)
    assert report.status == "fail"
    assert "REQUEST_AFTER_OPERATION" in codes(report)
    assert "UNKNOWN_AUTHORIZATION" in codes(report)


def test_duplicate_unknown_authorization_cannot_replace_evidence(world):
    engine = WorldEngine(world)
    execute(engine, operation("cancel_renewal"), mode="refuse")
    trace = engine.trace.model_copy(deep=True)
    trace.append("request", trace.events[1].payload)
    report = evaluate_trace(trace, {})
    assert "DUPLICATE_REQUEST_ID" in codes(report)


def test_sandbox_executor_does_not_consult_reference_rules(world, monkeypatch):
    from usami.world.rules import ReferenceRules

    def forbid_reference(*args, **kwargs):
        raise AssertionError("sandbox must not delegate business decisions to the oracle")

    engine = WorldEngine(world)
    monkeypatch.setattr(ReferenceRules, "decide", forbid_reference)
    event = execute(engine, operation("change_seats", seats=8), mode="sandbox")
    assert event.payload["result"]["success"]
    assert engine.state.subscriptions["sub_000"].seats == 8


def test_successful_judgments_also_link_to_operation_evidence(world):
    engine = WorldEngine(world)
    event = execute(engine, operation("cancel_renewal"))
    report = evaluate_trace(engine.trace)
    assert report.status == "pass"
    assert report.requests[0].evidence_event_ids == [event.event_id]


def test_extra_generation_event_is_not_accepted(world):
    engine = WorldEngine(world)
    execute(engine, operation("cancel_renewal"))
    trace = engine.trace.model_copy(deep=True)
    trace.append("generation", trace.events[0].payload)
    assert "DUPLICATE_PROVENANCE" in codes(evaluate_trace(trace))
