"""Executable P0 contracts. All evidence here is an offline test fixture."""
import pytest
from pydantic import ValidationError

from vast_coevo.schema import digest
from usami_v2.contracts import (
    CommunicationObligation, FactEvidence, PublicRequest, ServiceMessage, SourceRef,
)
from usami_v2.evaluation.communication import evaluate_communication, render_fact
from usami_v2.evaluation.outcomes import GoalResult, evaluate_outcome
from usami_v2.runtime.observations import ContextItem, WorkingFact, WorkingState, project
from usami_v2.scenarios.distribution import (
    BusinessSource, BusinessStratum, DistributionSpec, GenerationAttempt,
    audit_sampling, sample_schedule,
)


def reseal(row, **updates):
    value = row.model_dump(mode="json", exclude={"record_hash"})
    value.update(updates)
    return type(row).model_validate(value)


def source(kind="public_tool", index=1):
    return SourceRef(source_id="event_" + str(index), source_kind=kind,
                     source_hash=digest([kind, index]), event_index=index)


def fact(kind="refund_amount", value=2000, subject="line_B", **updates):
    fields = dict(fact_id="fact_" + kind, subject_id=subject, kind=kind, value=value,
                  unit="CNY_cents" if kind in ("refund_amount", "extra_charge") else "",
                  valid_from=1, source=source())
    fields.update(updates)
    return FactEvidence(**fields)


def obligation(kind="refund_amount", subject="line_B", due=10):
    request = PublicRequest(request_id="request_" + kind, subject_id=subject, fact_kind=kind,
                            source=source("customer_message", 0),
                            public_text="Please tell me the " + kind + " for " + subject)
    return CommunicationObligation(obligation_id="obligation_" + kind, request=request, due_event_index=due)


def message(text, index=3, subjects=("line_B",), refs=(), id="reply"):
    return ServiceMessage(message_id=id, event_index=index, text=text, subject_ids=subjects, fact_refs=refs)


@pytest.mark.parametrize("text", [
    "line_B: refund amount is CNY 20.00.",
    "退款金额为20元。",
    "退款金额是人民币20.00元。",
])
def test_actual_money_and_obligation_are_verified(text):
    result = evaluate_communication([fact()], [message(text)], [obligation()], end_event_index=10)
    assert result.success is True
    assert result.fulfilled == 1
    assert result.claims[0].status == "CORRECT"
    assert result.checked() == result


def test_correct_reference_cannot_hide_wrong_prose_or_wrong_completed_state():
    evidence = fact()
    result = evaluate_communication(
        [evidence], [message("line_B: refund amount is CNY 200.00.", refs=(evidence.fact_id,))],
        [obligation()], end_event_index=10)
    assert result.success is False
    assert result.incorrect_facts == 1
    pending = fact("refund_status", "requested")
    result = evaluate_communication([pending], [message("The refund has been released.")],
                                    [obligation("refund_status")], end_event_index=10)
    assert result.false_completions == 1
    assert result.success is False


def test_facts_must_be_bound_to_subject_and_actual_time():
    facts = [fact(), fact(value=9000, subject="line_A", fact_id="fact_A")]
    result = evaluate_communication(
        facts, [message("line_A: refund amount is CNY 20.00.", subjects=("line_A", "line_B"))],
        [obligation(subject="line_A")], end_event_index=10)
    assert result.incorrect_facts == 1
    future = fact(valid_from=5, source=source(index=5))
    result = evaluate_communication([future], [message("refund amount is CNY 20.00.")],
                                    [obligation()], end_event_index=10)
    assert result.success is None
    with pytest.raises(ValueError, match="not yet observed"):
        evaluate_communication([future], [message("hello", refs=(future.fact_id,))], [],
                               end_event_index=10)


def test_missing_unknown_historical_and_future_due_are_distinct():
    missing = evaluate_communication([fact()], [message("Thank you.")], [obligation()], end_event_index=10)
    assert missing.missing == 1 and missing.success is False
    ambiguous = evaluate_communication([fact()], [message("Your money is on its way.")],
                                       [obligation()], end_event_index=10)
    assert ambiguous.success is None and ambiguous.unresolved_message_ids == ("reply",)
    historical = evaluate_communication([fact()], [message("As of yesterday, refund amount is CNY 50.00.")],
                                        [obligation()], end_event_index=10)
    assert historical.incorrect_facts == 0 and historical.success is None
    not_due = evaluate_communication([fact()], [], [obligation(due=20)], end_event_index=10, terminal=False)
    assert not_due.required_due == 0 and not_due.success is True
    early_finish = evaluate_communication([fact()], [], [obligation(due=20)], end_event_index=10)
    assert early_finish.required_due == 1 and early_finish.success is False
    question = evaluate_communication([fact()], [message("Could you confirm refund amount is CNY 200.00?")],
                                      [obligation()], end_event_index=10)
    assert question.incorrect_facts == 0 and question.success is None


def test_optional_render_roundtrip_and_recovered_error_not_erased():
    rows = [fact(), fact("arrival_date", "2026-01-18", subject="line_A")]
    text = "\n".join(render_fact(row) for row in rows)
    result = evaluate_communication(rows, [message(text, subjects=("line_A", "line_B"))],
                                    [obligation(), obligation("arrival_date", "line_A")], end_event_index=10)
    assert result.success is True and result.fulfilled == 2
    result = evaluate_communication(
        [fact()], [message("refund amount is CNY 200.00.", index=2, id="wrong"),
                   message("refund amount is CNY 20.00.", index=4, id="correction")],
        [obligation()], end_event_index=10)
    assert result.success is False and result.incorrect_facts == 1


def test_information_only_and_completed_transaction_with_process_error():
    communication = evaluate_communication([fact()], [message(render_fact(fact()))],
                                            [obligation()], end_event_index=10)
    pure_query = evaluate_outcome([], [], communication, replay_verified=True, valid_end=True)
    assert pure_query.final_goal_reached is None
    assert pure_query.state_goal_applicability == "NOT_APPLICABLE"
    assert pure_query.strict_task_success is True
    completed = GoalResult(goal_id="refund", status="PASS", evidence_refs=("ledger",))
    violation = GoalResult(goal_id="consent", status="FAIL", evidence_refs=("attempt",))
    score = evaluate_outcome([completed], [violation], communication, replay_verified=True, valid_end=True)
    assert score.final_goal_reached is True and score.strict_task_success is False
    invalid = evaluate_outcome([completed], [], communication, replay_verified=False, valid_end=True)
    assert invalid.evaluation_status == "INVALID_EVIDENCE" and invalid.strict_task_success is None


def test_private_metadata_cannot_create_communication_obligation_or_fact_reference():
    with pytest.raises(ValueError, match="public request"):
        PublicRequest(request_id="hidden", subject_id="line_B", fact_kind="refund_amount",
                      source=source("task_spec"), public_text="secret requirement")
    hidden = fact(source=source("oracle"), visible_to=("oracle",))
    with pytest.raises(ValueError, match="private"):
        evaluate_communication([hidden], [message("hello", refs=(hidden.fact_id,))], [], end_event_index=10)


def test_source_tampering_cannot_be_resealed_implicitly():
    original = fact()
    tampered = original.model_copy(update={"value": 9000})
    with pytest.raises(ValueError, match="hash differs"):
        evaluate_communication([tampered], [], [], end_event_index=10)
    with pytest.raises(ValueError):
        fact(value=True)
    with pytest.raises(ValueError):
        fact(valid_from=0, source=source(index=1))


def test_visibility_and_working_memory_never_import_secret_answers():
    visible = ContextItem(name="request", value="compare replacements", source=source("customer_message"),
                          audience=("service", "customer"), owner_id="buyer")
    hidden = ContextItem(name="answer", value="SKU_CORRECT", source=source("task_spec"),
                         audience=("service", "oracle"), owner_id="buyer")
    other = ContextItem(name="other_customer", value="private", source=source("public_tool"),
                        audience=("service",), owner_id="other")
    left = project("service", "buyer", (visible, hidden, other))
    right = project("service", "buyer", (visible, reseal(hidden, value="CHANGED_ANSWER"), other))
    assert left == right and left.items == (visible,)
    with pytest.raises(ValueError, match="Working Memory"):
        WorkingFact(fact=fact(source=source("oracle")), observed_at=2)
    with pytest.raises(ValueError, match="actual public"):
        WorkingState(identity="buyer", goals=(hidden,))
    observed = WorkingFact(fact=fact(valid_until=3), observed_at=2)
    state = WorkingState(identity="buyer", goals=(visible,), facts=(observed,))
    assert state.current_facts(2) == (observed.fact,)
    assert state.current_facts(4) == ()
    heldout = ContextItem(name="exam", value="answer", source=source("heldout"), audience=("learner",))
    with pytest.raises(ValueError, match="held-out"):
        project("learner", "buyer", (heldout,))


def distribution():
    return DistributionSpec(
        distribution_id="synthetic-test", version=1, population_scope="offline contract fixtures",
        sources=(BusinessSource(source_id="assumption", kind="synthetic_assumption",
                                reference="unit-test assumptions", version="1", supported_claims=("synthetic",)),),
        strata=(BusinessStratum(stratum_id="multi_goal", weight=1, weight_basis_source_id="assumption",
                                contexts=("two goals",)),
                BusinessStratum(stratum_id="information", weight=1, weight_basis_source_id="assumption",
                                contexts=("asked for money and date",))),
        split="train")


def test_distribution_provenance_and_deterministic_sampling():
    spec = distribution()
    first = sample_schedule(spec, 1, 6)
    assert first == sample_schedule(spec, 1, 6)
    assert sum(row.stratum_id == "multi_goal" for row in first) == 3
    assert first != sample_schedule(spec, 2, 6)
    with pytest.raises(ValueError, match="representativeness"):
        reseal(spec, claim="empirical_business_population")
    with pytest.raises(ValueError, match="known provenance"):
        reseal(spec, strata=[dict(spec.strata[0].model_dump(), weight_basis_source_id="imaginary", record_hash="")])


def test_sampling_keeps_unknown_rejection_and_prevents_best_run_replacement():
    spec = distribution()
    schedule = sample_schedule(spec, 1, 2)
    unknown = GenerationAttempt(slot=schedule[0], attempt=0, status="UNKNOWN",
                                accepted=False, reason="search budget exhausted")
    success = GenerationAttempt(slot=schedule[0], attempt=1, status="SAT", accepted=True, world_hash="worldA")
    audit = audit_sampling(spec, schedule, (unknown, success))
    assert audit.total_attempts == 2 and not audit.all_slots_filled
    assert ("UNKNOWN", 1) in audit.status_counts
    overwrite = GenerationAttempt(slot=schedule[0], attempt=2, status="SAT", accepted=True, world_hash="worldB")
    with pytest.raises(ValueError, match="replace"):
        audit_sampling(spec, schedule, (unknown, success, overwrite))
    same_world = GenerationAttempt(slot=schedule[1], attempt=0, status="SAT", accepted=True, world_hash="worldA")
    with pytest.raises(ValueError, match="same world"):
        audit_sampling(spec, schedule, (unknown, success, same_world))


def test_contract_bundle_exports_task_compiler_contracts_without_legacy_designer():
    from usami_v2.cli import contract_schemas

    schemas = contract_schemas()
    assert "TaskDesignCapabilities" in schemas
    assert "CustomerTaskScope" in schemas
    assert "ScenarioDraft" in schemas and "TaskDraft" in schemas
    assert "DesignerRun" not in schemas
    assert "ChallengeQualityVectorV1" not in schemas
    assert "ChallengeQualityVector" not in schemas
    assert "PreRunQualityV1" not in schemas and "PreRunQuality" not in schemas
    assert schemas["TaskDesignCapabilities"]["properties"]["protocol"]["default"] == \
        "commerce-task-design-capabilities-v1"

def test_actual_schema_export_is_immutable_offline_and_tamper_detecting(tmp_path):
    import json
    from usami_v2.cli import export_schemas, verify_schemas
    bundle = tmp_path / "schemas"
    manifest = export_schemas(bundle)
    result = verify_schemas(bundle)
    assert result["verified"] and result["schema_count"] == len(manifest["schema_hashes"])
    assert result["model_calls"] == 0
    with pytest.raises(FileExistsError):
        export_schemas(bundle)
    file = bundle / "TaskDesignCapabilities.json"
    content = json.loads(file.read_text())
    content["title"] = "tampered title"
    file.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="schema content"):
        verify_schemas(bundle)


def test_cli_does_not_pretend_unimplemented_run_or_smoke_commands_exist():
    from usami_v2.cli import main
    with pytest.raises(SystemExit) as exc:
        main(["evolve", "--execute-live"])
    assert exc.value.code == 2
