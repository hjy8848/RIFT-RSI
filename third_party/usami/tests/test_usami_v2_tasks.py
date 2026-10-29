"""Task goals score actual multi-object business outcomes and delivered information."""
from datetime import timedelta

import pytest

from usami_harness.engine import Engine, replay_trace
from usami_harness.models import Request, digest
from usami_harness.oracle import evaluate
from usami_v2.commerce import adapter
from usami_v2.commerce.models import Consent, Customer, Delegation, Operation, PREPARATIONS
from usami_v2.contracts import FactEvidence, ServiceMessage, SourceRef
from usami_v2.evaluation.outcomes import evaluate_outcome
from usami_v2.scenarios.tasks import (
    InformationObjective, RemedyGoal, TaskSpec, check_task_scope, evaluate_state_goals,
    evaluate_task_compliance, evaluate_task_communication,
)
from test_usami_v2_builder import scene
from usami_v2.scenarios.builder import build_world


def built_world():
    result = build_world(scene(), 13)
    assert result.status == "SAT"
    value, aliases = result.world, dict(result.private_aliases)
    customer_id = aliases["customer"]
    line_ids = tuple(sorted(aliases[name] for name in ("damaged", "accessory")))
    value.delegations["task-customer-request"] = Delegation(
        id="task-customer-request", customer_id=customer_id, line_ids=line_ids,
        allowed_preparations=tuple(sorted(PREPARATIONS)),
        expires_at=value.clock + timedelta(days=1),
        source_customer_event_hash=digest(["frozen task request", result.world_hash]))
    return value, aliases


def execute_one(value, name, resource, params, actor, request_no):
    operation = Operation(name=name, scope_id=next(iter(value.merchants)),
                          resource_id=resource, params=params)
    request = Request(id="task-proof-" + str(request_no), identity=actor,
                      operation=operation.model_dump(mode="json"))
    engine = Engine(adapter, value, operation_budget=1)
    engine.register(request)
    event = engine.execute(actor, operation, request.id)
    trace = engine.finish()
    report = evaluate(adapter, trace, {request.id: request})
    assert report["trace_valid"] and report["status"] == "pass"
    assert replay_trace(adapter, trace, {request.id: request}).state == engine.state
    return engine.state, event.payload["result"]


def make_task(initial, customer_id, remedies=(), information=()):
    request = ("Please resolve the damaged item and refund its accessory." if not information
               else "Please respond to this request: " + " ".join(row.public_request for row in information))
    return TaskSpec(
        task_id="paired-lines",
        world_hash=initial.state_hash,
        customer_id=customer_id,
        customer_request=request,
        remedy_goals=tuple(remedies), information_goals=tuple(information),
        partition="train", source_distribution_hash=digest("test distribution"))


def finish_remedy(state, line_id, actor, *, remedy="exchange", target="D", warehouse="fast", request_offset=0):
    issue = next(row.issue for row in state.evidence.values() if row.line_id == line_id)
    evidence_id = next(row.id for row in state.evidence.values() if row.line_id == line_id)
    state, opened = execute_one(state, "open_remedy_case", line_id,
                                {"quantity": 1, "issue": issue}, actor, request_offset)
    case_id = opened["data"]["case_id"]
    state, verified = execute_one(state, "verify_submitted_evidence", case_id,
                                  {"evidence_id": evidence_id}, actor, request_offset + 1)
    assert verified["success"]
    state, prepared = execute_one(state, "prepare_remedy_quote", case_id, {
        "remedy": remedy,
        "target_sku_id": target if remedy != "refund" else None,
        "warehouse_id": warehouse if remedy != "refund" else None,
        "pickup_slot": None,
    }, actor, request_offset + 2)
    if not prepared["success"]:
        return state, prepared
    quote_id = prepared["data"]["quote"]["id"]
    quote = state.quotes[quote_id]
    consent_id = "task-consent-" + quote_id
    state.consents[consent_id] = Consent(
        id=consent_id, customer_id=actor, quote_id=quote_id, terms_hash=quote.terms_hash,
        operation="commit_accepted_quote",
        presented_offer_event_hash=digest(["customer saw exact quote", quote_id, quote.terms_hash]),
        source_customer_event_hash=digest(["customer accepted exact quote", quote_id, quote.terms_hash]),
        expires_at=quote.valid_until)
    state, result = execute_one(state, "commit_accepted_quote", quote_id, {}, actor, request_offset + 3)
    return state, result


def test_joint_task_accepts_both_valid_skus_and_scores_two_order_lines():
    for target, charge in (("D", 2000), ("E", 1000)):
        initial, aliases = built_world()
        customer_id = aliases["customer"]
        damaged, accessory = aliases["damaged"], aliases["accessory"]
        task = make_task(initial, customer_id, remedies=(
            RemedyGoal(goal_id="damaged-item", line_id=damaged, quantity=1,
                       allowed_remedies=("exchange",),
                       required_attributes=({"name": "interface", "value": "USB_C"},),
                       maximum_extra_charge_cents=2000,
                       latest_arrival=initial.clock + timedelta(days=3)),
            RemedyGoal(goal_id="accessory-refund", line_id=accessory, quantity=1,
                       allowed_remedies=("refund",), maximum_extra_charge_cents=0),
        ))
        # A valid plan may use either compliant SKU and independently refund the second line.
        value, result = finish_remedy(initial, damaged, customer_id,
                                      target=aliases["correct_choice_" + target],
                                      warehouse=aliases["fast"])
        assert result["success"]
        exchange_quote = next(row for row in value.quotes.values() if row.remedy == "exchange")
        assert exchange_quote.extra_charge_cents == charge
        value, result = finish_remedy(value, accessory, customer_id, remedy="refund", request_offset=4)
        assert result["success"]
        scores = {row.goal_id: row.status for row in evaluate_state_goals(initial, value, task)}
        assert scores == {"damaged-item": "PASS", "accessory-refund": "PASS"}
        assert evaluate_task_compliance(initial, value, task)[0].status == "PASS"


@pytest.mark.parametrize("target,warehouse", [("wrong", "fast"), ("late", "slow"), ("expensive", "fast")])
def test_joint_goal_rejects_wrong_attribute_late_or_over_budget_solutions(target, warehouse):
    initial, aliases = built_world()
    customer_id = aliases["customer"]
    line_id = aliases["damaged"]
    task = make_task(initial, customer_id, remedies=(RemedyGoal(
        goal_id="bounded-replacement", line_id=line_id, quantity=1,
        allowed_remedies=("exchange",),
        required_attributes=({"name": "interface", "value": "USB_C"},),
        maximum_extra_charge_cents=2000,
        latest_arrival=initial.clock + timedelta(days=3)),))
    value, result = finish_remedy(initial, line_id, customer_id,
                                  target=aliases[target], warehouse=aliases[warehouse])
    assert result["success"]
    scores = {row.goal_id: row.status for row in evaluate_state_goals(initial, value, task)}
    assert scores["bounded-replacement"] == "FAIL"


def test_information_only_task_uses_real_delivered_text_and_does_not_need_a_transaction():
    initial, aliases = built_world()
    line_id = aliases["damaged"]
    task = make_task(initial, aliases["customer"], information=(InformationObjective(
        goal_id="refund-balance", subject_id=line_id, subject_mention="this item", fact_kind="refund_amount",
        public_request="How much was paid for this item?"),))
    fact = FactEvidence(
        fact_id="balance-a", subject_id=line_id, kind="refund_amount", value=20000,
        unit="CNY_cents", valid_from=2,
        source=SourceRef(source_id="tool-event-2", source_kind="public_tool",
                         source_hash=digest("actual get_refundable_balance result"), event_index=1),
        visible_to=("service", "oracle"))
    message = ServiceMessage(
        message_id="answer-1", event_index=3, text=line_id + ": refund amount is CNY 200.00.",
        subject_ids=(line_id,), fact_refs=(fact.fact_id,))
    scores = evaluate_state_goals(initial, initial, task)
    assert scores == ()
    assert evaluate_task_compliance(initial, initial, task)[0].status == "PASS"
    communication = evaluate_task_communication(
        initial, task, facts=(fact,), messages=(message,), end_event_index=4)
    assert communication.success is True
    outcome = evaluate_outcome(scores, evaluate_task_compliance(initial, initial, task), communication,
                               replay_verified=True, valid_end=True)
    assert outcome.final_goal_reached is None
    assert outcome.state_goal_applicability == "NOT_APPLICABLE"
    assert outcome.strict_task_success is True
    assert not initial.settlements


def test_information_goal_marks_wrong_claim_missing_answer_and_ambiguity_separately():
    initial, aliases = built_world()
    line_id = aliases["damaged"]
    task = make_task(initial, aliases["customer"], information=(InformationObjective(
        goal_id="refund-balance", subject_id=line_id, subject_mention="this item", fact_kind="refund_amount",
        public_request="How much was paid for this item?"),))
    fact = FactEvidence(
        fact_id="balance-a", subject_id=line_id, kind="refund_amount", value=20000,
        unit="CNY_cents", valid_from=2,
        source=SourceRef(source_id="tool-event-2", source_kind="public_tool",
                         source_hash=digest("actual tool result"), event_index=1))
    wrong = ServiceMessage(message_id="wrong", event_index=3,
                           text=line_id + ": refund amount is CNY 99.00.",
                           subject_ids=(line_id,), fact_refs=(fact.fact_id,))
    wrong_result = evaluate_task_communication(
        initial, task, facts=(fact,), messages=(wrong,), end_event_index=4)
    assert wrong_result.success is False and wrong_result.incorrect_facts == 1
    missing_result = evaluate_task_communication(initial, task, facts=(fact,), messages=(), end_event_index=4)
    assert missing_result.success is False and missing_result.missing == 1
    ambiguous = ServiceMessage(message_id="ambiguous", event_index=3,
                               text=line_id + ": The refund may be around 200 yuan.",
                               subject_ids=(line_id,))
    unresolved = evaluate_task_communication(
        initial, task, facts=(fact,), messages=(ambiguous,), end_event_index=4)
    assert unresolved.success is None and unresolved.unresolved > 0


def test_information_only_task_does_not_count_unrequested_settlement_as_success():
    initial, aliases = built_world()
    customer_id = aliases["customer"]
    task = make_task(initial, customer_id, information=(InformationObjective(
        goal_id="order-status", subject_id=aliases["purchase"], subject_mention="the order",
        fact_kind="refund_status",
        public_request="Please tell me the order status only."),))
    value, result = finish_remedy(initial, aliases["damaged"], customer_id, remedy="refund")
    assert result["success"]
    assert evaluate_state_goals(initial, value, task) == ()
    assert evaluate_task_compliance(initial, value, task)[0].status == "FAIL"


def test_task_cannot_cross_customer_or_mutate_frozen_initial_facts():
    initial, aliases = built_world()
    wrong_customer = initial.clone()
    other_id = "other_customer"
    wrong_customer.customers[other_id] = Customer(id=other_id, name="Other Customer")
    order_id = aliases["purchase"]
    wrong_customer.orders[order_id].customer_id = other_id
    for row in wrong_customer.evidence.values():
        row.customer_id = other_id
    wrong_customer.delegations.clear()
    task = make_task(wrong_customer, aliases["customer"], remedies=(RemedyGoal(
        goal_id="refund", line_id=aliases["damaged"], quantity=1, allowed_remedies=("refund",)),))
    with pytest.raises(ValueError, match="frozen Customer identity"):
        check_task_scope(wrong_customer, task)

    initial, aliases = built_world()
    task = make_task(initial, aliases["customer"], remedies=(RemedyGoal(
        goal_id="refund", line_id=aliases["damaged"], quantity=1, allowed_remedies=("refund",)),))
    changed = initial.clone()
    changed.skus[aliases["original"]].attributes["interface"] = "TAMPERED"
    with pytest.raises(ValueError, match="native business invariants|frozen source business facts"):
        evaluate_state_goals(initial, changed, task)


def test_task_world_binding_and_preexisting_line_commitment_are_checked():
    initial, aliases = built_world()
    with pytest.raises(ValueError, match="grounded in the public Customer request"):
        TaskSpec(
            task_id="hidden-question", world_hash=initial.state_hash,
            customer_id=aliases["customer"], customer_request="Please refund the accessory.",
            information_goals=(InformationObjective(
                goal_id="hidden-amount", subject_id=aliases["damaged"], subject_mention="this item",
                fact_kind="refund_amount", public_request="How much did I pay for this item?"),),
            partition="train", source_distribution_hash=digest("test distribution"))
    # Cross-customer information references and invented targets are rejected.
    with pytest.raises(ValueError, match="not about an owned"):
        check_task_scope(initial, make_task(initial, aliases["customer"], information=(
            InformationObjective(goal_id="other", subject_id="made-up", subject_mention="How many",
                                 fact_kind="quantity",
                                 public_request="How many?"),)))

    task = make_task(initial, aliases["customer"], remedies=(RemedyGoal(
        goal_id="too-many", line_id=aliases["damaged"], quantity=2,
        allowed_remedies=("refund",)),))
    with pytest.raises(ValueError, match="uncommitted purchased quantity"):
        check_task_scope(initial, task)

    foreign = initial.clone()
    foreign.seed += 1
    with pytest.raises(ValueError, match="world binding"):
        check_task_scope(foreign, make_task(initial, aliases["customer"], information=(
            InformationObjective(goal_id="order-status", subject_id=aliases["purchase"], subject_mention="Status",
                                 fact_kind="refund_status", public_request="Status?"),)))
