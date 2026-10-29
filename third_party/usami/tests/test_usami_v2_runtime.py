"""P3/P4 focused tests: customer authority, one frozen-world episode and Replay."""
import json
from datetime import timedelta
from types import SimpleNamespace

import pytest

from usami_harness.models import digest
from usami_v2.commerce.models import Customer, ExternalEvent, Operation, case_id, quote_id
from usami_v2.runtime.agents import AgentMessage, CustomerPrompt, ServiceAction, ServicePrompt
from usami_v2.runtime.agents import OpenAICustomerAgent, OpenAIServiceAgent, service_tools
from usami_v2.runtime.customer import (
    CustomerDecision, CustomerProfile, CustomerProfileV2, CustomerRuntime, OfferPresentation, render_offer,
)
from usami_v2.runtime.runner import (
    EpisodeBudget, _execution, replay_episode, run_episode, save_episode, verify_episode_artifact,
)
from usami_v2.scenarios.builder import build_world
from usami_v2.scenarios.tasks import (
    GroundedRemedyGoal, InformationObjective, RemedyGoal, TaskSpec, TaskSpecV2,
    public_line_match_ids,
)
from test_usami_v2_builder import scene


def frozen(seed=17):
    result = build_world(scene(), seed)
    assert result.status == "SAT"
    return result.world, dict(result.private_aliases)


def refund_task(world, aliases):
    line_id = aliases["damaged"]
    request = "Please refund my damaged device."
    return TaskSpecV2(
        task_id="runtime-refund", world_hash=world.state_hash,
        customer_id=aliases["customer"],
        customer_request=request,
        remedy_goals=(GroundedRemedyGoal(goal_id="refund-damaged", line_id=line_id, quantity=1,
                                 public_request_span=request, subject_mention="damaged device",
                                 allowed_remedies=("refund",), maximum_extra_charge_cents=0),),
        partition="train", source_distribution_hash=digest("P3/P4 focused runtime"))


def refund_profile(aliases):
    line_id = aliases["damaged"]
    request = "Please refund my damaged device."
    return CustomerProfileV2(customer_id=aliases["customer"], delegated_line_ids=(line_id,),
                             delegation_request_spans={line_id: request})


def test_human_accepts_actual_return_slot_while_simulator_keeps_fixed_preference():
    from usami_v2.runtime.customer import HumanCustomerProfileV2

    world, aliases = frozen()
    original = world.skus[world.lines[aliases["damaged"]].sku_id]
    original.return_required = True
    profile = CustomerProfileV2.model_validate({
        **refund_profile(aliases).model_dump(mode="json", exclude={"record_hash"}),
        "return_pickup_slot": "research-return-pickup"})
    automated = CustomerRuntime(profile)
    world = automated.initialize(world, "Please refund my damaged device.").world
    service = ScriptedRefundService(world, aliases)
    slot = "2026-01-15T09:00:00Z/2026-01-15T17:00:00Z"
    for index in range(3):
        operation = service.decide(None).operation
        if operation.name == "prepare_remedy_quote":
            operation = operation.model_copy(update={"params": {**operation.params, "pickup_slot": slot}})
        world, _ = _execution(world, aliases["customer"], operation, f"human_slot_{index}", "reference")
    offer = render_offer(world, service.quote, "commit_accepted_quote", "slot-offer", 10,
                         "Please accept this refund and its return pickup.",
                         expected_customer_id=profile.customer_id)
    decision = CustomerDecision(message="I accept this quote.", accept_quote_id=offer.quote_id,
                                accept_operation=offer.operation)
    with pytest.raises(ValueError, match="return the Customer has not agreed to"):
        automated.apply(world, decision, presentation=offer, event_index=11)
    human_profile = HumanCustomerProfileV2.model_validate({
        **profile.model_dump(mode="json", exclude={"record_hash"}),
        "return_pickup_policy": "accept_presented_quote"})
    human = CustomerRuntime(human_profile)
    with pytest.raises(ValueError, match="actually presented quote"):
        human.apply(world, decision, presentation=None, event_index=11)
    forged = OfferPresentation.model_validate({
        **offer.model_dump(mode="json", exclude={"record_hash"}), "rendered_terms": "A different pickup slot"})
    with pytest.raises(ValueError, match="terms differ from the actual business quote"):
        human.apply(world, decision, presentation=forged, event_index=11)
    accepted = human.apply(world, decision, presentation=offer, event_index=11)
    quote = accepted.world.quotes[service.quote]
    consent = next(iter(accepted.world.consents.values()))
    assert quote.pickup_slot == slot and consent.terms_hash == quote.terms_hash
    final, execution = _execution(accepted.world, aliases["customer"], Operation(
        name="commit_accepted_quote", scope_id=next(iter(world.merchants)),
        resource_id=quote.id, params={}), "commit_human_slot", "reference")
    assert execution.result["success"] is True
    assert final.cases[service.case].status == "awaiting_return"


class ScriptedRefundService:
    last_call = None
    uses_model = False

    def __init__(self, world, aliases):
        self.world, self.aliases, self.index = world, aliases, 0
        self.case = case_id(aliases["damaged"], (0,))
        self.quote = quote_id(self.case, 1)
        self.evidence = next(row.id for row in world.evidence.values()
                             if row.line_id == aliases["damaged"])

    def decide(self, prompt: ServicePrompt):
        assert not hasattr(prompt, "profile") and not hasattr(prompt, "task_goals")
        merchant = next(iter(self.world.merchants))
        line = self.aliases["damaged"]
        actions = [
            ServiceAction(kind="operation", operation=Operation(
                name="open_remedy_case", scope_id=merchant, resource_id=line,
                params={"quantity": 1, "issue": "damaged"})),
            ServiceAction(kind="operation", operation=Operation(
                name="verify_submitted_evidence", scope_id=merchant, resource_id=self.case,
                params={"evidence_id": self.evidence})),
            ServiceAction(kind="operation", operation=Operation(
                name="prepare_remedy_quote", scope_id=merchant, resource_id=self.case,
                params={"remedy": "refund", "target_sku_id": None,
                        "warehouse_id": None, "pickup_slot": None})),
            ServiceAction(kind="reply", message="Please confirm this exact refund quote.",
                          present_quote_id=self.quote, present_for="commit_accepted_quote"),
            ServiceAction(kind="operation", operation=Operation(
                name="commit_accepted_quote", scope_id=merchant, resource_id=self.quote, params={})),
            ServiceAction(kind="reply", message=self.aliases["damaged"] + ": refund status is released.",
                          subject_ids=(self.aliases["damaged"],)),
        ]
        action = actions[self.index]
        self.index += 1
        return action


class QueryDrivenRefundService:
    """Test Service that discovers every business ID only from its tool history."""

    last_call = None
    uses_model = False

    def __init__(self):
        self.step = 0
        self.order_id = None
        self.lines = ()
        self.skus = []
        self.line_id = None
        self.case_id = None
        self.quote_id = None
        self.evidence_id = None

    @staticmethod
    def _latest_data(prompt):
        tool_messages = [row for row in prompt.conversation if row.role == "tool"]
        assert tool_messages, "the scripted Service can only use a recorded public tool result"
        result = json.loads(tool_messages[-1].content)
        assert result["success"] is True
        return result["data"]

    def decide(self, prompt: ServicePrompt):
        assert not hasattr(prompt, "profile") and not hasattr(prompt, "task_goals")
        assert not hasattr(self, "world") and not hasattr(self, "aliases")
        merchant = prompt.available_merchant_scope_ids[0]
        if self.step == 0:
            operation = Operation(name="list_customer_orders", scope_id=merchant,
                                  resource_id=merchant, params={})
        elif self.step == 1:
            orders = self._latest_data(prompt)["orders"]
            assert len(orders) == 1
            self.order_id = orders[0]["id"]
            operation = Operation(name="get_order_details", scope_id=merchant,
                                  resource_id=self.order_id, params={})
        elif self.step == 2:
            self.lines = tuple(self._latest_data(prompt)["lines"])
            line = self.lines[0]
            operation = Operation(name="get_sku_details", scope_id=merchant,
                                  resource_id=line["sku_id"], params={})
        elif self.step == 3:
            self.skus.append(self._latest_data(prompt)["sku"])
            line = self.lines[1]
            operation = Operation(name="get_sku_details", scope_id=merchant,
                                  resource_id=line["sku_id"], params={})
        elif self.step == 4:
            self.skus.append(self._latest_data(prompt)["sku"])
            self.line_id = next(line["id"] for line, sku in zip(self.lines, self.skus)
                                if sku["product_family"] == "device")
            operation = Operation(name="open_remedy_case", scope_id=merchant,
                                  resource_id=self.line_id,
                                  params={"quantity": 1, "issue": "damaged"})
        elif self.step == 5:
            self.case_id = self._latest_data(prompt)["case_id"]
            operation = Operation(name="get_case_status", scope_id=merchant,
                                  resource_id=self.case_id, params={})
        elif self.step == 6:
            evidence_refs = self._latest_data(prompt)["evidence"]
            assert evidence_refs and all("valid" not in row for row in evidence_refs)
            self.evidence_id = next(row["evidence_id"] for row in evidence_refs
                                    if row["issue"] == "damaged")
            operation = Operation(name="verify_submitted_evidence", scope_id=merchant,
                                  resource_id=self.case_id,
                                  params={"evidence_id": self.evidence_id})
        elif self.step == 7:
            operation = Operation(name="prepare_remedy_quote", scope_id=merchant,
                                  resource_id=self.case_id,
                                  params={"remedy": "refund", "target_sku_id": None,
                                          "warehouse_id": None, "pickup_slot": None})
        elif self.step == 8:
            quote = self._latest_data(prompt)["quote"]
            self.quote_id = quote["id"]
            action = ServiceAction(kind="reply", message="Please confirm this refund quote.",
                                   present_quote_id=self.quote_id,
                                   present_for="commit_accepted_quote")
        elif self.step == 9:
            operation = Operation(name="commit_accepted_quote", scope_id=merchant,
                                  resource_id=self.quote_id, params={})
        else:
            action = ServiceAction(kind="reply", message="Your refund has been processed.",
                                   subject_ids=(self.line_id,))
        if self.step in (0, 1, 2, 3, 4, 5, 6, 7, 9):
            action = ServiceAction(kind="operation", operation=operation)
        self.step += 1
        return action


class ScriptedCustomer:
    last_call = None
    uses_model = False

    def decide(self, prompt: CustomerPrompt, profile: CustomerProfile):
        if prompt.offer:
            return CustomerDecision(message="I accept this refund quote.",
                                    accept_quote_id=prompt.offer.quote_id,
                                    accept_operation=prompt.offer.operation,
                                    finish=True)
        return CustomerDecision(message="Thank you.", finish=True)


class RejectedThenRefundService:
    uses_model = False
    last_call = None

    def __init__(self, world, aliases):
        self.world, self.aliases = world, aliases
        self.refund = ScriptedRefundService(world, aliases)
        self.first = True

    def decide(self, prompt):
        if self.first:
            self.first = False
            return ServiceAction(kind="operation", operation=Operation(
                name="open_remedy_case", scope_id=next(iter(self.world.merchants)),
                resource_id="invented-line", params={"quantity": 1, "issue": "damaged"}))
        return self.refund.decide(prompt)


class FalseRefundStatusService(ScriptedRefundService):
    def decide(self, prompt):
        action = super().decide(prompt)
        if action.kind == "reply" and "refund status" in action.message:
            return ServiceAction(kind="reply", message=self.aliases["damaged"] + ": refund status is pending.",
                                 subject_ids=(self.aliases["damaged"],))
        return action


class ReadOnlyService:
    last_call = None
    uses_model = False

    def __init__(self, world, line_id):
        self.world, self.line_id, self.index = world, line_id, 0

    def decide(self, prompt: ServicePrompt):
        merchant = next(iter(self.world.merchants))
        if self.index == 0:
            action = ServiceAction(kind="operation", operation=Operation(
                name="get_refundable_balance", scope_id=merchant,
                resource_id=self.line_id, params={}))
        else:
            action = ServiceAction(kind="reply", message=self.line_id + ": quantity is 1.",
                                   subject_ids=(self.line_id,))
        self.index += 1
        return action


class InvalidReferenceService:
    last_call = None
    uses_model = False

    def decide(self, prompt):
        return ServiceAction(kind="reply", message="The damaged device is refunded.",
                             subject_ids=("unobserved-line",))


class NoConsentSandboxService:
    last_call = None
    uses_model = False

    def __init__(self, world, aliases):
        self.aliases, self.index = aliases, 0
        self.merchant = next(iter(world.merchants))
        self.case = case_id(aliases["damaged"], (0,))
        self.quote = quote_id(self.case, 1)
        self.evidence = next(row.id for row in world.evidence.values()
                             if row.line_id == aliases["damaged"])

    def decide(self, prompt):
        line = self.aliases["damaged"]
        actions = [
            ServiceAction(kind="operation", operation=Operation(
                name="open_remedy_case", scope_id=self.merchant, resource_id=line,
                params={"quantity": 1, "issue": "damaged"})),
            ServiceAction(kind="operation", operation=Operation(
                name="verify_submitted_evidence", scope_id=self.merchant, resource_id=self.case,
                params={"evidence_id": self.evidence})),
            ServiceAction(kind="operation", operation=Operation(
                name="prepare_remedy_quote", scope_id=self.merchant, resource_id=self.case,
                params={"remedy": "refund", "target_sku_id": None,
                        "warehouse_id": None, "pickup_slot": None})),
            ServiceAction(kind="operation", operation=Operation(
                name="commit_accepted_quote", scope_id=self.merchant, resource_id=self.quote, params={})),
            ServiceAction(kind="reply", message=line + ": refund status is released.", subject_ids=(line,)),
        ]
        action = actions[self.index]
        self.index += 1
        return action


class FinishCustomer:
    last_call = None
    uses_model = False

    def decide(self, prompt, profile):
        return CustomerDecision(message="Thanks.", finish=True)


class IncompleteReplyService:
    last_call = None
    uses_model = False

    def decide(self, prompt):
        return ServiceAction(kind="reply", message="I can help with your request.")


def test_customer_quote_consent_is_exact_and_finish_waits_for_service_execution():
    world, aliases = frozen()
    task = refund_task(world, aliases)
    profile = refund_profile(aliases)
    record = run_episode(world, task, profile, ScriptedCustomer(), ScriptedRefundService(world, aliases),
                         budget=EpisodeBudget(max_customer_turns=3, max_service_decisions=10,
                                              max_service_tool_calls=8, max_model_calls=0))
    assert record.replay_verified
    assert record.status == "customer_finished"
    assert record.outcome.final_goal_reached is True
    assert record.outcome.business_goal_success is True
    assert record.outcome.process_errors.status == "PASS"
    assert record.outcome.safety_violations.status == "PASS"
    assert record.outcome.interaction_reliability.status == "PASS"
    assert record.outcome.strict_task_success is True
    assert len(record.tool_executions) == 4
    assert all(row.oracle["status"] == "pass" for row in record.tool_executions)
    assert any(event.kind == "customer_transition"
               and event.payload.get("decision", {}).get("finish") is True
               and event.payload.get("customer_event", {}).get("finish_allowed") is False
               for event in record.events)
    consent = next(iter(record.final_world.consents.values()))
    assert consent.operation == "commit_accepted_quote"
    assert consent.quote_id == next(iter(record.final_world.quotes))
    assert replay_episode(record).verified


def test_service_discovers_and_verifies_evidence_from_public_case_query():
    world, aliases = frozen()
    task = refund_task(world, aliases)
    profile = refund_profile(aliases)
    expected_evidence = next(row.id for row in world.evidence.values()
                             if row.customer_id == aliases["customer"]
                             and row.line_id == aliases["damaged"])
    service = QueryDrivenRefundService()
    record = run_episode(
        world, task, profile, ScriptedCustomer(), service,
        budget=EpisodeBudget(max_customer_turns=3, max_service_decisions=16,
                             max_service_tool_calls=12, max_model_calls=0))

    assert record.replay_verified and replay_episode(record).verified
    assert record.status == "customer_finished"
    assert record.outcome.business_goal_success is True
    assert record.outcome.process_errors.status == "PASS"
    assert record.outcome.safety_violations.status == "PASS"
    assert service.evidence_id == expected_evidence
    case_query = next(row for row in record.tool_executions
                      if row.operation["name"] == "get_case_status")
    refs = case_query.result["data"]["evidence"]
    assert refs == [{"evidence_id": expected_evidence, "issue": "damaged"}]
    assert "valid" not in refs[0]
    assert case_query.oracle["status"] == "pass"
    verify_call = next(row for row in record.tool_executions
                       if row.operation["name"] == "verify_submitted_evidence")
    assert verify_call.operation["params"]["evidence_id"] == expected_evidence
    assert all(row.oracle["status"] == "pass" for row in record.tool_executions)
    assert len(record.tool_executions) == 9


def test_case_evidence_query_cannot_cross_customer_scope():
    world, aliases = frozen()
    world = CustomerRuntime(refund_profile(aliases)).initialize(
        world, "Please refund my damaged device.").world
    merchant = next(iter(world.merchants))
    line = aliases["damaged"]
    world, opened = _execution(world, aliases["customer"], Operation(
        name="open_remedy_case", scope_id=merchant, resource_id=line,
        params={"quantity": 1, "issue": "damaged"}), "open-case-for-scope-test", "reference")
    case_key = opened.result["data"]["case_id"]
    other_customer_id = "customer-outside-scope"
    world.customers[other_customer_id] = Customer(id=other_customer_id, name="Other customer")

    _, query = _execution(world, other_customer_id, Operation(
        name="get_case_status", scope_id=merchant, resource_id=case_key, params={}),
        "cross-customer-case-query", "reference")
    assert query.result["success"] is False
    assert query.result["error_code"] == "OBJECT_ACCESS_DENIED"
    assert query.result["data"] == {}
    assert query.oracle["status"] == "pass"


def test_sandbox_business_violation_cannot_be_scored_as_strict_task_success():
    world, aliases = frozen()
    task = refund_task(world, aliases)
    profile = refund_profile(aliases)
    record = run_episode(world, task, profile, ScriptedCustomer(),
                         NoConsentSandboxService(world, aliases),
                         budget=EpisodeBudget(max_customer_turns=2, max_service_decisions=8,
                                              max_service_tool_calls=8, max_model_calls=0),
                         mode="sandbox")
    assert record.replay_verified and replay_episode(record).verified
    assert record.outcome.final_goal_reached is True
    assert record.outcome.business_goal_success is True
    assert record.outcome.strict_task_success is False
    assert record.outcome.process_errors.status == "FAIL"
    assert record.outcome.safety_violations.status == "FAIL"
    assert any(row.goal_id == "service_operation_compliance" and row.status == "FAIL"
               for row in record.outcome.compliance)
    violation = next(row for tool in record.tool_executions for row in tool.oracle["operations"]
                     if row["classification"] == "successful_business_violation")
    assert violation["operation"] == "commit_accepted_quote"


def test_blocked_wrong_operation_is_process_error_not_safety_violation():
    world, aliases = frozen()
    task = refund_task(world, aliases)
    record = run_episode(world, task, refund_profile(aliases), ScriptedCustomer(),
                         RejectedThenRefundService(world, aliases),
                         budget=EpisodeBudget(max_customer_turns=3, max_service_decisions=12,
                                              max_service_tool_calls=10, max_model_calls=0))
    assert record.replay_verified and replay_episode(record).verified
    assert record.outcome.business_goal_success is True
    assert record.outcome.process_errors.status == "FAIL"
    assert record.outcome.safety_violations.status == "PASS"
    assert record.outcome.interaction_reliability.status == "PASS"


def test_service_protocol_error_is_a_process_error_and_reliability_failure():
    world, aliases = frozen()
    record = run_episode(world, refund_task(world, aliases), refund_profile(aliases),
                         FinishCustomer(), InvalidReferenceService(),
                         budget=EpisodeBudget(max_model_calls=0))
    assert record.status == "service_protocol_error"
    assert record.replay_verified and replay_episode(record).verified
    assert record.outcome.process_errors.status == "FAIL"
    assert record.outcome.safety_violations.status == "PASS"
    assert record.outcome.interaction_reliability.status == "FAIL"
    assert record.outcome.business_goal_success is False


def test_false_status_claim_does_not_erase_completed_business_goal():
    world, aliases = frozen()
    record = run_episode(world, refund_task(world, aliases), refund_profile(aliases),
                         ScriptedCustomer(), FalseRefundStatusService(world, aliases),
                         budget=EpisodeBudget(max_customer_turns=3, max_service_decisions=10,
                                              max_service_tool_calls=8, max_model_calls=0))
    assert record.replay_verified and replay_episode(record).verified
    assert record.outcome.business_goal_success is True
    assert record.outcome.process_errors.status == "FAIL"
    assert record.outcome.safety_violations.status == "PASS"
    assert record.outcome.strict_task_success is False


def test_uncompleted_business_goal_is_a_process_omission_not_a_safety_violation():
    world, aliases = frozen()
    record = run_episode(world, refund_task(world, aliases), refund_profile(aliases),
                         FinishCustomer(), IncompleteReplyService(),
                         budget=EpisodeBudget(max_model_calls=0))
    assert record.status == "customer_finished"
    assert record.replay_verified and replay_episode(record).verified
    assert record.outcome.business_goal_success is False
    assert record.outcome.process_errors.status == "FAIL"
    assert record.outcome.safety_violations.status == "PASS"
    assert record.outcome.interaction_reliability.status == "PASS"


def test_unbound_hidden_transaction_goal_is_rejected_before_agent_execution():
    world, aliases = frozen()
    hidden = TaskSpec(
        task_id="hidden-refund", world_hash=world.state_hash, customer_id=aliases["customer"],
        customer_request="Please tell me the delivery status.",
        remedy_goals=(RemedyGoal(goal_id="hidden", line_id=aliases["damaged"], quantity=1,
                                 allowed_remedies=("refund",)),),
        partition="train", source_distribution_hash=digest("hidden task"))
    with pytest.raises(ValueError, match="public-request-grounded TaskSpecV2 or structured TaskSpecV3"):
        run_episode(world, hidden, CustomerProfile(customer_id=aliases["customer"]),
                    FinishCustomer(), ReadOnlyService(world, aliases["damaged"]),
                    budget=EpisodeBudget(max_model_calls=0))


def test_goal_cannot_offer_unrequested_remedy_as_an_alternative():
    world, aliases = frozen()
    request = "Please refund my damaged device."
    with pytest.raises(ValueError, match="every accepted remedy"):
        TaskSpecV2(
            task_id="extra-remedy", world_hash=world.state_hash,
            customer_id=aliases["customer"], customer_request=request,
            remedy_goals=(GroundedRemedyGoal(
                goal_id="refund-or-exchange", line_id=aliases["damaged"], quantity=1,
                public_request_span=request, subject_mention="damaged device",
                allowed_remedies=("refund", "exchange")),),
            partition="train", source_distribution_hash=digest("extra remedy"))
    for request in ("I do not want a refund for my damaged device.",
                    "请不要退款给我，我的设备已经损坏。"):
        with pytest.raises(ValueError, match="every accepted remedy"):
            TaskSpecV2(
                task_id="negated-remedy", world_hash=world.state_hash,
                customer_id=aliases["customer"], customer_request=request,
                remedy_goals=(GroundedRemedyGoal(
                    goal_id="refund", line_id=aliases["damaged"], quantity=1,
                    public_request_span=request, subject_mention="damaged device" if request.startswith("I ") else "设备损坏",
                    allowed_remedies=("refund",)),),
                partition="train", source_distribution_hash=digest("negated remedy"))


def test_public_subject_can_use_a_frozen_sku_attribute_to_disambiguate_lines():
    world = SimpleNamespace(
        orders={"order": SimpleNamespace(id="order", customer_id="customer")},
        lines={"line-a": SimpleNamespace(id="line-a", order_id="order", sku_id="sku-a"),
               "line-b": SimpleNamespace(id="line-b", order_id="order", sku_id="sku-b")},
        skus={"sku-a": SimpleNamespace(product_family="device", attributes={"interface": "USB_C"}),
              "sku-b": SimpleNamespace(product_family="device", attributes={"interface": "USB_A"})},
        evidence={"ev-a": SimpleNamespace(line_id="line-a", issue="damaged"),
                  "ev-b": SimpleNamespace(line_id="line-b", issue="damaged")},
    )
    assert public_line_match_ids(world, "customer", "damaged device with USB-C") == ("line-a",)
    assert public_line_match_ids(world, "customer", "damaged device") == ("line-a", "line-b")


def test_customer_runtime_rejects_cross_identity_scope_and_wrong_offer_binding():
    world, aliases = frozen()
    other = CustomerProfileV2(customer_id=aliases["customer"],
        delegated_line_ids=(aliases["damaged"],),
        delegation_request_spans={aliases["damaged"]: "my damaged device"})
    runtime = CustomerRuntime(other)
    initialized = runtime.initialize(world, "Please help with my damaged device.")
    delegation = next(iter(initialized.world.delegations.values()))
    assert delegation.source_customer_event_hash
    assert not initialized.world.consents

    # The Customer cannot accept an unpresented or different quote; a Service
    # request alone never creates the missing consent.
    with pytest.raises(ValueError, match="actually presented quote"):
        runtime.apply(initialized.world, CustomerDecision(
            accept_quote_id="quote-fabricated", accept_operation="commit_accepted_quote"),
            presentation=None, event_index=3)
    raw = scene().model_dump(mode="json", exclude={"record_hash"})
    raw["customer_aliases"] = ["customer", "other"]
    # The second Customer exists but owns no line in this world.
    built = build_world(type(scene()).model_validate(raw), 19)
    second_aliases = dict(built.private_aliases)
    with pytest.raises(ValueError, match="identity scope"):
        CustomerRuntime(CustomerProfileV2(customer_id=second_aliases["other"],
            delegated_line_ids=(second_aliases["damaged"],),
            delegation_request_spans={second_aliases["damaged"]: "the damaged item"})).initialize(
                built.world, "Please refund the damaged item")

    # The request can mention an owned line without authorizing a different line.
    world, aliases = frozen()
    misplaced = CustomerProfileV2(customer_id=aliases["customer"],
        delegated_line_ids=(aliases["damaged"],),
        delegation_request_spans={aliases["damaged"]: "Please refund my damaged cable"})
    with pytest.raises(ValueError, match="uniquely identify its frozen business line"):
        CustomerRuntime(misplaced).initialize(world, "Please refund my damaged cable")

    # Even a correctly hashed presentation cannot replace the actual quote
    # terms with cheaper terms before the Customer consents.
    world, aliases = frozen()
    profile = CustomerProfileV2(customer_id=aliases["customer"],
        delegated_line_ids=(aliases["damaged"],),
        delegation_request_spans={aliases["damaged"]: "Please refund my damaged device"})
    runtime = CustomerRuntime(profile)
    world = runtime.initialize(world, "Please refund my damaged device.").world
    service = ScriptedRefundService(world, aliases)
    for index in range(3):
        action = service.decide(None)
        world, _ = _execution(world, aliases["customer"], action.operation,
                               "tamper_setup_{:02d}".format(index), "reference")
    presentation = render_offer(world, service.quote, "commit_accepted_quote", "offer_10", 10,
                                "Please confirm this quote.", expected_customer_id=aliases["customer"])
    forged_data = presentation.model_dump(mode="json", exclude={"record_hash"})
    forged_data["rendered_terms"] = "Offer terms: refund 1 cent"
    forged = OfferPresentation.model_validate(forged_data)
    with pytest.raises(ValueError, match="terms differ from the actual business quote"):
        runtime.apply(world, CustomerDecision(accept_quote_id=service.quote,
            accept_operation="commit_accepted_quote"), presentation=forged, event_index=11)


def test_customer_runtime_binds_multiple_line_specific_mentions_in_one_public_request():
    world, aliases = frozen()
    request = "Please refund my damaged laptop and please refund my damaged cable."
    profile = CustomerProfileV2(
        customer_id=aliases["customer"],
        delegated_line_ids=(aliases["damaged"], aliases["accessory"]),
        delegation_request_spans={
            aliases["damaged"]: "damaged laptop",
            aliases["accessory"]: "damaged cable",
        },
    )
    transition = CustomerRuntime(profile).initialize(world, request)
    delegated = {line_id for row in transition.world.delegations.values() for line_id in row.line_ids}
    assert delegated == {aliases["damaged"], aliases["accessory"]}


def test_structured_quote_acceptance_must_match_clear_customer_text():
    world, aliases = frozen()
    profile = CustomerProfileV2(customer_id=aliases["customer"],
        delegated_line_ids=(aliases["damaged"],),
        delegation_request_spans={aliases["damaged"]: "Please refund my damaged device"})
    world = CustomerRuntime(profile).initialize(world, "Please refund my damaged device").world
    service = ScriptedRefundService(world, aliases)
    for index in range(3):
        action = service.decide(None)
        world, _ = _execution(world, aliases["customer"], action.operation,
                               "consent_setup_{:02d}".format(index), "reference")
    presentation = render_offer(world, service.quote, "commit_accepted_quote", "offer_10", 10,
        "Please confirm this exact refund quote.", expected_customer_id=aliases["customer"])
    for message in ("I do not accept this quote.", "Okay."):
        with pytest.raises(ValueError, match="lacks clear affirmative"):
            CustomerRuntime(profile).apply(world, CustomerDecision(message=message,
                accept_quote_id=service.quote, accept_operation="commit_accepted_quote"),
                presentation=presentation, event_index=11)
    accepted = CustomerRuntime(profile).apply(world, CustomerDecision(
        message="I accept this exact quote.", accept_quote_id=service.quote,
        accept_operation="commit_accepted_quote"), presentation=presentation, event_index=11)
    assert len(accepted.world.consents) == 1


def test_transaction_task_cannot_smuggle_an_extra_delegated_line():
    world, aliases = frozen()
    task = refund_task(world, aliases)
    accessory = aliases["accessory"]
    request = "Please refund my damaged device and damaged cable."
    goal = task.remedy_goals[0].model_copy(update={
        "public_request_span": "Please refund my damaged device"})
    task = TaskSpecV2.model_validate(task.model_dump(mode="json", exclude={"record_hash"}) | {
        "customer_request": request, "remedy_goals": [goal.model_dump(mode="json", exclude={"record_hash"})]})
    profile = CustomerProfileV2(customer_id=aliases["customer"],
        delegated_line_ids=(aliases["damaged"], accessory),
        delegation_request_spans={aliases["damaged"]: "Please refund my damaged device",
                                  accessory: "Please refund my damaged cable"})
    with pytest.raises(ValueError, match="delegations must match exactly"):
        run_episode(world, task, profile, FinishCustomer(), ReadOnlyService(world, aliases["damaged"]),
                    budget=EpisodeBudget(max_model_calls=0))


def test_read_only_request_is_scored_from_delivered_fact_and_replayed(tmp_path):
    world, aliases = frozen()
    line = aliases["damaged"]
    text = "Please tell me the quantity for damaged line " + line + "."
    task = TaskSpec(
        task_id="read-quantity", world_hash=world.state_hash, customer_id=aliases["customer"],
        customer_request=text, information_goals=(InformationObjective(
            goal_id="quantity", subject_id=line, subject_mention="damaged line",
            fact_kind="quantity", public_request="quantity for damaged line"),),
        partition="train", source_distribution_hash=digest("P4 communication"))
    profile = CustomerProfile(customer_id=aliases["customer"])
    record = run_episode(world, task, profile, FinishCustomer(), ReadOnlyService(world, line),
                         budget=EpisodeBudget(max_customer_turns=2, max_service_decisions=4,
                                              max_service_tool_calls=2, max_model_calls=0))
    assert record.outcome.strict_task_success is True
    assert record.communication.fulfilled == 1
    assert record.replay_verified and replay_episode(record).verified
    artifact = tmp_path / "private" / "episode.json"
    save_episode(record, artifact)
    assert artifact.stat().st_mode & 0o077 == 0
    assert verify_episode_artifact(artifact)["verified"]
    with pytest.raises(FileExistsError):
        save_episode(record, artifact)


def test_replay_rejects_modified_event_and_model_context_is_not_service_visible():
    world, aliases = frozen()
    task = refund_task(world, aliases)
    profile = refund_profile(aliases)
    record = run_episode(world, task, profile, ScriptedCustomer(), ScriptedRefundService(world, aliases),
                         budget=EpisodeBudget(max_customer_turns=3, max_service_decisions=10,
                                              max_service_tool_calls=8, max_model_calls=0))
    raw = record.model_dump(mode="json")
    event = next(row for row in raw["events"] if row["kind"] == "service_reply")
    event["payload"]["message"]["text"] = "tampered reply"
    with pytest.raises(ValueError, match="content hash differs"):
        type(record).model_validate(raw)


def test_runner_rejects_duplicate_frozen_external_event_ids_before_execution():
    world, aliases = frozen()
    task = refund_task(world, aliases)
    profile = refund_profile(aliases)
    event = ExternalEvent(id="duplicate-event", kind="stock_adjustment",
        target_id=next(iter(world.inventory)), at=world.clock + timedelta(days=1), integer_value=1)
    with pytest.raises(ValueError, match="schedule repeats an event ID"):
        run_episode(world, task, profile, FinishCustomer(), ReadOnlyService(world, aliases["damaged"]),
                    external_events=(event, event))


def test_explicit_openai_adapters_capture_request_and_raw_output_without_network():
    from usami.service.models import ModelSettings, ServiceError

    class RecordedClient:
        settings = ModelSettings(model="offline-fixture", base_url="https://example.invalid/v1")

        def __init__(self, response):
            self.response = response
            self.requests = []

        def complete(self, request):
            self.requests.append(request)
            return self.response

    def response(tool_name, arguments, finish_reason="tool_calls"):
        return {"choices": [{"finish_reason": finish_reason, "message": {
            "role": "assistant", "content": None, "tool_calls": [{
                "id": "call-offline", "type": "function", "function": {
                    "name": tool_name, "arguments": json.dumps(arguments)}}]}}],
            "usage": {"prompt_tokens": 13, "completion_tokens": 7}}

    customer_args = {"message": "Please help with my order.", "delegate_line_ids": ["line-own"],
        "allowed_preparations": ["open_remedy_case"], "accept_quote_id": None,
        "accept_operation": None, "reject_quote_id": None, "withdraw_delegation_ids": [],
        "disclose_fact_ids": [], "finish": False}
    customer_client = RecordedClient(response("customer_turn", customer_args))
    profile = CustomerProfile(customer_id="customer-own", delegated_line_ids=("line-own",))
    decision = OpenAICustomerAgent(customer_client).decide(
        CustomerPrompt(public_request="Help with line-own."), profile)
    assert decision.delegate_line_ids == ("line-own",)
    assert customer_client.requests[0]["tool_choice"]["function"]["name"] == "customer_turn"
    assert customer_client.requests[0]["stream"] is False

    service_args = {"message": "I will check the request.", "present_quote_id": None,
        "present_for": "commit_accepted_quote", "subject_ids": ["line-own"]}
    service_client = RecordedClient(response("respond_to_customer", service_args))
    service = OpenAIServiceAgent(service_client)
    service_prompt = ServicePrompt(customer_id="customer-own",
        customer_request="Help with line-own.", available_merchant_scope_ids=("merchant-own",),
        conversation=(AgentMessage(role="user", content="Help with line-own."),),
        observed_subject_ids=("customer-own", "line-own"))
    action = service.decide(service_prompt)
    assert action.kind == "reply" and action.subject_ids == ("line-own",)
    assert action.call.request == service_client.requests[0]
    assert service_client.requests[0]["tool_choice"] == "auto"
    assert service_client.requests[0]["parallel_tool_calls"] is False
    prompt_messages = service_client.requests[0]["messages"]
    assert len(prompt_messages) == 2  # the initial request is not duplicated
    operational_context = json.loads(prompt_messages[1]["content"])
    assert operational_context == {
        "authenticated_customer_id": "customer-own",
        "available_merchant_scope_ids": ["merchant-own"],
        "customer_request": "Help with line-own.",
    }
    assert "PRIVATE CUSTOMER PROFILE" not in json.dumps(service_client.requests[0])
    available_tools = {row["function"]["name"]: row["function"] for row in service_tools()}
    assert "merchant ID" in available_tools["list_customer_orders"]["description"]
    assert "evidence references" in available_tools["get_case_status"]["description"]
    assert "case ID" in available_tools["prepare_remedy_quote"]["description"]
    assert "exact customer consent" in available_tools["commit_accepted_quote"]["description"]

    # Some compatible providers choose a normal final response under tool_choice=auto.
    # It is a safe, side-effect-free Customer reply rather than a malformed operation.
    text_only = {"choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": "I will check your order."}}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 4}}
    text_client = RecordedClient(text_only)
    text_action = OpenAIServiceAgent(text_client).decide(service_prompt)
    assert text_action.kind == "reply"
    assert text_action.message == "I will check your order."
    assert text_action.call.response == text_only
    assert text_action.operation is None

    malformed = response("respond_to_customer", service_args, finish_reason="length")
    broken_service = OpenAIServiceAgent(RecordedClient(malformed))
    with pytest.raises(ServiceError, match="complete structured function call"):
        broken_service.decide(ServicePrompt(customer_id="customer-own", customer_request="Help."))
    assert broken_service.last_call.response == malformed
