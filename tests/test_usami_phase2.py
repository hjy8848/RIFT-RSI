"""Phase 2 tests: partial, multi-line, multi-order and multi-goal atomic refunds.

Every test exercises the real Engine + Reference + Oracle + Replay path; nothing
is mocked.  The only scripted parts are the Service and Customer agents that a
model would normally play.
"""
import json

import pytest

from usami_harness.engine import Engine, replay_trace
from usami_harness.models import Request, digest
from usami_harness.oracle import evaluate
from usami_v2.commerce import adapter
from usami_v2.commerce.invariants import invariants
from usami_v2.commerce.models import Operation, unit_credit
from usami_v2.commerce.reference import decide
from usami_v2.runtime.agents import ServiceAction
from usami_v2.runtime.customer import (
    CustomerDecision,
    CustomerProfile,
    CustomerProfileV2,
    CustomerRuntime,
    bind_customer_profile,
)
from usami_v2.runtime.runner import (
    EpisodeBudget,
    EpisodeRecord,
    replay_episode,
    run_episode,
)
from usami_v2.scenarios.builder import build_world
from usami_v2.scenarios.schema import (
    Attribute,
    BuildBudget,
    IntegerRange,
    LineVariable,
    OrderVariable,
    ScenarioSpec,
    SKUVariable,
    WarehouseVariable,
)
from usami_v2.scenarios.task_certification import (
    CustomerScenario,
    ProofBudget,
    certification_cost,
    certify_task,
)
from usami_v2.scenarios.tasks import (
    CustomerRequestBundle,
    GroundedRemedyGoal,
    RemedyGoal,
    TaskSpecV2,
    TaskSpecV3,
    evaluate_state_goals,
    evaluate_task_compliance,
    render_customer_request,
    request_intent_hash,
)


def _fixed(value):
    return IntegerRange(low=value, high=value)


FAST = WarehouseVariable(alias="fast", delivery_days=_fixed(2), delivery_fee_cents=_fixed(0))


def _sku(alias, family, price_cents, **attributes):
    return SKUVariable(
        alias=alias,
        family=family,
        price_cents=_fixed(price_cents),
        attributes=tuple(Attribute(name=name, value=value) for name, value in attributes.items()),
    )


def _build(spec, seed):
    result = build_world(spec, seed, BuildBudget(max_nodes=256))
    assert result.status == "SAT", result.reasons
    return result.world, dict(result.private_aliases)


def multiline_world(seed=41):
    """One order with three lines: two laptop units, one mouse, two cables."""
    spec = ScenarioSpec(
        scenario_id="phase2-multiline",
        provenance_kind="offline_fixture",
        customer_aliases=("customer",),
        skus=(
            _sku("laptop", "laptop", 8000, category="electronics", product="laptop", interface="USB_C"),
            _sku("mouse", "mouse", 1200, category="electronics", product="mouse", interface="USB_C"),
            _sku("cable", "cable", 1000, category="accessories", product="cable", interface="USB_C"),
        ),
        warehouses=(FAST,),
        orders=(
            OrderVariable(
                alias="purchase",
                customer_alias="customer",
                delivered_days_ago=_fixed(1),
                lines=(
                    LineVariable(alias="laptop_line", original_sku="laptop",
                                 quantity=_fixed(2), paid_cents=_fixed(16000)),
                    LineVariable(alias="mouse_line", original_sku="mouse",
                                 quantity=_fixed(1), paid_cents=_fixed(1200)),
                    LineVariable(alias="cable_line", original_sku="cable",
                                 quantity=_fixed(2), paid_cents=_fixed(2000)),
                ),
            ),
        ),
        default_stock=_fixed(0),
    )
    return _build(spec, seed)


def multiorder_world(seed=43):
    """Two orders for the same customer plus one order owned by a stranger."""
    spec = ScenarioSpec(
        scenario_id="phase2-multiorder",
        provenance_kind="offline_fixture",
        customer_aliases=("customer", "stranger"),
        skus=(
            _sku("laptop", "laptop", 8000, category="electronics", product="laptop", interface="USB_C"),
            _sku("mouse", "mouse", 1200, category="electronics", product="mouse", interface="USB_C"),
            _sku("cable", "cable", 1000, category="accessories", product="cable", interface="USB_C"),
        ),
        warehouses=(FAST,),
        orders=(
            OrderVariable(
                alias="first", customer_alias="customer", delivered_days_ago=_fixed(1),
                lines=(LineVariable(alias="first_laptop", original_sku="laptop",
                                    quantity=_fixed(1), paid_cents=_fixed(8000)),
                       LineVariable(alias="first_cable", original_sku="cable",
                                    quantity=_fixed(2), paid_cents=_fixed(2000))),
            ),
            OrderVariable(
                alias="second", customer_alias="customer", delivered_days_ago=_fixed(2),
                lines=(LineVariable(alias="second_mouse", original_sku="mouse",
                                    quantity=_fixed(1), paid_cents=_fixed(1200)),),
            ),
            OrderVariable(
                alias="foreign", customer_alias="stranger", delivered_days_ago=_fixed(1),
                lines=(LineVariable(alias="foreign_laptop", original_sku="laptop",
                                    quantity=_fixed(1), paid_cents=_fixed(8000)),),
            ),
        ),
        default_stock=_fixed(0),
    )
    return _build(spec, seed)


TWO_ORDER_REQUEST = (
    "Please refund the damaged laptop from my first order and refund the damaged mouse from my second order."
)


def _goal(goal_id, line_id, quantity, span, mention):
    return GroundedRemedyGoal(
        goal_id=goal_id,
        line_id=line_id,
        quantity=quantity,
        public_request_span=span,
        subject_mention=mention,
        allowed_remedies=("refund",),
        maximum_extra_charge_cents=0,
    )


def _task(world, goals, request):
    return TaskSpecV2(
        task_id="phase2-task",
        world_hash=world.state_hash,
        customer_id=next(row.id for row in world.customers.values() if row.id != "stranger"),
        customer_request=request,
        remedy_goals=tuple(goals),
        partition="train",
        source_distribution_hash=digest("phase2 regression"),
    )


def _v3_task(world, goals, request=None):
    """Structured-intent task; the public request is rendered from the goals."""
    goals = tuple(goals)
    customer_id = next(row.id for row in world.customers.values() if row.id != "stranger")
    if request is None:
        request = render_customer_request(world, customer_id, goals, ())
    bundle = CustomerRequestBundle(
        customer_id=customer_id,
        goal_ids=tuple(goal.goal_id for goal in goals),
        intent_hash=request_intent_hash(customer_id, goals, ()),
        request_hash=digest(request),
    )
    return TaskSpecV3(
        task_id="phase2-task-v3",
        world_hash=world.state_hash,
        customer_id=customer_id,
        customer_request=request,
        remedy_goals=goals,
        partition="train",
        source_distribution_hash=digest("phase2 regression"),
        request_bundle=bundle,
    )


def _profile(task, line_ids, request, *, spans=None):
    line_ids = tuple(line_ids)
    resolved = dict(spans or {})
    for goal in task.remedy_goals:
        resolved.setdefault(goal.line_id, goal.public_request_span)
    for line_id in line_ids:
        resolved.setdefault(line_id, request)
    return CustomerProfileV2(
        customer_id=task.customer_id,
        delegated_line_ids=line_ids,
        delegation_request_spans={line_id: resolved[line_id] for line_id in line_ids},
    )


def _delegated(world, task, profile):
    return CustomerRuntime(profile, task=task).initialize(world, task.customer_request).world


def _refund(world, actor, line_id, quantity=1):
    operation = Operation(
        name="return_delivered_order_items",
        scope_id=next(iter(world.merchants)),
        resource_id=line_id,
        params={"quantity": quantity, "payment_method": "original"},
    )
    request = Request(id="phase2-refund-1", identity=actor,
                      operation=operation.model_dump(mode="json"), granted=True)
    engine = Engine(adapter, world, operation_budget=1)
    engine.register(request)
    event = engine.execute(actor, operation, request.id, mode="reference")
    trace = engine.finish()
    oracle = evaluate(adapter, trace, {request.id: request})
    replayed = replay_trace(adapter, trace, {request.id: request})
    assert replayed.state == engine.state
    return engine.state, event.payload["result"], oracle


# ---------------------------------------------------------------------------
# P2.2 同订单多商品
# ---------------------------------------------------------------------------

def test_multiline_refund_touches_only_the_selected_line():
    world, aliases = multiline_world()
    customer = aliases["customer"]
    task = _task(world, [_goal("mouse-goal", aliases["mouse_line"], 1,
                               "refund the damaged mouse", "damaged mouse")], "Please refund the damaged mouse.")
    profile = _profile(task, [aliases["mouse_line"]], "Please refund the damaged mouse.")
    world = _delegated(world, task, profile)

    after, result, oracle = _refund(world, customer, aliases["mouse_line"], 1)

    assert result["success"] and result["data"]["refund_cents"] == 1200
    assert oracle["status"] == "pass" and not invariants(after)
    added = [row for key, row in after.settlements.items() if key not in world.settlements]
    assert len(added) == 1 and added[0].line_id == aliases["mouse_line"]
    assert after.lines[aliases["laptop_line"]] == world.lines[aliases["laptop_line"]]
    assert after.lines[aliases["cable_line"]] == world.lines[aliases["cable_line"]]
    assert not [row for row in after.cases.values() if row.line_id != aliases["mouse_line"]]


def test_multiline_two_of_three_units_can_be_refunded_in_two_calls():
    world, aliases = multiline_world()
    customer = aliases["customer"]
    task = _task(world, [_goal("cable-goal", aliases["cable_line"], 2,
                               "refund the damaged cable", "damaged cable")], "Please refund the damaged cable.")
    profile = _profile(task, [aliases["cable_line"]], "Please refund the damaged cable.")
    world = _delegated(world, task, profile)

    first, first_result, _ = _refund(world, customer, aliases["cable_line"], 1)
    second, second_result, oracle = _refund(first, customer, aliases["cable_line"], 1)

    assert first_result["data"]["refund_cents"] == 1000
    assert second_result["data"]["refund_cents"] == 1000
    assert oracle["status"] == "pass" and not invariants(second)
    units = set()
    total = 0
    for key, row in second.settlements.items():
        if key in world.settlements:
            continue
        assert row.line_id == aliases["cable_line"]
        assert not units.intersection(row.unit_indices)
        units.update(row.unit_indices)
        refunds = [entry for entry in second.ledger.values()
                   if entry.quote_id == row.quote_id and entry.kind == "refund"]
        total += sum(entry.amount_cents for entry in refunds)
    assert units == {0, 1} and total == 2000


def test_multiline_quantity_above_line_units_is_rejected():
    world, aliases = multiline_world()
    customer = aliases["customer"]
    task = _task(world, [_goal("laptop-goal", aliases["laptop_line"], 2,
                               "refund the damaged laptop", "damaged laptop")],
                 "Please refund the damaged laptop.")
    profile = _profile(task, [aliases["laptop_line"]], "Please refund the damaged laptop.")
    world = _delegated(world, task, profile)

    after, result, oracle = _refund(world, customer, aliases["laptop_line"], 3)

    assert not result["success"] and result["error_code"] == "PURCHASED_QUANTITY_UNAVAILABLE"
    assert after == world and oracle["status"] == "pass"


def test_public_sku_attributes_disambiguate_two_devices():
    world, aliases = multiline_world()
    order = world.orders[aliases["purchase"]]
    lines = [world.lines[aliases["laptop_line"]], world.lines[aliases["mouse_line"]]]
    descriptors = [(line, world.skus[line.sku_id]) for line in lines]
    device_lines = [(line, sku) for line, sku in descriptors
                    if sku.product_family == "device" or sku.attributes.get("category") == "electronics"]
    assert len(device_lines) == 2  # ambiguous by category alone

    mouse = next(line for line, sku in device_lines if sku.attributes.get("product") == "mouse")
    assert mouse.id == aliases["mouse_line"]
    assert all(line.order_id == order.id for line, _ in device_lines)


def test_refunding_the_wrong_item_does_not_satisfy_the_goal():
    world, aliases = multiline_world()
    customer = aliases["customer"]
    request = "Please refund the damaged laptop and refund the damaged mouse."
    task = _task(world, [_goal("mouse-goal", aliases["mouse_line"], 1,
                               "refund the damaged mouse", "damaged mouse")], request)
    profile = _profile(task, [aliases["laptop_line"], aliases["mouse_line"]], request,
                       spans={aliases["laptop_line"]: "refund the damaged laptop"})
    base, world = world, _delegated(world, task, profile)
    initial = base

    after, result, _ = _refund(world, customer, aliases["laptop_line"], 1)

    assert result["success"]
    goals = evaluate_state_goals(initial, after, task)
    assert [(row.goal_id, row.status) for row in goals] == [("mouse-goal", "FAIL")]
    compliance = evaluate_task_compliance(initial, after, task)
    assert [(row.goal_id, row.status) for row in compliance] == [("unrequested_settlements", "FAIL")]


def test_goal_quantity_must_match_the_refunded_units():
    world, aliases = multiline_world()
    customer = aliases["customer"]
    task = _task(world, [_goal("laptop-goal", aliases["laptop_line"], 1,
                               "refund the damaged laptop", "damaged laptop")],
                 "Please refund the damaged laptop.")
    profile = _profile(task, [aliases["laptop_line"]], "Please refund the damaged laptop.")
    base, world = world, _delegated(world, task, profile)
    initial = base

    after, result, _ = _refund(world, customer, aliases["laptop_line"], 2)

    assert result["success"]
    goals = evaluate_state_goals(initial, after, task)
    assert [(row.goal_id, row.status) for row in goals] == [("laptop-goal", "FAIL")]


# ---------------------------------------------------------------------------
# P2.3 多订单组合
# ---------------------------------------------------------------------------

def _two_goal_task(world, aliases):
    return _task(
        world,
        [
            _goal("laptop-goal", aliases["first_laptop"], 1,
                  "refund the damaged laptop from my first order", "damaged laptop"),
            _goal("mouse-goal", aliases["second_mouse"], 1,
                  "refund the damaged mouse from my second order", "mouse"),
        ],
        TWO_ORDER_REQUEST,
    )


def test_multiorder_refunds_are_independent_and_both_goals_pass():
    world, aliases = multiorder_world()
    task = _two_goal_task(world, aliases)
    profile = _profile(task, [aliases["first_laptop"], aliases["second_mouse"]],
                       TWO_ORDER_REQUEST)
    base, world = world, _delegated(world, task, profile)
    initial = base

    after_first, first_result, _ = _refund(world, task.customer_id, aliases["first_laptop"], 1)
    after_second, second_result, oracle = _refund(after_first, task.customer_id,
                                                  aliases["second_mouse"], 1)

    assert first_result["data"]["refund_cents"] == 8000
    assert second_result["data"]["refund_cents"] == 1200
    assert oracle["status"] == "pass" and not invariants(after_second)
    goals = evaluate_state_goals(initial, after_second, task)
    assert [(row.goal_id, row.status) for row in goals] == [
        ("laptop-goal", "PASS"), ("mouse-goal", "PASS")]
    assert [row.status for row in evaluate_task_compliance(initial, after_second, task)] == ["PASS"]


def test_partial_multiorder_completion_fails_the_task():
    world, aliases = multiorder_world()
    task = _two_goal_task(world, aliases)
    profile = _profile(task, [aliases["first_laptop"], aliases["second_mouse"]],
                       TWO_ORDER_REQUEST)
    base, world = world, _delegated(world, task, profile)
    initial = base

    after, result, _ = _refund(world, task.customer_id, aliases["first_laptop"], 1)

    assert result["success"]
    goals = evaluate_state_goals(initial, after, task)
    assert [(row.goal_id, row.status) for row in goals] == [
        ("laptop-goal", "PASS"), ("mouse-goal", "FAIL")]


def test_a_line_from_another_customer_is_rejected():
    world, aliases = multiorder_world()
    task = _two_goal_task(world, aliases)
    profile = _profile(task, [aliases["first_laptop"], aliases["second_mouse"]],
                       TWO_ORDER_REQUEST)
    world = _delegated(world, task, profile)

    after, result, oracle = _refund(world, task.customer_id, aliases["foreign_laptop"], 1)

    assert not result["success"] and result["error_code"] == "OBJECT_ACCESS_DENIED"
    assert after == world and oracle["status"] == "pass"


def test_an_order_id_cannot_be_used_as_a_line_id():
    world, aliases = multiorder_world()
    task = _two_goal_task(world, aliases)
    profile = _profile(task, [aliases["first_laptop"], aliases["second_mouse"]],
                       TWO_ORDER_REQUEST)
    world = _delegated(world, task, profile)

    after, result, _ = _refund(world, task.customer_id, aliases["first"], 1)

    assert not result["success"] and result["error_code"] in (
        "RESOURCE_NOT_FOUND", "OBJECT_ACCESS_DENIED")
    assert after == world


# ---------------------------------------------------------------------------
# P2.4 多目标 Certification
# ---------------------------------------------------------------------------

def _scenario(task, line_ids):
    return CustomerScenario(
        customer_id=task.customer_id,
        delegated_line_ids=tuple(line_ids),
        accepted_remedies=("refund",),
    )


def test_multigoal_certification_witnesses_every_goal_via_public_tools():
    world, aliases = multiorder_world()
    task = _two_goal_task(world, aliases)
    scenario = _scenario(task, [aliases["first_laptop"], aliases["second_mouse"]])

    certificate = certify_task(world, task, scenario, ProofBudget())

    assert certificate.status == "CERTIFIED", certificate.blockers
    assert certificate.full_task_reachability_certified
    witness = next(row for row in certificate.attempts if row.status == "WITNESS")
    assert witness.goal_results == (
        ("laptop-goal", "PASS"), ("mouse-goal", "PASS"))
    operations = [step.evidence["request"]["operation"]["name"] for step in witness.steps
                  if step.kind == "operation"]
    assert operations == ["list_customer_orders", "get_order_details",
                          "return_delivered_order_items", "list_customer_orders",
                          "get_order_details", "return_delivered_order_items"]
    # one confirmation exchange per refund, plus one closing exchange
    assert certification_cost(certificate) == {
        "tool_calls": 6, "service_decisions": 9, "customer_turns": 3, "model_calls": 12}


def test_multigoal_certification_fails_when_one_goal_is_unreachable():
    world, aliases = multiorder_world()
    sku = world.skus[aliases["mouse"]]
    modified = world.clone()
    modified.skus[sku.id] = sku.model_copy(update={"return_required": True})
    task = _two_goal_task(modified, aliases)
    scenario = _scenario(task, [aliases["first_laptop"], aliases["second_mouse"]])

    certificate = certify_task(modified, task, scenario, ProofBudget())

    assert certificate.status == "UNKNOWN"
    assert not certificate.full_task_reachability_certified
    assert "physical_return_path_not_migrated_in_phase_1" in certificate.blockers


def test_exhausted_proof_budget_is_unknown_not_unsat():
    world, aliases = multiorder_world()
    task = _two_goal_task(world, aliases)
    scenario = _scenario(task, [aliases["first_laptop"], aliases["second_mouse"]])

    certificate = certify_task(world, task, scenario, ProofBudget(max_operation_steps=1))

    assert certificate.status == "UNKNOWN"
    assert not certificate.full_task_reachability_certified
    assert "proof_operation_budget_exhausted" in certificate.blockers


# ---------------------------------------------------------------------------
# P2.3 运行时：两个订单、两次确认、两次退款
# ---------------------------------------------------------------------------

class MultiOrderService:
    """Scripted Service: list -> details -> confirm/refund for each order."""

    uses_model = False
    last_call = None

    def __init__(self):
        self.index = 0
        self.scope = None
        self.orders = []
        self.lines = {}

    @staticmethod
    def _last_result(prompt):
        for message in reversed(prompt.conversation):
            if message.role == "tool" and message.content:
                return json.loads(message.content)["data"]
        raise AssertionError("public Service observed no tool result")

    def _call(self, name, resource, params=None):
        return ServiceAction(kind="operation", operation=Operation(
            name=name, scope_id=self.scope, resource_id=resource, params=params or {}))

    def decide(self, prompt):
        self.scope = self.scope or prompt.available_merchant_scope_ids[0]
        if self.index == 0:
            self.index = 1
            return self._call("list_customer_orders", self.scope)
        if self.index == 1:
            self.orders = self._last_result(prompt)["orders"]
            self.current_order = self.orders[0]["id"]
            self.index = 2
            return self._call("get_order_details", self.current_order)
        if self.index == 2:
            self.lines["laptop"] = self._last_result(prompt)["lines"]
            self.index = 3
            return ServiceAction(kind="reply",
                                 message="I can refund one damaged laptop (8000) to the original "
                                         "card. Shall I proceed?")
        if self.index == 3:
            line = next(row for row in self.lines["laptop"]
                        if row["sku"]["attributes"].get("product") == "laptop")
            self.index = 4
            return self._call("return_delivered_order_items", line["id"],
                              {"quantity": 1, "payment_method": "original"})
        if self.index == 4:
            other = next(order for order in self.orders
                         if order["id"] != self.current_order)
            self.index = 5
            return self._call("get_order_details", other["id"])
        if self.index == 5:
            self.lines["mouse"] = self._last_result(prompt)["lines"]
            self.index = 6
            return ServiceAction(kind="reply",
                                 message="I can refund one mouse (1200) to the original card. "
                                         "Shall I proceed?")
        if self.index == 6:
            line = next(row for row in self.lines["mouse"]
                        if row["sku"]["attributes"].get("product") == "mouse")
            self.index = 7
            return self._call("return_delivered_order_items", line["id"],
                              {"quantity": 1, "payment_method": "original"})
        self.index += 1
        return ServiceAction(kind="reply",
                             message="Both refunds are issued to the original card.")


class TwoTurnConfirmingCustomer:
    """Agrees twice, then finishes after the last acknowledgement."""

    uses_model = False
    last_call = None

    def __init__(self):
        self.turns = 0

    def decide(self, prompt, profile):
        self.turns += 1
        if self.turns == 1:
            return CustomerDecision(message="Yes, please refund the laptop.")
        if self.turns == 2:
            return CustomerDecision(message="Yes, refund the mouse too.")
        return CustomerDecision(message="Thanks, that is all.", finish=True)


def test_two_order_episode_refunds_both_goals_and_replays():
    world, aliases = multiorder_world()
    task = _two_goal_task(world, aliases)
    profile = _profile(task, [aliases["first_laptop"], aliases["second_mouse"]],
                       TWO_ORDER_REQUEST)

    record = run_episode(world, task, profile, TwoTurnConfirmingCustomer(),
                         MultiOrderService(),
                         budget=EpisodeBudget(max_customer_turns=3, max_service_decisions=12,
                                              max_service_tool_calls=12, max_model_calls=12))

    assert record.status == "customer_finished", record.terminal_reason
    assert record.final_world != world
    assert len([key for key in record.final_world.settlements
                if key not in world.settlements]) == 2
    goals = evaluate_state_goals(record.initial_world, record.final_world, record.task)
    assert [(row.goal_id, row.status) for row in goals] == [
        ("laptop-goal", "PASS"), ("mouse-goal", "PASS")]
    replay = replay_episode(record)
    assert replay.verified, replay.errors
    # list, details, refund, details, refund
    assert replay.operation_count == 5


def test_a_customer_refusal_prevents_the_refund():
    class AskingService:
        uses_model = False
        last_call = None

        def __init__(self):
            self.index = 0

        def decide(self, prompt):
            if self.index == 0:
                self.index = 1
                return ServiceAction(kind="reply",
                                     message="I can refund the damaged laptop. Shall I proceed?")
            if self.index == 1:
                self.index = 2
                last_user = next((row.content for row in reversed(prompt.conversation)
                                  if row.role == "user" and row.content), "")
                if last_user.lower().startswith("no"):
                    return ServiceAction(kind="reply", message="Understood, no refund.")
                return ServiceAction(kind="operation", operation=Operation(
                    name="return_delivered_order_items",
                    scope_id=prompt.available_merchant_scope_ids[0],
                    resource_id=prompt.observed_subject_ids[0],
                    params={"quantity": 1, "payment_method": "original"}))
            return ServiceAction(kind="reply", message="Done.")

    class RefusingCustomer:
        uses_model = False
        last_call = None

        def __init__(self):
            self.turns = 0

        def decide(self, prompt, profile):
            self.turns += 1
            if self.turns == 1:
                return CustomerDecision(message="No, please do not refund anything.")
            return CustomerDecision(message="Thanks.", finish=True)

    world, aliases = multiline_world()
    task = _task(world, [_goal("mouse-goal", aliases["mouse_line"], 1,
                               "refund the damaged mouse", "damaged mouse")], "Please refund the damaged mouse.")
    profile = _profile(task, [aliases["mouse_line"]], "Please refund the damaged mouse.")

    record = run_episode(world, task, profile, RefusingCustomer(), AskingService(),
                         budget=EpisodeBudget(max_customer_turns=3, max_service_decisions=6,
                                              max_service_tool_calls=4, max_model_calls=4))

    assert record.status == "customer_finished", record.terminal_reason
    assert record.final_world.settlements == world.settlements
    assert record.final_world.cases == world.cases


def test_tampered_episode_trace_is_rejected_by_replay():
    world, aliases = multiorder_world()
    task = _two_goal_task(world, aliases)
    profile = _profile(task, [aliases["first_laptop"], aliases["second_mouse"]],
                       TWO_ORDER_REQUEST)
    record = run_episode(world, task, profile, TwoTurnConfirmingCustomer(),
                         MultiOrderService(),
                         budget=EpisodeBudget(max_customer_turns=3, max_service_decisions=12,
                                              max_service_tool_calls=12, max_model_calls=12))
    assert replay_episode(record).verified

    payload = record.model_dump(mode="json")
    payload["service_messages"][0]["text"] += " (tampered)"
    payload["service_messages"][0]["record_hash"] = ""
    tampered = EpisodeRecord.model_validate({**payload, "record_hash": ""})

    replay = replay_episode(tampered)
    assert not replay.verified
    assert replay.errors


def test_v3_multigoal_task_certifies_and_replays_end_to_end():
    world, aliases = multiorder_world()
    goals = [
        RemedyGoal(goal_id="laptop-goal", line_id=aliases["first_laptop"], quantity=1,
                   allowed_remedies=("refund",)),
        RemedyGoal(goal_id="mouse-goal", line_id=aliases["second_mouse"], quantity=1,
                   allowed_remedies=("refund",)),
    ]
    task = _v3_task(world, goals)
    scenario = _scenario(task, [aliases["first_laptop"], aliases["second_mouse"]])

    certificate = certify_task(world, task, scenario, ProofBudget())

    assert certificate.status == "CERTIFIED", certificate.blockers
    assert certification_cost(certificate) == {
        "tool_calls": 6, "service_decisions": 9, "customer_turns": 3, "model_calls": 12}

    profile = bind_customer_profile(CustomerProfile(customer_id=task.customer_id), task)
    record = run_episode(world, task, profile, TwoTurnConfirmingCustomer(), MultiOrderService(),
                         budget=EpisodeBudget(max_customer_turns=4, max_service_decisions=12,
                                              max_service_tool_calls=12, max_model_calls=12))

    assert record.status == "customer_finished", record.terminal_reason
    assert record.outcome.business_goal_success is True
    goals_out = evaluate_state_goals(record.initial_world, record.final_world, record.task)
    assert [(row.goal_id, row.status) for row in goals_out] == [
        ("laptop-goal", "PASS"), ("mouse-goal", "PASS")]
    assert replay_episode(record).verified
