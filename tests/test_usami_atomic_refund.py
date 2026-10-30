"""CPU-only acceptance tests for the Phase 1 atomic refund tool.

These tests exercise the real Engine + Reference + Oracle path; nothing is mocked.
"""
import json

import pytest

from rift_usami_fixtures import (
    ScriptedCustomer, customer_scenario, frozen, frozen_world, make_task, refund_profile,
    refund_task,
)
from usami_v2.commerce import adapter
from usami_v2.commerce.invariants import invariants
from usami_v2.commerce.models import Operation
from usami_v2.commerce.reference import decide
from usami_v2.runtime.agents import ServiceAction, service_tools
from usami_v2.runtime.customer import CustomerRuntime
from usami_v2.runtime.runner import EpisodeBudget, replay_episode, run_episode
from usami_harness.engine import Engine
from usami_harness.models import Request
from usami_harness.oracle import evaluate


def _delegated_world(seed=17):
    world, aliases = frozen(seed)
    task = refund_task(world, aliases)
    profile = refund_profile(aliases)
    transition = CustomerRuntime(profile, task=task).initialize(world, task.customer_request)
    return transition.world, aliases, task, profile


def _run_operation(world, actor, line_id, params):
    operation = Operation(name="return_delivered_order_items",
                          scope_id=next(iter(world.merchants)),
                          resource_id=line_id, params=params)
    request = Request(id="atomic-refund-1", identity=actor,
                      operation=operation.model_dump(mode="json"), granted=True)
    engine = Engine(adapter, world, operation_budget=1)
    engine.register(request)
    event = engine.execute(actor, operation, request.id, mode="reference")
    trace = engine.finish()
    oracle = evaluate(adapter, trace, {request.id: request})
    return engine.state, event.payload["result"], oracle


class AtomicRefundService:
    """Scripted Service using the public tool surface only (no private aliases)."""

    uses_model = False
    last_call = None

    def __init__(self, line_id):
        self.line_id, self.index = line_id, 0

    def decide(self, prompt):
        actions = [
            ServiceAction(kind="operation", operation=Operation(
                name="return_delivered_order_items",
                scope_id=next(iter(prompt.available_merchant_scope_ids)),
                resource_id=self.line_id,
                params={"quantity": 1, "payment_method": "original"})),
            ServiceAction(kind="reply",
                          message="Your return is accepted and the refund is issued to the original card."),
        ]
        action = actions[self.index]
        self.index += 1
        return action


def test_atomic_refund_episode_changes_state_and_replays():
    world, aliases = frozen(17)
    task = refund_task(world, aliases)
    profile = refund_profile(aliases)
    episode = run_episode(world, task, profile, ScriptedCustomer(),
                          AtomicRefundService(aliases["damaged"]),
                          budget=EpisodeBudget(max_customer_turns=3, max_service_decisions=10,
                                               max_service_tool_calls=8, max_model_calls=0))
    replay = replay_episode(episode)
    assert replay.verified
    assert episode.outcome.business_goal_success is True
    assert invariants(episode.final_world) == []
    settlements = list(episode.final_world.settlements.values())
    assert len(settlements) == 1 and settlements[0].remedy == "refund"
    case = episode.final_world.cases[settlements[0].case_id]
    assert case.evidence_id is None                 # no fabricated Evidence
    assert case.status == "resolved"
    ledger = list(episode.final_world.ledger.values())
    assert [row.kind for row in ledger] == ["refund"]
    assert ledger[0].amount_cents == 20000
    assert not any(row.line_id == case.line_id and row.customer_id != profile.customer_id
                   for row in episode.final_world.evidence.values())


def test_reference_and_engine_agree_on_atomic_refund():
    world, aliases, task, profile = _delegated_world()
    line_id = aliases["damaged"]
    operation = Operation(name="return_delivered_order_items",
                          scope_id=next(iter(world.merchants)), resource_id=line_id,
                          params={"quantity": 1, "payment_method": "original"})
    decision = decide(world, profile.customer_id, operation)
    assert decision.allowed, decision.error_code
    assert invariants(decision.expected_state) == []
    after, result, oracle = _run_operation(world, profile.customer_id, line_id,
                                           {"quantity": 1, "payment_method": "original"})
    assert result["success"] and result["data"] == decision.data
    assert after == decision.expected_state
    assert oracle.get("status") == "pass"


def test_return_window_expired_is_rejected():
    world, aliases, task, profile = _delegated_world()
    line_id = aliases["damaged"]
    line = world.lines[line_id]
    order = world.orders[line.order_id]
    expired = world.clone()
    expired.orders[order.id] = order.model_copy(
        update={"delivered_at": world.clock - __import__("datetime").timedelta(days=45)})
    after, result, _ = _run_operation(expired, profile.customer_id, line_id,
                                      {"quantity": 1, "payment_method": "original"})
    assert not result["success"] and result["error_code"] == "RETURN_WINDOW_EXPIRED"
    assert after == expired


def test_not_delivered_is_rejected():
    world, aliases, task, profile = _delegated_world()
    line_id = aliases["damaged"]
    line = world.lines[line_id]
    order = world.orders[line.order_id]
    pending = world.clone()
    pending.orders[order.id] = order.model_copy(
        update={"delivered_at": world.clock + __import__("datetime").timedelta(days=2)})
    after, result, _ = _run_operation(pending, profile.customer_id, line_id,
                                      {"quantity": 1, "payment_method": "original"})
    assert not result["success"] and result["error_code"] == "ORDER_NOT_DELIVERED"
    assert after == pending


def test_quantity_above_purchased_is_rejected():
    world, aliases, task, profile = _delegated_world()
    line_id = aliases["damaged"]
    after, result, _ = _run_operation(world, profile.customer_id, line_id,
                                      {"quantity": 2, "payment_method": "original"})
    assert not result["success"] and result["error_code"] == "PURCHASED_QUANTITY_UNAVAILABLE"
    assert after == world


def test_duplicate_refund_is_rejected():
    world, aliases, task, profile = _delegated_world()
    line_id = aliases["damaged"]
    settled, result, _ = _run_operation(world, profile.customer_id, line_id,
                                        {"quantity": 1, "payment_method": "original"})
    assert result["success"]
    after, duplicate, _ = _run_operation(settled, profile.customer_id, line_id,
                                         {"quantity": 1, "payment_method": "original"})
    assert not duplicate["success"] and duplicate["error_code"] == "ALREADY_REFUNDED"
    assert after == settled


def test_approval_limit_requires_policy_review():
    world, aliases, task, profile = _delegated_world()
    line_id = aliases["damaged"]
    line = world.lines[line_id]
    order = world.orders[line.order_id]
    merchant = world.merchants[order.merchant_id]
    limited = world.clone()
    limited.merchants[merchant.id] = merchant.model_copy(update={"refund_approval_limit_cents": 100})
    after, result, _ = _run_operation(limited, profile.customer_id, line_id,
                                      {"quantity": 1, "payment_method": "original"})
    assert not result["success"] and result["error_code"] == "APPROVAL_REQUIRED"
    assert after == limited


def test_payment_method_is_restricted_to_original_channel():
    raw = {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "return_delivered_order_items",
                        "arguments": json.dumps({"line_id": "line-x", "quantity": 1,
                                                 "payment_method": "store-credit"}),
                    },
                }],
            },
        }]
    }
    with pytest.raises(Exception) as excinfo:
        ServiceAction.from_tool_message(raw, "customer", (), ("merchant",))
    assert getattr(excinfo.value, "code", "") == "MODEL_OUTPUT_INVALID"


def test_public_tool_surface_matches_engine_and_reference():
    tools = {row["function"]["name"]: row["function"] for row in service_tools()}
    assert set(tools) == {"list_customer_orders", "get_order_details",
                          "return_delivered_order_items", "respond_to_customer"}
    refund = tools["return_delivered_order_items"]["parameters"]
    assert refund["required"] == ["line_id", "quantity"]
    assert set(refund["properties"]) == {"line_id", "quantity", "payment_method"}
    assert refund["properties"]["payment_method"]["enum"] == ["original"]
    raw = {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call-2",
                    "type": "function",
                    "function": {
                        "name": "return_delivered_order_items",
                        "arguments": json.dumps({"line_id": "line-x", "quantity": 1}),
                    },
                }],
            },
        }]
    }
    action = ServiceAction.from_tool_message(raw, "customer", (), ("merchant",))
    assert action.operation.name == "return_delivered_order_items"
    assert action.operation.params["quantity"] == 1
    assert action.operation.params["payment_method"] == "original"
    assert action.operation.scope_id == "merchant"


def test_certified_witness_cost_fits_episode_budget():
    world, aliases = frozen_world(13)
    task = make_task(world, aliases, remedy=("refund",))
    scenario = customer_scenario(aliases)
    from usami_v2.scenarios.task_certification import ProofBudget, certify_task, certification_cost
    certificate = certify_task(world, task, scenario, ProofBudget(max_plan_candidates=16,
                                                                  max_operation_steps=16,
                                                                  max_witnesses=1))
    cost = certification_cost(certificate)
    assert cost == {"tool_calls": 1, "service_decisions": 2, "customer_turns": 1, "model_calls": 2}
    assert certificate.full_task_reachability_certified
