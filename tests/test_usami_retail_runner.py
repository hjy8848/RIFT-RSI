"""Retail episodes through the real Engine with scripted Services of different skill."""
import pytest

from usami_v2.runtime.agents import ServiceAction
from usami_v2.runtime.retail_runner import run_retail_episode
from usami_v2.scenarios.retail_tasks import FAMILIES, expected_total_text, generate_retail_task, gold_operations


class Script:
    """A Service that proposes, waits for the customer's yes, performs its operations, then reports."""

    uses_model = False

    def __init__(self, operations, report="Done."):
        self.operations, self.report = list(operations), report
        self.stage = 0

    def decide(self, prompt):
        confirmed = any(row.role == "user" and "go ahead" in (row.content or "") for row in prompt.conversation)
        if not confirmed:
            return ServiceAction(kind="reply", message="Here is what I will do. Shall I proceed?")
        done = sum(row.role == "tool" for row in prompt.conversation)
        if done < len(self.operations):
            return ServiceAction(kind="operation", operation=self.operations[done])
        return ServiceAction(kind="reply", message=self.report)


@pytest.mark.parametrize("family", FAMILIES)
def test_an_ideal_service_succeeds_in_every_family(family):
    world, task, _ = generate_retail_task(3, family)
    ops, _ = gold_operations(world, task)
    report = "Done. New total CNY " + (expected_total_text(world, task) if task.ask_total else "")
    episode = run_retail_episode(world, task, Script(ops, report))
    assert episode.status == "completed" and episode.success, (episode.status, episode.state_match, episode.told_total)
    assert episode.tool_calls == len(ops)


@pytest.mark.parametrize("family", ["address_stated", "items_simple", "items_two_lines", "items_fallback_cancel"])
def test_a_service_that_does_nothing_fails_unless_nothing_was_right(family):
    world, task, evidence = generate_retail_task(3, family)
    episode = run_retail_episode(world, task, Script([]))
    assert episode.status == "completed" and not episode.success and evidence["changes_state"]


def test_the_naive_literal_first_choice_fails_where_there_is_a_trap():
    for family in ("items_gift_card_fallback", "items_stock_fallback", "items_fallback_cancel"):
        world, task, evidence = generate_retail_task(5, family)
        goal = task.goals[0]
        naive = Script([__import__("usami_v2.commerce.models", fromlist=["Operation"]).Operation(
            name="modify_order_items", scope_id=next(iter(world.merchants)), resource_id=goal.order_id,
            params={"changes": [{"line_id": w.line_id, "target_sku_id": w.targets[0]} for w in goal.wishes],
                    "payment_method": goal.payment_preferences[0]})])
        episode = run_retail_episode(world, task, naive)
        assert evidence["has_trap"] and not episode.success
        assert any(not result["success"] for _, result in episode.tool_log) or not episode.state_match


def test_acting_on_a_delivered_order_is_refused_by_the_engine_and_counts_as_right_only_if_nothing_changed():
    world, task, _ = generate_retail_task(2, "address_delivered")
    goal = task.goals[0]
    from usami_v2.commerce.models import Operation
    op = Operation(name="modify_order_address", scope_id=next(iter(world.merchants)), resource_id=goal.order_id,
                   params={"street": goal.address.street, "city": goal.address.city, "state": goal.address.state,
                           "postal_code": goal.address.postal_code})
    episode = run_retail_episode(world, task, Script([op]))
    assert episode.tool_log[0][1]["error_code"] == "ORDER_NOT_MODIFIABLE" and episode.success


def test_patience_ends_the_episode_before_the_work_is_done():
    world, task, _ = generate_retail_task(3, "items_simple")
    ops, _ = gold_operations(world, task)
    episode = run_retail_episode(world, task, Script(ops), patience=2)
    assert episode.status == "customer_left" and not episode.success
