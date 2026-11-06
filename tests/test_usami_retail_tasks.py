"""Retail tasks: generated from seeds, certified natively, judged by the golden final state."""
import pytest

from usami_v2.scenarios.retail_tasks import (
    FAMILIES, certify_retail_task, delegated, evaluate_retail_task, expected_total_text, generate_retail_task,
    gold_operations, projection,
)
from usami_v2.commerce.reference import decide

SEEDS = range(1, 7)


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_generates_certifies_and_is_reproducible(family):
    for seed in SEEDS:
        world, task, evidence = generate_retail_task(seed, family)
        assert evidence["certified"]
        again_world, again_task, _ = generate_retail_task(seed, family)
        assert again_task == task and again_world == world
        assert evidence["changes_state"] == (family != "address_delivered")


@pytest.mark.parametrize("family", FAMILIES)
def test_golden_state_succeeds_and_doing_nothing_fails_when_something_must_change(family):
    for seed in SEEDS:
        world, task, evidence = generate_retail_task(seed, family)
        _, gold = gold_operations(world, task)
        texts = (expected_total_text(world, task),) if task.ask_total else ()
        assert evaluate_retail_task(world, gold, task, texts)["success"]
        idle = evaluate_retail_task(world, world, task, texts)
        assert idle["state_match"] == (not evidence["changes_state"])


def test_the_total_must_actually_be_told():
    world, task, _ = generate_retail_task(3, "items_ask_total")
    _, gold = gold_operations(world, task)
    assert not evaluate_retail_task(world, gold, task, ("something else",))["success"]
    assert evaluate_retail_task(world, gold, task, ("Your new total is CNY " + expected_total_text(world, task),))["success"]


def test_preference_chains_end_where_the_engine_says():
    for seed in SEEDS:
        world, task, evidence = generate_retail_task(seed, "items_gift_card_fallback")
        ops, gold = gold_operations(world, task)
        pp = next(k for k, v in world.payment_methods.items() if v.kind == "paypal")
        assert ops[0].name == "modify_order_items" and ops[0].params["payment_method"] == pp
        assert evidence["has_trap"]                                   # the literal first preference fails
        world, task, evidence = generate_retail_task(seed, "items_stock_fallback")
        ops, _ = gold_operations(world, task)
        wish = task.goals[0].wishes[0]
        assert ops[0].params["changes"][0]["target_sku_id"] == wish.targets[1] and evidence["has_trap"]
        world, task, evidence = generate_retail_task(seed, "items_fallback_cancel")
        ops, gold = gold_operations(world, task)
        assert [op.name for op in ops] == ["cancel_order"] and evidence["has_trap"]


def test_cheaper_variant_refunds_the_gift_card_and_two_lines_change_in_one_call():
    world, task, _ = generate_retail_task(2, "items_cheaper_to_gift_card")
    ops, gold = gold_operations(world, task)
    gc = next(k for k, v in world.payment_methods.items() if v.kind == "gift_card")
    assert ops[0].params["payment_method"] == gc
    assert gold.payment_methods[gc].balance_cents > world.payment_methods[gc].balance_cents
    world, task, _ = generate_retail_task(2, "items_two_lines")
    ops, _ = gold_operations(world, task)
    assert len(ops) == 1 and len(ops[0].params["changes"]) == 2


def test_splitting_the_one_shot_change_is_not_the_golden_state():
    world, task, _ = generate_retail_task(4, "items_two_lines")
    ops, gold = gold_operations(world, task)
    state = delegated(world, task.customer_id)
    for change in ops[0].params["changes"]:
        op = ops[0].model_copy(update={"params": {"changes": [change], "payment_method": "original"}})
        decision = decide(state, task.customer_id, op)
        if decision.allowed:
            state = decision.expected_state
        else:
            assert decision.error_code == "ORDER_LOCKED"
    assert projection(state) != projection(gold)
    assert not evaluate_retail_task(world, state, task)["state_match"]


def test_addresses_come_from_other_orders_and_the_profile():
    world, task, _ = generate_retail_task(5, "address_from_order")
    _, gold = gold_operations(world, task)
    goal = task.goals[0]
    assert gold.orders[goal.order_id].shipping_address == world.orders[goal.source_order_id].shipping_address
    world, task, _ = generate_retail_task(5, "address_profile")
    _, gold = gold_operations(world, task)
    assert gold.orders[task.goals[0].order_id].shipping_address == world.customers[task.customer_id].default_address
    world, task, _ = generate_retail_task(5, "address_delivered")
    assert "address" in task.customer_request and projection(gold_operations(world, task)[1]) == projection(world)


def test_public_text_names_no_internal_ids_and_is_deterministic():
    for family in FAMILIES:
        world, task, _ = generate_retail_task(7, family)
        text = task.customer_request
        for key in list(world.orders) + list(world.lines) + list(world.skus) + list(world.payment_methods):
            assert key not in text
        assert text.startswith("Please help me with the following request(s):")


def test_a_tampered_task_is_refused():
    world, task, _ = generate_retail_task(1, "address_stated")
    forged = task.model_copy(update={"customer_request": task.customer_request.replace("Please", "Kindly")})
    with pytest.raises(ValueError):
        certify_retail_task(world, forged.checked())
