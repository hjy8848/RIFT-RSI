"""Retail episodes through the real Engine with scripted Services of different skill."""
import pytest

from usami_v2.runtime.agents import ServiceAction
from usami_v2.runtime.retail_runner import RetailEpisodeRecord, replay_retail_episode, run_retail_episode
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


def test_acting_on_a_delivered_order_is_refused_and_does_not_complete_the_mixed_task():
    world, task, _ = generate_retail_task(2, "address_delivered_mixed")
    goal = task.goals[0]
    from usami_v2.commerce.models import Operation
    op = Operation(name="modify_order_address", scope_id=next(iter(world.merchants)), resource_id=goal.order_id,
                   params={"street": goal.address.street, "city": goal.address.city, "state": goal.address.state,
                           "postal_code": goal.address.postal_code})
    episode = run_retail_episode(world, task, Script([op]))
    assert episode.tool_log[0][1]["error_code"] == "ORDER_NOT_MODIFIABLE" and not episode.success


def test_patience_ends_the_episode_before_the_work_is_done():
    world, task, _ = generate_retail_task(3, "items_simple")
    ops, _ = gold_operations(world, task)
    episode = run_retail_episode(world, task, Script(ops), patience=2)
    assert episode.status == "customer_left" and not episode.success


def test_record_round_trip_and_replay_detects_tampering():
    from rift.benchmark.usami import public_episode
    from rift.schemas import Outcome
    from rift.training.service_data import _replayed_outcome
    world, task, _ = generate_retail_task(3, "items_simple")
    ops, _ = gold_operations(world, task)
    episode = run_retail_episode(world, task, Script(ops))
    restored = RetailEpisodeRecord.model_validate_json(episode.record.model_dump_json())
    assert replay_retail_episode(restored).verified
    assert public_episode(restored).outcome is Outcome.SUCCESS
    assert _replayed_outcome(restored) == (True, Outcome.SUCCESS)
    corrupted = restored.model_copy(update={"final_world": world})
    assert not replay_retail_episode(corrupted).verified
    assert public_episode(corrupted).outcome is Outcome.UNKNOWN


def test_stepwise_retail_session_success_idle_and_parser_failure():
    import asyncio
    import json
    from types import SimpleNamespace
    from rift.benchmark.usami import public_episode
    from rift.schemas import Outcome
    from rift.verl.usami_session import NativeUsamiSession
    from usami_v2.runtime.retail_runner import ScriptedRetailCustomer
    from usami_v2.runtime.runner import EpisodeBudget

    world, task, _ = generate_retail_task(8, 'address_stated')
    op = gold_operations(world, task)[0][0]
    class Parser:
        def __init__(self, mode):
            self.step, self.mode = 0, mode
        async def extract_tool_calls(self, token_ids, schemas):
            self.step += 1
            if self.mode == 'failure':
                raise RuntimeError('parser failure')
            if self.step == 2 and self.mode == 'success':
                return '', [SimpleNamespace(name='modify_order_address', tool_call_id='call',
                    arguments=json.dumps({'order_id': op.resource_id, **op.params}))]
            return ('Shall I proceed?' if self.step == 1 else 'Done.'), []

    async def exercise(mode):
        session = NativeUsamiSession(world, task, None, ScriptedRetailCustomer(), Parser(mode),
                                     budget=EpisodeBudget(), tool_profile='retail')
        try:
            await session.start()
            done, step = False, 0
            while not done:
                _, _, done = await session.advance([step + 1], [10, step + 1], {})
                step += 1
            record = await session.finish()
            assert replay_retail_episode(record).verified
            assert len(record.model_calls) == step
            return public_episode(record).outcome
        finally:
            await session.abort()
    assert asyncio.run(exercise('success')) is Outcome.SUCCESS
    assert asyncio.run(exercise('idle')) is Outcome.FAILURE
    assert asyncio.run(exercise('failure')) is Outcome.UNKNOWN


def test_retail_episode_capture_persists_resolves_and_builds_training_rows(tmp_path):
    import asyncio
    import json
    from types import SimpleNamespace
    from rift.schemas import Checkpoint
    from rift.training.service_data import prepare_service_trajectory_update
    from rift.verl.capture import ContinuousCapture
    from rift.verl.capture_store import NativeCaptureStore
    from rift.verl.usami_session import NativeUsamiSession
    from usami_v2.runtime.retail_runner import ScriptedRetailCustomer, replay_retail_episode
    from usami_v2.runtime.runner import EpisodeBudget

    world, task, _ = generate_retail_task(19, 'address_stated')
    operation = gold_operations(world, task)[0][0]
    checkpoint = Checkpoint('service', 'retail-test', 'tokenizer', 'weights')
    class Attestor:
        def verify_capture(self, trace, request, response):
            return True
    store = NativeCaptureStore(tmp_path / 'captures', serialization='retail-fixture', attestor=Attestor())

    async def episode(seed):
        class Parser:
            step = 0
            async def extract_tool_calls(self, token_ids, schemas):
                self.step += 1
                if self.step == 2:
                    return '', [SimpleNamespace(name='modify_order_address', tool_call_id='call',
                        arguments=json.dumps({'order_id': operation.resource_id, **operation.params}))]
                return ('Shall I proceed?' if self.step == 1 else 'Done.'), []
        rollout = 'rollout-' + str(seed)
        session = NativeUsamiSession(world, task, None, ScriptedRetailCustomer(), Parser(),
            budget=EpisodeBudget(), tool_profile='retail', rollout_id=rollout,
            group_id='retail-group', group_size=2, task_ref=task.record_hash)
        capture = ContinuousCapture([1], checkpoint)
        try:
            await session.start()
            done, index = False, 0
            while not done:
                prompt = list(capture.current)
                action = [seed + index + 10]
                capture.add_generation(prompt, action, None, prompt + action, list(capture.mask) + [1])
                _, _, done = await session.advance(action, prompt, {'temperature': 1.0},
                                                   generation_id=f'{seed}-{index}')
                if not done:
                    capture.add_context(list(capture.current) + [99], list(capture.mask) + [0])
                index += 1
            record = await session.finish()
            await store.persist(record, checkpoint, tuple(capture.decisions), capture.output())
            restored = store.resolve_episode(record.record_hash)
            assert replay_retail_episode(restored).verified and store.verify_trajectory(restored, checkpoint)
            return restored
        finally:
            await session.abort()

    async def pair():
        return [await episode(100), await episode(200)]
    records = asyncio.run(pair())
    update = prepare_service_trajectory_update('retail-package', {task.record_hash}, records,
                                               checkpoint, store)
    assert len(update.rollouts) == 2 and {row.reward for row in update.rollouts} == {1.0}
