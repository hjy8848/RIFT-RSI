"""CPU integration of retail Designer certification with the self-teaching control loop."""
import json
from types import SimpleNamespace

from rift.designer.retail_prompt import example_retail_plans
from rift.runtime import curriculum_run, self_teach, teacher
from rift.runtime.curriculum_run import WorldKit
from rift.schemas import Checkpoint


def test_two_retail_actions_certify_select_and_reward_real_plans(tmp_path, monkeypatch):
    def checkpoint(n):
        return Checkpoint('service', str(n), 'tokenizer', f'weights-{n}')
    class Context:
        tool_profile = 'retail'
        group_size = 2
        root = tmp_path
        world = WorldKit(tmp_path)
        capture_store = None
        def initial_checkpoint(self):
            return checkpoint(0)
        def close_runtime(self):
            pass
        def resync(self, checkpoint):
            pass
        def forget_checkpoints(self, checkpoints):
            pass
    context = Context()
    seen = {'solver': [], 'teacher': []}
    def generate(ctx, cp, failures, *, action, **kwargs):
        plans = example_retail_plans(100 + action, 2)
        samples = []
        for index, plan in enumerate(plans):
            value = plan.model_dump(mode='json')
            value['gift_card_balance_cents'] += 17
            for order in value['orders']:
                for item in order['items']:
                    item['price_cents'] += 123
            for variant in value['variants']:
                variant['price_cents'] += 123
            samples.append(teacher.TeacherSample(f'a{action:03d}-g0-p{index}', f'a{action:03d}-g0', cp,
                (1, 2), (3, 4), (-0.1, -0.2), json.dumps(value), f'plan-{action}-{index}', action * 10 + index))
        return samples, [900 + action]
    def solve(ctx, cp, *, action, index, teacher_refs, **kwargs):
        assert teacher_refs and all(ctx.world.task_resolver.resolve(ref) for ref in teacher_refs)
        seen['solver'].append(tuple(teacher_refs))
        counts = {ref: (1, 2) for ref in teacher_refs}
        return checkpoint(100 + action), [{'task_counts': counts, 'episodes': []}], index + 1
    def update(ctx, samples, rewards, before, ledger, index):
        assert all(sample.row['status'] == 'certified' for sample in samples)
        seen['teacher'].append(rewards)
        return SimpleNamespace(policy_after=checkpoint(len(seen['teacher']))), len(samples)
    monkeypatch.setattr(curriculum_run, 'sample_rows', lambda *a, **kw: [])
    monkeypatch.setattr(self_teach, 'failures_from_rows', lambda *a, **kw: [])
    monkeypatch.setattr(self_teach, 'generate_plans', generate)
    monkeypatch.setattr(self_teach, 'run_solver_steps', solve)
    monkeypatch.setattr(self_teach, 'teacher_update', update)
    result = self_teach.self_teach(context,
        train_specs=[{'retail_family': 'items_simple', 'instances': 2, 'seed': 20}], probe_specs=None,
        actions=2, groups=1, plans=2, steps=1, tasks_per_step=2, mix_fraction=0.5,
        seed=1, calibrate_every=0, plan_format='retail')
    assert result['status'] == 'self_teach_completed' and len(seen['solver']) == len(seen['teacher']) == 2
    assert all(any(reward == 1.0 for reward in rewards.values()) for rewards in seen['teacher'])


def test_deduplication_scope_defaults_keep_legacy_runs_cumulative():
    assert self_teach.resolve_dedup_scope('legacy') == 'run'
    assert self_teach.resolve_dedup_scope('retail') == 'action'
    assert self_teach.resolve_dedup_scope('legacy', 'action') == 'action'
    assert self_teach.resolve_dedup_scope('retail', 'run') == 'run'


def test_reward_needs_a_normally_ended_episode_even_when_the_state_is_right():
    from rift.benchmark.usami import public_episode
    from rift.schemas import Outcome
    from usami_v2.runtime.agents import ServiceAction
    from usami_v2.runtime.retail_runner import run_retail_episode
    from usami_v2.scenarios.retail_tasks import generate_retail_task, gold_operations

    class NoFarewell:
        """Asks, then does the golden work, then never speaks again: the customer never gets to finish."""
        uses_model = False

        def __init__(self, operations):
            self.operations = list(operations)

        def decide(self, prompt):
            if not any(row.role == 'user' and 'go ahead' in (row.content or '') for row in prompt.conversation):
                return ServiceAction(kind='reply', message='Shall I proceed?')
            done = sum(row.role == 'tool' for row in prompt.conversation)
            if done < len(self.operations):
                return ServiceAction(kind='operation', operation=self.operations[done])
            return ServiceAction(kind='operation', operation=self.operations[0])    # keeps working, never replies

    world, task, _ = generate_retail_task(5, 'address_stated')
    ops = gold_operations(world, task)[0]
    episode = run_retail_episode(world, task, NoFarewell(ops), max_service_decisions=len(ops) + 2)
    assert episode.state_match and episode.status != 'completed'
    assert public_episode(episode.record).outcome is Outcome.FAILURE


def test_address_source_is_constrained_by_the_plan_schema():
    import pytest
    from rift.designer.format import PlanError
    from rift.designer.retail_format import parse_retail_plan
    base = example_retail_plans(3, 1)[0].model_dump(mode='json')
    order = base['orders'][0]['ref']

    def plan_with(request):
        value = json.loads(json.dumps(base))
        value['tasks'] = [{'requests': [request], 'ask_total': False}]
        return value
    for good in ({'type': 'set_order_address', 'order': order, 'to': 'new'},
                 {'type': 'set_order_address', 'order': order, 'to': 'profile'},
                 {'type': 'set_order_address', 'order': order, 'to': 'order:' + order},   # schema allows it; validate_scene refuses
                 {'type': 'set_default_address', 'to': 'new'}):
        parse_retail_plan(plan_with(good))
    for bad in ({'type': 'set_order_address', 'order': order, 'to': 'profile:1'},
                {'type': 'set_order_address', 'order': order, 'to': 'order:'},
                {'type': 'set_default_address', 'to': 'profile'},
                {'type': 'set_default_address', 'to': 'default'}):
        with pytest.raises(PlanError):
            parse_retail_plan(plan_with(bad))


def test_a_total_can_only_be_asked_in_a_task_that_changes_items():
    import pytest
    from rift.designer.format import PlanError
    from rift.designer.retail_format import retail_plan_scene
    base = example_retail_plans(4, 1)[0].model_dump(mode='json')
    order = base['orders'][0]['ref']
    value = json.loads(json.dumps(base))
    value['tasks'] = [{'requests': [{'type': 'set_order_address', 'order': order, 'to': 'new'}], 'ask_total': True}]
    with pytest.raises(PlanError) as caught:
        retail_plan_scene(value)
    assert caught.value.code == 'ASK_TOTAL_INVALID'
    value['tasks'][0]['ask_total'] = False
    retail_plan_scene(value)


def test_the_designer_prompt_states_the_rules_models_broke():
    from rift.designer.retail_prompt import SYSTEM
    for rule in ('ask_total may be true only', "'of' is exactly that change's item", 'never profile:something',
                 "must never be that same order"):
        assert rule in SYSTEM


def test_episode_diagnostics_understand_retail_records():
    from rift.runtime.diagnose_episodes import classify_episode, confirmation_order_compliance, episode_metrics
    from usami_v2.runtime.agents import ServiceAction
    from usami_v2.runtime.retail_runner import run_retail_episode
    from usami_v2.scenarios.retail_tasks import generate_retail_task, gold_operations

    class Script:
        uses_model = False

        def __init__(self, ops, ask_first):
            self.ops, self.ask_first = list(ops), ask_first

        def decide(self, prompt):
            confirmed = any(row.role == 'user' and 'go ahead' in (row.content or '') for row in prompt.conversation)
            if self.ask_first and not confirmed:
                return ServiceAction(kind='reply', message='Shall I proceed?')
            done = sum(row.role == 'tool' for row in prompt.conversation)
            if done < len(self.ops):
                return ServiceAction(kind='operation', operation=self.ops[done])
            return ServiceAction(kind='reply', message='Done.')

    world, task, _ = generate_retail_task(6, 'address_stated')
    ops = gold_operations(world, task)[0]
    asked = run_retail_episode(world, task, Script(ops, True)).record
    assert confirmation_order_compliance(asked) is True
    assert classify_episode(asked) == 'success'
    assert episode_metrics(asked)['tool_calls'] == len(ops)
    rushed = run_retail_episode(world, task, Script(ops, False)).record
    assert confirmation_order_compliance(rushed) is False
    idle = run_retail_episode(world, task, Script([], True)).record
    assert confirmation_order_compliance(idle) is None
    assert classify_episode(idle) == 'customer_finished_unsuccessful'
