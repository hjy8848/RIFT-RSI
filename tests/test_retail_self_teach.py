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
