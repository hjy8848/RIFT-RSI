import json
import random

import pytest

from rift.benchmark.generator import balanced_seeds, draft_instance, generate_instance, task_outcome
from rift.designer.format import PlanError, parse_plan, plan_from_drafts, plan_proposal, plan_schema
from rift.designer.prompt import designer_messages
from rift.designer.stage0 import certify_designer_plan, certify_saved_plans, dry_run_generate, run
from rift.runtime.ascend_910b3 import _proposed_lines
from rift.runtime.curriculum_run import WorldKit, designer_round, parse_task_specs, spec_key


def example_plan(family='F6', seed=5):
    data, task = draft_instance(family, seed)
    return plan_from_drafts(data, task, 'test plan').model_dump(mode='json')


@pytest.mark.parametrize('family', ['F1', 'F4', 'F5', 'F6', 'F8', 'F9', 'F11'])
def test_generator_plan_round_trips_to_the_same_answer(tmp_path, family):
    expected = task_outcome(generate_instance(family, 5).task)
    row = certify_designer_plan(WorldKit(tmp_path), json.dumps(example_plan(family, 5)), 'plan-a', 5)
    assert row['status'] == 'certified', row
    assert row['answers'] == [expected]


def test_plan_errors_are_readable_codes():
    with pytest.raises(PlanError, match='NO_JSON'):
        parse_plan('I cannot do that')
    plan = example_plan()
    plan['tasks'][0]['requests'][0]['line'] = 'nowhere'
    with pytest.raises(PlanError, match='UNKNOWN_LINE'):
        plan_proposal(plan, 'bad-line')
    plan = example_plan('F1', 3)
    plan['tasks'][0]['requests'][0]['attributes'] = {'color': 'red'}
    with pytest.raises(PlanError, match='REQUEST_INVALID'):
        plan_proposal(plan, 'bad-request')
    plan = example_plan()
    plan['world']['skus'].append(dict(plan['world']['skus'][0]))
    with pytest.raises(PlanError, match='WORLD_INVALID'):
        plan_proposal(plan, 'duplicate-alias')
    assert plan_schema()['title'] == 'DesignerPlan'


def test_history_steps_must_carry_the_fields_their_operation_needs():
    base = example_plan('F1', 3)
    for bad in ({'operation': 'return_delivered_order_items'},                      # no line
                {'operation': 'exchange_delivered_order_items', 'line': 'line_0'},  # no target_sku
                {'operation': 'cancel_order'},                                      # no order
                {'operation': 'cancel_order', 'order': 'order_0', 'quantity': 2}):  # a cancel has no quantity
        plan = json.loads(json.dumps(base))
        plan['world']['history'] = [bad]
        with pytest.raises(PlanError, match='SCHEMA'):
            parse_plan(plan)
    plan = json.loads(json.dumps(base))
    plan['world']['history'] = [{'operation': 'cancel_order', 'order': 'order_0'}]
    assert parse_plan(plan).world.history[0].order == 'order_0'
    schema = json.dumps(plan_schema())
    assert 'HistoryLine' in schema and 'HistoryExchange' in schema and 'HistoryCancel' in schema


def test_a_plan_with_engine_history_round_trips_and_certifies(tmp_path):
    data, task = draft_instance('F1', 3, near_distractors=3)
    assert data.get('history')                                  # the generator recorded a settled distractor
    plan = plan_from_drafts(data, task, 'history round trip').model_dump(mode='json')
    row = certify_designer_plan(WorldKit(tmp_path), json.dumps(plan), 'plan-history', 3)
    assert row['status'] == 'certified', row
    from rift.designer.format import plan_drafts
    rebuilt, _ = plan_drafts(parse_plan(plan), 'plan-history')
    assert rebuilt['history'] == data['history']


def test_a_saved_registry_with_designer_tasks_reloads(tmp_path):
    """Training workers reload the registry from disk; a task with several requests on one line is v4 (covered by the real-pool rehearsal)."""
    kit = WorldKit(tmp_path)
    data, task = draft_instance('F1', 3, near_distractors=3)
    plan = plan_from_drafts(data, task, 'reload').model_dump(mode='json')
    row = certify_designer_plan(kit, json.dumps(plan), 'plan-reload', 3)
    assert row['status'] == 'certified', row
    kit.save()
    reloaded = WorldKit(tmp_path)
    ref = row['task_refs'][0]
    assert (reloaded.task_registry.resolve_record(ref).task.protocol
            == kit.task_registry.resolve_record(ref).task.protocol)


def test_uncertifiable_request_is_rejected_with_a_reason_not_a_crash(tmp_path):
    plan = example_plan('F1', 3)
    # Naming an item by its order date needs an order date the plan never gave.
    for order in plan['world']['orders']:
        order.pop('ordered_days_ago', None)
    plan['tasks'][0]['requests'][0]['refer_by'] = 'order_date'
    row = certify_designer_plan(WorldKit(tmp_path), json.dumps(plan), 'no-order-date', 3)
    assert row['status'] == 'tasks_rejected' and row['task_refs'] == [] and row['reasons']


def test_stage0_dry_run_and_stage1_recertify_the_same_tasks(tmp_path):
    report = run(WorldKit(tmp_path / 'stage0'), lambda prompts: dry_run_generate(prompts, seed=11), [],
                 samples=4, seed=11, out_dir=tmp_path / 'stage0', signatures=False)
    assert report['plans'] == 4 and report['certified_tasks'] == 4
    rows = [json.loads(line) for line in (tmp_path / 'stage0' / 'designer-plans.jsonl').read_text().splitlines()]
    refs = certify_saved_plans(WorldKit(tmp_path / 'stage1'), tmp_path / 'stage0' / 'designer-plans.jsonl')
    assert refs == [ref for row in rows for ref in row['task_refs']]


def test_prompt_contains_rules_failures_and_examples():
    case = {'request': 'Please refund 1 item', 'success_rate': 0.25, 'agent_steps': ['list_customer_orders:ok'],
            'last_message': 'Done', 'goals': [{'item': 'damaged tablet', 'action': 'refund', 'quantity': 1,
                                              'answer': 'refuse (APPROVAL_REQUIRED)'}]}
    system, user = designer_messages([case], seed=3)
    assert 'refund_approval_limit_cents' in system['content']
    assert 'refuse (APPROVAL_REQUIRED)' in user['content'] and user['content'].count('"hypothesis"') == 2


def test_copy_score_flags_a_copied_example_and_not_an_invented_plan():
    from rift.designer.prompt import copy_score, example_plans
    examples = example_plans(5, 2)
    copied = examples[0].model_dump(mode='json')
    assert copy_score(copied, examples) == 1.0
    invented = json.loads(json.dumps(copied))
    for sku in invented['world']['skus']:
        sku['price_cents'] += 7919
    for order in invented['world']['orders']:
        for line in order['lines']:
            line['paid_cents'] += 6101
    invented['world']['refund_approval_limit_cents'] += 4409
    assert copy_score(invented, examples) < 0.5
    assert 'example' not in designer_messages([], seed=5)[1]['content'].split('Examples of valid plans:')[1].lower()


def test_focus_is_deterministic_and_covers_every_action_and_structure():
    from collections import Counter
    from rift.designer.focus import ACTION_RULES, STRUCTURES, focus_for, focus_text
    assert focus_for(7) == focus_for(7)
    foci = [focus_for(seed) for seed in range(400)]
    assert {f['action'] for f in foci} == set(ACTION_RULES)                 # reship and cancel are asked for
    assert {f['structure'] for f in foci} == set(STRUCTURES)                # history and multi-request too
    assert min(Counter(f['action'] for f in foci).values()) > 40
    assert 'reship' in focus_text({**foci[0], 'action': 'reship'})
    messages = designer_messages([], seed=5, focus=True)
    assert focus_text(focus_for(5)) in messages[1]['content']
    assert 'Focus of this plan' not in designer_messages([], seed=5)[1]['content']


def test_selection_drops_copies_and_repeated_signatures():
    from rift.designer.stage0 import select_tasks
    rows = [{'task_refs': ['a', 'b'], 'signatures': ['s1', 's2'], 'copy_score': 0.1},
            {'task_refs': ['c'], 'signatures': ['s1'], 'copy_score': 0.0},                  # same signature as 'a'
            {'task_refs': ['d'], 'signatures': ['s9'], 'copy_score': 0.9},                  # copied from an example
            {'task_refs': ['e', 'f'], 'signatures': ['unmeasured:ValueError', 'unmeasured:ValueError'],
             'copy_score': None}]                                                           # unmeasured counts as unique
    assert select_tasks(rows) == 4
    assert [row['selected'] for row in rows] == [[True, True], [False], [False], [True, True]]


def test_stage0_with_focus_selects_and_can_reload_only_the_selection(tmp_path):
    report = run(WorldKit(tmp_path / 's0'), lambda prompts: dry_run_generate(prompts, seed=41), [],
                 samples=6, seed=41, out_dir=tmp_path / 's0', focus=True)
    assert report['plans'] == 6 and report['selected_tasks'] <= report['certified_tasks']
    assert 'focus_asked' in report and report['actions']
    rows = [json.loads(line) for line in (tmp_path / 's0' / 'designer-plans.jsonl').read_text().splitlines()]
    assert all(len(row['selected']) == len(row['task_refs']) == len(row['actions']) for row in rows)
    plans = tmp_path / 's0' / 'designer-plans.jsonl'
    everything = certify_saved_plans(WorldKit(tmp_path / 'a'), plans)
    chosen = certify_saved_plans(WorldKit(tmp_path / 'b'), plans, selected_only=True)
    assert set(chosen) <= set(everything) and len(chosen) == report['selected_tasks']
    legacy = [dict(row) for row in rows]
    for row in legacy:
        row.pop('selected')
    (tmp_path / 'old.jsonl').write_text('\n'.join(json.dumps(row) for row in legacy), encoding='utf-8')
    with pytest.raises(ValueError, match='no selection'):
        certify_saved_plans(WorldKit(tmp_path / 'c'), tmp_path / 'old.jsonl', selected_only=True)


def test_merge_selects_across_files_and_keeps_copies_on_request(tmp_path):
    from rift.designer.merge import merge_plans
    for name, seed in (('a', 61), ('b', 62)):
        run(WorldKit(tmp_path / name), lambda prompts, s=seed: dry_run_generate(prompts, seed=s), [],
            samples=5, seed=seed, out_dir=tmp_path / name)
    inputs = [tmp_path / 'a' / 'designer-plans.jsonl', tmp_path / 'b' / 'designer-plans.jsonl']
    report = merge_plans(inputs, tmp_path / 'm' / 'designer-plans.jsonl', copy_threshold=1.01)
    assert report['input_plans'] == 10 and report['plans_kept'] == 10
    assert 0 < report['selected_tasks'] <= report['certified_tasks']
    merged = [json.loads(line) for line in (tmp_path / 'm' / 'designer-plans.jsonl').read_text().splitlines()]
    signatures = [sig for row in merged for sig, keep in zip(row['signatures'], row['selected']) if keep]
    assert len(signatures) == len(set(signatures))                      # distinct over BOTH files
    refs = certify_saved_plans(WorldKit(tmp_path / 'check'), tmp_path / 'm' / 'designer-plans.jsonl', selected_only=True)
    assert len(refs) == report['selected_tasks']
    with pytest.raises(ValueError, match='two input files'):
        merge_plans([inputs[0], inputs[0]], tmp_path / 'x.jsonl')


def test_balanced_seeds_split_answers_evenly():
    seeds = balanced_seeds('F5', 6, 100)
    answers = [task_outcome(generate_instance('F5', seed).task) for seed in seeds]
    assert answers.count('success') == 3 and answers.count('refusal') == 3


def test_generator_specs_parse_and_certify(tmp_path):
    specs = parse_task_specs([{'generator': 'F11', 'instances': 2, 'seed': 0, 'balanced': True}])
    refs = WorldKit(tmp_path, patience_slack=2).certify_specs(specs)
    assert list(refs) == [spec_key(specs[0])] == ['gen:F11:{"balanced":true}'] and len(refs[spec_key(specs[0])]) == 2


def test_designer_round_mixes_a_fixed_share():
    rng = random.Random(0)
    chosen = designer_round(rng, ['d1', 'd2'], [('gen:F1', 'm1'), ('gen:F2', 'm2'), ('gen:F3', 'm3')], 8, 0.25)
    assert [key for key, _ in chosen].count('designer') == 6 and len(chosen) == 8


def test_failure_cases_show_hidden_goals_and_agent_steps(tmp_path):
    from types import SimpleNamespace
    from usami_v2.commerce.models import Operation
    from usami_v2.runtime.agents import ServiceAction
    from usami_v2.runtime.runner import run_episode
    from rift.designer.failures import failure_cases
    from rift.runtime.ascend_910b3 import _UsamiCustomer
    kit = WorldKit(tmp_path)
    ref = kit.certify_specs([{'generator': 'F1', 'instances': 1, 'seed': 0}])['gen:F1'][0]
    record = kit.task_registry.resolve_record(ref)

    class LazyService:
        uses_model, last_call = False, None

        def __init__(self):
            self.turn = 0

        def decide(self, prompt):
            self.turn += 1
            merchant = next(iter(record.world.merchants))
            if self.turn == 1:
                return ServiceAction(kind='operation', operation=Operation(
                    name='list_customer_orders', scope_id=merchant, resource_id=merchant))
            return ServiceAction(kind='reply', message='All done.')
    episode = run_episode(record.world, record.task, record.profile, _UsamiCustomer(1), LazyService())
    store = SimpleNamespace(resolve_episode=lambda episode_id: episode)
    rows = [{'task_ref': ref, 'outcome': 'failure', 'episode_id': episode.record_hash},
            {'task_ref': ref, 'outcome': 'success', 'episode_id': 'x'}]
    [case] = failure_cases(rows, kit.task_registry, store)
    assert case['success_rate'] == 0.5 and case['agent_steps'] == ['list_customer_orders:ok']
    assert case['goals'][0]['answer'] in ('carry it out', 'refuse (APPROVAL_REQUIRED)')
    assert case['last_message'] == 'All done.'


def test_scripted_customer_finds_the_item_from_natural_wording():
    references = {'line_a': 'smart watch delivered about 12 days ago',
                  'line_b': 'wireless headphones delivered about 3 days ago'}
    assert _proposed_lines('Your watch from 12 days ago costs CNY 30 more. Shall I proceed?', references) == ['line_a']
    assert _proposed_lines('I can refund the headphones now, okay?', references) == ['line_b']
    assert sorted(_proposed_lines('Shall I proceed with both items?', references)) == []
    assert _proposed_lines('Proceed with line_b?', references) == ['line_b']
