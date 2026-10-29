"""Offline experiment plumbing tests; scripted replies are not live evidence."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / 'experiments/vast_coevo/examples'
sys.path.insert(0, str(EXAMPLES))
import multigen_heldout as experiment
from multigen_worlds import environment, tasks
from multigen_budget import CallLedger, BudgetClient
from integrations.usami import UsamiProfile, ServiceTaskProfile, UsamiAdapter
from integrations.usami.domains import native_task
from vast_coevo.schema import digest
from vast_coevo.evolution.models import EvolutionRecord, RunConfig
from vast_coevo.storage.runs import BudgetExhausted
from test_vast_controller import setup_run
from test_vast_usami import SPEC, ScriptedClient, answer, call, libraries, service_arguments


def test_fixed_customer_uses_same_controller_without_any_customer_reflection(tmp_path):
    controller, config, adapter, c, s, store, archive, runs = setup_run(tmp_path, service_modes=('add',))
    config = RunConfig.model_validate({**config.model_dump(mode='json'), 'customer_updates': False})
    c.complete = lambda *_: pytest.fail('fixed Customer cannot run Reflection')
    report = controller.run(config)
    assert report.status == 'completed', report.error
    assert not c.calls and len(s.calls) == 2
    assert report.selected_customer_hash == config.initial_customer_hash
    for g in range(2):
        row = runs.load_evolution(config.run_id, g, 'customer')
        assert row.outcome == 'fixed' and row.reflection is None and row.patch is None
        assert row.consumption.llm_calls == 0
        assert runs.load_evolution(config.run_id, g, 'service').dynamic_validation.passed
    assert controller.run(config) == report


@pytest.mark.parametrize('index', list(range(6)))
def test_new_worlds_are_valid_and_frozen_tasks_have_original_certificates(index):
    w, actor, pid, blocked = environment(20000 + index)
    goals = tasks(w, actor, pid, blocked, 20000 + index)
    p = UsamiProfile(config_id='new', domain='cloudops', seed=w.seed, identity=actor,
        mode='sandbox', split='heldout', frozen_world=w.model_dump(mode='json'))
    for task in goals:
        t = ServiceTaskProfile(task_ref='t', config_id='new', scenario=task.family, frozen_task=task.model_dump(mode='json'))
        world, actual, _ = native_task(p, t)
        assert world.state_hash == w.state_hash and actual == task
    raw = goals[0].model_dump(mode='json'); raw['witnesses'][0]['final_state_hash'] = 'f' * 64
    with pytest.raises(ValueError, match='certificate'):
        native_task(p, ServiceTaskProfile(task_ref='tampered', config_id='new', scenario='bad', frozen_task=raw))


def test_prepare_is_offline_immutable_and_keeps_heldout_out_of_training(tmp_path):
    root = tmp_path / 'registered'; experiment.prepare(root); m = experiment.pinned(root)
    train = set(sum(m['training_profiles'], []) + m['smoke_profiles'])
    assert not train.intersection(m['heldout_profiles'])
    assert len(m['heldout_profiles']) == 24 and m['heldout_jobs'] == 288
    assert len({r['business_shape_hash'] for r in m['worlds']}) == 30
    assert (root / 'MANIFEST.json').exists() and not (root / 'EXECUTION_STARTED.json').exists()
    with pytest.raises(ValueError, match='overwrite'): experiment.prepare(root)


@pytest.mark.parametrize('kind', ['completion', 'policy_refusal'])
def test_heldout_score_uses_actual_goal_oracle_authority_and_explicit_refusal(tmp_path, kind):
    w, actor, pid, blocked = environment(20000)
    goals = tasks(w, actor, pid, blocked, 20000); task = next(t for t in goals if t.kind == kind)
    p = UsamiProfile(config_id='heldout', domain='cloudops', seed=w.seed, identity=actor,
        mode='sandbox', split='heldout', frozen_world=w.model_dump(mode='json'))
    t = ServiceTaskProfile(task_ref='heldout.task', config_id=p.config_id, scenario=task.family, frozen_task=task.model_dump(mode='json'))
    replies = [answer(calls=[call('w' + str(i), op.name, service_arguments('cloudops', op.model_dump(mode='json')))])
        for i, op in enumerate(task.witnesses[0].operations)] if kind == 'completion' else []
    replies.append(answer('Completed the exact authorized business request.' if kind == 'completion'
        else 'I cannot perform this operation: ' + task.refusal_code))
    adapter = UsamiAdapter(tmp_path / 'native', (p,), (t,), lambda spec, *_: ScriptedClient(spec, replies))
    c, s = libraries('cloudops')
    config = adapter.config_for(p.config_id, SPEC, SPEC, experiment.TASK, task_ref=t.task_ref)
    evidence = adapter.run_service_task(config, t.task_ref, c, s, 'heldout-fixture')
    assert evidence.config.partition == 'heldout' and evidence.replay.verified
    row = experiment.scored(evidence, 0, 'S0', t.model_dump(mode='json'))
    assert row['correct'], row
    assert not row['successful_business_violations'] and not row['unauthorized_successful_writes']


def test_call_ledger_stops_before_budget_and_never_retries_started_request(tmp_path):
    limits = {'calls': 1, 'tokens': 100_000, 'buckets': {'one': {'calls': 1, 'tokens': 100_000}}}
    ledger = CallLedger(tmp_path, limits)
    calls = []
    inner = SimpleNamespace(settings=None, close=lambda: None,
        complete=lambda request: calls.append(request) or {'usage': {'total_tokens': 10}, 'choices': []})
    client = BudgetClient(ledger, inner, 'service', 'tag', 'one')
    client.complete({'messages': [], 'max_tokens': 32})
    with pytest.raises(BudgetExhausted): client.complete({'messages': [], 'max_tokens': 32})
    assert len(calls) == 1 and ledger.state['buckets']['one']['tokens'] == 10
    assert not ledger.state['pending']


def test_pair_statistics_preserve_worlds_runs_and_identical_snapshot_caveat():
    m = {'heldout_profiles': ['w0', 'w1'], 'analysis': {'seed': 1, 'resamples': 100, 'limitations': 'two clusters'}}
    rows = [dict(repeat=r, service=s, world=w, kind=k, correct=s == 'B', service_hash=s * 64)
        for r in range(2) for s in ('S0', 'A', 'B') for w in m['heldout_profiles'] for k in ('legal', 'refusal')]
    stats = experiment.analyze(m, rows)
    assert stats['comparisons']['B-A']['mean_difference'] == 1
    assert stats['comparisons']['B-A']['adjusted_interval'] == [1, 1]
    for row in rows: row['service_hash'] = 'a' * 64
    assert not experiment.analyze(m, rows)['comparisons']['B-A']['supports_preregistered_gain']


def test_actual_controller_driver_and_all_reflection_static_dynamic_replays(tmp_path):
    from integrations.usami.domains import model_settings
    from test_vast_usami import customer_action, stopped
    root = tmp_path / 'end-to-end'; experiment.prepare(root); m = experiment.pinned(root)
    class FixtureLedger:
        def client(self, spec, role, tag, bucket):
            if role.startswith('reflection_'):
                def reflect(request):
                    value = json.loads(request['messages'][1]['content'])
                    skill = copy.deepcopy(value['library']['skills'][0]); skill.pop('content_hash')
                    skill['version'] += 1
                    skill['description'] = 'Explicitly labelled CI fixture; reusable permitted decision boundaries.'
                    patch = dict(action='UPDATE', role=value['role'], base_snapshot_hash=value['library']['snapshot_hash'],
                        skill=skill, rationale='One deterministic plumbing fixture, never an online research result.')
                    return answer(json.dumps(patch))
                return SimpleNamespace(settings=model_settings(spec), close=lambda: None, complete=reflect)
            if role == 'customer':
                replies = [customer_action('cloudops', 'ask', 'Please summarize visible state without writes.'), stopped('cloudops')]
            else:
                task = next((t['frozen_task'] for t in m['service_tasks'] if t['task_ref'] in tag), None)
                if task and task['kind'] == 'completion':
                    replies = [answer(calls=[call('op' + str(i), op['name'], service_arguments('cloudops', op))])
                        for i, op in enumerate(task['witnesses'][0]['operations'])] + [answer('Completed the authorized task.')]
                elif task: replies = [answer('I cannot do this: ' + task['refusal_code'])]
                else: replies = [answer('No business write requested.')]
            return ScriptedClient(spec, replies)
    result = experiment.one_run(root, m, FixtureLedger(), 'fixture-B', 'B', m['smoke_profiles'], 1, 'smoke')
    assert result['report']['status'] == 'completed', result
    assert [r['outcome'] for r in result['evolution']] == ['accepted', 'accepted']
    audit = experiment.replay(root, m)
    assert audit['status'] == 'PASS' and audit['reflection_calls_reproduced'] == 2
    assert audit['service_gates_reproduced'] == 1 and audit['online_calls'] == 0


def test_small_protocol_matches_authorized_scale_and_preserves_rules(tmp_path):
    root = tmp_path / 'small'; experiment.prepare(root, small=True); m = experiment.pinned(root)
    assert (m['repetitions'], m['generations'], m['max_workers'], m['heldout_jobs']) == (1, 2, 4, 72)
    assert len(m['heldout_profiles']) == 12
    assert m['customer_selection'] == 'customer-static-only-v1'
    assert m['service_gate_policy'] == {} and m['customer_diagnostics'] is False
    assert m['amendment']['prompt_policy'].startswith('unchanged')
    assert m['budget']['tokens'] == sum(v['tokens'] for v in m['budget']['buckets'].values())
    assert m['amendment']['cumulative_token_cap'] < 25_000_000
    rows = [dict(repeat=0, service=s, world=w, kind=k, correct=s == 'B', service_hash=s * 64)
        for s in ('S0', 'A', 'B') for w in m['heldout_profiles'] for k in ('legal', 'refusal')]
    result = experiment.analyze(m, rows)
    assert result['independent_training_runs'] == 1
    assert result['comparisons']['B-A']['mean_difference'] == 1
    assert not result['comparisons']['B-A']['supports_preregistered_gain']
