from types import SimpleNamespace as NS

import pytest

from rift.runtime.diagnose_episodes import classify_episode, episode_metrics


def episode(status, reason="", *, calls=(), tools=(), success=None):
    return NS(status=status, terminal_reason=reason, model_calls=calls,
              tool_executions=tools, outcome=NS(business_goal_success=success), events=())


def call(response=None, completion_tokens=None):
    return NS(role="service", response=response or {}, completion_tokens=completion_tokens)


@pytest.mark.parametrize(('record', 'expected'), [
    (episode('agent_error', 'ServiceError: invalid', calls=(call({'choices': [
        {'message': {'tool_calls': [{}, {}]}}]}),)), 'multiple_tool_calls'),
    (episode('agent_error', 'Service reply references an object it never observed'), 'unobserved_object'),
    (episode('agent_error', 'Service tool arguments do not match the native schema'), 'schema_violation'),
    (episode('budget_exhausted', 'service_tool_call_budget_exhausted'), 'budget_exhausted'),
    (episode('customer_finished', success=False, tools=(NS(result={'success': False}),)),
     'finished_after_rejected_operation'),
    (episode('customer_finished', success=False), 'customer_finished_unsuccessful'),
    (episode('customer_finished', success=True), 'success'),
])
def test_classify_episode(record, expected):
    assert classify_episode(record) == expected


def test_episode_metrics_only_reports_recorded_lengths():
    record = episode('customer_finished', calls=(call({'generated_token_ids': [1, 2]}),
        call(completion_tokens=5), call()), tools=(NS(result={'success': True}),), success=True)
    assert episode_metrics(record) == {'category': 'success', 'tool_calls': 1,
                                       'response_token_lengths': [2, 5]}


def test_confirmation_order_is_diagnostic_only_and_ignores_read_only_tools():
    from rift.runtime.diagnose_episodes import confirmation_order_compliance
    def event(kind, before='a', after='a', **payload):
        return NS(kind=kind, before_state_hash=before, after_state_hash=after, payload=payload)
    record = episode('customer_finished', success=True)
    record.events = (event('service_tool'), event('service_reply'),
        event('customer_transition', transition_kind='decision'), event('service_tool', 'a', 'b'))
    assert confirmation_order_compliance(record) is True
    record.events = (event('service_tool', 'a', 'b'), event('service_reply'))
    assert confirmation_order_compliance(record) is False
    record.events = (event('service_reply'), event('customer_transition', transition_kind='decision'))
    assert confirmation_order_compliance(record) is None


def test_actor_backend_records_gradient_norms_when_verl_exposes_them():
    from rift.verl.actor_backend import VerlGRPOBackend

    metrics = []
    backend = object.__new__(VerlGRPOBackend)
    backend.role = 'service'
    backend.round_index = 4
    backend.metric_sink = lambda name, value, **fields: metrics.append((name, value, fields))
    backend._report_gradient_norms({'actor/grad_norm': 1.25, 'loss': .5,
                                   'actor/grad_norm_invalid': float('nan')})
    assert metrics == [('service_gradient_norm', 1.25,
                        {'round_index': 4, 'role': 'service', 'metric_name': 'actor/grad_norm'})]


def test_collector_emits_group_metrics_and_returns_recorded_package(monkeypatch):
    from rift.runtime import evaluator
    from rift.verl.sampling import rollout_seed
    from hashlib import sha256

    from rift.schemas import Checkpoint
    checkpoint = Checkpoint('service', 'r0', 'tokenizer', 'weights')
    group = NS(task_ref='task', group_id='group', group_size=2, source='designer')
    package = NS(package_id='package', task_groups=(group,), sampling_seed=3)
    episodes = {}
    for member in range(2):
        rollout_id = sha256(f'rift-service-rollout-v1:package:task:{member}'.encode()).hexdigest()
        request = {'rift_mode': 'train', 'task_ref': 'task', 'rollout_id': rollout_id,
                   'group_id': 'group', 'group_size': 2,
                   'sampling_seed': rollout_seed(3, rollout_id),
                   'customer_seed': rollout_seed(3, 'task')}
        model_call = NS(role='service', request=request, response={'generated_token_ids': [1, 2]},
                        completion_tokens=None)
        record = episode('customer_finished', calls=(model_call,), success=bool(member))
        record.task = NS(record_hash='task')
        record.record_hash = f'episode-{member}'
        episodes[rollout_id] = record
    store = NS(resolve_rollout=lambda key, cp: episodes[key],
               verify_trajectory=lambda record, cp: True, resolve_episode=lambda _: None)
    batch = NS(generate=lambda _: pytest.fail('all rollouts were recovered'))
    request = NS(rollouts=(NS(group_id='group', reward=0), NS(group_id='group', reward=1)))
    recorded = object()
    monkeypatch.setattr(evaluator, 'prepare_package_service_update',
                        lambda *args: (request, recorded))
    metrics = []
    collector = evaluator.ProductionServiceRolloutCollector(batch, task_registry=object(),
        capture_store=store, metric_sink=lambda name, value, **fields: metrics.append((name, value, fields)),
        round_index=4)
    assert collector.collect(package, checkpoint) == (request, recorded)
    group_metric = next(row for row in metrics if row[0] == 'service_group_successes')
    assert group_metric[1] == 1
    assert group_metric[2]['source'] == 'designer' and group_metric[2]['informative'] is True
    assert next(row for row in metrics if row[0] == 'service_uniform_group_fraction')[1] == 0
