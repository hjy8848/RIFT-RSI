import pytest

from rift.benchmark.families import FAMILIES, synthetic_distribution
from rift.benchmark.usami import CommerceHost
from rift.replay.splits import SplitRegistry
from rift.schemas import INSTANCE_DISJOINT_WITHIN_FAMILY_V1
from usami_v2.runtime.runner import EpisodeBudget
from usami_v2.scenarios.task_certification import ProofBudget


def host_for(instance):
    distribution = synthetic_distribution()
    registry = SplitRegistry(policy_version=INSTANCE_DISJOINT_WITHIN_FAMILY_V1)
    host = CommerceHost((), distribution, None, None, None, registry=registry,
        identity_factory=instance.identity, proof_budget=ProofBudget(),
        episode_budget=EpisodeBudget(), return_receipt_after_days=(
            instance.knobs.get('receipt_delay_days', 1) if instance.family_id == 'F9' else None))
    return host


@pytest.mark.parametrize('family_id', list(FAMILIES))
def test_family_default_builds_grounds_and_certifies(family_id):
    instance = FAMILIES[family_id].instance({}, 41)
    assert instance == FAMILIES[family_id].instance({}, 41)
    sibling = FAMILIES[family_id].sibling(instance, 42)
    assert sibling.knobs == instance.knobs and sibling.draft.record_hash != instance.draft.record_hash
    host = host_for(instance)
    result = host.certify_training_plan(instance.proposal(), world_seed=instance.world_seed)
    assert result.build.status == 'SAT', result.build
    assert len(result.tasks) == 1, result.challenge_results
    record = host.task_registry.resolve_record(result.tasks[0].record_hash)
    assert record.identity.family.startswith(family_id + ':')
    assert record.certificate.status == 'CERTIFIED'


@pytest.mark.parametrize(('family_id', 'knobs'), [
    ('F3', {'amount_ratio': 1.1, 'refusal': True}),
    ('F4', {'age_days': 31, 'refusal': True}),
    ('F5', {'fulfillment_status': 'shipped', 'refusal': True}),
    ('F8', {'stock': 0, 'refusal': True}),
    ('F8', {'evidence_valid': False, 'refusal': True}),
    ('F6', {'extra_charge_cents': 3000, 'customer_limit_cents': 1000, 'refusal': True}),
])
def test_rule_refusal_family_builds_and_certifies(family_id, knobs):
    instance = FAMILIES[family_id].instance(knobs, 42)
    host = host_for(instance)
    result = host.certify_training_plan(instance.proposal(), world_seed=instance.world_seed)
    assert result.build.status == 'SAT'
    assert len(result.tasks) == 1, result.challenge_results
    record = host.task_registry.resolve_record(result.tasks[0].record_hash)
    assert record.certificate.protocol == 'commerce-refusal-certificate-v1'


def test_customer_decline_requires_recorded_customer_turn_and_replays():
    from rift.runtime.ascend_910b3 import _UsamiCustomer
    from usami_v2.commerce.models import Operation
    from usami_v2.runtime.agents import ServiceAction
    from usami_v2.runtime.runner import replay_episode, run_episode
    instance = FAMILIES['F6'].instance({'extra_charge_cents': 3000,
        'customer_limit_cents': 1000, 'refusal': True}, 83)
    host = host_for(instance)
    result = host.certify_training_plan(instance.proposal(), world_seed=83)
    assert len(result.tasks) == 1, result.challenge_results
    record = host.task_registry.resolve_record(result.tasks[0].record_hash)
    assert record.profile.accepts_proposals is False
    assert record.scenario.maximum_extra_charge_cents == 1000
    assert record.certificate.attempts[0]['kind'] == 'customer_declines'
    class Service:
        uses_model = False
        last_call = None
        def __init__(self): self.count = 0
        def decide(self, prompt):
            self.count += 1
            merchant = next(iter(record.world.merchants))
            line = record.task.refusal_goals[0].line_ids[0]
            if self.count == 1:
                return ServiceAction(kind='operation', operation=Operation(
                    name='list_customer_orders', scope_id=merchant, resource_id=merchant))
            if self.count == 2:
                return ServiceAction(kind='operation', operation=Operation(
                    name='get_order_details', scope_id=merchant,
                    resource_id=record.world.lines[line].order_id))
            return ServiceAction(kind='reply', message='The replacement costs more. Shall I proceed?')
    episode = run_episode(record.world, record.task, record.profile, _UsamiCustomer(1), Service())
    assert episode.outcome.business_goal_success is True
    assert replay_episode(episode).verified


def test_refusal_fails_when_another_unrequested_line_is_settled():
    from datetime import timedelta
    from types import SimpleNamespace
    from usami_harness.models import digest
    from usami_v2.commerce import adapter
    from usami_v2.commerce.models import Delegation, Operation, PREPARATIONS
    from usami_v2.scenarios.tasks import evaluate_state_goals
    instance = FAMILIES['F6'].instance({'extra_charge_cents': 3000,
        'customer_limit_cents': 1000, 'refusal': True}, 84, n_tasks=2)
    host = host_for(instance)
    result = host.certify_training_plan(instance.proposal(n_tasks=2), world_seed=84)
    assert len(result.tasks) == 2, result.challenge_results
    task, other_task = result.tasks
    initial = result.build.world
    target = task.refusal_goals[0].line_ids[0]
    other = other_task.refusal_goals[0].line_ids[0]
    state = initial.clone()
    state.delegations['explicit-test-authority'] = Delegation(id='explicit-test-authority',
        customer_id=task.customer_id, line_ids=(target, other),
        allowed_preparations=tuple(sorted(PREPARATIONS)), expires_at=state.clock + timedelta(days=1),
        source_customer_event_hash=digest('explicit test authority'))
    merchant = state.orders[state.lines[other].order_id].merchant_id
    operation = Operation(name='return_delivered_order_items', scope_id=merchant,
                          resource_id=other, params={'quantity': 1})
    final, outcome = adapter.apply(state, task.customer_id, operation, 0)
    assert outcome.success
    read = SimpleNamespace(operation={'name': 'get_order_details',
        'resource_id': initial.lines[target].order_id}, result={'success': True})
    declined = SimpleNamespace(kind='customer_transition', payload={'transition_kind': 'decision',
        'decision': {'declines_proposal': True}})
    results = evaluate_state_goals(initial, final, task, tool_executions=(read,),
                                   episode_status='customer_finished', events=(declined,))
    assert results[0].status == 'FAIL'


@pytest.mark.parametrize(('family_id', 'knobs'), [
    ('F1', {'distractor_orders': 3}),
    ('F2', {'purchased_quantity': 5, 'refund_quantity': 2}),
    ('F3', {'amount_ratio': 1}),
    ('F4', {'age_days': 30}),
    ('F5', {'fulfillment_status': 'delivered', 'refusal': True}),
    ('F6', {'extra_charge_cents': 500, 'customer_limit_cents': 500}),
    ('F7', {'fast_stock': 1, 'deadline_days': 2}),
    ('F8', {'stock': 1}),
    ('F9', {'receipt_delay_days': 3}),
    ('F10', {'distractor_lines': 3}),
    ('F11', {'target_count': 3, 'order_count': 2}),
])
@pytest.mark.parametrize('seed', [13, 27])
def test_family_knob_levels_are_certified_across_seeds(family_id, knobs, seed):
    instance = FAMILIES[family_id].instance(knobs, seed)
    host = host_for(instance)
    result = host.certify_training_plan(instance.proposal(), world_seed=seed)
    assert result.build.status == 'SAT'
    assert len(result.tasks) == 1, result.challenge_results


@pytest.mark.parametrize('family_id', list(FAMILIES))
def test_default_family_builds_a_full_distinct_training_package(family_id):
    instance = FAMILIES[family_id].instance({}, 17, n_tasks=8)
    host = host_for(instance)
    result = host.certify_training_plan(instance.proposal(n_tasks=8), world_seed=17)
    assert result.build.status == 'SAT'
    assert len(result.tasks) == 8, result.challenge_results
    targets = []
    for task in result.tasks:
        if task.remedy_goals:
            targets.append(tuple(goal.line_id for goal in task.remedy_goals))
        elif task.cancellation_goals:
            targets.append(tuple(goal.order_id for goal in task.cancellation_goals))
        else:
            targets.append(tuple(line for goal in task.refusal_goals for line in goal.line_ids))
    assert len(set(targets)) == 8


def test_siblings_keep_capability_but_change_visible_product_instance():
    from math import inf
    from rift.probes.fingerprint import fingerprint_distance, task_fingerprint
    family = FAMILIES['F1']
    original = family.instance({'distractor_orders': 1}, 31)
    sibling = family.sibling(original, 32)
    records = []
    for instance in (original, sibling):
        host = host_for(instance)
        result = host.certify_training_plan(instance.proposal(), world_seed=instance.world_seed)
        assert len(result.tasks) == 1
        records.append(host.task_registry.resolve_record(result.tasks[0].record_hash))
    assert records[0].identity.family == records[1].identity.family
    assert records[0].identity.lineage != records[1].identity.lineage
    assert fingerprint_distance(*(task_fingerprint(row.world, row.task) for row in records)) == inf
