import dataclasses

import pytest

from rift.benchmark.families import FAMILIES, synthetic_distribution
from rift.benchmark.usami import CommerceHost
from rift.replay.splits import SplitRegistry
from rift.runtime.ascend_910b3 import _UsamiCustomer
from rift.schemas import INSTANCE_DISJOINT_WITHIN_FAMILY_V1
from usami_v2.commerce.models import Operation
from usami_v2.runtime.agents import ServiceAction
from usami_v2.runtime.customer import with_patience
from usami_v2.runtime.runner import PATIENCE_WARNING, EpisodeBudget, replay_episode, run_episode
from usami_v2.scenarios.refusal_certification import minimum_service_decisions
from usami_v2.scenarios.task_certification import ProofBudget


def certified_refund(patience_slack):
    instance = FAMILIES['F1'].instance({}, 41)
    host = CommerceHost((), synthetic_distribution(), None, None, None,
        registry=SplitRegistry(policy_version=INSTANCE_DISJOINT_WITHIN_FAMILY_V1),
        identity_factory=instance.identity, proof_budget=ProofBudget(),
        episode_budget=EpisodeBudget(), patience_slack=patience_slack)
    result = host.certify_training_plan(instance.proposal(), world_seed=instance.world_seed)
    assert len(result.tasks) == 1, result.challenge_results
    return host, host.task_registry.resolve_record(result.tasks[0].record_hash)


class ScriptedService:
    """Replays a fixed plan of Service actions and records every prompt it saw."""
    uses_model = False
    last_call = None

    def __init__(self, record, plan):
        self.record, self.plan, self.prompts = record, list(plan), []

    def decide(self, prompt):
        self.prompts.append(prompt)
        world, merchant = self.record.world, next(iter(self.record.world.merchants))
        line = self.record.task.remedy_goals[0].line_id
        step = self.plan.pop(0) if self.plan else 'list'
        if step == 'list':
            return ServiceAction(kind='operation', operation=Operation(
                name='list_customer_orders', scope_id=merchant, resource_id=merchant))
        if step == 'details':
            return ServiceAction(kind='operation', operation=Operation(
                name='get_order_details', scope_id=merchant, resource_id=world.lines[line].order_id))
        if step == 'refund':
            return ServiceAction(kind='operation', operation=Operation(
                name='return_delivered_order_items', scope_id=merchant, resource_id=line,
                params={'quantity': self.record.task.remedy_goals[0].quantity, 'payment_method': 'original'}))
        return ServiceAction(kind='reply', message=step)


def test_patience_is_off_by_default_and_profile_hash_is_unchanged():
    _, legacy = certified_refund(None)
    _, patient = certified_refund(0)
    assert legacy.profile.patience_service_decisions is None
    assert 'patience_service_decisions' not in legacy.profile.model_dump(mode='json')
    assert patient.profile.patience_service_decisions == minimum_service_decisions(
        patient.world, patient.task, patient.certificate)


def test_minimum_path_succeeds_within_exact_patience():
    _, record = certified_refund(0)
    plan = ['list', 'details', 'I will refund it now, is that okay?', 'refund', 'Your refund is done.']
    assert record.profile.patience_service_decisions == len(plan)
    episode = run_episode(record.world, record.task, record.profile, _UsamiCustomer(1),
                          ScriptedService(record, plan))
    assert episode.status == 'customer_finished'
    assert episode.outcome.business_goal_success is True
    assert replay_episode(episode).verified


def test_customer_leaves_when_patience_runs_out():
    _, record = certified_refund(0)
    service = ScriptedService(record, ['list'] * 20)
    episode = run_episode(record.world, record.task, record.profile, _UsamiCustomer(1), service)
    assert episode.status == 'customer_left'
    assert episode.terminal_reason == 'customer_patience_exhausted'
    assert len(service.prompts) == record.profile.patience_service_decisions
    assert episode.outcome.business_goal_success is not True
    assert replay_episode(episode).verified


def test_customer_warns_once_before_leaving():
    _, record = certified_refund(0)
    profile = with_patience(record.profile, 3)
    service = ScriptedService(record, ['Let me look into this for you.', 'list', 'list', 'list'])
    episode = run_episode(record.world, record.task, profile, _UsamiCustomer(1), service)
    assert episode.status == 'customer_left'
    seen = [message.content for message in service.prompts[-1].conversation if message.role == 'user']
    assert seen.count(PATIENCE_WARNING) == 1


def test_registry_rejects_patience_below_certified_minimum():
    host, record = certified_refund(0)
    impatient = with_patience(record.profile, record.profile.patience_service_decisions - 1)
    assert host.task_registry._check(record) is True
    assert host.task_registry._check(dataclasses.replace(record, profile=impatient)) is False
