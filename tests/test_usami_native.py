"""CPU only: native Usami fixtures exercise actual certification and engine/replay."""
import sys

from rift_usami_fixtures import (
    NoConsentSandboxService, RejectedThenRefundService, ScriptedCustomer,
    ScriptedRefundService, customer_scenario, distribution_and_slot, frozen,
    frozen_world, make_task, refund_profile, refund_task, scenario_draft, scene,
)
from usami_v2.scenarios.task_certification import ProofBudget
from usami_v2.runtime.runner import EpisodeBudget
from rift.benchmark.usami import certify, run_episode, public_episode
from rift.schemas import Outcome
from rift.tasks import CertificationStatus

def test_native_certification():
    world, aliases = frozen_world()
    task = make_task(world, aliases, remedy=('refund',))
    result = certify('proposal', world, task, customer_scenario(aliases), ProofBudget(max_plan_candidates=16, max_operation_steps=16, max_witnesses=2))
    assert result.status is CertificationStatus.CERTIFIED and result.task_ref == task.record_hash
    assert set(result.__dict__) == {'proposal_id', 'status', 'task_ref'}

def test_designer_diagnostic_removal_preserves_frozen_usami_results():
    from usami_v2.scenarios.task_certification import certify_task, verify_task_certificate
    from usami_v2.runtime.runner import run_episode, replay_episode
    from rift.benchmark.private_evaluation import public_business_outcome
    from rift.benchmark.public_feedback import service_reward

    world, aliases = frozen_world(13)
    task = make_task(world, aliases, remedy=('refund',))
    scenario = customer_scenario(aliases)
    certificate = certify_task(world, task, scenario,
        ProofBudget(max_plan_candidates=16, max_operation_steps=16, max_witnesses=2),
        minimum_witnesses=2)
    assert world.state_hash == '136651c71bc21af5ffb39fce47eba0a6d438078faaf9338087296a8f3c50a21c'
    assert task.record_hash == 'a265fdaa9318b247b15cf916a386146fcba7f39610182c2734092f09c9d7a70c'
    assert certificate.record_hash == 'a9d581d3f85eb2e1d56866380a19b41b8453b6bc06036fbcf25f1b1017ee81ad'
    assert certificate.status == 'CERTIFIED'
    assert verify_task_certificate(world, task, scenario, certificate)['verified']

    episode_world, episode_aliases = frozen(17)
    episode_task = refund_task(episode_world, episode_aliases)
    profile = refund_profile(episode_aliases)
    episode = run_episode(episode_world, episode_task, profile, ScriptedCustomer(),
        ScriptedRefundService(episode_world, episode_aliases), budget=EpisodeBudget(
            max_customer_turns=3, max_service_decisions=10, max_service_tool_calls=8,
            max_model_calls=0))
    replay = replay_episode(episode)
    assert episode_world.state_hash == '2b76cf04eb0c03c1f43284e5431a66cef43ac05dea76de45b790e031cc2fb238'
    assert episode_task.record_hash == '4a8f7ee4e3b7cb89382145b1554d8b4f7f79db3d26d3dbcaf34e3b4b6aa44861'
    assert episode.record_hash == 'c9211ae1c07b72c0be47cd0ad9bc9b938596aa5bb63d7d1da26429f7d8310584'
    assert episode.outcome.record_hash == '0ff86d2ee98792a35102bcd115ad1d69c9b9fc1df97af80dbc572517cd0d6c0e'
    assert replay.verified and episode.outcome.business_goal_success is True
    assert public_business_outcome(episode, replay.verified) is Outcome.SUCCESS
    assert service_reward(Outcome.SUCCESS) == 1.0

def test_native_episode_execution_replay_and_projection():
    world, aliases = frozen()
    episode = run_episode(world, refund_task(world, aliases), refund_profile(aliases), ScriptedCustomer(), ScriptedRefundService(world, aliases), budget=EpisodeBudget(max_customer_turns=3,max_service_decisions=10,max_service_tool_calls=8,max_model_calls=0))
    public = public_episode(episode)
    assert public.outcome is Outcome.SUCCESS
    assert any(row.kind == 'tool_result' for row in public.events)
    assert 'oracle' not in repr(public).lower()

def test_designer_receives_only_native_projected_training_episode():
    from rift.agents import TrainingExperience, propose
    from rift.replay.splits import SplitRegistry
    from rift.schemas import Identity, Split
    from rift.tasks import Proposal
    world, aliases = frozen()
    record = run_episode(world, refund_task(world, aliases), refund_profile(aliases), ScriptedCustomer(), ScriptedRefundService(world, aliases), budget=EpisodeBudget(max_customer_turns=3,max_service_decisions=10,max_service_tool_calls=8,max_model_calls=0))
    identity = Identity(world.state_hash, record.task.record_hash, 'family', 'lineage', Split.TRAIN)
    class Designer:
        def propose(self, episodes):
            assert len(episodes) == 1 and episodes[0].outcome is Outcome.SUCCESS
            assert 'oracle' not in repr(episodes).lower()
            assert not hasattr(episodes[0], 'task') and not hasattr(episodes[0], 'final_world')
            return [Proposal('p', 'hypothesis', 'request', '{}')]
    assert propose(Designer(), [TrainingExperience(identity, record)], SplitRegistry([identity]))[0].proposal_id == 'p'

def test_native_business_success_does_not_use_privileged_strict_score():
    from rift.benchmark.public_feedback import service_reward
    world, aliases = frozen()
    record = run_episode(world, refund_task(world, aliases), refund_profile(aliases), ScriptedCustomer(),
        NoConsentSandboxService(world, aliases), mode='sandbox',
        budget=EpisodeBudget(max_customer_turns=2,max_service_decisions=8,max_service_tool_calls=8,max_model_calls=0))
    assert record.outcome.business_goal_success is True and record.outcome.strict_task_success is False
    projected = public_episode(record)
    assert projected.outcome is Outcome.SUCCESS and service_reward(projected.outcome) == 1.0
    assert 'oracle' not in repr(projected).lower() and 'strict_task_success' not in repr(projected)

def test_native_public_tool_error_survives_without_oracle_fields():
    import json
    world, aliases = frozen()
    record = run_episode(world, refund_task(world, aliases), refund_profile(aliases), ScriptedCustomer(),
        RejectedThenRefundService(world, aliases),
        budget=EpisodeBudget(max_customer_turns=3,max_service_decisions=12,max_service_tool_calls=10,max_model_calls=0))
    failed = next(tool.result for tool in record.tool_executions if not tool.result['success'])
    observed = [json.loads(row.content) for row in public_episode(record).events if row.kind == 'tool_result']
    assert failed in observed
    assert 'oracle' not in repr(observed).lower()

def test_native_service_prompts_exclude_private_state_and_future_results():
    import json
    world, aliases = frozen()
    class PromptRecorder:
        uses_model = False
        last_call = None
        def __init__(self):
            self.base = ScriptedRefundService(world, aliases)
            self.prompts = []
        def decide(self, prompt):
            self.prompts.append(prompt.model_dump(mode='json'))
            return self.base.decide(prompt)
    service = PromptRecorder()
    record = run_episode(world, refund_task(world, aliases), refund_profile(aliases), ScriptedCustomer(), service,
        budget=EpisodeBudget(max_customer_turns=3,max_service_decisions=10,max_service_tool_calls=8,max_model_calls=0))
    assert public_episode(record).outcome is Outcome.SUCCESS
    first = json.dumps(service.prompts[0], ensure_ascii=False)
    assert service.base.quote not in first
    assert not any(row.get('role') == 'tool' for row in service.prompts[0]['conversation'])
    assert any(row.get('role') == 'tool' for row in service.prompts[-1]['conversation'])
    for prompt in service.prompts:
        assert set(prompt) == {'customer_id', 'customer_request', 'conversation', 'available_merchant_scope_ids', 'observed_subject_ids', 'record_hash'}
        assert not any(key in prompt for key in ('task', 'profile', 'oracle', 'final_world', 'outcome'))

def test_native_grounding():
    import json
    from usami_v2.scenarios.builder import build_world
    from rift.tasks import Proposal
    from rift.benchmark.usami import ground
    spec = scene(); build = build_world(spec, 13)
    data = {'task_id':'grounded-task', 'customer_alias':'customer', 'hypothesis':'preconditions',
        'remedy_goals':[{'goal_id':'g','line_alias':'item','quantity':1,'allowed_remedies':['refund']}],
        'line_selectors':[{'alias':'item','family':'device','issue':'damaged'}]}
    proposal = Proposal('p', 'preconditions', 'Please help', json.dumps(data))
    result = ground(proposal, [('world', spec, build)])
    assert result.status == 'GROUNDED' and result.candidates
    assert result.candidates[0].world_hash == build.world_hash

def test_native_ground_compile_certify_chain():
    import json
    from usami_v2.scenarios.builder import build_world
    from usami_v2.scenarios.task_certification import CustomerScenario
    from rift.tasks import Proposal
    from rift.benchmark.usami import ground, compile_candidate
    distribution, _ = distribution_and_slot(13)
    spec = scenario_draft().compile(distribution); build = build_world(spec, 13)
    data = {'task_id':'grounded-task','customer_alias':'customer','hypothesis':'missing precondition',
        'remedy_goals':[{'goal_id':'g','line_alias':'item','quantity':1,'allowed_remedies':['refund']}],
        'line_selectors':[{'alias':'item','family':'device','issue':'damaged'}]}
    proposal = Proposal('p', 'missing precondition', 'Please refund', json.dumps(data))
    candidates = ground(proposal, [('world', spec, build)]).candidates
    task = compile_candidate(candidates[0], spec, build, distribution)
    scenario = CustomerScenario(customer_id=task.customer_id, delegated_line_ids=tuple(g.line_id for g in task.remedy_goals), maximum_extra_charge_cents=0)
    result = certify(proposal.proposal_id, build.world, task, scenario, ProofBudget())
    assert result.status is CertificationStatus.CERTIFIED and result.task_ref == task.record_hash

def test_scripted_native_episode_cannot_be_replay_demonstration():
    from rift.replay.trajectory import extract_demonstrations, DecisionEvidenceVerifier
    from rift.replay.splits import SplitRegistry
    from rift.schemas import Identity, Split, Checkpoint
    import pytest
    world, aliases = frozen()
    record = run_episode(world, refund_task(world, aliases), refund_profile(aliases), ScriptedCustomer(), ScriptedRefundService(world, aliases), budget=EpisodeBudget(max_customer_turns=3,max_service_decisions=10,max_service_tool_calls=8,max_model_calls=0))
    identity = Identity(world.state_hash, record.task.record_hash, 'f', 'source', Split.REPLAY)
    class Verifier(DecisionEvidenceVerifier):
        def verify_source(self, episode, checkpoint): return True
    with pytest.raises(ValueError, match='real Service model'):
        extract_demonstrations(record, identity, Checkpoint('service','r','t','w'), SplitRegistry([identity]), Verifier())

def test_scripted_episode_cannot_generate_service_training_data():
    from rift.training.service_data import prepare_service_update
    from rift.schemas import Checkpoint
    import pytest
    world, aliases = frozen()
    record = run_episode(world, refund_task(world, aliases), refund_profile(aliases), ScriptedCustomer(), ScriptedRefundService(world, aliases), budget=EpisodeBudget(max_customer_turns=3,max_service_decisions=10,max_service_tool_calls=8,max_model_calls=0))
    class Recorder:
        def verify_source(self, episode, checkpoint): return True
    with pytest.raises(ValueError, match='real Service model'):
        prepare_service_update('package', {record.task.record_hash}, [record], Checkpoint('service','r','t','w'), Recorder())
    with pytest.raises(ValueError, match='duplicate Service episode'):
        prepare_service_update('package', {record.task.record_hash}, [record, record], Checkpoint('service','r','t','w'), Recorder())

def test_native_commerce_host_certification():
    import json
    from usami_v2.scenarios.builder import build_world
    from rift.tasks import Proposal
    from rift.benchmark.usami import CommerceHost
    distribution, _ = distribution_and_slot(13)
    spec = scenario_draft().compile(distribution); build = build_world(spec, 13)
    data = {'task_id':'grounded-task','customer_alias':'customer','hypothesis':'missing precondition',
        'remedy_goals':[{'goal_id':'g','line_alias':'item','quantity':1,'allowed_remedies':['refund']}],
        'line_selectors':[{'alias':'item','family':'device','issue':'damaged'}]}
    from rift.replay.splits import SplitRegistry
    from rift.schemas import Identity, Split
    registry = SplitRegistry()
    def identity(task, spec, build):
        return Identity(build.world_hash, task.record_hash, 'family', 'source', Split.TRAIN)
    host = CommerceHost([('world', spec, build)], distribution, None, None, None, registry=registry, identity_factory=identity, proof_budget=ProofBudget(), episode_budget=EpisodeBudget())
    cert, task = host.certify_proposal(Proposal('p','missing precondition','refund',json.dumps(data)))
    assert cert.status is CertificationStatus.CERTIFIED and cert.task_ref == task.record_hash
    assert registry.contains(identity(task, spec, build))
    assert 'rsi_core' not in sys.modules
    task_registry = host.task_registry
    w, t, p, _ = task_registry.resolve(task.record_hash)
    assert task_registry.verify_task(task.record_hash, w, t, p)
    import pytest
    for conflicting in (
        Identity(build.world_hash, 'other-task', 'other-family', 'other-source', Split.REPLAY),
        Identity('other-world', 'other-task', 'family', 'other-source', Split.TEST),
        Identity('other-world', 'other-task', 'other-family', 'source', Split.VALIDATION),
    ):
        blocked_registry = SplitRegistry([conflicting])
        blocked = CommerceHost([('world', spec, build)], distribution, None, None, None,
            registry=blocked_registry, identity_factory=identity,
            proof_budget=ProofBudget(), episode_budget=EpisodeBudget())
        with pytest.raises(ValueError, match='cross-split'):
            blocked.certify_proposal(Proposal('p','missing precondition','refund',json.dumps(data)))

def test_default_host_ground_certify_and_scripted_rollout():
    import json
    from usami_v2.scenarios.builder import build_world
    from rift.tasks import Proposal
    from rift.benchmark.usami import CommerceHost
    from rift.replay.splits import SplitRegistry
    from rift.schemas import Identity, Split
    distribution, _ = distribution_and_slot(13)
    spec = scenario_draft().compile(distribution); build = build_world(spec, 13)
    data = {'task_id':'grounded-task','customer_alias':'customer','hypothesis':'precondition',
        'remedy_goals':[{'goal_id':'g','line_alias':'item','quantity':1,'allowed_remedies':['refund']}],
        'line_selectors':[{'alias':'item','family':'device','issue':'damaged'}]}
    def identity(task, spec, build):
        return Identity(build.world_hash, task.record_hash, 'family', 'lineage', Split.TRAIN)
    host = CommerceHost([('world', spec, build)], distribution, None, None, ScriptedCustomer(),
        registry=SplitRegistry(), identity_factory=identity, proof_budget=ProofBudget(),
        episode_budget=EpisodeBudget(max_customer_turns=3, max_service_decisions=10,
            max_service_tool_calls=8, max_model_calls=0))
    cert, task = host.certify_proposal(Proposal('p','precondition','refund',json.dumps(data)))
    assert cert.status is CertificationStatus.CERTIFIED
    record = host.rollout(task, ScriptedRefundService(build.world, dict(build.private_aliases)))
    assert public_episode(record).outcome is Outcome.SUCCESS

def test_host_certifies_next_grounding_candidate_after_first_rejection(monkeypatch):
    from types import SimpleNamespace
    from usami_v2.scenarios.builder import build_world
    from usami_v2.scenarios.designer import TaskDraft, compile_task
    from rift import benchmark
    from rift.benchmark.usami import CommerceHost
    from rift.replay.splits import SplitRegistry
    from rift.schemas import Identity, Split
    from rift.tasks import CertificationStatus, Proposal
    import usami_v2.scenarios.task_certification as task_certification
    distribution, _ = distribution_and_slot(13)
    spec = scenario_draft().compile(distribution)
    build = build_world(spec, 13)
    task_a = compile_task(TaskDraft(task_id='candidate-a', customer_alias='customer', remedy_goals=(
        {'goal_id':'g','line_alias':'damaged','quantity':1,'allowed_remedies':['refund']},)),
        spec, build, distribution)
    task_b = compile_task(TaskDraft(task_id='candidate-b', customer_alias='customer', remedy_goals=(
        {'goal_id':'g','line_alias':'damaged','quantity':1,'allowed_remedies':['refund']},)),
        spec, build, distribution)
    candidate_a = SimpleNamespace(key='a', world_ref='world', world_hash=build.world_hash)
    candidate_b = SimpleNamespace(key='b', world_ref='world', world_hash=build.world_hash)
    monkeypatch.setattr(benchmark.usami, 'ground', lambda *args: SimpleNamespace(candidates=(candidate_a, candidate_b)))
    monkeypatch.setattr(benchmark.usami, 'compile_candidate',
        lambda candidate, *args: task_a if candidate.key == 'a' else task_b)
    native_certify = benchmark.usami.certify_native
    attempted = []
    def certify_candidate(proposal_id, world, task, scenario, budget):
        attempted.append(task.task_id)
        if task == task_a:
            return SimpleNamespace(status='UNSAT',
                full_task_reachability_certified=False, blockers=())
        return native_certify(proposal_id, world, task, scenario, budget)
    monkeypatch.setattr(benchmark.usami, 'certify_native', certify_candidate)
    verify_certificate = task_certification.verify_task_certificate
    verified_tasks = []
    def verify_candidate(world, task, scenario, certificate):
        if task == task_a:
            return {'verified': True}
        verified_tasks.append(task.record_hash)
        return verify_certificate(world, task, scenario, certificate)
    monkeypatch.setattr(task_certification, 'verify_task_certificate', verify_candidate)
    host = CommerceHost([('world', spec, build)], distribution, None, None, ScriptedCustomer(),
        registry=SplitRegistry(), identity_factory=lambda task, spec, build: Identity(
            build.world_hash, task.record_hash, 'family', 'lineage', Split.TRAIN),
        proof_budget=ProofBudget(), episode_budget=EpisodeBudget())
    result, selected = host.certify_proposal(Proposal('p','preconditions','refund','{}'))
    assert attempted == ['candidate-a', 'candidate-b']
    assert verified_tasks == [task_b.record_hash]
    assert result.status is CertificationStatus.CERTIFIED and selected == task_b
    assert host.task_registry.resolve_record(task_b.record_hash).task == task_b

def test_default_customer_runs_information_only_native_usami_task():
    import json
    from usami_v2.scenarios.builder import build_world
    from usami_v2.scenarios.designer import TaskDraft, compile_task, task_design_capabilities
    from usami_v2.runtime.agents import Operation, ServiceAction
    from usami_v2.runtime.runner import EpisodeBudget
    from usami_v2.scenarios.task_certification import ProofBudget, certify_task
    from rift.benchmark.usami import default_customer_profile, default_customer_scenario
    from rift.schemas import Outcome
    distribution, _ = distribution_and_slot(23)
    spec = scenario_draft().compile(distribution)
    build = build_world(spec, 23)
    capabilities = task_design_capabilities(
        spec, build, interaction_certifier_available=True, information_goals_certifiable=True)
    task = compile_task(TaskDraft(task_id='information-only', customer_alias='customer', information_goals=(
        {'goal_id':'refund-amount','subject_alias':'damaged','fact_kind':'refund_amount'},)),
        spec, build, distribution, capabilities=capabilities)
    scenario = default_customer_scenario(task, spec, build)
    profile = default_customer_profile(task, spec, build)
    assert scenario.accepted_remedies == profile.accepted_remedies == ()
    assert scenario.delegated_line_ids == profile.delegated_line_ids == ()
    mixed = compile_task(TaskDraft(task_id='mixed-remedy-information', customer_alias='customer',
        remedy_goals=({'goal_id':'refund','line_alias':'damaged','quantity':1,
                       'allowed_remedies':['refund']},),
        information_goals=({'goal_id':'refund-amount','subject_alias':'damaged',
                            'fact_kind':'refund_amount'},)),
        spec, build, distribution, capabilities=capabilities)
    mixed_scenario = default_customer_scenario(mixed, spec, build)
    mixed_profile = default_customer_profile(mixed, spec, build)
    assert mixed_scenario.accepted_remedies == mixed_profile.accepted_remedies == ('refund',)
    assert mixed_scenario.delegated_line_ids == mixed_profile.delegated_line_ids == (
        dict(build.private_aliases)['damaged'],)
    # Runtime support does not claim that the pinned Reference certifier proves
    # communication objectives; it explicitly reports that gap as UNKNOWN.
    assert certify_task(build.world, task, scenario, ProofBudget()).status == 'UNKNOWN'

    class InformationService:
        uses_model = False
        last_call = None
        def __init__(self): self.queried = False
        def decide(self, prompt):
            line_id = dict(build.private_aliases)['damaged']
            merchant = prompt.available_merchant_scope_ids[0]
            if not self.queried:
                self.queried = True
                return ServiceAction(kind='operation', operation=Operation(
                    name='get_refundable_balance', scope_id=merchant, resource_id=line_id, params={}))
            result = json.loads(next(row.content for row in reversed(prompt.conversation)
                                     if row.role == 'tool'))
            amount = result['data']['remaining_original_credit_cents']
            return ServiceAction(kind='reply', message=f'{line_id}: refund amount is CNY {amount / 100:.2f}.',
                                 subject_ids=(line_id,))

    episode = run_episode(build.world, task, profile, ScriptedCustomer(), InformationService(),
        budget=EpisodeBudget(max_customer_turns=2, max_service_decisions=4,
            max_service_tool_calls=2, max_model_calls=0))
    assert episode.replay_verified and episode.status == 'customer_finished'
    assert public_episode(episode).outcome is Outcome.SUCCESS

def test_offline_round_reaches_ready_with_explicit_fake_recording():
    """Exercise wiring only; the recorded calls and token verifier are test doubles."""
    import json
    from usami_v2.scenarios.builder import build_world
    from usami_v2.runtime.agents import AgentCall
    from rift.agents import TrainingExperience
    from rift.benchmark.usami import CommerceHost
    from rift.loop import Stage
    from rift.orchestrator import prepare_round
    from rift.replay.splits import SplitRegistry
    from rift.schemas import Checkpoint, Identity, Split, Tokens
    from rift.tasks import Proposal
    old_world, aliases = frozen()
    history = run_episode(old_world, refund_task(old_world, aliases), refund_profile(aliases), ScriptedCustomer(),
        ScriptedRefundService(old_world, aliases), budget=EpisodeBudget(max_customer_turns=3,
            max_service_decisions=10,max_service_tool_calls=8,max_model_calls=0))
    past = Identity(old_world.state_hash, history.task.record_hash, 'past-family', 'past-lineage', Split.TRAIN)
    registry = SplitRegistry([past])
    distribution, _ = distribution_and_slot(13)
    spec = scenario_draft().compile(distribution); build = build_world(spec, 13)
    data = {'task_id':'grounded-task','customer_alias':'customer','hypothesis':'precondition',
        'remedy_goals':[{'goal_id':'g','line_alias':'item','quantity':1,'allowed_remedies':['refund']}],
        'line_selectors':[{'alias':'item','family':'device','issue':'damaged'}]}
    class Designer:
        def propose(self, observations):
            assert observations[0].outcome is Outcome.SUCCESS
            return [Proposal('p', 'precondition', 'refund', json.dumps(data))]
    class RecordedScript:
        uses_model = True
        last_call = None
        def __init__(self): self.base = ScriptedRefundService(build.world, dict(build.private_aliases))
        def decide(self, prompt):
            action = self.base.decide(prompt)
            call = AgentCall(role='service', request={'fixture': True}, response={'fixture': True})
            return action.model_copy(update={'call': call})
    class FakeRecorder:
        def verify_source(self, episode, checkpoint): return True
        def tokens_for_call(self, call, checkpoint): return Tokens((1,), (2,), (True,))
    host = CommerceHost([('world', spec, build)], distribution, None, None, ScriptedCustomer(),
        registry=registry, identity_factory=lambda task, spec, build: Identity(build.world_hash, task.record_hash, 'new-family', 'new-lineage', Split.TRAIN),
        proof_budget=ProofBudget(), episode_budget=EpisodeBudget(max_customer_turns=3,
            max_service_decisions=10,max_service_tool_calls=8,max_model_calls=10))
    episodes = []
    native_rollout = host.rollout
    def capture_episode(task, service):
        record = native_rollout(task, service)
        episodes.append(record)
        return record
    host.rollout = capture_episode
    checkpoint = Checkpoint('service','v1','tokenizer','test-weight-id')
    state = prepare_round('package', [TrainingExperience(past, history)], registry, Designer(), host,
        RecordedScript(), checkpoint, FakeRecorder())
    assert state.stage is Stage.READY_FOR_TRAINING
    assert state.service_update is None and state.service_request.rollouts
    assert all(row.response_id and row.reward_ref == row.episode_id for row in state.service_request.rollouts)
    # The real-model and token-source claims here are intentionally supplied
    # by a fake verifier. No real-model provenance is established by this test.
    from dataclasses import replace
    from rift.replay.trajectory import extract_demonstrations, NativeReplayEvidenceVerifier
    from rift.replay.bank import ReplayBank
    import pytest
    record = episodes[0]
    replay_id = Identity(record.initial_world.state_hash, record.task.record_hash, 'replay-family', 'replay-lineage', Split.REPLAY)
    replay_registry = SplitRegistry([replay_id])
    captured = extract_demonstrations(record, replay_id, checkpoint, replay_registry, FakeRecorder())
    source_verifier = NativeReplayEvidenceVerifier(lambda episode_id: record if episode_id == record.record_hash else None,
        replay_registry, FakeRecorder())
    bank = ReplayBank([captured[0]], verifier=source_verifier)
    assert bank._rows[0] == captured[0]
    altered = replace(captured[0], tokens=Tokens((1,), (99,), (True,)))
    with pytest.raises(ValueError, match='demonstration evidence'):
        ReplayBank([altered], verifier=source_verifier)
