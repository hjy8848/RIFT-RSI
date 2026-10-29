"""Offline native-Usami coverage for the single-World mixed training package."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'third_party' / 'usami' / 'tests'))
from test_usami_v2_designer import distribution_and_slot, scenario_draft
from usami_v2.scenarios.builder import build_world
from usami_v2.scenarios.task_certification import ProofBudget

from rift.benchmark.usami import CommerceHost
from rift.loop import LoopState, PendingDesignerAction
from rift.banks.records import EvidenceLevel, ObservedTrial
from rift.probes.records import ProbeSnapshot, TaskProbe
from rift.replay.splits import SplitRegistry
from rift.schemas import (Checkpoint, DataPurpose, Identity, INSTANCE_DISJOINT_WITHIN_FAMILY_V1,
                          Outcome, Split, Tokens)
from rift.tasks import CertificationStatus, Proposal
from rift.training.contracts import Rollout, UpdateRequest
from rift.training.package import (FrozenBenchmarkTrainingPool, TrainingPackageConfig,
                                   assemble_training_package)
from usami_v2.runtime.runner import EpisodeBudget


def _challenge(task_id, issue='damaged'):
    return {
        'task_id': task_id,
        'customer_alias': 'customer',
        'hypothesis': 'check the evidence before committing a remedy',
        'remedy_goals': [{'goal_id': 'refund', 'line_alias': 'item', 'quantity': 1,
                          'allowed_remedies': ['refund']}],
        'line_selectors': [{'alias': 'item', 'family': 'device', 'issue': issue}],
    }


def _host(distribution, registry, task_registry=None, *, worlds=()):
    return CommerceHost(worlds, distribution, None, None, None,
        registry=registry,
        identity_factory=lambda task, spec, build: Identity(build.world_hash,
            task.record_hash, 'commerce', f'lineage-{task.task_id}', Split.TRAIN,
            DataPurpose.TRAIN),
        proof_budget=ProofBudget(), episode_budget=EpisodeBudget(),
        task_registry=task_registry)


def _native_fixture():
    distribution, _ = distribution_and_slot(73)
    split_registry = SplitRegistry(policy_version=INSTANCE_DISJOINT_WITHIN_FAMILY_V1)
    designer_host = _host(distribution, split_registry)
    plan = Proposal('decision-1', 'evidence before action', ability_id='NEW',
        diagnosis_task_refs=('failure-1', 'failure-2'),
        world_draft=scenario_draft('designer-world').model_dump(mode='json', exclude={'record_hash'}),
        challenges=tuple(_challenge(f'focused-{index}') for index in range(8)))
    designer_world = designer_host.certify_training_plan(plan, world_seed=73)
    assert designer_world.build.status == 'SAT'
    assert designer_world.build_verification['verified'] is True
    assert len(designer_world.tasks) == len(designer_world.certifications) == 8
    assert designer_world.tasks[0].world_hash == designer_world.tasks[1].world_hash

    task_registry = designer_host.task_registry
    benchmark_spec = scenario_draft('benchmark-world').compile(distribution)
    benchmark_build = build_world(benchmark_spec, 91)
    benchmark_host = _host(distribution, split_registry, task_registry,
        worlds=(('benchmark-world', benchmark_spec, benchmark_build),))
    benchmark_refs = []
    for index in range(8):
        task_id = f'general-{index}'
        challenge = _challenge(task_id)
        proposal = Proposal(task_id, challenge['hypothesis'], 'Please refund',
            json.dumps(_challenge(task_id)))
        certification, task = benchmark_host.certify_proposal(proposal)
        assert certification.status is CertificationStatus.CERTIFIED
        assert task is not None
        benchmark_refs.append(task.record_hash)
    pool = FrozenBenchmarkTrainingPool.freeze('frozen-train-v1',
        tuple(benchmark_refs), task_registry)
    config = TrainingPackageConfig(tasks_per_round=4, designer_fraction=.5,
        benchmark_fraction=.5, sampling_seed=109, group_size=2)
    checkpoint = Checkpoint('service', 'step-0', 'tokenizer-v1', 'weights-0')
    return (plan, designer_host, designer_world, task_registry, pool,
            split_registry, config, checkpoint)


def _package(fixture, package_id='package-1'):
    plan, _, result, task_registry, pool, _, config, checkpoint = fixture
    return assemble_training_package(package_id=package_id, proposal=plan,
        designer_action_ref='a' * 32, target_ability_id='A000001',
        designer_result=result, benchmark_pool=pool, task_registry=task_registry,
        service_checkpoint_before=checkpoint, config=config, round_index=3)


@pytest.fixture(scope='module')
def native_fixture():
    return _native_fixture()


def test_native_one_world_multi_challenge_mixed_package_and_state_restore(tmp_path, native_fixture):
    fixture = native_fixture
    plan, host, result, task_registry, pool, split_registry, config, checkpoint = fixture
    package = _package(fixture)
    assert len(result.tasks) == 8
    assert len(package.designer_task_refs) == len(package.benchmark_task_refs) == 2
    assert set(package.designer_task_refs).isdisjoint(package.benchmark_task_refs)
    assert set(package.task_refs) == set(package.designer_task_refs) | set(package.benchmark_task_refs)
    assert package.sampling_seed == config.sampling_seed + 3
    assert package.verify_service_request is not None
    assert all(task_registry.resolve_record(ref).task.partition == 'train'
               for ref in package.task_refs)
    assert host.task_registry is task_registry

    full_config = TrainingPackageConfig(tasks_per_round=16, designer_fraction=.5,
        benchmark_fraction=.5, sampling_seed=211, group_size=4)
    full_package = assemble_training_package(package_id='package-default-8-plus-8',
        proposal=plan, designer_action_ref='d' * 32, target_ability_id='A000001',
        designer_result=result, benchmark_pool=pool, task_registry=task_registry,
        service_checkpoint_before=package.service_checkpoint_before,
        config=full_config, round_index=0)
    assert len(full_package.designer_task_refs) == len(full_package.benchmark_task_refs) == 8
    assert len(full_package.task_refs) == 16
    assert all(task_registry.resolve_record(ref).certificate.status == 'CERTIFIED'
               for ref in full_package.task_refs)

    # Both episodes start from the same frozen World record. Native Usami clones
    # its initial state for each run, so the stored World remains unchanged.
    from test_usami_v2_runtime import ScriptedCustomer, ScriptedRefundService
    from rift.benchmark.usami import public_episode
    host.customer_agent = ScriptedCustomer()
    record_a = task_registry.resolve_record(package.designer_task_refs[0])
    record_b = task_registry.resolve_record(package.designer_task_refs[1])
    initial_hash = record_a.world.state_hash
    episode_a = host.rollout(record_a.task, ScriptedRefundService(
        record_a.world, dict(result.build.private_aliases)))
    episode_b = host.rollout(record_b.task, ScriptedRefundService(
        record_b.world, dict(result.build.private_aliases)))
    assert episode_a.final_world.state_hash == episode_b.final_world.state_hash
    assert task_registry.resolve_record(record_a.task.record_hash).world.state_hash == initial_hash
    assert public_episode(episode_a).outcome.value in {'SUCCESS', 'FAILURE', 'UNKNOWN'}

    action = PendingDesignerAction(Checkpoint('designer', 'step-2', 'tok', 'dweights'),
        package.designer_action_ref, Tokens((1,), (2, 3), (True, True)))
    state = LoopState(package.package_id, (plan.proposal_id,),
        target_ability_id=package.target_ability_id, designer_action=action)
    state = state.certify_training_package(package, result.certifications)
    state_path = tmp_path / 'package-state.json'
    state.save(state_path)
    class ActionVerifier:
        def verify_designer_action(self, *args): return True
    restored = LoopState.load(state_path, action_verifier=ActionVerifier())
    assert restored == state
    assert restored.task_refs == package.task_refs
    assert restored.training_package.distribution_hash == result.distribution_hash


def test_one_on_policy_capture_binds_world_and_all_challenges(native_fixture, tmp_path):
    """Mock attestation checks one response contains the complete package plan."""
    from dataclasses import asdict
    from rift.training.trace import ExactPolicyTrace, digest
    from rift.verl.designer import DesignerCaptureStore, PendingDesignerRollout

    plan = native_fixture[0]
    checkpoint = Checkpoint('designer', 'step-4', 'tokenizer-v1', 'designer-weights')
    rollout_id = 'c' * 32
    prompt_ids, response_ids = (11, 12), (21, 22, 23)
    request = {'prompt_ids': list(prompt_ids), 'package_id': 'decision-package',
        'rollout_id': rollout_id, 'group_id': None, 'visible_failure_refs': ['failure-1', 'failure-2'],
        'ability_options': []}
    response = {'generated_token_ids': list(response_ids), 'log_probs': None,
        'parsed_proposals': [asdict(plan)], 'serving_fields': {}}
    token_trace = Tokens(prompt_ids, response_ids, (True, True, True))

    class MockAttestor:
        def verify_capture(self, *args): return True

    store = DesignerCaptureStore(tmp_path / 'designer-capture', attestor=MockAttestor())
    trace = ExactPolicyTrace('designer', rollout_id, 'decision-1', checkpoint,
        token_trace, digest(request), digest(response), 'native-token-ids-v1',
        str(store._path(rollout_id)), 'decision-package')
    pending = PendingDesignerRollout('decision-package', rollout_id, None,
        checkpoint, (plan,), trace, None)
    store.save(pending, request, response)
    restored = store.load_pending(rollout_id)
    assert restored.proposals == (plan,)
    assert restored.trace.tokens == token_trace
    assert store.verify_designer_proposal(rollout_id, plan,
        visible_failure_refs=('failure-1', 'failure-2'), ability_options=())


def test_seeded_benchmark_mix_stats_and_training_overlap_checks(native_fixture):
    from dataclasses import asdict
    fixture = native_fixture
    package = _package(fixture)
    second = _package(fixture, 'package-2')
    assert package.benchmark_task_refs == second.benchmark_task_refs
    _, _, designer_result, task_registry, pool, _, config, checkpoint = fixture
    too_small = FrozenBenchmarkTrainingPool.freeze('one-task-only',
        pool.task_refs[:1], task_registry)
    with pytest.raises(ValueError, match='Benchmark pool is too small'):
        assemble_training_package(package_id='too-small', proposal=fixture[0],
            designer_action_ref='a' * 32, target_ability_id='A000001',
            designer_result=designer_result, benchmark_pool=too_small,
            task_registry=task_registry, service_checkpoint_before=checkpoint,
            config=config, round_index=3)

    rollouts = []
    for task_index, task_ref in enumerate(package.task_refs):
        group = package.group_by_task_ref[task_ref]
        for member in range(group.group_size):
            rollouts.append(Rollout(Tokens((100 + task_index,), (200 + member,), (True,)),
                package.service_checkpoint_before, task_ref, f'{task_ref}-{member}',
                float(member), 'business_outcome_v2', group.group_id,
                f'{task_ref}-{member}', f'{task_ref}-{member}'))
    request = UpdateRequest(package.service_checkpoint_before, package.package_id,
        tuple(rollouts), 'grpo')
    recorded = package.record_service_request(request)
    assert recorded.stats.rollout_count == 8
    assert recorded.stats.designer_rollout_count == recorded.stats.benchmark_rollout_count == 4
    assert recorded.stats.designer_token_count + recorded.stats.benchmark_token_count == 8
    assert type(recorded).from_dict(asdict(recorded)) == recorded

    source_identity = package.task_sources[0].identity
    probe_identity = replace(source_identity, purpose=DataPurpose.BANK_CANDIDATE)
    snapshot = ProbeSnapshot.create(snapshot_id='overlap',
        purpose=DataPurpose.RECOVERY_FEEDBACK, probes=(
            TaskProbe(
                task_ref=probe_identity.task, identity=probe_identity,
                purpose=DataPurpose.RECOVERY_FEEDBACK,
                baseline_checkpoint=package.service_checkpoint_before,
                baseline_trials=(ObservedTrial(
                    task_ref=probe_identity.task, trial_id='trial',
                    checkpoint=package.service_checkpoint_before, seed=1,
                    outcome=Outcome.FAILURE,
                    protocol_digest='probe-protocol', evidence_ref='evidence',
                    replay_verified=True,
                    evidence_level=EvidenceLevel.REAL_SERVICE),),
                protocol_ref='probe-v1', protocol_digest='probe-protocol'),),
        random_seed=1, split_policy_version=INSTANCE_DISJOINT_WITHIN_FAMILY_V1,
        baseline_checkpoint=package.service_checkpoint_before,
        protocol_digest='probe-protocol', candidate_count=1)
    with pytest.raises(ValueError, match='overlaps'):
        package.assert_disjoint_from(snapshot)


def test_grpo_group_cannot_mix_task_refs_and_frozen_pool_rejects_test_identity():
    cp = Checkpoint('service', 'r1', 'tok', 'weights')
    token_a, token_b = Tokens((1,), (2,), (True,)), Tokens((1,), (3,), (True,))
    with pytest.raises(ValueError, match='mix different TaskRefs'):
        UpdateRequest(cp, 'package', (
            Rollout(token_a, cp, 'train-a', 'episode-a', 1., 'business', 'same-group'),
            Rollout(token_b, cp, 'train-b', 'episode-b', 0., 'business', 'same-group')),
            'grpo')

    class TestRecord:
        identity = Identity('world', 'heldout-task', 'family', 'lineage', Split.TEST,
                            DataPurpose.TEST)
        task = type('Task', (), {'partition':'heldout'})()
        certificate = type('Certificate', (), {'status':'CERTIFIED',
            'full_task_reachability_certified':True, 'record_hash':'certificate'})()
    class TestRegistry:
        def resolve_record(self, ref): return TestRecord()
        def verify_task(self, *args, **kwargs): return True
    with pytest.raises(ValueError, match='certified Usami TRAIN'):
        FrozenBenchmarkTrainingPool.freeze('bad-test-pool', ('heldout-task',), TestRegistry())


def test_unknown_source_groups_may_be_discarded_without_relabeling_reward(native_fixture):
    fixture = native_fixture
    package = _package(fixture)
    rows = []
    for source_refs in (package.designer_task_refs, package.benchmark_task_refs):
        task_ref = source_refs[0]
        group = package.group_by_task_ref[task_ref]
        prompt = (10 + package.task_refs.index(task_ref),)
        rows.extend(Rollout(Tokens(prompt, (20 + i,), (True,)),
            package.service_checkpoint_before, task_ref, f'known-{len(rows)}-{i}', float(i),
            'business_outcome_v2', group.group_id) for i in range(group.group_size))
    partial = package.record_service_request(UpdateRequest(
        package.service_checkpoint_before, package.package_id, tuple(rows), 'grpo'))
    assert partial.stats.rollout_count == 2 * package.mix_configuration.group_size
    assert partial.stats.designer_rollout_count == partial.stats.benchmark_rollout_count


def test_package_service_data_wrapper_uses_existing_native_task_registry(native_fixture, monkeypatch):
    """Mocked rollout capture exercises the adapter; it does not claim model training."""
    from types import SimpleNamespace
    import usami_v2.runtime.runner as native_runner
    import rift.training.service_data as service_data

    fixture = native_fixture
    package = _package(fixture)
    task_registry = fixture[3]
    episodes = []
    for task_index, task_ref in enumerate(package.task_refs):
        group = package.group_by_task_ref[task_ref]
        for member in range(group.group_size):
            call = SimpleNamespace(role='service', record_hash=f'call-{task_index}-{member}',
                request={'group_id': group.group_id, 'group_size': group.group_size})
            episodes.append(SimpleNamespace(task=SimpleNamespace(record_hash=task_ref),
                record_hash=f'episode-{task_index}-{member}', model_calls=(call,)))

    class Capture:
        def verify_trajectory(self, episode, checkpoint): return True
        def trajectory_for(self, episode, checkpoint):
            task_index = package.task_refs.index(episode.task.record_hash)
            member = int(episode.record_hash.rsplit('-', 1)[1])
            return Tokens((100 + task_index,), (200 + member,), (True,))
        def group_for(self, episode, checkpoint):
            return package.group_by_task_ref[episode.task.record_hash].group_id

    monkeypatch.setattr(native_runner, 'replay_episode', lambda episode: SimpleNamespace(verified=True))
    monkeypatch.setattr(service_data, 'public_business_outcome', lambda episode, verified: Outcome.SUCCESS)
    request, recorded = service_data.prepare_package_service_update(package, task_registry,
        tuple(episodes), package.service_checkpoint_before, Capture())
    assert {row.task_id for row in request.rollouts} == set(package.task_refs)
    assert recorded.stats.rollout_count == len(request.rollouts) == 8
    assert recorded.stats.designer_rollout_count == recorded.stats.benchmark_rollout_count == 4


def test_native_package_grounding_falls_through_rejected_candidate(native_fixture, monkeypatch):
    from types import SimpleNamespace
    import usami_v2.scenarios.grounding as native_grounding
    import rift.benchmark.usami as adapter
    from usami_v2.scenarios.designer import TaskDraft, compile_task
    from rift.tasks import Certification, CertificationStatus

    plan, host, result, *_ = native_fixture
    proposal = Proposal('candidate-fallback', 'package-level ability',
        world_draft=plan.world_draft, challenges=(_challenge('fallback-challenge'),))
    candidates = (SimpleNamespace(key='first'), SimpleNamespace(key='second'))
    monkeypatch.setattr(native_grounding, 'ground_challenge', lambda *args: SimpleNamespace(
        status='GROUNDED', candidates=candidates, search_truncated=False))
    def compile_task_candidate(candidate, spec, build, distribution, allowed=()):
        task_id = 'fallback-a' if candidate.key == 'first' else 'fallback-b'
        return compile_task(TaskDraft(task_id=task_id, customer_alias='customer',
            remedy_goals=({'goal_id':'refund', 'line_alias':'damaged', 'quantity':1,
                           'allowed_remedies':['refund']},)), spec, build, distribution, allowed)
    monkeypatch.setattr(adapter, 'compile_candidate', compile_task_candidate)
    native_certifier = adapter.certify_native
    attempted = []
    def certify(proposal_id, world, task, scenario, budget):
        attempted.append(task.task_id)
        if task.task_id == 'fallback-a':
            return Certification(proposal_id, CertificationStatus.REJECTED), None
        return native_certifier(proposal_id, world, task, scenario, budget)
    monkeypatch.setattr(adapter, 'certify_native', certify)

    trained = host.certify_training_plan(proposal, world_seed=121)
    assert attempted == ['fallback-a', 'fallback-b']
    assert tuple(task.task_id for task in trained.tasks) == ('fallback-b',)
    assert trained.challenge_results[0]['attempts'][0]['status'] == 'REJECTED'
    assert trained.challenge_results[0]['attempts'][1]['status'] == 'CERTIFIED'


def test_unverified_world_is_rejected_before_package_assembly(native_fixture, monkeypatch):
    import usami_v2.scenarios.builder as builder
    from rift.training.package import assemble_training_package

    plan, host, _, task_registry, pool, _, config, checkpoint = native_fixture
    monkeypatch.setattr(builder, 'verify_build', lambda *args: {
        'verified': False, 'status': 'UNSAT', 'world_hash': 'unverified'})
    result = host.certify_training_plan(plan, world_seed=131)
    with pytest.raises(ValueError, match='World did not pass native build verification'):
        assemble_training_package(package_id='invalid-world', proposal=plan,
            designer_action_ref='a' * 32, target_ability_id='A000001',
            designer_result=result, benchmark_pool=pool, task_registry=task_registry,
            service_checkpoint_before=checkpoint, config=config, round_index=4)
