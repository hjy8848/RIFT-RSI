import pytest

from rift.evolution.curriculum import (BanditPolicy, DifficultyOnlyPolicy, FixedFamilyPolicy,
    LearningProgressPolicy, Observation, RandomPolicy, Selection)


ARMS = tuple(Selection(f'family-{i}', {'difficulty': i}, 8) for i in range(3))


@pytest.mark.parametrize('policy', [RandomPolicy(ARMS, seed=31),
    FixedFamilyPolicy(ARMS, family_id='family-1', seed=31),
    DifficultyOnlyPolicy(ARMS, seed=31), LearningProgressPolicy(ARMS, seed=31),
    BanditPolicy(ARMS, seed=31)])
def test_policy_is_deterministic_and_preserves_budget(policy):
    history = [Observation(ARMS[0], .2, .64, 8), Observation(ARMS[1], .5, 1., 8)]
    assert policy.propose(4, history) == policy.propose(4, history)
    assert policy.propose(4, history).n_tasks == 8


def test_bandit_finds_best_synthetic_arm():
    policy = BanditPolicy(ARMS, seed=8)
    values = [.15, .45, .9]
    history = []
    for round_index in range(150):
        selection = policy.propose(round_index, history)
        value = values[ARMS.index(selection)]
        history.append(Observation(selection, .5, value, 8))
    assert sum(row.selection == ARMS[2] for row in history[-50:]) >= 45


def test_bandit_forgets_an_arm_when_its_learnability_decays():
    policy = BanditPolicy(ARMS[:2], seed=8, window=4)
    history = ([Observation(ARMS[0], .5, .95, 8) for _ in range(12)]
               + [Observation(ARMS[0], .5, .05, 8) for _ in range(4)]
               + [Observation(ARMS[1], .5, .6, 8) for _ in range(4)])
    alpha_a, beta_a = policy._posterior([row for row in history if row.selection == ARMS[0]])
    alpha_b, beta_b = policy._posterior([row for row in history if row.selection == ARMS[1]])
    assert alpha_a / (alpha_a + beta_a) < alpha_b / (alpha_b + beta_b)
    assert policy.window == 4


def test_difficulty_uses_observed_success_and_progress_uses_recent_change():
    difficulty = DifficultyOnlyPolicy(ARMS, seed=0)
    history = [Observation(arm, mean, .3, 4) for arm, mean in zip(ARMS, [.1, .45, .8])]
    assert difficulty.propose(3, history) == ARMS[1]
    progress = LearningProgressPolicy(ARMS, window=1)
    history = [Observation(arm, .5, score, 4) for arm, pair in zip(ARMS,
        [(.2, .21), (.2, .8), (.4, .3)]) for score in pair]
    assert progress.propose(6, history) == ARMS[1]


def test_mismatched_arm_budgets_are_rejected():
    with pytest.raises(ValueError, match='equal-budget'):
        RandomPolicy((ARMS[0], Selection('other', {}, 4)))


def test_runner_uses_certified_packages_and_resumes_without_retraining(tmp_path, monkeypatch):
    from rift.benchmark.families import FAMILIES
    from rift.benchmark.usami import CommerceHost
    from rift.evolution.curriculum import CurriculumRunner
    from rift.evolution.matched_control import UpdateBudget
    from rift.schemas import Checkpoint, Tokens
    from rift.training.contracts import CompletedUpdate, Rollout, UpdateRequest
    from rift.training.task_stats import TaskStatsStore
    from usami_v2.runtime.runner import EpisodeBudget
    from usami_v2.scenarios.task_certification import ProofBudget
    from test_training_package import _native_fixture

    fixture = _native_fixture()
    _, _, _, task_registry, pool, split_registry, config, initial = fixture
    distribution = fixture[1].distribution
    selection = Selection('F1', {}, config.designer_task_count)
    policy = FixedFamilyPolicy((selection,), family_id='F1', seed=7)
    def host_factory(instance):
        return CommerceHost((), distribution, None, None, None, registry=split_registry,
            identity_factory=instance.identity, proof_budget=ProofBudget(),
            episode_budget=EpisodeBudget(), task_registry=task_registry)
    class Collector:
        def collect(self, package, checkpoint):
            rows = []
            for index, group in enumerate(package.task_groups):
                for member in range(group.group_size):
                    rows.append(Rollout(Tokens((index + 1,), (member + 1,), (True,)), checkpoint,
                        group.task_ref, f'episode-{index}-{member}', float(member), 'outcome', group.group_id))
            request = UpdateRequest(checkpoint, package.package_id, tuple(rows), 'grpo')
            return request, package.record_service_request(request)
    updates = []
    stages = []
    class Backend:
        def budget_for(self, request): return UpdateBudget(8, 1, 1e-6)
        def train(self, request):
            stages.append('train')
            index = len(updates) + 1
            after = Checkpoint('service', str(index), request.policy_before.tokenizer, 'weights-' + str(index))
            result = CompletedUpdate(request, after, 'run-' + str(index), 'evidence-' + str(index))
            updates.append(result)
            return result
    verifier = type('Verifier', (), {'verify_update': lambda self, update: update in updates,
        'verify_update_budget': lambda self, update, budget: update in updates and budget == UpdateBudget(8, 1, 1e-6)})()
    store = type('Store', (), {'verify': lambda self, checkpoint: True})()
    class Validation:
        def prepare(self, index, request, package):
            stages.append('prepare')
            return {'baseline': request.policy_before.weights_digest}
        def finish(self, index, update, raw, preparation):
            stages.append('finish')
            assert preparation['baseline'] == update.request.policy_before.weights_digest
            return {'validated': True}
    runner = CurriculumRunner(policy=policy, families=FAMILIES, host_factory=host_factory,
        benchmark_pool=pool, task_registry=task_registry, package_config=config,
        collector_factory=lambda checkpoint, index: Collector(),
        backend_factory=lambda index, checkpoint, path: Backend(),
        recover_update=lambda index, request: next((u for u in updates if u.request == request), None),
        checkpoint_store=store,
        update_verifier=verifier, run_root=tmp_path / 'run', seed=30,
        validation_every=1, validation_callback=Validation(), update_budget=UpdateBudget(8, 1, 1e-6),
        benchmark_selector=TaskStatsStore(tmp_path / 'statistics.jsonl'))
    after, history = runner.run(initial, 2)
    assert len(history) == 2 and after == updates[-1].policy_after
    assert len(updates) == 2
    assert stages == ['prepare', 'train', 'finish', 'prepare', 'train', 'finish']
    assert runner.run(initial, 2) == (after, history)
    assert len(updates) == 2
    assert len(stages) == 6
    assert len((tmp_path / 'run' / 'rounds.jsonl').read_text().splitlines()) == 2
    assert len(runner.benchmark_selector.read()) == 8
    import json
    recorded = json.loads((tmp_path / 'run' / 'round-000000' / 'round.json').read_text())
    designer_tasks = [task_registry.resolve_record(ref).task
                      for ref in recorded['package']['designer_task_refs']]
    assert len({task.remedy_goals[0].line_id for task in designer_tasks}) == 2
    runner.run_root = tmp_path / 'interrupted'
    runner.seed = 90
    original_observe = policy.observe
    def interrupted(*args):
        raise RuntimeError('simulated interruption after verified update')
    monkeypatch.setattr(policy, 'observe', interrupted)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        runner.run(initial, 1)
    assert len(updates) == 3
    monkeypatch.setattr(policy, 'observe', original_observe)
    runner.run(initial, 1)
    assert len(updates) == 3


def test_runner_rejects_low_coverage_before_training(tmp_path):
    from dataclasses import replace
    from rift.benchmark.families import FAMILIES
    from rift.benchmark.usami import CommerceHost
    from rift.evolution.curriculum import CurriculumRunner
    from rift.evolution.matched_control import UpdateBudget
    from rift.rewards.learning_value import InsufficientEvidenceError
    from rift.schemas import Tokens
    from rift.training.contracts import Rollout, UpdateRequest
    from usami_v2.runtime.runner import EpisodeBudget
    from usami_v2.scenarios.task_certification import ProofBudget
    from test_training_package import _native_fixture
    fixture = _native_fixture()
    _, host, _, registry, pool, splits, original, checkpoint = fixture
    config = replace(original, tasks_per_round=8)
    policy = FixedFamilyPolicy((Selection('F1', {}, 4),), family_id='F1')
    def host_factory(instance):
        return CommerceHost((), host.distribution, None, None, None, registry=splits,
            identity_factory=instance.identity, proof_budget=ProofBudget(),
            episode_budget=EpisodeBudget(), task_registry=registry)
    class Collector:
        def collect(self, package, cp):
            chosen = [next(group for group in package.task_groups if group.source == source)
                      for source in ('designer', 'benchmark')]
            rows = tuple(Rollout(Tokens((i + 1,), (member + 1,), (True,)), cp, group.task_ref,
                f'episode-{i}-{member}', float(member), 'outcome', group.group_id)
                for i, group in enumerate(chosen) for member in range(group.group_size))
            request = UpdateRequest(cp, package.package_id, rows, 'grpo')
            return request, package.record_service_request(request)
    verifier = type('Verifier', (), {'verify_update_budget': lambda self, update, budget: True})()
    runner = CurriculumRunner(policy=policy, families=FAMILIES, host_factory=host_factory,
        benchmark_pool=pool, task_registry=registry, package_config=config,
        collector_factory=lambda cp, i: Collector(),
        backend_factory=lambda *args: pytest.fail('must not create backend on insufficient evidence'),
        recover_update=lambda *args: None, checkpoint_store=type('Store', (), {'verify': lambda s, cp: True})(),
        update_verifier=verifier, run_root=tmp_path, update_budget=UpdateBudget(4, 1, 1e-6))
    with pytest.raises(InsufficientEvidenceError):
        runner.run(checkpoint, 1)
