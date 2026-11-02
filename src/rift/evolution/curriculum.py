"""Deterministic structured curriculum policies over explicitly specified arms."""
from dataclasses import dataclass
import json
import random


def _write_new_json(path, value):
    """Publish a complete immutable JSON record without replacing an existing one."""
    import os
    import tempfile
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n'
    descriptor, temporary = tempfile.mkstemp(prefix='.curriculum-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)

from rift.rewards.learnability import LearnabilityStats


@dataclass(frozen=True)
class Selection:
    family_id: str
    knobs: dict
    n_tasks: int

    def __post_init__(self):
        if not self.family_id or not isinstance(self.knobs, dict) or type(self.n_tasks) is not int or self.n_tasks < 1:
            raise ValueError('a family, knob mapping and positive task count are required')
        json.dumps(self.knobs, sort_keys=True, allow_nan=False)

    @property
    def arm_key(self):
        return self.family_id + ':' + json.dumps(self.knobs, sort_keys=True, separators=(',', ':'), allow_nan=False)


@dataclass(frozen=True)
class Observation:
    selection: Selection
    mean_success: float
    learnability: float
    valid_groups: int

    def __post_init__(self):
        if (not isinstance(self.selection, Selection) or not 0 <= self.mean_success <= 1
                or not 0 <= self.learnability <= 1 or type(self.valid_groups) is not int
                or self.valid_groups < 1):
            raise ValueError('invalid curriculum observation')


class CurriculumPolicy:
    def __init__(self, arms, *, seed=0):
        self.arms = tuple(arms)
        if (not self.arms or any(not isinstance(arm, Selection) for arm in self.arms)
                or len({arm.arm_key for arm in self.arms}) != len(self.arms)
                or len({arm.n_tasks for arm in self.arms}) != 1
                or type(seed) is not int or seed < 0):
            raise ValueError('unique equal-budget curriculum arms and a non-negative seed are required')
        self.seed = seed

    def _random(self, round_index):
        if type(round_index) is not int or round_index < 0:
            raise ValueError('non-negative round index required')
        return random.Random(f'curriculum-v1:{self.seed}:{round_index}')

    def _by_arm(self, history):
        rows = {arm.arm_key: [] for arm in self.arms}
        for obs in history:
            if not isinstance(obs, Observation) or obs.selection.arm_key not in rows:
                raise ValueError('history contains an unknown curriculum arm')
            if obs.selection.n_tasks != self.arms[0].n_tasks:
                raise ValueError('history task budget differs from the current policy')
            rows[obs.selection.arm_key].append(obs)
        return rows

    def observe(self, selection, stats):
        if selection not in self.arms or not isinstance(stats, LearnabilityStats):
            raise ValueError('configured selection and verified learnability statistics required')
        designer = tuple(row for row in stats.tasks if row.source == 'designer')
        if not designer or len(designer) != stats.designer_valid_groups:
            raise ValueError('Designer observations do not match the valid-group count')
        return Observation(selection, sum(row.successes / row.group_size for row in designer) / len(designer),
                           stats.designer_mean, len(designer))


class RandomPolicy(CurriculumPolicy):
    def propose(self, round_index, history):
        self._by_arm(history)
        return self._random(round_index).choice(self.arms)


class FixedFamilyPolicy(CurriculumPolicy):
    def __init__(self, arms, *, family_id, seed=0):
        super().__init__(arms, seed=seed)
        matching = [arm for arm in self.arms if arm.family_id == family_id]
        if len(matching) != 1:
            raise ValueError('fixed family must identify exactly one configured arm')
        self.fixed = matching[0]

    def propose(self, round_index, history):
        self._random(round_index)
        self._by_arm(history)
        return self.fixed


class DifficultyOnlyPolicy(CurriculumPolicy):
    def propose(self, round_index, history):
        rng, rows = self._random(round_index), self._by_arm(history)
        unseen = [arm for arm in self.arms if not rows[arm.arm_key]]
        if unseen:
            return rng.choice(unseen)
        distances = {arm.arm_key: abs(rows[arm.arm_key][-1].mean_success - .5) for arm in self.arms}
        best = min(distances.values())
        return rng.choice([arm for arm in self.arms if distances[arm.arm_key] == best])


class LearningProgressPolicy(CurriculumPolicy):
    def __init__(self, arms, *, seed=0, window=3):
        super().__init__(arms, seed=seed)
        if type(window) is not int or window < 1:
            raise ValueError('positive learning progress window required')
        self.window = window

    def propose(self, round_index, history):
        rng, rows = self._random(round_index), self._by_arm(history)
        incomplete = [arm for arm in self.arms if len(rows[arm.arm_key]) < 2 * self.window]
        if incomplete:
            least = min(len(rows[arm.arm_key]) for arm in incomplete)
            return rng.choice([arm for arm in incomplete if len(rows[arm.arm_key]) == least])
        def progress(arm):
            values = [item.learnability for item in rows[arm.arm_key][-2 * self.window:]]
            return abs(sum(values[self.window:]) - sum(values[:self.window])) / self.window
        scores = {arm.arm_key: progress(arm) for arm in self.arms}
        best = max(scores.values())
        return rng.choice([arm for arm in self.arms if scores[arm.arm_key] == best])


class BanditPolicy(CurriculumPolicy):
    def propose(self, round_index, history):
        rng, rows = self._random(round_index), self._by_arm(history)
        # Beta pseudocounts accept fractional learnability observations; use
        # one update per valid group so a larger observed sample counts more.
        def sample(arm):
            history_for_arm = rows[arm.arm_key]
            alpha = 1 + sum(row.learnability * row.valid_groups for row in history_for_arm)
            beta = 1 + sum((1 - row.learnability) * row.valid_groups for row in history_for_arm)
            return rng.betavariate(alpha, beta)
        return max(self.arms, key=sample)


def _decode_update(data):
    from rift.schemas import Checkpoint, Tokens
    from rift.training.contracts import CompletedUpdate, Rollout, UpdateRequest
    raw = data['request']
    def rollout(row):
        tokens = row['tokens']
        return Rollout(Tokens(tuple(tokens['prompt']), tuple(tokens['response']), tuple(tokens['mask'])),
            Checkpoint(**row['policy']), row['task_id'], row['episode_id'], row['reward'],
            row['reward_source'], row.get('group_id'), row.get('response_id'),
            row.get('reward_ref'), row.get('target_ability_id'))
    request = UpdateRequest(Checkpoint(**raw['policy_before']), raw['package_id'],
        tuple(rollout(row) for row in raw['rollouts']), raw['algorithm'], raw.get('advantage_baseline'),
        raw.get('reward_protocol'), raw.get('credit_assignment_mode'))
    return CompletedUpdate(request, Checkpoint(**data['policy_after']), data['run_id'], data['evidence_ref'])


class CurriculumRunner:
    """Run verified Service updates without an LLM Designer or LoopState.

    The injected factories own deployment-specific host, capture and backend
    resources. ``recover_update`` must reconcile an existing run's update
    ledger and checkpoint before returning None; the backend must reject an
    already-used output path. Round records bind the exact package and request.
    """
    def __init__(self, *, policy, families, host_factory, benchmark_pool, task_registry,
                 package_config, collector_factory, backend_factory, recover_update,
                 checkpoint_store, update_verifier, run_root, seed=42,
                 benchmark_selector=None, validation_every=0, validation_callback=None,
                 update_budget=None):
        from pathlib import Path
        from rift.training.package import TrainingPackageConfig
        from rift.evolution.matched_control import UpdateBudget
        if not isinstance(policy, CurriculumPolicy) or not isinstance(package_config, TrainingPackageConfig):
            raise TypeError('curriculum policy and frozen package configuration required')
        if any(not callable(item) for item in (host_factory, collector_factory, backend_factory, recover_update)):
            raise TypeError('explicit Host, collector, backend and recovery factories required')
        if any(arm.n_tasks != package_config.designer_task_count for arm in policy.arms):
            raise ValueError('every arm must use the package Designer task budget')
        if not isinstance(update_budget, UpdateBudget):
            raise ValueError('an explicit shared Service update budget is required')
        if not callable(getattr(update_verifier, 'verify_update_budget', None)):
            raise ValueError('curriculum update verifier must check recorded update budgets')
        if (type(seed) is not int or seed < 0 or type(validation_every) is not int or validation_every < 0
                or (validation_every and (not callable(getattr(validation_callback, 'prepare', None))
                    or not callable(getattr(validation_callback, 'finish', None))))):
            raise ValueError('periodic validation requires preparation before training and a finish callback')
        self.policy, self.families, self.host_factory = policy, dict(families), host_factory
        self.benchmark_pool, self.task_registry, self.package_config = benchmark_pool, task_registry, package_config
        self.collector_factory, self.backend_factory, self.recover_update = collector_factory, backend_factory, recover_update
        self.checkpoint_store, self.update_verifier = checkpoint_store, update_verifier
        self.run_root, self.seed, self.benchmark_selector = Path(run_root), seed, benchmark_selector
        self.validation_every, self.validation_callback = validation_every, validation_callback
        self.update_budget = update_budget

    def _round_record(self, path, expected_checkpoint, expected_index):
        from dataclasses import asdict
        from rift.rewards.learnability import score_learnability
        from rift.training.package import TrainingPackage, _record_for_train
        raw = json.loads(path.read_text(encoding='utf-8'))
        package, update = TrainingPackage.from_dict(raw['package']), _decode_update(raw['update'])
        selection = Selection(**raw['selection'])
        if (raw['round_index'] != expected_index or update.request.policy_before != expected_checkpoint
                or not package.verify_service_request(update.request)
                or package.mix_configuration != self.package_config
                or package.benchmark_pool_id != self.benchmark_pool.pool_id
                or package.benchmark_pool_digest != self.benchmark_pool.pool_digest
                or not set(package.benchmark_task_refs).issubset(self.benchmark_pool.task_refs)
                or self.update_verifier.verify_update(update) is not True
                or self.update_verifier.verify_update_budget(update, self.update_budget) is not True
                or self.checkpoint_store.verify(update.policy_after) is not True):
            raise ValueError('persisted curriculum round does not bind verified update evidence')
        for source in package.task_sources:
            record = _record_for_train(self.task_registry, source.task_ref)
            if (record.identity != source.identity or record.world.state_hash != source.world_hash
                    or record.certificate.record_hash != source.certificate_hash):
                raise ValueError('persisted curriculum task differs from its certified registry entry')
        stats = score_learnability(package, update.request)
        record_stats = getattr(self.benchmark_selector, 'record_request', None)
        if callable(record_stats):
            record_stats(package, update.request, expected_index)
        observation = self.policy.observe(selection, stats)
        if asdict(observation) != raw['observation']:
            raise ValueError('persisted curriculum observation differs from recomputation')
        return update, observation, raw

    def run(self, initial_checkpoint, rounds):
        from dataclasses import asdict
        from rift.rewards.learnability import score_learnability
        from rift.training.package import assemble_training_package
        from rift.training.contracts import CompletedUpdate
        if type(rounds) is not int or rounds < 1 or self.checkpoint_store.verify(initial_checkpoint) is not True:
            raise ValueError('positive round budget and verified starting checkpoint required')
        if not self.benchmark_pool.verify(self.task_registry):
            raise ValueError('frozen benchmark pool failed certified registry verification')
        self.run_root.mkdir(parents=True, exist_ok=True)
        manifest = {'schema_version': 1, 'policy': type(self.policy).__name__, 'policy_seed': self.policy.seed,
                    'policy_window': getattr(self.policy, 'window', None),
                    'fixed_arm': (self.policy.fixed.arm_key if isinstance(self.policy, FixedFamilyPolicy) else None),
                    'arms': [asdict(arm) for arm in self.policy.arms], 'seed': self.seed,
                    'package_config': asdict(self.package_config), 'initial_checkpoint': asdict(initial_checkpoint),
                    'benchmark_pool_id': self.benchmark_pool.pool_id,
                    'benchmark_pool_digest': self.benchmark_pool.pool_digest,
                    'update_budget': asdict(self.update_budget),
                    'validation_every': self.validation_every}
        manifest_path = self.run_root / 'manifest.json'
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
                raise ValueError('curriculum run manifest differs from the requested configuration')
        else:
            _write_new_json(manifest_path, manifest)
        history, checkpoint = [], initial_checkpoint
        for index in range(rounds):
            directory = self.run_root / f'round-{index:06d}'
            directory.mkdir(parents=True, exist_ok=True)
            record_path = directory / 'round.json'
            validation_due = bool(self.validation_every and (index + 1) % self.validation_every == 0)
            preparation_path = directory / 'validation-preparation.json'
            if record_path.exists():
                update, observation, raw = self._round_record(record_path, checkpoint, index)
                if observation.selection != self.policy.propose(index, tuple(history)):
                    raise ValueError('persisted curriculum selection differs from configured policy')
                if raw['package']['world_seed'] != self.seed + index:
                    raise ValueError('persisted curriculum World seed differs from the run manifest')
                if validation_due and not preparation_path.exists():
                    raise ValueError('completed round is missing its pre-update validation preparation')
            else:
                selection = self.policy.propose(index, tuple(history))
                family = self.families[selection.family_id]
                instance = family.instance(selection.knobs, self.seed + index, n_tasks=selection.n_tasks)
                proposal = instance.proposal(n_tasks=selection.n_tasks)
                host = self.host_factory(instance)
                if host.task_registry is not self.task_registry:
                    raise ValueError('curriculum Host and package must share the certified task registry')
                sources = {source.source_id: source for source in host.distribution.sources}
                if (host.distribution.claim != 'synthetic_business_reference'
                        or any(sources[row.weight_basis_source_id].kind != 'synthetic_assumption'
                               for row in host.distribution.strata)):
                    raise ValueError('synthetic family tasks require synthetic-assumption distribution provenance')
                if (selection.family_id == 'F9'
                        and host.return_receipt_after_days != selection.knobs.get('receipt_delay_days', 1)):
                    raise ValueError('physical-return Host receipt delay differs from its family knob')
                result = host.certify_training_plan(proposal, world_seed=instance.world_seed,
                    max_candidate_challenges=self.package_config.max_candidate_challenges)
                package_id = f'curriculum-{index:06d}-{proposal.proposal_id}'
                package = assemble_training_package(package_id=package_id, proposal=proposal,
                    designer_action_ref=f'structured-curriculum-{index:06d}', target_ability_id='curriculum',
                    designer_result=result, benchmark_pool=self.benchmark_pool, task_registry=self.task_registry,
                    service_checkpoint_before=checkpoint, config=self.package_config, round_index=index,
                    benchmark_selector=self.benchmark_selector)
                collector = self.collector_factory(checkpoint, index)
                request, recorded = collector.collect(package, checkpoint)
                if not recorded.verify_service_request(request):
                    raise ValueError('Service request differs from its frozen certified package')
                # Reject insufficient Designer evidence before producing an
                # expensive checkpoint that cannot update the curriculum.
                stats = score_learnability(recorded, request)
                record_stats = getattr(self.benchmark_selector, 'record_request', None)
                if callable(record_stats):
                    record_stats(recorded, request, index)
                backend = self.backend_factory(index, checkpoint, directory)
                if (not callable(getattr(backend, 'budget_for', None))
                        or backend.budget_for(request) != self.update_budget):
                    raise ValueError('Service backend differs from the shared curriculum update budget')
                self.update_budget.optimizer_steps(request)
                if validation_due and not preparation_path.exists():
                    preparation = self.validation_callback.prepare(index, request, recorded)
                    _write_new_json(preparation_path, preparation)
                update = self.recover_update(index, request)
                if update is None:
                    update = backend.train(request)
                if (not isinstance(update, CompletedUpdate) or update.request != request
                        or self.update_verifier.verify_update(update) is not True
                        or self.update_verifier.verify_update_budget(update, self.update_budget) is not True
                        or self.checkpoint_store.verify(update.policy_after) is not True):
                    raise ValueError('curriculum update failed checkpoint or update verification')
                observation = self.policy.observe(selection, stats)
                raw = {'round_index': index, 'selection': asdict(selection),
                       'observation': asdict(observation), 'package': asdict(recorded), 'update': asdict(update)}
                _write_new_json(record_path, raw)
            history.append(observation)
            checkpoint = update.policy_after
            # Summary is separate from exact evidence, and can be reconstructed
            # from the committed per-round records after interruption.
            summary = self.run_root / 'rounds.jsonl'
            summaries = [json.loads(line) for line in summary.read_text(encoding='utf-8').splitlines()] if summary.exists() else []
            expected_summary = {'round_index': index, 'selection': raw['selection'],
                                'observation': raw['observation'], 'checkpoint': asdict(checkpoint)}
            if len(summaries) == index:
                with summary.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(expected_summary, sort_keys=True,
                                            ensure_ascii=False, allow_nan=False) + '\n')
            elif len(summaries) <= index or summaries[index] != expected_summary:
                raise ValueError('curriculum summary is inconsistent with committed rounds')
            if validation_due:
                validation_path = directory / 'validation.json'
                if not validation_path.exists():
                    preparation = json.loads(preparation_path.read_text(encoding='utf-8'))
                    validation = self.validation_callback.finish(index, update, raw, preparation)
                    _write_new_json(validation_path, validation)
        return checkpoint, tuple(history)
