"""Real 8x910B3 entry points for the dense-proxy curriculum.

``train`` runs ``CurriculumRunner`` rounds on a real VERL Service policy.
``eval`` measures a frozen checkpoint on certified synthetic families without
updating it (experiments X0/X1/X3 in docs/PLAN_DENSE_PROXY_CURRICULUM.md).

VERL starts a fresh Ray session per Service runtime and its AgentLoop workers
read the certified-task registry and the checkpoint manifest once at start-up.
Each round therefore certifies on the driver, saves the registries, and only then
starts one runtime that serves both the rollouts and the update of that round.
"""
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path

from rift.runtime.ascend_910b3 import CHAT_TEMPLATE_KWARGS, WORLD_SIZE
DEFAULT_ARMS = ({'family_id': 'F1', 'knobs': {}}, {'family_id': 'F4', 'knobs': {'age_days': 20}})
DEFAULT_BENCH = ({'family': 'F1', 'knobs': {}, 'instances': 8, 'seed': 1000000},
                 {'family': 'F4', 'knobs': {'age_days': 20}, 'instances': 8, 'seed': 1001000})
POLICIES = ('random', 'fixed', 'difficulty', 'learning_progress', 'bandit')


def load_json_argument(value):
    """Accept inline JSON or a path to a JSON file."""
    text = value.strip()
    if text.startswith(('{', '[')):
        return json.loads(text)
    return json.loads(Path(value).read_text(encoding='utf-8'))


def parse_task_specs(rows):
    """Validate ``[{family, knobs, instances, seed}]`` certification requests."""
    specs = []
    for row in rows:
        if (not isinstance(row, dict) or set(row) - {'family', 'knobs', 'instances', 'seed'}
                or type(row.get('instances')) is not int or row['instances'] < 1
                or type(row.get('seed')) is not int or row['seed'] < 0):
            raise ValueError('each task spec needs family, instances >= 1 and a non-negative seed')
        knobs = row.get('knobs', {})
        if not isinstance(knobs, dict):
            raise ValueError('task spec knobs must be a mapping')
        specs.append({'family': row['family'], 'knobs': knobs,
                      'instances': row['instances'], 'seed': row['seed']})
    if not specs:
        raise ValueError('at least one task spec is required')
    return tuple(specs)


def spec_key(spec):
    return spec['family'] + ':' + json.dumps(spec['knobs'], sort_keys=True, separators=(',', ':'))


def wilson_interval(successes, total, z=1.96):
    if total < 1:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize_outcomes(rows):
    """Aggregate ``(group_key, task_ref, outcome_value)`` rows; UNKNOWN is kept apart."""
    groups = {}
    for key, task_ref, outcome in rows:
        entry = groups.setdefault(key, {'tasks': {}})
        counts = entry['tasks'].setdefault(task_ref, {'success': 0, 'failure': 0, 'unknown': 0})
        # rift.schemas.Outcome values are upper case ("SUCCESS"); accept either spelling.
        normalized = str(getattr(outcome, 'value', outcome)).lower()
        counts[normalized if normalized in ('success', 'failure') else 'unknown'] += 1
    report = {}
    for key, entry in sorted(groups.items()):
        success = sum(row['success'] for row in entry['tasks'].values())
        failure = sum(row['failure'] for row in entry['tasks'].values())
        unknown = sum(row['unknown'] for row in entry['tasks'].values())
        scored = success + failure
        low, high = wilson_interval(success, scored)
        report[key] = {'tasks': len(entry['tasks']), 'success': success, 'failure': failure,
            'unknown': unknown, 'success_rate': (success / scored) if scored else None,
            'wilson95': [low, high], 'per_task': entry['tasks']}
    return report


class WorldKit:
    """Certified-task and split registries plus the Host factory for family instances."""
    def __init__(self, root, *, proof_budget=None, episode_budget=None):
        from rift.benchmark.families import synthetic_distribution
        from rift.replay.splits import SplitRegistry
        from rift.schemas import INSTANCE_DISJOINT_WITHIN_FAMILY_V1
        from rift.verl.task_registry import CertifiedTaskRegistry
        from usami_v2.runtime.runner import EpisodeBudget
        from usami_v2.scenarios.task_certification import ProofBudget
        directory = Path(root) / 'task_registry'
        directory.mkdir(parents=True, exist_ok=True)
        self.splits_path, self.tasks_path = directory / 'splits.json', directory / 'certified-tasks.json'
        if self.splits_path.exists() != self.tasks_path.exists():
            raise ValueError('task registry state is incomplete; refusing to continue')
        if self.splits_path.exists():
            self.split_registry = SplitRegistry.load(self.splits_path)
            self.task_registry = CertifiedTaskRegistry.load(self.tasks_path, self.split_registry)
        else:
            self.split_registry = SplitRegistry(policy_version=INSTANCE_DISJOINT_WITHIN_FAMILY_V1)
            self.task_registry = CertifiedTaskRegistry(self.split_registry)
        self.split_registry.persistence_path = self.splits_path
        self.task_registry.storage_path = self.tasks_path
        self.distribution = synthetic_distribution('train')
        self.proof_budget = proof_budget or ProofBudget()
        self.episode_budget = episode_budget or EpisodeBudget()

    def host(self, instance):
        from rift.benchmark.usami import CommerceHost
        return CommerceHost((), self.distribution, None, None, None, registry=self.split_registry,
            identity_factory=instance.identity, proof_budget=self.proof_budget,
            episode_budget=self.episode_budget, task_registry=self.task_registry,
            return_receipt_after_days=(instance.knobs.get('receipt_delay_days', 1)
                                       if instance.family_id == 'F9' else None))

    def certify_specs(self, specs):
        """Certify one single-task World per instance; returns ``{spec_key: [task_ref]}``."""
        from rift.benchmark.families import FAMILIES
        refs = {}
        for spec in parse_task_specs(specs):
            family = FAMILIES[spec['family']]
            for offset in range(spec['instances']):
                seed = spec['seed'] + offset
                instance = family.instance(spec['knobs'], seed, n_tasks=1)
                result = self.host(instance).certify_training_plan(
                    instance.proposal(n_tasks=1), world_seed=seed)
                if result.build.status != 'SAT' or len(result.tasks) != 1:
                    raise RuntimeError('family %s seed %d did not certify: %s'
                                       % (spec['family'], seed, result.build.status))
                refs.setdefault(spec_key(spec), []).append(result.tasks[0].record_hash)
        return refs

    def save(self):
        self.split_registry.save()
        self.task_registry.save()


class ExperimentContext:
    """Checkpoint/attestation stores shared by every runtime of one run directory."""
    def __init__(self, *, root, service_model, verl_config_dir, episode_budget, update_budget,
                 tool_profile='full9', runtime_factory=None, group_size=4, response_length=8192,
                 keep_runtime=True, persist_every=1, persist_final=True):
        from rift.verl.checkpoints import FSDPCheckpointStore
        self.keep_runtime = keep_runtime
        self.persist_final = persist_final
        self.persist_every, self.total_rounds = persist_every, None
        self.root, self.service_model = Path(root).resolve(), str(service_model)
        self.verl_config_dir, self.episode_budget = Path(verl_config_dir), episode_budget
        self.update_budget, self.tool_profile, self.group_size = update_budget, tool_profile, group_size
        self.response_length = response_length
        self.runtime_factory = runtime_factory or self._ascend_runtime
        for name in ('task_registry', 'checkpoints', 'episodes', 'generations', 'serving', 'updates'):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.world = WorldKit(self.root, episode_budget=episode_budget)
        self.checkpoint_store = FSDPCheckpointStore({}, world_size=WORLD_SIZE)
        self.runtime = None
        self._wire_stores()

    # -- stores -----------------------------------------------------------
    def _wire_stores(self):
        from rift.verl.capture_store import NativeCaptureStore
        from rift.verl.generation_ledger import GenerationLedger
        from rift.verl.serving import VerlServingAttestor
        from rift.verl.update_ledger import UpdateLedger
        self.serialization = 'verl-v0.9.1-vllm-ascend-0.23.0-native-token-ids-v1'
        self.serving_attestor = VerlServingAttestor(self.root / 'serving', self.checkpoint_store)
        self.generation_attestor = GenerationLedger(self.root / 'generations', self.serving_attestor)
        self.capture_store = NativeCaptureStore(self.root / 'episodes',
            serialization=self.serialization, attestor=self.generation_attestor)
        self.ledger = UpdateLedger(self.root / 'updates', self.checkpoint_store)

    def adopt_manifest(self, path):
        """Replace the checkpoint store with a verified manifest and rebuild dependent stores."""
        from rift.verl.checkpoints import FSDPCheckpointStore
        self.checkpoint_store = FSDPCheckpointStore.load_manifest(path)
        self._wire_stores()

    def manifests(self):
        found = []
        for path in (self.root / 'checkpoints').glob('*.json'):
            if path.name == 'initial.json' or path.name.startswith('service-step-'):
                found.append(path)
        def order(path):
            return 0 if path.name == 'initial.json' else int(path.stem.rsplit('-', 1)[1])
        return sorted(found, key=order)

    def latest_manifest(self):
        found = self.manifests()
        return found[-1] if found else None

    def metric_sink(self, name, value, **fields):
        row = {'metric': name, 'value': value, **fields}
        with (self.root / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, sort_keys=True, default=str, allow_nan=False) + '\n')

    # -- checkpoints ------------------------------------------------------
    def initial_checkpoint(self):
        """Create the starting FSDP2 checkpoint, or reload the one this run already made."""
        from rift.schemas import Checkpoint
        marker = self.root / 'initial-checkpoint.json'
        if marker.exists():
            checkpoint = Checkpoint(**json.loads(marker.read_text(encoding='utf-8')))
            latest = self.latest_manifest()
            if latest is None:
                raise ValueError('initial checkpoint exists without a checkpoint manifest')
            self.adopt_manifest(latest)
            if self.checkpoint_store.verify(checkpoint) is not True:
                raise ValueError('stored initial checkpoint failed verification')
            return checkpoint
        from rift.runtime.ascend_910b3 import initialize_checkpoint
        checkpoint = initialize_checkpoint('service', self.service_model,
            self.root / 'checkpoints' / 'service-initial', checkpoint_store=self.checkpoint_store,
            serving_attestor=self.serving_attestor, run_root=self.root,
            initialize_vllm=False, synchronize=False)
        if checkpoint.tokenizer != str(Path(self.service_model).resolve()):
            raise RuntimeError('initial Service checkpoint is bound to a different tokenizer')
        self.checkpoint_store.save_manifest(self.root / 'checkpoints' / 'initial.json')
        marker.write_text(json.dumps(asdict(checkpoint), sort_keys=True), encoding='utf-8')
        return checkpoint

    def adopt_initial_from(self, run_root):
        """Start this run from another run's base checkpoint instead of writing a new one."""
        checkpoint = self.checkpoint_from_run(run_root)
        self.checkpoint_store.save_manifest(self.root / 'checkpoints' / 'initial.json')
        (self.root / 'initial-checkpoint.json').write_text(
            json.dumps(asdict(checkpoint), sort_keys=True), encoding='utf-8')
        return checkpoint

    def checkpoint_from_run(self, run_root):
        """Load a finished ``train`` run's final checkpoint, or an earlier run's initial one.

        A run directory without ``final.json`` (an ``eval`` run) offers its starting
        checkpoint, which saves rewriting the 16 GB base-model checkpoint.
        """
        from rift.schemas import Checkpoint
        root = Path(run_root)
        if (root / 'final.json').exists():
            final = json.loads((root / 'final.json').read_text(encoding='utf-8'))
            raw, manifest = final['checkpoint'], final['manifest']
        else:
            raw = json.loads((root / 'initial-checkpoint.json').read_text(encoding='utf-8'))
            manifest = root / 'checkpoints' / 'initial.json'
        checkpoint = Checkpoint(**raw)
        self.adopt_manifest(manifest)
        if self.checkpoint_store.verify(checkpoint) is not True:
            raise ValueError('checkpoint of the source run failed verification')
        return checkpoint

    # -- runtime ----------------------------------------------------------
    def _worker_manifest(self, checkpoint):
        from rift.runtime.service_protocol import service_profile_descriptor
        manifest_path = self.latest_manifest() if self.manifests() else None
        if manifest_path is None:
            manifest_path = self.root / 'checkpoints' / 'initial.json'
            self.checkpoint_store.save_manifest(manifest_path)
        records = json.loads(manifest_path.read_text(encoding='utf-8'))['records']
        if asdict(checkpoint) not in [row['checkpoint'] for row in records]:
            # The active checkpoint came from an adopted external manifest.
            manifest_path = self.root / 'checkpoints' / 'adopted.json'
            if not manifest_path.exists():
                self.checkpoint_store.save_manifest(manifest_path)
        # One fixed path: a runtime that stays up across rounds keeps the path it was
        # started with, and its workers re-read this file whenever it changes.
        path = self.root / 'worker-manifest.json'
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_text(json.dumps({
            'active_checkpoint': asdict(checkpoint),
            'checkpoint_manifest': str(manifest_path),
            'split_registry': str(self.world.splits_path),
            'task_registry': str(self.world.tasks_path),
            'serving_receipts': str(self.root / 'serving'),
            'generation_ledger': str(self.root / 'generations'),
            'capture_store': str(self.root / 'episodes'),
            'serialization': self.serialization,
            'episode_budget': self.episode_budget.model_dump(mode='json', exclude={'record_hash'}),
            'tool_profile': self.tool_profile,
            'service_profile': service_profile_descriptor(self.tool_profile),
            'parser_format': 'hermes', 'usami_mode': 'reference',
            'chat_template_kwargs': dict(CHAT_TEMPLATE_KWARGS),
        }, sort_keys=True, indent=2), encoding='utf-8')
        os.replace(temporary, path)   # workers never see a half-written manifest
        return path

    def _ascend_runtime(self, checkpoint):
        from rift.runtime.ascend_910b3 import Ascend910B3RoleRuntime
        return Ascend910B3RoleRuntime('service', checkpoint, model_path=self.service_model,
            run_root=self.root, checkpoint_store=self.checkpoint_store,
            serving_attestor=self.serving_attestor,
            agent_loop_config=self.verl_config_dir / 'agent_loop.yaml',
            worker_manifest_path=self._worker_manifest(checkpoint), open_agent_loop=True,
            tool_profile=self.tool_profile, response_length=self.response_length,
            service_update={'mini_batch_size': self.update_budget.mini_batch_size,
                            'epochs': self.update_budget.epochs,
                            'learning_rate': self.update_budget.learning_rate})

    def persists(self, index):
        """Whether the update of round ``index`` writes real checkpoint files."""
        if self.persist_every and (index + 1) % self.persist_every == 0:
            return True
        if not self.persist_final:
            return False
        return self.total_rounds is not None and index + 1 >= self.total_rounds   # 0 = last round only

    def open_runtime(self, checkpoint):
        """Start one runtime for ``checkpoint``; registries must already hold every task."""
        import time
        if checkpoint in getattr(self.checkpoint_store, 'virtual', ()):
            raise ValueError('revision %s exists only in a runtime that has ended; start a new run '
                             '(or one from the last persisted checkpoint)' % checkpoint.revision)
        self.close_runtime()
        self.world.save()
        started = time.perf_counter()
        self.runtime = self.runtime_factory(checkpoint)
        self.metric_sink('runtime_start_seconds', time.perf_counter() - started)
        return self.runtime

    def ensure_runtime(self, checkpoint):
        """Reuse the live runtime when it already serves ``checkpoint``.

        Starting one costs most of a round (Ray, FSDP workers, checkpoint load, vLLM), and
        after an update the runtime itself holds the new weights, so it stays up across
        rounds. Its workers pick up the round's new tasks and checkpoint from the
        rewritten worker manifest and registries.
        """
        runtime = self.runtime
        if runtime is not None and self.keep_runtime and runtime_checkpoint(runtime) == checkpoint:
            self.world.save()
            self._worker_manifest(checkpoint)
            return runtime
        return self.open_runtime(checkpoint)

    def close_runtime(self):
        runtime, self.runtime = self.runtime, None
        if runtime is not None:
            runtime.close()


def runtime_checkpoint(runtime):
    """The checkpoint a runtime currently holds (it moves forward after each update)."""
    return getattr(runtime, '_loaded_checkpoint', None) or runtime.checkpoint


class LazyBatch:
    """Defers runtime start-up so a fully recovered round never touches the NPUs."""
    def __init__(self, context, checkpoint):
        self.context, self.checkpoint = context, checkpoint

    def generate(self, requests):
        return self.context.ensure_runtime(self.checkpoint).service_batch.generate(requests)


class RoundBackend:
    """Runs the real VERL update on the live runtime, which stays up for the next round."""
    def __init__(self, context, index, checkpoint):
        self.context, self.index, self.checkpoint = context, index, checkpoint

    def budget_for(self, request):
        return self.context.update_budget

    def train(self, request):
        from rift.verl.serving import checkpoint_step
        from rift.verl.rollout_evidence import NativeServiceRolloutVerifier
        context, step = self.context, checkpoint_step(request.policy_before) + 1
        runtime = context.ensure_runtime(self.checkpoint)
        try:
            runtime.backend_bindings = {
                'rollout_verifier': NativeServiceRolloutVerifier(
                    context.capture_store.resolve_episode, context.capture_store),
                'run_recorder': context.ledger, 'update_verifier': context.ledger}
            backend = runtime.create_update_backend(request,
                output_dir=context.root / 'checkpoints' / 'service' / ('global-step-%08d' % step),
                after_revision=str(step), run_id=run_id_for(self.index), round_index=self.index,
                metric_sink=context.metric_sink,
                checkpoint_manifest_path=context.root / 'checkpoints' / ('service-step-%d.json' % step),
                mini_batch_size=context.update_budget.mini_batch_size,
                epochs=context.update_budget.epochs,
                persist=context.persists(self.index))
            completed = backend.train(request)
        except BaseException:
            context.close_runtime()    # a half-updated runtime must not serve another round
            raise
        if not context.keep_runtime:
            context.close_runtime()
        return completed


def run_id_for(index):
    return 'curriculum-%06d' % index


def build_recover_update(context):
    """Return an update recovered from a durable ledger receipt, else None."""
    from rift.schemas import Checkpoint
    from rift.training.contracts import CompletedUpdate

    def recover(index, request):
        run_id = run_id_for(index)
        path = context.ledger.root / (run_id + '.json')
        if not path.exists():
            return None
        row = json.loads(path.read_text(encoding='utf-8'))
        update = CompletedUpdate(request, Checkpoint(**row['after']), run_id, str(path.resolve()))
        if context.ledger.verify_update(update) is not True:
            raise ValueError('existing update receipt does not match the frozen request')
        return update
    return recover


def build_policy(name, arms, *, seed, window, fixed_family=None):
    from rift.evolution import curriculum
    if name == 'random':
        return curriculum.RandomPolicy(arms, seed=seed)
    if name == 'fixed':
        if not fixed_family:
            raise ValueError('the fixed policy needs --fixed-family')
        return curriculum.FixedFamilyPolicy(arms, family_id=fixed_family, seed=seed)
    if name == 'difficulty':
        return curriculum.DifficultyOnlyPolicy(arms, seed=seed)
    if name == 'learning_progress':
        return curriculum.LearningProgressPolicy(arms, seed=seed, window=window)
    if name == 'bandit':
        return curriculum.BanditPolicy(arms, seed=seed, window=window)
    raise ValueError('unknown policy: %s' % name)


def train(context, *, policy_name, arms, bench_specs, rounds, seed, package_config, window=10,
          fixed_family=None, init_from_run=None):
    """Run ``rounds`` curriculum rounds; resumable from the run directory."""
    from rift.benchmark.families import FAMILIES
    from rift.evolution.curriculum import CurriculumRunner, Selection
    from rift.training.package import FrozenBenchmarkTrainingPool
    from rift.training.task_stats import TaskStatsStore
    context.total_rounds = rounds
    if init_from_run and not (context.root / 'initial-checkpoint.json').exists():
        initial = context.adopt_initial_from(init_from_run)
    else:
        initial = context.initial_checkpoint()
    pool_path = context.root / 'bench-pool.json'
    if pool_path.exists():
        refs = tuple(json.loads(pool_path.read_text(encoding='utf-8'))['task_refs'])
    else:
        by_spec = context.world.certify_specs(bench_specs)
        refs = tuple(ref for key in sorted(by_spec) for ref in by_spec[key])
        context.world.save()
        pool_path.write_text(json.dumps({'task_refs': list(refs)}, sort_keys=True), encoding='utf-8')
    pool = FrozenBenchmarkTrainingPool.freeze('bench-' + sha256('|'.join(refs).encode()).hexdigest()[:12],
                                              refs, context.world.task_registry)
    selections = tuple(Selection(row['family_id'], row.get('knobs', {}), package_config.designer_task_count)
                       for row in arms)
    policy = build_policy(policy_name, selections, seed=seed, window=window, fixed_family=fixed_family)

    def collector_factory(checkpoint, index):
        from rift.runtime.evaluator import ProductionServiceRolloutCollector
        context.world.save()
        return ProductionServiceRolloutCollector(LazyBatch(context, checkpoint),
            task_registry=context.world.task_registry, capture_store=context.capture_store,
            metric_sink=context.metric_sink, round_index=index)

    runner = CurriculumRunner(policy=policy, families=FAMILIES, host_factory=context.world.host,
        benchmark_pool=pool, task_registry=context.world.task_registry, package_config=package_config,
        collector_factory=collector_factory,
        backend_factory=lambda index, checkpoint, directory: RoundBackend(context, index, checkpoint),
        recover_update=build_recover_update(context), checkpoint_store=context.checkpoint_store,
        update_verifier=context.ledger, run_root=context.root / 'curriculum', seed=seed,
        update_budget=context.update_budget,
        benchmark_selector=TaskStatsStore(context.root / 'task-stats.jsonl'))
    try:
        final, history = runner.run(initial, rounds)
    finally:
        context.close_runtime()
    manifest = context.latest_manifest()
    (context.root / 'final.json').write_text(json.dumps({
        'checkpoint': asdict(final), 'manifest': str(manifest.resolve()), 'rounds': rounds},
        sort_keys=True), encoding='utf-8')
    return {'status': 'curriculum_completed', 'rounds': len(history),
            'final_checkpoint': asdict(final),
            'observations': [asdict(item) for item in history], 'run_root': str(context.root)}


class PoolGroup:
    def __init__(self, task_ref, group_id, group_size, source):
        self.task_ref, self.group_id, self.group_size, self.source = task_ref, group_id, group_size, source


class PoolBatch:
    """The tasks of one round, drawn straight from certified single-task worlds.

    It has only what the rollout collector reads (ids, seed, groups); no Designer world,
    bandit or training package is involved.
    """
    def __init__(self, package_id, groups, sampling_seed):
        self.package_id, self.task_groups, self.sampling_seed = package_id, tuple(groups), sampling_seed

    @property
    def task_refs(self):
        return tuple(group.task_ref for group in self.task_groups)


def fit_request_to_batches(request, mini_batch_size, world_size=WORLD_SIZE):
    """Drop whole GRPO groups until the rollout count divides into complete mini-batches.

    A group whose rollouts are all right or all wrong has zero advantage, so those go first;
    informative groups are only dropped when nothing else is left to drop.
    """
    from dataclasses import replace
    unit = mini_batch_size * world_size // math.gcd(mini_batch_size, world_size)
    groups = {}
    for row in request.rollouts:
        groups.setdefault(row.group_id, []).append(row)
    sizes = {len(rows) for rows in groups.values()}
    if len(sizes) != 1:
        raise ValueError('groups of different sizes cannot be balanced into mini-batches')
    per_group = sizes.pop()
    keep = (len(groups) * per_group // unit) * unit // per_group      # groups that fit whole batches
    if keep == len(groups):
        return request
    if keep < 1:
        raise ValueError('fewer than one mini-batch of complete groups is left (%d rollouts)'
                         % len(request.rollouts))
    order = {group_id: index for index, group_id in enumerate(groups)}
    informative = lambda rows: len({row.reward for row in rows}) > 1
    ranked = sorted(groups, key=lambda group_id: (not informative(groups[group_id]), order[group_id]))
    kept = set(ranked[:keep])
    rows = tuple(row for row in request.rollouts if row.group_id in kept)
    return replace(request, rollouts=rows)


def _pool_collector(context, checkpoint, index):
    from rift.runtime.evaluator import ProductionServiceRolloutCollector
    from rift.training.service_data import prepare_service_trajectory_update

    class PoolCollector(ProductionServiceRolloutCollector):
        def _build_update(self, package, episodes, loaded):
            request = prepare_service_trajectory_update(package.package_id, set(package.task_refs),
                episodes, loaded, self.capture_store, algorithm='grpo', allow_partial_groups=True)
            return request, None

    return PoolCollector(LazyBatch(context, checkpoint), task_registry=context.world.task_registry,
        capture_store=context.capture_store, metric_sink=context.metric_sink, round_index=index)


def _pool_stats(request, groups):
    """Per-group learnability of one round, shaped for the curriculum policies."""
    from rift.rewards.learnability import LearnabilityStats, TaskLearnability
    by_id = {group.group_id: group for group in groups}
    observed = {}
    for row in request.rollouts:
        observed.setdefault(row.group_id, []).append(row)
    tasks = []
    for group_id, rows in observed.items():
        group = by_id[group_id]
        k = sum(int(row.reward) for row in rows)
        p = k / group.group_size
        tasks.append(TaskLearnability(group.task_ref, 'designer', k, group.group_size, 4 * p * (1 - p)))
    return LearnabilityStats(tuple(tasks), sum(t.learnability for t in tasks) / len(tasks), None,
                             sum(t.informative for t in tasks) / len(tasks), len(tasks), len(groups))


def train_pool(context, *, train_specs, rounds, tasks_per_round, seed, init_from_run=None,
               collector_factory=None, policy=None, heldout_specs=None, heldout_groups=4,
               heldout_seed=2000, heldout_label='x2base'):
    """Plain GRPO on a fixed pool of certified tasks.

    Without ``policy`` each round is a balanced sample of every class (X2). With one of the
    curriculum policies each round is ``tasks_per_round`` tasks of the single class it picks,
    scored by the learnability of that round's groups (X5). With ``heldout_specs`` the final
    policy is evaluated on those tasks by the still-running runtime, which also makes a
    checkpoint file unnecessary. A crashed run starts again.
    """
    import random
    import time
    rounds_path = context.root / 'rounds.jsonl'
    if rounds_path.exists():
        raise FileExistsError('this run directory already trained; use a new one')
    context.total_rounds = rounds
    if init_from_run and not (context.root / 'initial-checkpoint.json').exists():
        checkpoint = context.adopt_initial_from(init_from_run)
    else:
        checkpoint = context.initial_checkpoint()
    by_key = context.world.certify_specs(train_specs)
    heldout = context.world.certify_specs(heldout_specs) if heldout_specs else None
    context.world.save()
    keys = sorted(by_key)
    if policy is None:
        if tasks_per_round % len(keys) or any(len(by_key[key]) < tasks_per_round // len(keys) for key in keys):
            raise ValueError('tasks per round must split evenly over the classes and fit their pools')
        per_key = tasks_per_round // len(keys)
    else:
        arm_keys = {arm.arm_key for arm in policy.arms}
        if (not arm_keys <= set(by_key) or {arm.n_tasks for arm in policy.arms} != {tasks_per_round}
                or any(len(by_key[key]) < tasks_per_round for key in arm_keys)):
            raise ValueError('every arm needs a task pool of at least tasks_per_round certified instances')
    make_collector = collector_factory or (lambda loaded, index: _pool_collector(context, loaded, index))
    history, observations, report = [], [], None
    try:
        for index in range(rounds):
            started = time.perf_counter()
            rng = random.Random(seed + index)
            selection, groups = None, []
            if policy is None:
                chosen = [(key, ref) for key in keys for ref in rng.sample(by_key[key], per_key)]
            else:
                selection = policy.propose(index, tuple(observations))
                chosen = [(selection.arm_key, ref) for ref in rng.sample(by_key[selection.arm_key], tasks_per_round)]
            for key, ref in chosen:
                groups.append(PoolGroup(ref, sha256(('rift-pool-group-v1:%d:%s' % (index, ref)).encode()
                                                    ).hexdigest()[:32], context.group_size, key))
            batch = PoolBatch('pool-%06d-%s' % (index, sha256(
                '|'.join(group.task_ref for group in groups).encode()).hexdigest()[:12]), groups, seed + index)
            request, _ = make_collector(checkpoint, index).collect(batch, checkpoint)
            source = {group.group_id: group.source for group in groups}
            rewards = {}
            for row in request.rollouts:
                rewards.setdefault(source[row.group_id], []).append(row.reward)
            sampled = time.perf_counter()
            collected = len(request.rollouts)
            stats = _pool_stats(request, groups) if policy is not None else None
            request = fit_request_to_batches(request, context.update_budget.mini_batch_size)
            update = RoundBackend(context, index, checkpoint).train(request)
            checkpoint = update.policy_after
            row = {'round': index, 'revision': checkpoint.revision,
                   'train_success': {key: sum(values) / len(values) for key, values in sorted(rewards.items())},
                   'train_success_all': sum(sum(v) for v in rewards.values()) / sum(len(v) for v in rewards.values()),
                   'rollouts': collected, 'trained_rollouts': len(request.rollouts),
                   'sample_seconds': round(sampled - started, 1),
                   'update_seconds': round(time.perf_counter() - sampled, 1)}
            if policy is not None:
                observations.append(policy.observe(selection, stats))
                row.update(arm=selection.arm_key, learnability=stats.designer_mean,
                           informative_fraction=stats.designer_informative_fraction)
            history.append(row)
            with rounds_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
        if heldout:
            report = sample_outcomes(context, heldout, checkpoint, groups=heldout_groups,
                                     seed=heldout_seed, label=heldout_label)
    finally:
        context.close_runtime()
    manifest = context.latest_manifest()
    (context.root / 'final.json').write_text(json.dumps({
        'checkpoint': asdict(checkpoint), 'manifest': str(manifest.resolve()), 'rounds': rounds,
        'in_memory_only': checkpoint in context.checkpoint_store.virtual}, sort_keys=True), encoding='utf-8')
    result = {'status': 'pool_training_completed', 'rounds': rounds, 'final_checkpoint': asdict(checkpoint),
              'history': history, 'run_root': str(context.root)}
    if report is not None:
        result['heldout'] = {'status': 'evaluation_completed', 'label': heldout_label, 'seed': heldout_seed,
                             'groups_per_task': heldout_groups, 'group_size': context.group_size,
                             'checkpoint': asdict(checkpoint), 'families': report}
    return result


def build_eval_requests(by_key, checkpoint, *, groups, group_size, seed, label):
    """Train-mode GRPO groups on frozen tasks; the update step is never run."""
    from rift.runtime.evaluator import checkpoint_json
    from rift.verl.sampling import rollout_seed
    requests, owners = [], {}
    session_id = 0
    for key in sorted(by_key):
        for task_ref in by_key[key]:
            for group in range(groups):
                group_id = 'eval-%s-%s-%d' % (label, task_ref[:16], group)
                for member in range(group_size):
                    rollout_id = sha256(('rift-eval-v1:%s:%s:%d:%d' % (
                        label, task_ref, group, member)).encode()).hexdigest()
                    requests.append({'rift_request_id': rollout_id, 'rift_task_ref': task_ref,
                        'rift_mode': 'train', 'rift_group_size': group_size,
                        'rift_rollout_id': rollout_id,
                        'rift_sampling_seed': rollout_seed(seed, rollout_id),
                        'rift_customer_seed': rollout_seed(seed, task_ref),
                        'rift_expected_checkpoint': checkpoint_json(checkpoint),
                        'uid': group_id, 'session_id': session_id, 'raw_prompt': []})
                    owners[rollout_id] = (key, task_ref)
                    session_id += 1
    return requests, owners


def sample_outcomes(context, by_key, checkpoint, *, groups, seed, label):
    """Sample ``groups`` x group_size episodes per task on the live runtime; per-class report."""
    requests, owners = build_eval_requests(by_key, checkpoint, groups=groups,
        group_size=context.group_size, seed=seed, label=label)
    # One generate call only: the runtime puts vLLM to sleep after every call and
    # nothing wakes it before the next one, so a second call would wait forever.
    outputs = LazyBatch(context, checkpoint).generate(requests)
    if len(outputs) != len(requests):
        raise ValueError('VERL omitted evaluation requests')
    rows = []
    for request, output in zip(requests, outputs):
        if output.get('rift_request_id') != request['rift_request_id']:
            raise ValueError('evaluation output lost its request identity')
        key, task_ref = owners[request['rift_request_id']]
        outcome = output.get('rift_outcome')
        if output.get('rift_replay_verified') is not True:
            outcome = 'unknown'
        rows.append((key, task_ref, outcome))
    return summarize_outcomes(rows)


def evaluate(context, *, specs, groups, seed, label, from_run=None):
    """Sample ``groups`` x group_size episodes per task and report success rates."""
    by_key = context.world.certify_specs(specs)
    context.world.save()
    checkpoint = (context.checkpoint_from_run(from_run) if from_run else context.initial_checkpoint())
    try:
        report = sample_outcomes(context, by_key, checkpoint, groups=groups, seed=seed, label=label)
    finally:
        context.close_runtime()
    return {'status': 'evaluation_completed', 'label': label, 'checkpoint': asdict(checkpoint),
            'groups_per_task': groups, 'group_size': context.group_size, 'seed': seed,
            'from_run': None if from_run is None else str(from_run),
            'families': report, 'run_root': str(context.root)}


def _common(parser):
    parser.add_argument('--service-model', required=True)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--verl-config-dir', required=True)
    parser.add_argument('--episode-budget-config')
    parser.add_argument('--tool-profile', default='full9', choices=('full9', 'core4'))
    parser.add_argument('--seed', type=int, default=2000)


def _context(args, update_budget, group_size, response_length=8192, keep_runtime=True, persist_every=1):
    from usami_v2.runtime.runner import EpisodeBudget
    from rift.runtime.ascend_910b3 import _default_smoke_episode_budget
    budget = (EpisodeBudget.model_validate(json.loads(Path(args.episode_budget_config).read_text(encoding='utf-8')))
              if args.episode_budget_config else _default_smoke_episode_budget())
    return ExperimentContext(root=args.run_root, service_model=args.service_model,
        verl_config_dir=args.verl_config_dir, episode_budget=budget,
        update_budget=update_budget, tool_profile=args.tool_profile, group_size=group_size,
        response_length=response_length, keep_runtime=keep_runtime, persist_every=persist_every)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='RIFT curriculum training and checkpoint evaluation on 8 Ascend 910B3 NPUs')
    sub = parser.add_subparsers(dest='command', required=True)
    train_parser = sub.add_parser('train', help='run CurriculumRunner rounds')
    _common(train_parser)
    train_parser.add_argument('--policy', choices=POLICIES, default='bandit')
    train_parser.add_argument('--fixed-family')
    train_parser.add_argument('--rounds', type=int, required=True)
    train_parser.add_argument('--arms', help='JSON list of {family_id, knobs}; inline or file')
    train_parser.add_argument('--bench-specs', help='JSON list of benchmark task specs; inline or file')
    train_parser.add_argument('--tasks-per-round', type=int, default=16)
    train_parser.add_argument('--group-size', type=int, default=4)
    train_parser.add_argument('--window', type=int, default=10)
    train_parser.add_argument('--mini-batch-size', type=int, default=WORLD_SIZE)
    train_parser.add_argument('--epochs', type=int, default=1)
    train_parser.add_argument('--learning-rate', type=float, default=1e-6)
    train_parser.add_argument('--response-length', type=int, default=12288,
        help='VERL aborts the whole round when one episode outgrows this')
    train_parser.add_argument('--persist-every', type=int, default=0,
        help='write real checkpoint files every N rounds; 0 (default) writes only after the '
             'last round. Rounds in between update the live runtime only, which is much faster')
    train_parser.add_argument('--init-from-run',
        help='an earlier run directory whose base checkpoint to start from (skips writing a new one)')
    train_parser.add_argument('--restart-each-round', action='store_true',
        help='tear the NPU runtime down after every update (slow; the default keeps it up)')
    pool_parser = sub.add_parser('train-pool', help='plain GRPO on a fixed pool of tasks (X2)')
    _common(pool_parser)
    pool_parser.add_argument('--train-specs', required=True, help='JSON list of task specs; inline or file')
    pool_parser.add_argument('--rounds', type=int, required=True)
    pool_parser.add_argument('--tasks-per-round', type=int, default=16)
    pool_parser.add_argument('--group-size', type=int, default=4)
    pool_parser.add_argument('--mini-batch-size', type=int, default=16)
    pool_parser.add_argument('--epochs', type=int, default=1)
    pool_parser.add_argument('--learning-rate', type=float, default=1e-6)
    pool_parser.add_argument('--response-length', type=int, default=12288)
    pool_parser.add_argument('--persist-every', type=int, default=0)
    pool_parser.add_argument('--no-final-checkpoint', action='store_true',
        help='never write checkpoint files; use with --heldout-specs, which evaluates in place')
    pool_parser.add_argument('--policy', choices=POLICIES,
        help='pick one class per round with a curriculum policy (X5); default is a balanced mix (X2)')
    pool_parser.add_argument('--arms', help='JSON list of {family_id, knobs}; inline or file (with --policy)')
    pool_parser.add_argument('--window', type=int, default=10)
    pool_parser.add_argument('--heldout-specs', help='evaluate the final policy on these tasks, in place')
    pool_parser.add_argument('--heldout-groups', type=int, default=4)
    pool_parser.add_argument('--init-from-run')
    pool_parser.add_argument('--restart-each-round', action='store_true')
    eval_parser = sub.add_parser('eval', help='evaluate a frozen checkpoint without updating it')
    _common(eval_parser)
    eval_parser.add_argument('--specs', help='JSON list of {family, knobs, instances, seed}; inline or file')
    eval_parser.add_argument('--groups', type=int, default=2, help='GRPO groups sampled per task')
    eval_parser.add_argument('--group-size', type=int, default=4)
    eval_parser.add_argument('--from-run', help='train run directory whose final.json supplies the checkpoint')
    eval_parser.add_argument('--label', default='x0')
    eval_parser.add_argument('--response-length', type=int, default=20480,
        help='VERL aborts the whole batch when one episode outgrows this, so evaluation is generous')
    args = parser.parse_args(argv)
    from rift.evolution.matched_control import UpdateBudget
    from rift.verl.version import require_verl_version
    require_verl_version()
    if args.command == 'train':
        from rift.training.package import TrainingPackageConfig
        budget = UpdateBudget(args.mini_batch_size, args.epochs, args.learning_rate)
        package_config = TrainingPackageConfig(tasks_per_round=args.tasks_per_round,
            sampling_seed=args.seed, group_size=args.group_size)
        context = _context(args, budget, args.group_size, response_length=args.response_length,
                           keep_runtime=not args.restart_each_round, persist_every=args.persist_every)
        result = train(context, policy_name=args.policy, rounds=args.rounds, seed=args.seed,
            arms=load_json_argument(args.arms) if args.arms else list(DEFAULT_ARMS),
            bench_specs=load_json_argument(args.bench_specs) if args.bench_specs else list(DEFAULT_BENCH),
            package_config=package_config, window=args.window, fixed_family=args.fixed_family,
            init_from_run=args.init_from_run)
    elif args.command == 'train-pool':
        from rift.evolution.curriculum import Selection
        budget = UpdateBudget(args.mini_batch_size, args.epochs, args.learning_rate)
        context = _context(args, budget, args.group_size, response_length=args.response_length,
                           keep_runtime=not args.restart_each_round, persist_every=args.persist_every)
        context.persist_final = not args.no_final_checkpoint
        policy = None
        if args.policy:
            if not args.arms:
                parser.error('--policy needs --arms')
            selections = tuple(Selection(row['family_id'], row.get('knobs', {}), args.tasks_per_round)
                               for row in load_json_argument(args.arms))
            policy = build_policy(args.policy, selections, seed=args.seed, window=args.window)
        result = train_pool(context, train_specs=load_json_argument(args.train_specs),
            rounds=args.rounds, tasks_per_round=args.tasks_per_round, seed=args.seed,
            init_from_run=args.init_from_run, policy=policy,
            heldout_specs=load_json_argument(args.heldout_specs) if args.heldout_specs else None,
            heldout_groups=args.heldout_groups)
        if 'heldout' in result:
            (context.root / 'eval-result.json').write_text(
                json.dumps(result['heldout'], sort_keys=True, indent=2), encoding='utf-8')
    else:
        context = _context(args, UpdateBudget(WORLD_SIZE, 1, 1e-6), args.group_size,
                           response_length=args.response_length)
        result = evaluate(context,
            specs=load_json_argument(args.specs) if args.specs else list(DEFAULT_BENCH),
            groups=args.groups, seed=args.seed, label=args.label, from_run=args.from_run)
    (context.root / ('%s-result.json' % args.command)).write_text(
        json.dumps(result, sort_keys=True, indent=2), encoding='utf-8')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
