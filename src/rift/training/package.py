"""Frozen, source-auditable task packages for one dual-bank Service update."""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path
import random

from rift.schemas import Checkpoint, DataPurpose, Identity, Split


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return sha256(_canonical(value).encode("utf-8")).hexdigest()


def _record_for_train(task_registry, task_ref):
    record = task_registry.resolve_record(task_ref)
    if (record.identity.task != task_ref or record.identity.split is not Split.TRAIN
            or record.identity.purpose is not DataPurpose.TRAIN or record.task.partition != "train"
            or record.certificate.status != "CERTIFIED"
            or not record.certificate.full_task_reachability_certified
            or task_registry.verify_task(task_ref, record.world, record.task, record.profile,
                                         purpose=DataPurpose.TRAIN) is not True):
        raise ValueError("training packages require certified Usami TRAIN tasks")
    return record


@dataclass(frozen=True)
class TrainingPackageConfig:
    """Per-round task and rollout budget. Fractions must yield exact task counts."""
    designer_worlds_per_round: int = 1
    tasks_per_round: int = 16
    designer_fraction: float = 0.5
    benchmark_fraction: float = 0.5
    sampling_seed: int = 42
    group_size: int = 4
    max_candidate_challenges: int = 32

    def __post_init__(self):
        if (type(self.designer_worlds_per_round) is not int or self.designer_worlds_per_round != 1
                or type(self.tasks_per_round) is not int
                or self.tasks_per_round < 2 or type(self.sampling_seed) is not int
                or self.sampling_seed < 0 or type(self.group_size) is not int
                or self.group_size < 2 or type(self.max_candidate_challenges) is not int
                or self.max_candidate_challenges < 1
                or not isfinite(self.designer_fraction) or not isfinite(self.benchmark_fraction)
                or self.designer_fraction <= 0 or self.benchmark_fraction <= 0
                or not isclose(self.designer_fraction + self.benchmark_fraction, 1.0)):
            raise ValueError("one World, positive task/group budgets and valid training mix required")
        if (not isclose(self.designer_fraction * self.tasks_per_round,
                        round(self.designer_fraction * self.tasks_per_round))
                or not isclose(self.benchmark_fraction * self.tasks_per_round,
                               round(self.benchmark_fraction * self.tasks_per_round))):
            raise ValueError("training fractions must produce exact task counts")
        if self.designer_task_count < 1 or self.benchmark_task_count < 1:
            raise ValueError("both Designer and Benchmark sources need training groups")

    @property
    def designer_task_count(self):
        return round(self.tasks_per_round * self.designer_fraction)

    @property
    def benchmark_task_count(self):
        return round(self.tasks_per_round * self.benchmark_fraction)

    def seed_for_round(self, round_index):
        if type(round_index) is not int or round_index < 0:
            raise ValueError("non-negative round index required")
        return self.sampling_seed + round_index


@dataclass(frozen=True)
class FrozenBenchmarkTrainingPool:
    """Immutable manifest of previously certified native Usami TRAIN tasks."""
    pool_id: str
    task_refs: tuple[str, ...]
    pool_digest: str

    def __post_init__(self):
        if (not self.pool_id or not self.task_refs or len(set(self.task_refs)) != len(self.task_refs)
                or any(not ref for ref in self.task_refs)
                or len(self.pool_digest) != 64
                or any(ch not in "0123456789abcdef" for ch in self.pool_digest)):
            raise ValueError("frozen Benchmark task manifest required")

    @classmethod
    def freeze(cls, pool_id, task_refs, task_registry):
        refs = tuple(task_refs)
        if not refs or len(set(refs)) != len(refs):
            raise ValueError("unique pre-certified Benchmark TaskRefs required")
        records = [_record_for_train(task_registry, ref) for ref in refs]
        payload = [{"task_ref": ref, "world_hash": row.world.state_hash,
                    "identity": asdict(row.identity), "certificate_hash": row.certificate.record_hash}
                   for ref, row in zip(refs, records)]
        return cls(pool_id, refs, _digest(payload))

    def verify(self, task_registry):
        try:
            rebuilt = self.freeze(self.pool_id, self.task_refs, task_registry)
            return rebuilt == self
        except (KeyError, TypeError, ValueError):
            return False

    def save(self, path):
        Path(path).write_text(_canonical(asdict(self)), encoding="utf-8")

    @classmethod
    def load(cls, path, task_registry):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if set(raw) != {"pool_id", "task_refs", "pool_digest"}:
            raise ValueError("unsupported frozen Benchmark pool manifest")
        pool = cls(raw["pool_id"], tuple(raw["task_refs"]), raw["pool_digest"])
        if not pool.verify(task_registry):
            raise ValueError("frozen Benchmark pool no longer matches certified task records")
        return pool


@dataclass(frozen=True)
class PackageTaskSource:
    task_ref: str
    source: str
    world_hash: str
    identity: Identity
    certificate_hash: str

    def __post_init__(self):
        if (not self.task_ref or self.source not in ("designer", "benchmark")
                or not self.world_hash or self.identity.task != self.task_ref
                or self.identity.world != self.world_hash or self.identity.split is not Split.TRAIN
                or self.identity.purpose is not DataPurpose.TRAIN
                or len(self.certificate_hash) != 64
                or any(ch not in "0123456789abcdef" for ch in self.certificate_hash)):
            raise ValueError("package task source must bind a certified TRAIN identity")


@dataclass(frozen=True)
class PackageTaskGroup:
    task_ref: str
    group_id: str
    group_size: int
    source: str

    def __post_init__(self):
        if (not self.task_ref or not self.group_id or type(self.group_size) is not int
                or self.group_size < 2 or self.source not in ("designer", "benchmark")):
            raise ValueError("complete single-task GRPO group assignment required")


@dataclass(frozen=True)
class TrainingPackageStats:
    rollout_count: int
    token_count: int
    optimization_sample_count: int
    designer_rollout_count: int
    benchmark_rollout_count: int
    designer_token_count: int
    benchmark_token_count: int
    designer_optimization_samples: int
    benchmark_optimization_samples: int

    def __post_init__(self):
        values = (self.rollout_count, self.token_count, self.optimization_sample_count,
                  self.designer_rollout_count, self.benchmark_rollout_count,
                  self.designer_token_count, self.benchmark_token_count,
                  self.designer_optimization_samples, self.benchmark_optimization_samples)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("non-negative actual Service training counts required")
        if (self.designer_rollout_count + self.benchmark_rollout_count != self.rollout_count
                or self.designer_token_count + self.benchmark_token_count != self.token_count
                or self.designer_optimization_samples + self.benchmark_optimization_samples
                    != self.optimization_sample_count
                or self.optimization_sample_count > self.token_count):
            raise ValueError("per-source Service training counts do not add up")


@dataclass(frozen=True)
class TrainingPackage:
    """Versioned evidence and source manifest for a single Service update."""
    package_id: str
    proposal_id: str
    ability_choice: str
    target_ability_id: str
    designer_action_ref: str
    designer_world_ref: str
    designer_world_hash: str
    distribution_hash: str
    world_seed: int
    scenario_hash: str
    build_hash: str
    build_result_json: str
    build_verification_json: str
    challenge_results_json: tuple[str, ...]
    designer_task_refs: tuple[str, ...]
    benchmark_task_refs: tuple[str, ...]
    task_sources: tuple[PackageTaskSource, ...]
    task_groups: tuple[PackageTaskGroup, ...]
    benchmark_pool_id: str
    benchmark_pool_digest: str
    sampling_seed: int
    mix_configuration: TrainingPackageConfig
    service_checkpoint_before: Checkpoint
    stats: TrainingPackageStats | None = None
    schema_version: int = 1

    def __post_init__(self):
        if (self.schema_version != 1 or not all((self.package_id, self.proposal_id,
                self.ability_choice, self.target_ability_id, self.designer_action_ref,
                self.designer_world_ref, self.designer_world_hash, self.distribution_hash,
                self.scenario_hash, self.build_hash, self.benchmark_pool_id))
                or type(self.world_seed) is not int or type(self.sampling_seed) is not int
                or self.world_seed < 0 or self.sampling_seed < 0
                or not isinstance(self.service_checkpoint_before, Checkpoint)
                or not isinstance(self.mix_configuration, TrainingPackageConfig)):
            raise ValueError("incomplete versioned TrainingPackage evidence")
        if self.ability_choice != "NEW" and self.ability_choice != self.target_ability_id:
            raise ValueError("selected existing ability differs from the package target")
        config = self.mix_configuration
        if (self.sampling_seed < config.sampling_seed
                or len(self.designer_task_refs) != config.designer_task_count
                or len(self.benchmark_task_refs) != config.benchmark_task_count
                or len(self.task_refs) != config.tasks_per_round
                or len(set(self.task_refs)) != len(self.task_refs)
                or len(self.task_sources) != len(self.task_refs)
                or len(self.task_groups) != len(self.task_refs)
                or len({row.task_ref for row in self.task_sources}) != len(self.task_sources)
                or len({row.task_ref for row in self.task_groups}) != len(self.task_groups)
                or len({row.group_id for row in self.task_groups}) != len(self.task_groups)):
            raise ValueError("training package task counts and source records differ from its mix")
        if (len(self.benchmark_pool_digest) != 64
                or any(ch not in "0123456789abcdef" for ch in self.benchmark_pool_digest)):
            raise ValueError("Benchmark pool digest must be SHA-256")
        sources = {row.task_ref: row.source for row in self.task_sources}
        groups = {row.task_ref: row for row in self.task_groups}
        if (set(sources) != set(self.task_refs) or set(groups) != set(self.task_refs)
                or any(sources[ref] != "designer" for ref in self.designer_task_refs)
                or any(sources[ref] != "benchmark" for ref in self.benchmark_task_refs)
                or any(row.group_size != config.group_size or row.source != sources[row.task_ref]
                       for row in self.task_groups)
                or any(row.world_hash != self.designer_world_hash for row in self.task_sources
                       if row.source == "designer")):
            raise ValueError("package source, frozen World and per-task GRPO groups disagree")
        build = json.loads(self.build_result_json)
        verification = json.loads(self.build_verification_json)
        if (build.get("status") != "SAT" or build.get("world_hash") != self.designer_world_hash
                or build.get("scenario_hash") != self.scenario_hash or build.get("seed") != self.world_seed
                or verification.get("verified") is not True
                or verification.get("status") != "SAT"
                or verification.get("world_hash") != self.designer_world_hash):
            raise ValueError("package World must be the verified native SAT BuildResult")
        for item in self.challenge_results_json:
            value = json.loads(item)
            if not isinstance(value, dict) or "grounding_status" not in value:
                raise ValueError("native challenge statuses and reasons must be retained")
        if self.stats is not None:
            if not isinstance(self.stats, TrainingPackageStats):
                raise ValueError("invalid Service package statistics")
            expected_designer = len(self.designer_task_refs) * config.group_size
            expected_benchmark = len(self.benchmark_task_refs) * config.group_size
            actual_designer_groups = self.stats.designer_rollout_count // config.group_size
            actual_benchmark_groups = self.stats.benchmark_rollout_count // config.group_size
            if (self.stats.rollout_count > expected_designer + expected_benchmark
                    or self.stats.designer_rollout_count > expected_designer
                    or self.stats.benchmark_rollout_count > expected_benchmark
                    or self.stats.designer_rollout_count % config.group_size
                    or self.stats.benchmark_rollout_count % config.group_size
                    or actual_designer_groups * config.benchmark_task_count
                        != actual_benchmark_groups * config.designer_task_count):
                raise ValueError("actual groups exceed or fragment the frozen source mix")

    @property
    def task_refs(self):
        return self.designer_task_refs + self.benchmark_task_refs

    @property
    def task_source_by_ref(self):
        return {row.task_ref: row.source for row in self.task_sources}

    @property
    def group_by_task_ref(self):
        return {row.task_ref: row for row in self.task_groups}

    def assert_disjoint_from(self, *snapshots):
        for snapshot in snapshots:
            for probe in snapshot.probes:
                if any(source.identity.overlaps_under(probe.identity, snapshot.split_policy_version)
                       for source in self.task_sources):
                    raise ValueError("mixed training package overlaps Recovery/Retention scoring tasks")
        return True

    def record_service_request(self, request):
        from rift.training.contracts import UpdateRequest
        if (not isinstance(request, UpdateRequest) or request.package_id != self.package_id
                or request.policy_before != self.service_checkpoint_before
                or request.algorithm.lower() != "grpo"
                or not {row.task_id for row in request.rollouts}.issubset(set(self.task_refs))):
            raise ValueError("Service GRPO request is outside the frozen training package")
        groups = self.group_by_task_ref
        grouped = {}
        for row in request.rollouts:
            expected = groups[row.task_id]
            if row.group_id != expected.group_id:
                raise ValueError("Service rollout is bound to another task's GRPO group")
            grouped.setdefault(row.task_id, []).append(row)
        if any(len(rows) != groups[task_ref].group_size for task_ref, rows in grouped.items()):
            raise ValueError("actual Service group size differs from the package mix")
        if not grouped:
            raise ValueError("no complete known-outcome package GRPO groups remain")
        realized_designer_tasks = sum(groups[ref].source == "designer" for ref in grouped)
        realized_benchmark_tasks = sum(groups[ref].source == "benchmark" for ref in grouped)
        if (realized_designer_tasks * self.mix_configuration.benchmark_task_count
                != realized_benchmark_tasks * self.mix_configuration.designer_task_count):
            raise ValueError("UNKNOWN group removal changed the configured source ratio")
        rows_by_source = {"designer": [], "benchmark": []}
        for row in request.rollouts:
            rows_by_source[groups[row.task_id].source].append(row)
        counts = {}
        for source, rows in rows_by_source.items():
            counts[source] = (len(rows), sum(len(row.tokens.response) for row in rows),
                              sum(sum(row.tokens.mask) for row in rows))
        stats = TrainingPackageStats(
            rollout_count=len(request.rollouts),
            token_count=sum(value[1] for value in counts.values()),
            optimization_sample_count=sum(value[2] for value in counts.values()),
            designer_rollout_count=counts["designer"][0],
            benchmark_rollout_count=counts["benchmark"][0],
            designer_token_count=counts["designer"][1],
            benchmark_token_count=counts["benchmark"][1],
            designer_optimization_samples=counts["designer"][2],
            benchmark_optimization_samples=counts["benchmark"][2])
        return replace(self, stats=stats)

    def verify_service_request(self, request):
        try:
            return self.stats is not None and self.record_service_request(request) == self
        except (KeyError, TypeError, ValueError):
            return False

    @classmethod
    def from_dict(cls, value):
        raw = dict(value)
        raw["mix_configuration"] = TrainingPackageConfig(**raw["mix_configuration"])
        raw["service_checkpoint_before"] = Checkpoint(**raw["service_checkpoint_before"])
        raw["task_sources"] = tuple(PackageTaskSource(
            row["task_ref"], row["source"], row["world_hash"],
            Identity(row["identity"]["world"], row["identity"]["task"],
                row["identity"]["family"], row["identity"]["lineage"],
                Split(row["identity"]["split"]),
                None if row["identity"].get("purpose") is None else
                    DataPurpose(row["identity"]["purpose"])), row["certificate_hash"])
            for row in raw["task_sources"])
        raw["task_groups"] = tuple(PackageTaskGroup(**row) for row in raw["task_groups"])
        raw["designer_task_refs"] = tuple(raw["designer_task_refs"])
        raw["benchmark_task_refs"] = tuple(raw["benchmark_task_refs"])
        raw["challenge_results_json"] = tuple(raw["challenge_results_json"])
        if raw.get("stats") is not None:
            raw["stats"] = TrainingPackageStats(**raw["stats"])
        return cls(**raw)


def assemble_training_package(*, package_id, proposal, designer_action_ref,
        target_ability_id, designer_result, benchmark_pool, task_registry,
        service_checkpoint_before, config=TrainingPackageConfig(), round_index=0):
    """Select an exact Designer/Benchmark task mix from certified task records."""
    from rift.tasks import Proposal
    if (not isinstance(proposal, Proposal) or not proposal.is_training_plan
            or proposal.proposal_id != designer_result.proposal_id
            or proposal.ability_id not in ("NEW", target_ability_id)
            or not designer_action_ref or not target_ability_id):
        raise ValueError("one captured Designer plan and its resolved ability are required")
    if len(proposal.challenges) > config.max_candidate_challenges:
        raise ValueError("Designer challenge list exceeds its bounded screening budget")
    if (designer_result.build.status != "SAT"
            or designer_result.build_verification.get("verified") is not True
            or designer_result.build_verification.get("status") != "SAT"):
        raise ValueError("Designer World did not pass native build verification")
    if not benchmark_pool.verify(task_registry):
        raise ValueError("Benchmark pool is not a frozen set of certified Usami TRAIN tasks")
    successes = tuple(designer_result.tasks)
    if len(successes) < config.designer_task_count:
        raise ValueError("certified Designer tasks are insufficient; exact source ratio cannot be met")
    if len(benchmark_pool.task_refs) < config.benchmark_task_count:
        raise ValueError("certified Benchmark pool is too small; exact source ratio cannot be met")

    designer_tasks = successes[:config.designer_task_count]
    seed = config.seed_for_round(round_index)
    benchmark_refs = tuple(random.Random(seed).sample(
        sorted(benchmark_pool.task_refs), config.benchmark_task_count))
    sources = []
    for source, refs in (("designer", tuple(row.record_hash for row in designer_tasks)),
                         ("benchmark", benchmark_refs)):
        for task_ref in refs:
            record = _record_for_train(task_registry, task_ref)
            if source == "designer" and record.world.state_hash != designer_result.build.world_hash:
                raise ValueError("Designer task escaped the single frozen World")
            sources.append(PackageTaskSource(task_ref, source, record.world.state_hash,
                                             record.identity, record.certificate.record_hash))
    refs = tuple(row.task_ref for row in sources)
    groups = tuple(PackageTaskGroup(task_ref=task_ref,
        group_id=sha256(f"rift-grpo-package-v1:{package_id}:{task_ref}".encode()).hexdigest()[:32],
        group_size=config.group_size, source=source.source) for task_ref, source in
        ((row.task_ref, row) for row in sources))
    build = designer_result.build
    return TrainingPackage(
        package_id=package_id, proposal_id=proposal.proposal_id,
        ability_choice=proposal.ability_id, target_ability_id=target_ability_id,
        designer_action_ref=designer_action_ref,
        designer_world_ref=designer_result.world_ref,
        designer_world_hash=build.world_hash,
        distribution_hash=designer_result.distribution_hash,
        world_seed=build.seed, scenario_hash=build.scenario_hash,
        build_hash=build.record_hash,
        build_result_json=_canonical(build.model_dump(mode="json")),
        build_verification_json=_canonical(designer_result.build_verification),
        challenge_results_json=tuple(_canonical(row) for row in designer_result.challenge_results),
        designer_task_refs=tuple(row.record_hash for row in designer_tasks),
        benchmark_task_refs=benchmark_refs, task_sources=tuple(sources), task_groups=groups,
        benchmark_pool_id=benchmark_pool.pool_id, benchmark_pool_digest=benchmark_pool.pool_digest,
        sampling_seed=seed, mix_configuration=config,
        service_checkpoint_before=service_checkpoint_before)
