"""Production runtime assembly from explicit deployment-owned trust roots."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rift.banks.bootstrap import BootstrapResult, BootstrapRunner
from rift.evolution.cli import EvolutionRunRuntime
from rift.evolution.scheduler import EvolutionScheduler
from rift.runtime.config import ProductionConfig
from rift.runtime.evidence import JsonlMetricSink, ProductionBankUpdater, prepare_run_directory
from rift.runtime.plan_factory import ProductionPlanFactory
from rift.schemas import Checkpoint


@dataclass(frozen=True)
class ProductionDeployment:
    """Real environment bindings supplied by the target application image.

    This object contains actual model/tokenizer paths, certified Usami data,
    split/attestation roots, and role factories that create official VERL
    WorkerGroups. It has no test-double or synthetic default.
    """
    paths: dict
    bootstrap: BootstrapResult
    bootstrap_runner: BootstrapRunner
    initial_service: Checkpoint
    initial_designer: Checkpoint
    resources: object
    task_evidence_store: object
    plan_dependencies: dict
    rounds: int
    bank_updater: Callable | None = None
    independent_test_runner: object | None = None
    test_every: int = 0

    def __post_init__(self):
        if (not isinstance(self.bootstrap, BootstrapResult)
                or not isinstance(self.bootstrap_runner, BootstrapRunner)
                or not isinstance(self.initial_service, Checkpoint)
                or not isinstance(self.initial_designer, Checkpoint)
                or self.bootstrap.checkpoint != self.initial_service
                or not callable(getattr(self.resources, "acquire", None))
                or not callable(getattr(self.task_evidence_store, "save", None))
                or not callable(getattr(self.task_evidence_store, "load", None))
                or not isinstance(self.plan_dependencies, dict)
                or type(self.rounds) is not int or self.rounds < 1
                or type(self.test_every) is not int or self.test_every < 0):
            raise ValueError("complete real Bootstrap, checkpoints, evidence and role runtime bindings required")
        if self.test_every and not callable(getattr(self.independent_test_runner, "run", None)):
            raise ValueError("independent TEST schedule requires a production test runner")


def _config_snapshot(config):
    import yaml
    return yaml.safe_dump({"rift": config.rift, "verl": config.verl,
        "ascend": config.ascend, "performance_enabled": config.performance_enabled},
        allow_unicode=True, sort_keys=True)


def build_production_runtime(config, deployment, *, run_root):
    """Create directories and wire the existing scheduler/coordinator/backend path."""
    if not isinstance(config, ProductionConfig):
        raise TypeError("validated ProductionConfig required")
    config.validate()
    if not isinstance(deployment, ProductionDeployment):
        raise TypeError("deployment factory must return ProductionDeployment")
    config.require_deployment_paths(deployment.paths)
    if Path(deployment.paths['run_root']).resolve() != Path(run_root).resolve():
        raise ValueError("deployment run_root and CLI run_root must identify the same run directory")
    root = prepare_run_directory(run_root, config_snapshot=_config_snapshot(config),
        manifest={"requested_verl_commit": config.ascend["verl_commit"],
                  "target_platform": config.ascend.get("platform"),
                  "target_accelerator": config.ascend.get("accelerator"),
                  "target_device_count": config.ascend.get("devices"),
                  "requested_versions": {key: config.ascend.get(key) for key in (
                      'python', 'cann', 'torch', 'torch_npu', 'verl', 'vllm',
                      'vllm_ascend', 'triton_ascend', 'numpy', 'transformers')}})
    metrics = JsonlMetricSink(root / 'metrics.jsonl')
    deployment.bootstrap.bank_manager.save(root / "bootstrap" / "banks.json")
    for record in deployment.bootstrap.records:
        deployment.task_evidence_store.save(record)

    dependencies = dict(deployment.plan_dependencies)
    checkpoint_store = dependencies.get('checkpoint_store')
    if (not callable(getattr(checkpoint_store, 'verify', None))
            or not callable(getattr(checkpoint_store, 'save_manifest', None))
            or not callable(getattr(checkpoint_store, 'resolve', None))):
        raise ValueError('production deployment requires the real FSDP checkpoint store')
    for checkpoint in (deployment.initial_service, deployment.initial_designer):
        if checkpoint_store.verify(checkpoint) is not True:
            raise ValueError(f'initial {checkpoint.policy} checkpoint is not registered and verified')
    initial_manifest = root / 'checkpoints' / 'initial.json'
    if initial_manifest.exists():
        from rift.verl.checkpoints import FSDPCheckpointStore
        restored_checkpoints = FSDPCheckpointStore.load_manifest(initial_manifest)
        for checkpoint in (deployment.initial_service, deployment.initial_designer):
            if (not restored_checkpoints.verify(checkpoint)
                    or restored_checkpoints.resolve(checkpoint) != checkpoint_store.resolve(checkpoint)):
                raise ValueError('initial checkpoint manifest differs from deployment checkpoint store')
    else:
        checkpoint_store.save_manifest(initial_manifest)
    from rift.replay.splits import SplitRegistry
    from rift.verl.task_registry import CertifiedTaskRegistry
    task_registry = dependencies.get("task_registry")
    split_registry = dependencies.get("split_registry")
    commerce_host = dependencies.get("commerce_host")
    if (not isinstance(task_registry, CertifiedTaskRegistry)
            or not isinstance(split_registry, SplitRegistry)
            or getattr(commerce_host, "task_registry", None) is not task_registry
            or task_registry.split_registry is not split_registry):
        raise ValueError("CommerceHost, CertifiedTaskRegistry and SplitRegistry must share one authority")
    task_registry_path = root / "task_registry" / "certified-tasks.json"
    split_registry_path = root / "task_registry" / "splits.json"
    if task_registry_path.exists():
        restored = CertifiedTaskRegistry.load(task_registry_path, split_registry)
        if (task_registry.task_refs != restored.task_refs
                or any(task_registry.resolve_record(ref) != restored.resolve_record(ref)
                       for ref in task_registry.task_refs)):
            raise ValueError("deployment task registry differs from the persisted run registry")
    else:
        task_registry.save(task_registry_path)
    task_registry.storage_path = task_registry_path
    if split_registry_path.exists():
        restored_splits = SplitRegistry.load(split_registry_path)
        if (restored_splits.policy_version != split_registry.policy_version
                or set(restored_splits.identities) != set(split_registry.identities)):
            raise ValueError("deployment split registry differs from the persisted run registry")
    else:
        split_registry.save(split_registry_path)
    split_registry.persistence_path = split_registry_path
    if dependencies.get("resources", deployment.resources) is not deployment.resources:
        raise ValueError("plan and deployment must use the same exclusive NPU resource manager")
    dependencies["resources"] = deployment.resources
    existing_checkpoint_root = dependencies.get('checkpoint_root')
    if (existing_checkpoint_root is not None
            and Path(existing_checkpoint_root).resolve() != (root / 'checkpoints').resolve()):
        raise ValueError('checkpoint outputs must be stored under this run root')
    dependencies['checkpoint_root'] = root / 'checkpoints'
    if dependencies.get('metric_sink') is not None and dependencies['metric_sink'] is not metrics:
        raise ValueError("production metrics must be written to this run's metrics.jsonl")
    dependencies['metric_sink'] = metrics
    plan_factory = ProductionPlanFactory(**dependencies)
    if deployment.bank_updater is None:
        bank_updater = ProductionBankUpdater(
            deployment.task_evidence_store, dependencies["task_registry"],
            dependencies["trial_verifier"])
    else:
        bank_updater = deployment.bank_updater
    scheduler = EvolutionScheduler(
        bootstrap_runner=deployment.bootstrap_runner,
        task_evidence_store=deployment.task_evidence_store,
        state_root=root / "round_states",
        success_threshold=config.rift["banks"]["success_threshold"],
        failure_threshold=config.rift["banks"]["failure_threshold"],
        failure_context_limit=config.rift["banks"]["failure_context_limit"],
        seed=deployment.bootstrap.seed, metric_sink=metrics)
    return EvolutionRunRuntime(
        scheduler=scheduler,
        bootstrap=deployment.bootstrap,
        initial_service=deployment.initial_service,
        initial_designer=deployment.initial_designer,
        plan_factory=plan_factory,
        bank_updater=bank_updater,
        rounds=deployment.rounds,
        bank_manifest_path=root / "evolving-banks.json",
        independent_test_runner=deployment.independent_test_runner,
        test_every=deployment.test_every,
        ability_registry_path=root / config.rift["ability_registry"]["persistence_file"])
