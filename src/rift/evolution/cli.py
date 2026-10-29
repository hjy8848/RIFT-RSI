"""Multi-round entry point over an application-owned, evidence-backed runtime.

Importing this module starts no training and loads no model. A runtime factory is
the explicit boundary where a deployment wires real Usami, VERL and trust roots.
"""
import argparse
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Callable

from rift.banks.bootstrap import BootstrapResult
from rift.evolution.scheduler import EvolutionScheduler, EvolutionPlanFactory
from rift.schemas import Checkpoint


@dataclass(frozen=True)
class EvolutionRunRuntime:
    scheduler: EvolutionScheduler
    bootstrap: BootstrapResult
    initial_service: Checkpoint
    initial_designer: Checkpoint
    plan_factory: EvolutionPlanFactory
    bank_updater: Callable
    rounds: int
    bank_manifest_path: str | Path | None = None
    independent_test_runner: object | None = None
    test_every: int = 0
    ability_registry_path: str | Path | None = None

    def __post_init__(self):
        if not isinstance(self.scheduler, EvolutionScheduler):
            raise TypeError("EvolutionScheduler runtime required")
        if not isinstance(self.bootstrap, BootstrapResult):
            raise TypeError("verified BootstrapResult required")
        if not isinstance(self.initial_service, Checkpoint) or not isinstance(self.initial_designer, Checkpoint):
            raise TypeError("initial Service and Designer checkpoints required")
        if not callable(getattr(self.plan_factory, "build", None)) or not callable(self.bank_updater):
            raise TypeError("EvolutionPlanFactory and bank updater required")
        if type(self.rounds) is not int or self.rounds < 1:
            raise ValueError("positive evolution round count required")
        if type(self.test_every) is not int or self.test_every < 0:
            raise ValueError("test interval must be a non-negative integer")
        if self.test_every and not callable(getattr(self.independent_test_runner, "run", None)):
            raise ValueError("scheduled independent testing requires an isolated test runner")


def import_runtime_factory(spec: str):
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError("runtime factory must use module:callable syntax")
    module_name, attribute = spec.split(":", 1)
    if not module_name or not attribute:
        raise ValueError("runtime factory must use module:callable syntax")
    factory = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(factory):
        raise ValueError("runtime factory target is not callable")
    return factory


def execute(runtime: EvolutionRunRuntime):
    if not isinstance(runtime, EvolutionRunRuntime):
        raise TypeError("runtime factory must return EvolutionRunRuntime")
    return runtime.scheduler.run(bootstrap=runtime.bootstrap,
        initial_service=runtime.initial_service,
        initial_designer=runtime.initial_designer,
        plan_factory=runtime.plan_factory,
        bank_updater=runtime.bank_updater,
        rounds=runtime.rounds,
        bank_manifest_path=runtime.bank_manifest_path,
        ability_registry_path=runtime.ability_registry_path,
        independent_test_runner=runtime.independent_test_runner,
        test_every=runtime.test_every)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run RIFT dual-bank evolution using a trusted runtime factory.")
    factory_group = parser.add_mutually_exclusive_group(required=True)
    factory_group.add_argument("--runtime-factory",
        help="legacy Python EvolutionRunRuntime factory in module:callable form")
    factory_group.add_argument("--deployment-factory",
        help="production binding factory in module:callable form; returns ProductionDeployment")
    parser.add_argument("--rift-config", default="configs/dual_bank.yaml")
    parser.add_argument("--verl-config", default="configs/verl/fsdp2_vllm.yaml")
    parser.add_argument("--ascend-config", default="configs/verl/ascend_910b3.yaml")
    parser.add_argument("--performance-config", default="configs/verl/performance_trial.yaml")
    parser.add_argument("--enable-performance", action="store_true",
        help="opt in to measured Dynamic Batch settings in the performance overlay")
    parser.add_argument("--run-root")
    args = parser.parse_args(argv)
    if args.runtime_factory:
        runtime = import_runtime_factory(args.runtime_factory)(args)
    else:
        if not args.run_root:
            parser.error("--run-root is required with --deployment-factory")
        from rift.runtime.config import load_production_config
        from rift.runtime.factory import ProductionDeployment, build_production_runtime
        config = load_production_config(
            rift_path=args.rift_config,
            verl_path=args.verl_config,
            ascend_path=args.ascend_config,
            performance_path=args.performance_config if args.enable_performance else None,
            enable_performance=args.enable_performance)
        deployment_factory = import_runtime_factory(args.deployment_factory)
        deployment = deployment_factory(args, config)
        if not isinstance(deployment, ProductionDeployment):
            raise TypeError("deployment factory must return ProductionDeployment")
        runtime = build_production_runtime(config, deployment, run_root=args.run_root)
    states = execute(runtime)
    summary = {"rounds_completed": len(states),
        "last_service_revision": states[-1].service_update.policy_after.revision,
        "last_designer_revision": states[-1].designer_update.policy_after.revision,
        "last_reward": states[-1].learning_reward.value}
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
