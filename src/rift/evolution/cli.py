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
    parser.add_argument("--runtime-factory", required=True,
        help="Python runtime factory in module:callable form")
    args = parser.parse_args(argv)
    runtime = import_runtime_factory(args.runtime_factory)(args)
    states = execute(runtime)
    summary = {"rounds_completed": len(states),
        "last_service_revision": states[-1].service_update.policy_after.revision,
        "last_designer_revision": states[-1].designer_update.policy_after.revision,
        "last_reward": states[-1].learning_reward.value}
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
