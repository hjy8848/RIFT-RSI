"""Load the existing RIFT and VERL YAML files as one validated runtime config."""
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any


def _yaml(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("install rift-rsi[runtime] to load YAML runtime settings") from exc
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return value


def _merge(base, overlay):
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass(frozen=True)
class ProductionConfig:
    rift: dict[str, Any]
    verl: dict[str, Any]
    ascend: dict[str, Any]
    performance_enabled: bool = False

    @property
    def package(self):
        return self.rift["training_package"]

    @property
    def group_size(self):
        return self.package["grpo_group_size"]

    def validate(self):
        from rift.training.package import TrainingPackageConfig
        package = self.rift.get("training_package", {})
        service = self.rift.get("service", {})
        designer = self.rift.get("designer", {})
        actor = self.verl.get("actor_rollout_ref", {})
        model = actor.get("model", {})
        actor_config = actor.get("actor", {})
        rollout = actor.get("rollout", {})
        algorithm = self.verl.get("algorithm", {})
        trainer = self.verl.get("trainer", {})
        checks = (
            (package.get("grpo_group_size") == rollout.get("n"),
             "RIFT grpo_group_size must equal VERL rollout.n"),
            (rollout.get("name") == "vllm", "production rollout backend must be vllm"),
            (actor_config.get("strategy") == "fsdp2", "actor strategy must be fsdp2"),
            (model.get("lora_rank") == 0, "LoRA must be disabled for full-parameter RL"),
            (algorithm.get("adv_estimator") == "grpo", "Service VERL estimator must be GRPO"),
            (service.get("algorithm") == "grpo" and service.get("full_parameter") is True
             and service.get("lora_rank") == 0, "RIFT Service must use full-parameter GRPO"),
            (designer.get("algorithm") == "reinforce_baseline"
             and designer.get("full_parameter") is True and designer.get("lora_rank") == 0,
             "RIFT Designer must use full-parameter reinforce_baseline"),
            (designer.get("credit_assignment") == "sequential_curriculum"
             and designer.get("batch_whitening") is False,
             "sequential Designer credit requires scalar-baseline REINFORCE without whitening"),
            (trainer.get("use_v1") is True and trainer.get("device") == "npu"
             and trainer.get("n_gpus_per_node") == 8,
             "target trainer must use VERL v1 on eight NPU devices"),
            (self.ascend.get("verl_commit") ==
             "1876b06d0a3e4e71e06230be10af14492ca8a75b",
             "Ascend manifest must pin VERL v0.9.1 commit"),
        )
        failures = [message for valid, message in checks if not valid]
        if failures:
            raise ValueError("invalid production config: " + "; ".join(failures))
        try:
            TrainingPackageConfig(
                designer_worlds_per_round=package["designer_worlds_per_round"],
                tasks_per_round=package["tasks_per_round"],
                designer_fraction=package["designer_fraction"],
                benchmark_fraction=package["benchmark_fraction"],
                sampling_seed=package["sampling_seed"],
                group_size=package["grpo_group_size"],
                max_candidate_challenges=package["max_candidate_challenges"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("RIFT training package configuration is invalid") from exc
        if self.performance_enabled:
            token_budget = actor_config.get("ppo_max_token_len_per_gpu")
            if type(token_budget) is not int or token_budget < 1:
                raise ValueError("performance overlay requires a measured integer token budget")
            if actor_config.get("use_dynamic_bsz") is not True:
                raise ValueError("performance overlay token budget requires Dynamic Batch")
        elif actor_config.get("use_dynamic_bsz") or actor_config.get("ppo_max_token_len_per_gpu"):
            raise ValueError("Dynamic Batch is only allowed through the explicit measured overlay")
        return self

    def require_deployment_paths(self, deployment):
        """Fail before runtime construction if actual model/data roots are absent."""
        required = ("service_model", "designer_model", "service_tokenizer", "designer_tokenizer",
                    "bootstrap_manifest", "benchmark_manifest", "run_root")
        missing = [key for key in required if not deployment.get(key)]
        if missing:
            raise ValueError("deployment paths are required: " + ", ".join(missing))
        for key in required:
            path = Path(deployment[key])
            if key.endswith("manifest") and not path.is_file():
                raise FileNotFoundError(f"deployment {key} does not exist: {path}")
            if key.endswith(("_model", "_tokenizer")) and not path.exists():
                raise FileNotFoundError(f"deployment {key} does not exist: {path}")
        return deployment


def load_production_config(*, rift_path, verl_path, ascend_path,
                           performance_path=None, enable_performance=False):
    """Merge existing config files and validate the algorithm/runtime contract."""
    if enable_performance and performance_path is None:
        raise ValueError("performance overlay path is required when explicitly enabled")
    if not enable_performance and performance_path is not None:
        raise ValueError("performance config must be explicitly opted into")
    verl_config = _yaml(verl_path)
    if enable_performance:
        overlay = _yaml(performance_path)
        token_budget = overlay.get("actor_rollout_ref", {}).get("actor", {}).get(
            "ppo_max_token_len_per_gpu")
        if isinstance(token_budget, str) and token_budget.startswith("${oc.env:"):
            variable = token_budget[len("${oc.env:"):-1].split(",", 1)[0].strip()
            if not variable or variable not in os.environ:
                raise ValueError(f"measured token-budget environment variable is missing: {variable}")
            try:
                overlay["actor_rollout_ref"]["actor"]["ppo_max_token_len_per_gpu"] = int(os.environ[variable])
            except ValueError as exc:
                raise ValueError(f"measured token budget must be an integer: {variable}") from exc
        verl_config = _merge(verl_config, overlay)
    return ProductionConfig(_yaml(rift_path), verl_config, _yaml(ascend_path),
                            performance_enabled=enable_performance).validate()
