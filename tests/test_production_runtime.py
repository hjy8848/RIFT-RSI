import asyncio
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from rift.runtime.config import load_production_config
from rift.runtime.evaluator import ProductionProbeEvaluator, VerlAgentLoopBatch
from rift.runtime.resources import ExclusivePolicyResources
from rift.runtime.ascend_910b3 import (
    _check_ray_socket_path,
    _forward_ascend_worker_environment,
    initialize_checkpoint,
    _require_cann_atb_library,
)
from rift.schemas import Checkpoint, Outcome


def _config_kwargs():
    return {
        "rift_path": "configs/dual_bank.yaml",
        "verl_path": "configs/verl/fsdp2_vllm.yaml",
        "ascend_path": "configs/verl/ascend_910b3.yaml",
    }


def test_ray_socket_path_guard_accepts_short_dir_and_rejects_long_dir():
    assert _check_ray_socket_path("/tmp/r12345678") < 107
    with pytest.raises(RuntimeError, match="AF_UNIX limit is 107"):
        _check_ray_socket_path("/tmp/" + "x" * 100)


def test_ray_runtime_env_forwards_sourced_cann_paths_without_device_binding():
    runtime_env = {"env_vars": {"VERL_KEEP": "yes"}}
    source = {
        "PATH": "/cann/bin:/usr/bin",
        "LD_LIBRARY_PATH": "/cann/lib64:/cann/opp/vendors/customize/op_api/lib",
        "ASCEND_HOME_PATH": "/usr/local/Ascend/ascend-toolkit/latest",
        "ASCEND_OPP_PATH": "/usr/local/Ascend/ascend-toolkit/latest/opp",
        "ASCEND_AICPU_PATH": "/usr/local/Ascend/ascend-toolkit/latest",
        "ASCEND_CUSTOM_PATH": "/usr/local/Ascend/ascend-toolkit/latest",
        "ASCEND_TOOLKIT_HOME": "/usr/local/Ascend/ascend-toolkit/latest",
        "ATB_HOME_PATH": "/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1",
        "ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE": "0",
        "LCCL_DETERMINISTIC": "0",
        "ASCEND_DEVICE_ID": "7",
        "ASCEND_RT_VISIBLE_DEVICES": "7",
    }
    result = _forward_ascend_worker_environment(runtime_env, source)

    assert result["env_vars"]["VERL_KEEP"] == "yes"
    assert result["env_vars"]["LD_LIBRARY_PATH"] == source["LD_LIBRARY_PATH"]
    assert result["env_vars"]["ASCEND_HOME_PATH"] == source["ASCEND_HOME_PATH"]
    assert result["env_vars"]["ATB_HOME_PATH"] == source["ATB_HOME_PATH"]
    assert result["env_vars"]["ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE"] == "0"
    assert result["env_vars"]["LCCL_DETERMINISTIC"] == "0"
    assert "ASCEND_DEVICE_ID" not in result["env_vars"]
    assert "ASCEND_RT_VISIBLE_DEVICES" not in result["env_vars"]


def test_missing_cann_atb_library_fails_before_ray_with_setup_command(monkeypatch):
    import ctypes

    def missing_library(path):
        assert path == "libatb.so"
        raise OSError("not found")

    monkeypatch.setattr(ctypes, "CDLL", missing_library)
    with pytest.raises(RuntimeError, match="/nnal/atb/set_env\\.sh"):
        _require_cann_atb_library()


def test_initial_checkpoint_uses_resolved_model_identity(monkeypatch, tmp_path):
    from rift.runtime import ascend_910b3

    model_path = tmp_path / "model-link"
    seen = {}

    class Runtime:
        def __init__(self, *args, **kwargs):
            seen["runtime_options"] = kwargs

        def save_initial_checkpoint(self, output_dir, *, tokenizer_id, synchronize=True):
            seen["tokenizer_id"] = tokenizer_id
            seen["synchronize"] = synchronize
            return "checkpoint"

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(ascend_910b3, "Ascend910B3RoleRuntime", Runtime)
    result = initialize_checkpoint("service", model_path, tmp_path / "checkpoint",
        checkpoint_store=None, serving_attestor=None, run_root=tmp_path,
        initialize_vllm=False, synchronize=False)

    assert result == "checkpoint"
    assert seen["tokenizer_id"] == str(model_path.resolve())
    assert seen["runtime_options"]["initialize_vllm"] is False
    assert seen["synchronize"] is False
    assert seen["closed"] is True


def test_checkpoint_only_runtime_skips_vllm_and_tokenizer(monkeypatch, tmp_path):
    from rift.runtime import ascend_910b3

    monkeypatch.setattr(ascend_910b3, "_verl_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(ascend_910b3, "_start_ray", lambda *args, **kwargs: None)
    monkeypatch.setattr(ascend_910b3, "_stop_ray", lambda: None)
    monkeypatch.setattr(ascend_910b3.Ascend910B3RoleRuntime, "_construct_workers",
        lambda runtime: setattr(runtime, "worker_group", SimpleNamespace(world_size=8)))
    monkeypatch.setattr(ascend_910b3.Ascend910B3RoleRuntime, "_construct_vllm",
        lambda runtime: pytest.fail("checkpoint-only initialization must not start vLLM"))

    runtime = ascend_910b3.Ascend910B3RoleRuntime("service", None,
        model_path=tmp_path, run_root=tmp_path,
        checkpoint_store=object(), serving_attestor=object(), initialize_vllm=False)

    assert runtime.llm_server_manager is None
    assert runtime.checkpoint_manager is None
    assert runtime.tokenizer is None
    runtime.close()


def test_resident_training_state_requires_matching_runtime_checkpoint():
    from rift.runtime.ascend_910b3 import Ascend910B3RoleRuntime

    checkpoint = object()
    worker = SimpleNamespace(world_size=8)
    runtime = object.__new__(Ascend910B3RoleRuntime)
    runtime.role = "service"
    runtime.worker_group = worker
    runtime.closed = False
    runtime._loaded_checkpoint = checkpoint
    runtime.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(
        actor=SimpleNamespace(strategy="fsdp2"),
        rollout=SimpleNamespace(name="vllm")))
    runtime.serving_attestor = SimpleNamespace(
        verify_serving=lambda candidate: candidate is checkpoint)

    assert runtime.verify_resident_training_state("service", worker, checkpoint)
    assert not runtime.verify_resident_training_state("designer", worker, checkpoint)
    assert not runtime.verify_resident_training_state("service", worker, object())
    assert not runtime.verify_resident_training_state(
        "service", SimpleNamespace(world_size=8), checkpoint)


def test_production_config_pins_complete_grpo_and_measured_overlay(monkeypatch):
    config = load_production_config(**_config_kwargs())
    assert config.group_size == 4
    assert config.verl["actor_rollout_ref"]["rollout"]["n"] == config.group_size

    monkeypatch.setenv("RIFT_PPO_MAX_TOKEN_LEN_PER_GPU", "8192")
    measured = load_production_config(**_config_kwargs(),
        performance_path="configs/verl/performance_trial.yaml", enable_performance=True)
    actor = measured.verl["actor_rollout_ref"]["actor"]
    assert actor["use_dynamic_bsz"] is True
    assert actor["ppo_max_token_len_per_gpu"] == 8192
    assert measured.verl["actor_rollout_ref"]["rollout"]["n"] == 4


def test_production_config_rejects_unmeasured_overlay():
    with pytest.raises(ValueError, match="measured token-budget environment variable"):
        load_production_config(**_config_kwargs(),
            performance_path="configs/verl/performance_trial.yaml", enable_performance=True)


def test_agent_loop_manager_adapter_preserves_and_reorders_opaque_rows(monkeypatch):
    class Tensor:
        def __init__(self, shape):
            self.shape = shape

    fake_torch = SimpleNamespace(
        long=object(),
        full=lambda shape, value, dtype: Tensor(shape),
        ones_like=lambda value: Tensor(value.shape),
        zeros_like=lambda value: Tensor(value.shape),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    # Python special methods are looked up on the type, so use a concrete batch.
    class InputBatch:
        @classmethod
        def from_dict(cls, *, tensors, non_tensors, meta_info):
            row = cls()
            row.tensors, row.non_tensor_batch, row.meta_info = tensors, non_tensors, meta_info
            return row

        def __len__(self):
            return len(self.non_tensor_batch["uid"])

    fake_verl = types.ModuleType("verl")
    fake_verl.DataProto = InputBatch
    monkeypatch.setitem(sys.modules, "verl", fake_verl)

    class Manager:
        async def generate_sequences(self, batch):
            assert batch.tensors["prompts"].shape == (2, 1)
            assert list(batch.non_tensor_batch["agent_name"]) == ["rift_usami_service"] * 2
            assert batch.meta_info == {"validate": False}
            indices = [1, 0]
            output = {key: [values[i] for i in indices]
                for key, values in batch.non_tensor_batch.items()}
            output["rift_episode_id"] = ["episode-b", "episode-a"]
            return SimpleNamespace(non_tensor_batch=output, __len__=lambda: 2)

    class Result:
        def __init__(self, values):
            self.non_tensor_batch = values

        def __len__(self):
            return len(self.non_tensor_batch["uid"])

    manager = Manager()
    original = manager.generate_sequences

    async def generate(batch):
        response = await original(batch)
        return Result(response.non_tensor_batch)

    manager.generate_sequences = generate
    adapter = VerlAgentLoopBatch(manager, prompt_token_id=0)
    common = {
        "rift_mode": "train", "rift_task_ref": "task-F1", "rift_group_size": 4,
        "rift_rollout_id": "rollout", "rift_sampling_seed": 7,
        "rift_customer_seed": 19,
        "rift_expected_checkpoint": '{"policy":"service","revision":"r1",'
            '"tokenizer":"tok","weights_digest":"digest"}',
    }
    rows = tuple({**common, "rift_request_id": f"request-{letter}", "uid": "group-1",
                  "session_id": i, "raw_prompt": []}
        for i, letter in enumerate(("a", "b")))
    result = adapter.generate(rows)
    assert [row["rift_request_id"] for row in result] == ["request-a", "request-b"]
    assert [row["rift_episode_id"] for row in result] == ["episode-a", "episode-b"]


def test_agent_loop_manager_adapter_rejects_missing_or_forged_request_id(monkeypatch):
    class Tensor:
        def __init__(self, shape): self.shape = shape

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(long=object(),
        full=lambda shape, value, dtype: Tensor(shape), ones_like=lambda x: x,
        zeros_like=lambda x: x))

    class DataProto:
        @classmethod
        def from_dict(cls, *, tensors, non_tensors, meta_info):
            row = cls(); row.non_tensor_batch = non_tensors; row.tensors = tensors
            return row
        def __len__(self): return len(self.non_tensor_batch["uid"])

    fake_verl = types.ModuleType("verl"); fake_verl.DataProto = DataProto
    monkeypatch.setitem(sys.modules, "verl", fake_verl)

    class Manager:
        async def generate_sequences(self, batch):
            return DataProto.from_dict(tensors={}, non_tensors={
                "rift_request_id": ["forged"], "uid": ["group"]}, meta_info={})

    adapter = VerlAgentLoopBatch(Manager(), prompt_token_id=0)
    row = {"rift_request_id": "original", "rift_task_ref": "task", "rift_mode": "train",
        "rift_group_size": 4, "rift_rollout_id": "rollout", "rift_sampling_seed": 1,
        "rift_customer_seed": 2, "rift_expected_checkpoint": "checkpoint", "uid": "group"}
    with pytest.raises(ValueError, match="changed or omitted request identities"):
        adapter.generate((row,))


def test_probe_adapter_downgrades_unverified_business_outcome_to_unknown():
    checkpoint = Checkpoint("service", "1", "tok", "digest")

    class Batch:
        def generate(self, requests):
            return tuple({
                "rift_request_id": row["rift_request_id"],
                "rift_task_ref": row["rift_task_ref"],
                "rift_mode": "probe",
                "rift_checkpoint": row["rift_checkpoint"],
                "rift_seed": row["rift_sampling_seed"],
                "rift_protocol_ref": row["rift_protocol_ref"],
                "rift_protocol_digest": row["rift_protocol_digest"],
                "rift_episode_id": f"episode-{i}",
                "rift_outcome": Outcome.SUCCESS.value,
                "rift_replay_verified": False,
                "rift_evidence_ref": None,
            } for i, row in enumerate(requests))

    trial, = ProductionProbeEvaluator(Batch()).evaluate_many((
        ("bank-task", checkpoint, 13, "protocol", "protocol-digest"),))
    assert trial.outcome is Outcome.UNKNOWN
    assert trial.replay_verified is False
    assert trial.evidence_ref is None


def test_role_runtime_is_exclusive_and_always_closed():
    checkpoint = Checkpoint("service", "r1", "tok", "digest")
    events = []

    class Runtime:
        role = "service"
        def __init__(self): self.checkpoint = checkpoint
        def verify_loaded_state(self, expected):
            events.append(("verified", expected))
            return True
        def close(self): events.append(("closed",))

    resources = ExclusivePolicyResources(lambda _: Runtime(), lambda _: Runtime())
    with resources.acquire("service", checkpoint):
        with pytest.raises(RuntimeError, match="already owned"):
            with resources.acquire("designer", checkpoint):
                pass
    assert events == [("verified", checkpoint), ("closed",)]


def test_role_bound_backend_builds_per_update_and_closes_after_failure(tmp_path):
    from rift.runtime.resources import RoleBoundBackend
    checkpoint = Checkpoint("service", "1", "tok", "digest")
    events = []

    class Backend:
        def train(self, request):
            events.append(("train", request))
            raise RuntimeError("explicit simulated worker failure")

    class Runtime:
        role = "service"
        def __init__(self): self.checkpoint = checkpoint
        def verify_loaded_state(self, expected): return expected == checkpoint
        def create_update_backend(self, request, **kwargs):
            events.append(("backend", request, kwargs["round_index"], kwargs["metric_sink"]))
            return Backend()
        def close(self): events.append(("closed",))

    resources = ExclusivePolicyResources(lambda _: Runtime(), lambda _: Runtime())
    request = SimpleNamespace(policy_before=checkpoint, package_id="package")
    backend = RoleBoundBackend(resources, "service", checkpoint_root=tmp_path / 'checkpoints',
                               round_index=2)
    with pytest.raises(RuntimeError, match="simulated worker failure"):
        backend.train(request)
    assert events[0] == ("backend", request, 2, None)
    assert events[1] == ("train", request)
    assert events[2] == ("closed",)


def test_agent_loop_batch_rejects_incomplete_probe_result():
    class Batch:
        def generate(self, requests): return ()

    with pytest.raises(ValueError, match="omitted one or more frozen requests"):
        ProductionProbeEvaluator(Batch()).evaluate_many((
            ("task", Checkpoint("service", "r1", "tok", "digest"), 2,
             "protocol", "protocol-digest"),))


def test_runtime_deployment_paths_require_real_models_and_manifests(tmp_path):
    config = load_production_config(**_config_kwargs())
    model = tmp_path / "missing-model"
    tokenizer = tmp_path / "missing-tokenizer"
    manifest_a = tmp_path / "bootstrap.json"
    manifest_b = tmp_path / "benchmark.json"
    manifest_a.write_text("{}", encoding="utf-8")
    manifest_b.write_text("{}", encoding="utf-8")
    paths = {"service_model": str(model), "designer_model": str(model),
        "service_tokenizer": str(tokenizer), "designer_tokenizer": str(tokenizer),
        "bootstrap_manifest": str(manifest_a), "benchmark_manifest": str(manifest_b),
        "run_root": str(tmp_path / "run")}
    with pytest.raises(FileNotFoundError, match="service_model"):
        config.require_deployment_paths(paths)
