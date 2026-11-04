import sys
import types

import pytest

from rift import hardware
from rift.runtime import ascend_910b3 as runtime


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    for name in ('RIFT_PLATFORM', 'RIFT_WORLD_SIZE', 'RIFT_ROLLOUT_TP', 'RIFT_VLLM_MEMORY_UTILIZATION'):
        monkeypatch.delenv(name, raising=False)
    hardware.reset()
    yield
    hardware.reset()


def use(monkeypatch, name, world_size=None):
    monkeypatch.setenv('RIFT_PLATFORM', name)
    if world_size is not None:
        monkeypatch.setenv('RIFT_WORLD_SIZE', str(world_size))
    hardware.reset()


def test_rollout_tensor_parallelism_follows_the_platform(monkeypatch):
    use(monkeypatch, 'ascend')
    assert runtime._rollout_tensor_parallel_size(8) == 8              # unchanged on the NPU host
    use(monkeypatch, 'cuda', 3)
    assert runtime._rollout_tensor_parallel_size(3) == 1              # one replica per GPU, any GPU count
    monkeypatch.setenv('RIFT_ROLLOUT_TP', '2')
    with pytest.raises(ValueError, match='divide'):
        runtime._rollout_tensor_parallel_size(3)
    assert runtime._rollout_tensor_parallel_size(4) == 2
    monkeypatch.setenv('RIFT_ROLLOUT_TP', '3')
    with pytest.raises(ValueError, match='1, 2, 4 or 8'):
        runtime._rollout_tensor_parallel_size(6)


def test_worker_environment_forwarding_is_explicit():
    env = {'PATH': '/bin', 'PYTHONPATH': '/src', 'ASCEND_HOME_PATH': '/cann', 'ATB_HOME_PATH': '/atb',
           'SECRET_TOKEN': 'x', 'LD_LIBRARY_PATH': ''}
    cuda = runtime._forward_worker_environment({}, runtime.CUDA_WORKER_ENV_KEYS, env)['env_vars']
    assert cuda == {'PATH': '/bin', 'PYTHONPATH': '/src'}              # nothing else leaks, empty values dropped
    nccl = runtime._forward_cuda_worker_environment({}, {**env, 'NCCL_P2P_DISABLE': '1', 'NCCL_DEBUG': ''})
    assert nccl['env_vars'] == {'PATH': '/bin', 'PYTHONPATH': '/src', 'NCCL_P2P_DISABLE': '1'}
    ascend = runtime._forward_ascend_worker_environment({}, env)['env_vars']
    assert ascend == {'PATH': '/bin', 'PYTHONPATH': '/src', 'ASCEND_HOME_PATH': '/cann', 'ATB_HOME_PATH': '/atb'}


def _config(tmp_path):
    return runtime._verl_config('service', tmp_path, run_root=tmp_path / 'run')


def test_verl_config_for_ascend_is_unchanged(monkeypatch, tmp_path):
    pytest.importorskip('verl')
    pytest.importorskip('hydra')
    monkeypatch.setitem(sys.modules, 'torch_npu', types.ModuleType('torch_npu'))
    use(monkeypatch, 'ascend')
    config = _config(tmp_path)
    assert config.trainer.device == 'npu' and config.trainer.n_gpus_per_node == 8
    assert config.actor_rollout_ref.rollout.tensor_model_parallel_size == 8
    assert config.actor_rollout_ref.actor.ppo_mini_batch_size == 8
    assert config.actor_rollout_ref.rollout.agent.num_workers == 8
    assert config.actor_rollout_ref.rollout.gpu_memory_utilization == 0.45
    assert config.trainer.experiment_name == 'ascend910b3-service'
    assert 'attn_implementation' not in config.actor_rollout_ref.model.override_config   # NPU keeps VERL's default
    assert config.actor_rollout_ref.model.use_remove_padding is True


def test_verl_config_for_cuda_uses_the_gpu_count(monkeypatch, tmp_path):
    pytest.importorskip('verl')
    pytest.importorskip('hydra')
    use(monkeypatch, 'cuda', 4)
    monkeypatch.setenv('RIFT_VLLM_MEMORY_UTILIZATION', '0.3')
    config = _config(tmp_path)
    assert config.trainer.device == 'cuda' and config.trainer.n_gpus_per_node == 4
    assert config.actor_rollout_ref.rollout.tensor_model_parallel_size == 1
    assert config.actor_rollout_ref.actor.ppo_mini_batch_size == 4
    assert config.actor_rollout_ref.rollout.agent.num_workers == 4
    assert config.actor_rollout_ref.rollout.gpu_memory_utilization == 0.3
    assert config.trainer.experiment_name == 'cuda-service'
    assert config.actor_rollout_ref.actor.strategy == 'fsdp2' and config.actor_rollout_ref.rollout.name == 'vllm'
    # SDPA needs padded inputs: packed samples would attend to each other.
    assert config.actor_rollout_ref.model.override_config.attn_implementation == 'sdpa'
    assert config.actor_rollout_ref.model.use_remove_padding is False
    monkeypatch.setenv('RIFT_AGENT_WORKERS', '8')
    assert _config(tmp_path).actor_rollout_ref.rollout.agent.num_workers == 8     # independent of the 4 GPUs
    monkeypatch.setenv('RIFT_ATTN_IMPLEMENTATION', 'flash_attention_2')
    packed = _config(tmp_path)
    assert packed.actor_rollout_ref.model.use_remove_padding is True
