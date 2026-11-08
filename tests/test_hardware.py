import pytest

from rift import hardware
from rift.hardware import ASCEND_SERIALIZATION, CUDA_SERIALIZATION, PlatformNotReady


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    for name in ('RIFT_PLATFORM', 'RIFT_WORLD_SIZE'):
        monkeypatch.delenv(name, raising=False)
    hardware.reset()
    yield
    hardware.reset()


def test_ascend_keeps_the_npu_contract(monkeypatch):
    monkeypatch.setenv('RIFT_PLATFORM', 'ascend')
    platform = hardware.detect()
    assert (platform.device, platform.ray_resource, platform.world_size) == ('npu', 'NPU', 8)
    assert platform.serialization == ASCEND_SERIALIZATION and platform.runtime_ready


def test_cuda_world_size_follows_the_environment_and_has_its_own_identity(monkeypatch):
    monkeypatch.setenv('RIFT_PLATFORM', 'cuda')
    monkeypatch.setenv('RIFT_WORLD_SIZE', '3')
    platform = hardware.detect()
    assert (platform.device, platform.ray_resource, platform.world_size) == ('cuda', 'GPU', 3)
    assert platform.serialization == CUDA_SERIALIZATION != ASCEND_SERIALIZATION
    assert platform.runtime_ready                       # one GRPO update was run on 2 GPUs
    monkeypatch.setenv('RIFT_WORLD_SIZE', 'many')
    with pytest.raises(ValueError, match='positive integer'):
        hardware.detect()


def test_rehearsal_has_eight_ranks_and_no_runtime(monkeypatch):
    monkeypatch.setenv('RIFT_PLATFORM', 'rehearsal')
    platform = hardware.detect()
    assert platform.world_size == 8 and platform.serialization == ASCEND_SERIALIZATION
    with pytest.raises(PlatformNotReady):
        platform.require_runtime('writing the base checkpoint')


def test_unknown_platform_is_an_error_not_a_guess(monkeypatch):
    monkeypatch.setenv('RIFT_PLATFORM', 'tpu')
    with pytest.raises(ValueError, match='RIFT_PLATFORM'):
        hardware.detect()


def test_a_cpu_only_machine_defaults_to_rehearsal():
    assert hardware.detect().name in ('rehearsal', 'cuda', 'ascend')    # follows what is installed
    assert hardware.current() is hardware.current()                     # decided once


def test_the_run_follows_the_platform_for_rank_count_and_identity(monkeypatch, tmp_path):
    from rift.runtime import curriculum_run as run
    from usami_v2.runtime.runner import EpisodeBudget
    from rift.evolution.matched_control import UpdateBudget
    monkeypatch.setenv('RIFT_PLATFORM', 'cuda')
    monkeypatch.setenv('RIFT_WORLD_SIZE', '4')
    hardware.reset()
    context = run.ExperimentContext(root=tmp_path, service_model='m', verl_config_dir='.',
        episode_budget=EpisodeBudget(), update_budget=UpdateBudget(4, 1, 1e-6))
    assert context.checkpoint_store.world_size == 4
    assert context.serialization == CUDA_SERIALIZATION
    # A CPU rehearsal has the NPU contract's ranks but can start no runtime.
    monkeypatch.setenv('RIFT_PLATFORM', 'rehearsal')
    hardware.reset()
    rehearsal = run.ExperimentContext(root=tmp_path / 'cpu', service_model='m', verl_config_dir='.',
        episode_budget=EpisodeBudget(), update_budget=UpdateBudget(8, 1, 1e-6))
    with pytest.raises(PlatformNotReady, match='base checkpoint'):
        rehearsal.initial_checkpoint()
    with pytest.raises(PlatformNotReady, match='Service runtime'):
        rehearsal.runtime_factory(None)


def test_ascend_world_size_can_be_reduced_for_shared_hosts(monkeypatch):
    import pytest
    from rift import hardware
    monkeypatch.setenv('RIFT_PLATFORM', 'ascend')
    monkeypatch.delenv('RIFT_WORLD_SIZE', raising=False)
    assert hardware.detect().world_size == 8
    monkeypatch.setenv('RIFT_WORLD_SIZE', '4')
    assert hardware.detect().world_size == 4
    for bad in ('0', '9', 'x'):
        monkeypatch.setenv('RIFT_WORLD_SIZE', bad)
        with pytest.raises(ValueError):
            hardware.detect()
