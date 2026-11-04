"""The accelerator platform of this process.

Usami, the generators, the Designer, certification, the registries and the curriculum
policies do not depend on the accelerator. Only starting VERL and vLLM workers, writing
the FSDP base checkpoint and the Ray resource map do, and they go through this module.

``RIFT_PLATFORM`` (``ascend``, ``cuda`` or ``rehearsal``) forces the choice, which is
recorded with every run; without it the platform follows what is installed. ``rehearsal``
is the CPU stand-in used by the tests: it has the NPU contract's eight ranks so that a
rehearsal exercises the sizes of the real run, but it cannot start a runtime.
"""
from dataclasses import dataclass
import importlib.util
import os

# Qwen3 thinks by default; every Service and Designer prompt switches that off.
CHAT_TEMPLATE_KWARGS = {'enable_thinking': False}

ASCEND_WORLD_SIZE = 8
# Part of every stored trace and checkpoint identity. Rehearsal keeps the Ascend value so
# that rehearsal and NPU artifacts of the same run layout have the same identity.
ASCEND_SERIALIZATION = 'verl-v0.9.1-vllm-ascend-0.23.0-native-token-ids-v1'
CUDA_SERIALIZATION = 'verl-v0.9.1-vllm-cuda-native-token-ids-v1'
NAMES = ('ascend', 'cuda', 'rehearsal')


class PlatformNotReady(RuntimeError):
    """The requested operation needs a runtime this platform does not have yet."""


@dataclass(frozen=True)
class Platform:
    name: str                 # ascend | cuda | rehearsal
    device: str               # VERL trainer.device
    ray_resource: str | None  # Ray resource name of one accelerator
    world_size: int           # FSDP ranks = accelerators used
    serialization: str
    runtime_ready: bool       # a VERL/vLLM runtime for this platform exists and was run

    def require_runtime(self, what):
        if not self.runtime_ready:
            raise PlatformNotReady(
                f'{what} needs a VERL/vLLM runtime for the {self.name!r} platform, which is not '
                'available. Only the Ascend runtime has been implemented and run; Designer stage 0 '
                '(python -m rift.designer.stage0) needs no runtime and works on any platform.')


def _installed(module):
    return importlib.util.find_spec(module) is not None


def _cuda_world_size():
    forced = os.environ.get('RIFT_WORLD_SIZE')
    if forced:
        if not forced.isdigit() or int(forced) < 1:
            raise ValueError('RIFT_WORLD_SIZE must be a positive integer')
        return int(forced)
    import torch
    return max(1, torch.cuda.device_count())


def detect():
    name = os.environ.get('RIFT_PLATFORM')
    if name is not None and name not in NAMES:
        raise ValueError(f'RIFT_PLATFORM must be one of {NAMES}, got {name!r}')
    if name is None:
        if _installed('torch_npu'):
            name = 'ascend'
        elif _installed('torch'):
            import torch
            name = 'cuda' if torch.cuda.is_available() else 'rehearsal'
        else:
            name = 'rehearsal'
    if name == 'ascend':
        return Platform('ascend', 'npu', 'NPU', ASCEND_WORLD_SIZE, ASCEND_SERIALIZATION, True)
    if name == 'cuda':
        return Platform('cuda', 'cuda', 'GPU', _cuda_world_size(), CUDA_SERIALIZATION, False)
    return Platform('rehearsal', 'cpu', None, ASCEND_WORLD_SIZE, ASCEND_SERIALIZATION, False)


_CURRENT = None


def current():
    """The platform of this process, decided once."""
    global _CURRENT
    if _CURRENT is None:
        _CURRENT = detect()
    return _CURRENT


def reset():
    """Forget the decision (tests change RIFT_PLATFORM)."""
    global _CURRENT
    _CURRENT = None
