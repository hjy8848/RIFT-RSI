"""Exclusive, failure-safe lifecycle for full-parameter policy runtimes."""
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
import time

from rift.schemas import Checkpoint


class ExclusivePolicyResources:
    """Own one role runtime at a time; runtime factories must close real VERL workers.

    A factory returns a handle with ``role``, ``checkpoint`` and ``close()``. The
    factory is deployment-owned because Ray/FSDP worker construction depends on
    the fully resolved VERL model config. RIFT enforces the exclusive boundary
    around that official runtime and never starts a fallback worker.
    """
    def __init__(self, service_factory, designer_factory):
        if not callable(service_factory) or not callable(designer_factory):
            raise TypeError("explicit Service and Designer VERL runtime factories required")
        self._factories = {"service": service_factory, "designer": designer_factory}
        self._lock = Lock()
        self._active = None

    @contextmanager
    def acquire(self, role, checkpoint):
        if role not in self._factories or not isinstance(checkpoint, Checkpoint):
            raise ValueError("known policy role and exact checkpoint required")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError(f"NPU runtime is already owned by {self._active or 'another thread'}")
        handle = None
        try:
            self._active = role
            handle = self._factories[role](checkpoint)
            if (getattr(handle, "role", None) != role
                    or getattr(handle, "checkpoint", None) != checkpoint
                    or not callable(getattr(handle, "close", None))):
                raise ValueError("policy runtime handle must attest role/checkpoint and provide close()")
            verify = getattr(handle, "verify_loaded_state", None)
            if not callable(verify) or verify(checkpoint) is not True:
                raise ValueError("full model/optimizer/scheduler/RNG state was not verified after restore")
            yield handle
        finally:
            try:
                if handle is not None:
                    handle.close()
            finally:
                self._active = None
                self._lock.release()


class RoleBoundBackend:
    """Build a request-bound FSDP2 backend while its role owns the devices.

    Each update needs a unique output checkpoint and run identity. The runtime
    handle therefore constructs a fresh backend for every UpdateRequest.
    """
    def __init__(self, resources, role, *, checkpoint_root, metric_sink=None,
                 round_index=None):
        if role not in ("service", "designer"):
            raise ValueError("unknown VERL policy role")
        self.resources, self.role = resources, role
        self.checkpoint_root = Path(checkpoint_root)
        self.metric_sink, self.round_index = metric_sink, round_index

    def train(self, request):
        started = time.perf_counter()
        from rift.verl.serving import checkpoint_step
        after_step = checkpoint_step(request.policy_before) + 1
        round_name = 'unknown' if self.round_index is None else f'{self.round_index:06d}'
        run_id = f'rift-{self.role}-round-{round_name}-{request.package_id}'
        output_dir = self.checkpoint_root / self.role / f'global-step-{after_step:08d}'
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_manifest_path = self.checkpoint_root / f'{self.role}-global-step-{after_step:08d}.json'
        if output_dir.exists() or checkpoint_manifest_path.exists():
            raise RuntimeError('update checkpoint output already exists; reconcile persisted update evidence before retry')
        with self.resources.acquire(self.role, request.policy_before) as runtime:
            create_backend = getattr(runtime, "create_update_backend", None)
            if not callable(create_backend):
                raise ValueError(f"{self.role} runtime must create a checkpoint-bound VERL backend per update")
            backend = create_backend(request, metric_sink=self.metric_sink,
                round_index=self.round_index, output_dir=output_dir,
                after_revision=str(after_step), run_id=run_id,
                checkpoint_manifest_path=checkpoint_manifest_path)
            if not callable(getattr(backend, "train", None)):
                raise ValueError(f"{self.role} runtime did not construct its pinned VERL backend")
            result = backend.train(request)
        if self.metric_sink is not None:
            self.metric_sink(f"{self.role}_update_seconds", time.perf_counter() - started,
                round_index=self.round_index, package_id=request.package_id)
        return result
