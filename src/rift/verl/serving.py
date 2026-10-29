"""Bind VERL rollout generations to completed weight-sync revisions.

VERL v0.9.1's vLLM server reports ``global_steps`` on each TokenOutput. The
weight-sync manager itself does not return a tensor digest, so this records a
host-side sync receipt and requires each generation to acknowledge the same
step. It is version evidence, not a cryptographic attestation of server RAM.
"""
import asyncio
import inspect
import json
from pathlib import Path

from rift.schemas import Checkpoint
from rift.training.trace import digest
from dataclasses import asdict


def checkpoint_step(checkpoint: Checkpoint) -> int:
    """Map the numeric checkpoint revision to VERL's global step."""
    try:
        step = int(checkpoint.revision)
    except (TypeError, ValueError) as exc:
        raise ValueError('VERL checkpoint revision must be a canonical nonnegative global step') from exc
    if step < 0 or str(step) != checkpoint.revision:
        raise ValueError('VERL checkpoint revision must be a canonical nonnegative global step')
    return step


def serving_fields(extra_fields):
    """Extract the mandatory version fields from an official TokenOutput."""
    if not isinstance(extra_fields, dict):
        raise ValueError('VERL TokenOutput.extra_fields is required')
    fields = {key: extra_fields.get(key) for key in ('global_steps', 'min_global_steps', 'max_global_steps')}
    if any(type(value) is not int or value < 0 for value in fields.values()):
        raise ValueError('VERL generation did not report complete global-step bounds')
    if len(set(fields.values())) != 1:
        raise ValueError('generation crossed Service checkpoint revisions')
    return fields


class VerlServingAttestor:
    """Persist completed sync receipts and verify per-generation step bounds."""
    def __init__(self, root, checkpoint_store):
        self.root = Path(root)
        self.checkpoint_store = checkpoint_store
        self._active = None

    def _path(self, checkpoint):
        return self.root / (digest(asdict(checkpoint)) + '.json')

    def _read(self, checkpoint):
        try:
            row = json.loads(self._path(checkpoint).read_text(encoding='utf-8'))
            expected_artifact = self.checkpoint_store.artifact_digests.get(checkpoint)
            if (row.get('checkpoint') != asdict(checkpoint) or not expected_artifact
                    or row.get('artifact_digest') != expected_artifact):
                return None
            return row
        except (OSError, ValueError, TypeError):
            return None

    def activate_after_sync(self, checkpoint, global_step, artifact_digest):
        if not isinstance(checkpoint, Checkpoint) or type(global_step) is not int or global_step < 0:
            raise ValueError('checkpoint and nonnegative synchronized step are required')
        if checkpoint_step(checkpoint) != global_step:
            raise ValueError('checkpoint revision and VERL global step differ')
        if (self.checkpoint_store.verify(checkpoint) is not True
                or not artifact_digest
                or self.checkpoint_store.artifact_digests.get(checkpoint) != artifact_digest):
            raise ValueError('synchronized checkpoint artifact is not registered and verified')
        row = {'checkpoint': asdict(checkpoint), 'global_steps': global_step,
               'artifact_digest': artifact_digest,
               'sync_status': 'verl_checkpoint_engine_update_weights_returned'}
        payload = json.dumps(row, sort_keys=True, allow_nan=False)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self._path(checkpoint)
        try:
            with path.open('x', encoding='utf-8') as stream:
                stream.write(payload)
        except FileExistsError:
            if path.read_text(encoding='utf-8') != payload:
                raise ValueError('conflicting checkpoint synchronization receipt')
        self._active = checkpoint

    def verify_serving(self, checkpoint):
        row = self._read(checkpoint)
        return self._active == checkpoint and row is not None and row.get('global_steps') == checkpoint_step(checkpoint)

    def verify_generation(self, checkpoint, fields):
        row = self._read(checkpoint)
        if row is None:
            return False
        try:
            parsed = serving_fields(fields)
        except ValueError:
            return False
        expected = row.get('global_steps')
        return type(expected) is int and all(value == expected for value in parsed.values())


class VerlCheckpointWeightSync:
    """Call the pinned VERL manager and activate its checkpoint revision.

    ``checkpoint_manager`` must be the official v0.9.1 CheckpointEngineManager
    wired to the same actor worker group passed to ``synchronize``.
    """
    def __init__(self, checkpoint_manager, serving_attestor):
        self.checkpoint_manager = checkpoint_manager
        self.serving_attestor = serving_attestor

    @staticmethod
    def _resolve(value):
        if not inspect.isawaitable(value):
            return value
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(value)
        raise RuntimeError('synchronous RIFT update cannot await VERL weight sync inside a running event loop')

    def synchronize(self, worker_group, checkpoint):
        manager = self.checkpoint_manager
        if getattr(manager, 'actor_wg', None) is not worker_group:
            raise ValueError('VERL weight-sync manager is not bound to this actor worker group')
        step = checkpoint_step(checkpoint)
        if self.checkpoint_manager.actor_wg is not worker_group:
            raise ValueError('VERL weight-sync manager changed actor worker group during synchronization')
        if self.serving_attestor.checkpoint_store.verify(checkpoint) is not True:
            raise ValueError('checkpoint artifact failed verification before weight synchronization')
        artifact_digest = self.serving_attestor.checkpoint_store.artifact_digests.get(checkpoint)
        if not artifact_digest:
            raise ValueError('verified checkpoint artifact digest is unavailable')
        # The pinned manager waits for all replica updates before returning. Its
        # vLLM TokenOutput global_steps is checked independently per request.
        self._resolve(manager.update_weights(global_steps=step))
        self.serving_attestor.activate_after_sync(checkpoint, step, artifact_digest)
        return True
