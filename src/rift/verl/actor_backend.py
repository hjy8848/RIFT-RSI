"""Explicit GRPO update against an initialized VERL 0.9.1 FSDP2 actor.

No worker or model is created by importing this module. The host must supply
real rollout and update attestors; fake evidence is never accepted implicitly.
"""
from pathlib import Path
import time
from rift.training.contracts import CompletedUpdate, UpdateRequest
from rift.training.records import comparable_designer_rollouts
from rift.verl.scorer import padded_batch
from rift.verl.serving import checkpoint_step
from rift.verl.version import require_verl_version


def validate_request(request, role, rollout_verifier):
    if not isinstance(request, UpdateRequest) or request.algorithm.lower() not in ('grpo', 'reinforce_plus_plus', 'reinforce_baseline'):
        raise ValueError('unsupported on-policy advantage estimator')
    if role not in ('service', 'designer'): raise ValueError('unknown policy role')
    if role == 'service' and request.algorithm.lower() != 'grpo':
        raise ValueError('Service updates require verified same-task GRPO groups')
    if role == 'designer' and request.reward_protocol == 'dual_bank_outcome_v1':
        comparable_designer_rollouts(request.rollouts, algorithm=request.algorithm,
            credit_assignment_mode=request.credit_assignment_mode)
        if request.algorithm.lower() == 'reinforce_baseline' and request.advantage_baseline is None:
            raise ValueError('sequential Designer policy gradient requires an explicit scalar baseline')
    for row in request.rollouts:
        if (request.algorithm.lower() == 'grpo' and not row.group_id) or not row.response_id or not row.reward_ref:
            raise ValueError('rollout, reward and group provenance are required')
        if role == 'service' and (row.reward_source != 'business_outcome_v2' or row.reward_ref != row.episode_id):
            raise ValueError('Service requires business-only outcomes')
        if rollout_verifier.verify_rollout(row, request.policy_before) is not True:
            raise ValueError('on-policy rollout evidence was not verified')
    if request.algorithm.lower() == 'grpo':
        groups = {}
        for row in request.rollouts: groups.setdefault(row.group_id, []).append(row)
        for group in groups.values():
            if role == 'service' and len({r.task_id for r in group}) != 1:
                raise ValueError('Service GRPO groups must use the same task')
            if rollout_verifier.verify_group(group, request.policy_before) is not True:
                raise ValueError('GRPO group provenance was not verified')
    if role == 'designer' and request.reward_protocol != 'dual_bank_outcome_v1':
        comparable_designer_rollouts(request.rollouts)
    return request


class VerlGRPOBackend:
    """One explicit full-parameter update. The caller owns the worker lifecycle.

    A runtime may expose verify_resident_training_state() to prove that model,
    optimizer, scheduler and RNG state are already at policy_before. Only then
    can a resident FSDP2 worker skip restoring that checkpoint from disk.
    """
    def __init__(self, role, worker_group, checkpoint_store, rollout_verifier,
                 run_recorder, update_verifier, runtime_verifier, weight_sync, *, output_dir, after_revision,
                 run_id, pad_token_id, mini_batch_size, epochs=1,
                 metric_sink=None, round_index=None, checkpoint_manifest_path=None):
        self.role, self.worker_group = role, worker_group
        self.checkpoint_store, self.rollout_verifier = checkpoint_store, rollout_verifier
        self.run_recorder, self.update_verifier = run_recorder, update_verifier
        self.runtime_verifier = runtime_verifier
        self.weight_sync = weight_sync
        self.output_dir, self.after_revision, self.run_id = Path(output_dir), after_revision, run_id
        self.pad_token_id, self.mini_batch_size, self.epochs = pad_token_id, mini_batch_size, epochs
        self.metric_sink, self.round_index = metric_sink, round_index
        self.checkpoint_manifest_path = (None if checkpoint_manifest_path is None
                                         else Path(checkpoint_manifest_path))

    def _metric(self, name, value, **fields):
        if self.metric_sink is not None:
            self.metric_sink(name, value, round_index=self.round_index,
                             role=self.role, **fields)

    def train(self, request):
        update_started = time.perf_counter()
        validate_request(request, self.role, self.rollout_verifier)
        if not self.checkpoint_store.verify(request.policy_before): raise ValueError('before checkpoint failed verification')
        if not self.run_id or not self.after_revision or self.output_dir.exists():
            raise ValueError('unique run, revision and unused checkpoint destination required')
        if (self.mini_batch_size < 1 or self.epochs < 1
                or self.mini_batch_size % self.checkpoint_store.world_size
                or len(request.rollouts) % self.checkpoint_store.world_size
                or self.mini_batch_size > len(request.rollouts)
                or len(request.rollouts) % self.mini_batch_size):
            raise ValueError('rollout and mini-batch sizes must divide across the FSDP world and each other')
        verify_environment = getattr(self.runtime_verifier, 'verify_runtime_environment', None)
        if (not callable(verify_environment)
                or verify_environment(self.role, self.worker_group) is not True):
            raise ValueError('VERL worker role and FSDP2 runtime environment were not verified')
        before_step = checkpoint_step(request.policy_before)
        # VERL's global_steps counts outer trainer batches, not the inner
        # optimizer minibatches/epochs executed by one update_actor call.
        after_step = before_step + 1
        if self.after_revision != str(after_step):
            raise ValueError('after checkpoint revision must equal the actual VERL optimizer step')
        require_verl_version()
        rows = request.rollouts
        encoded = padded_batch([row.tokens for row in rows], self.pad_token_id)
        import numpy as np
        import torch
        from verl import DataProto
        from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, compute_reinforce_plus_plus_outcome_advantage
        from verl.utils import tensordict_utils as tu
        from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding
        tensors = {key: torch.tensor(value, dtype=torch.long) for key, value in encoded.items()}
        verify_resident = getattr(self.runtime_verifier, 'verify_resident_training_state', None)
        resident = (callable(verify_resident)
                    and verify_resident(self.role, self.worker_group, request.policy_before) is True)
        if not resident:
            self.worker_group.load_checkpoint(self.checkpoint_store.resolve(request.policy_before))
            if self.runtime_verifier.verify_runtime(
                    self.role, self.worker_group, request.policy_before) is not True:
                raise ValueError('restored FSDP2 actor does not match the requested checkpoint')
        proto = DataProto.from_dict(tensors=tensors)
        infer = left_right_2_no_padding(proto.to_tensordict())
        tu.assign_non_tensor(infer, calculate_entropy=False, compute_loss=False, temperature=1.0)
        logprob_started = time.perf_counter()
        result = self.worker_group.compute_log_prob(infer)
        self._metric(f'{self.role}_logprob_seconds', time.perf_counter() - logprob_started)
        old = no_padding_2_padding(tu.get(result, 'log_probs'), infer).float()
        if not torch.isfinite(old).all(): raise ValueError('nonfinite old policy log probabilities')
        rewards = torch.zeros_like(tensors['response_mask'], dtype=torch.float32)
        for i, row in enumerate(rows):
            last = max(j for j, enabled in enumerate(row.tokens.mask) if enabled)
            rewards[i, last] = row.reward
        if request.algorithm.lower() == 'grpo':
            advantages, _ = compute_grpo_outcome_advantage(rewards, tensors['response_mask'],
                np.asarray([row.group_id for row in rows], dtype=object))
        elif request.algorithm.lower() == 'reinforce_baseline':
            # Sequential Service baselines are nonstationary. Apply a scalar
            # control variate directly to each captured Designer action token;
            # do not use GRPO or batch whitening across incomparable rounds.
            advantages = torch.zeros_like(tensors['response_mask'], dtype=torch.float32)
            for i, row in enumerate(rows):
                selected = torch.as_tensor(row.tokens.mask, dtype=torch.bool,
                                           device=advantages.device)
                width = min(selected.numel(), advantages.shape[1])
                selected = selected[:width]
                advantages[i, :width] = torch.where(selected,
                    torch.full((width,), row.reward - request.advantage_baseline,
                               dtype=torch.float32, device=advantages.device),
                    torch.zeros((width,), dtype=torch.float32, device=advantages.device))
        else:
            from types import SimpleNamespace
            advantages, _ = compute_reinforce_plus_plus_outcome_advantage(
                rewards, tensors['response_mask'], config=SimpleNamespace(gamma=1.0))
        tensors['old_log_probs'], tensors['advantages'] = old, advantages
        update_batch = left_right_2_no_padding(DataProto.from_dict(tensors=tensors).to_tensordict())
        tu.assign_non_tensor(update_batch, calculate_entropy=False, compute_loss=True,
            global_batch_size=len(rows), mini_batch_size=self.mini_batch_size,
            epochs=self.epochs, seed=42, dataloader_kwargs={'shuffle': False}, temperature=1.0)
        backward_started = time.perf_counter()
        metrics = self.worker_group.update_actor(update_batch)
        self._metric(f'{self.role}_backward_seconds', time.perf_counter() - backward_started)
        checkpoint_started = time.perf_counter()
        self.worker_group.save_checkpoint(str(self.output_dir), global_step=after_step)
        self._metric('checkpoint_save_seconds', time.perf_counter() - checkpoint_started)
        verification_started = time.perf_counter()
        after = self.checkpoint_store.publish(request.policy_before.policy,
            self.after_revision, request.policy_before.tokenizer, self.output_dir)
        if not self.checkpoint_store.verify(after): raise ValueError('after checkpoint failed verification')
        if self.checkpoint_manifest_path is not None:
            self.checkpoint_store.save_manifest(self.checkpoint_manifest_path)
        self._metric('checkpoint_verify_seconds', time.perf_counter() - verification_started,
                     checkpoint_revision=after.revision)
        if self.runtime_verifier.verify_runtime(self.role, self.worker_group, after) is not True:
            raise ValueError('saved checkpoint does not match the full-parameter actor runtime')
        sync_started = time.perf_counter()
        if self.weight_sync.synchronize(self.worker_group, after) is not True:
            raise ValueError('rollout weight synchronization was not verified')
        self._metric('weight_sync_seconds', time.perf_counter() - sync_started,
                     checkpoint_revision=after.revision)
        peak_memory = getattr(self.worker_group, 'get_peak_npu_memory_bytes', None)
        self._metric('npu_peak_memory_bytes', peak_memory() if callable(peak_memory) else None,
                     available=callable(peak_memory))
        evidence = self.run_recorder.record_update(self.run_id, request, after, metrics)
        completed = CompletedUpdate(request, after, self.run_id, evidence)
        if self.update_verifier.verify_update(completed) is not True:
            raise ValueError('completed update evidence failed verification')
        self._metric(f'{self.role}_update_backend_seconds', time.perf_counter() - update_started)
        return completed


class VerlServiceRLBackend(VerlGRPOBackend):
    def __init__(self, *args, **kwargs): super().__init__('service', *args, **kwargs)

class VerlDesignerRLBackend(VerlGRPOBackend):
    def __init__(self, *args, **kwargs): super().__init__('designer', *args, **kwargs)
