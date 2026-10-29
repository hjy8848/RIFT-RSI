"""Trusted worker-side resolution of opaque VERL dataset task references."""
from rift.verl.usami_session import NativeUsamiSession
from rift.schemas import DataPurpose
from hashlib import sha256

class NativeHostProvider:
    """Dependencies must be constructed inside the trusted worker process.

    The resolver and authorizer own private task/profile records. Only opaque
    identifiers enter VERL's dataset/TransferQueue; no native object is put in
    a sample field or an AgentLoopOutput.
    """
    def __init__(self, task_resolver, task_authorizer, customer_factory, checkpoint_source,
                 checkpoint_store, capture_store, *, budget, parser_format, mode='reference'):
        self.task_resolver, self.task_authorizer = task_resolver, task_authorizer
        self.customer_factory, self.checkpoint_source = customer_factory, checkpoint_source
        self.checkpoint_store, self.capture_store = checkpoint_store, capture_store
        self.budget, self.parser_format, self.mode = budget, parser_format, mode
    async def open_rollout(self, task_ref, group_id, session_id, tokenizer, *,
                           sampling_seed, customer_seed, rollout_id,
                           expected_checkpoint=None, group_size=None):
        if (not task_ref or not group_id or type(session_id) is not int or session_id < 0
                or type(sampling_seed) is not int or sampling_seed < 0
                or type(customer_seed) is not int or customer_seed < 0
                or not isinstance(rollout_id, str) or not rollout_id
                or type(group_size) is not int or group_size < 2):
            raise ValueError('opaque task/group, rollout/session IDs, seeds and complete GRPO group are required')
        world, task, profile, external_events = self.task_resolver.resolve(task_ref)
        purpose = getattr(self.task_resolver, 'purpose', lambda _: DataPurpose.TRAIN)(task_ref)
        if (purpose not in (None, DataPurpose.TRAIN)
                or task.partition != 'train'
                or self.task_authorizer.verify_task(task_ref, world, task, profile) is not True):
            raise ValueError('task is not an authorized certified training task')
        checkpoint = self.checkpoint_source.current()
        if expected_checkpoint is not None:
            from rift.schemas import Checkpoint
            if isinstance(expected_checkpoint, str):
                import json
                expected_checkpoint = json.loads(expected_checkpoint)
            expected = (expected_checkpoint if isinstance(expected_checkpoint, Checkpoint)
                        else Checkpoint(**expected_checkpoint))
            if checkpoint != expected:
                raise ValueError('Service checkpoint changed after the rollout batch was frozen')
        if self.checkpoint_store.verify(checkpoint) is not True:
            raise ValueError('Service checkpoint failed verification')
        # Customer randomness is common within a GRPO group so the first
        # Service prompt is identical. Only model sampling varies per member.
        customer = self.customer_factory.create(task_ref, seed=customer_seed)
        session = NativeUsamiSession.with_verl_parser(world, task, profile, customer, tokenizer,
            self.parser_format, budget=self.budget, mode=self.mode, external_events=external_events,
            group_id=group_id, group_size=group_size, rollout_id=rollout_id,
            sampling_seed=sampling_seed,
            sampling_scope=f'rift-service:{task_ref}:{group_id}:{rollout_id}',
            task_ref=task_ref, rift_mode='train', customer_seed=customer_seed)
        return session, checkpoint, self.checkpoint_store, self.capture_store

    async def open_probe(self, task_ref, checkpoint, seed, protocol_ref, tokenizer,
                         protocol_digest=None):
        """Open one gradient-free task evaluation against an explicitly selected checkpoint."""
        if (not task_ref or type(seed) is not int or seed < 0 or not protocol_ref
                or not protocol_digest):
            raise ValueError('opaque TaskRef, evaluation seed and frozen protocol identity required')
        purpose_resolver = getattr(self.task_resolver, 'purpose', None)
        if not callable(purpose_resolver):
            raise ValueError('task resolver must expose Host-private logical task purpose')
        purpose = purpose_resolver(task_ref)
        if purpose is not DataPurpose.BANK_CANDIDATE:
            raise ValueError('only frozen bank candidates may enter the Probe runner')
        world, task, profile, external_events = self.task_resolver.resolve(task_ref)
        if self.task_authorizer.verify_task(task_ref, world, task, profile, purpose=purpose) is not True:
            raise ValueError('Probe task authorization failed')
        if self.checkpoint_store.verify(checkpoint) is not True:
            raise ValueError('selected Probe checkpoint failed verification')
        customer = self.customer_factory.create(task_ref, seed=seed)
        rollout_id = sha256(f"rift-probe:{task_ref}:{checkpoint.weights_digest}:{seed}:{protocol_ref}".encode()).hexdigest()[:32]
        sampling_scope = sha256(f"rift-probe-sampling-v1:{task_ref}:{seed}:{protocol_ref}".encode()).hexdigest()
        session = NativeUsamiSession.with_verl_parser(world, task, profile, customer, tokenizer,
            self.parser_format, budget=self.budget, mode=self.mode, external_events=external_events,
            rollout_id=rollout_id, sampling_seed=seed, sampling_scope=sampling_scope,
            task_ref=task_ref, rift_mode='probe', protocol_ref=protocol_ref,
            protocol_digest=protocol_digest)
        return session, checkpoint, self.checkpoint_store, self.capture_store
