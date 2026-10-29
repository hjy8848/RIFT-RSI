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
    async def open_rollout(self, task_ref, group_id, session_id, tokenizer):
        if not task_ref or not group_id or type(session_id) is not int or session_id < 0:
            raise ValueError('opaque task, group and sampled session identity are required')
        world, task, profile, external_events = self.task_resolver.resolve(task_ref)
        purpose = getattr(self.task_resolver, 'purpose', lambda _: DataPurpose.TRAIN)(task_ref)
        if (purpose not in (None, DataPurpose.TRAIN)
                or task.partition != 'train'
                or self.task_authorizer.verify_task(task_ref, world, task, profile) is not True):
            raise ValueError('task is not an authorized certified training task')
        checkpoint = self.checkpoint_source.current()
        if self.checkpoint_store.verify(checkpoint) is not True:
            raise ValueError('Service checkpoint failed verification')
        customer = self.customer_factory.create(task_ref)
        session = NativeUsamiSession.with_verl_parser(world, task, profile, customer, tokenizer,
            self.parser_format, budget=self.budget, mode=self.mode, external_events=external_events,
            group_id=group_id)
        return session, checkpoint, self.checkpoint_store, self.capture_store

    async def open_probe(self, task_ref, checkpoint, seed, protocol_ref, tokenizer):
        """Open one gradient-free task evaluation against an explicitly selected checkpoint."""
        if not task_ref or type(seed) is not int or seed < 0 or not protocol_ref:
            raise ValueError('opaque TaskRef, evaluation seed and protocol reference required')
        purpose_resolver = getattr(self.task_resolver, 'purpose', None)
        if not callable(purpose_resolver):
            raise ValueError('task resolver must expose Host-private logical task purpose')
        purpose = purpose_resolver(task_ref)
        if purpose not in (DataPurpose.RECOVERY_FEEDBACK, DataPurpose.RETENTION_FEEDBACK):
            raise ValueError('only frozen Recovery or Retention tasks may enter the Probe runner')
        world, task, profile, external_events = self.task_resolver.resolve(task_ref)
        if self.task_authorizer.verify_task(task_ref, world, task, profile, purpose=purpose) is not True:
            raise ValueError('Probe task authorization failed')
        if self.checkpoint_store.verify(checkpoint) is not True:
            raise ValueError('selected Probe checkpoint failed verification')
        customer = self.customer_factory.create(task_ref)
        rollout_id = sha256(f"rift-probe:{task_ref}:{checkpoint.weights_digest}:{seed}:{protocol_ref}".encode()).hexdigest()[:32]
        session = NativeUsamiSession.with_verl_parser(world, task, profile, customer, tokenizer,
            self.parser_format, budget=self.budget, mode=self.mode, external_events=external_events,
            rollout_id=rollout_id)
        return session, checkpoint, self.checkpoint_store, self.capture_store
