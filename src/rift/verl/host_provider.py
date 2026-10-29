"""Trusted worker-side resolution of opaque VERL dataset task references."""
from rift.verl.usami_session import NativeUsamiSession

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
        if task.partition != 'train' or self.task_authorizer.verify_task(task_ref, world, task, profile) is not True:
            raise ValueError('task is not an authorized certified training task')
        checkpoint = self.checkpoint_source.current()
        if self.checkpoint_store.verify(checkpoint) is not True:
            raise ValueError('Service checkpoint failed verification')
        customer = self.customer_factory.create(task_ref)
        session = NativeUsamiSession.with_verl_parser(world, task, profile, customer, tokenizer,
            self.parser_format, budget=self.budget, mode=self.mode, external_events=external_events,
            group_id=group_id)
        return session, checkpoint, self.checkpoint_store, self.capture_store
