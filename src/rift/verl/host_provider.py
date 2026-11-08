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
                 checkpoint_store, capture_store, *, budget, parser_format, mode='reference',
                 tool_profile='full9', evaluation_protocol=None):
        self.task_resolver, self.task_authorizer = task_resolver, task_authorizer
        self.customer_factory, self.checkpoint_source = customer_factory, checkpoint_source
        self.checkpoint_store, self.capture_store = checkpoint_store, capture_store
        self.budget, self.parser_format, self.mode = budget, parser_format, mode
        from usami_v2.runtime.agents import service_tools
        self.tools = tuple(row['function']['name'] for row in service_tools(tool_profile))
        self.tool_profile = tool_profile
        if tool_profile == 'core4' and evaluation_protocol is None:
            raise ValueError('core4 requires a profile-bound evaluation protocol')
        from rift.runtime.service_protocol import service_protocol_digest
        self.protocol_digest = (None if evaluation_protocol is None else
                                service_protocol_digest(evaluation_protocol, tool_profile))
    def _customer(self, task_ref, task, seed):
        from usami_v2.scenarios.retail_tasks import RetailTask
        if isinstance(task, RetailTask):
            import os
            if self.tool_profile != 'retail':
                raise ValueError('retail tasks require the retail tool profile')
            if os.environ.get('RIFT_CUSTOMER_URL'):
                raise ValueError('retail tasks do not yet support an LLM customer')
            from usami_v2.runtime.retail_runner import ScriptedRetailCustomer
            return ScriptedRetailCustomer()
        return self.customer_factory.create(task_ref, seed=seed)

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
        if self.tool_profile == 'core4':
            from usami_v2.scenarios.designer import ensure_task_tools
            ensure_task_tools(task, world, self.tools)
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
        customer = self._customer(task_ref, task, customer_seed)
        session = NativeUsamiSession.with_verl_parser(world, task, profile, customer, tokenizer,
            self.parser_format, budget=self.budget, mode=self.mode, external_events=external_events,
            group_id=group_id, group_size=group_size, rollout_id=rollout_id,
            sampling_seed=sampling_seed,
            sampling_scope=f'rift-service:{task_ref}:{group_id}:{rollout_id}',
            task_ref=task_ref, rift_mode='train', customer_seed=customer_seed,
            tool_profile=self.tool_profile)
        return session, checkpoint, self.checkpoint_store, self.capture_store

    async def open_probe(self, task_ref, checkpoint, seed, protocol_ref, tokenizer,
                         protocol_digest=None):
        """Open one gradient-free task evaluation against an explicitly selected checkpoint."""
        if (not task_ref or type(seed) is not int or seed < 0 or not protocol_ref
                or not protocol_digest):
            raise ValueError('opaque TaskRef, evaluation seed and frozen protocol identity required')
        if self.protocol_digest is not None and protocol_digest != self.protocol_digest:
            raise ValueError('Probe protocol does not match the active Service tool profile')
        purpose_resolver = getattr(self.task_resolver, 'purpose', None)
        if not callable(purpose_resolver):
            raise ValueError('task resolver must expose Host-private logical task purpose')
        purpose = purpose_resolver(task_ref)
        if purpose is not DataPurpose.BANK_CANDIDATE:
            raise ValueError('only frozen bank candidates may enter the Probe runner')
        world, task, profile, external_events = self.task_resolver.resolve(task_ref)
        if self.tool_profile == 'core4':
            from usami_v2.scenarios.designer import ensure_task_tools
            ensure_task_tools(task, world, self.tools)
        if self.task_authorizer.verify_task(task_ref, world, task, profile, purpose=purpose) is not True:
            raise ValueError('Probe task authorization failed')
        if self.checkpoint_store.verify(checkpoint) is not True:
            raise ValueError('selected Probe checkpoint failed verification')
        customer = self._customer(task_ref, task, seed)
        rollout_id = sha256(f"rift-probe:{task_ref}:{checkpoint.weights_digest}:{seed}:{protocol_ref}".encode()).hexdigest()[:32]
        sampling_scope = sha256(f"rift-probe-sampling-v1:{task_ref}:{seed}:{protocol_ref}".encode()).hexdigest()
        session = NativeUsamiSession.with_verl_parser(world, task, profile, customer, tokenizer,
            self.parser_format, budget=self.budget, mode=self.mode, external_events=external_events,
            rollout_id=rollout_id, sampling_seed=seed, sampling_scope=sampling_scope,
            task_ref=task_ref, rift_mode='probe', protocol_ref=protocol_ref,
            protocol_digest=protocol_digest, tool_profile=self.tool_profile)
        return session, checkpoint, self.checkpoint_store, self.capture_store
