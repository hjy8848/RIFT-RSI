"""Trusted in-process registry for native Usami task certificates."""
from dataclasses import dataclass
from datetime import timedelta
from rift.schemas import Identity, Split

@dataclass(frozen=True)
class CertifiedNativeTask:
    world: object
    task: object
    profile: object
    scenario: object
    certificate: object
    identity: Identity
    external_events: tuple = ()

class CertifiedTaskRegistry:
    def __init__(self, split_registry):
        self.split_registry = split_registry
        self._records = {}
    def register(self, record):
        if not isinstance(record, CertifiedNativeTask): raise TypeError('native certified task required')
        ref = record.task.record_hash
        if ref in self._records: raise ValueError('task already registered')
        if not self._check(record): raise ValueError('task or certificate is not authorized')
        self._records[ref] = record
    def _check(self, record):
        from usami_v2.scenarios.task_certification import verify_task_certificate
        identity, task, world, profile, scenario, cert = (record.identity, record.task, record.world,
            record.profile, record.scenario, record.certificate)
        if (not isinstance(identity, Identity) or identity.split is not Split.TRAIN
                or not self.split_registry.contains(identity) or identity.task != task.record_hash
                or identity.world != world.state_hash or task.partition != 'train'
                or task.world_hash != world.state_hash or profile.customer_id != task.customer_id
                or scenario.customer_id != task.customer_id or cert.task_hash != task.record_hash
                or cert.world_hash != world.state_hash or cert.status != 'CERTIFIED'
                or not cert.full_task_reachability_certified): return False
        if task.protocol != 'commerce-task-v3' or profile.protocol != 'commerce-customer-profile-v3': return False
        lines = tuple(goal.line_id for goal in task.remedy_goals)
        if (profile.delegated_line_ids != lines or scenario.delegated_line_ids != lines
                or profile.intent_bundle_hash != task.request_bundle.record_hash
                or set(profile.allowed_preparations) != set(scenario.allowed_preparations)
                or profile.accepted_remedies != scenario.accepted_remedies
                or profile.maximum_extra_charge_cents != scenario.maximum_extra_charge_cents
                or profile.return_pickup_slot != scenario.return_pickup_slot): return False
        if ((scenario.latest_arrival is None) != (profile.latest_arrival_days is None)
                or (scenario.latest_arrival is not None
                    and world.clock + timedelta(days=profile.latest_arrival_days) < scenario.latest_arrival)):
            return False
        return verify_task_certificate(world, task, scenario, cert)['verified'] is True
    def resolve(self, task_ref):
        record = self._records.get(task_ref)
        if record is None or not self._check(record): raise ValueError('certified task evidence unavailable')
        return record.world, record.task, record.profile, record.external_events
    def verify_task(self, task_ref, world, task, profile):
        record = self._records.get(task_ref)
        return bool(record is not None and record.world == world and record.task == task
                    and record.profile == profile and self._check(record))
