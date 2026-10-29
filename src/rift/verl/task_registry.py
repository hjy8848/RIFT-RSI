"""Trusted in-process registry for native Usami task certificates."""
from dataclasses import dataclass
from datetime import timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
from rift.schemas import DataPurpose, Identity, Split

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
    def __init__(self, split_registry, *, storage_path=None):
        self.split_registry = split_registry
        self.storage_path = None if storage_path is None else Path(storage_path)
        self._records = {}
    @property
    def task_refs(self):
        return tuple(sorted(self._records))
    def register(self, record):
        if not isinstance(record, CertifiedNativeTask): raise TypeError('native certified task required')
        ref = record.task.record_hash
        existing = self._records.get(ref)
        if existing is not None:
            if existing == record and self._check(existing):
                return existing
            raise ValueError('TaskRef is already bound to different certified task evidence')
        if not self._check(record): raise ValueError('task or certificate is not authorized')
        self._records[ref] = record
        try:
            if self.storage_path is not None:
                self.save()
        except BaseException:
            del self._records[ref]
            raise
        return record

    @staticmethod
    def _encode_record(record):
        identity = record.identity
        return {
            'world': record.world.model_dump(mode='json'),
            'task': record.task.model_dump(mode='json'),
            'profile': record.profile.model_dump(mode='json'),
            'scenario': record.scenario.model_dump(mode='json'),
            'certificate': record.certificate.model_dump(mode='json'),
            'identity': {'world': identity.world, 'task': identity.task,
                'family': identity.family, 'lineage': identity.lineage,
                'split': identity.split.value,
                'purpose': None if identity.purpose is None else identity.purpose.value},
            'external_events': [event.model_dump(mode='json') for event in record.external_events],
        }

    def save(self, path=None):
        """Persist validated native task certificates for exact deployment recovery."""
        target = Path(path) if path is not None else self.storage_path
        if target is None:
            raise ValueError('a persistent CertifiedTaskRegistry path is required')
        entries = {ref: self._encode_record(record)
                   for ref, record in sorted(self._records.items())}
        canonical = json.dumps(entries, sort_keys=True, ensure_ascii=False,
                               separators=(',', ':'))
        payload = json.dumps({'schema_version': 1,
            'split_policy_version': self.split_registry.policy_version,
            'records_digest': sha256(canonical.encode('utf-8')).hexdigest(),
            'records': entries}, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = target.with_name(target.name + f'.{os.getpid()}.tmp')
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        return sha256(payload.encode('utf-8')).hexdigest()

    @classmethod
    def load(cls, path, split_registry):
        """Reload and re-certify every persisted native task before it is usable."""
        source = Path(path)
        raw = json.loads(source.read_text(encoding='utf-8'))
        if (not isinstance(raw, dict) or raw.get('schema_version') != 1
                or raw.get('split_policy_version') != split_registry.policy_version
                or not isinstance(raw.get('records'), dict)):
            raise ValueError('unsupported or mismatched CertifiedTaskRegistry store')
        canonical = json.dumps(raw['records'], sort_keys=True, ensure_ascii=False,
                               separators=(',', ':'))
        if sha256(canonical.encode('utf-8')).hexdigest() != raw.get('records_digest'):
            raise ValueError('CertifiedTaskRegistry content digest mismatch')
        from usami_v2.commerce.models import ExternalEvent, World
        from usami_v2.runtime.customer import CustomerProfileV3
        from usami_v2.scenarios.task_certification import CustomerScenario, TaskCertification
        from usami_v2.scenarios.tasks import TaskSpecV3
        registry = cls(split_registry)
        for task_ref, row in raw['records'].items():
            identity_raw = row['identity']
            identity = Identity(identity_raw['world'], identity_raw['task'],
                identity_raw['family'], identity_raw['lineage'],
                Split(identity_raw['split']),
                None if identity_raw['purpose'] is None else DataPurpose(identity_raw['purpose']))
            record = CertifiedNativeTask(
                World.model_validate(row['world']),
                TaskSpecV3.model_validate(row['task']),
                CustomerProfileV3.model_validate(row['profile']),
                CustomerScenario.model_validate(row['scenario']),
                TaskCertification.model_validate(row['certificate']), identity,
                tuple(ExternalEvent.model_validate(event) for event in row['external_events']))
            if record.task.record_hash != task_ref:
                raise ValueError('CertifiedTaskRegistry key differs from native TaskRef')
            registry.register(record)
        registry.storage_path = source
        return registry
    def _check(self, record):
        from usami_v2.scenarios.task_certification import verify_task_certificate
        identity, task, world, profile, scenario, cert = (record.identity, record.task, record.world,
            record.profile, record.scenario, record.certificate)
        if not isinstance(identity, Identity): return False
        expected = {
            DataPurpose.TRAIN: (Split.TRAIN, ("train",)),
            DataPurpose.BANK_CANDIDATE: (None, ("train", "validation")),
            DataPurpose.REPLAY_LOGP_FEEDBACK: (Split.REPLAY, ("train", "validation")),
            DataPurpose.VALIDATION: (Split.VALIDATION, ("validation",)),
            DataPurpose.TEST: (Split.TEST, ("heldout",)),
        }
        purpose = identity.purpose
        if purpose is None:
            # Legacy training records predate logical purposes and remain readable.
            expected_split, partitions = Split.TRAIN, ("train",)
        else:
            expected_split, partitions = expected[purpose]
            if purpose is DataPurpose.BANK_CANDIDATE:
                expected_split = (Split.TRAIN if task.partition == "train"
                                  else Split.VALIDATION if task.partition == "validation"
                                  else None)
        if (identity.split is not expected_split
                or not self.split_registry.contains(identity) or identity.task != task.record_hash
                or identity.world != world.state_hash or task.partition not in partitions
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
    def resolve_record(self, task_ref):
        record = self._records.get(task_ref)
        if record is None or not self._check(record):
            raise ValueError('certified task evidence unavailable')
        return record
    def purpose(self, task_ref):
        return self.resolve_record(task_ref).identity.purpose
    def verify_task(self, task_ref, world, task, profile, *, purpose=None):
        record = self._records.get(task_ref)
        return bool(record is not None and record.world == world and record.task == task
                    and record.profile == profile and self._check(record)
                    and (purpose is None or record.identity.purpose is purpose))
