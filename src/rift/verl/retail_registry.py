"""Persistent registry for re-certified retail tasks."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from rift.schemas import DataPurpose, Identity, Split
from .task_registry import CertifiedTaskRegistry


@dataclass(frozen=True)
class CertifiedRetailTask:
    world: object
    task: object
    certificate: dict
    identity: Identity


class RetailTaskRegistry(CertifiedTaskRegistry):
    def register(self, record):
        if not isinstance(record, CertifiedRetailTask):
            raise TypeError("certified retail task required")
        return self._register(record, add_identity=True)

    @staticmethod
    def _encode_record(record):
        identity = record.identity
        return {"world": record.world.model_dump(mode="json"), "task": record.task.model_dump(mode="json"),
                "certificate": record.certificate,
                "identity": {"world": identity.world, "task": identity.task, "family": identity.family,
                             "lineage": identity.lineage, "split": identity.split.value,
                             "purpose": None if identity.purpose is None else identity.purpose.value}}

    def _check(self, record, *, require_registered_identity=True):
        from usami_v2.scenarios.retail_tasks import RetailTask, certify_retail_task
        identity, world, task = record.identity, record.world, record.task
        if not isinstance(identity, Identity) or not isinstance(task, RetailTask):
            return False
        allowed = {DataPurpose.TRAIN: (Split.TRAIN, "train"),
                   DataPurpose.VALIDATION: (Split.VALIDATION, "validation"),
                   DataPurpose.TEST: (Split.TEST, "heldout")}
        if (identity.purpose not in allowed or allowed[identity.purpose] != (identity.split, task.partition)
                or identity.world != world.state_hash or identity.task != task.record_hash
                or task.world_hash != world.state_hash
                or (require_registered_identity and not self.split_registry.contains(identity))):
            return False
        return certify_retail_task(world, task) == record.certificate

    @classmethod
    def load(cls, path, split_registry):
        from usami_v2.commerce.models import World
        from usami_v2.scenarios.retail_tasks import RetailTask
        source = Path(path)
        raw = json.loads(source.read_text(encoding="utf-8"))
        if (not isinstance(raw, dict) or raw.get("schema_version") != 1
                or raw.get("split_policy_version") != split_registry.policy_version
                or not isinstance(raw.get("records"), dict)):
            raise ValueError("unsupported or mismatched RetailTaskRegistry store")
        canonical = json.dumps(raw["records"], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if sha256(canonical.encode("utf-8")).hexdigest() != raw.get("records_digest"):
            raise ValueError("RetailTaskRegistry content digest mismatch")
        registry = cls(split_registry)
        for ref, row in raw["records"].items():
            ident = row["identity"]
            identity = Identity(ident["world"], ident["task"], ident["family"], ident["lineage"],
                                Split(ident["split"]), DataPurpose(ident["purpose"]))
            record = CertifiedRetailTask(World.model_validate(row["world"]), RetailTask.model_validate(row["task"]),
                                         row["certificate"], identity)
            if record.task.record_hash != ref or not split_registry.contains(identity):
                raise ValueError("retail registry key or split identity differs")
            registry._register(record, add_identity=False)
        registry.storage_path = source
        return registry

    def resolve(self, task_ref):
        record = self.resolve_record(task_ref)
        return record.world, record.task, None, ()

    def verify_task(self, task_ref, world, task, profile, *, purpose=None):
        try:
            record = self.resolve_record(task_ref)
        except ValueError:
            return False
        return (profile is None and record.world == world and record.task == task
                and (purpose is None or record.identity.purpose is purpose))


class TaskResolver:
    """Resolve either protocol without altering legacy registry validation."""
    def __init__(self, legacy, retail):
        self.legacy, self.retail = legacy, retail

    @property
    def task_refs(self):
        return tuple(sorted(set(self.legacy.task_refs) | set(self.retail.task_refs)))

    def _registry(self, ref):
        if ref in self.retail.task_refs:
            return self.retail
        return self.legacy

    def resolve(self, ref):
        return self._registry(ref).resolve(ref)

    def resolve_record(self, ref):
        return self._registry(ref).resolve_record(ref)

    def purpose(self, ref):
        return self._registry(ref).purpose(ref)

    def verify_task(self, ref, world, task, profile, *, purpose=None):
        return self._registry(ref).verify_task(ref, world, task, profile, purpose=purpose)
