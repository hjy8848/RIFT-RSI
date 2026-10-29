"""Researcher-only final evaluation, separate from feedback and bank updates."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json

from rift.banks.records import EvidenceLevel, ObservedTrial
from rift.schemas import Checkpoint, DataPurpose, Identity, Outcome, Split


@dataclass(frozen=True)
class IndependentTestResult:
    checkpoint: Checkpoint
    manifest_digest: str
    protocol_digest: str
    trials: tuple[ObservedTrial, ...]
    evidence_level: EvidenceLevel

    def __post_init__(self):
        if not self.manifest_digest or not self.protocol_digest or not self.trials:
            raise ValueError('complete independent test result required')
        if any(row.checkpoint != self.checkpoint or row.protocol_digest != self.protocol_digest
               for row in self.trials):
            raise ValueError('test trials differ from checkpoint or protocol')
        if len({(row.task_ref, row.seed) for row in self.trials}) != len(self.trials):
            raise ValueError('duplicate independent test task/seed')
        if any(row.evidence_level is not self.evidence_level for row in self.trials):
            raise ValueError('independent test mixes evidence levels')

    @property
    def task_success_rate(self):
        by_task = {}
        for row in self.trials:
            if row.valid:
                by_task.setdefault(row.task_ref, []).append(row.outcome is Outcome.SUCCESS)
        if not by_task:
            return None
        return sum(sum(values) / len(values) for values in by_task.values()) / len(by_task)


@dataclass(frozen=True)
class IndependentTestTask:
    task_ref: str
    identity: Identity

    def __post_init__(self):
        if (not self.task_ref or not isinstance(self.identity, Identity)
                or self.identity.split is not Split.TEST
                or self.identity.purpose is not DataPurpose.TEST):
            raise ValueError('isolated TEST TaskRef and identity required')


class IndependentTestRunner:
    def __init__(self, evaluator, evidence_verifier, *, split_registry):
        self.evaluator, self.evidence_verifier, self.split_registry = evaluator, evidence_verifier, split_registry

    def run(self, checkpoint, tasks, *, protocol_digest, trials_per_task=3, seed=0):
        tasks = tuple(tasks)
        if (not tasks or type(trials_per_task) is not int or trials_per_task < 1
                or type(seed) is not int or seed < 0 or not protocol_digest):
            raise ValueError('independent test manifest and evaluation protocol required')
        if any(not isinstance(task, IndependentTestTask) for task in tasks):
            raise TypeError('IndependentTestTask records required')
        if len({task.task_ref for task in tasks}) != len(tasks):
            raise ValueError('duplicate independent test TaskRef')
        for task in tasks:
            task_ref, identity = task.task_ref, task.identity
            if (not isinstance(identity, Identity) or identity.split is not Split.TEST
                    or identity.purpose is not DataPurpose.TEST or not self.split_registry.contains(identity)):
                raise ValueError('final Test task is not registered in the isolated TEST purpose')
        manifest = [(task.task_ref, asdict(task.identity)) for task in tasks]
        manifest_digest = sha256(json.dumps(manifest, sort_keys=True, default=str,
            separators=(',', ':')).encode()).hexdigest()
        trials = []
        for task in tasks:
            task_ref, identity = task.task_ref, task.identity
            for index in range(trials_per_task):
                derived = sha256(f"rift-test-v1:{seed}:{task_ref}:{index}".encode()).digest()
                trial_seed = int.from_bytes(derived[:4], 'big')
                row = self.evaluator.evaluate(task_ref, checkpoint, trial_seed, protocol_digest)
                if (not isinstance(row, ObservedTrial) or row.task_ref != task_ref
                        or row.checkpoint != checkpoint or row.seed != trial_seed
                        or row.protocol_digest != protocol_digest or row.public_summary is not None):
                    raise ValueError('independent test evaluator returned mismatched/private data')
                if self.evidence_verifier.verify_trial(row, identity) is not True:
                    raise ValueError('independent test Episode evidence failed verification')
                trials.append(row)
        levels = {row.evidence_level for row in trials}
        if len(levels) != 1:
            raise ValueError('independent test must use one evidence level')
        return IndependentTestResult(checkpoint, manifest_digest, protocol_digest,
                                     tuple(trials), next(iter(levels)))
