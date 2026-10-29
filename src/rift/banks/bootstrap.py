"""Checkpoint-relative task evaluation and initial bank construction."""
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Protocol, Sequence

from rift.banks.core import BankManager
from rift.banks.records import ObservedTrial, TaskEvidenceRecord
from rift.schemas import Checkpoint, DataPurpose, Identity, Split


@dataclass(frozen=True)
class BootstrapTask:
    task_ref: str
    identity: Identity

    def __post_init__(self):
        if (not self.task_ref or not isinstance(self.identity, Identity)
                or self.identity.purpose is not DataPurpose.BANK_CANDIDATE
                or self.identity.split is Split.TEST):
            raise ValueError("bootstrap tasks must be unlabelled non-TEST bank candidates")


class BootstrapEvaluator(Protocol):
    def evaluate(self, task_ref: str, checkpoint: Checkpoint, seed: int,
                 protocol_digest: str) -> ObservedTrial: ...


class BootstrapEvidenceVerifier(Protocol):
    def verify_trial(self, trial: ObservedTrial, task: BootstrapTask) -> bool: ...


@dataclass(frozen=True)
class BootstrapResult:
    checkpoint: Checkpoint
    protocol_digest: str
    seed: int
    evaluation_trials: int
    records: tuple[TaskEvidenceRecord, ...]
    bank_manager: BankManager

    @property
    def unresolved(self):
        return tuple(row for row in self.records if row.success_rate is None)


def _trial_seed(base_seed, task_ref, index):
    raw = f"rift-bootstrap-v1:{base_seed}:{task_ref}:{index}".encode("utf-8")
    return int.from_bytes(sha256(raw).digest()[:4], "big")


def _protocol_digest(protocol):
    encoded = json.dumps(protocol, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), default=str)
    return sha256(encoded.encode("utf-8")).hexdigest()


class BootstrapRunner:
    """Bootstrap observations only; this class has no trainer or model API."""
    def __init__(self, evaluator, evidence_verifier, *, success_threshold=0.75,
                 failure_threshold=0.25):
        self.evaluator, self.evidence_verifier = evaluator, evidence_verifier
        self.success_threshold, self.failure_threshold = success_threshold, failure_threshold
        if not callable(getattr(evaluator, "evaluate", None)):
            raise TypeError("BootstrapEvaluator required")
        if not callable(getattr(evidence_verifier, "verify_trial", None)):
            raise TypeError("BootstrapEvidenceVerifier required")

    def run(self, task_manifest: Sequence[BootstrapTask], checkpoint: Checkpoint, *,
            protocol, trials_per_task=3, seed=0):
        tasks = tuple(task_manifest)
        if not tasks or len({row.task_ref for row in tasks}) != len(tasks):
            raise ValueError("non-empty task manifest with distinct TaskRefs required")
        if type(trials_per_task) is not int or trials_per_task < 1 or type(seed) is not int or seed < 0:
            raise ValueError("positive trial budget and non-negative integer seed required")
        protocol_hash = _protocol_digest(protocol)
        manager = BankManager(success_threshold=self.success_threshold,
                              failure_threshold=self.failure_threshold)
        records = []
        observed_at = datetime.now(timezone.utc).isoformat()
        for task in tasks:
            trials = []
            for index in range(trials_per_task):
                trial_seed = _trial_seed(seed, task.task_ref, index)
                result = self.evaluator.evaluate(task.task_ref, checkpoint, trial_seed, protocol_hash)
                if (not isinstance(result, ObservedTrial) or result.task_ref != task.task_ref
                        or result.checkpoint != checkpoint or result.seed != trial_seed
                        or result.protocol_digest != protocol_hash):
                    raise ValueError("bootstrap evaluator returned mismatched task/checkpoint/seed/protocol")
                if self.evidence_verifier.verify_trial(result, task) is not True:
                    raise ValueError("bootstrap trial evidence was not verified")
                trials.append(result)
            record = TaskEvidenceRecord(task.task_ref, task.identity,
                checkpoint, protocol_hash, tuple(trials), observed_at)
            manager.observe(record, event_id=f"bootstrap-{len(records):06d}-{task.task_ref}")
            records.append(record)
        return BootstrapResult(checkpoint, protocol_hash, seed, trials_per_task,
                               tuple(records), manager)
