"""Seeded random selection and freezing of independent before-update probes."""
from hashlib import sha256
import random

from rift.banks.records import CapabilityState, EvidenceLevel, TaskEvidenceRecord
from rift.schemas import DataPurpose
from rift.probes.records import ProbeSnapshot, TaskProbe


class ProbeSelector:
    def __init__(self, *, success_threshold=0.75, failure_threshold=0.25,
                 min_valid_fraction=0.8, allow_nonreal=False, record_verifier=None):
        if not 0 <= failure_threshold < success_threshold <= 1:
            raise ValueError("probe classification thresholds must be ordered")
        if not 0 < min_valid_fraction <= 1:
            raise ValueError("min_valid_fraction must be in (0, 1]")
        self.success_threshold, self.failure_threshold = success_threshold, failure_threshold
        self.min_valid_fraction, self.allow_nonreal = min_valid_fraction, allow_nonreal
        self.record_verifier = record_verifier
        if not callable(getattr(record_verifier, "verify_record", None)):
            raise TypeError("trusted baseline record verifier required")

    def freeze(self, *, recovery_candidates, retention_candidates,
               diagnosis_identities, train_package_identities, split_registry,
               before_checkpoint, recovery_tasks, retention_tasks, seed,
               protocol_ref, protocol_digest, frozen_at_ns=0):
        if type(seed) is not int or seed < 0:
            raise ValueError("non-negative integer sampling seed required")
        if type(recovery_tasks) is not int or recovery_tasks < 1 or type(retention_tasks) is not int or retention_tasks < 1:
            raise ValueError("positive recovery and retention task counts required")
        if not protocol_ref or not protocol_digest or not split_registry.policy_version:
            raise ValueError("protocol and versioned SplitRegistry required")
        recovery, recovery_reasons = self._eligible(recovery_candidates,
            DataPurpose.RECOVERY_FEEDBACK, before_checkpoint, protocol_digest,
            split_registry, train_package_identities, diagnosis_identities)
        retention, retention_reasons = self._eligible(retention_candidates,
            DataPurpose.RETENTION_FEEDBACK, before_checkpoint, protocol_digest,
            split_registry, train_package_identities, tuple(diagnosis_identities) + tuple(r.identity for r in recovery))
        rng = random.Random(seed)
        recovery_chosen = rng.sample(sorted(recovery, key=lambda r: r.task_ref), min(recovery_tasks, len(recovery)))
        retention_chosen = rng.sample(sorted(retention, key=lambda r: r.task_ref), min(retention_tasks, len(retention)))
        if len(recovery_chosen) != recovery_tasks or len(retention_chosen) != retention_tasks:
            raise ValueError("insufficient eligible independent candidates for frozen Probe snapshots")
        recovery_probes = tuple(self._as_probe(row, protocol_ref, protocol_digest) for row in recovery_chosen)
        retention_probes = tuple(self._as_probe(row, protocol_ref, protocol_digest) for row in retention_chosen)
        recovery_id = self._snapshot_id("recovery", seed, recovery_probes, protocol_digest,
                                        split_registry.policy_version)
        retention_id = self._snapshot_id("retention", seed, retention_probes, protocol_digest,
                                         split_registry.policy_version)
        recovery_snapshot = ProbeSnapshot.create(snapshot_id=recovery_id,
            purpose=DataPurpose.RECOVERY_FEEDBACK, probes=recovery_probes, random_seed=seed,
            split_policy_version=split_registry.policy_version, baseline_checkpoint=before_checkpoint,
            protocol_digest=protocol_digest, candidate_count=len(recovery),
            exclusions=tuple(sorted(recovery_reasons.items())), frozen_at_ns=frozen_at_ns)
        retention_snapshot = ProbeSnapshot.create(snapshot_id=retention_id,
            purpose=DataPurpose.RETENTION_FEEDBACK, probes=retention_probes, random_seed=seed,
            split_policy_version=split_registry.policy_version, baseline_checkpoint=before_checkpoint,
            protocol_digest=protocol_digest, candidate_count=len(retention),
            exclusions=tuple(sorted(retention_reasons.items())), frozen_at_ns=frozen_at_ns)
        self._validate_pair(recovery_snapshot, retention_snapshot, split_registry.policy_version)
        return recovery_snapshot, retention_snapshot

    def _eligible(self, candidates, purpose, checkpoint, protocol_digest,
                  registry, training, diagnosis):
        rows, rejected = [], {}
        def reject(reason): rejected[reason] = rejected.get(reason, 0) + 1
        for row in candidates:
            if not isinstance(row, TaskEvidenceRecord):
                reject("not_task_evidence"); continue
            if row.purpose is not purpose or row.identity.purpose is not purpose:
                reject("wrong_purpose"); continue
            state = row.state(success_threshold=self.success_threshold,
                              failure_threshold=self.failure_threshold)
            target = CapabilityState.ACTIVE_WEAKNESS if purpose is DataPurpose.RECOVERY_FEEDBACK else CapabilityState.MASTERED_CANDIDATE
            if state is not target:
                reject("outside_capability_threshold"); continue
            if row.checkpoint != checkpoint:
                reject("baseline_checkpoint_mismatch"); continue
            if row.protocol_digest != protocol_digest:
                reject("evaluation_protocol_mismatch"); continue
            if not registry.contains(row.identity):
                reject("identity_not_registered"); continue
            if self.record_verifier.verify_record(row) is not True:
                reject("baseline_evidence_unverified"); continue
            if any(trial.valid and trial.evidence_level is not EvidenceLevel.REAL_SERVICE
                   for trial in row.trials):
                if not self.allow_nonreal:
                    reject("baseline_not_real_or_complete"); continue
            total = len(row.trials)
            if total == 0 or row.valid_trials / total < self.min_valid_fraction:
                reject("baseline_valid_fraction"); continue
            if any(row.identity.overlaps_under(other, registry.policy_version)
                   for other in tuple(training) + tuple(diagnosis)):
                reject("training_or_diagnosis_overlap"); continue
            rows.append(row)
        return rows, rejected

    @staticmethod
    def _as_probe(record, protocol_ref, protocol_digest):
        return TaskProbe(record.task_ref, record.identity, record.purpose,
            record.checkpoint, record.trials, protocol_ref, protocol_digest)

    @staticmethod
    def _snapshot_id(label, seed, probes, protocol_digest, policy):
        raw = "|".join((label, str(seed), protocol_digest, policy,
                         *(p.task_ref for p in probes)))
        return sha256(raw.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _validate_pair(recovery, retention, policy_version):
        if recovery.baseline_checkpoint != retention.baseline_checkpoint:
            raise ValueError("Recovery and Retention snapshots require one Service baseline")
        if recovery.split_policy_version != retention.split_policy_version:
            raise ValueError("Probe split-policy versions differ")
        if recovery.protocol_digest != retention.protocol_digest:
            raise ValueError("Recovery and Retention evaluation protocols differ")
        for left in recovery.probes:
            for right in retention.probes:
                if left.identity.overlaps_under(right.identity, policy_version):
                    raise ValueError("Recovery and Retention Probe identities overlap")
