"""Seeded, ability-prioritized Recovery and random Retention Probe freezing."""
from hashlib import sha256
import random
from dataclasses import replace

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
               protocol_ref, protocol_digest, frozen_at_ns=0,
               target_ability_id=None, ability_task_refs=(),
               current_baseline_candidates=()):
        if type(seed) is not int or seed < 0:
            raise ValueError("non-negative integer sampling seed required")
        if type(recovery_tasks) is not int or recovery_tasks < 1 or type(retention_tasks) is not int or retention_tasks < 1:
            raise ValueError("positive recovery and retention task counts required")
        if not protocol_ref or not protocol_digest or not split_registry.policy_version:
            raise ValueError("protocol and versioned SplitRegistry required")
        current_baselines = {}
        for row in current_baseline_candidates:
            if not isinstance(row, TaskEvidenceRecord):
                continue
            if row.task_ref in current_baselines:
                raise ValueError("multiple current baseline records for one task")
            current_baselines[row.task_ref] = row
        recovery, recovery_reasons = self._eligible(recovery_candidates,
            DataPurpose.RECOVERY_FEEDBACK, before_checkpoint, protocol_digest,
            split_registry, train_package_identities, diagnosis_identities,
            current_baselines)
        retention, retention_reasons = self._eligible(retention_candidates,
            DataPurpose.RETENTION_FEEDBACK, before_checkpoint, protocol_digest,
            split_registry, train_package_identities,
            tuple(diagnosis_identities) + tuple(r.identity for r in recovery),
            current_baselines)
        if target_ability_id is not None and not target_ability_id:
            raise ValueError("non-empty target ability ID required")
        ability_task_refs = frozenset(ability_task_refs)
        if ability_task_refs and target_ability_id is None:
            raise ValueError("ability-linked failure tasks require a target ability ID")
        rng = random.Random(seed)
        priority_candidates = [row for row in recovery if row.task_ref in ability_task_refs]
        fallback_candidates = [row for row in recovery if row.task_ref not in ability_task_refs]
        chosen_priority = rng.sample(sorted(priority_candidates, key=lambda r: r.task_ref),
                                     min(recovery_tasks, len(priority_candidates)))
        remaining = recovery_tasks - len(chosen_priority)
        chosen_fallback = rng.sample(sorted(fallback_candidates, key=lambda r: r.task_ref),
                                     min(remaining, len(fallback_candidates)))
        recovery_chosen = chosen_priority + chosen_fallback
        rng.shuffle(recovery_chosen)
        # Retention is sampled independently so target-ability prioritization
        # cannot change which preserved capabilities are tested.
        retention_rng = random.Random(seed ^ 0x5EED5EED)
        retention_chosen = retention_rng.sample(sorted(retention, key=lambda r: r.task_ref),
                                                min(retention_tasks, len(retention)))
        if len(recovery_chosen) != recovery_tasks or len(retention_chosen) != retention_tasks:
            raise ValueError("insufficient eligible independent candidates for frozen Probe snapshots")
        recovery_probes = tuple(self._as_probe(row, protocol_ref, protocol_digest) for row in recovery_chosen)
        retention_probes = tuple(self._as_probe(row, protocol_ref, protocol_digest) for row in retention_chosen)
        priority_refs = tuple(row.task_ref for row in chosen_priority)
        recovery_id = self._snapshot_id("recovery", seed, recovery_probes, protocol_digest,
                                        split_registry.policy_version, target_ability_id, priority_refs)
        retention_id = self._snapshot_id("retention", seed, retention_probes, protocol_digest,
                                         split_registry.policy_version)
        recovery_snapshot = ProbeSnapshot.create(snapshot_id=recovery_id,
            purpose=DataPurpose.RECOVERY_FEEDBACK, probes=recovery_probes, random_seed=seed,
            split_policy_version=split_registry.policy_version, baseline_checkpoint=before_checkpoint,
            protocol_digest=protocol_digest, candidate_count=len(recovery),
            exclusions=tuple(sorted(recovery_reasons.items())), frozen_at_ns=frozen_at_ns,
            target_ability_id=target_ability_id, priority_task_refs=priority_refs)
        retention_snapshot = ProbeSnapshot.create(snapshot_id=retention_id,
            purpose=DataPurpose.RETENTION_FEEDBACK, probes=retention_probes, random_seed=seed,
            split_policy_version=split_registry.policy_version, baseline_checkpoint=before_checkpoint,
            protocol_digest=protocol_digest, candidate_count=len(retention),
            exclusions=tuple(sorted(retention_reasons.items())), frozen_at_ns=frozen_at_ns)
        self._validate_pair(recovery_snapshot, retention_snapshot, split_registry.policy_version)
        return recovery_snapshot, retention_snapshot

    def _eligible(self, candidates, purpose, checkpoint, protocol_digest,
                  registry, training, diagnosis, current_baselines):
        rows, rejected, history_by_task = [], {}, {}
        def reject(reason): rejected[reason] = rejected.get(reason, 0) + 1
        for history in candidates:
            if not isinstance(history, TaskEvidenceRecord):
                reject("not_task_evidence"); continue
            if history.purpose is not purpose or history.identity.purpose is not purpose:
                reject("wrong_purpose"); continue
            state = history.state(success_threshold=self.success_threshold,
                              failure_threshold=self.failure_threshold)
            target = CapabilityState.ACTIVE_WEAKNESS if purpose is DataPurpose.RECOVERY_FEEDBACK else CapabilityState.MASTERED_CANDIDATE
            if state is not target:
                reject("outside_capability_threshold"); continue
            if not registry.contains(history.identity):
                reject("identity_not_registered"); continue
            if self.record_verifier.verify_record(history) is not True:
                reject("historical_evidence_unverified"); continue
            if any(trial.valid and trial.evidence_level is not EvidenceLevel.REAL_SERVICE
                   for trial in history.trials):
                if not self.allow_nonreal:
                    reject("historical_evidence_not_real"); continue
            total = len(history.trials)
            if total == 0 or history.valid_trials / total < self.min_valid_fraction:
                reject("historical_valid_fraction"); continue
            if any(history.identity.overlaps_under(other, registry.policy_version)
                   for other in tuple(training) + tuple(diagnosis)):
                reject("training_or_diagnosis_overlap"); continue

            baseline = current_baselines.get(history.task_ref)
            if baseline is None and history.checkpoint == checkpoint and history.protocol_digest == protocol_digest:
                baseline = history
            if baseline is None:
                reject("current_checkpoint_baseline_missing"); continue
            if (baseline.task_ref != history.task_ref or baseline.identity != history.identity
                    or baseline.purpose is not purpose or baseline.identity.purpose is not purpose):
                reject("current_baseline_task_identity_mismatch"); continue
            if baseline.checkpoint != checkpoint:
                reject("current_baseline_checkpoint_mismatch"); continue
            if baseline.protocol_digest != protocol_digest:
                reject("evaluation_protocol_mismatch"); continue
            if self.record_verifier.verify_record(baseline) is not True:
                reject("current_baseline_evidence_unverified"); continue
            if any(trial.valid and trial.evidence_level is not EvidenceLevel.REAL_SERVICE
                   for trial in baseline.trials):
                if not self.allow_nonreal:
                    reject("current_baseline_not_real"); continue
            baseline_total = len(baseline.trials)
            if baseline_total == 0 or baseline.valid_trials / baseline_total < self.min_valid_fraction:
                reject("current_baseline_valid_fraction"); continue
            previous = history_by_task.get(history.task_ref)
            if previous is None or history.observed_at > previous[0]:
                history_by_task[history.task_ref] = (history.observed_at, baseline)

        rows.extend(baseline for _, baseline in history_by_task.values())
        return rows, rejected

    @staticmethod
    def _as_probe(record, protocol_ref, protocol_digest):
        # Hidden Recovery/Retention observations may be retained Host-side to
        # support ability attribution. Frozen evaluation snapshots contain no
        # Designer-visible summaries.
        trials = tuple(replace(trial, public_summary=None) for trial in record.trials)
        return TaskProbe(record.task_ref, record.identity, record.purpose,
            record.checkpoint, trials, protocol_ref, protocol_digest)

    @staticmethod
    def _snapshot_id(label, seed, probes, protocol_digest, policy,
                     target_ability_id=None, priority_task_refs=()):
        parts = (label, str(seed), protocol_digest, policy)
        if target_ability_id is not None or priority_task_refs:
            parts += (target_ability_id or "", *priority_task_refs)
        raw = "|".join(parts + tuple(p.task_ref for p in probes))
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
