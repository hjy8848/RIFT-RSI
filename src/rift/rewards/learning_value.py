"""Verified dual-bank task-success reward; independent from Replay LogP scoring."""
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from math import isfinite, isclose

from rift.banks.records import EvidenceLevel, ObservedTrial
from rift.probes.records import ProbeResultSet, ProbeSnapshot, ProbeTaskResult
from rift.schemas import Checkpoint, DataPurpose, Identity, Outcome, Split
from rift.training.contracts import CompletedUpdate


class RewardMode(str, Enum):
    DUAL_BANK_OUTCOME_V1 = "dual_bank_outcome_v1"
    REPLAY_LOGP_V1 = "replay_logp_v1"


class RewardStatus(str, Enum):
    COMPLETE = "COMPLETE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class InsufficientEvidenceError(ValueError):
    status = RewardStatus.INSUFFICIENT_EVIDENCE


@dataclass(frozen=True)
class DualBankOutcomeReward:
    package_id: str
    service_update_run_id: str
    service_before: Checkpoint
    service_after: Checkpoint
    recovery_snapshot_digest: str
    retention_snapshot_digest: str
    recovery_before_evidence: str
    recovery_after_evidence: str
    retention_before_evidence: str
    retention_after_evidence: str
    recovery_before_sr: float
    recovery_after_sr: float
    retention_before_sr: float
    retention_after_sr: float
    gain: float
    retention_delta: float
    lambda_retention: float
    total: float
    recovery_valid_fraction: float
    retention_valid_fraction: float
    min_valid_fraction: float
    expected_trials_per_probe: int
    split_policy_version: str
    protocol_digest: str
    credit_assignment_mode: str = "sequential_curriculum"
    reward_protocol: str = RewardMode.DUAL_BANK_OUTCOME_V1.value

    def __post_init__(self):
        refs = (self.package_id, self.service_update_run_id, self.recovery_snapshot_digest,
            self.retention_snapshot_digest, self.recovery_before_evidence,
            self.recovery_after_evidence, self.retention_before_evidence,
            self.retention_after_evidence, self.split_policy_version, self.protocol_digest)
        values = (self.recovery_before_sr, self.recovery_after_sr, self.retention_before_sr,
            self.retention_after_sr, self.gain, self.retention_delta, self.lambda_retention,
            self.total, self.recovery_valid_fraction, self.retention_valid_fraction,
            self.min_valid_fraction)
        if (not all(refs) or self.reward_protocol != RewardMode.DUAL_BANK_OUTCOME_V1.value
                or self.credit_assignment_mode != "sequential_curriculum"
                or any(not isfinite(x) for x in values)
                or any(x < 0 or x > 1 for x in (self.recovery_before_sr,
                    self.recovery_after_sr, self.retention_before_sr, self.retention_after_sr,
                    self.recovery_valid_fraction, self.retention_valid_fraction,
                    self.min_valid_fraction))
                or self.lambda_retention < 0 or type(self.expected_trials_per_probe) is not int
                or self.expected_trials_per_probe < 1):
            raise ValueError("invalid or incomplete dual-bank reward evidence")
        gain = self.recovery_after_sr - self.recovery_before_sr
        retention = self.retention_after_sr - self.retention_before_sr
        total = gain - self.lambda_retention * max(0.0, -retention)
        if (not isclose(self.gain, gain, abs_tol=1e-12)
                or not isclose(self.retention_delta, retention, abs_tol=1e-12)
                or not isclose(self.total, total, abs_tol=1e-12)):
            raise ValueError("dual-bank reward components do not match the declared formula")
        if (self.recovery_valid_fraction < self.min_valid_fraction
                or self.retention_valid_fraction < self.min_valid_fraction):
            raise ValueError("insufficient Probe evidence cannot receive a scalar reward")

    @property
    def value(self):
        return self.total

    @property
    def scalar(self):
        return self.total

    @property
    def reward_source_digest(self):
        payload = {key: getattr(self, key) for key in self.__dataclass_fields__}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return sha256(encoded.encode("utf-8")).hexdigest()


def score_dual_bank(update, recovery_snapshot, retention_snapshot,
                    recovery_before, recovery_after, retention_before, retention_after,
                    *, update_verifier, checkpoint_store, snapshot_verifier,
                    trial_verifier, training_identities, lambda_retention=0.5,
                    min_valid_fraction=0.8, credit_assignment_mode="sequential_curriculum"):
    if credit_assignment_mode != "sequential_curriculum":
        raise ValueError("dual-bank v1 implements sequential credit only; matched-baseline is not enabled")
    if not isinstance(update, CompletedUpdate):
        raise ValueError("verified CompletedUpdate required")
    if update_verifier.verify_update(update) is not True:
        raise ValueError("Service update evidence was not verified")
    training_identities = tuple(training_identities)
    if (not training_identities or any(not isinstance(i, Identity) or i.split is not Split.TRAIN
                                       for i in training_identities)):
        raise ValueError("registered Service training identities are required for Probe isolation")
    if {i.task for i in training_identities} != {row.task_id for row in update.request.rollouts}:
        raise ValueError("Service training identity manifest differs from the actual update request")
    if not checkpoint_store.verify(update.request.policy_before) or not checkpoint_store.verify(update.policy_after):
        raise ValueError("Service before/after checkpoint evidence was not verified")
    snapshots = (recovery_snapshot, retention_snapshot)
    if any(not isinstance(row, ProbeSnapshot) or not row.verify() for row in snapshots):
        raise ValueError("frozen Probe snapshot verification failed")
    if (recovery_snapshot.purpose is not DataPurpose.RECOVERY_FEEDBACK
            or retention_snapshot.purpose is not DataPurpose.RETENTION_FEEDBACK
            or recovery_snapshot.baseline_checkpoint != update.request.policy_before
            or retention_snapshot.baseline_checkpoint != update.request.policy_before
            or recovery_snapshot.split_policy_version != retention_snapshot.split_policy_version
            or recovery_snapshot.protocol_digest != retention_snapshot.protocol_digest):
        raise ValueError("Probe purpose, baseline, split policy or protocol mismatch")
    for snapshot in snapshots:
        if snapshot_verifier.verify_frozen_before(snapshot, update) is not True:
            raise ValueError("Probe snapshot was not verified as frozen before Service update")
        for probe in snapshot.probes:
            if any(probe.identity.overlaps_under(identity, snapshot.split_policy_version)
                   for identity in training_identities):
                raise ValueError("frozen Probe overlaps a Service training task")
    if _expected_trials(recovery_snapshot) != _expected_trials(retention_snapshot):
        raise ValueError("Recovery and Retention Probe budgets must match")
    _verify_independent(recovery_snapshot, retention_snapshot)
    pairs = ((recovery_snapshot, recovery_before, recovery_after),
             (retention_snapshot, retention_before, retention_after))
    for snapshot, before, after in pairs:
        _validate_pair(snapshot, before, after, update, trial_verifier)
    if lambda_retention < 0 or not isfinite(lambda_retention):
        raise ValueError("lambda_retention must be finite and non-negative")
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in (0, 1]")
    for label, result_set in (("Recovery before", recovery_before),
            ("Recovery after", recovery_after), ("Retention before", retention_before),
            ("Retention after", retention_after)):
        if (result_set.valid_fraction < min_valid_fraction
                or any(row.success_rate is None for row in result_set.results)):
            raise InsufficientEvidenceError(f"{label} Probe valid fraction is below the configured floor")
    recovery_before_sr = _weighted_success_rate(recovery_before, recovery_snapshot)
    recovery_after_sr = _weighted_success_rate(recovery_after, recovery_snapshot)
    retention_before_sr = _weighted_success_rate(retention_before, retention_snapshot)
    retention_after_sr = _weighted_success_rate(retention_after, retention_snapshot)
    gain, retention_delta, total = dual_bank_formula(recovery_before_sr,
        recovery_after_sr, retention_before_sr, retention_after_sr, lambda_retention)
    refs = (recovery_before.evidence_digest, recovery_after.evidence_digest,
            retention_before.evidence_digest, retention_after.evidence_digest)
    return DualBankOutcomeReward(update.request.package_id, update.run_id,
        update.request.policy_before, update.policy_after,
        recovery_snapshot.content_digest, retention_snapshot.content_digest,
        *refs, recovery_before_sr, recovery_after_sr, retention_before_sr,
        retention_after_sr, gain, retention_delta, lambda_retention, total,
        min(recovery_before.valid_fraction, recovery_after.valid_fraction),
        min(retention_before.valid_fraction, retention_after.valid_fraction),
        min_valid_fraction, _expected_trials(recovery_snapshot),
        recovery_snapshot.split_policy_version, recovery_snapshot.protocol_digest,
        credit_assignment_mode)


def _verify_independent(recovery, retention):
    for left in recovery.probes:
        for right in retention.probes:
            if left.identity.overlaps_under(right.identity, recovery.split_policy_version):
                raise ValueError("Recovery and Retention snapshots share a protected identity")


def _expected_trials(snapshot):
    sizes = {len(probe.baseline_trials) for probe in snapshot.probes}
    if len(sizes) != 1:
        raise ValueError("all probes in a snapshot require the same rollout budget")
    return next(iter(sizes))


def _validate_pair(snapshot, before, after, update, trial_verifier):
    if (not isinstance(before, ProbeResultSet) or not isinstance(after, ProbeResultSet)
            or before.phase != "before" or after.phase != "after"
            or before.snapshot_digest != snapshot.content_digest
            or after.snapshot_digest != snapshot.content_digest
            or before.purpose != snapshot.purpose or after.purpose != snapshot.purpose
            or before.checkpoint != update.request.policy_before
            or after.checkpoint != update.policy_after
            or before.protocol_digest != snapshot.protocol_digest
            or after.protocol_digest != snapshot.protocol_digest):
        raise ValueError("before/after Probe result binding mismatch")
    probes = {probe.task_ref: probe for probe in snapshot.probes}
    if set(probes) != {row.task_ref for row in before.results} or set(probes) != {row.task_ref for row in after.results}:
        raise ValueError("Probe task coverage differs from its frozen snapshot")
    if len({len(row.trials) for row in before.results + after.results}) != 1:
        raise ValueError("Probe trial budget differs between tasks")
    for result in before.results + after.results:
        probe = probes[result.task_ref]
        if result.identity != probe.identity or len(result.trials) != len(probe.baseline_trials):
            raise ValueError("Probe identity or rollout count differs from its frozen member")
        expected_seeds = set(probe.seeds)
        if {trial.seed for trial in result.trials} != expected_seeds:
            raise ValueError("before/after Probe seeds differ from frozen protocol")
        if result.phase == "before" and result.trials != probe.baseline_trials:
            raise ValueError("before baseline differs from frozen, pre-update evidence")
        for trial in result.trials:
            if trial.evidence_level is not EvidenceLevel.REAL_SERVICE:
                raise ValueError("Mock or scripted results cannot produce a real learning reward")
            if trial.outcome is not Outcome.UNKNOWN and (not trial.replay_verified or not trial.evidence_ref):
                raise ValueError("known business outcome lacks verified Episode evidence")
            if trial_verifier.verify_probe_trial(trial, probe, result.checkpoint) is not True:
                raise ValueError("Probe Episode evidence failed trusted verification")


def _weighted_success_rate(result_set, snapshot):
    probes = {probe.task_ref: probe for probe in snapshot.probes}
    rows = [(row.success_rate, probes[row.task_ref].weight) for row in result_set.results]
    if not rows or any(rate is None for rate, _ in rows):
        raise ValueError("Probe contains a task with no valid outcomes")
    total_weight = sum(weight for _, weight in rows)
    return sum(rate * weight for rate, weight in rows) / total_weight


class DualBankRewardVerifier:
    """Re-load and recompute reward inputs from the trusted Host evidence store."""
    def __init__(self, *, evidence_store, update_verifier, checkpoint_store,
                 snapshot_verifier, trial_verifier, training_identity_resolver,
                 lambda_retention=0.5, min_valid_fraction=0.8):
        self.evidence_store = evidence_store
        self.update_verifier, self.checkpoint_store = update_verifier, checkpoint_store
        self.snapshot_verifier, self.trial_verifier = snapshot_verifier, trial_verifier
        self.training_identity_resolver = training_identity_resolver
        self.lambda_retention, self.min_valid_fraction = lambda_retention, min_valid_fraction

    def verify_reward(self, reward, update):
        if not isinstance(reward, DualBankOutcomeReward) or not isinstance(update, CompletedUpdate):
            return False
        try:
            recovery = self.evidence_store.load_snapshot(reward.recovery_snapshot_digest)
            retention = self.evidence_store.load_snapshot(reward.retention_snapshot_digest)
            recovery_before = self.evidence_store.load_results(reward.recovery_before_evidence)
            recovery_after = self.evidence_store.load_results(reward.recovery_after_evidence)
            retention_before = self.evidence_store.load_results(reward.retention_before_evidence)
            retention_after = self.evidence_store.load_results(reward.retention_after_evidence)
            recomputed = score_dual_bank(update, recovery, retention,
                recovery_before, recovery_after, retention_before, retention_after,
                update_verifier=self.update_verifier, checkpoint_store=self.checkpoint_store,
                snapshot_verifier=self.snapshot_verifier, trial_verifier=self.trial_verifier,
                training_identities=self.training_identity_resolver(update),
                lambda_retention=self.lambda_retention,
                min_valid_fraction=self.min_valid_fraction,
                credit_assignment_mode=reward.credit_assignment_mode)
            return recomputed == reward
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            # Do not expose Host-private task or Oracle diagnostics through this boundary.
            return False


def dual_bank_formula(recovery_before_sr, recovery_after_sr,
                      retention_before_sr, retention_after_sr,
                      lambda_retention):
    """Pure arithmetic used by the evidence-backed scorer and offline tests."""
    values = (recovery_before_sr, recovery_after_sr, retention_before_sr,
              retention_after_sr, lambda_retention)
    if any(not isfinite(value) for value in values) or lambda_retention < 0:
        raise ValueError("finite success rates and non-negative retention weight required")
    if any(value < 0 or value > 1 for value in values[:4]):
        raise ValueError("success-rate values must be in [0, 1]")
    gain = recovery_after_sr - recovery_before_sr
    retention_delta = retention_after_sr - retention_before_sr
    total = gain - lambda_retention * max(0.0, -retention_delta)
    return gain, retention_delta, total
