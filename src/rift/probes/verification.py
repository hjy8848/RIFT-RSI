"""Host-side verifier combining immutable storage, Episode attestation and ordering."""
from rift.probes.records import ProbeResultSet, ProbeSnapshot


class StoredProbeEvidenceVerifier:
    def __init__(self, evidence_store, trial_verifier, freeze_order_verifier):
        self.evidence_store = evidence_store
        self.trial_verifier = trial_verifier
        self.freeze_order_verifier = freeze_order_verifier

    def verify_snapshot(self, snapshot):
        if not isinstance(snapshot, ProbeSnapshot) or not snapshot.verify():
            return False
        try:
            return self.evidence_store.load_snapshot(snapshot.content_digest) == snapshot
        except (OSError, ValueError, TypeError, KeyError):
            return False

    def verify_frozen_before(self, snapshot, update):
        if not self.verify_snapshot(snapshot):
            return False
        return self.freeze_order_verifier.verify_frozen_before(snapshot, update) is True

    def verify_result(self, result, snapshot):
        if (not isinstance(result, ProbeResultSet) or not self.verify_snapshot(snapshot)
                or result.snapshot_digest != snapshot.content_digest
                or result.purpose is not snapshot.purpose
                or result.protocol_digest != snapshot.protocol_digest):
            return False
        try:
            if self.evidence_store.load_results(result.evidence_digest) != result:
                return False
        except (OSError, ValueError, TypeError, KeyError):
            return False
        probes = {row.task_ref: row for row in snapshot.probes}
        if set(probes) != {row.task_ref for row in result.results}:
            return False
        return all(row.identity == probes[row.task_ref].identity
            and all(self.trial_verifier.verify_probe_trial(trial, probes[row.task_ref], result.checkpoint) is True
                    for trial in row.trials)
            for row in result.results)
