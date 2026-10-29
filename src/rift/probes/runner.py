"""Task-level before/after evaluation using an injected real Usami Service runner."""
from typing import Protocol

from rift.banks.records import ObservedTrial
from rift.probes.records import ProbeTaskResult, make_result_set


class ProbeEvaluator(Protocol):
    def evaluate(self, task_ref: str, checkpoint, seed: int,
                 protocol_ref: str, protocol_digest: str) -> ObservedTrial: ...


class ProbeTrialVerifier(Protocol):
    def verify_probe_trial(self, trial: ObservedTrial, probe, checkpoint) -> bool: ...


class ProbeRunner:
    """No model is loaded here; the Host injects checkpoint-aware Usami execution."""
    def __init__(self, evaluator, trial_verifier):
        self.evaluator, self.trial_verifier = evaluator, trial_verifier
        if not callable(getattr(evaluator, "evaluate", None)):
            raise TypeError("ProbeEvaluator required")
        if not callable(getattr(trial_verifier, "verify_probe_trial", None)):
            raise TypeError("ProbeTrialVerifier required")

    def baseline(self, snapshot):
        if not snapshot.verify():
            raise ValueError("frozen Probe snapshot failed verification")
        results = []
        for probe in snapshot.probes:
            if any(self.trial_verifier.verify_probe_trial(t, probe, snapshot.baseline_checkpoint) is not True
                   for t in probe.baseline_trials):
                raise ValueError("cached before-update Probe evidence failed verification")
            results.append(ProbeTaskResult(probe.task_ref, probe.identity,
                snapshot.baseline_checkpoint, probe.baseline_trials,
                snapshot.protocol_digest, "before"))
        return make_result_set(snapshot, snapshot.baseline_checkpoint, "before", results)

    def after(self, snapshot, checkpoint):
        return self.after_many((snapshot,), checkpoint)[0]

    def after_many(self, snapshots, checkpoint):
        """Evaluate frozen Recovery/Retention jobs in one VERL-compatible batch."""
        snapshots = tuple(snapshots)
        if not snapshots:
            raise ValueError("at least one frozen Probe snapshot is required")
        requests = []
        owners = []
        for snapshot_index, snapshot in enumerate(snapshots):
            if not snapshot.verify():
                raise ValueError("frozen Probe snapshot failed verification")
            if checkpoint == snapshot.baseline_checkpoint:
                raise ValueError("after evaluation requires the updated Service checkpoint")
            for probe in snapshot.probes:
                for seed in probe.seeds:
                    requests.append((probe.task_ref, checkpoint, seed,
                                     probe.protocol_ref, snapshot.protocol_digest))
                    owners.append((snapshot_index, probe))

        # Optional production extension: submit this sequence to one VERL
        # Agent Loop batch. Results must preserve request order.
        evaluate_many = getattr(self.evaluator, "evaluate_many", None)
        if callable(evaluate_many):
            trials = tuple(evaluate_many(tuple(requests)))
            if len(trials) != len(requests):
                raise ValueError("batched Probe evaluator returned an incomplete result batch")
        else:
            trials = tuple(self.evaluator.evaluate(*request) for request in requests)

        by_snapshot = [[] for _ in snapshots]
        by_probe = [{} for _ in snapshots]
        for request, (snapshot_index, probe), trial in zip(requests, owners, trials):
            task_ref, expected_checkpoint, seed, _, protocol_digest = request
            if (not isinstance(trial, ObservedTrial) or trial.task_ref != task_ref
                    or trial.checkpoint != expected_checkpoint or trial.seed != seed
                    or trial.protocol_digest != protocol_digest
                    or trial.public_summary is not None):
                raise ValueError("after evaluator returned a mismatched or public Probe record")
            if self.trial_verifier.verify_probe_trial(
                    trial, probe, expected_checkpoint) is not True:
                raise ValueError("after Probe trial evidence failed verification")
            by_probe[snapshot_index].setdefault(probe.task_ref, []).append(trial)

        for snapshot_index, snapshot in enumerate(snapshots):
            for probe in snapshot.probes:
                probe_trials = tuple(by_probe[snapshot_index].get(probe.task_ref, ()))
                if len(probe_trials) != len(probe.seeds):
                    raise ValueError("after evaluator omitted a frozen Probe seed")
                by_snapshot[snapshot_index].append(ProbeTaskResult(
                    probe.task_ref, probe.identity, checkpoint, probe_trials,
                    snapshot.protocol_digest, "after"))
        return tuple(make_result_set(snapshot, checkpoint, "after", rows)
                     for snapshot, rows in zip(snapshots, by_snapshot))
