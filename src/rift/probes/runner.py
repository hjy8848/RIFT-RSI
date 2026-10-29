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
        if not snapshot.verify():
            raise ValueError("frozen Probe snapshot failed verification")
        if checkpoint == snapshot.baseline_checkpoint:
            raise ValueError("after evaluation requires the updated Service checkpoint")
        results = []
        for probe in snapshot.probes:
            trials = []
            for seed in probe.seeds:
                trial = self.evaluator.evaluate(probe.task_ref, checkpoint, seed,
                                                probe.protocol_ref, snapshot.protocol_digest)
                if (not isinstance(trial, ObservedTrial) or trial.task_ref != probe.task_ref
                        or trial.checkpoint != checkpoint or trial.seed != seed
                        or trial.protocol_digest != snapshot.protocol_digest
                        or trial.public_summary is not None):
                    raise ValueError("after evaluator returned a mismatched or public Probe record")
                if self.trial_verifier.verify_probe_trial(trial, probe, checkpoint) is not True:
                    raise ValueError("after Probe trial evidence failed verification")
                trials.append(trial)
            results.append(ProbeTaskResult(probe.task_ref, probe.identity,
                checkpoint, tuple(trials), snapshot.protocol_digest, "after"))
        return make_result_set(snapshot, checkpoint, "after", results)
