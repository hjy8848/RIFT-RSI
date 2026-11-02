"""Paired, frozen-probe comparison of two updates from one checkpoint."""
from dataclasses import asdict, dataclass
import argparse
import importlib
import json
from math import isfinite

from rift.rewards.learning_value import score_dual_bank
from rift.schemas import DataPurpose, Identity, Split
from rift.training.contracts import CompletedUpdate, UpdateRequest


@dataclass(frozen=True)
class UpdateBudget:
    mini_batch_size: int
    epochs: int
    learning_rate: float

    def __post_init__(self):
        if (type(self.mini_batch_size) is not int or self.mini_batch_size < 1
                or type(self.epochs) is not int or self.epochs < 1
                or isinstance(self.learning_rate, bool)
                or not isinstance(self.learning_rate, (int, float))
                or not isfinite(self.learning_rate) or self.learning_rate <= 0):
            raise ValueError('positive mini-batch, epochs and learning rate required')

    def optimizer_steps(self, request):
        if len(request.rollouts) % self.mini_batch_size:
            raise ValueError('rollout count must divide into complete mini-batches')
        return len(request.rollouts) // self.mini_batch_size * self.epochs


@dataclass(frozen=True)
class MatchedArmResult:
    update_run_id: str
    after_revision: str
    gain: float
    retention_delta: float
    reward: float


@dataclass(frozen=True)
class MatchedControlResult:
    recovery_snapshot_digest: str
    retention_snapshot_digest: str
    starting_weights_digest: str
    rollout_count: int
    optimizer_steps: int
    budget: UpdateBudget
    treatment: MatchedArmResult
    control: MatchedArmResult
    gain_difference: float
    retention_difference: float
    reward_difference: float


def run_matched_control(treatment_request, control_request, *, recovery_snapshot,
        retention_snapshot, treatment_budget, control_budget, backend_factory,
        probe_runner, update_verifier, checkpoint_store, snapshot_verifier,
        trial_verifier, training_identity_resolver, lambda_retention=0.5,
        min_valid_fraction=0.8):
    """Train two independent arms and score both against exactly the same probes.

    The deployment-owned backend factory takes ``(arm_name, request, budget)``
    and must produce a fresh checkpoint-bound backend for each arm. The normal
    update verifier and frozen-probe scorers validate both results.
    """
    requests = (treatment_request, control_request)
    if any(not isinstance(row, UpdateRequest) or row.algorithm.lower() != 'grpo' for row in requests):
        raise ValueError('two Service GRPO requests are required')
    if (treatment_request.policy_before != control_request.policy_before
            or treatment_request.policy_before.policy.lower() != 'service'
            or treatment_request.package_id == control_request.package_id):
        raise ValueError('distinct arm packages must start from the same Service checkpoint')
    if (not isinstance(treatment_budget, UpdateBudget) or treatment_budget != control_budget
            or len(treatment_request.rollouts) != len(control_request.rollouts)):
        raise ValueError('matched arms must use identical rollout and update budgets')
    steps = treatment_budget.optimizer_steps(treatment_request)
    if control_budget.optimizer_steps(control_request) != steps:
        raise ValueError('matched arms must execute the same number of optimizer steps')
    verify_budget = getattr(update_verifier, 'verify_update_budget', None)
    if not callable(verify_budget):
        raise ValueError('matched updates require a trusted verifier for their recorded budgets')
    if (not recovery_snapshot.verify() or not retention_snapshot.verify()
            or recovery_snapshot.purpose is not DataPurpose.RECOVERY_FEEDBACK
            or retention_snapshot.purpose is not DataPurpose.RETENTION_FEEDBACK
            or recovery_snapshot.baseline_checkpoint != treatment_request.policy_before
            or retention_snapshot.baseline_checkpoint != treatment_request.policy_before
            or recovery_snapshot.protocol_digest != retention_snapshot.protocol_digest
            or recovery_snapshot.split_policy_version != retention_snapshot.split_policy_version):
        raise ValueError('both frozen probes must share the starting checkpoint and protocol')
    identities_by_arm = []
    for request in requests:
        identities = tuple(training_identity_resolver(request))
        if (not identities or any(not isinstance(row, Identity) or row.split is not Split.TRAIN
                                  for row in identities)
                or {row.task for row in identities} != {row.task_id for row in request.rollouts}):
            raise ValueError('each arm requires its exact registered training identities')
        for snapshot in (recovery_snapshot, retention_snapshot):
            if any(probe.identity.overlaps_under(identity, snapshot.split_policy_version)
                   for probe in snapshot.probes for identity in identities):
                raise ValueError('training identities overlap a frozen Probe')
        identities_by_arm.append(identities)
    before_recovery = probe_runner.baseline(recovery_snapshot)
    before_retention = probe_runner.baseline(retention_snapshot)
    backends = tuple(backend_factory(name, request, treatment_budget)
                     for name, request in zip(('treatment', 'control'), requests))
    if backends[0] is backends[1]:
        raise ValueError('matched arms require separate update backends')
    for backend, request in zip(backends, requests):
        if (not callable(getattr(backend, 'train', None))
                or not callable(getattr(backend, 'budget_for', None))
                or backend.budget_for(request) != treatment_budget):
            raise ValueError('backend must attest the actual matched update budget')
    paths = tuple(getattr(backend, 'checkpoint_root', None) for backend in backends)
    if paths[0] is not None and paths[1] is not None and paths[0] == paths[1]:
        raise ValueError('matched arms require separate checkpoint output roots')
    results = []
    for request, identities, backend in zip(requests, identities_by_arm, backends):
        update = backend.train(request)
        if (not isinstance(update, CompletedUpdate) or update.request != request
                or update_verifier.verify_update(update) is not True
                or verify_budget(update, treatment_budget) is not True):
            raise ValueError('arm update is not bound to its verified starting request')
        recovery_after, retention_after = probe_runner.after_many(
            (recovery_snapshot, retention_snapshot), update.policy_after)
        score = score_dual_bank(update, recovery_snapshot, retention_snapshot,
            before_recovery, recovery_after, before_retention, retention_after,
            update_verifier=update_verifier, checkpoint_store=checkpoint_store,
            snapshot_verifier=snapshot_verifier, trial_verifier=trial_verifier,
            training_identities=identities, lambda_retention=lambda_retention,
            min_valid_fraction=min_valid_fraction)
        results.append(MatchedArmResult(update.run_id, update.policy_after.revision,
            score.gain, score.retention_delta, score.total))
    treatment, control = results
    return MatchedControlResult(recovery_snapshot.content_digest, retention_snapshot.content_digest,
        treatment_request.policy_before.weights_digest, len(treatment_request.rollouts), steps,
        treatment_budget, treatment, control, treatment.gain - control.gain,
        treatment.retention_delta - control.retention_delta, treatment.reward - control.reward)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run a matched-control evaluation with deployment bindings')
    parser.add_argument('--factory', required=True, help='module:function returning run_matched_control kwargs')
    parser.add_argument('--output', required=True, help='new JSON result path')
    args = parser.parse_args(argv)
    from pathlib import Path
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(f'matched-control output already exists: {path}')
    module, separator, name = args.factory.partition(':')
    if not separator or not module or not name:
        parser.error('--factory must be module:function')
    factory = getattr(importlib.import_module(module), name)
    result = run_matched_control(**factory())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(asdict(result), stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
