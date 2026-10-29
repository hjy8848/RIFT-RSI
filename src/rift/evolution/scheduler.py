"""Multi-round dual-bank scheduler over injected Designer, Usami and VERL runtimes."""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Callable, Protocol

from rift.abilities.registry import AbilityRegistry
from rift.abilities.records import AbilityPrompt
from rift.banks.bootstrap import BootstrapResult, BootstrapRunner
from rift.banks.core import BankManager, JsonTaskEvidenceStore, public_projection
from rift.banks.records import FailureObservation, TaskEvidenceRecord
from rift.banks.records import CapabilityState
from rift.loop import LoopState, Stage
from rift.probes.records import ProbeSnapshot
from rift.rewards.learning_value import RewardMode
from rift.schemas import Checkpoint
from rift.training.contracts import UpdateRequest
from rift.training.records import DesignerFeedback
from rift.tasks import Proposal
from rift.verl.coordinator import prepare_dual_bank_round, run_round


@dataclass(frozen=True)
class EvolutionRoundPlan:
    """Prepared actions before training; callbacks hold application-owned trust roots."""
    state: LoopState
    service_request: UpdateRequest
    recovery_snapshot: ProbeSnapshot
    retention_snapshot: ProbeSnapshot
    service_backend: object
    designer_backend: object
    probe_runner: object
    probe_evidence_store: object
    probe_verifier: object
    snapshot_verifier: object
    trial_verifier: object
    checkpoint_store: object
    update_verifier: object
    reward_verifier: object
    action_verifier: object
    designer_checkpoint: Checkpoint
    target_ability_id: str
    designer_selection: Proposal
    feedback_provider: Callable
    training_identity_resolver: Callable
    designer_batch_id: str
    designer_reward_baseline: float
    lambda_retention: float = 0.5
    min_valid_fraction: float = 0.8

    def __post_init__(self):
        resumable_stages = {Stage.CERTIFIED, Stage.PROBES_FROZEN, Stage.BASELINE_READY,
            Stage.READY_FOR_TRAINING, Stage.SERVICE_TRAINING_COMPLETED,
            Stage.PROBES_EVALUATED, Stage.LEARNING_REWARDED, Stage.DESIGNER_READY,
            Stage.DESIGNER_TRAINING_COMPLETED}
        if (self.state.stage not in resumable_stages or not self.state.designer_action
                or (self.state.reward_mode != RewardMode.REPLAY_LOGP_V1.value
                    and self.state.training_package is None)):
            raise ValueError('round plan must start with a certified package and its captured Designer action')
        if (self.state.designer_action.checkpoint != self.designer_checkpoint
                or self.state.designer_action.group_id is not None
                or self.state.package_id != self.service_request.package_id
                or not self.target_ability_id
                or self.state.target_ability_id != self.target_ability_id
                or not isinstance(self.designer_selection, Proposal)
                or self.state.proposal_ids != (self.designer_selection.proposal_id,)
                or self.recovery_snapshot.target_ability_id != self.target_ability_id
                or self.retention_snapshot.target_ability_id is not None
                or self.service_request.policy_before != self.recovery_snapshot.baseline_checkpoint
                or self.recovery_snapshot.baseline_checkpoint != self.retention_snapshot.baseline_checkpoint):
            raise ValueError('round plan package action, request and Probe baselines differ')
        if self.state.training_package is not None:
            self.state.training_package.assert_disjoint_from(self.recovery_snapshot,
                                                             self.retention_snapshot)
            # Validate the requested complete groups before any baseline/model work.
            self.state.training_package.record_service_request(self.service_request)
        if (not self.designer_batch_id or not callable(self.feedback_provider)
                or not callable(self.training_identity_resolver)
                or not isinstance(self.designer_reward_baseline, (int, float))):
            raise ValueError('explicit Designer credit and trusted runtime adapters required')


class EvolutionPlanFactory(Protocol):
    def build(self, round_index: int, service_checkpoint: Checkpoint,
              designer_checkpoint: Checkpoint, banks: BankManager,
              ability_registry: AbilityRegistry,
              failures: tuple[FailureObservation, ...],
              failure_records: tuple[TaskEvidenceRecord, ...],
              ability_options: tuple[AbilityPrompt, ...],
              previous_state: LoopState | None
              ) -> EvolutionRoundPlan: ...


class EvolutionScheduler:
    """Own bootstrap, per-round persistence, credit, bank refresh and checkpoint handoff.

    The factory performs actual on-policy Designer generation, Usami certification
    and captured Service rollout collection. The scheduler supplies no fake models.
    """
    def __init__(self, *, bootstrap_runner: BootstrapRunner,
                 task_evidence_store: JsonTaskEvidenceStore, state_root,
                 success_threshold=0.75, failure_threshold=0.25,
                 failure_context_limit=8, seed=0, metric_sink=None):
        self.bootstrap_runner = bootstrap_runner
        self.task_evidence_store = task_evidence_store
        self.state_root = Path(state_root)
        self.success_threshold, self.failure_threshold = success_threshold, failure_threshold
        self.failure_context_limit, self.seed = failure_context_limit, seed
        self.metric_sink = metric_sink
        if type(failure_context_limit) is not int or failure_context_limit < 1 or type(seed) is not int or seed < 0:
            raise ValueError('positive failure context limit and non-negative scheduler seed required')

    def bootstrap(self, task_manifest, checkpoint, *, protocol, trials_per_task=3,
                  seed=None, banks_path=None):
        result = self.bootstrap_runner.run(task_manifest, checkpoint, protocol=protocol,
            trials_per_task=trials_per_task, seed=self.seed if seed is None else seed)
        for record in result.records:
            self.task_evidence_store.save(record)
        manifest = {"schema_version": 1, "evidence_level": sorted(
                level.value for record in result.records for level in record.evidence_levels),
            "checkpoint": asdict(result.checkpoint), "protocol_digest": result.protocol_digest,
            "seed": result.seed, "evaluation_trials": result.evaluation_trials,
            "task_count": len(result.records),
            "classification_counts": {state.value: sum(record.state(
                success_threshold=result.bank_manager.success_bank.success_threshold,
                failure_threshold=result.bank_manager.failure_bank.failure_threshold) is state
                for record in result.records) for state in CapabilityState}}
        encoded = json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        manifest["manifest_digest"] = sha256(encoded.encode("utf-8")).hexdigest()
        self.state_root.mkdir(parents=True, exist_ok=True)
        path = Path(banks_path) if banks_path else self.state_root / 'bootstrap-banks.json'
        result.bank_manager.save(path)
        meta_path = path.with_suffix('.manifest.json')
        meta_path.write_text(json.dumps(manifest, sort_keys=True, ensure_ascii=False), encoding='utf-8')
        return result

    def execute_round(self, plan: EvolutionRoundPlan, state_path):
        path = Path(state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            state = LoopState.load(path, update_verifier=plan.update_verifier,
                reward_verifier=plan.reward_verifier, action_verifier=plan.action_verifier,
                probe_verifier=plan.probe_verifier)
            if (state.package_id != plan.state.package_id
                    or state.proposal_ids != plan.state.proposal_ids
                    or state.designer_action != plan.state.designer_action):
                raise ValueError('resumed round differs from the persisted Designer package action')
            if plan.state.training_package is not None:
                if (state.training_package is None or replace(state.training_package, stats=None)
                        != replace(plan.state.training_package, stats=None)):
                    raise ValueError('resumed package differs from the frozen Designer/Benchmark mix')
        else:
            state = plan.state
            action_args = (plan.state.designer_action.tokens, plan.designer_checkpoint,
                plan.state.package_id, plan.state.designer_action.rollout_id,
                plan.state.designer_action.group_id)
            if plan.action_verifier.verify_designer_action(*action_args) is not True:
                raise ValueError('round plan Designer action evidence failed verification')
            state.save(path)
        verify_proposal = getattr(plan.action_verifier, 'verify_designer_proposal', None)
        if (not callable(verify_proposal)
                or verify_proposal(state.designer_action.rollout_id, plan.designer_selection) is not True):
            raise ValueError('selected ability choice is not bound to captured Designer output')

        def target_bound_feedback(current_state):
            rows = tuple(plan.feedback_provider(current_state))
            if any(not isinstance(row, DesignerFeedback)
                   or row.target_ability_id not in (None, plan.target_ability_id)
                   for row in rows):
                raise ValueError('Designer feedback target differs from verified round attribution')
            return tuple(replace(row, target_ability_id=plan.target_ability_id) for row in rows)

        if state.stage in (Stage.CERTIFIED, Stage.PROBES_FROZEN, Stage.BASELINE_READY, Stage.READY_FOR_TRAINING):
            state = prepare_dual_bank_round(state,
                recovery_snapshot=plan.recovery_snapshot,
                retention_snapshot=plan.retention_snapshot,
                service_request=plan.service_request, probe_runner=plan.probe_runner,
                probe_evidence_store=plan.probe_evidence_store,
                probe_verifier=plan.probe_verifier,
                snapshot_verifier=plan.snapshot_verifier,
                save_state=lambda updated: updated.save(path))
        return run_round(state, service_backend=plan.service_backend,
            designer_backend=plan.designer_backend, checkpoint_store=plan.checkpoint_store,
            update_verifier=plan.update_verifier, reward_verifier=plan.reward_verifier,
            action_verifier=plan.action_verifier, designer_checkpoint=plan.designer_checkpoint,
            feedback_provider=target_bound_feedback, designer_batch_id=plan.designer_batch_id,
            designer_algorithm='reinforce_baseline', probe_runner=plan.probe_runner,
            probe_evidence_store=plan.probe_evidence_store, probe_verifier=plan.probe_verifier,
            snapshot_verifier=plan.snapshot_verifier, trial_verifier=plan.trial_verifier,
            training_identity_resolver=lambda update: plan.training_identity_resolver(update),
            lambda_retention=plan.lambda_retention,
            min_valid_fraction=plan.min_valid_fraction,
            designer_reward_baseline=plan.designer_reward_baseline,
            save_state=lambda updated: updated.save(path))

    def run(self, *, bootstrap: BootstrapResult, initial_service: Checkpoint,
            initial_designer: Checkpoint, plan_factory: EvolutionPlanFactory,
            bank_updater, rounds, bank_manifest_path=None,
            ability_registry_path=None,
            independent_test_runner=None, test_every=0):
        if not isinstance(bootstrap, BootstrapResult) or not callable(getattr(plan_factory, 'build', None)):
            raise TypeError('verified BootstrapResult and EvolutionPlanFactory required')
        if not callable(bank_updater) or type(rounds) is not int or rounds < 1:
            raise ValueError('bank updater and positive round count required')
        if test_every < 0 or (test_every and not callable(independent_test_runner)):
            raise ValueError('independent test schedule requires an isolated test runner')
        banks = bootstrap.bank_manager
        self.state_root.mkdir(parents=True, exist_ok=True)
        banks_path = Path(bank_manifest_path) if bank_manifest_path else self.state_root / 'evolving-banks.json'
        registry_path = (Path(ability_registry_path) if ability_registry_path
                         else self.state_root / 'ability-registry.json')
        if banks_path.exists():
            from rift.banks.core import BankManager
            banks = BankManager.load(banks_path)
        else:
            banks = bootstrap.bank_manager
        ability_registry = (AbilityRegistry.load(registry_path) if registry_path.exists()
                            else AbilityRegistry())
        banks.save(banks_path)
        service_checkpoint, designer_checkpoint, previous = initial_service, initial_designer, None
        completed = []
        for index in range(rounds):
            state_path = self.state_root / f'round-{index:06d}.json'
            finalized_path = state_path.with_name(state_path.stem + '.finalized.json')
            if finalized_path.exists():
                load_completed = getattr(plan_factory, 'load_completed_state', None)
                if not callable(load_completed):
                    raise ValueError('completed rounds require a trusted state recovery adapter')
                restored = load_completed(state_path)
                marker = json.loads(finalized_path.read_text(encoding='utf-8'))
                state_digest = sha256(state_path.read_bytes()).hexdigest()
                if (restored.stage is not Stage.DESIGNER_TRAINING_COMPLETED
                        or restored.round_index != index
                        or marker.get('state_digest') != state_digest
                        or marker.get('service_checkpoint') != asdict(restored.service_update.policy_after)
                        or marker.get('designer_checkpoint') != asdict(restored.designer_update.policy_after)):
                    raise ValueError('round finalization marker differs from verified persisted state')
                service_checkpoint = restored.service_update.policy_after
                designer_checkpoint = restored.designer_update.policy_after
                previous = restored
                completed.append(restored)
                continue
            round_started = time.perf_counter()
            failure_records = banks.failure_bank.sample_records(
                limit=self.failure_context_limit, seed=self.seed + index)
            failures = tuple(public_projection(row) for row in failure_records)
            ability_options = ability_registry.prompts()
            ability_registry.save(registry_path)
            if state_path.exists():
                resume = getattr(plan_factory, 'resume', None)
                if not callable(resume):
                    raise ValueError('an incomplete persisted round requires exact-action resume support')
                plan = resume(index, service_checkpoint, designer_checkpoint,
                    banks, ability_registry, failures, failure_records,
                    ability_options, previous, state_path)
            else:
                plan = plan_factory.build(index, service_checkpoint, designer_checkpoint,
                    banks, ability_registry, failures, failure_records,
                    ability_options, previous)
            if not isinstance(plan, EvolutionRoundPlan):
                raise TypeError('plan factory must return EvolutionRoundPlan')
            if plan.state.round_index != index:
                raise ValueError('round plan index does not match scheduler history')
            verify_proposal = getattr(plan.action_verifier, 'verify_designer_proposal', None)
            if (not callable(verify_proposal)
                    or verify_proposal(plan.state.designer_action.rollout_id,
                        plan.designer_selection, visible_failure_refs=tuple(row.task_ref for row in failures),
                        ability_options=ability_options) is not True):
                raise ValueError('Designer choice is not bound to captured failure and ability context')
            selected = ability_registry.apply_designer_choice(
                ability_id=plan.designer_selection.ability_id,
                hypothesis=plan.designer_selection.hypothesis,
                add_failure_task_refs=plan.designer_selection.diagnosis_task_refs,
                remove_failure_task_refs=plan.designer_selection.remove_failure_task_refs,
                visible_failures=failure_records, evidence_verifier=banks,
                round_index=index,
                designer_rollout_id=plan.state.designer_action.rollout_id)
            if selected.ability_id != plan.target_ability_id:
                raise ValueError('resolved Designer ability ID differs from prepared round')
            priority_refs = ability_registry.failure_task_refs(plan.target_ability_id)
            if not set(plan.recovery_snapshot.priority_task_refs).issubset(priority_refs):
                raise ValueError('Recovery priority tasks lack an explicit target-ability link')
            if plan.state.expected_service != (None if previous is None else service_checkpoint):
                raise ValueError('round plan does not continue the latest Service checkpoint')
            if plan.state.expected_designer != (None if previous is None else designer_checkpoint):
                raise ValueError('round plan does not continue the latest Designer checkpoint')
            result = self.execute_round(plan, state_path)
            if result.stage is not Stage.DESIGNER_TRAINING_COMPLETED:
                raise ValueError('round did not complete both verified policy updates')
            bank_updater(result, banks)
            ability_registry.record_round(result, update_verifier=plan.update_verifier,
                                          reward_verifier=plan.reward_verifier)
            ability_registry.save(registry_path)
            banks.save(banks_path)
            if self.metric_sink is not None:
                self.metric_sink('round_wall_time_seconds', time.perf_counter() - round_started,
                    round_index=index, service_checkpoint=result.service_update.policy_after.revision,
                    designer_checkpoint=result.designer_update.policy_after.revision)
            marker = {"schema_version": 1,
                "state_digest": sha256(state_path.read_bytes()).hexdigest(),
                "service_checkpoint": asdict(result.service_update.policy_after),
                "designer_checkpoint": asdict(result.designer_update.policy_after)}
            temporary = finalized_path.with_suffix(finalized_path.suffix + '.tmp')
            temporary.write_text(json.dumps(marker, sort_keys=True), encoding='utf-8')
            os.replace(temporary, finalized_path)
            completed.append(result)
            service_checkpoint, designer_checkpoint = result.service_update.policy_after, result.designer_update.policy_after
            previous = result
            if test_every and (index + 1) % test_every == 0:
                # Test output is researcher-only and is deliberately not passed to plan_factory or bank_updater.
                independent_test_runner.run(service_checkpoint, round_index=index)
        return tuple(completed)
