"""Production assembly for one existing RIFT dual-bank round."""
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import inspect
from pathlib import Path
import time
from types import SimpleNamespace

from rift.banks.bootstrap import _trial_seed
from rift.banks.records import TaskEvidenceRecord
from rift.evolution.scheduler import EvolutionRoundPlan
from rift.loop import LoopState, PendingDesignerAction, Stage
from rift.probes.runner import ProbeRunner
from rift.schemas import Checkpoint, DataPurpose, Split
from rift.tasks import Proposal
from rift.training.package import TrainingPackageConfig, assemble_training_package
from rift.training.records import DesignerFeedback
from rift.runtime.evaluator import ProductionProbeEvaluator, ProductionServiceRolloutCollector
from rift.runtime.resources import RoleBoundBackend
from rift.verl.designer import PendingDesignerRollout


def _await_sync(value):
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    raise RuntimeError("synchronous evolution scheduler cannot await an active event loop")


def _digest_id(*parts):
    material = ":".join(str(part) for part in parts).encode("utf-8")
    return sha256(material).hexdigest()[:32]


class _RoleProbeEvaluator:
    def __init__(self, resources, *, seed_base=0, metric_sink=None, round_index=None):
        self.resources, self.seed_base = resources, seed_base
        self.metric_sink, self.round_index = metric_sink, round_index

    def evaluate(self, task_ref, checkpoint, seed, protocol_ref, protocol_digest):
        return self.evaluate_many(((task_ref, checkpoint, seed, protocol_ref, protocol_digest),))[0]

    def evaluate_many(self, requests):
        requests = tuple(requests)
        if not requests:
            return ()
        checkpoints = {row[1] for row in requests}
        if len(checkpoints) != 1:
            raise ValueError("one batched Probe call must use a single checkpoint")
        checkpoint = next(iter(checkpoints))
        with self.resources.acquire("service", checkpoint) as runtime:
            batch = getattr(runtime, "probe_batch", None)
            if not callable(getattr(batch, "generate", None)):
                raise ValueError("Service runtime has no pinned VERL Probe Agent Loop batch")
            return ProductionProbeEvaluator(batch, seed_base=self.seed_base,
                metric_sink=self.metric_sink, round_index=self.round_index,
                metric_name='probe_after_seconds').evaluate_many(requests)


class ProductionPlanFactory:
    """Connect captured Designer decisions, certified Usami tasks and VERL batches.

    Deployment code supplies only environment roots and official VERL worker
    construction. Sampling, package formation, Probe freezing and RIFT state
    transitions stay in the existing project implementations.
    """
    def __init__(self, *, resources, commerce_host, benchmark_pool, task_registry,
                 split_registry, task_evidence_store, probe_selector, probe_evidence_store,
                 capture_store, designer_capture_store, checkpoint_store,
                 update_verifier, reward_verifier, action_verifier,
                 snapshot_verifier, trial_verifier, freeze_order_verifier,
                 protocol_ref, protocol_digest, package_config=TrainingPackageConfig(),
                 recovery_tasks=16, retention_tasks=16, trials_per_probe=5,
                 probe_seed=4100, designer_sampling_params=None,
                 designer_reward_baseline=0.0, lambda_retention=0.5,
                 min_valid_fraction=0.8, allowed_training_hashes=(), metric_sink=None,
                 checkpoint_root=None):
        required = (resources, commerce_host, benchmark_pool, task_registry, split_registry,
            task_evidence_store, probe_selector, probe_evidence_store, capture_store,
            designer_capture_store, checkpoint_store, update_verifier, reward_verifier,
            action_verifier, snapshot_verifier, trial_verifier, freeze_order_verifier)
        if any(value is None for value in required) or checkpoint_root is None:
            raise ValueError("all trusted production task, VERL, checkpoint and evidence dependencies are required")
        if (not protocol_ref or not protocol_digest or type(recovery_tasks) is not int
                or recovery_tasks < 1 or type(retention_tasks) is not int or retention_tasks < 1
                or type(trials_per_probe) is not int or trials_per_probe < 1
                or type(probe_seed) is not int or probe_seed < 0
                or not isinstance(package_config, TrainingPackageConfig)):
            raise ValueError("invalid frozen training and Probe configuration")
        if lambda_retention < 0 or not 0 < min_valid_fraction <= 1:
            raise ValueError("invalid dual-bank reward settings")
        self.resources, self.commerce_host, self.benchmark_pool = resources, commerce_host, benchmark_pool
        self.task_registry, self.split_registry = task_registry, split_registry
        self.task_evidence_store, self.probe_selector = task_evidence_store, probe_selector
        self.probe_evidence_store, self.capture_store = probe_evidence_store, capture_store
        self.designer_capture_store, self.checkpoint_store = designer_capture_store, checkpoint_store
        self.update_verifier, self.reward_verifier = update_verifier, reward_verifier
        self.action_verifier, self.snapshot_verifier = action_verifier, snapshot_verifier
        self.trial_verifier, self.freeze_order_verifier = trial_verifier, freeze_order_verifier
        self.probe_verifier = getattr(probe_evidence_store, "verifier", None)
        if self.probe_verifier is None:
            from rift.probes.verification import StoredProbeEvidenceVerifier
            self.probe_verifier = StoredProbeEvidenceVerifier(
                probe_evidence_store, trial_verifier, freeze_order_verifier)
        self.protocol_ref, self.protocol_digest = protocol_ref, protocol_digest
        self.package_config = package_config
        self.recovery_tasks, self.retention_tasks = recovery_tasks, retention_tasks
        self.trials_per_probe, self.probe_seed = trials_per_probe, probe_seed
        self.designer_sampling_params = dict(designer_sampling_params or {
            "temperature": 1.0, "top_p": 1.0, "top_k": -1,
            "repetition_penalty": 1.0, "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        })
        self.designer_reward_baseline = designer_reward_baseline
        self.lambda_retention, self.min_valid_fraction = lambda_retention, min_valid_fraction
        self.allowed_training_hashes = tuple(allowed_training_hashes)
        self.metric_sink = metric_sink
        self.checkpoint_root = Path(checkpoint_root)

    def _action(self, round_index, service_checkpoint, designer_checkpoint,
                banks, ability_registry, failures, ability_options):
        if not failures:
            raise ValueError("Designer cannot generate a package without verified Failure Bank evidence")
        package_id = _digest_id("rift-package-v1", round_index,
            service_checkpoint.weights_digest, designer_checkpoint.weights_digest)
        rollout_id = _digest_id("rift-designer-action-v1", package_id,
            designer_checkpoint.weights_digest)
        with self.resources.acquire("designer", designer_checkpoint) as runtime:
            sampler = getattr(runtime, "designer_sampler", None)
            if not callable(getattr(sampler, "sample", None)):
                raise ValueError("Designer runtime has no captured on-policy DesignerSampler")
            path = sampler.capture_store._path(rollout_id)
            if path.exists():
                pending = sampler.capture_store.load_pending(rollout_id)
            else:
                started = time.perf_counter()
                pending = _await_sync(sampler.sample(failures, ability_options,
                    designer_checkpoint, package_id, self.designer_sampling_params,
                    group_id=None, group_size=None, rollout_id=rollout_id))
                if self.metric_sink is not None:
                    self.metric_sink('designer_generation_seconds', time.perf_counter() - started,
                        round_index=round_index, rollout_id=rollout_id)
        if not isinstance(pending, PendingDesignerRollout) or len(pending.proposals) != 1:
            raise ValueError("one captured package-level Designer decision is required")
        proposal = pending.proposals[0]
        if not proposal.is_training_plan:
            raise ValueError("Designer output must contain its World and challenges in one generation")
        if proposal.ability_id == "NEW":
            target_ability_id = ability_registry.next_id()
        else:
            ability_registry.get(proposal.ability_id)
            target_ability_id = proposal.ability_id
        if self.action_verifier.verify_designer_proposal(
                pending.rollout_id, proposal,
                visible_failure_refs=tuple(row.task_ref for row in failures),
                ability_options=ability_options) is not True:
            raise ValueError("captured Designer output is not bound to the current visible context")
        return package_id, pending, proposal, target_ability_id

    def _current_baselines(self, histories, checkpoint, round_index, batch):
        from rift.banks.records import TaskEvidenceRecord
        candidates = {}
        for history in histories:
            if (history.identity.split is Split.TEST
                    or history.purpose is not DataPurpose.BANK_CANDIDATE
                    or not self.split_registry.contains(history.identity)):
                continue
            old = candidates.get(history.task_ref)
            if old is None or old.observed_at < history.observed_at:
                candidates[history.task_ref] = history
        if not candidates:
            raise ValueError("no non-TEST Failure/Success Bank task is available for Probe selection")

        verified = {}
        missing = []
        for task_ref, history in sorted(candidates.items()):
            try:
                cached = self.task_evidence_store.load(task_ref, checkpoint)
            except FileNotFoundError:
                cached = None
            if (cached is not None and cached.identity == history.identity
                    and cached.protocol_digest == self.protocol_digest
                    and len(cached.trials) == self.trials_per_probe
                    and all(self.trial_verifier.verify_probe_trial(
                        trial, SimpleNamespace(task_ref=task_ref, identity=history.identity,
                            protocol_ref=self.protocol_ref, protocol_digest=self.protocol_digest),
                        checkpoint) is True for trial in cached.trials)):
                verified[task_ref] = cached
            else:
                missing.append((task_ref, history.identity))

        if missing:
            requests = []
            for task_ref, _identity in missing:
                for trial_index in range(self.trials_per_probe):
                    seed = _trial_seed(self.probe_seed + round_index, task_ref, trial_index)
                    requests.append((task_ref, checkpoint, seed, self.protocol_ref, self.protocol_digest))
            started = time.perf_counter()
            trials = tuple(batch.evaluate_many(tuple(requests)))
            if self.metric_sink is not None:
                self.metric_sink('probe_before_seconds', time.perf_counter() - started,
                    round_index=round_index, request_count=len(requests))
            if len(trials) != len(requests):
                raise ValueError("current-Checkpoint Probe baseline batch is incomplete")
            offset = 0
            observed_at = datetime.now(timezone.utc).isoformat()
            for task_ref, identity in missing:
                task_trials = trials[offset:offset + self.trials_per_probe]
                offset += self.trials_per_probe
                probe_context = SimpleNamespace(task_ref=task_ref, identity=identity,
                    protocol_ref=self.protocol_ref, protocol_digest=self.protocol_digest)
                for trial in task_trials:
                    if self.trial_verifier.verify_probe_trial(trial, probe_context, checkpoint) is not True:
                        raise ValueError("new current-checkpoint Probe baseline evidence failed verification")
                record = TaskEvidenceRecord(task_ref, identity, checkpoint, self.protocol_digest,
                                            tuple(task_trials), observed_at)
                self.task_evidence_store.save(record)
                verified[task_ref] = record
        return tuple(verified.values())

    def _freeze_snapshots(self, round_index, service_checkpoint, target_ability_id,
                          proposal, failure_records, recovery_histories,
                          retention_histories, training_package, batch,
                          ability_task_refs):
        baselines = self._current_baselines(
            tuple(recovery_histories) + tuple(retention_histories),
            service_checkpoint, round_index, batch)
        diagnosis_refs = set(proposal.diagnosis_task_refs)
        linked_refs = set(ability_task_refs) | diagnosis_refs
        identities = tuple(source.identity for source in training_package.task_sources)
        recovery, retention = self.probe_selector.freeze(
            recovery_candidates=tuple(recovery_histories),
            retention_candidates=tuple(retention_histories),
            train_package_identities=identities,
            split_registry=self.split_registry,
            before_checkpoint=service_checkpoint,
            recovery_tasks=self.recovery_tasks,
            retention_tasks=self.retention_tasks,
            seed=self.probe_seed + round_index,
            protocol_ref=self.protocol_ref,
            protocol_digest=self.protocol_digest,
            frozen_at_ns=time.time_ns(),
            target_ability_id=target_ability_id,
            ability_task_refs=tuple(linked_refs),
            current_baseline_candidates=baselines)
        return recovery, retention

    def _service_request_and_snapshots(self, round_index, package, proposal,
            target_ability_id, service_checkpoint, banks, ability_registry,
            failure_records, ability_registry_record, recovery_snapshot=None,
            retention_snapshot=None, service_request=None):
        recovery_histories = tuple(banks.failure_bank.records)
        retention_histories = tuple(banks.success_bank.records)
        with self.resources.acquire("service", service_checkpoint) as runtime:
            probe_batch = getattr(runtime, "probe_batch", None)
            if not callable(getattr(probe_batch, "generate", None)):
                raise ValueError("Service runtime has no VERL Probe Agent Loop batch")
            baseline_batch = ProductionProbeEvaluator(probe_batch)
            if recovery_snapshot is None or retention_snapshot is None:
                ability_refs = set(proposal.diagnosis_task_refs)
                if ability_registry_record is not None:
                    ability_refs.update(ability_registry_record.failure_task_refs)
                    ability_refs.difference_update(proposal.remove_failure_task_refs)
                # Feed the exact explicit target links into the existing selector.
                recovery_snapshot, retention_snapshot = self._freeze_snapshots(
                    round_index, service_checkpoint, target_ability_id, proposal,
                    failure_records, recovery_histories, retention_histories,
                    package, baseline_batch, ability_refs)
            if service_request is None:
                service_batch = getattr(runtime, "service_batch", None)
                if not callable(getattr(service_batch, "generate", None)):
                    raise ValueError("Service runtime has no VERL rollout Agent Loop batch")
                collector = ProductionServiceRolloutCollector(
                    service_batch, task_registry=self.task_registry,
                    capture_store=self.capture_store, metric_sink=self.metric_sink,
                    round_index=round_index)
                service_request, _realized_package = collector.collect(package, service_checkpoint)
        return recovery_snapshot, retention_snapshot, service_request

    def _make_plan(self, *, state, package, proposal, target_ability_id,
                   designer_checkpoint, service_request, recovery_snapshot,
                   retention_snapshot):
        from rift.probes.verification import StoredProbeEvidenceVerifier
        trial_verifier = self.trial_verifier
        probe_verifier = self.probe_verifier or StoredProbeEvidenceVerifier(
            self.probe_evidence_store, trial_verifier, self.freeze_order_verifier)
        probe_runner = ProbeRunner(_RoleProbeEvaluator(self.resources,
            seed_base=self.probe_seed, metric_sink=self.metric_sink,
            round_index=state.round_index), trial_verifier)
        designer_rollout_id = state.designer_action.rollout_id
        def feedback_provider(current_state):
            pending = self.designer_capture_store.load_pending(designer_rollout_id)
            if len(pending.proposals) != 1:
                raise ValueError("persisted Designer action is not a single package decision")
            if current_state.learning_reward is None or current_state.service_update is None:
                raise ValueError("delayed Designer credit is not yet verified")
            return (DesignerFeedback(pending.trace.tokens, current_state.service_update,
                current_state.learning_reward, designer_rollout_id, None, target_ability_id),)
        def training_identity_resolver(update):
            refs = tuple(dict.fromkeys(row.task_id for row in update.request.rollouts))
            identities = tuple(self.task_registry.resolve_record(ref).identity for ref in refs)
            if any(row.split is Split.TEST for row in identities):
                raise ValueError("TEST identities cannot enter a Service training update")
            return identities
        return EvolutionRoundPlan(
            state=state,
            service_request=service_request,
            recovery_snapshot=recovery_snapshot,
            retention_snapshot=retention_snapshot,
            service_backend=RoleBoundBackend(self.resources, "service",
                checkpoint_root=self.checkpoint_root,
                metric_sink=self.metric_sink, round_index=state.round_index),
            designer_backend=RoleBoundBackend(self.resources, "designer",
                checkpoint_root=self.checkpoint_root,
                metric_sink=self.metric_sink, round_index=state.round_index),
            probe_runner=probe_runner,
            probe_evidence_store=self.probe_evidence_store,
            probe_verifier=probe_verifier,
            snapshot_verifier=self.snapshot_verifier,
            trial_verifier=trial_verifier,
            checkpoint_store=self.checkpoint_store,
            update_verifier=self.update_verifier,
            reward_verifier=self.reward_verifier,
            action_verifier=self.action_verifier,
            designer_checkpoint=designer_checkpoint,
            target_ability_id=target_ability_id,
            designer_selection=proposal,
            feedback_provider=feedback_provider,
            training_identity_resolver=training_identity_resolver,
            designer_batch_id=f"designer-update-{state.package_id}",
            designer_reward_baseline=self.designer_reward_baseline,
            lambda_retention=self.lambda_retention,
            min_valid_fraction=self.min_valid_fraction)

    def build(self, round_index, service_checkpoint, designer_checkpoint, banks,
              ability_registry, failures, failure_records, ability_options, previous_state):
        package_id, pending, proposal, target_ability_id = self._action(
            round_index, service_checkpoint, designer_checkpoint, banks,
            ability_registry, failures, ability_options)
        world_seed = self.package_config.seed_for_round(round_index)
        designer_result = self.commerce_host.certify_training_plan(
            proposal, world_seed=world_seed,
            max_candidate_challenges=self.package_config.max_candidate_challenges,
            allowed_training_hashes=self.allowed_training_hashes,
            metrics=self.metric_sink)
        package = assemble_training_package(
            package_id=package_id, proposal=proposal,
            designer_action_ref=pending.rollout_id,
            target_ability_id=target_ability_id,
            designer_result=designer_result,
            benchmark_pool=self.benchmark_pool,
            task_registry=self.task_registry,
            service_checkpoint_before=service_checkpoint,
            config=self.package_config,
            round_index=round_index)
        if proposal.ability_id == "NEW":
            ability_record = None
        else:
            ability_record = ability_registry.get(proposal.ability_id)
        recovery_snapshot, retention_snapshot, service_request = self._service_request_and_snapshots(
            round_index, package, proposal, target_ability_id, service_checkpoint, banks,
            ability_registry, failure_records, ability_record)
        previous_updates = () if previous_state is None else (
            previous_state.service_update, previous_state.designer_update)
        state = LoopState(
            package_id=package.package_id,
            proposal_ids=(proposal.proposal_id,),
            target_ability_id=target_ability_id,
            round_index=round_index,
            expected_service=None if previous_state is None else service_checkpoint,
            expected_designer=None if previous_state is None else designer_checkpoint,
            previous_updates=previous_updates,
            protocol_version=3,
            reward_mode="dual_bank_outcome_v1",
            designer_action=PendingDesignerAction(designer_checkpoint, pending.rollout_id,
                                                  pending.trace.tokens, None),
            training_package=package)
        state = state.certify_training_package(package, designer_result.certifications)
        return self._make_plan(state=state, package=package, proposal=proposal,
            target_ability_id=target_ability_id, designer_checkpoint=designer_checkpoint,
            service_request=service_request, recovery_snapshot=recovery_snapshot,
            retention_snapshot=retention_snapshot)

    def load_completed_state(self, state_path):
        return LoopState.load(state_path, update_verifier=self.update_verifier,
            reward_verifier=self.reward_verifier, action_verifier=self.action_verifier,
            probe_verifier=self.probe_verifier)

    def resume(self, round_index, service_checkpoint, designer_checkpoint, banks,
               ability_registry, failures, failure_records, ability_options,
               previous_state, state_path):
        """Rebuild adapters around the exact persisted plan and on-policy action."""
        state = LoopState.load(state_path, update_verifier=self.update_verifier,
            reward_verifier=self.reward_verifier, action_verifier=self.action_verifier,
            probe_verifier=self.probe_verifier)
        expected_service = None if previous_state is None else service_checkpoint
        expected_designer = None if previous_state is None else designer_checkpoint
        if (state.round_index != round_index
                or state.expected_service != expected_service
                or state.expected_designer != expected_designer
                or state.designer_action.checkpoint != designer_checkpoint):
            raise ValueError("persisted round does not continue the verified scheduler checkpoint")
        pending = self.designer_capture_store.load_pending(state.designer_action.rollout_id)
        if len(pending.proposals) != 1 or pending.proposals[0].proposal_id != state.proposal_ids[0]:
            raise ValueError("persisted round is not bound to its original package proposal")
        proposal = pending.proposals[0]
        if self.action_verifier.verify_designer_proposal(
                pending.rollout_id, proposal,
                visible_failure_refs=tuple(row.task_ref for row in failures),
                ability_options=ability_options) is not True:
            raise ValueError("persisted Designer action differs from the current visible context")
        package = state.training_package
        if package is None or package.package_id != state.package_id:
            raise ValueError("persisted production round is missing its frozen TrainingPackage")
        ability_record = (None if proposal.ability_id == "NEW"
                          else ability_registry.get(proposal.ability_id))
        recovery_snapshot, retention_snapshot = state.recovery_snapshot, state.retention_snapshot
        request = state.service_request
        if request is None:
            recovery_snapshot, retention_snapshot, request = self._service_request_and_snapshots(
                round_index, replace(package, stats=None), proposal, state.target_ability_id,
                service_checkpoint, banks, ability_registry, failure_records, ability_record,
                recovery_snapshot, retention_snapshot, None)
        elif recovery_snapshot is None or retention_snapshot is None:
            raise ValueError("persisted Service request is missing its frozen Probe snapshots")
        return self._make_plan(state=state, package=package, proposal=proposal,
            target_ability_id=state.target_ability_id, designer_checkpoint=designer_checkpoint,
            service_request=request, recovery_snapshot=recovery_snapshot,
            retention_snapshot=retention_snapshot)
