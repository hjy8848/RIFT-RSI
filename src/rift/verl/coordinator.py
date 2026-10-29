"""Explicit alternating-round orchestration; no work occurs on import."""
from rift.loop import Stage
from rift.replay.reward import score_update
from rift.training.records import prepare_designer_update
from rift.rewards.learning_value import RewardMode, score_dual_bank


def prepare_dual_bank_round(state, *, recovery_snapshot, retention_snapshot,
    service_request, probe_runner, probe_evidence_store, probe_verifier,
    snapshot_verifier, save_state):
    """Durably freeze Probes and cache same-checkpoint baselines before training."""
    if state.stage is Stage.CERTIFIED:
        probe_evidence_store.save_snapshot(recovery_snapshot)
        probe_evidence_store.save_snapshot(retention_snapshot)
        state = state.freeze_probes(recovery_snapshot, retention_snapshot,
                                    snapshot_verifier=snapshot_verifier)
        save_state(state)
    elif state.stage not in (Stage.PROBES_FROZEN, Stage.BASELINE_READY, Stage.READY_FOR_TRAINING):
        raise ValueError('round is not at a resumable pre-training Probe phase')
    elif (state.recovery_snapshot != recovery_snapshot
            or state.retention_snapshot != retention_snapshot):
        raise ValueError('resumed round cannot replace its frozen Probe snapshots')
    if state.stage is Stage.PROBES_FROZEN:
        recovery_before = probe_runner.baseline(state.recovery_snapshot)
        retention_before = probe_runner.baseline(state.retention_snapshot)
        probe_evidence_store.save_results(recovery_before)
        probe_evidence_store.save_results(retention_before)
        state = state.baseline_ready(recovery_before, retention_before,
                                     probe_verifier=probe_verifier)
        save_state(state)
    if state.stage is Stage.BASELINE_READY:
        state = state.ready(service_request)
        save_state(state)
    elif state.stage is Stage.READY_FOR_TRAINING and state.service_request != service_request:
        raise ValueError('resumed Service request differs from persisted request')
    return state


def run_round(state, *, service_backend, designer_backend, replay_snapshot=None, scorer=None,
              checkpoint_store=None, update_verifier, replay_verifier=None, reward_verifier,
              action_verifier, designer_checkpoint, feedback_provider, save_state,
              designer_batch_id, designer_algorithm=None, normalization='sum',
              probe_runner=None, probe_evidence_store=None, probe_verifier=None,
              snapshot_verifier=None, trial_verifier=None, training_identity_resolver=None,
              lambda_retention=0.5, min_valid_fraction=0.8, designer_reward_baseline=None):
    """Advance a previously persisted round using supplied trusted infrastructure.

    Every transition is persisted. Backend failures leave the preceding state on
    disk; a partially written checkpoint must be reconciled by the caller rather
    than silently rerunning or treating it as complete.
    """
    for update in (state.service_update, state.designer_update):
        if update is not None and update_verifier.verify_update(update) is not True:
            raise ValueError('existing update evidence was not verified')
    if state.replay_reward is not None and reward_verifier.verify_reward(state.replay_reward, state.service_update) is not True:
        raise ValueError('existing replay reward evidence was not verified')
    if state.learning_reward is not None and reward_verifier.verify_reward(state.learning_reward, state.service_update) is not True:
        raise ValueError('existing dual-bank reward evidence was not verified')
    if state.designer_request is not None:
        restored = prepare_designer_update(state.designer_request.package_id, state.designer_request.policy_before,
            state.designer_feedback, update_verifier=update_verifier, reward_verifier=reward_verifier,
            action_verifier=action_verifier, algorithm=state.designer_request.algorithm,
            credit_assignment_mode=state.designer_request.credit_assignment_mode,
            reward_baseline=state.designer_request.advantage_baseline)
        if restored != state.designer_request:
            raise ValueError('existing Designer request differs from verified evidence')
    if state.stage is Stage.READY_FOR_TRAINING:
        update = service_backend.train(state.service_request)
        state = state.service_completed(update, update_verifier=update_verifier)
        save_state(state)
    if state.stage is Stage.SERVICE_TRAINING_COMPLETED:
        if state.reward_mode == RewardMode.REPLAY_LOGP_V1.value:
            if replay_snapshot is None or scorer is None or checkpoint_store is None or replay_verifier is None:
                raise ValueError('Replay LogP mode requires its frozen snapshot, scorer and evidence stores')
            reward = score_update(state.service_update, replay_snapshot, scorer, checkpoint_store,
                update_verifier=update_verifier, replay_verifier=replay_verifier, normalization=normalization)
            state = state.replay_scored(reward, reward_verifier=reward_verifier)
            save_state(state)
        else:
            if probe_runner is None or probe_evidence_store is None or probe_verifier is None:
                raise ValueError('dual-bank mode requires its Probe runner, evidence store and verifier')
            recovery_after = probe_runner.after(state.recovery_snapshot, state.service_update.policy_after)
            retention_after = probe_runner.after(state.retention_snapshot, state.service_update.policy_after)
            probe_evidence_store.save_results(recovery_after)
            probe_evidence_store.save_results(retention_after)
            state = state.probes_evaluated(recovery_after, retention_after,
                probe_verifier=probe_verifier)
            save_state(state)
    if state.stage is Stage.PROBES_EVALUATED:
        if snapshot_verifier is None or trial_verifier is None or training_identity_resolver is None:
            raise ValueError('dual-bank reward needs frozen-snapshot, Episode and task-identity verifiers')
        reward = score_dual_bank(state.service_update, state.recovery_snapshot,
            state.retention_snapshot, state.recovery_before, state.recovery_after,
            state.retention_before, state.retention_after,
            update_verifier=update_verifier, checkpoint_store=checkpoint_store,
            snapshot_verifier=snapshot_verifier, trial_verifier=trial_verifier,
            training_identities=training_identity_resolver(state.service_update),
            lambda_retention=lambda_retention, min_valid_fraction=min_valid_fraction)
        state = state.learning_rewarded(reward, reward_verifier=reward_verifier)
        save_state(state)
    if state.stage in (Stage.REPLAY_SCORED, Stage.LEARNING_REWARDED):
        feedback = tuple(feedback_provider(state))
        request = prepare_designer_update(designer_batch_id, designer_checkpoint, feedback,
            update_verifier=update_verifier, reward_verifier=reward_verifier,
            action_verifier=action_verifier, algorithm=designer_algorithm,
            reward_baseline=designer_reward_baseline)
        state = state.designer_ready(request, feedback, update_verifier=update_verifier,
            reward_verifier=reward_verifier, action_verifier=action_verifier)
        save_state(state)
    if state.stage is Stage.DESIGNER_READY:
        update = designer_backend.train(state.designer_request)
        state = state.designer_completed(update, update_verifier=update_verifier)
        save_state(state)
    if state.stage is not Stage.DESIGNER_TRAINING_COMPLETED:
        raise ValueError('round must start at a supported training phase')
    return state
