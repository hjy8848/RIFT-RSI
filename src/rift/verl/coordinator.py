"""Explicit alternating-round orchestration; no work occurs on import."""
from rift.loop import Stage
from rift.replay.reward import score_update
from rift.training.records import prepare_designer_update


def run_round(state, *, service_backend, designer_backend, replay_snapshot, scorer,
              checkpoint_store, update_verifier, replay_verifier, reward_verifier,
              action_verifier, designer_checkpoint, feedback_provider, save_state,
              designer_batch_id, designer_algorithm='reinforce_plus_plus', normalization='sum'):
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
    if state.designer_request is not None:
        restored = prepare_designer_update(state.designer_request.package_id, state.designer_request.policy_before,
            state.designer_feedback, update_verifier=update_verifier, reward_verifier=reward_verifier,
            action_verifier=action_verifier, algorithm=state.designer_request.algorithm)
        if restored != state.designer_request:
            raise ValueError('existing Designer request differs from verified evidence')
    if state.stage is Stage.READY_FOR_TRAINING:
        update = service_backend.train(state.service_request)
        state = state.service_completed(update, update_verifier=update_verifier)
        save_state(state)
    if state.stage is Stage.SERVICE_TRAINING_COMPLETED:
        reward = score_update(state.service_update, replay_snapshot, scorer, checkpoint_store,
            update_verifier=update_verifier, replay_verifier=replay_verifier, normalization=normalization)
        state = state.replay_scored(reward, reward_verifier=reward_verifier)
        save_state(state)
    if state.stage is Stage.REPLAY_SCORED:
        feedback = tuple(feedback_provider(state))
        request = prepare_designer_update(designer_batch_id, designer_checkpoint, feedback,
            update_verifier=update_verifier, reward_verifier=reward_verifier,
            action_verifier=action_verifier, algorithm=designer_algorithm)
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
