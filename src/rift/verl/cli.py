"""Single-round entry point with an explicit, application-owned runtime factory.

The factory constructs certified Usami stores and real VERL worker groups. This
module never invents model weights or launches work merely by being imported.
"""
import argparse
from dataclasses import dataclass
import importlib
import json
from pathlib import Path

from rift.loop import LoopState
from rift.verl.coordinator import run_round


@dataclass
class VerlRoundRuntime:
    """Trusted dependencies required to execute one persisted RIFT round."""
    initial_state: LoopState | None
    service_backend: object
    designer_backend: object
    replay_snapshot: object
    scorer: object
    checkpoint_store: object
    update_verifier: object
    replay_verifier: object
    reward_verifier: object
    action_verifier: object
    designer_checkpoint: object
    feedback_provider: object
    designer_batch_id: str
    designer_algorithm: str = 'reinforce_plus_plus'
    normalization: str = 'sum'

    def __post_init__(self):
        if not callable(self.feedback_provider) or not self.designer_batch_id:
            raise ValueError('runtime must provide a feedback factory and Designer batch ID')
        if self.designer_algorithm not in ('grpo', 'reinforce_plus_plus'):
            raise ValueError('unsupported Designer advantage estimator')


def import_runtime_factory(spec):
    if not isinstance(spec, str) or ':' not in spec:
        raise ValueError('runtime factory must use module:callable syntax')
    module_name, attribute = spec.split(':', 1)
    if not module_name or not attribute:
        raise ValueError('runtime factory must use module:callable syntax')
    factory = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(factory):
        raise ValueError('runtime factory target is not callable')
    return factory


def execute_round(runtime, state_path):
    if not isinstance(runtime, VerlRoundRuntime):
        raise TypeError('runtime factory must return VerlRoundRuntime')
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        state = LoopState.load(path, update_verifier=runtime.update_verifier,
            reward_verifier=runtime.reward_verifier, action_verifier=runtime.action_verifier)
    else:
        state = runtime.initial_state
        if not isinstance(state, LoopState):
            raise ValueError('a new state file requires a prepared, evidence-backed initial LoopState')
        state.save(path)
    result = run_round(state,
        service_backend=runtime.service_backend,
        designer_backend=runtime.designer_backend,
        replay_snapshot=runtime.replay_snapshot,
        scorer=runtime.scorer,
        checkpoint_store=runtime.checkpoint_store,
        update_verifier=runtime.update_verifier,
        replay_verifier=runtime.replay_verifier,
        reward_verifier=runtime.reward_verifier,
        action_verifier=runtime.action_verifier,
        designer_checkpoint=runtime.designer_checkpoint,
        feedback_provider=runtime.feedback_provider,
        save_state=lambda updated: updated.save(path),
        designer_batch_id=runtime.designer_batch_id,
        designer_algorithm=runtime.designer_algorithm,
        normalization=runtime.normalization)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Execute or resume one RIFT round using a trusted runtime factory.')
    parser.add_argument('--runtime-factory', required=True, help='Python factory in module:callable form')
    parser.add_argument('--state', required=True, help='LoopState JSON path for this round')
    args = parser.parse_args(argv)
    runtime = import_runtime_factory(args.runtime_factory)(args)
    result = execute_round(runtime, args.state)
    summary = {'package_id': result.package_id, 'stage': result.stage.value,
        'service_revision': result.service_update.policy_after.revision,
        'designer_revision': result.designer_update.policy_after.revision}
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
