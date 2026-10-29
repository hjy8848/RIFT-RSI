"""Resumable orchestration; transition methods never invoke a trainer or model."""
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import json
from rift.schemas import Checkpoint, Tokens
from rift.tasks import Certification, CertificationStatus
from rift.training.contracts import CompletedUpdate, Rollout, UpdateRequest
from rift.training.records import DesignerFeedback, comparable_designer_rollouts, prepare_designer_update, replay_source
from rift.replay.reward import ReplayReward, SampleScore

class Stage(str, Enum):
    PROPOSED = 'PROPOSED'
    CERTIFIED = 'CERTIFIED'
    READY_FOR_TRAINING = 'READY_FOR_TRAINING'
    SERVICE_TRAINING_COMPLETED = 'SERVICE_TRAINING_COMPLETED'
    REPLAY_SCORED = 'REPLAY_SCORED'
    DESIGNER_READY = 'DESIGNER_READY_FOR_TRAINING'
    DESIGNER_TRAINING_COMPLETED = 'DESIGNER_TRAINING_COMPLETED'

@dataclass(frozen=True)
class LoopState:
    package_id: str
    proposal_ids: tuple[str, ...]
    stage: Stage = Stage.PROPOSED
    task_refs: tuple[str, ...] = ()
    service_request: UpdateRequest | None = None
    service_update: CompletedUpdate | None = None
    replay_reward: ReplayReward | None = None
    designer_request: UpdateRequest | None = None
    designer_update: CompletedUpdate | None = None
    designer_feedback: tuple[DesignerFeedback, ...] = ()
    expected_service: Checkpoint | None = None
    expected_designer: Checkpoint | None = None
    previous_updates: tuple[CompletedUpdate, ...] = ()
    def __post_init__(self):
        if not self.package_id or not self.proposal_ids or len(set(self.proposal_ids)) != len(self.proposal_ids): raise ValueError('distinct proposals and package required')
        if (self.expected_service is None) != (self.expected_designer is None): raise ValueError('both prior policy checkpoints are required')
        if self.expected_service is None and self.previous_updates: raise ValueError('unexpected prior updates')
        if self.expected_service is not None and (len(self.previous_updates) != 2 or self.previous_updates[0].policy_after != self.expected_service or self.previous_updates[1].policy_after != self.expected_designer): raise ValueError('prior updates do not bind the next checkpoints')
        if self.stage is Stage.PROPOSED and self.task_refs: raise ValueError('proposed state cannot contain tasks')
        if self.stage is not Stage.PROPOSED and (len(self.task_refs) != len(self.proposal_ids) or any(not t for t in self.task_refs)): raise ValueError('certified task refs required')
        if len(set(self.task_refs)) != len(self.task_refs): raise ValueError('certified tasks must be distinct')
        stages = list(Stage); index = stages.index(self.stage)
        for start, value in ((Stage.READY_FOR_TRAINING,self.service_request), (Stage.SERVICE_TRAINING_COMPLETED,self.service_update), (Stage.REPLAY_SCORED,self.replay_reward), (Stage.DESIGNER_READY,self.designer_request), (Stage.DESIGNER_TRAINING_COMPLETED,self.designer_update)):
            if (index >= stages.index(start)) != (value is not None): raise ValueError('state does not match stored evidence')
        if self.service_request and (self.service_request.package_id != self.package_id or any(r.task_id not in self.task_refs or r.reward_source != 'business_outcome_v2' or not r.response_id or r.reward_ref != r.episode_id for r in self.service_request.rollouts)): raise ValueError('invalid Service request binding')
        if self.service_request and {r.task_id for r in self.service_request.rollouts} != set(self.task_refs): raise ValueError('Service request does not cover the certified package')
        if self.expected_service is not None and self.service_request is not None and self.service_request.policy_before != self.expected_service: raise ValueError('Service request does not continue the prior checkpoint')
        if self.service_update and self.service_update.request != self.service_request: raise ValueError('Service update differs from request')
        if self.replay_reward and (self.replay_reward.package_id != self.package_id or self.replay_reward.update_run_id != self.service_update.run_id): raise ValueError('replay reward differs from update')
        if self.designer_request:
            if not self.designer_feedback: raise ValueError('Designer batch evidence required')
            if self.expected_designer is not None and self.designer_request.policy_before != self.expected_designer: raise ValueError('Designer request does not continue the prior checkpoint')
            if self.designer_request.policy_before.policy == self.service_request.policy_before.policy: raise ValueError("Service and Designer policies must be distinct")
            comparable_designer_rollouts(self.designer_request.rollouts)
            reward = self.replay_reward
            expected = replay_source(self.service_update.request.policy_before, reward.snapshot_digest, reward.normalization)
            own = [r for r in self.designer_request.rollouts if r.task_id == self.package_id]
            if len(own) != 1 or any(r.reward_ref != self.service_update.run_id or not r.response_id or r.reward_source != expected or r.reward != reward.mean for r in own): raise ValueError('Designer credit differs from replay result')
        if self.designer_update and self.designer_update.request != self.designer_request: raise ValueError('Designer update differs from request')
        if self.designer_request is None and self.designer_feedback: raise ValueError('Designer batch evidence without request')
    def certify(self, results):
        if self.stage is not Stage.PROPOSED or len(results) != len(self.proposal_ids): raise ValueError('invalid certification transition')
        if any(not isinstance(r, Certification) for r in results) or tuple(r.proposal_id for r in results) != self.proposal_ids: raise ValueError('certification must bind each proposal in order')
        if any(r.status is not CertificationStatus.CERTIFIED for r in results): raise ValueError('uncertified proposal')
        return self._next(Stage.CERTIFIED, task_refs=tuple(r.task_ref for r in results))
    def ready(self, request):
        if self.stage is not Stage.CERTIFIED or not isinstance(request, UpdateRequest): raise ValueError('invalid training request')
        return self._next(Stage.READY_FOR_TRAINING, service_request=request)
    def service_completed(self, update, *, update_verifier):
        if self.stage is not Stage.READY_FOR_TRAINING or not isinstance(update, CompletedUpdate): raise ValueError('completed update required')
        if update_verifier.verify_update(update) is not True: raise ValueError('Service update evidence was not verified')
        return self._next(Stage.SERVICE_TRAINING_COMPLETED, service_update=update)
    def replay_scored(self, reward, *, reward_verifier):
        if self.stage is not Stage.SERVICE_TRAINING_COMPLETED or not isinstance(reward, ReplayReward): raise ValueError('replay result required')
        if reward_verifier.verify_reward(reward, self.service_update) is not True: raise ValueError('Replay reward evidence was not verified')
        return self._next(Stage.REPLAY_SCORED, replay_reward=reward)
    def designer_ready(self, request, feedback, *, update_verifier, reward_verifier, action_verifier):
        if self.stage is not Stage.REPLAY_SCORED or not isinstance(request, UpdateRequest): raise ValueError('Designer request required')
        feedback = tuple(feedback)
        rebuilt = prepare_designer_update(request.package_id, request.policy_before, feedback,
            update_verifier=update_verifier, reward_verifier=reward_verifier, action_verifier=action_verifier, algorithm=request.algorithm)
        if rebuilt != request: raise ValueError('Designer request differs from verified batch evidence')
        return self._next(Stage.DESIGNER_READY, designer_request=request, designer_feedback=feedback)
    def designer_completed(self, update, *, update_verifier):
        if self.stage is not Stage.DESIGNER_READY or not isinstance(update, CompletedUpdate): raise ValueError('Designer update required')
        if update_verifier.verify_update(update) is not True: raise ValueError('Designer update evidence was not verified')
        return self._next(Stage.DESIGNER_TRAINING_COMPLETED, designer_update=update)
    def start_next_round(self, package_id, proposal_ids, *, update_verifier):
        if self.stage is not Stage.DESIGNER_TRAINING_COMPLETED:
            raise ValueError('both policy updates must be completed before the next round')
        if (update_verifier.verify_update(self.service_update) is not True
                or update_verifier.verify_update(self.designer_update) is not True):
            raise ValueError('prior policy update evidence was not verified')
        return LoopState(package_id, tuple(proposal_ids),
            expected_service=self.service_update.policy_after,
            expected_designer=self.designer_update.policy_after,
            previous_updates=(self.service_update, self.designer_update))
    def _next(self, stage, **updates):
        from dataclasses import replace
        return replace(self, stage=stage, **updates)
    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, sort_keys=True), encoding='utf-8')
    @classmethod
    def load(cls, path, *, update_verifier=None, reward_verifier=None, action_verifier=None):
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        def checkpoint(v): return Checkpoint(**v)
        def request(v):
            if v is None: return None
            rows = []
            for r in v['rollouts']:
                t = r['tokens']; tok = Tokens(tuple(t['prompt']), tuple(t['response']), tuple(t['mask']))
                rows.append(Rollout(tok, checkpoint(r['policy']), r['task_id'], r['episode_id'], r['reward'], r['reward_source'], r['group_id'], r.get('response_id'), r.get('reward_ref')))
            return UpdateRequest(checkpoint(v['policy_before']), v['package_id'], tuple(rows), v['algorithm'])
        def update(v):
            return None if v is None else CompletedUpdate(request(v['request']), checkpoint(v['policy_after']), v['run_id'], v['evidence_ref'])
        def replay_reward(value):
            if value is None: return None
            return ReplayReward(value['package_id'], value['update_run_id'], value['snapshot_digest'], value['binding_digest'], value['normalization'], tuple(SampleScore(**r) for r in value['samples']), value['mean'])
        feedback = []
        for item in d.get('designer_feedback', []):
            t = item['tokens']
            feedback.append(DesignerFeedback(Tokens(tuple(t['prompt']), tuple(t['response']), tuple(t['mask'])), update(item['update']), replay_reward(item['reward']), item['designer_rollout_id'], item['group_id']))
        state = cls(d['package_id'], tuple(d['proposal_ids']), Stage(d['stage']), tuple(d['task_refs']), request(d['service_request']), update(d['service_update']), replay_reward(d['replay_reward']), request(d['designer_request']), update(d['designer_update']), tuple(feedback), checkpoint(d['expected_service']) if d.get('expected_service') else None, checkpoint(d['expected_designer']) if d.get('expected_designer') else None, tuple(update(item) for item in d.get('previous_updates', [])))
        for completed in (state.service_update, state.designer_update, *state.previous_updates, *(item.update for item in state.designer_feedback)):
            if completed is not None and (update_verifier is None or update_verifier.verify_update(completed) is not True):
                raise ValueError('persisted update evidence was not verified')
        reward_pairs = ([] if state.replay_reward is None else [(state.replay_reward, state.service_update)])
        reward_pairs.extend((item.reward, item.update) for item in state.designer_feedback)
        for recorded_reward, completed in reward_pairs:
            if reward_verifier is None or reward_verifier.verify_reward(recorded_reward, completed) is not True:
                raise ValueError('persisted Replay reward evidence was not verified')
        if state.designer_request is not None:
            if action_verifier is None: raise ValueError('persisted Designer action evidence was not verified')
            rebuilt = prepare_designer_update(state.designer_request.package_id, state.designer_request.policy_before,
                state.designer_feedback, update_verifier=update_verifier, reward_verifier=reward_verifier, action_verifier=action_verifier, algorithm=state.designer_request.algorithm)
            if rebuilt != state.designer_request: raise ValueError('persisted Designer batch differs from evidence')
        return state
