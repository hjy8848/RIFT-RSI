"""Resumable orchestration; transition methods never invoke a trainer or model."""
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import json
from rift.schemas import Checkpoint, Tokens
from rift.tasks import Certification, CertificationStatus
from rift.training.contracts import CompletedUpdate, Rollout, UpdateRequest
from rift.training.records import comparable_designer_rollouts, replay_source
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
    def __post_init__(self):
        if not self.package_id or not self.proposal_ids or len(set(self.proposal_ids)) != len(self.proposal_ids): raise ValueError('distinct proposals and package required')
        if self.stage is Stage.PROPOSED and self.task_refs: raise ValueError('proposed state cannot contain tasks')
        if self.stage is not Stage.PROPOSED and (len(self.task_refs) != len(self.proposal_ids) or any(not t for t in self.task_refs)): raise ValueError('certified task refs required')
        stages = list(Stage); index = stages.index(self.stage)
        for start, value in ((Stage.READY_FOR_TRAINING,self.service_request), (Stage.SERVICE_TRAINING_COMPLETED,self.service_update), (Stage.REPLAY_SCORED,self.replay_reward), (Stage.DESIGNER_READY,self.designer_request), (Stage.DESIGNER_TRAINING_COMPLETED,self.designer_update)):
            if (index >= stages.index(start)) != (value is not None): raise ValueError('state does not match stored evidence')
        if self.service_request and (self.service_request.package_id != self.package_id or any(r.task_id not in self.task_refs or r.reward_source != 'business_outcome_v2' for r in self.service_request.rollouts)): raise ValueError('invalid Service request binding')
        if self.service_update and self.service_update.request != self.service_request: raise ValueError('Service update differs from request')
        if self.replay_reward and (self.replay_reward.package_id != self.package_id or self.replay_reward.update_run_id != self.service_update.run_id): raise ValueError('replay reward differs from update')
        if self.designer_request:
            if self.designer_request.policy_before.policy == self.service_request.policy_before.policy: raise ValueError("Service and Designer policies must be distinct")
            comparable_designer_rollouts(self.designer_request.rollouts)
            reward = self.replay_reward
            expected = replay_source(self.service_update.request.policy_before, reward.snapshot_digest, reward.normalization)
            if self.designer_request.package_id != self.package_id or any(r.task_id != self.package_id or r.episode_id != self.service_update.run_id or r.reward_source != expected or r.reward != reward.mean for r in self.designer_request.rollouts): raise ValueError('Designer credit differs from replay result')
        if self.designer_update and self.designer_update.request != self.designer_request: raise ValueError('Designer update differs from request')
    def certify(self, results):
        if self.stage is not Stage.PROPOSED or len(results) != len(self.proposal_ids): raise ValueError('invalid certification transition')
        if any(not isinstance(r, Certification) for r in results) or tuple(r.proposal_id for r in results) != self.proposal_ids: raise ValueError('certification must bind each proposal in order')
        if any(r.status is not CertificationStatus.CERTIFIED for r in results): raise ValueError('uncertified proposal')
        return self._next(Stage.CERTIFIED, task_refs=tuple(r.task_ref for r in results))
    def ready(self, request):
        if self.stage is not Stage.CERTIFIED or not isinstance(request, UpdateRequest): raise ValueError('invalid training request')
        return self._next(Stage.READY_FOR_TRAINING, service_request=request)
    def service_completed(self, update):
        if self.stage is not Stage.READY_FOR_TRAINING or not isinstance(update, CompletedUpdate): raise ValueError('completed update required')
        return self._next(Stage.SERVICE_TRAINING_COMPLETED, service_update=update)
    def replay_scored(self, reward):
        if self.stage is not Stage.SERVICE_TRAINING_COMPLETED or not isinstance(reward, ReplayReward): raise ValueError('replay result required')
        return self._next(Stage.REPLAY_SCORED, replay_reward=reward)
    def designer_ready(self, request):
        if self.stage is not Stage.REPLAY_SCORED or not isinstance(request, UpdateRequest): raise ValueError('Designer request required')
        return self._next(Stage.DESIGNER_READY, designer_request=request)
    def designer_completed(self, update):
        if self.stage is not Stage.DESIGNER_READY or not isinstance(update, CompletedUpdate): raise ValueError('Designer update required')
        return self._next(Stage.DESIGNER_TRAINING_COMPLETED, designer_update=update)
    def _next(self, stage, **updates):
        from dataclasses import replace
        return replace(self, stage=stage, **updates)
    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, sort_keys=True), encoding='utf-8')
    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        def checkpoint(v): return Checkpoint(**v)
        def request(v):
            if v is None: return None
            rows = []
            for r in v['rollouts']:
                t = r['tokens']; tok = Tokens(tuple(t['prompt']), tuple(t['response']), tuple(t['mask']))
                rows.append(Rollout(tok, checkpoint(r['policy']), r['task_id'], r['episode_id'], r['reward'], r['reward_source'], r['group_id']))
            return UpdateRequest(checkpoint(v['policy_before']), v['package_id'], tuple(rows), v['algorithm'])
        def update(v):
            return None if v is None else CompletedUpdate(request(v['request']), checkpoint(v['policy_after']), v['run_id'], v['evidence_ref'])
        reward = d['replay_reward']
        if reward is not None:
            reward = ReplayReward(reward['package_id'], reward['update_run_id'], reward['snapshot_digest'], reward['binding_digest'], reward['normalization'], tuple(SampleScore(**r) for r in reward['samples']), reward['mean'])
        return cls(d['package_id'], tuple(d['proposal_ids']), Stage(d['stage']), tuple(d['task_refs']), request(d['service_request']), update(d['service_update']), reward, request(d['designer_request']), update(d['designer_update']))
