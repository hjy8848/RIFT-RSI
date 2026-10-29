"""Versioned, resumable state for legacy Replay LogP and dual-bank rounds."""
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
import json

from rift.schemas import Checkpoint, Tokens
from rift.tasks import Certification, CertificationStatus
from rift.training.contracts import CompletedUpdate, Rollout, UpdateRequest
from rift.training.records import (DesignerFeedback, comparable_designer_rollouts,
    designer_reward_source, prepare_designer_update, reward_scalar)
from rift.replay.reward import ReplayReward, SampleScore
from rift.rewards.learning_value import DualBankOutcomeReward, RewardMode
from rift.probes.records import ProbeResultSet, ProbeSnapshot
from rift.probes.storage import _result_set, _snapshot
from rift.training.package import TrainingPackage


class Stage(str, Enum):
    PROPOSED = 'PROPOSED'
    CERTIFIED = 'CERTIFIED'
    PROBES_FROZEN = 'PROBES_FROZEN'
    BASELINE_READY = 'BASELINE_READY'
    READY_FOR_TRAINING = 'READY_FOR_TRAINING'
    SERVICE_TRAINING_COMPLETED = 'SERVICE_TRAINING_COMPLETED'
    PROBES_EVALUATED = 'PROBES_EVALUATED'
    REPLAY_SCORED = 'REPLAY_SCORED'
    LEARNING_REWARDED = 'LEARNING_REWARDED'
    DESIGNER_READY = 'DESIGNER_READY_FOR_TRAINING'
    DESIGNER_TRAINING_COMPLETED = 'DESIGNER_TRAINING_COMPLETED'


_LEGACY_ORDER = (Stage.PROPOSED, Stage.CERTIFIED, Stage.READY_FOR_TRAINING,
    Stage.SERVICE_TRAINING_COMPLETED, Stage.REPLAY_SCORED, Stage.DESIGNER_READY,
    Stage.DESIGNER_TRAINING_COMPLETED)
_DUAL_ORDER = (Stage.PROPOSED, Stage.CERTIFIED, Stage.PROBES_FROZEN,
    Stage.BASELINE_READY, Stage.READY_FOR_TRAINING, Stage.SERVICE_TRAINING_COMPLETED,
    Stage.PROBES_EVALUATED, Stage.LEARNING_REWARDED, Stage.DESIGNER_READY,
    Stage.DESIGNER_TRAINING_COMPLETED)


@dataclass(frozen=True)
class PendingDesignerAction:
    checkpoint: Checkpoint
    rollout_id: str
    tokens: Tokens
    group_id: str | None = None

    def __post_init__(self):
        if not self.rollout_id or not isinstance(self.tokens, Tokens):
            raise ValueError('captured on-policy Designer action required')


@dataclass(frozen=True)
class LoopState:
    package_id: str
    proposal_ids: tuple[str, ...]
    stage: Stage = Stage.PROPOSED
    target_ability_id: str | None = None
    round_index: int = 0
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
    protocol_version: int = 2
    reward_mode: str = RewardMode.REPLAY_LOGP_V1.value
    recovery_snapshot: ProbeSnapshot | None = None
    retention_snapshot: ProbeSnapshot | None = None
    recovery_before: ProbeResultSet | None = None
    retention_before: ProbeResultSet | None = None
    recovery_after: ProbeResultSet | None = None
    retention_after: ProbeResultSet | None = None
    learning_reward: DualBankOutcomeReward | None = None
    designer_action: PendingDesignerAction | None = None
    training_package: TrainingPackage | None = None

    def __post_init__(self):
        if not self.package_id or not self.proposal_ids or len(set(self.proposal_ids)) != len(self.proposal_ids):
            raise ValueError('distinct proposals and package required')
        if ((self.target_ability_id is not None and not self.target_ability_id)
                or type(self.round_index) is not int or self.round_index < 0):
            raise ValueError('valid target ability ID and non-negative round index required')
        if self.protocol_version not in (2, 3):
            raise ValueError('unsupported LoopState protocol version')
        if self.protocol_version == 3:
            if (self.reward_mode != RewardMode.DUAL_BANK_OUTCOME_V1.value
                    or not isinstance(self.training_package, TrainingPackage)
                    or self.proposal_ids != (self.training_package.proposal_id,)
                    or self.package_id != self.training_package.package_id
                    or self.target_ability_id != self.training_package.target_ability_id
                    or self.designer_action is None
                    or self.designer_action.rollout_id != self.training_package.designer_action_ref):
                raise ValueError('package protocol must bind one dual-bank Designer decision')
        elif self.training_package is not None:
            raise ValueError('version 2 states cannot contain package evidence')
        if self.reward_mode not in (RewardMode.REPLAY_LOGP_V1.value, RewardMode.DUAL_BANK_OUTCOME_V1.value):
            raise ValueError('unknown Designer reward mode')
        if self.designer_action is not None and not isinstance(self.designer_action, PendingDesignerAction):
            raise ValueError('invalid pending Designer action')
        if (self.expected_service is None) != (self.expected_designer is None):
            raise ValueError('both prior policy checkpoints are required')
        if self.expected_service is None and self.previous_updates:
            raise ValueError('unexpected prior updates')
        if self.expected_service is not None and (len(self.previous_updates) != 2
                or self.previous_updates[0].policy_after != self.expected_service
                or self.previous_updates[1].policy_after != self.expected_designer):
            raise ValueError('prior updates do not bind the next checkpoints')
        if self.stage is Stage.PROPOSED and self.task_refs:
            raise ValueError('proposed state cannot contain tasks')
        if self.stage is not Stage.PROPOSED:
            if self.protocol_version == 3:
                if self.task_refs != self.training_package.task_refs:
                    raise ValueError('package state task refs differ from frozen package')
            elif len(self.task_refs) != len(self.proposal_ids):
                raise ValueError('legacy certified task refs must bind each proposal')
            if any(not t for t in self.task_refs):
                raise ValueError('certified task refs required')
        if len(set(self.task_refs)) != len(self.task_refs):
            raise ValueError('certified tasks must be distinct')
        order = _DUAL_ORDER if self.reward_mode == RewardMode.DUAL_BANK_OUTCOME_V1.value else _LEGACY_ORDER
        if self.stage not in order:
            raise ValueError('stage is not valid for the selected reward protocol')
        index = order.index(self.stage)
        if self.reward_mode == RewardMode.REPLAY_LOGP_V1.value:
            for start, value in ((Stage.READY_FOR_TRAINING, self.service_request),
                (Stage.SERVICE_TRAINING_COMPLETED, self.service_update),
                (Stage.REPLAY_SCORED, self.replay_reward), (Stage.DESIGNER_READY, self.designer_request),
                (Stage.DESIGNER_TRAINING_COMPLETED, self.designer_update)):
                if (index >= order.index(start)) != (value is not None):
                    raise ValueError('legacy state does not match stored replay evidence')
            if any((x is not None for x in (self.recovery_snapshot, self.retention_snapshot,
                    self.recovery_before, self.retention_before, self.recovery_after,
                    self.retention_after, self.learning_reward))):
                raise ValueError('dual-bank evidence cannot appear in a Replay LogP state')
        else:
            if self.replay_reward is not None:
                raise ValueError('Replay LogP reward cannot be reinterpreted as dual-bank reward')
            for start, value in ((Stage.PROBES_FROZEN, (self.recovery_snapshot, self.retention_snapshot)),
                (Stage.BASELINE_READY, (self.recovery_before, self.retention_before)),
                (Stage.SERVICE_TRAINING_COMPLETED, self.service_update),
                (Stage.PROBES_EVALUATED, (self.recovery_after, self.retention_after)),
                (Stage.LEARNING_REWARDED, self.learning_reward),
                (Stage.DESIGNER_READY, self.designer_request),
                (Stage.DESIGNER_TRAINING_COMPLETED, self.designer_update)):
                present = all(v is not None for v in value) if isinstance(value, tuple) else value is not None
                if (index >= order.index(start)) != present:
                    raise ValueError('dual-bank state does not match stored probe evidence')
            if self.recovery_snapshot and not self.recovery_snapshot.verify():
                raise ValueError('invalid Recovery snapshot or baseline binding')
            if self.retention_snapshot and not self.retention_snapshot.verify():
                raise ValueError('invalid Retention snapshot or baseline binding')
            if self.recovery_before and self.recovery_before.snapshot_digest != self.recovery_snapshot.content_digest:
                raise ValueError('Recovery baseline differs from frozen snapshot')
            if self.retention_before and self.retention_before.snapshot_digest != self.retention_snapshot.content_digest:
                raise ValueError('Retention baseline differs from frozen snapshot')
            if self.recovery_after and (self.recovery_after.snapshot_digest != self.recovery_snapshot.content_digest
                    or self.recovery_after.phase != 'after'):
                raise ValueError('Recovery after result differs from frozen snapshot')
            if self.retention_after and (self.retention_after.snapshot_digest != self.retention_snapshot.content_digest
                    or self.retention_after.phase != 'after'):
                raise ValueError('Retention after result differs from frozen snapshot')
            if self.learning_reward and (self.learning_reward.package_id != self.package_id
                    or self.learning_reward.service_update_run_id != self.service_update.run_id
                    or self.learning_reward.recovery_snapshot_digest != self.recovery_snapshot.content_digest
                    or self.learning_reward.retention_snapshot_digest != self.retention_snapshot.content_digest
                    or self.learning_reward.recovery_before_evidence != self.recovery_before.evidence_digest
                    or self.learning_reward.recovery_after_evidence != self.recovery_after.evidence_digest
                    or self.learning_reward.retention_before_evidence != self.retention_before.evidence_digest
                    or self.learning_reward.retention_after_evidence != self.retention_after.evidence_digest):
                raise ValueError('dual-bank reward does not bind stored Probe evidence')
            if index >= order.index(Stage.PROBES_FROZEN) and self.designer_action is None:
                raise ValueError('dual-bank package must retain its original on-policy Designer action')
        self._validate_updates_and_feedback()

    def _validate_updates_and_feedback(self):
        if self.service_request and (self.service_request.package_id != self.package_id
                or any(r.task_id not in self.task_refs or r.reward_source != 'business_outcome_v2'
                       or not r.response_id or r.reward_ref != r.episode_id
                       for r in self.service_request.rollouts)):
            raise ValueError('invalid Service request binding')
        if self.service_request:
            service_refs = {r.task_id for r in self.service_request.rollouts}
            if (not service_refs or (self.training_package is None
                    and service_refs != set(self.task_refs))
                    or not service_refs.issubset(set(self.task_refs))):
                raise ValueError('Service request does not cover the certified package')
        if self.training_package is not None:
            if self.stage.value in {Stage.READY_FOR_TRAINING.value,
                    Stage.SERVICE_TRAINING_COMPLETED.value, Stage.PROBES_EVALUATED.value,
                    Stage.LEARNING_REWARDED.value, Stage.DESIGNER_READY.value,
                    Stage.DESIGNER_TRAINING_COMPLETED.value}:
                if (self.service_request is None
                        or not self.training_package.verify_service_request(self.service_request)):
                    raise ValueError('persisted package statistics differ from its Service GRPO request')
        if self.expected_service is not None and self.service_request is not None and self.service_request.policy_before != self.expected_service:
            raise ValueError('Service request does not continue the prior checkpoint')
        if self.service_update and self.service_update.request != self.service_request:
            raise ValueError('Service update differs from request')
        if self.replay_reward and (self.replay_reward.package_id != self.package_id
                or self.replay_reward.update_run_id != self.service_update.run_id):
            raise ValueError('replay reward differs from update')
        if self.designer_request:
            if not self.designer_feedback:
                raise ValueError('Designer batch evidence required')
            if self.expected_designer is not None and self.designer_request.policy_before != self.expected_designer:
                raise ValueError('Designer request does not continue the prior checkpoint')
            if self.designer_request.policy_before.policy == self.service_request.policy_before.policy:
                raise ValueError('Service and Designer policies must be distinct')
            if (self.reward_mode == RewardMode.DUAL_BANK_OUTCOME_V1.value
                    and (self.designer_action is None
                         or self.designer_request.policy_before != self.designer_action.checkpoint)):
                raise ValueError('dual-bank update must use the checkpoint that generated its package')
            comparable_designer_rollouts(self.designer_request.rollouts,
                algorithm=self.designer_request.algorithm,
                credit_assignment_mode=self.designer_request.credit_assignment_mode)
            if self.reward_mode == RewardMode.REPLAY_LOGP_V1.value:
                reward = self.replay_reward
                expected = designer_reward_source(reward, self.service_update)
            else:
                reward = self.learning_reward
                expected = designer_reward_source(reward, self.service_update)
            own = [r for r in self.designer_request.rollouts if r.task_id == self.package_id]
            if (len(own) != 1 or any(r.reward_ref != self.service_update.run_id
                    or not r.response_id or r.reward_source != expected
                    or r.reward != reward_scalar(reward)
                    or (self.reward_mode == RewardMode.DUAL_BANK_OUTCOME_V1.value
                        and r.target_ability_id != self.target_ability_id)
                    for r in own)):
                raise ValueError('Designer credit differs from verified learning reward')
        if self.designer_update and self.designer_update.request != self.designer_request:
            raise ValueError('Designer update differs from request')
        if self.designer_request is None and self.designer_feedback:
            raise ValueError('Designer feedback without request')

    def certify(self, results):
        if self.stage is not Stage.PROPOSED or len(results) != len(self.proposal_ids):
            raise ValueError('invalid certification transition')
        if any(not isinstance(r, Certification) for r in results) or tuple(r.proposal_id for r in results) != self.proposal_ids:
            raise ValueError('certification must bind each proposal in order')
        if any(r.status is not CertificationStatus.CERTIFIED for r in results):
            raise ValueError('uncertified proposal')
        return self._next(Stage.CERTIFIED, task_refs=tuple(r.task_ref for r in results))

    def certify_training_package(self, package, designer_certifications):
        """Bind one captured Designer choice to its certified tasks and frozen mix."""
        if self.stage is not Stage.PROPOSED or not isinstance(package, TrainingPackage):
            raise ValueError('a proposed state and frozen TrainingPackage are required')
        rows = tuple(designer_certifications)
        if (self.proposal_ids != (package.proposal_id,)
                or package.package_id != self.package_id
                or package.target_ability_id != self.target_ability_id
                or self.designer_action is None
                or self.designer_action.rollout_id != package.designer_action_ref
                or self.reward_mode not in (RewardMode.REPLAY_LOGP_V1.value,
                                             RewardMode.DUAL_BANK_OUTCOME_V1.value)
                or package.stats is not None
                or len(rows) < len(package.designer_task_refs)
                or len({row.task_ref for row in rows if isinstance(row, Certification)}) != len(rows)
                or any(not isinstance(row, Certification)
                       or row.proposal_id != package.proposal_id
                       or row.status is not CertificationStatus.CERTIFIED
                       for row in rows)
                or tuple(row.task_ref for row in rows[:len(package.designer_task_refs)])
                    != package.designer_task_refs):
            raise ValueError('Designer certifications do not bind the one frozen task package')
        return self._next(Stage.CERTIFIED, task_refs=package.task_refs,
            protocol_version=3, reward_mode=RewardMode.DUAL_BANK_OUTCOME_V1.value,
            training_package=package)

    def freeze_probes(self, recovery, retention, *, snapshot_verifier):
        if (self.stage is not Stage.CERTIFIED
                or (self.reward_mode != RewardMode.REPLAY_LOGP_V1.value
                    and not (self.protocol_version == 3
                             and self.reward_mode == RewardMode.DUAL_BANK_OUTCOME_V1.value))):
            raise ValueError('Probe snapshots can only be frozen once after certification')
        if self.designer_action is None:
            raise ValueError('dual-bank credit requires the captured on-policy Designer action')
        if self.designer_action.group_id is not None:
            raise ValueError('sequential dual-bank Designer feedback cannot use a fabricated GRPO group')
        if (not isinstance(recovery, ProbeSnapshot) or not isinstance(retention, ProbeSnapshot)
                or not recovery.verify() or not retention.verify()
                or recovery.baseline_checkpoint != retention.baseline_checkpoint
                or recovery.target_ability_id != self.target_ability_id
                or retention.target_ability_id is not None
                or (self.expected_service is not None and recovery.baseline_checkpoint != self.expected_service)
                or snapshot_verifier.verify_snapshot(recovery) is not True
                or snapshot_verifier.verify_snapshot(retention) is not True):
            raise ValueError('frozen Probe evidence failed verification')
        return self._next(Stage.PROBES_FROZEN,
            reward_mode=RewardMode.DUAL_BANK_OUTCOME_V1.value,
            recovery_snapshot=recovery, retention_snapshot=retention)

    def baseline_ready(self, recovery_before, retention_before, *, probe_verifier):
        if self.reward_mode != RewardMode.DUAL_BANK_OUTCOME_V1.value or self.stage is not Stage.PROBES_FROZEN:
            raise ValueError('dual-bank frozen snapshots required before baseline')
        for snapshot, result in ((self.recovery_snapshot, recovery_before),
                                 (self.retention_snapshot, retention_before)):
            if (not isinstance(result, ProbeResultSet) or result.phase != 'before'
                    or result.snapshot_digest != snapshot.content_digest
                    or result.checkpoint != snapshot.baseline_checkpoint
                    or probe_verifier.verify_result(result, snapshot) is not True):
                raise ValueError('before-update Probe baseline evidence failed verification')
        return self._next(Stage.BASELINE_READY, recovery_before=recovery_before,
                          retention_before=retention_before)

    def ready(self, request):
        expected = Stage.BASELINE_READY if self.reward_mode == RewardMode.DUAL_BANK_OUTCOME_V1.value else Stage.CERTIFIED
        if self.stage is not expected or not isinstance(request, UpdateRequest):
            raise ValueError('invalid training request transition')
        if self.expected_service is not None and request.policy_before != self.expected_service:
            raise ValueError('Service request does not use the prior checkpoint')
        if (self.reward_mode == RewardMode.DUAL_BANK_OUTCOME_V1.value
                and request.policy_before != self.recovery_snapshot.baseline_checkpoint):
            raise ValueError('Probe baseline checkpoint differs from Service training start')
        package = self.training_package
        if package is not None:
            package = package.record_service_request(request)
        return self._next(Stage.READY_FOR_TRAINING, service_request=request,
                          training_package=package)

    def service_completed(self, update, *, update_verifier):
        if self.stage is not Stage.READY_FOR_TRAINING or not isinstance(update, CompletedUpdate):
            raise ValueError('completed update required')
        if update_verifier.verify_update(update) is not True:
            raise ValueError('Service update evidence was not verified')
        return self._next(Stage.SERVICE_TRAINING_COMPLETED, service_update=update)

    def replay_scored(self, reward, *, reward_verifier):
        if (self.reward_mode != RewardMode.REPLAY_LOGP_V1.value
                or self.stage is not Stage.SERVICE_TRAINING_COMPLETED
                or not isinstance(reward, ReplayReward)):
            raise ValueError('Replay LogP result required for this reward mode')
        if reward_verifier.verify_reward(reward, self.service_update) is not True:
            raise ValueError('Replay reward evidence was not verified')
        return self._next(Stage.REPLAY_SCORED, replay_reward=reward)

    def probes_evaluated(self, recovery_after, retention_after, *, probe_verifier):
        if self.reward_mode != RewardMode.DUAL_BANK_OUTCOME_V1.value or self.stage is not Stage.SERVICE_TRAINING_COMPLETED:
            raise ValueError('Service update must complete before after-Probe evaluation')
        for snapshot, result in ((self.recovery_snapshot, recovery_after),
                                 (self.retention_snapshot, retention_after)):
            if (not isinstance(result, ProbeResultSet) or result.phase != 'after'
                    or result.snapshot_digest != snapshot.content_digest
                    or result.checkpoint != self.service_update.policy_after
                    or probe_verifier.verify_result(result, snapshot) is not True):
                raise ValueError('after-update Probe evidence failed verification')
        return self._next(Stage.PROBES_EVALUATED, recovery_after=recovery_after,
                          retention_after=retention_after)

    def learning_rewarded(self, reward, *, reward_verifier):
        if (self.reward_mode != RewardMode.DUAL_BANK_OUTCOME_V1.value
                or self.stage is not Stage.PROBES_EVALUATED
                or not isinstance(reward, DualBankOutcomeReward)):
            raise ValueError('evaluated dual-bank evidence required')
        if reward_verifier.verify_reward(reward, self.service_update) is not True:
            raise ValueError('dual-bank reward evidence was not verified')
        return self._next(Stage.LEARNING_REWARDED, learning_reward=reward)

    def designer_ready(self, request, feedback, *, update_verifier, reward_verifier,
                       action_verifier):
        expected = Stage.LEARNING_REWARDED if self.reward_mode == RewardMode.DUAL_BANK_OUTCOME_V1.value else Stage.REPLAY_SCORED
        if self.stage is not expected or not isinstance(request, UpdateRequest):
            raise ValueError('verified learning reward required before Designer training')
        feedback = tuple(feedback)
        if self.reward_mode == RewardMode.DUAL_BANK_OUTCOME_V1:
            if (self.designer_action is None or len(feedback) != 1
                    or feedback[0].designer_rollout_id != self.designer_action.rollout_id
                    or feedback[0].tokens != self.designer_action.tokens
                    or feedback[0].group_id != self.designer_action.group_id
                    or feedback[0].update != self.service_update
                    or feedback[0].reward != self.learning_reward
                    or feedback[0].target_ability_id != self.target_ability_id
                    or request.policy_before != self.designer_action.checkpoint):
                raise ValueError('Designer feedback must credit the captured action that generated this package')
        rebuilt = prepare_designer_update(request.package_id, request.policy_before, feedback,
            update_verifier=update_verifier, reward_verifier=reward_verifier,
            action_verifier=action_verifier, algorithm=request.algorithm,
            credit_assignment_mode=request.credit_assignment_mode,
            reward_baseline=request.advantage_baseline)
        if rebuilt != request:
            raise ValueError('Designer request differs from verified feedback batch')
        return self._next(Stage.DESIGNER_READY, designer_request=request,
                          designer_feedback=feedback)

    def designer_completed(self, update, *, update_verifier):
        if self.stage is not Stage.DESIGNER_READY or not isinstance(update, CompletedUpdate):
            raise ValueError('Designer update required')
        if update_verifier.verify_update(update) is not True:
            raise ValueError('Designer update evidence was not verified')
        return self._next(Stage.DESIGNER_TRAINING_COMPLETED, designer_update=update)

    def start_next_round(self, package_id, proposal_ids, *, update_verifier):
        if self.stage is not Stage.DESIGNER_TRAINING_COMPLETED:
            raise ValueError('both policy updates must complete before the next round')
        if (update_verifier.verify_update(self.service_update) is not True
                or update_verifier.verify_update(self.designer_update) is not True):
            raise ValueError('prior policy update evidence was not verified')
        return LoopState(package_id, tuple(proposal_ids),
            target_ability_id=None,
            round_index=self.round_index + 1,
            expected_service=self.service_update.policy_after,
            expected_designer=self.designer_update.policy_after,
            previous_updates=(self.service_update, self.designer_update))

    def _next(self, stage, **updates):
        return replace(self, stage=stage, **updates)

    def save(self, path):
        payload = {"schema_version": 4, **asdict(self)}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding='utf-8')

    @classmethod
    def load(cls, path, *, update_verifier=None, reward_verifier=None,
             action_verifier=None, probe_verifier=None):
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
        if "schema_version" not in raw:
            # Legacy LoopState used only ReplayReward and has no protocol tag.
            raw = {"schema_version": 1, **raw}
        if raw["schema_version"] not in (1, 2, 3, 4):
            raise ValueError('unsupported LoopState schema')

        def checkpoint(value): return None if value is None else Checkpoint(**value)
        def rollout(value):
            token = value['tokens']
            return Rollout(Tokens(tuple(token['prompt']), tuple(token['response']), tuple(token['mask'])),
                checkpoint(value['policy']), value['task_id'], value['episode_id'], value['reward'],
                value['reward_source'], value.get('group_id'), value.get('response_id'),
                value.get('reward_ref'), value.get('target_ability_id'))
        def request(value):
            if value is None: return None
            return UpdateRequest(checkpoint(value['policy_before']), value['package_id'],
                tuple(rollout(row) for row in value['rollouts']), value['algorithm'],
                value.get('advantage_baseline'), value.get('reward_protocol'),
                value.get('credit_assignment_mode'))
        def update(value):
            return None if value is None else CompletedUpdate(request(value['request']),
                checkpoint(value['policy_after']), value['run_id'], value['evidence_ref'])
        def reward(value):
            if value is None: return None
            if value.get('reward_protocol') == RewardMode.DUAL_BANK_OUTCOME_V1.value:
                value = dict(value)
                value['service_before'] = checkpoint(value['service_before'])
                value['service_after'] = checkpoint(value['service_after'])
                return DualBankOutcomeReward(**value)
            return ReplayReward(value['package_id'], value['update_run_id'], value['snapshot_digest'],
                value['binding_digest'], value['normalization'],
                tuple(SampleScore(**row) for row in value['samples']), value['mean'])
        schema = raw['schema_version']
        dual = schema in (2, 3, 4) and raw.get('reward_mode') == RewardMode.DUAL_BANK_OUTCOME_V1.value
        recovery_snapshot = _snapshot(raw['recovery_snapshot']) if dual and raw.get('recovery_snapshot') else None
        retention_snapshot = _snapshot(raw['retention_snapshot']) if dual and raw.get('retention_snapshot') else None
        recovery_before = _result_set(raw['recovery_before']) if dual and raw.get('recovery_before') else None
        retention_before = _result_set(raw['retention_before']) if dual and raw.get('retention_before') else None
        recovery_after = _result_set(raw['recovery_after']) if dual and raw.get('recovery_after') else None
        retention_after = _result_set(raw['retention_after']) if dual and raw.get('retention_after') else None
        feedback = tuple(DesignerFeedback(Tokens(tuple(row['tokens']['prompt']),
            tuple(row['tokens']['response']), tuple(row['tokens']['mask'])), update(row['update']),
            reward(row['reward']), row['designer_rollout_id'], row.get('group_id'),
            row.get('target_ability_id'))
            for row in raw.get('designer_feedback', ()))
        state = cls(package_id=raw['package_id'], proposal_ids=tuple(raw['proposal_ids']),
            stage=Stage(raw['stage']), task_refs=tuple(raw.get('task_refs', ())),
            target_ability_id=raw.get('target_ability_id'), round_index=raw.get('round_index', 0),
            service_request=request(raw.get('service_request')), service_update=update(raw.get('service_update')),
            replay_reward=reward(raw.get('replay_reward')), designer_request=request(raw.get('designer_request')),
            designer_update=update(raw.get('designer_update')), designer_feedback=feedback,
            expected_service=checkpoint(raw.get('expected_service')),
            expected_designer=checkpoint(raw.get('expected_designer')),
            previous_updates=tuple(update(row) for row in raw.get('previous_updates', ())),
            protocol_version=(raw.get('protocol_version', 2) if schema == 4 else 2),
            reward_mode=(raw.get('reward_mode') if schema in (2, 3, 4) else RewardMode.REPLAY_LOGP_V1.value),
            recovery_snapshot=recovery_snapshot, retention_snapshot=retention_snapshot,
            recovery_before=recovery_before, retention_before=retention_before,
            recovery_after=recovery_after, retention_after=retention_after,
            learning_reward=reward(raw.get('learning_reward')) if dual else None,
            designer_action=(PendingDesignerAction(checkpoint(raw['designer_action']['checkpoint']),
                raw['designer_action']['rollout_id'], Tokens(tuple(raw['designer_action']['tokens']['prompt']),
                    tuple(raw['designer_action']['tokens']['response']), tuple(raw['designer_action']['tokens']['mask'])),
                raw['designer_action'].get('group_id')) if raw.get('designer_action') else None),
            training_package=(TrainingPackage.from_dict(raw['training_package'])
                if schema == 4 and raw.get('training_package') else None))
        updates = (state.service_update, state.designer_update, *state.previous_updates,
                   *(item.update for item in state.designer_feedback))
        for item in updates:
            if item is not None and (update_verifier is None or update_verifier.verify_update(item) is not True):
                raise ValueError('persisted update evidence was not verified')
        for recorded, completed in ((state.replay_reward, state.service_update),
                *((item.reward, item.update) for item in state.designer_feedback)):
            if recorded is not None and (reward_verifier is None or reward_verifier.verify_reward(recorded, completed) is not True):
                raise ValueError('persisted Designer reward evidence was not verified')
        if state.learning_reward is not None:
            if reward_verifier is None or reward_verifier.verify_reward(state.learning_reward, state.service_update) is not True:
                raise ValueError('persisted dual-bank reward evidence was not verified')
        if state.reward_mode == RewardMode.DUAL_BANK_OUTCOME_V1.value and state.recovery_snapshot is not None:
            if probe_verifier is None:
                raise ValueError('persisted Probe snapshot evidence requires a trusted verifier')
            if (probe_verifier.verify_snapshot(state.recovery_snapshot) is not True
                    or probe_verifier.verify_snapshot(state.retention_snapshot) is not True):
                raise ValueError('persisted frozen Probe snapshot was not verified')
            for snapshot, result in ((state.recovery_snapshot, state.recovery_before),
                    (state.retention_snapshot, state.retention_before),
                    (state.recovery_snapshot, state.recovery_after),
                    (state.retention_snapshot, state.retention_after)):
                if snapshot is not None and result is not None and probe_verifier.verify_result(result, snapshot) is not True:
                    raise ValueError('persisted Probe trial evidence was not verified')
        if state.designer_request is not None:
            if action_verifier is None:
                raise ValueError('persisted Designer action evidence was not verified')
            rebuilt = prepare_designer_update(state.designer_request.package_id,
                state.designer_request.policy_before, state.designer_feedback,
                update_verifier=update_verifier, reward_verifier=reward_verifier,
                action_verifier=action_verifier, algorithm=state.designer_request.algorithm,
                credit_assignment_mode=state.designer_request.credit_assignment_mode,
                reward_baseline=state.designer_request.advantage_baseline)
            if rebuilt != state.designer_request:
                raise ValueError('persisted Designer batch differs from verified evidence')
        if state.designer_action is not None:
            if action_verifier is None:
                raise ValueError('persisted package-generating Designer action was not verified')
            verify_args = (state.designer_action.tokens, state.designer_action.checkpoint,
                state.package_id, state.designer_action.rollout_id,
                state.designer_action.group_id)
            if action_verifier.verify_designer_action(*verify_args) is not True:
                raise ValueError('persisted package-generating Designer action was not verified')
        return state
