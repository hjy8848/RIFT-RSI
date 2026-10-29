"""Host-owned ability records linked only by explicit Designer IDs and TaskRefs."""
from dataclasses import asdict, replace
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path

from rift.abilities.records import (AbilityEvaluationEvent, AbilityPrompt, AbilityRecord,
    AbilityRevision, AbilityStatus, AbilityTrainingEvent)
from rift.banks.records import EvidenceLevel, TaskEvidenceRecord
from rift.schemas import Checkpoint, DataPurpose, Outcome, Split


def _jsonable(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _canonical(value):
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def _record_from_json(row):
    evaluations = tuple(AbilityEvaluationEvent(
        value["round_index"], value["service_update_id"], Checkpoint(**value["checkpoint"]),
        value["recovery_before_sr"], value["recovery_after_sr"],
        value["retention_before_sr"], value["retention_after_sr"],
        tuple(value["recovery_task_refs"]), tuple(value["recovery_evidence_refs"]),
        None if value.get("checkpoint_before") is None else Checkpoint(**value["checkpoint_before"]),
        tuple(value.get("retention_task_refs", ())), tuple(value.get("retention_evidence_refs", ())),
        tuple(value.get("target_ability_task_refs", ())), tuple(value.get("fallback_task_refs", ())),
        value.get("target_recovery_before_sr"), value.get("target_recovery_after_sr"))
        for value in row.get("evaluation_history", ()))
    return AbilityRecord(row["ability_id"], row["description"], AbilityStatus(row["status"]),
        row["created_round"], row["last_observed_round"],
        tuple(row.get("failure_task_refs", ())),
        row.get("creator_rollout_id"),
        tuple(AbilityTrainingEvent(**value) for value in row.get("training_history", ())),
        evaluations,
        tuple(AbilityRevision(value["round_index"], value["previous_description"],
            value["revised_description"], tuple(value.get("added_failure_task_refs", ())),
            tuple(value.get("removed_failure_task_refs", ())))
            for value in row.get("revisions", ())))


class AbilityRegistry:
    """Persist explicit Designer choices; the Host never infers semantic matches."""
    def __init__(self, records=()):
        self._records = {}
        for record in records:
            self.add(record)

    @property
    def records(self):
        return tuple(self._records[key] for key in sorted(self._records))

    def prompts(self):
        return tuple(AbilityPrompt(row.ability_id, row.description, row.failure_task_refs)
                     for row in self.records)

    def fingerprint(self):
        payload = {"schema_version": 2,
            "records": [_jsonable(asdict(row)) for row in self.records]}
        return sha256(_canonical(payload).encode("utf-8")).hexdigest()

    def add(self, record):
        if not isinstance(record, AbilityRecord):
            raise TypeError("AbilityRecord required")
        if record.ability_id in self._records:
            raise ValueError("duplicate ability ID")
        self._records[record.ability_id] = record
        return record

    def get(self, ability_id):
        try:
            return self._records[ability_id]
        except KeyError as exc:
            raise KeyError("unknown ability ID") from exc

    def create(self, description, *, round_index, status=AbilityStatus.PROVISIONAL,
               creator_rollout_id=None):
        if not description or type(round_index) is not int or round_index < 0:
            raise ValueError("ability description and non-negative creation round required")
        number = max((int(key[1:]) for key in self._records
                      if key.startswith("A") and key[1:].isdigit()), default=0) + 1
        return self.add(AbilityRecord(f"A{number:06d}", description, status,
            round_index, round_index, creator_rollout_id=creator_rollout_id))

    def apply_designer_choice(self, *, ability_id, hypothesis, add_failure_task_refs,
                              remove_failure_task_refs=(), visible_failures,
                              evidence_verifier, round_index, designer_rollout_id):
        """Apply the selected ID and refs after checking generation context and real failures."""
        if (not isinstance(ability_id, str) or not ability_id or not hypothesis.strip()
                or type(round_index) is not int or round_index < 0
                or not designer_rollout_id
                or not callable(getattr(evidence_verifier, "verify_record", None))):
            raise ValueError("explicit ability choice, hypothesis and evidence verifier required")
        visible = {row.task_ref: row for row in visible_failures
                   if isinstance(row, TaskEvidenceRecord)}
        raw_additions, raw_removals = tuple(add_failure_task_refs), tuple(remove_failure_task_refs)
        additions = tuple(dict.fromkeys(raw_additions))
        removals = tuple(dict.fromkeys(raw_removals))
        if (len(additions) != len(raw_additions)
                or len(removals) != len(raw_removals)
                or set(additions).intersection(removals)
                or not set(additions).issubset(visible)):
            raise ValueError("failure links must be distinct references visible to Designer")

        record = None if ability_id == "NEW" else self.get(ability_id)
        if record is None and removals:
            raise ValueError("a new ability cannot remove existing failure links")
        if record is not None and not set(removals).issubset(record.failure_task_refs):
            raise ValueError("revised failure links must already belong to the selected ability")
        for ref in additions:
            failure = visible[ref]
            valid_failures = [trial for trial in failure.trials
                if trial.valid and trial.outcome is Outcome.FAILURE]
            if (failure.purpose is not DataPurpose.BANK_CANDIDATE
                    or failure.identity.split is Split.TEST or not valid_failures
                    or any(trial.valid and trial.evidence_level is not EvidenceLevel.REAL_SERVICE
                           for trial in failure.trials)
                    or evidence_verifier.verify_record(failure) is not True):
                raise ValueError("ability links require verified real Service failure evidence")

        if record is None:
            replayed = next((row for row in self.records
                             if row.creator_rollout_id == designer_rollout_id), None)
            if replayed is not None:
                if (replayed.description != hypothesis or removals
                        or not set(additions).issubset(replayed.failure_task_refs)):
                    raise ValueError("replayed NEW choice conflicts with its saved Designer action")
                return replayed
            record = self.create(hypothesis, round_index=round_index,
                                 creator_rollout_id=designer_rollout_id)
        old_description = record.description
        old_refs = set(record.failure_task_refs)
        new_refs = tuple(ref for ref in record.failure_task_refs if ref not in set(removals))
        new_refs += tuple(ref for ref in additions if ref not in old_refs)
        changed = hypothesis != old_description or set(new_refs) != old_refs
        revisions = record.revisions
        if changed:
            revisions += (AbilityRevision(round_index, old_description, hypothesis,
                tuple(ref for ref in additions if ref not in old_refs), removals),)
        updated = replace(record, description=hypothesis, failure_task_refs=new_refs,
            revisions=revisions, last_observed_round=max(record.last_observed_round, round_index))
        self._records[record.ability_id] = updated
        return updated

    def record_round(self, state, *, update_verifier, reward_verifier,
                     mastery_threshold=0.75):
        """Append verified Service training and independent probe outcomes once."""
        if (getattr(getattr(state, "stage", None), "value", None) != "DESIGNER_TRAINING_COMPLETED"
                or not state.target_ability_id or state.service_update is None
                or state.learning_reward is None):
            raise ValueError("completed dual-bank round with a target ability is required")
        if update_verifier.verify_update(state.service_update) is not True:
            raise ValueError("Service update evidence was not verified")
        if reward_verifier.verify_reward(state.learning_reward, state.service_update) is not True:
            raise ValueError("dual-bank learning reward was not verified")
        if not 0 <= mastery_threshold <= 1:
            raise ValueError("ability stability threshold must be in [0, 1]")
        record = self.get(state.target_ability_id)
        reward = state.learning_reward
        training = AbilityTrainingEvent(state.round_index, state.package_id,
                                        state.service_update.run_id, reward.value)
        recovery_refs = tuple(probe.task_ref for probe in state.recovery_snapshot.probes)
        evidence_refs = tuple(dict.fromkeys(ref for result in state.recovery_after.results
                                            for ref in result.evidence_refs))
        target_refs = tuple(state.recovery_snapshot.priority_task_refs)
        fallback_refs = tuple(ref for ref in recovery_refs if ref not in set(target_refs))
        before_by_task = {result.task_ref: result.success_rate
                          for result in state.recovery_before.results}
        after_by_task = {result.task_ref: result.success_rate
                         for result in state.recovery_after.results}
        target_before_rates = [before_by_task.get(ref) for ref in target_refs]
        target_after_rates = [after_by_task.get(ref) for ref in target_refs]
        target_before = (None if not target_refs or any(value is None for value in target_before_rates)
                         else sum(target_before_rates) / len(target_before_rates))
        target_after = (None if not target_refs or any(value is None for value in target_after_rates)
                        else sum(target_after_rates) / len(target_after_rates))
        retention_refs = tuple(probe.task_ref for probe in state.retention_snapshot.probes)
        retention_evidence_refs = tuple(dict.fromkeys(ref for result in state.retention_after.results
                                                       for ref in result.evidence_refs))
        evaluation = AbilityEvaluationEvent(state.round_index,
            state.service_update.run_id, state.service_update.policy_after,
            reward.recovery_before_sr, reward.recovery_after_sr,
            reward.retention_before_sr, reward.retention_after_sr,
            recovery_refs, evidence_refs,
            checkpoint_before=state.recovery_snapshot.baseline_checkpoint,
            retention_task_refs=retention_refs,
            retention_evidence_refs=retention_evidence_refs,
            target_ability_task_refs=target_refs,
            fallback_task_refs=fallback_refs,
            target_recovery_before_sr=target_before,
            target_recovery_after_sr=target_after)
        old_training = next((row for row in record.training_history
                             if row.service_update_id == training.service_update_id), None)
        old_evaluation = next((row for row in record.evaluation_history
                               if row.service_update_id == evaluation.service_update_id), None)
        if old_training is not None or old_evaluation is not None:
            if old_training != training or old_evaluation != evaluation:
                raise ValueError("ability round evidence conflicts with persisted history")
            return record
        if target_after is None:
            status = AbilityStatus.UNCERTAIN
        elif target_after >= mastery_threshold:
            status = AbilityStatus.STABLE
        elif target_before is not None and target_after > target_before:
            status = AbilityStatus.IMPROVING
        else:
            status = AbilityStatus.PERSISTENT_WEAKNESS
        updated = replace(record, status=status,
            last_observed_round=max(record.last_observed_round, state.round_index),
            training_history=record.training_history + (training,),
            evaluation_history=record.evaluation_history + (evaluation,))
        self._records[record.ability_id] = updated
        return updated

    def save(self, path):
        path = Path(path)
        payload = {"schema_version": 2,
            "records": [_jsonable(asdict(row)) for row in self.records]}
        envelope = {"payload": payload,
                    "sha256": sha256(_canonical(payload).encode("utf-8")).hexdigest()}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_canonical(envelope), encoding="utf-8")

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = raw.get("payload")
        if not isinstance(payload, dict) or payload.get("schema_version") != 2:
            raise ValueError("unsupported Ability Registry schema")
        if raw.get("sha256") != sha256(_canonical(payload).encode("utf-8")).hexdigest():
            raise ValueError("Ability Registry digest mismatch")
        return cls(_record_from_json(row) for row in payload["records"])
