"""Durable Ability Registry with explicit failure links and learning history."""
from dataclasses import asdict, replace
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path

from rift.abilities.records import (AbilityEvaluationEvent, AbilityEvidenceLink,
    AbilityRecord, AbilityRevision, AbilityStatus, AbilityTrainingEvent)
from rift.schemas import Checkpoint, Identity, Split, DataPurpose
from rift.banks.records import EvidenceLevel, TaskEvidenceRecord
from rift.schemas import Outcome


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


class AbilityRegistry:
    """Host-owned hypotheses; identity and evidence never enter Designer inputs."""
    def __init__(self, records=(), *, diagnosis_seen=()):
        self._records = {}
        self._diagnosis_seen = {}
        for record in records:
            self.add(record)
        for task_ref, identity in diagnosis_seen:
            self.mark_diagnosis_seen(task_ref, identity)
    @property
    def records(self):
        return tuple(self._records[key] for key in sorted(self._records))

    @property
    def seen_diagnosis_identities(self):
        return tuple(self._diagnosis_seen[key] for key in sorted(self._diagnosis_seen))

    def fingerprint(self):
        payload = {"schema_version": 1,
            "records": [_jsonable(asdict(row)) for row in self.records],
            "diagnosis_seen": [{"task_ref": key, "identity": _jsonable(asdict(value))}
                               for key, value in sorted(self._diagnosis_seen.items())]}
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

    def create(self, description, *, round_index, status=AbilityStatus.PROVISIONAL):
        if not description or type(round_index) is not int or round_index < 0:
            raise ValueError("ability description and non-negative creation round required")
        number = max((int(key[1:]) for key in self._records
                      if key.startswith("A") and key[1:].isdigit()), default=0) + 1
        record = AbilityRecord(f"A{number:06d}", description, status,
                               round_index, round_index)
        self.add(record)
        return record

    def mark_diagnosis_seen(self, task_ref, identity):
        if (not task_ref or not isinstance(identity, Identity)
                or identity.purpose is not DataPurpose.DIAGNOSIS
                or identity.split is not Split.TRAIN):
            raise ValueError("Designer-visible diagnosis identity required")
        previous = self._diagnosis_seen.get(task_ref)
        if previous is not None and previous != identity:
            raise ValueError("diagnosis TaskRef identity changed across rounds")
        self._diagnosis_seen[task_ref] = identity

    def failure_task_refs(self, ability_id):
        """Return links explicitly confirmed after the current evidence check."""
        return frozenset(self.get(ability_id).explicit_failure_task_refs)

    def link_failure_task(self, ability_id, failure_record, evidence_refs, *,
                          evidence_verifier, round_index, confidence=None, rationale=""):
        record = self.get(ability_id)
        if (not isinstance(failure_record, TaskEvidenceRecord)
                or failure_record.purpose not in (DataPurpose.DIAGNOSIS, DataPurpose.RECOVERY_FEEDBACK)
                or any(trial.valid and trial.evidence_level is not EvidenceLevel.REAL_SERVICE
                       for trial in failure_record.trials)
                or evidence_verifier.verify_record(failure_record) is not True):
            raise ValueError("ability links require verified real Service failure evidence")
        valid_failure_refs = {trial.evidence_ref for trial in failure_record.trials
            if trial.valid and trial.outcome is Outcome.FAILURE
            and trial.evidence_level is EvidenceLevel.REAL_SERVICE}
        evidence_refs = tuple(dict.fromkeys(evidence_refs))
        if not evidence_refs or not set(evidence_refs).issubset(valid_failure_refs):
            raise ValueError("ability link evidence must cite verified failure trials")
        task_ref = failure_record.task_ref
        link = AbilityEvidenceLink(task_ref, tuple(dict.fromkeys(evidence_refs)),
                                   confidence, round_index, rationale)
        links = list(record.failure_links)
        duplicate = next((index for index, row in enumerate(links)
                          if row.task_ref == link.task_ref
                          and row.evidence_refs == link.evidence_refs), None)
        if duplicate is None:
            links.append(link)
        elif (link.confidence is not None and
              (links[duplicate].confidence is None
               or link.confidence > links[duplicate].confidence)):
            links[duplicate] = link
        task_refs = tuple(dict.fromkeys(record.failure_task_refs + (task_ref,)))
        explicit_refs = tuple(dict.fromkeys(record.explicit_failure_task_refs + (task_ref,)))
        self._records[ability_id] = replace(record, failure_task_refs=task_refs,
                                            failure_links=tuple(links),
                                            explicit_failure_task_refs=explicit_refs,
                                            last_observed_round=max(record.last_observed_round, round_index))

    def record_round(self, state, *, update_verifier, reward_verifier,
                     mastery_threshold=0.75):
        """Append verified Designer training and independent probe outcomes once."""
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
        payload = {"schema_version": 1,
            "records": [_jsonable(asdict(row)) for row in self.records],
            "diagnosis_seen": [{"task_ref": key, "identity": _jsonable(asdict(value))}
                               for key, value in sorted(self._diagnosis_seen.items())]}
        envelope = {"payload": payload, "sha256": self.fingerprint()}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_canonical(envelope), encoding="utf-8")

    @classmethod
    def load(cls, path):
        from rift.schemas import DataPurpose
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = raw.get("payload")
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("unsupported Ability Registry schema")
        if raw.get("sha256") != sha256(_canonical(payload).encode("utf-8")).hexdigest():
            raise ValueError("Ability Registry digest mismatch")
        records = []
        for row in payload["records"]:
            records.append(AbilityRecord(row["ability_id"], row["description"],
                AbilityStatus(row["status"]), row["created_round"], row["last_observed_round"],
                tuple(row.get("aliases", ())), tuple(row.get("previous_descriptions", ())),
                tuple(row.get("failure_task_refs", ())),
                tuple(AbilityEvidenceLink(value["task_ref"], tuple(value["evidence_refs"]),
                    value["confidence"], value["round_index"], value["rationale"])
                    for value in row.get("failure_links", ())),
                tuple(AbilityTrainingEvent(**value) for value in row.get("training_history", ())),
                tuple(AbilityEvaluationEvent(value["round_index"], value["service_update_id"],
                    Checkpoint(**value["checkpoint"]), value["recovery_before_sr"],
                    value["recovery_after_sr"], value["retention_before_sr"],
                    value["retention_after_sr"], tuple(value["recovery_task_refs"]),
                    tuple(value["recovery_evidence_refs"]),
                    None if value.get("checkpoint_before") is None else
                        Checkpoint(**value["checkpoint_before"]),
                    tuple(value.get("retention_task_refs", ())),
                    tuple(value.get("retention_evidence_refs", ())),
                    tuple(value.get("target_ability_task_refs", ())),
                    tuple(value.get("fallback_task_refs", ())),
                    value.get("target_recovery_before_sr"),
                    value.get("target_recovery_after_sr"))
                    for value in row.get("evaluation_history", ())),
                tuple(AbilityRevision(value["round_index"], value["previous_description"],
                    value["revised_description"], value["rationale"], tuple(value["evidence_refs"]))
                    for value in row.get("revisions", ()))))
            records[-1] = replace(records[-1],
                                  explicit_failure_task_refs=tuple(
                                      row.get("explicit_failure_task_refs", ())))
        seen = []
        for row in payload.get("diagnosis_seen", ()):
            identity = row["identity"]
            seen.append((row["task_ref"], Identity(identity["world"], identity["task"],
                identity["family"], identity["lineage"], Split(identity["split"]),
                None if identity.get("purpose") is None else DataPurpose(identity["purpose"]))))
        return cls(records, diagnosis_seen=seen)
