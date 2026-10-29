"""Exact controlled starts are distinct from autonomous whole-task evaluation."""
from typing import Callable, Literal, Optional, Tuple

from pydantic import Field, model_validator

from usami_v2.contracts import Sealed


class DecisionCheckpoint(Sealed):
    protocol: Literal["commerce-decision-checkpoint-v1"] = "commerce-decision-checkpoint-v1"
    checkpoint_id: str
    provenance: Literal["diagnostic_training", "independent_validation"]
    partition: Literal["train", "validation"]
    source_record_hash: str
    world_prestate_hash: str
    rules_version: str
    identity: str
    delegation_hash: str
    consent_hash: str
    public_goal_history_hash: str
    visible_history_hash: str
    initial_working_state_hash: str
    baseline_prompt_hash: str
    virtual_time: str
    event_cursor_hash: str
    tool_schema_hash: str
    model_sampling_hash: str
    objective: str
    subject_binding: str
    evaluation_window: Literal["NEXT_DECISION", "BOUNDED_CONTINUATION"]
    continuation_policy_hash: str
    max_model_calls: int = Field(ge=1, strict=True)
    max_tool_calls: int = Field(ge=1, strict=True)
    training_lineage_excluded: bool

    @model_validator(mode="after")
    def independent_validation(self):
        if self.provenance == "independent_validation" and (
                self.partition != "validation" or not self.training_lineage_excluded):
            raise ValueError("validation checkpoint must exclude training provenance")
        if self.provenance == "diagnostic_training" and self.partition != "train":
            raise ValueError("diagnostic training cannot masquerade as independent validation")
        from datetime import datetime
        from usami_v2.contracts import aware
        aware(datetime.fromisoformat(self.virtual_time))
        return self


class CheckpointAttempt(Sealed):
    checkpoint: DecisionCheckpoint
    system_snapshot_hash: str
    injected_experience_hash: str
    execution_hash: str
    replay_verified: bool
    outcome: Literal[
        "CORRECT", "INCORRECT", "REQUEST_MORE_INFORMATION", "NOT_REACHED",
        "PROTOCOL_ERROR", "UNKNOWN",
    ]
    acquired_observation_hashes: Tuple[str, ...] = ()


class CheckpointComparison(Sealed):
    protocol: Literal["commerce-checkpoint-comparison-v1"] = "commerce-checkpoint-comparison-v1"
    parent_hash: str
    candidate_hash: str
    equivalent_start: bool
    differing_fields: Tuple[str, ...]
    local_improvement: Optional[bool]
    independent_validation: bool
    end_to_end_improvement: Literal["not_evaluated"] = "not_evaluated"
    conclusion: str


def compare_checkpoints(parent: CheckpointAttempt, candidate: CheckpointAttempt, *,
                        verify_attempt: Callable[[CheckpointAttempt], bool]):
    parent, candidate = parent.checked(), candidate.checked()
    if (not parent.replay_verified or not candidate.replay_verified
            or not verify_attempt(parent) or not verify_attempt(candidate)):
        raise ValueError("checkpoint comparison requires independently replayed attempts")
    left = parent.checkpoint.model_dump(mode="json", exclude={"record_hash", "checkpoint_id"})
    right = candidate.checkpoint.model_dump(mode="json", exclude={"record_hash", "checkpoint_id"})
    differences = tuple(key for key in left if left[key] != right[key])
    independent = parent.checkpoint.provenance == candidate.checkpoint.provenance == "independent_validation"
    if differences:
        improved, conclusion = None, "NOT_EQUIVALENT: same tool names do not establish the same decision problem"
    elif parent.outcome == "INCORRECT" and candidate.outcome == "CORRECT":
        improved, conclusion = True, "Controlled local improvement only; whole-task evidence still required"
    elif parent.outcome == "CORRECT" and candidate.outcome == "INCORRECT":
        improved, conclusion = False, "Controlled local regression"
    elif parent.outcome == candidate.outcome and parent.outcome in ("CORRECT", "INCORRECT"):
        improved, conclusion = False, "No controlled local improvement"
    else:
        improved, conclusion = None, "Information gathering or an unreached decision is not an established repair"
    return CheckpointComparison(
        parent_hash=parent.record_hash, candidate_hash=candidate.record_hash,
        equivalent_start=not differences, differing_fields=differences,
        local_improvement=improved, independent_validation=independent, conclusion=conclusion)


class WholeTaskPair(Sealed):
    parent_execution_hash: str
    candidate_execution_hash: str
    initial_world_hash: str
    customer_scenario_hash: str
    tool_schema_hash: str
    model_sampling_hash: str
    external_process_hash: str
    budget_hash: str
    parent_success: Optional[bool]
    candidate_success: Optional[bool]
    parent_protocol_failure: bool
    candidate_protocol_failure: bool
    # All paired episodes stay in the denominator, including NOT_REACHED.
    parent_target_status: Literal["REACHED", "NOT_REACHED", "UNKNOWN"]
    candidate_target_status: Literal["REACHED", "NOT_REACHED", "UNKNOWN"]
