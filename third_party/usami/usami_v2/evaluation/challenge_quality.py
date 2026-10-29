"""Evidence vectors, not a failure-count reward or implicit Designer Skill gate."""
from typing import Callable, Literal, Optional, Tuple

from pydantic import Field, model_validator

from vast_coevo.schema import digest
from usami_v2.contracts import Sealed


class Consequence(Sealed):
    phase: str
    goal_effects: Tuple[str, ...]
    constraint_effects: Tuple[str, ...]
    money_delta_cents: int = Field(strict=True)
    delay_seconds: int = Field(ge=0, strict=True)
    information_gained: Tuple[str, ...] = ()
    resource_effects: Tuple[str, ...] = ()

    @property
    def consequence_class(self):
        # No SKU ID, generated object ID or seed belongs in equivalence classes.
        data = self.model_dump(mode="json", exclude={"record_hash"})
        for field in ("goal_effects", "constraint_effects", "information_gained", "resource_effects"):
            data[field] = sorted(data[field])
        return digest(data)


class AlternativeEvidence(Sealed):
    alternative_id: str
    world_hash: str
    task_hash: str
    proof_hash: str
    consequence: Consequence


class ChallengeDescriptor(Sealed):
    world_hash: str
    task_hash: str
    task_kind: Literal["transaction", "information", "refusal", "mixed"]
    business_certificate_hash: str
    interaction_certificate_hash: str
    goal_relations: Tuple[str, ...]
    constraint_relations: Tuple[str, ...]
    information_relations: Tuple[str, ...]
    event_relations: Tuple[str, ...] = ()
    alternatives: Tuple[AlternativeEvidence, ...] = ()
    search_complete: bool = False
    search_completion_proof_hash: Optional[str] = None
    replay_inputs_hash: str

    @model_validator(mode="after")
    def bound_proofs(self):
        if any((row.world_hash, row.task_hash) != (self.world_hash, self.task_hash)
               for row in self.alternatives):
            raise ValueError("alternative proof belongs to another task or world")
        if len({row.alternative_id for row in self.alternatives}) != len(self.alternatives):
            raise ValueError("duplicate alternative IDs")
        if self.search_complete and not self.search_completion_proof_hash:
            raise ValueError("exact consequence count needs a search-completion proof")
        return self


class PreRunQualityV1(Sealed):
    """Read-only parser for archived quality records that lacked an index state."""

    protocol: Literal["commerce-challenge-quality-v1"] = "commerce-challenge-quality-v1"
    descriptor_hash: str
    world_hash: str
    task_hash: str
    business_valid: bool
    consequence_class_lower_bound: int
    consequence_count_exact: bool
    diagnostic_inputs_available: bool
    structural_signature: str
    novel_against_history: bool
    history_index_hash: str
    proof_refs: Tuple[str, ...]
    selection_power: Literal["diagnostic_only"] = "diagnostic_only"


class PreRunQuality(Sealed):
    protocol: Literal["commerce-challenge-quality-v2"] = "commerce-challenge-quality-v2"
    descriptor_hash: str
    world_hash: str
    task_hash: str
    business_valid: bool
    consequence_class_lower_bound: int
    consequence_count_exact: bool
    diagnostic_inputs_available: bool
    structural_signature: str
    novelty_status: Literal["EVALUATED_NOVEL", "EVALUATED_SEEN", "NOT_EVALUATED"]
    novel_against_history: Optional[bool] = None
    history_index_hash: str
    proof_refs: Tuple[str, ...]
    selection_power: Literal["diagnostic_only"] = "diagnostic_only"


def evaluate_challenge(descriptor: ChallengeDescriptor, *,
                       verify_proof: Callable[[str], bool], history_signatures=None):
    descriptor = descriptor.checked()
    refs = (descriptor.business_certificate_hash, descriptor.interaction_certificate_hash) + tuple(
        row.proof_hash for row in descriptor.alternatives)
    if descriptor.search_completion_proof_hash:
        refs += (descriptor.search_completion_proof_hash,)
    if not refs[0] or not refs[1] or any(not verify_proof(ref) for ref in refs):
        raise ValueError("quality requires independently verified certificates")
    # Within-task consequences retain real money/time. Across-task novelty uses
    # their ordering and constraint relations, not a price/seed renaming.
    costs = sorted({row.consequence.money_delta_cents for row in descriptor.alternatives})
    times = sorted({row.consequence.delay_seconds for row in descriptor.alternatives})
    qualitative = []
    for row in descriptor.alternatives:
        data = row.consequence.model_dump(mode="json", exclude={"record_hash", "money_delta_cents", "delay_seconds"})
        for name in ("goal_effects", "constraint_effects", "information_gained", "resource_effects"):
            data[name] = sorted(data[name])
        data["relative_cost_rank"] = costs.index(row.consequence.money_delta_cents)
        data["relative_delay_rank"] = times.index(row.consequence.delay_seconds)
        qualitative.append(digest(data))
    structural = digest({
        "task_kind": descriptor.task_kind,
        "goals": sorted(descriptor.goal_relations),
        "constraints": sorted(descriptor.constraint_relations),
        "information": sorted(descriptor.information_relations),
        "events": sorted(descriptor.event_relations),
        "consequences": sorted(set(qualitative)),
    })
    history_available = history_signatures is not None
    history = tuple(sorted(set(history_signatures or ())))
    novelty_status = ("NOT_EVALUATED" if not history_available else
                      "EVALUATED_NOVEL" if structural not in history else "EVALUATED_SEEN")
    return PreRunQuality(
        descriptor_hash=descriptor.record_hash, world_hash=descriptor.world_hash,
        task_hash=descriptor.task_hash, business_valid=True,
        consequence_class_lower_bound=len({row.consequence.consequence_class for row in descriptor.alternatives}),
        consequence_count_exact=descriptor.search_complete,
        diagnostic_inputs_available=bool(descriptor.replay_inputs_hash),
        structural_signature=structural, novelty_status=novelty_status,
        novel_against_history=(structural not in history) if history_available else None,
        history_index_hash=digest({"available": history_available, "signatures": history}),
        proof_refs=refs)


class PostRunUtility(Sealed):
    pre_run_quality_hash: str
    execution_hash: str
    verified_business_error_refs: Tuple[str, ...]
    recovered_error_refs: Tuple[str, ...]
    protocol_error_refs: Tuple[str, ...]
    customer_error_refs: Tuple[str, ...]
    transport_error_refs: Tuple[str, ...]
    successful_task: bool
    causal_assessment: Literal["not_evaluated"] = "not_evaluated"
    selection_power: Literal["diagnostic_only"] = "diagnostic_only"
