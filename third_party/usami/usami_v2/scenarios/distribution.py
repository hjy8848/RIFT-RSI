"""Explicit synthetic/reference populations and auditable bounded sampling."""
import random
from collections import Counter
from typing import Literal, Optional, Tuple

from pydantic import Field, model_validator

from usami_v2.contracts import Sealed


class BusinessSource(Sealed):
    source_id: str = Field(min_length=1)
    kind: Literal["synthetic_assumption", "public_policy", "public_benchmark", "human_review", "empirical"]
    reference: str = Field(min_length=1)
    version: str = Field(min_length=1)
    supported_claims: Tuple[str, ...]
    restrictions: Tuple[str, ...] = ()
    empirical_sample_count: Optional[int] = Field(default=None, ge=1, strict=True)

    @model_validator(mode="after")
    def empirical_needs_data(self):
        if self.kind == "empirical" and self.empirical_sample_count is None:
            raise ValueError("empirical provenance needs a declared sample count")
        return self


class BusinessStratum(Sealed):
    stratum_id: str = Field(min_length=1)
    weight: int = Field(gt=0, strict=True)
    weight_basis_source_id: str
    contexts: Tuple[str, ...] = Field(min_length=1)
    dependencies: Tuple[str, ...] = ()


class DistributionSpec(Sealed):
    protocol: Literal["commerce-distribution-v1"] = "commerce-distribution-v1"
    distribution_id: str = Field(min_length=1)
    version: int = Field(ge=1, strict=True)
    population_scope: str = Field(min_length=1)
    claim: Literal["synthetic_business_reference", "empirical_business_population"] = "synthetic_business_reference"
    sources: Tuple[BusinessSource, ...] = Field(min_length=1)
    strata: Tuple[BusinessStratum, ...] = Field(min_length=1)
    exclusions: Tuple[str, ...] = ()
    max_generation_attempts: int = Field(default=100, ge=1, strict=True)
    split: Literal["train", "validation", "heldout"]

    @model_validator(mode="after")
    def provenance(self):
        source_ids = {row.source_id for row in self.sources}
        if len(source_ids) != len(self.sources) or len({row.stratum_id for row in self.strata}) != len(self.strata):
            raise ValueError("duplicate source or stratum IDs")
        if any(row.weight_basis_source_id not in source_ids for row in self.strata):
            raise ValueError("stratum weights need a known provenance source")
        if self.claim == "empirical_business_population":
            empirical = {row.source_id for row in self.sources if row.kind == "empirical"}
            if any(row.weight_basis_source_id not in empirical for row in self.strata):
                raise ValueError("synthetic weights cannot claim empirical population representativeness")
        return self


class SamplingSlot(Sealed):
    distribution_hash: str
    slot_id: int = Field(ge=0, strict=True)
    stratum_id: str
    seed: int = Field(strict=True)


class GenerationAttempt(Sealed):
    slot: SamplingSlot
    attempt: int = Field(ge=0, strict=True)
    status: Literal["SAT", "UNSAT", "UNKNOWN", "ERROR"]
    accepted: bool
    world_hash: Optional[str] = None
    reason: str = ""

    @model_validator(mode="after")
    def actual_accepted_world(self):
        if self.accepted and (self.status != "SAT" or not self.world_hash):
            raise ValueError("accepted worlds require a SAT witness and world hash")
        if self.status != "SAT" and not self.reason:
            raise ValueError("failed or uncertain generation needs a reason")
        return self


class SamplingAudit(Sealed):
    distribution_hash: str
    planned_slots: int
    completed_slots: int
    total_attempts: int
    by_stratum: Tuple[Tuple[str, int, int, int], ...]  # id, planned, attempted, accepted
    status_counts: Tuple[Tuple[str, int], ...]
    accepted_worlds: Tuple[str, ...]
    all_slots_filled: bool
    population_claim: str
    warning: str = "Report accepted and target strata separately; rejection can change the sampled population."


def sample_schedule(spec: DistributionSpec, seed: int, count: int):
    """Exact stratified allocation, randomized order; never react to model results."""
    spec = spec.checked()
    if type(seed) is not int or type(count) is not int or count < 1:
        raise ValueError("integer seed and positive sample count required")
    rows = sorted(spec.strata, key=lambda row: row.stratum_id)
    total = sum(row.weight for row in rows)
    allocations = {row.stratum_id: count * row.weight // total for row in rows}
    remaining = count - sum(allocations.values())
    ranked = sorted(rows, key=lambda row: (-(count * row.weight % total), row.stratum_id))
    for row in ranked[:remaining]:
        allocations[row.stratum_id] += 1
    labels = [row.stratum_id for row in rows for _ in range(allocations[row.stratum_id])]
    rng = random.Random(seed)
    rng.shuffle(labels)
    return tuple(SamplingSlot(distribution_hash=spec.record_hash, slot_id=index,
                              stratum_id=label, seed=rng.randrange(2**31))
                 for index, label in enumerate(labels))


def audit_sampling(spec: DistributionSpec, schedule, attempts):
    spec = spec.checked()
    schedule = tuple(row.checked() for row in schedule)
    attempts = tuple(row.checked() for row in attempts)
    slots = {row.slot_id: row for row in schedule}
    if len(slots) != len(schedule) or any(row.distribution_hash != spec.record_hash for row in schedule):
        raise ValueError("schedule not bound to this distribution")
    if any(row.stratum_id not in {s.stratum_id for s in spec.strata} for row in schedule):
        raise ValueError("unknown sampling stratum")
    if len(attempts) > spec.max_generation_attempts:
        raise ValueError("generation budget exceeded")
    grouped = {}
    for row in attempts:
        if slots.get(row.slot.slot_id) != row.slot:
            raise ValueError("generation attempt differs from its pre-registered slot")
        grouped.setdefault(row.slot.slot_id, []).append(row)
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda row: row.attempt)
        if [row.attempt for row in ordered] != list(range(len(ordered))):
            raise ValueError("attempt history has gaps or duplicates")
        if any(row.accepted for row in ordered[:-1]):
            raise ValueError("cannot silently replace a previously accepted world")
    accepted = [row for row in attempts if row.accepted]
    if len({row.world_hash for row in accepted}) != len(accepted):
        raise ValueError("the same world cannot masquerade as independent sampling slots")
    planned = Counter(row.stratum_id for row in schedule)
    attempted = Counter(row.slot.stratum_id for row in attempts)
    included = Counter(row.slot.stratum_id for row in accepted)
    return SamplingAudit(
        distribution_hash=spec.record_hash, planned_slots=len(schedule),
        completed_slots=len(accepted), total_attempts=len(attempts),
        by_stratum=tuple((row.stratum_id, planned[row.stratum_id],
                          attempted[row.stratum_id], included[row.stratum_id]) for row in spec.strata),
        status_counts=tuple(sorted(Counter(row.status for row in attempts).items())),
        accepted_worlds=tuple(row.world_hash for row in accepted),
        all_slots_filled=len(accepted) == len(schedule), population_claim=spec.claim)
