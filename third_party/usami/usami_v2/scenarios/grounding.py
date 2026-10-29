"""ID-free commerce challenges grounded through the existing TaskDraft compiler.

A selector denotes a line variable, not a preselected World object. Multiple
bindings are valid; their deterministic order and proof attempts are recorded.
The legacy benchmark Grounder remains the implementation for legacy Worlds.
"""
from itertools import product
from typing import Literal, Optional, Tuple

from pydantic import Field, model_validator

from usami_benchmark.grounder import GroundingBudget
from usami_harness.models import digest
from usami_v2.contracts import Sealed
from .designer import TaskDraft, compile_task, task_design_capabilities
from .schema import Attribute


class LineSelector(Sealed):
    alias: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    family: Optional[str] = None
    issue: Optional[Literal["damaged", "wrong_item", "missing_parts"]] = None
    attributes: Tuple[Attribute, ...] = ()
    minimum_quantity: int = Field(default=1, ge=1, strict=True)
    maximum_quantity: Optional[int] = Field(default=None, ge=1, strict=True)
    minimum_age_days: int = Field(default=0, ge=0, strict=True)
    maximum_age_days: Optional[int] = Field(default=None, ge=0, strict=True)
    return_required: Optional[bool] = None

    @model_validator(mode="after")
    def ordered(self):
        if self.maximum_quantity is not None and self.maximum_quantity < self.minimum_quantity:
            raise ValueError("selector quantity bounds are reversed")
        if self.maximum_age_days is not None and self.maximum_age_days < self.minimum_age_days:
            raise ValueError("selector age bounds are reversed")
        return self


class LineRelation(Sealed):
    left: str
    right: str
    kind: Literal["same_order", "different_order"]


class AbstractChallenge(TaskDraft):
    """Reuse the native structured-intent TaskDraft contract.

    customer_alias and line_alias are symbolic variables here. Only Grounder
    supplies the real ScenarioSpec aliases to the native compiler.
    """
    challenge_protocol: Literal["commerce-abstract-challenge-v2"] = "commerce-abstract-challenge-v2"
    hypothesis: str = Field(min_length=1)
    line_selectors: Tuple[LineSelector, ...] = Field(min_length=1, max_length=8)
    line_relations: Tuple[LineRelation, ...] = ()

    @model_validator(mode="after")
    def symbolic_scope(self):
        names = {row.alias for row in self.line_selectors}
        if len(names) != len(self.line_selectors):
            raise ValueError("challenge repeats a line variable")
        targets = {row.line_alias for row in self.remedy_goals}
        targets.update(row.subject_alias for row in self.information_goals)
        if targets != names:
            raise ValueError("every goal must target a declared selector and every selector needs a goal")
        if any(row.left not in names or row.right not in names or row.left == row.right
               for row in self.line_relations):
            raise ValueError("line relation references unavailable variables")
        return self


class GroundedCandidate(Sealed):
    world_ref: str
    world_hash: str
    customer_alias: str
    line_binding: Tuple[Tuple[str, str], ...]
    draft: TaskDraft


class GroundingResult(Sealed):
    challenge_hash: str
    budget: dict
    status: Literal["GROUNDED", "NO_MATCH", "BUDGET_EXHAUSTED"]
    worlds_considered: int
    bindings_explored: int
    search_truncated: bool
    candidates: Tuple[GroundedCandidate, ...]


def _matches(selector, spec, build, order, line):
    sku = next(row for row in spec.skus if row.alias == line.original_sku)
    ids = dict(build.private_aliases)
    native_line = build.world.lines[ids[line.alias]]
    age = (build.world.clock - build.world.orders[ids[order.alias]].delivered_at).days
    return ((selector.family is None or sku.family == selector.family)
            and (selector.issue is None or line.issue == selector.issue)
            and all({r.name: r.value for r in sku.attributes}.get(r.name) == r.value for r in selector.attributes)
            and native_line.quantity >= selector.minimum_quantity
            and (selector.maximum_quantity is None or native_line.quantity <= selector.maximum_quantity)
            and age >= selector.minimum_age_days
            and (selector.maximum_age_days is None or age <= selector.maximum_age_days)
            and (selector.return_required is None or sku.return_required == selector.return_required))


def ground_challenge(challenge, worlds, budget=None):
    """Match over authorized frozen Worlds without reading any proof witness.

    worlds contains (opaque public world_ref, ScenarioSpec, BuildResult) triples.
    Distinct variables bind distinct lines of one authenticated Customer.
    """
    challenge = AbstractChallenge.model_validate(challenge).checked()
    budget = GroundingBudget.model_validate(budget or {})
    worlds = tuple(worlds)
    candidates, explored, considered, truncated = [], 0, 0, False
    for world_ref, spec, build in worlds[:budget.max_worlds]:
        spec, build = spec.checked(), build.checked()
        if build.status != "SAT" or build.scenario_hash != spec.record_hash:
            raise ValueError("Grounder requires a verified SAT World binding")
        considered += 1
        for customer in spec.customer_aliases:
            rows = [(order, line) for order in spec.orders if order.customer_alias == customer for line in order.lines]
            choices = [[(order, line) for order, line in rows if _matches(sel, spec, build, order, line)]
                       for sel in challenge.line_selectors]
            for combination in product(*choices):
                if explored >= budget.max_bindings:
                    truncated = True
                    break
                explored += 1
                assigned = dict(zip((row.alias for row in challenge.line_selectors), combination))
                if len({line.alias for _, line in combination}) != len(combination):
                    continue
                if any((assigned[r.left][0].alias == assigned[r.right][0].alias) != (r.kind == "same_order")
                       for r in challenge.line_relations):
                    continue
                binding = {name: line.alias for name, (_, line) in assigned.items()}
                data = challenge.model_dump(mode="json", exclude={"record_hash", "challenge_protocol", "hypothesis",
                                                                 "line_selectors", "line_relations"})
                data["customer_alias"] = customer
                for goal in data["remedy_goals"]:
                    goal.pop("record_hash", None)
                    goal["line_alias"] = binding[goal["line_alias"]]
                for goal in data["information_goals"]:
                    goal.pop("record_hash", None)
                    goal["subject_alias"] = binding[goal["subject_alias"]]
                draft = TaskDraft.model_validate(data)
                candidates.append(GroundedCandidate(world_ref=world_ref, world_hash=build.world_hash,
                    customer_alias=customer, line_binding=tuple(sorted(binding.items())), draft=draft))
            if truncated:
                break
        if truncated:
            break
    truncated |= len(worlds) > budget.max_worlds
    candidates.sort(key=lambda row: digest([budget.selection_seed, row.record_hash]))
    truncated |= len(candidates) > budget.max_matches
    selected = tuple(candidates[:budget.max_matches])
    return GroundingResult(challenge_hash=challenge.record_hash, budget=budget.model_dump(mode="json"),
        status="GROUNDED" if selected else "BUDGET_EXHAUSTED" if truncated else "NO_MATCH",
        worlds_considered=considered, bindings_explored=explored, search_truncated=truncated, candidates=selected)


def compile_grounded(candidate, spec, build, distribution, allowed_training_hashes=(), *, capability_options=None):
    """All business/task compilation remains owned by the native compiler."""
    if candidate.world_hash != build.world_hash:
        raise ValueError("grounded candidate belongs to another frozen World")
    capabilities = task_design_capabilities(spec, build, **(capability_options or {}))
    return compile_task(candidate.draft, spec, build, distribution, allowed_training_hashes, capabilities=capabilities)
