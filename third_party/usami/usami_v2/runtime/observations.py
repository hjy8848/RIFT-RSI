"""Field-level visibility and provenance; never project hidden task answers."""
from typing import Literal, Tuple

from pydantic import Field, model_validator

from usami_v2.contracts import FactEvidence, Role, Sealed, SourceRef


class ContextItem(Sealed):
    name: str = Field(min_length=1)
    value: str
    source: SourceRef
    audience: Tuple[Role, ...]
    owner_id: str = ""


class RoleView(Sealed):
    role: Role
    identity: str
    items: Tuple[ContextItem, ...]


_SERVICE_SOURCES = frozenset({
    "customer_message", "public_policy", "public_tool", "transaction_receipt",
})


def project(role: Role, identity: str, items: Tuple[ContextItem, ...]) -> RoleView:
    """Only include explicitly addressed, role-permitted data from trusted sources."""
    selected = []
    for raw in items:
        item = raw.checked()
        if role not in item.audience:
            continue
        if item.owner_id and item.owner_id != identity and role in ("customer", "service"):
            continue
        kind = item.source.source_kind
        if role == "service" and kind not in _SERVICE_SOURCES:
            continue
        if role == "customer" and kind not in _SERVICE_SOURCES | {"private_scenario"}:
            continue
        if role in ("designer", "learner") and kind == "heldout":
            raise ValueError("held-out material cannot enter training context")
        selected.append(item)
    return RoleView(role=role, identity=identity, items=tuple(sorted(selected, key=lambda i: i.name)))


class WorkingFact(Sealed):
    fact: FactEvidence
    observed_at: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def public_only(self):
        if self.fact.source.source_kind not in _SERVICE_SOURCES:
            raise ValueError("Working Memory cannot import private task or Oracle facts")
        if "service" not in self.fact.visible_to:
            raise ValueError("fact is not Service visible")
        if self.observed_at < self.fact.source.event_index:
            raise ValueError("fact not yet observed")
        return self


class WorkingState(Sealed):
    protocol: Literal["commerce-working-state-v1"] = "commerce-working-state-v1"
    identity: str
    facts: Tuple[WorkingFact, ...] = ()
    # Goals contain actual public requests, not hidden TaskSpec targets.
    goals: Tuple[ContextItem, ...] = ()

    @model_validator(mode="after")
    def goals_are_public(self):
        for goal in self.goals:
            if goal.source.source_kind != "customer_message" or "service" not in goal.audience:
                raise ValueError("working goals require actual public Customer messages")
            if goal.owner_id and goal.owner_id != self.identity:
                raise ValueError("working goal belongs to another Customer")
        return self

    def current_facts(self, event_index: int):
        self.checked()
        return tuple(row.fact for row in self.facts
                     if row.observed_at <= event_index
                     and row.fact.valid_from <= event_index
                     and (row.fact.valid_until is None or event_index <= row.fact.valid_until))
