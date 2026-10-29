"""Resolve routing metadata from trusted experiment inputs, not model guesses.

The wire proposal may omit scope or domains. The published SkillPatch remains a
complete, hashed v1 object. Explicit legacy declarations are checked, never
silently corrected; a scope change needs a separate experiment.
"""
from __future__ import annotations

from typing import Literal, Tuple
from pydantic import Field, field_validator

from ..schema import FrozenModel
from ..skills.models import NonemptyText, Scope, SkillPatch


class DomainPolicy(FrozenModel):
    add_domains: Tuple[NonemptyText, ...] = Field(min_length=1, max_length=32)
    update_domains: Literal["inherit"] = "inherit"

    @field_validator("add_domains")
    @classmethod
    def concrete_domains(cls, domains):
        domains = Scope(domains=domains).domains
        if "*" in domains:
            raise ValueError("ADD policy requires concrete experiment domains")
        return domains


class ScopeResolution(FrozenModel):
    source: Literal["parent_skill", "experiment_policy"]
    domains: Tuple[NonemptyText, ...]


def proposal_json_schema(role=None):
    if role not in (None, "customer", "service"):
        raise ValueError("unknown proposal role")
    schema = SkillPatch.model_json_schema()
    if role is not None:
        schema["properties"]["role"]["const"] = role
    schema["properties"]["skill"]["description"] = (
        "ADD/UPDATE require a complete non-null skill with id, type, trigger and all step arrays. "
        "REMOVE/NOOP must omit skill or use null. Scope metadata is program-managed."
    )
    scope = schema["$defs"]["Scope"]
    scope["required"] = [name for name in scope.get("required", ()) if name != "domains"]
    scope["properties"]["domains"]["description"] = (
        "Optional compatibility field. Omit it: UPDATE inherits the parent domains; "
        "ADD uses the experiment's add_domains. An explicit mismatch is rejected."
    )
    for name in ("ExplorationSkill", "DecisionSkill"):
        skill = schema["$defs"][name]
        skill["required"] = [key for key in skill["required"] if key != "scope"]
        skill["properties"]["content_hash"]["description"] = "Omit; the program computes and checks this hash."
    if role == "customer":
        fields = schema["$defs"]["ExplorationSkill"]["properties"]
        for name, description in {
            "hypotheses": "Testing hypotheses about Service decision-making; distinguish suspected from confirmed weaknesses.",
            "exploration": "Concrete, authorized probes of Service choices, not a business execution tutorial.",
            "feedback": "Interpret correct, suspicious, unknown and independently verified outcomes; branch to the next legal test.",
            "stop": "Stop at budget, authorization or evidence limits; finding a vulnerability is not required.",
        }.items():
            fields[name]["description"] = description
    return schema


def resolve_scope(data, library, policy):
    if not isinstance(data, dict):
        raise ValueError("Reflection must return one JSON patch object")
    if data.get("action") not in ("ADD", "UPDATE") or not isinstance(data.get("skill"), dict):
        return data, None  # the complete SkillPatch parser reports shape errors
    skill = data["skill"]
    if data["action"] == "UPDATE":
        parent = next((row for row in library.skills if row.id == skill.get("id")), None)
        if parent is None:
            raise ValueError("UPDATE references a missing skill")
        domains, tools, source = parent.scope.domains, parent.scope.tools, "parent_skill"
    else:
        if policy is None:
            raise ValueError("ADD requires a program-supplied domain policy")
        domains, tools, source = policy.add_domains, (), "experiment_policy"
    scope = skill.get("scope", {})
    if not isinstance(scope, dict):
        raise ValueError("scope must be an object when supplied")
    resolved = dict(scope)
    resolved.setdefault("domains", domains)
    resolved.setdefault("tools", tools)
    checked = Scope.model_validate(resolved)
    if set(checked.domains) != set(domains):
        raise ValueError("domain scope is program-managed; applicability changes require a separate experiment")
    skill["scope"] = checked.model_dump(mode="json")
    return data, ScopeResolution(source=source, domains=domains)
