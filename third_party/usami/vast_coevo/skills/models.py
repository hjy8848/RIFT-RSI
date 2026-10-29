"""Declarative, immutable skill contracts independent of any environment."""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Tuple, Union

from pydantic import ConfigDict, Field, StringConstraints, field_validator, model_validator

from ..schema import FrozenModel, digest


SkillRole = Literal["customer", "service", "designer"]
NonemptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
SkillId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")]
Hash = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Steps = Annotated[Tuple[NonemptyText, ...], Field(min_length=1, max_length=32)]


def _unique(values: Tuple[str, ...], label: str) -> Tuple[str, ...]:
    if len(values) != len(set(values)):
        raise ValueError("duplicate " + label)
    return values


class Scope(FrozenModel):
    domains: Tuple[NonemptyText, ...] = Field(min_length=1, max_length=32)
    tools: Tuple[NonemptyText, ...] = Field(default=(), max_length=128)

    @field_validator("domains")
    @classmethod
    def unique_domains(cls, values: Tuple[str, ...]) -> Tuple[str, ...]:
        return _unique(tuple(value.casefold() for value in values), "domains")

    @field_validator("tools")
    @classmethod
    def unique_tools(cls, values: Tuple[str, ...]) -> Tuple[str, ...]:
        return _unique(values, "tools")


class Trigger(FrozenModel):
    model_config = ConfigDict(json_schema_extra={"anyOf": [
        {"required": ["keywords"], "properties": {"keywords": {"minItems": 1}}},
        {"required": ["intents"], "properties": {"intents": {"minItems": 1}}},
    ]})

    keywords: Tuple[NonemptyText, ...] = Field(default=(), max_length=64)
    intents: Tuple[NonemptyText, ...] = Field(default=(), max_length=64)

    @field_validator("keywords", "intents")
    @classmethod
    def normalize(cls, values: Tuple[str, ...]) -> Tuple[str, ...]:
        return _unique(tuple(value.casefold() for value in values), "trigger terms")

    @model_validator(mode="after")
    def require_trigger(self) -> "Trigger":
        if not self.keywords and not self.intents:
            raise ValueError("a skill needs at least one keyword or intent trigger")
        return self


class _Skill(FrozenModel):
    schema_version: Literal[1] = 1
    id: SkillId
    version: int = Field(default=1, ge=1, strict=True)
    scope: Scope
    trigger: Trigger
    description: str = Field(default="", max_length=4000)
    content_hash: str = ""

    @model_validator(mode="after")
    def seal_content(self) -> "_Skill":
        expected = digest(self.model_dump(mode="json", exclude={"content_hash"}))
        if self.content_hash and self.content_hash != expected:
            raise ValueError("skill content hash does not match its content")
        object.__setattr__(self, "content_hash", expected)
        return self


class ExplorationSkill(_Skill):
    type: Literal["exploration"] = "exploration"
    hypotheses: Steps
    exploration: Steps
    feedback: Steps
    stop: Steps


class DecisionSkill(_Skill):
    type: Literal["decision"] = "decision"
    preconditions: Steps
    procedure: Steps
    postconditions: Steps
    recovery: Steps


class DesignerSkill(_Skill):
    """Test selection guidance, deliberately distinct from in-world Customer skills."""
    type: Literal["design"] = "design"
    selection: Steps
    transformations: Steps
    interpretation: Steps
    stop: Steps


Skill = Annotated[Union[ExplorationSkill, DecisionSkill, DesignerSkill], Field(discriminator="type")]
SkillType = Skill


def skill_role(skill: Union[ExplorationSkill, DecisionSkill, DesignerSkill]) -> SkillRole:
    return "designer" if isinstance(skill, DesignerSkill) else "customer" if isinstance(skill, ExplorationSkill) else "service"


def _patch_schema_constraints(schema):
    """Expose action/role shape invariants without changing stored v1 objects.

    A required nullable field still accepts null, so mutation branches also
    constrain skill to the concrete role-specific object. Snapshot, ID equality
    and content-hash checks remain in the parser and gates.
    """
    schema["allOf"] = [
        {
            "if": {"properties": {"action": {"enum": ["ADD", "UPDATE"]},
                                  "role": {"const": role}}, "required": ["action", "role"]},
            "then": {"required": ["skill"], "properties": {"skill": {
                "type": "object", "required": ["type"],
                "properties": {"type": {"const": kind}},
            }}},
        }
        for role, kind in (("customer", "exploration"), ("service", "decision"), ("designer", "design"))
    ] + [
        {
            "if": {"properties": {"action": {"const": "REMOVE"}}, "required": ["action"]},
            "then": {"required": ["skill_id"], "properties": {
                "skill_id": {"type": "string"}, "skill": {"type": "null"}}},
        },
        {
            "if": {"properties": {"action": {"const": "NOOP"}}, "required": ["action"]},
            "then": {"properties": {"skill_id": {"type": "null"}, "skill": {"type": "null"}}},
        },
    ]


class SkillPatch(FrozenModel):
    """Exactly one modification, bound to the snapshot reflected upon."""

    model_config = ConfigDict(json_schema_extra=_patch_schema_constraints)

    schema_version: Literal[1] = 1
    action: Literal["ADD", "UPDATE", "REMOVE", "NOOP"]
    role: SkillRole
    base_snapshot_hash: Hash
    skill_id: Optional[SkillId] = None
    skill: Optional[Skill] = None
    rationale: NonemptyText
    evidence_ids: Tuple[NonemptyText, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def one_modification(self) -> "SkillPatch":
        _unique(self.evidence_ids, "evidence IDs")
        if self.action in ("ADD", "UPDATE"):
            if self.skill is None:
                raise ValueError(self.action + " requires one skill")
            if skill_role(self.skill) != self.role:
                raise ValueError("patch role and skill type disagree")
            if self.skill_id is not None and self.skill_id != self.skill.id:
                raise ValueError("patch skill_id and skill.id disagree")
            object.__setattr__(self, "skill_id", self.skill.id)
        elif self.action == "REMOVE":
            if self.skill_id is None or self.skill is not None:
                raise ValueError("REMOVE requires only a skill_id")
        elif self.skill is not None or self.skill_id is not None:
            raise ValueError("NOOP cannot contain a skill or skill_id")
        return self


Patch = SkillPatch
