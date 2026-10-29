"""Deterministic retrieval from explicitly permitted context only."""

from __future__ import annotations

from typing import Tuple

from pydantic import Field

from ..schema import FrozenModel
from .library import LibrarySnapshot
from .models import Hash, NonemptyText, Skill, SkillId, SkillRole
from .renderer import render_skills


class SkillContext(FrozenModel):
    domain: NonemptyText
    public_text: str = ""
    intents: Tuple[NonemptyText, ...] = ()
    available_tools: Tuple[NonemptyText, ...] = ()


class SkillHit(FrozenModel):
    id: SkillId
    version: int = Field(ge=1, strict=True)
    content_hash: Hash


class RetrievalResult(FrozenModel):
    skills: Tuple[Skill, ...] = ()
    hits: Tuple[SkillHit, ...] = ()
    rendered_chars: int = Field(default=0, ge=0, strict=True)


class SkillRetriever:
    def __init__(self, role: SkillRole, max_skills: int = 4, max_chars: int = 6000):
        if role not in ("customer", "service", "designer"):
            raise ValueError("unknown skill role")
        if type(max_skills) is not int or max_skills < 0 or type(max_chars) is not int or max_chars < 0:
            raise ValueError("retrieval budgets must be nonnegative integers")
        self.role = role
        self.max_skills = max_skills
        self.max_chars = max_chars

    def retrieve(self, snapshot: LibrarySnapshot, context: SkillContext) -> RetrievalResult:
        snapshot = LibrarySnapshot.model_validate(snapshot.model_dump(mode="json"))
        context = SkillContext.model_validate(context.model_dump(mode="json"))
        if snapshot.role != self.role:
            raise ValueError("retriever role and library role disagree")
        public_text = context.public_text.casefold()
        intents = {intent.casefold() for intent in context.intents}
        available_tools = set(context.available_tools)
        ranked = []
        for skill in snapshot.skills:
            if context.domain.casefold() not in skill.scope.domains and "*" not in skill.scope.domains:
                continue
            if not set(skill.scope.tools).issubset(available_tools):
                continue
            keyword_hits = sum(term in public_text for term in skill.trigger.keywords)
            intent_hits = len(set(skill.trigger.intents) & intents)
            if not keyword_hits and not intent_hits:
                continue
            ranked.append((-(keyword_hits + 3 * intent_hits), skill.id, skill))
        selected = []
        for _, _, skill in sorted(ranked, key=lambda item: (item[0], item[1])):
            if len(selected) >= self.max_skills:
                break
            if len(render_skills(tuple(selected) + (skill,))) <= self.max_chars:
                selected.append(skill)
        skills = tuple(selected)
        return RetrievalResult(
            skills=skills,
            hits=tuple(SkillHit(id=skill.id, version=skill.version, content_hash=skill.content_hash) for skill in skills),
            rendered_chars=len(render_skills(skills)),
        )


Retriever = SkillRetriever
