"""Public skill schemas, version storage, deterministic retrieval and framing."""

from .library import LibrarySnapshot, LibraryStore, apply_patch, empty_snapshot
from .models import DecisionSkill, DesignerSkill, ExplorationSkill, Patch, Scope, Skill, SkillPatch, SkillRole, SkillType, Trigger
from .renderer import SkillRenderer, render_skills
from .retriever import RetrievalResult, Retriever, SkillContext, SkillHit, SkillRetriever
from .validation import StaticSkillGate, StaticValidation, ValidationResult, validate_patch


def json_schemas():
    """Export versioned declarative schemas for tools, documentation and reflectors."""
    return {
        name: model.model_json_schema()
        for name, model in (
            ("exploration_skill", ExplorationSkill), ("decision_skill", DecisionSkill),
            ("skill_patch", SkillPatch), ("library_snapshot", LibrarySnapshot),
            ("skill_context", SkillContext),
        )
    }


__all__ = [
    "DecisionSkill", "DesignerSkill", "ExplorationSkill", "LibrarySnapshot", "LibraryStore", "Patch",
    "RetrievalResult", "Retriever", "Scope", "Skill", "SkillContext", "SkillHit",
    "SkillPatch", "SkillRenderer", "SkillRetriever", "SkillRole", "SkillType",
    "StaticSkillGate", "StaticValidation", "Trigger", "ValidationResult",
    "apply_patch", "empty_snapshot", "json_schemas", "render_skills", "validate_patch",
]
