"""Content-addressed skill snapshots and immutable publication storage."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

from pydantic import Field, TypeAdapter, model_validator

from ..schema import FrozenModel, atomic_json, digest
from .models import Hash, NonemptyText, Skill, SkillPatch, SkillRole, skill_role


class LibrarySnapshot(FrozenModel):
    schema_version: Literal[1] = 1
    role: SkillRole
    skills: Tuple[Skill, ...] = Field(default=(), max_length=4096)
    parent_hash: Optional[Hash] = None
    provenance: NonemptyText = "initial"
    snapshot_hash: str = ""

    @model_validator(mode="after")
    def seal_snapshot(self) -> "LibrarySnapshot":
        # Nested model instances can otherwise be supplied via unchecked model_copy.
        verified_skills = tuple(TypeAdapter(Skill).validate_python(skill.model_dump(mode="json")) for skill in self.skills)
        object.__setattr__(self, "skills", verified_skills)
        if len({skill.id for skill in self.skills}) != len(self.skills):
            raise ValueError("duplicate skill IDs in snapshot")
        if any(skill_role(skill) != self.role for skill in self.skills):
            raise ValueError("library role and skill type disagree")
        ordered = tuple(sorted(self.skills, key=lambda skill: skill.id))
        object.__setattr__(self, "skills", ordered)
        expected = digest(self.model_dump(mode="json", exclude={"snapshot_hash"}))
        if self.snapshot_hash and self.snapshot_hash != expected:
            raise ValueError("snapshot hash does not match its content")
        object.__setattr__(self, "snapshot_hash", expected)
        return self


def empty_snapshot(role: SkillRole) -> LibrarySnapshot:
    return LibrarySnapshot(role=role)


def apply_patch(parent: LibrarySnapshot, patch: SkillPatch, source: str) -> LibrarySnapshot:
    """Validate first; a rejected modification cannot mutate its parent."""

    # Revalidate even model_copy-created objects, which can bypass Pydantic validators.
    original_parent = parent
    parent = LibrarySnapshot.model_validate(parent.model_dump(mode="json"))
    patch = SkillPatch.model_validate(patch.model_dump(mode="json"))
    if not isinstance(source, str) or not source.strip():
        raise ValueError("a patch requires a nonempty provenance source")
    if patch.role != parent.role:
        raise ValueError("patch role differs from parent library")
    if patch.base_snapshot_hash != parent.snapshot_hash:
        raise ValueError("patch was proposed against another snapshot")
    if patch.action == "NOOP":
        return original_parent
    skills = {skill.id: skill for skill in parent.skills}
    if patch.action == "ADD":
        if patch.skill_id in skills:
            raise ValueError("ADD references an existing skill")
        if patch.skill.version != 1:
            raise ValueError("ADD must start at version 1")
        skills[patch.skill.id] = patch.skill
    elif patch.action == "UPDATE":
        if patch.skill_id not in skills:
            raise ValueError("UPDATE references a missing skill")
        if patch.skill.version != skills[patch.skill_id].version + 1:
            raise ValueError("UPDATE must increment the skill version exactly once")
        if set(patch.skill.scope.domains) != set(skills[patch.skill_id].scope.domains):
            raise ValueError("UPDATE must inherit domain scope; applicability changes require a separate experiment")
        skills[patch.skill.id] = patch.skill
    elif patch.action == "REMOVE":
        if patch.skill_id not in skills:
            raise ValueError("REMOVE references a missing skill")
        del skills[patch.skill_id]
    return LibrarySnapshot(
        role=parent.role,
        skills=tuple(skills.values()),
        parent_hash=parent.snapshot_hash,
        provenance=source,
    )


class LibraryStore:
    """Historical snapshots remain loadable after process restarts."""

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)

    def _snapshot_path(self, snapshot_hash: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", snapshot_hash):
            raise ValueError("invalid snapshot hash")
        return self.root / "snapshots" / (snapshot_hash + ".json")

    def save(self, snapshot: LibrarySnapshot, *, publish: bool = True) -> str:
        snapshot = LibrarySnapshot.model_validate(snapshot.model_dump(mode="json"))
        path = self._snapshot_path(snapshot.snapshot_hash)
        publications = []
        if not publish:
            # Rejected proposals remain inspectable without reserving a published
            # skill version that a later single Reflection might legitimately use.
            atomic_json(path, snapshot.model_dump(mode="json"), immutable=True)
            return snapshot.snapshot_hash
        # Check every collision before publishing any part of this candidate.
        for skill in snapshot.skills:
            publication = self.root / "skills" / snapshot.role / skill.id / (str(skill.version) + ".json")
            if publication.exists():
                existing = TypeAdapter(Skill).validate_json(publication.read_text(encoding="utf-8"))
                if existing.model_dump(mode="json") != skill.model_dump(mode="json"):
                    raise ValueError("published skill version cannot be overwritten: " + skill.id)
            publications.append((publication, skill.model_dump(mode="json")))
        if path.exists():
            if self.load(snapshot.snapshot_hash) != snapshot:
                raise ValueError("published snapshot cannot be overwritten")
        for publication, content in publications:
            atomic_json(publication, content, immutable=True)
        atomic_json(path, snapshot.model_dump(mode="json"), immutable=True)
        return snapshot.snapshot_hash

    def load(self, snapshot_hash: str) -> LibrarySnapshot:
        path = self._snapshot_path(snapshot_hash)
        snapshot = LibrarySnapshot.model_validate_json(path.read_text(encoding="utf-8"))
        if snapshot.snapshot_hash != snapshot_hash:
            raise ValueError("snapshot filename and content hash disagree")
        return snapshot

    def hashes(self, role: Optional[SkillRole] = None) -> Tuple[str, ...]:
        paths = sorted((self.root / "snapshots").glob("*.json"))
        return tuple(path.stem for path in paths if role is None or self.load(path.stem).role == role)
