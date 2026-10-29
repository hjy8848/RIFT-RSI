"""Offline structural and instruction safety checks for one candidate patch.

The text checks are conservative heuristics, not a replacement for environment
authorization or dynamic validation. Entity literals are supplied by an adapter;
the core never imports a domain schema or guesses what an entity ID means.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Tuple, Union

from .library import LibrarySnapshot, apply_patch
from .models import SkillPatch
from ..schema import FrozenModel


class StaticValidation(FrozenModel):
    passed: bool
    checks: Tuple[str, ...] = ()
    reasons: Tuple[str, ...] = ()


ValidationResult = StaticValidation

_CODE_PATTERNS = (
    r"```\s*(?:python|py|javascript|js|typescript|ts|bash|sh|shell|powershell)\b",
    r"(?m)^\s*(?:from\s+[\w.]+\s+import\s+|import\s+[\w.]+(?:\s|$)|def\s+\w+\s*\(|class\s+\w+\s*[:(])",
    r"\b(?:exec|eval|compile|__import__)\s*\(",
    r"\b(?:subprocess\.(?:run|Popen|call)|os\.system)\s*\(",
    r"<\s*script\b",
    r"\b(?:python\d?|node|bash|sh)\s+-[ce]\b",
    r"\b(?:execute|run)\s+(?:arbitrary\s+)?(?:python|javascript|shell)\s+code\b",
    r"(?:执行|运行)(?:任意)?(?:Python|Javascript|Shell|脚本)代码",
)
_UNSAFE_PATTERNS = (
    r"\b(?:ignore|override|replace|discard|forget|disregard)\s+(?:(?:all|the|any|existing|previous|prior|original)\s+)*(?:system|developer|previous|prior)\s+(?:instructions?|messages?|prompts?)\b",
    r"\b(?:bypass|disable|skip|ignore|override|forge|fabricate|reuse)\s+(?:(?:all|the|any|existing|customer|user|write|exact|single.use)\s+)*(?:authorization|permissions?|grants?|consent|access\s+controls?|approval|auth(?:entication|orization)?\s+checks?)\b",
    r"\b(?:treat|interpret|use)\s+(?:this|the)\s+skill\s+as\s+(?:a\s+)?(?:system|developer)\s+(?:instruction|message|prompt)\b",
    r"\byou\s+are\s+(?:now\s+)?(?:the\s+)?(?:system|administrator|root)\b",
    r"\b(?:execute|write|act|proceed|operate)\b.{0,60}\bwithout\s+(?:(?:customer|user|explicit|exact|write|valid)\s+)*(?:authorization|permission|consent|approval|grant)\b",
    r"\b(?:grant|authorize)\s+(?:yourself|itself)\b",
    r"\b(?:service|assistant)\s+(?:can|may|must|should)\s+authorize\s+(?:itself|writes|operations|actions)\b",
    r"\b(?:read|access|inspect|expose|query|consult)\s+(?:(?:the|full|complete)\s+)*(?:hidden\s+(?:oracle|rules?|answers?|state)|private\s+(?:world\s+)?state)\b",
    r"\b(?:modify|mutate|edit|write)\s+(?:(?:the|private|full)\s+)*world\s+(?:state\s+)?directly\b",
    r"(?:绕过|跳过|忽略|伪造|禁用)(?:客户|用户|写操作|现有|所有)?(?:授权|权限|许可|审批|系统指令)",
    r"(?:读取|访问|查询|暴露)(?:隐藏的?|私有的?)(?:Oracle|规则|答案|世界状态)",
    r"(?:自行|自己)(?:授权|授予权限)",
)
_NEGATION = re.compile(r"\b(?:do\s+not|don't|never|must\s+not|cannot|can\s+not|should\s+not|avoid|refuse\s+to|prohibit)\b|不得|禁止|不要|不能|不应|避免", re.I)
# Commas are clause boundaries for this deliberately conservative safety
# heuristic: negation in "Do not request X, bypass authorization" must not
# scope over the second, unsafe instruction.
_CLAUSES = re.compile(r"[\n,.!?;:，。！？；：]|\b(?:but|however|then|and)\b|但是|然而|然后", re.I)


def _unsafe_instruction(text: str) -> bool:
    for clause in _CLAUSES.split(text):
        for pattern in _UNSAFE_PATTERNS:
            for match in re.finditer(pattern, clause, re.I):
                # A prohibition such as "do not bypass grants" is useful guidance.
                if not _NEGATION.search(clause[:match.start()]):
                    return True
    return False


class StaticSkillGate:
    def __init__(self, allowed_tools: Iterable[str] = (), forbidden_identifiers: Iterable[str] = (), allowed_domains=None):
        self.allowed_tools = frozenset(allowed_tools)
        self.forbidden_identifiers = tuple(value for value in forbidden_identifiers if value)
        self.allowed_domains = None if allowed_domains is None else frozenset(allowed_domains)

    def validate(self, parent: LibrarySnapshot, patch: Union[SkillPatch, dict]) -> StaticValidation:
        reasons = []
        checks = ["schema", "patch_base", "role", "skill_id_version", "scope_trigger"]
        try:
            patch = SkillPatch.model_validate(patch.model_dump(mode="json") if isinstance(patch, SkillPatch) else patch)
            apply_patch(parent, patch, "static-validation")
        except (ValueError, TypeError, AttributeError) as exc:
            return StaticValidation(passed=False, checks=tuple(checks), reasons=(str(exc),))
        if patch.skill is not None:
            skill = patch.skill
            checks.extend(("allowed_tools", "declarative_content", "authorization_boundary", "generic_entity_literals"))
            unknown = set(skill.scope.tools) - self.allowed_tools
            if unknown:
                reasons.append("undeclared or disallowed tools: " + ", ".join(sorted(unknown)))
            # UPDATE's domain scope is checked against the parent in apply_patch.
            # Only a new ADD is restricted to this experiment's configured domains.
            if patch.action == "ADD" and self.allowed_domains is not None and set(skill.scope.domains) - self.allowed_domains:
                reasons.append("new skill declares a domain outside the experiment's ADD policy")
            payload = skill.model_dump(mode="json", exclude={"content_hash"})
            # Check the actual natural-language fields and declarations, not JSON syntax.
            strings = []

            def visit(value):
                if isinstance(value, str):
                    strings.append(unicodedata.normalize("NFKC", value))
                elif isinstance(value, dict):
                    for nested in value.values():
                        visit(nested)
                elif isinstance(value, (list, tuple)):
                    for nested in value:
                        visit(nested)

            visit(payload)
            if any(re.search(pattern, value, re.I) for pattern in _CODE_PATTERNS for value in strings):
                reasons.append("executable code or code execution instructions are forbidden")
            if any("VAST_SKILL" in value.upper() or "[INST]" in value.upper() or "<|SYSTEM|>" in value.upper() for value in strings):
                reasons.append("reserved prompt framing tokens are forbidden")
            if any(_unsafe_instruction(value) for value in strings):
                reasons.append("instruction attempts to change authority, authorization or private-state access")
            for identifier in self.forbidden_identifiers:
                pattern = r"(?<!\w)" + re.escape(unicodedata.normalize("NFKC", identifier)) + r"(?!\w)"
                if any(re.search(pattern, value, re.I) for value in strings):
                    reasons.append("skill contains a concrete environment identifier: " + identifier)
        return StaticValidation(passed=not reasons, checks=tuple(checks), reasons=tuple(reasons))


def validate_patch(parent: LibrarySnapshot, patch: Union[SkillPatch, dict], allowed_tools: Iterable[str] = (), forbidden_identifiers: Iterable[str] = ()) -> StaticValidation:
    return StaticSkillGate(allowed_tools, forbidden_identifiers).validate(parent, patch)
