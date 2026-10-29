from __future__ import annotations
from typing import Literal, Optional, Tuple
from pydantic import Field, model_validator

from ..models import Consumption, EpisodeConfig, ModelSpec
from ..schema import FrozenModel
from ..skills.models import Hash, SkillPatch, SkillRole
from ..skills.validation import StaticValidation
from ..reflection.reflector import ReflectionRecord
from ..evaluation.customer import CustomerQualityPolicy, CustomerQualityResult


class ValidationCase(FrozenModel):
    id: str = Field(min_length=1)
    config: EpisodeConfig
    task_ref: Optional[str] = None
    purpose: Optional[Literal["legal_completion", "correct_refusal"]] = None

    @model_validator(mode="after")
    def development_only(self):
        if self.config.partition != "development":
            raise ValueError("heldout cases cannot enter candidate validation")
        if self.purpose is not None and self.task_ref is None:
            raise ValueError("normal task purpose requires a task reference")
        return self


class GatePolicy(FrozenModel):
    # Customer policies are retained for explicit diagnostics and old records.
    # They never control Customer selection in RunConfig v2 / current Evolver.
    customer_quality: Optional[CustomerQualityPolicy] = Field(default=None, exclude_if=lambda value: value is None)
    customer_excluded_world_hashes: Optional[Tuple[Hash, ...]] = Field(default=None, exclude_if=lambda value: value is None)
    customer_excluded_business_shapes: Optional[Tuple[Hash, ...]] = Field(default=None, exclude_if=lambda value: value is None)
    customer_allow_equal: bool = True
    service_allow_equal_without_failure: bool = True
    require_known_failure_reproduction: bool = True
    require_legal_tasks: bool = True
    require_refusal_tasks: bool = True
    # Paired checks are explicitly versioned so frozen historical configs keep
    # their original aggregate-only interpretation.
    service_task_regression_protocol: Literal["aggregate-v1", "paired-task-v1"] = Field(
        default="aggregate-v1", exclude_if=lambda value: value == "aggregate-v1")


class ValidationMetrics(FrozenModel):
    valid_actions: int = Field(default=0, ge=0, strict=True)
    invalid_attempts: int = Field(default=0, ge=0, strict=True)
    business_violations: int = Field(default=0, ge=0, strict=True)
    authorization_anomalies: int = Field(default=0, ge=0, strict=True)
    false_rejections: int = Field(default=0, ge=0, strict=True)
    execution_errors: int = Field(default=0, ge=0, strict=True)
    legal_task_successes: int = Field(default=0, ge=0, strict=True)
    correct_refusals: int = Field(default=0, ge=0, strict=True)
    fingerprints: Tuple[Hash, ...] = ()
    replay_verified: bool = True


class GateResult(FrozenModel):
    passed: bool
    reasons: Tuple[str, ...]
    baseline: ValidationMetrics
    candidate: ValidationMetrics
    baseline_evidence: Tuple[Hash, ...]
    candidate_evidence: Tuple[Hash, ...]
    repaired: Tuple[Hash, ...] = ()
    consumption: Consumption
    validated_domains: Optional[Tuple[str, ...]] = Field(default=None, exclude_if=lambda value: value is None)
    customer_quality: Optional[CustomerQualityResult] = Field(default=None, exclude_if=lambda value: value is None)


class DomainCoverage(FrozenModel):
    """Coverage describes executions, never a model's claim of generalization."""
    declared_domains: Tuple[str, ...]
    validated_domains: Tuple[str, ...]
    unvalidated_domains: Tuple[str, ...]


class EvolutionRecord(FrozenModel):
    schema_version: Literal[1] = 1
    generation: int = Field(ge=0, strict=True)
    role: SkillRole
    parent_customer_hash: Hash
    parent_service_hash: Hash
    reflection: Optional[ReflectionRecord]
    patch: Optional[SkillPatch] = None
    static_validation: Optional[StaticValidation] = None
    dynamic_validation: Optional[GateResult] = None
    candidate_snapshot_hash: Optional[Hash] = None
    selected_snapshot_hash: Hash
    outcome: Literal["accepted", "rejected", "inconclusive", "noop", "fixed"]
    reasons: Tuple[str, ...] = ()
    consumption: Consumption
    domain_coverage: Optional[DomainCoverage] = Field(default=None, exclude_if=lambda value: value is None)
    skill_updated: Optional[bool] = Field(default=None, exclude_if=lambda v:v is None)
    selection_mode: Optional[Literal['evidence_only_static']] = Field(default=None, exclude_if=lambda v:v is None)
    performance_validated: Optional[Literal[False]] = Field(default=None, exclude_if=lambda v:v is None)
    performance_improved: Optional[Literal['unknown']] = Field(default=None, exclude_if=lambda v:v is None)
    customer_selection: Optional[Literal["customer-static-only-v1"]] = Field(
        default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def static_only_selection(self):
        if self.reflection is None:
            if (self.role != "customer" or self.outcome != "fixed" or self.patch is not None or
                    self.static_validation is not None or self.dynamic_validation is not None or
                    self.candidate_snapshot_hash is not None or self.selected_snapshot_hash != self.parent_customer_hash or
                    self.consumption != Consumption()):
                raise ValueError("only a fixed Customer may skip Reflection, without patch, validation, cost or version change")
        elif self.outcome == "fixed":
            raise ValueError("fixed Customer must not fabricate a Reflection")
        if self.selection_mode is not None:
            if self.skill_updated != (self.selected_snapshot_hash!=self.parent_service_hash):
                raise ValueError('static-only update marker differs from selected version')
            if self.role!='service' or self.performance_validated is not False or self.performance_improved!='unknown' or self.dynamic_validation is not None:
                raise ValueError('static-only Service cannot carry performance validation')
            if self.outcome=='inconclusive':raise ValueError('static-only selection has no performance uncertainty gate')
            if self.outcome!='accepted' and self.selected_snapshot_hash!=self.parent_service_hash:
                raise ValueError('failed/static NOOP selection must retain the parent')
            if self.outcome=='accepted' and (not self.static_validation or not self.static_validation.passed or
                    self.candidate_snapshot_hash!=self.selected_snapshot_hash):
                raise ValueError('static-only selection requires static PASS and exact candidate')
        if self.customer_selection is not None:
            if self.role != "customer" or self.dynamic_validation is not None:
                raise ValueError("static-only Customer selection cannot carry a performance gate")
            if self.outcome == "inconclusive":
                raise ValueError("diagnostic uncertainty cannot be a static-only selection outcome")
            if self.outcome == "accepted" and (self.static_validation is None or
                    not self.static_validation.passed or self.candidate_snapshot_hash is None or
                    self.selected_snapshot_hash != self.candidate_snapshot_hash):
                raise ValueError("static-only acceptance requires static PASS and the exact candidate snapshot")
        return self


class RunConfig(FrozenModel):
    # v1-v3 keep their recorded gate semantics. v4 adds paired Service checks.
    schema_version: Literal[1, 2, 3, 4] = 3
    customer_selection: Optional[Literal["customer-static-only-v1"]] = Field(
        default=None, exclude_if=lambda value: value is None)
    training_protocol: Optional[Literal["recursive-training-v1"]] = Field(
        default=None, exclude_if=lambda value: value is None)
    customer_updates: bool = Field(default=True, exclude_if=lambda value: value is True, strict=True)
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
    generations: int = Field(default=1, ge=1, strict=True)
    exploration_cases: Tuple[ValidationCase, ...] = Field(min_length=1)
    validation_cases: Tuple[ValidationCase, ...] = Field(min_length=1)
    service_task_cases: Tuple[ValidationCase, ...] = ()
    customer_reflector_model: ModelSpec
    service_reflector_model: ModelSpec
    max_llm_calls: int = Field(default=500, ge=2, strict=True)
    max_tool_calls: int = Field(default=1000, ge=0, strict=True)
    policy: GatePolicy = Field(default_factory=GatePolicy)
    initial_customer_hash: Hash
    initial_service_hash: Hash

    @model_validator(mode="after")
    def fixed_models(self):
        if self.schema_version in (2, 3, 4):
            object.__setattr__(self, "customer_selection", "customer-static-only-v1")
        elif self.customer_selection is not None:
            raise ValueError("historical run v1 cannot declare static-only Customer selection")
        if self.schema_version in (3, 4):
            object.__setattr__(self, "training_protocol", "recursive-training-v1")
        elif self.training_protocol is not None:
            raise ValueError("historical runs cannot declare the new recursive protocol")
        if self.schema_version == 4 and self.policy.service_task_regression_protocol != "paired-task-v1":
            raise ValueError("run schema v4 requires paired per-task Service regression checks")
        if self.schema_version < 4 and self.policy.service_task_regression_protocol != "aggregate-v1":
            raise ValueError("paired per-task Service regression checks require run schema v4")
        if self.schema_version == 1 and not self.customer_updates:
            raise ValueError("fixed Customer ablation requires run v2")
        from ..models import ModelSpec
        object.__setattr__(self, "customer_reflector_model", ModelSpec.model_validate(self.customer_reflector_model))
        object.__setattr__(self, "service_reflector_model", ModelSpec.model_validate(self.service_reflector_model))
        if any(c.task_ref is not None for c in self.exploration_cases + self.validation_cases):
            raise ValueError("Discovery worlds cannot contain task references")
        if any(c.task_ref is None for c in self.service_task_cases):
            raise ValueError("Service regression cases require native task references")
        if self.schema_version == 1 and self.policy.customer_quality is not None:
            from ..evaluation.customer import CustomerGateManifest
            excluded = set(self.policy.customer_excluded_world_hashes or ())
            excluded.update(case.config.initial_world_hash for case in self.exploration_cases)
            CustomerGateManifest(policy=self.policy.customer_quality,
                baseline_customer_hash=self.initial_customer_hash, candidate_customer_hash=self.initial_customer_hash,
                service_hash=self.initial_service_hash, configs=tuple(case.config for case in self.validation_cases),
                excluded_world_hashes=tuple(sorted(excluded)))
        pairs = {(c.config.customer_model.model_dump_json(), c.config.service_model.model_dump_json()) for c in self.exploration_cases + self.validation_cases + self.service_task_cases}
        if len(pairs) != 1:
            raise ValueError("base Agent models must be frozen throughout a run")
        return self
