"""Versioned request-level Customer quality contracts, independent of tool volume."""
from typing import Dict, Literal, Optional, Tuple
from pydantic import Field, model_validator
from ..schema import FrozenModel, digest
from ..skills.models import Hash
from ..models import EpisodeConfig


class CustomerQualityPolicy(FrozenModel):
    protocol: Literal['quality-gated-discovery-v1', 'quality-gated-discovery-v2'] = 'quality-gated-discovery-v1'
    min_pairs: int = Field(default=20, ge=2, strict=True)
    min_requests: int = Field(default=30, ge=1, strict=True)
    min_assessable_rate: float = Field(default=0.8, ge=0, le=1, allow_inf_nan=False)
    min_valid_probe_rate: float = Field(default=0.8, ge=0, le=1, allow_inf_nan=False)
    quality_margin: float = Field(default=0.05, ge=0, le=1, allow_inf_nan=False)
    reliability_margin: float = Field(default=0.05, ge=0, le=1, allow_inf_nan=False)
    duplicate_margin: float = Field(default=0.05, ge=0, le=1, allow_inf_nan=False)
    max_cost_ratio: float = Field(default=1.2, ge=1, allow_inf_nan=False)
    min_cost_reduction: float = Field(default=0.1, gt=0, lt=1, allow_inf_nan=False)
    min_root_worlds: int = Field(default=2, ge=2, strict=True)
    confidence: float = Field(default=0.95, gt=0.5, lt=1, allow_inf_nan=False)
    bootstrap_samples: int = Field(default=4000, ge=100, strict=True)
    bootstrap_seed: int = Field(default=20261015, ge=0, strict=True)


class ProbeAssessment(FrozenModel):
    turn_id: str
    customer_message: str
    declared_intents: Tuple[dict, ...] = ()
    read_intents: Optional[Tuple[dict, ...]] = Field(default=None, exclude_if=lambda value: value is None)
    information_value: Optional[Tuple[dict, ...]] = Field(default=None, exclude_if=lambda value: value is None)
    visible_ids: Tuple[str, ...] = ()
    prestate_hash: Hash
    probe_validity: Literal['valid', 'invalid', 'unresolved', 'censored']
    reasons: Tuple[str, ...] = ()
    # Each write has its own Reference decision. One Customer request remains
    # one denominator unit, regardless of the number of tools/intentions.
    expected_outcomes: Tuple[dict, ...] = ()
    observed_service_outcomes: Tuple[dict, ...] = ()
    coverage: Tuple[Hash, ...] = ()
    duplicate: bool = False
    source_refs: Tuple[str, ...] = ()


class ProbeEpisode(FrozenModel):
    evaluator: Literal['cloudops-probes-v1', 'cloudops-probes-v2'] = 'cloudops-probes-v1'
    episode_id: str
    evidence_hash: Hash
    config: EpisodeConfig
    customer_snapshot_hash: Hash
    service_snapshot_hash: Hash
    probes: Tuple[ProbeAssessment, ...]
    direct_discoveries: Tuple[dict, ...] = ()
    incidental_discoveries: Tuple[dict, ...] = ()
    authorization_anomalies: Tuple[dict, ...] = ()
    protocol_errors: int = Field(default=0, ge=0)
    customer_calls: int = Field(default=0, ge=0)
    service_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    total_tokens: Optional[int] = Field(default=None, ge=0)
    cost_breakdown: Optional[dict] = Field(default=None, exclude_if=lambda value: value is None)
    termination: str
    replay_verified: bool
    injection_verified: bool
    injected_skill_hashes: Tuple[Hash, ...] = ()
    report_hash: str = ''

    @model_validator(mode='after')
    def seal(self):
        expected = digest(self.model_dump(mode='json', exclude={'report_hash'}))
        if self.report_hash and expected != self.report_hash:
            raise ValueError('Customer probe report hash mismatch')
        object.__setattr__(self, 'report_hash', expected)
        return self


class CustomerGateManifest(FrozenModel):
    schema_version: Literal[1] = 1
    policy: CustomerQualityPolicy
    baseline_customer_hash: Hash
    candidate_customer_hash: Hash
    service_hash: Hash
    required_candidate_skill_hashes: Tuple[Hash, ...] = ()
    configs: Tuple[EpisodeConfig, ...] = Field(min_length=1)
    excluded_world_hashes: Tuple[Hash, ...] = ()
    source_hashes: Dict[str, Hash] = Field(default_factory=dict)
    business_shape_hashes: Tuple[Hash, ...] = ()
    excluded_business_shapes: Tuple[Hash, ...] = ()
    manifest_hash: str = ''

    @model_validator(mode='after')
    def seal(self):
        if any(c.partition != 'development' or c.domain != 'cloudops' for c in self.configs):
            raise ValueError('QGD v1 requires CloudOps development Discovery worlds')
        worlds = [c.initial_world_hash for c in self.configs]
        if len(set(worlds)) != len(worlds):
            raise ValueError('duplicate initial worlds cannot inflate paired sample size')
        if set(worlds) & set(self.excluded_world_hashes):
            raise ValueError('validation world overlaps excluded learning/heldout evidence')
        if self.business_shape_hashes and (len(self.business_shape_hashes) != len(worlds) or
                len(set(self.business_shape_hashes)) != len(worlds)):
            raise ValueError('validation requires distinct business structures, not renamed worlds')
        if set(self.business_shape_hashes) & set(self.excluded_business_shapes):
            raise ValueError('validation business structure overlaps learning/heldout evidence')
        settings = {(c.customer_model.model_dump_json(), c.service_model.model_dump_json(),
                     c.budget.model_dump_json(), c.tools, c.mode) for c in self.configs}
        if len(settings) != 1:
            raise ValueError('models, tools, mode and budgets must be frozen across validation')
        expected = digest(self.model_dump(mode='json', exclude={'manifest_hash'}))
        if self.manifest_hash and expected != self.manifest_hash:
            raise ValueError('Customer gate manifest hash mismatch')
        object.__setattr__(self, 'manifest_hash', expected)
        return self


class CustomerQualityResult(FrozenModel):
    protocol: Literal['quality-gated-discovery-v1', 'quality-gated-discovery-v2'] = 'quality-gated-discovery-v1'
    decision: Literal['ACCEPT', 'REJECT', 'INCONCLUSIVE']
    reasons: Tuple[str, ...]
    manifest_hash: Hash
    baseline: dict
    candidate: dict
    intervals: Dict[str, Tuple[float, float]] = Field(default_factory=dict)
    improvements: Tuple[str, ...] = ()
    baseline_reports: Tuple[Hash, ...] = ()
    candidate_reports: Tuple[Hash, ...] = ()
