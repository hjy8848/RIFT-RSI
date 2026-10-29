"""Formal research-start gate; never a Skill selection/performance gate."""
from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Literal, Optional, Tuple

from pydantic import Field, model_validator

from usami_v2.contracts import Sealed


class ReadinessPolicy(Sealed):
    protocol: Literal["commerce-readiness-policy-v1"] = "commerce-readiness-policy-v1"
    status: Literal["DRAFT", "FROZEN"] = "DRAFT"
    environment_hash: str
    model_hash: str
    distribution_hash: str
    # id, pre-registered N, population weight.
    strata: Tuple[Tuple[str, int, int], ...] = ()
    confidence_level: Optional[float] = Field(default=None, gt=0.5, lt=1)
    required_half_width: Optional[float] = Field(default=None, gt=0, lt=0.5)
    maximum_nonbusiness_rate: Optional[float] = Field(default=None, ge=0, lt=1)
    maximum_customer_failure_rate: Optional[float] = Field(default=None, ge=0, lt=1)
    maximum_unresolved_rate: Optional[float] = Field(default=None, ge=0, lt=1)
    minimum_opportunity_rate: Optional[float] = Field(default=None, gt=0, le=1)
    minimum_verified_error_episodes: Optional[int] = Field(default=None, ge=1, strict=True)
    maximum_model_calls: Optional[int] = Field(default=None, ge=1, strict=True)
    maximum_tokens: Optional[int] = Field(default=None, ge=1, strict=True)
    maximum_wall_seconds: Optional[int] = Field(default=None, ge=1, strict=True)
    sample_size_justification: str = ""
    required_contract_checks: Tuple[str, ...] = (
        "multiple_consequences", "reference_execution", "communication",
        "consent", "customer_contract", "role_visibility", "replay",
    )
    analysis_plan: Literal["fixed_sample_one_look"] = "fixed_sample_one_look"
    confidence_method: Literal["clopper-pearson-bonferroni-v1"] = "clopper-pearson-bonferroni-v1"

    @model_validator(mode="after")
    def frozen_policy_complete(self):
        if self.strata:
            if (len({row[0] for row in self.strata}) != len(self.strata)
                    or any(type(n) is not int or type(w) is not int or n < 1 or w < 1
                           for _, n, w in self.strata)):
                raise ValueError("unique strata with positive integer N and weights required")
        if len(set(self.required_contract_checks)) != len(self.required_contract_checks) or not self.required_contract_checks:
            raise ValueError("contract suite must be explicit and unique")
        if self.status == "FROZEN":
            fields = ("confidence_level", "required_half_width", "maximum_nonbusiness_rate",
                      "maximum_customer_failure_rate", "maximum_unresolved_rate",
                      "minimum_opportunity_rate", "minimum_verified_error_episodes",
                      "maximum_model_calls", "maximum_tokens", "maximum_wall_seconds")
            if not self.strata or not self.sample_size_justification.strip() or any(
                    getattr(self, name) is None for name in fields):
                raise ValueError("frozen readiness policy cannot contain missing sampling/budget/threshold fields")
            if self.minimum_verified_error_episodes > sum(n for _, n, _ in self.strata):
                raise ValueError("requested learnable error count exceeds the sample plan")
        return self


class ReferenceContractCheck(Sealed):
    check_id: str
    passed: bool
    evidence_hash: str


class ReadinessEpisode(Sealed):
    episode_id: str
    world_hash: str
    independence_key: str
    stratum_id: str
    environment_hash: str
    model_hash: str
    distribution_hash: str
    origin: Literal["live_provider", "scripted_fixture"]
    purpose: Literal["smoke", "formal_readiness"]
    replay_hash: str
    provider_call_hashes: Tuple[str, ...] = ()
    decision_opportunity_proof_refs: Tuple[str, ...] = ()
    verified_business_error_refs: Tuple[str, ...] = ()
    nonbusiness_failure_refs: Tuple[str, ...] = ()
    customer_failure_refs: Tuple[str, ...] = ()
    unresolved_refs: Tuple[str, ...] = ()
    critical_integrity_or_visibility_failure: bool = False
    model_calls: int = Field(ge=0, strict=True)
    known_tokens: int = Field(ge=0, strict=True)
    usage_complete: bool
    wall_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def live_call_records(self):
        if self.origin == "live_provider" and (
                not self.provider_call_hashes or self.model_calls != len(self.provider_call_hashes)):
            raise ValueError("live evidence needs recorded provider calls")
        return self


class RateBound(Sealed):
    name: str
    lower: float
    upper: float
    point: float
    independent_worlds: int


class ReadinessReport(Sealed):
    policy_hash: str
    decision: Literal["PASS", "FAIL", "INCONCLUSIVE"]
    reasons: Tuple[str, ...]
    mode: Literal["formal", "contract_test"]
    formal_learning_allowed: bool
    evidence_hashes: Tuple[str, ...]
    reference_evidence_hashes: Tuple[str, ...]
    rate_bounds: Tuple[RateBound, ...]
    verified_error_episodes: int
    model_calls: int
    known_tokens: int
    unknown_usage_episodes: int
    elapsed_wall_seconds: float
    selection_power: Literal["research_start_only"] = "research_start_only"


def _binomial_cdf(k, n, p):
    if k < 0:
        return 0.0
    if k >= n or p == 0:
        return 1.0
    if p == 1:
        return 0.0
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + i * math.log(p) + (n - i) * math.log1p(-p) for i in range(k + 1)]
    largest = max(logs)
    return min(1.0, math.exp(largest) * sum(math.exp(value - largest) for value in logs))


def binomial_interval(successes: int, n: int, alpha: float):
    """Two-sided Clopper–Pearson endpoints via bounded numerical bisection."""
    if type(successes) is not int or type(n) is not int or not 0 <= successes <= n or n < 1 or not 0 < alpha < 1:
        raise ValueError("invalid binomial counts or alpha")
    lower, upper = 0.0, 1.0
    if successes > 0:
        lo, hi = 0.0, 1.0
        for _ in range(64):
            mid = (lo + hi) / 2
            if 1 - _binomial_cdf(successes - 1, n, mid) < alpha / 2:
                lo = mid
            else:
                hi = mid
        lower = (lo + hi) / 2
    if successes < n:
        lo, hi = 0.0, 1.0
        for _ in range(64):
            mid = (lo + hi) / 2
            if _binomial_cdf(successes, n, mid) > alpha / 2:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2
    return lower, upper


def check_readiness(policy: ReadinessPolicy, episodes, reference_checks, *,
                    verify_episode: Callable[[ReadinessEpisode], bool],
                    verify_reference: Callable[[ReferenceContractCheck], bool],
                    elapsed_wall_seconds: float,
                    mode: Literal["formal", "contract_test"] = "formal"):
    policy = policy.checked()
    if policy.status != "FROZEN":
        raise ValueError("formal readiness requires an explicitly frozen complete policy")
    if mode not in ("formal", "contract_test"):
        raise ValueError("unknown readiness execution mode")
    if not math.isfinite(elapsed_wall_seconds) or elapsed_wall_seconds < 0:
        raise ValueError("actual nonnegative elapsed run wall time required")
    episodes = tuple(row.checked() for row in episodes)
    references = tuple(row.checked() for row in reference_checks)
    reasons, hard = [], []
    reference_ids = {row.check_id for row in references}
    if len(reference_ids) != len(references):
        raise ValueError("duplicate reference suite checks")
    if set(policy.required_contract_checks) != reference_ids:
        reasons.append("reference_suite_incomplete_or_unregistered")
    if any(not row.passed or not row.evidence_hash or not verify_reference(row) for row in references):
        hard.append("reference_contract_or_integrity_failure")
    if len({row.episode_id for row in episodes}) != len(episodes):
        raise ValueError("duplicate Episode IDs")
    if len({row.world_hash for row in episodes}) != len(episodes) or len({
            row.independence_key for row in episodes}) != len(episodes):
        raise ValueError("correlated/repeated worlds are not independent readiness samples")
    planned = {name: n for name, n, _ in policy.strata}
    for row in episodes:
        if (row.environment_hash, row.model_hash, row.distribution_hash) != (
                policy.environment_hash, policy.model_hash, policy.distribution_hash):
            raise ValueError("readiness evidence differs from frozen environment/model/distribution")
        if row.stratum_id not in planned:
            raise ValueError("unregistered readiness sampling stratum")
        if row.critical_integrity_or_visibility_failure or not row.replay_hash or not verify_episode(row):
            hard.append("episode_integrity_or_visibility_failure")
        if row.purpose != "formal_readiness":
            reasons.append("smoke_is_not_formal_readiness")
        if mode == "formal" and row.origin != "live_provider":
            reasons.append("scripted_evidence_is_not_a_real_S0_baseline")
    counts = Counter(row.stratum_id for row in episodes)
    if any(counts[name] > n for name, n in planned.items()):
        raise ValueError("unregistered sample extension beyond frozen N")
    if counts != Counter(planned):
        reasons.append("formal_sample_plan_incomplete")
    calls = sum(row.model_calls for row in episodes)
    tokens = sum(row.known_tokens for row in episodes)
    unknown_usage = sum(not row.usage_complete for row in episodes)
    if calls > policy.maximum_model_calls or tokens > policy.maximum_tokens:
        hard.append("registered_model_budget_exceeded")
    # Concurrent episode durations are not elapsed run time.
    if elapsed_wall_seconds > policy.maximum_wall_seconds:
        hard.append("registered_wall_budget_exceeded")
    if unknown_usage:
        reasons.append("unknown_usage_prevents_certifying_total_budget")
    metrics = {
        "nonbusiness": lambda row: bool(row.nonbusiness_failure_refs),
        "customer_failure": lambda row: bool(row.customer_failure_refs),
        "unresolved": lambda row: bool(row.unresolved_refs),
        "decision_opportunity": lambda row: bool(row.decision_opportunity_proof_refs),
    }
    bounds = []
    # Simultaneous per-stratum bounds. No independence of actions is assumed.
    adjusted_alpha = (1 - policy.confidence_level) / (len(metrics) * len(policy.strata))
    weight_total = sum(weight for _, _, weight in policy.strata)
    if all(counts[name] for name in planned):
        for name, predicate in metrics.items():
            lower, upper, point = 0.0, 0.0, 0.0
            for stratum, _, weight in policy.strata:
                rows = [row for row in episodes if row.stratum_id == stratum]
                count = sum(predicate(row) for row in rows)
                lo, hi = binomial_interval(count, len(rows), adjusted_alpha)
                fraction = weight / weight_total
                lower += fraction * lo
                upper += fraction * hi
                point += fraction * count / len(rows)
            bounds.append(RateBound(name=name, lower=lower, upper=upper, point=point,
                                    independent_worlds=len(episodes)))
        limits = {"nonbusiness": policy.maximum_nonbusiness_rate,
                  "customer_failure": policy.maximum_customer_failure_rate,
                  "unresolved": policy.maximum_unresolved_rate}
        for bound in bounds:
            if (bound.upper - bound.lower) / 2 > policy.required_half_width:
                reasons.append("confidence_precision_insufficient:" + bound.name)
            if bound.name in limits and bound.upper > limits[bound.name]:
                (hard if bound.lower > limits[bound.name] else reasons).append(
                    "nonbusiness_reliability_not_established:" + bound.name)
            if bound.name == "decision_opportunity" and bound.lower < policy.minimum_opportunity_rate:
                reasons.append("decision_opportunity_coverage_not_established")
    errors = sum(bool(row.verified_business_error_refs) for row in episodes)
    if errors < policy.minimum_verified_error_episodes:
        reasons.append("insufficient_natural_learnable_business_errors")
    decision = "FAIL" if hard else "INCONCLUSIVE" if reasons else "PASS"
    return ReadinessReport(
        policy_hash=policy.record_hash, decision=decision,
        reasons=tuple(sorted(set(hard + reasons))), mode=mode,
        formal_learning_allowed=decision == "PASS" and mode == "formal",
        evidence_hashes=tuple(row.record_hash for row in episodes),
        reference_evidence_hashes=tuple(row.record_hash for row in references),
        rate_bounds=tuple(bounds), verified_error_episodes=errors, model_calls=calls,
        known_tokens=tokens, unknown_usage_episodes=unknown_usage,
        elapsed_wall_seconds=elapsed_wall_seconds)
