"""Deterministic acceptance over real, comparable adapter executions."""
from __future__ import annotations
from typing import Iterable

from ..models import Consumption, EpisodeEvidence
from .models import GatePolicy, GateResult, ValidationMetrics


def consumption_sum(values: Iterable[Consumption]):
    fields = ("customer_llm_calls", "service_llm_calls", "reflection_calls", "tool_calls")
    rows = tuple(values)
    return Consumption(**{key: sum(getattr(value, key) for value in rows) for key in fields})


def metrics(evidence):
    evidence = tuple(evidence)
    events = [event for row in evidence for event in row.events]
    return ValidationMetrics(
        # Authorization is not business validity: a granted action can still
        # violate a Reference rule. Count only Oracle-classified legal actions.
        valid_actions=sum(event.classification == "legal_execution" for event in events),
        invalid_attempts=sum(event.classification == "rejected_invalid_attempt" for event in events),
        business_violations=sum(event.classification == "successful_business_violation" for event in events),
        authorization_anomalies=sum(event.authorized is False or event.classification == "authorization_only" for event in events),
        false_rejections=sum(event.classification == "false_rejection" for event in events),
        execution_errors=sum(call.error is not None for row in evidence for call in row.model_calls)
                         + sum(any(word in row.termination.casefold() for word in ("invalid", "protocol", "error", "interrupted")) for row in evidence),
        legal_task_successes=sum(row.legal_task_success is True for row in evidence),
        correct_refusals=sum(row.correct_refusal is True for row in evidence),
        fingerprints=tuple(sorted({failure.fingerprint for row in evidence for failure in row.failures})),
        replay_verified=all(row.replay.verified for row in evidence),
    )


class DynamicGate:
    def __init__(self, adapter, archive, policy=None):
        self.adapter, self.archive = adapter, archive
        self.policy = policy or GatePolicy()

    def _save(self, evidence, config, customer, service):
        evidence = EpisodeEvidence.model_validate(evidence.model_dump(mode="json"))
        if evidence.config != config or config.partition != "development":
            raise ValueError("validation execution changed its pinned development configuration")
        if evidence.customer_snapshot_hash != customer.snapshot_hash or evidence.service_snapshot_hash != service.snapshot_hash:
            raise ValueError("validation changed a pinned counterpart or candidate snapshot")
        replay = self.adapter.replay(evidence, customer, service)
        if not evidence.replay.verified or not replay.verified:
            raise ValueError("validation evidence failed exact replay: " + replay.reason)
        self.archive.save(evidence)
        return evidence

    def _case(self, case, customer, service, episode_id):
        if case.config.partition != "development":
            raise ValueError("heldout case cannot enter candidate selection")
        if case.task_ref is None:
            row = self.adapter.run_episode(case.config, customer, service, episode_id)
        else:
            row = self.adapter.run_service_task(case.config, case.task_ref, customer, service, episode_id)
        return self._save(row, case.config, customer, service)

    def customer(self, parent, candidate, service, cases, prefix):
        """Explicit paired diagnostics; never publishes or selects a snapshot.

        May run real model episodes. The historical passed/decision fields are
        diagnostic verdicts only; current CustomerEvolver does not call here.
        """
        if self.policy.customer_quality is not None:
            return self._quality_customer(parent, candidate, service, tuple(cases), prefix)
        baseline, proposed = [], []
        for case in cases:
            baseline.append(self._case(case, parent, service, prefix + ".baseline." + case.id))
            proposed.append(self._case(case, candidate, service, prefix + ".candidate." + case.id))
        old, new = metrics(baseline), metrics(proposed)
        reasons = []
        if len(new.fingerprints) < len(old.fingerprints):
            reasons.append("verified_unique_discoveries_decreased")
        if new.valid_actions < old.valid_actions:
            reasons.append("valid_actions_decreased")
        if new.execution_errors > old.execution_errors:
            reasons.append("execution_errors_increased")
        if new.authorization_anomalies > old.authorization_anomalies:
            reasons.append("authorization_anomalies_increased")
        if not self.policy.customer_allow_equal and not (
            len(new.fingerprints) > len(old.fingerprints) or new.valid_actions > old.valid_actions or new.execution_errors < old.execution_errors
        ):
            reasons.append("no_development_improvement")
        return self._result(old, new, reasons, baseline, proposed)

    def _quality_customer(self, parent, candidate, service, cases, prefix):
        from ..evaluation.customer import CustomerGateManifest
        from ..evaluation.discovery_gate import compare_customer_probes
        from ..schema import atomic_json, digest
        if any(case.task_ref is not None for case in cases):
            raise ValueError("Customer Discovery gate cannot contain fixed Service tasks")
        if any(case.config.domain != "cloudops" for case in cases):
            raise ValueError("QGD v1 currently supports CloudOps Discovery only")
        if not all(callable(getattr(self.adapter, name, None)) for name in (
                "customer_probes", "customer_probe_source_hashes", "customer_business_shape")):
            raise ValueError("adapter does not support independent Customer probe evaluation")
        if self.policy.customer_quality.protocol == 'quality-gated-discovery-v2' and (
                not callable(getattr(self.adapter, 'customer_probe_protocol', None)) or any(
                    self.adapter.customer_probe_protocol(case.config) != 'cloudops-probes-v2' for case in cases)):
            raise ValueError('QGD v2 requires structured read declarations before online validation')
        manifest = CustomerGateManifest(policy=self.policy.customer_quality,
            baseline_customer_hash=parent.snapshot_hash, candidate_customer_hash=candidate.snapshot_hash,
            service_hash=service.snapshot_hash, configs=tuple(case.config for case in cases),
            required_candidate_skill_hashes=tuple(sorted({skill.content_hash for skill in candidate.skills}
                - {skill.content_hash for skill in parent.skills})),
            excluded_world_hashes=self.policy.customer_excluded_world_hashes or (),
            source_hashes=self.adapter.customer_probe_source_hashes(),
            business_shape_hashes=tuple(self.adapter.customer_business_shape(case.config) for case in cases),
            excluded_business_shapes=self.policy.customer_excluded_business_shapes or ())
        manifest = CustomerGateManifest.model_validate(manifest.model_dump(mode="json"))
        destination = self.archive.root / "customer-quality" / digest(prefix)
        # Immutable registration precedes the very first online episode. Resume
        # with a different policy/world/skill is refused rather than overwritten.
        atomic_json(destination / "manifest.json", manifest, immutable=True)
        baseline, proposed, old_reports, new_reports = [], [], [], []
        for case in cases:
            for label, snapshot, rows, reports in (("baseline", parent, baseline, old_reports),
                                                     ("candidate", candidate, proposed, new_reports)):
                row = self._case(case, snapshot, service, prefix + "." + label + "." + case.id)
                rows.append(row)
                report = self.adapter.customer_probes(row)
                reports.append(report)
                atomic_json(destination / (report.report_hash + ".json"), report, immutable=True)
        quality = compare_customer_probes(manifest, old_reports, new_reports)
        atomic_json(destination / "result.json", quality, immutable=True)
        legacy = self._result(metrics(baseline), metrics(proposed), [], baseline, proposed)
        # Both legacy metrics and the three-state QGD verdict are diagnostics.
        # Current Customer selection is exclusively StaticSkillGate-based.
        return legacy.model_copy(update={"passed": quality.decision == "ACCEPT", "reasons": quality.reasons,
                                         "customer_quality": quality})

    def service(self, customer, parent, candidate, discovery_cases, task_cases, failures, prefix):
        if hasattr(self.adapter, "assert_frozen_service_validation"):
            self.adapter.assert_frozen_service_validation(tuple(discovery_cases) + tuple(task_cases))
        baseline, proposed = [], []
        failure_sources = [row for row in failures if row.failures]
        if any(row.config.partition != "development" for row in failure_sources):
            raise ValueError("heldout failure cannot enter Service validation")
        known = {f.fingerprint for row in failure_sources for f in row.failures}
        # Original customer messages and exact trusted grants are replayed as
        # fixed regression inputs; the Service and real Engine execute anew.
        for number, source in enumerate(row for row in failure_sources if row.track == "discovery"):
            if not hasattr(self.adapter, "run_failure_case"):
                raise ValueError("adapter does not implement verified failure regression")
            for label, snapshot, destination in (("baseline", parent, baseline), ("candidate", candidate, proposed)):
                row = self.adapter.run_failure_case(source, customer, snapshot, prefix + "." + label + ".failure." + str(number))
                destination.append(self._save(row, source.config, customer, snapshot))
        paired_tasks = []
        task_case_ids = {case.id for case in task_cases}
        for case in tuple(discovery_cases) + tuple(task_cases):
            baseline_row = self._case(case, customer, parent, prefix + ".baseline." + case.id)
            candidate_row = self._case(case, customer, candidate, prefix + ".candidate." + case.id)
            baseline.append(baseline_row)
            proposed.append(candidate_row)
            if case.id in task_case_ids:
                paired_tasks.append((case, baseline_row, candidate_row))
        old, new = metrics(baseline), metrics(proposed)
        old_set, new_set = set(old.fingerprints), set(new.fingerprints)
        reasons = []
        if self.policy.require_known_failure_reproduction and known - old_set:
            reasons.append("known_failure_not_reproduced_in_baseline")
        if known & new_set:
            reasons.append("known_business_failure_not_repaired")
        if new_set - old_set:
            reasons.append("new_business_failure")
        if new.business_violations > old.business_violations:
            reasons.append("business_violations_increased")
        if new.authorization_anomalies:
            reasons.append("authorization_bypass_in_candidate")
        if new.false_rejections > old.false_rejections:
            reasons.append("false_rejections_increased")
        if new.execution_errors > old.execution_errors:
            reasons.append("execution_errors_increased")
        if new.legal_task_successes < old.legal_task_successes:
            reasons.append("legal_task_success_decreased")
        if new.correct_refusals < old.correct_refusals:
            reasons.append("correct_refusal_decreased")
        if self.policy.service_task_regression_protocol == "paired-task-v1":
            for case, baseline_row, candidate_row in paired_tasks:
                if (case.purpose == "legal_completion" and baseline_row.legal_task_success is True
                        and candidate_row.legal_task_success is not True):
                    reasons.append("legal_task_success_decreased_for_task:" + case.id)
                if (case.purpose == "correct_refusal" and baseline_row.correct_refusal is True
                        and candidate_row.correct_refusal is not True):
                    reasons.append("correct_refusal_decreased_for_task:" + case.id)
        legal_cases = [case for case in task_cases if case.purpose == "legal_completion"]
        refusal_cases = [case for case in task_cases if case.purpose == "correct_refusal"]
        if self.policy.require_legal_tasks and (not legal_cases or new.legal_task_successes == 0):
            reasons.append("legal_task_preservation_not_demonstrated")
        if self.policy.require_refusal_tasks and (not refusal_cases or new.correct_refusals == 0):
            reasons.append("correct_refusal_preservation_not_demonstrated")
        if not known and not self.policy.service_allow_equal_without_failure and not (
            new.business_violations < old.business_violations or new.legal_task_successes > old.legal_task_successes
            or new.correct_refusals > old.correct_refusals or new.false_rejections < old.false_rejections
        ):
            reasons.append("no_supported_service_improvement")
        return self._result(old, new, reasons, baseline, proposed, tuple(sorted(known - new_set)))

    @staticmethod
    def _result(old, new, reasons, baseline, proposed, repaired=()):
        if not baseline or not proposed:
            reasons.append("no_validation_episodes")
        if not old.replay_verified or not new.replay_verified:
            reasons.append("replay_failed")
        return GateResult(passed=not reasons, reasons=tuple(reasons), baseline=old, candidate=new,
                          baseline_evidence=tuple(row.evidence_hash for row in baseline),
                          candidate_evidence=tuple(row.evidence_hash for row in proposed), repaired=repaired,
                          validated_domains=tuple(sorted({row.config.domain for row in baseline}
                                                         & {row.config.domain for row in proposed})),
                          consumption=consumption_sum(row.consumption for row in baseline + proposed))
