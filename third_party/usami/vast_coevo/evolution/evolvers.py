"""Single-patch evolution: static Customer; Service gated by default, explicitly optional."""
from __future__ import annotations
import json

from ..models import Consumption, EpisodeEvidence
from ..reflection.reflector import ReflectionInput
from ..reflection.scope import DomainPolicy
from ..skills.library import apply_patch
from ..skills.validation import StaticSkillGate
from .gates import DynamicGate, consumption_sum
from .models import DomainCoverage, EvolutionRecord, GatePolicy


class Evolver:
    def __init__(self, role, adapter, reflector, library_store, archive, policy=None):
        self.role, self.adapter, self.reflector = role, adapter, reflector
        self.library_store, self.archive = library_store, archive
        self.policy = policy or GatePolicy()

    def update(self, generation, customer, service, evidence, discovery_cases, service_task_cases=(), episode_prefix=None, *, dynamic_gate=True):
        discovery_cases, service_task_cases = tuple(discovery_cases), tuple(service_task_cases)
        parent = customer if self.role == "customer" else service
        evidence = tuple(EpisodeEvidence.model_validate(row.model_dump(mode="json")) for row in evidence)
        if any(row.config.partition != "development" for row in evidence):
            raise ValueError("heldout evidence cannot influence evolution")
        for row in evidence:
            checked = self.adapter.replay(row, self.library_store.load(row.customer_snapshot_hash),
                                          self.library_store.load(row.service_snapshot_hash))
            if not checked.verified:
                raise ValueError("evolution evidence failed independent replay: " + checked.reason)
        cases = tuple(discovery_cases) + tuple(service_task_cases)
        gate_policy = self.policy
        if not discovery_cases:
            raise ValueError("evolution requires at least one Discovery validation case")
        if any(case.config.partition != "development" for case in cases):
            raise ValueError("heldout cases cannot influence evolution")
        prefix = episode_prefix or ("g" + str(generation) + "." + self.role + ".validation")
        preflight = []
        if self.role == "service" and dynamic_gate:
            # The actual normal-task results are part of the one Reflection,
            # including legal completion and correct refusal, not only attacks.
            gate = DynamicGate(self.adapter, self.archive, gate_policy)
            for case in service_task_cases:
                preflight.append(gate._case(case, customer, service, prefix + ".reflection-task." + case.id))
            evidence += tuple(preflight)
        tools = tuple(sorted({name for case in cases for name in self.adapter.public_capabilities(case.config)}))
        forbidden = tuple(sorted({name for case in cases for name in (
            self.adapter.forbidden_identifiers(case.config) if hasattr(self.adapter, "forbidden_identifiers") else ())}))
        policies = [self.adapter.public_policy(case.config) for case in cases if hasattr(self.adapter, "public_policy")]
        value = ReflectionInput(generation=generation, role=self.role, library=parent, evidence=evidence,
                                interaction_views=tuple(self.adapter.reflection_view(row) for row in evidence)
                                if hasattr(self.adapter, "reflection_view") else None,
                                domain_policy=DomainPolicy(add_domains=tuple(sorted({case.config.domain for case in cases}))),
                                allowed_tools=tools, public_policy=json.dumps(policies, ensure_ascii=False, sort_keys=True))
        reflection = self.reflector.reflect(value)
        costs = consumption_sum((Consumption(reflection_calls=reflection.calls), *(row.consumption for row in preflight)))
        fields = dict(generation=generation, role=self.role, parent_customer_hash=customer.snapshot_hash,
                      parent_service_hash=service.snapshot_hash, reflection=reflection, patch=reflection.patch,
                      selected_snapshot_hash=parent.snapshot_hash)
        if self.role == 'service' and not dynamic_gate:
            fields.update(skill_updated=False,selection_mode='evidence_only_static',performance_validated=False,performance_improved='unknown')
        if self.role == "customer":
            fields["customer_selection"] = "customer-static-only-v1"
        if reflection.error or reflection.patch is None:
            return EvolutionRecord(**fields, outcome="rejected", reasons=(reflection.error or "missing_patch",), consumption=costs)
        declared = reflection.patch.skill.scope.domains if reflection.patch.skill is not None else ()
        fields["domain_coverage"] = DomainCoverage(declared_domains=declared, validated_domains=(), unvalidated_domains=declared)
        static = StaticSkillGate(allowed_tools=tools, forbidden_identifiers=forbidden,
                                 allowed_domains={case.config.domain for case in cases}).validate(parent, reflection.patch)
        fields["static_validation"] = static
        if not static.passed:
            return EvolutionRecord(**fields, outcome="rejected", reasons=static.reasons, consumption=costs)
        if reflection.patch.action == "NOOP":
            return EvolutionRecord(**fields, outcome="noop", reasons=("explicit_noop",), consumption=costs)
        try:
            candidate = apply_patch(parent, reflection.patch, source=prefix + ".reflection")
            self.library_store.save(candidate, publish=False)
            fields["candidate_snapshot_hash"] = candidate.snapshot_hash
            if self.role == "customer" or not dynamic_gate:
                # Publication still verifies immutable contents/version collisions.
                # Performance diagnostics have no place in this selection path.
                self.library_store.save(candidate)
                fields["selected_snapshot_hash"] = candidate.snapshot_hash
                if self.role=='service':fields['skill_updated']=True
                return EvolutionRecord(**fields, outcome="accepted",
                                       reasons=("static_gate_passed_snapshot_updated",), consumption=costs)
            if (self.role == 'service' and gate_policy.customer_quality is not None and
                    gate_policy.customer_quality.protocol == 'quality-gated-discovery-v2' and
                    hasattr(self.adapter, 'service_scope_risks') and self.adapter.service_scope_risks(candidate)):
                return EvolutionRecord(**fields, outcome='rejected', reasons=('domain_specific_service_guidance_crosses_scope',), consumption=costs)
            gate = DynamicGate(self.adapter, self.archive, gate_policy)
            result = gate.service(customer, parent, candidate, discovery_cases, service_task_cases, evidence, prefix)
            fields["dynamic_validation"] = result
            validated = result.validated_domains or ()
            fields["domain_coverage"] = DomainCoverage(declared_domains=declared, validated_domains=validated,
                                                     unvalidated_domains=tuple(domain for domain in declared if domain not in validated))
            costs = consumption_sum((costs, result.consumption))
            if not result.passed:
                inconclusive = result.customer_quality is not None and result.customer_quality.decision == "INCONCLUSIVE"
                return EvolutionRecord(**fields, outcome="inconclusive" if inconclusive else "rejected",
                                       reasons=result.reasons, consumption=costs)
            self.library_store.save(candidate)
        except (ValueError, TypeError, OSError) as exc:
            return EvolutionRecord(**fields, outcome="rejected", reasons=("candidate_validation_error: " + str(exc),), consumption=costs)
        fields["selected_snapshot_hash"] = candidate.snapshot_hash
        return EvolutionRecord(**fields, outcome="accepted", reasons=("development_gate_passed",), consumption=costs)


class CustomerEvolver(Evolver):
    def __init__(self, adapter, reflector, library_store, archive, policy=None):
        super().__init__("customer", adapter, reflector, library_store, archive, policy)


class ServiceEvolver(Evolver):
    def __init__(self, adapter, reflector, library_store, archive, policy=None):
        super().__init__("service", adapter, reflector, library_store, archive, policy)
