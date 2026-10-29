"""Actual Evolvers + four-domain Engine/Oracle gates, with labelled CI LLM fixtures.

The response fixtures are not a new live LLM experiment. World execution,
Oracle, independent goal scoring, authorization and both gate reruns are real.
"""
import copy
import pytest

from integrations.usami import ServiceTaskProfile, UsamiProfile, UsamiAdapter
from integrations.usami.domains import native_task
from vast_coevo.evidence.archive import EvidenceArchive
from vast_coevo.evolution.evolvers import CustomerEvolver, ServiceEvolver
from vast_coevo.evolution.models import ValidationCase
from vast_coevo.models import Budget
from vast_coevo.reflection.reflector import CustomerReflector, ServiceReflector
from vast_coevo.skills.library import LibrarySnapshot, LibraryStore
from vast_coevo.skills.models import SkillPatch
from test_vast_evolution import PatchClient
from test_vast_usami import SPEC, LIMITS, ScriptedClient, answer, call, libraries, scripts, service_arguments


@pytest.mark.parametrize("domain,legal,refusal", [
    ("saas", "seat_activate", "owner_refusal"),
    ("cloudops", "dependency_boot", "viewer_refusal"),
    ("fulfillment", "paid_fulfillment", "stock_refusal"),
    ("stayops", "booking_confirmation", "deposit_refusal"),
])
def test_customer_static_selection_and_service_real_failure_and_normal_task_gates(tmp_path, domain, legal, refusal):
    identity = "user_001" if domain == "saas" else "u2" if domain == "cloudops" else "u0"
    profile = UsamiProfile(config_id=domain + "-real-gate", domain=domain, seed=42, identity=identity, mode="sandbox")
    tasks = [ServiceTaskProfile(task_ref=domain + "." + kind, config_id=profile.config_id, scenario=scenario)
             for kind, scenario in (("legal", legal), ("refusal", refusal))]
    task_responses = {}
    native_script = scripts(domain)
    corrected = native_script["service"][:2] + [answer("The required prerequisite is unsatisfied; I have not executed the write."),
                                                answer("Your question grants no exact write operation.")]

    def factory(spec, role, config, episode_id):
        if role == "customer":
            return ScriptedClient(spec, native_script[role])
        if config.config_hash in task_responses:
            replies = task_responses[config.config_hash]
        else:
            # Explicitly labelled CI fixture for this one candidate, not an
            # algorithm-side script or a claimed online model improvement.
            replies = corrected if ".service.candidate." in episode_id else native_script[role]
        return ScriptedClient(spec, replies)

    adapter = UsamiAdapter(tmp_path / "native", (profile,), tasks, factory)
    config = adapter.config_for(profile.config_id, SPEC, SPEC, LIMITS)
    cases = (ValidationCase(id=domain + "-discovery", config=config),)
    task_cases = []
    task_limits = Budget(max_customer_turns=1, max_customer_llm_calls=0, max_service_llm_calls=32, max_service_tool_calls=48)
    for task_profile in tasks:
        _, task, _ = native_task(profile, task_profile)
        frozen = adapter.config_for(profile.config_id, SPEC, SPEC, task_limits, task_ref=task_profile.task_ref)
        if task.kind == "completion":
            replies = []
            for index, operation in enumerate(task.witnesses[0].operations):
                raw = operation.model_dump(mode="json") if hasattr(operation, "model_dump") else operation
                replies.append(answer(calls=[call("normal_" + str(index), raw["name"], service_arguments(domain, raw))]))
            replies.append(answer("The actual business outcome is complete."))
            purpose = "legal_completion"
        else:
            code = task.policy_certificate.error_code if domain == "saas" else task.refusal_code
            replies, purpose = [answer("This request cannot be completed under the applicable policy: " + code)], "correct_refusal"
        task_responses[frozen.config_hash] = replies
        task_cases.append(ValidationCase(id=task_profile.task_ref, config=frozen, task_ref=task_profile.task_ref, purpose=purpose))

    customer, service = libraries(domain)
    # The same reusable applicability declaration is validated in one domain at
    # a time. This reproduces the live pilot's four-domain/CloudOps-only mismatch.
    domains = ("saas", "cloudops", "fulfillment", "stayops")
    def shared(snapshot):
        raw = snapshot.skills[0].model_dump(mode="json", exclude={"content_hash"})
        raw["scope"]["domains"] = domains
        return LibrarySnapshot(role=snapshot.role, skills=(type(snapshot.skills[0]).model_validate(raw),))
    customer, service = shared(customer), shared(service)
    store, archive = LibraryStore(tmp_path / "libraries"), EvidenceArchive(tmp_path / "evidence")
    store.save(customer); store.save(service)
    source = adapter.run_episode(config, customer, service, domain + ".source")
    assert source.failures and source.replay.verified
    archive.save(source)

    c_raw = customer.skills[0].model_dump(mode="json", exclude={"content_hash"})
    c_raw.update(version=2, description="Use actual public feedback before choosing the next independent request.")
    c_patch = SkillPatch(action="UPDATE", role="customer", base_snapshot_hash=customer.snapshot_hash,
                         skill=type(customer.skills[0]).model_validate(c_raw), rationale="One fixture proposal for real gate integration.",
                         evidence_ids=(source.episode_id,))
    c_client = PatchClient(c_patch)
    c_record = CustomerEvolver(adapter, CustomerReflector(c_client, SPEC), store, archive).update(
        0, customer, service, (source,), cases, task_cases, domain + ".customer")
    assert c_record.outcome == "accepted", c_record.reasons
    assert c_record.dynamic_validation is None and len(c_client.requests) == 1
    assert c_record.domain_coverage.declared_domains == domains
    assert c_record.domain_coverage.validated_domains == ()
    assert c_record.domain_coverage.unvalidated_domains == domains
    selected_customer = store.load(c_record.selected_snapshot_hash)

    s_raw = service.skills[0].model_dump(mode="json", exclude={"content_hash"})
    s_raw.update(version=2, procedure=["Inspect the public state prerequisites and refuse an unsatisfied write without changing state.",
                                      "Preserve exact write grants and complete legal requests with the original tools."])
    s_patch = SkillPatch(action="UPDATE", role="service", base_snapshot_hash=service.snapshot_hash,
                         skill=type(service.skills[0]).model_validate(s_raw), rationale="One fixture proposal for real regression integration.",
                         evidence_ids=(source.episode_id,))
    s_client = PatchClient(s_patch)
    s_record = ServiceEvolver(adapter, ServiceReflector(s_client, SPEC), store, archive).update(
        0, selected_customer, service, (source,), cases, task_cases, domain + ".service")
    assert s_record.outcome == "accepted", s_record.reasons
    gate = s_record.dynamic_validation
    assert gate.passed and gate.repaired and len(s_client.requests) == 1
    assert s_record.domain_coverage.declared_domains == domains
    assert s_record.domain_coverage.validated_domains == (domain,)
    assert gate.baseline.business_violations > 0 and gate.candidate.business_violations == 0
    assert gate.candidate.authorization_anomalies == 0
    assert gate.candidate.legal_task_successes == gate.candidate.correct_refusals == 1
    assert s_record.selected_snapshot_hash != service.snapshot_hash
    selected_service = store.load(s_record.selected_snapshot_hash)
    for evidence_hash in gate.candidate_evidence:
        row = archive.load(evidence_hash)
        assert row.replay.verified
        assert row.customer_snapshot_hash == selected_customer.snapshot_hash
        assert row.service_snapshot_hash == selected_service.snapshot_hash
        assert adapter.replay(row, selected_customer, selected_service).verified
