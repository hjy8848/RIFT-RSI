"""Program-managed applicability and execution-derived coverage, with no LLM claims."""
import json
from pathlib import Path
import tarfile

import pytest

from vast_coevo.evolution.evolvers import CustomerEvolver
from vast_coevo.evolution.gates import DynamicGate
from vast_coevo.reflection.reflector import CustomerReflector, ReflectionInput
from vast_coevo.reflection.scope import DomainPolicy, proposal_json_schema
from vast_coevo.skills import LibrarySnapshot, Scope, StaticSkillGate, apply_patch
from vast_coevo.skills.retriever import SkillContext, SkillRetriever
from vast_coevo.storage.runs import StoredEvolution
from test_vast_evolution import MODEL, customer_skill, proposal, setup
from test_vast_reflection import Client, completion


DOMAINS = ("generic", "other", "third", "fourth")


def scoped_skill(version=1, domains=DOMAINS):
    raw = customer_skill(version).model_dump(mode="json", exclude={"content_hash"})
    raw["scope"]["domains"] = domains
    return type(customer_skill()).model_validate(raw)


@pytest.mark.parametrize("omit", ["scope", "domains", "nothing"])
def test_update_inherits_four_domains_without_claiming_performance_validation(tmp_path, omit):
    parent = LibrarySnapshot(role="customer", skills=(scoped_skill(),))
    adapter, archive, store, customer, service, case, _ = setup(tmp_path, customer=parent)
    raw = proposal(parent, "UPDATE", skill=scoped_skill(2)).model_dump(mode="json")
    if omit == "scope":
        del raw["skill"]["scope"]
    elif omit == "domains":
        del raw["skill"]["scope"]["domains"]
    client = Client(completion(json.dumps(raw)))
    result = CustomerEvolver(adapter, CustomerReflector(client, MODEL), store, archive).update(
        0, customer, service, (), (case,))
    assert result.outcome == "accepted", result.reasons
    assert result.patch.skill.scope.domains == DOMAINS
    assert result.patch.skill.scope.tools == ("read_public",)
    assert result.reflection.scope_resolution.source == "parent_skill"
    assert result.reflection.proposal_protocol_version == 2
    assert result.domain_coverage.declared_domains == DOMAINS
    assert result.domain_coverage.validated_domains == ()
    assert result.dynamic_validation is None
    assert result.domain_coverage.unvalidated_domains == DOMAINS
    assert adapter.calls == [] and len(client.requests) == 1
    assert {store.load(result.selected_snapshot_hash).skills[0].scope.domains} == {DOMAINS}
    # The original model response is never replaced by the normalized patch.
    assert json.loads(result.reflection.call.response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]) == raw
    # Scope is routing metadata, not an extra available tool or domain execution.
    assert all(".validation." in call[-1] for call in adapter.calls)
    selected = store.load(result.selected_snapshot_hash)
    assert SkillRetriever("customer").retrieve(selected, SkillContext(domain="unknown", public_text="capacity")).hits == ()
    assert SkillRetriever("customer").retrieve(selected, SkillContext(domain="other", public_text="capacity")).hits == ()


@pytest.mark.parametrize("domains", [("generic",), DOMAINS + ("unknown",), ("*",)])
def test_update_cannot_silently_change_applicability_at_any_entry_point(tmp_path, domains):
    parent = LibrarySnapshot(role="customer", skills=(scoped_skill(),))
    adapter, archive, store, customer, service, case, _ = setup(tmp_path, customer=parent)
    patch = proposal(parent, "UPDATE", skill=scoped_skill(2, domains))
    with pytest.raises(ValueError, match="inherit domain scope"):
        apply_patch(parent, patch, "test")
    assert not StaticSkillGate(allowed_tools=case.config.tools, allowed_domains=("generic",)).validate(parent, patch).passed
    result = CustomerEvolver(adapter, CustomerReflector(Client(completion(patch.model_dump_json())), MODEL), store, archive).update(
        0, customer, service, (), (case,))
    assert result.outcome == "rejected" and "program-managed" in result.reflection.error
    assert result.dynamic_validation is None and adapter.calls == []
    assert result.selected_snapshot_hash == parent.snapshot_hash


def test_add_omits_domains_and_receives_only_experiment_scope(tmp_path):
    adapter, archive, store, customer, service, case, _ = setup(tmp_path)
    raw = proposal(customer, skill=customer_skill()).model_dump(mode="json")
    del raw["skill"]["scope"]["domains"]
    client = Client(completion(json.dumps(raw)))
    result = CustomerEvolver(adapter, CustomerReflector(client, MODEL), store, archive).update(
        0, customer, service, (), (case,))
    assert result.outcome == "accepted"
    assert result.reflection.scope_resolution.source == "experiment_policy"
    assert result.patch.skill.scope.domains == ("generic",)
    assert result.domain_coverage.unvalidated_domains == ("generic",)


def test_explicit_domain_order_is_not_an_applicability_change(tmp_path):
    parent = LibrarySnapshot(role="customer", skills=(scoped_skill(),))
    adapter, archive, store, customer, service, case, _ = setup(tmp_path, customer=parent)
    patch = proposal(parent, "UPDATE", skill=scoped_skill(2, tuple(reversed(DOMAINS))))
    result = CustomerEvolver(adapter, CustomerReflector(Client(completion(patch.model_dump_json())), MODEL), store, archive).update(
        0, customer, service, (), (case,))
    assert result.outcome == "accepted"
    assert result.patch == patch  # preserve the supplied valid hash and raw list order
    assert set(result.domain_coverage.declared_domains) == set(DOMAINS)
    assert result.domain_coverage.validated_domains == ()


@pytest.mark.parametrize("domains", [("typo",), ("generic", "other"), ("*",)])
def test_add_explicit_scope_mismatch_rejects_without_silent_repair(tmp_path, domains):
    adapter, archive, store, customer, service, case, _ = setup(tmp_path)
    patch = proposal(customer, skill=scoped_skill(domains=domains))
    result = CustomerEvolver(adapter, CustomerReflector(Client(completion(patch.model_dump_json())), MODEL), store, archive).update(
        0, customer, service, (), (case,))
    assert result.outcome == "rejected" and "program-managed" in result.reflection.error
    assert adapter.calls == [] and result.selected_snapshot_hash == customer.snapshot_hash


def test_scope_optional_only_on_wire_not_in_published_skill_and_hashes_still_checked(tmp_path):
    schema = proposal_json_schema()
    assert "domains" not in schema["$defs"]["Scope"]["required"]
    assert "scope" not in schema["$defs"]["ExplorationSkill"]["required"]
    assert "domains" in Scope.model_json_schema()["required"]
    adapter, archive, store, customer, service, case, _ = setup(tmp_path)
    raw = proposal(customer, skill=customer_skill()).model_dump(mode="json")
    raw["skill"]["content_hash"] = "a" * 64
    del raw["skill"]["scope"]
    result = CustomerEvolver(adapter, CustomerReflector(Client(completion(json.dumps(raw))), MODEL), store, archive).update(
        0, customer, service, (), (case,))
    assert result.outcome == "rejected" and "hash" in result.reflection.error
    assert adapter.calls == []


def test_add_without_trusted_domain_policy_is_rejected(tmp_path):
    adapter, _, _, customer, _, _, _ = setup(tmp_path)
    value = ReflectionInput(generation=0, role="customer", library=customer, evidence=())
    result = CustomerReflector(Client(completion(proposal(customer, skill=customer_skill()).model_dump_json())), MODEL).reflect(value)
    assert "program-supplied domain policy" in result.error
    assert adapter.calls == []


@pytest.mark.parametrize("domains", [("generic", "Generic"), ("*",), ()])
def test_program_policy_rejects_ambiguous_or_unbounded_domains(domains):
    with pytest.raises(ValueError):
        DomainPolicy(add_domains=domains)


def test_inherited_applicability_does_not_grant_unavailable_tools(tmp_path):
    parent = LibrarySnapshot(role="customer", skills=(scoped_skill(),))
    adapter, archive, store, customer, service, case, _ = setup(tmp_path, customer=parent)
    raw = proposal(parent, "UPDATE", skill=scoped_skill(2)).model_dump(mode="json")
    raw["skill"].pop("scope")
    raw["skill"].pop("content_hash")
    raw["skill"]["scope"] = {"tools": ["foreign_domain_write"]}
    result = CustomerEvolver(adapter, CustomerReflector(Client(completion(json.dumps(raw))), MODEL), store, archive).update(
        0, customer, service, (), (case,))
    assert result.outcome == "rejected" and not result.static_validation.passed
    assert "foreign_domain_write" in result.reasons[0]
    assert result.domain_coverage.validated_domains == ()
    assert result.domain_coverage.unvalidated_domains == DOMAINS
    assert adapter.calls == []


def test_no_execution_is_not_validation(tmp_path):
    adapter, archive, _, customer, service, _, _ = setup(tmp_path)
    candidate = apply_patch(customer, proposal(customer, skill=customer_skill()), "test")
    result = DynamicGate(adapter, archive).customer(customer, candidate, service, (), "empty")
    assert not result.passed and result.validated_domains == ()
    assert "no_validation_episodes" in result.reasons


def test_protocol_v1_live_record_keeps_exact_historical_hash_and_verdict():
    repo = Path(__file__).resolve().parents[1]
    with tarfile.open(repo / "experiments/vast_coevo/results/live-pilot-bundle.tar.gz") as bundle:
        names = [name for name in bundle.getnames() if name.endswith("cloudops-live-002/evolution/g0.customer.json")]
        assert len(names) == 1
        raw = json.load(bundle.extractfile(names[0]))
    stored = StoredEvolution.model_validate(raw)
    audit = json.loads((repo / "experiments/vast_coevo/results/CUSTOMER_UPDATE_SCOPE_AUDIT.json").read_text())
    assert audit["stored_record_hash"] == raw["record_hash"]
    assert audit["materialized_original_patch"] == raw["record"]["patch"]
    assert audit["raw_model_arguments"] == raw["record"]["reflection"]["call"]["response"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    assert stored.model_dump(mode="json") == raw
    assert stored.record_hash == "ea4da0918766221f06d4c294d6ee95da71bef154d70044884130cbf4fed1858b"
    assert stored.record.outcome == "rejected" and stored.record.domain_coverage is None
    assert stored.record.reasons == ("skill declares an unregistered validation domain",)
    # Recheck only structure under the corrected gate. This is not a new online
    # trial or a retroactive dynamic acceptance of the original proposal.
    parent = stored.record.reflection.input.library
    gate = StaticSkillGate(allowed_tools=stored.record.reflection.input.allowed_tools, allowed_domains=("cloudops",))
    assert gate.validate(parent, stored.record.patch).passed
