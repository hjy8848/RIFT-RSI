from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from usami.audit.report import render_report
from usami.audit.rules import RuleObserver, rule_statistics
from usami.audit.runner import ROOT, Census, audit_splits, diversity, overlap, run_audit, source_hashes
from usami.audit.signatures import analyze_scenario, world_signature
from usami.evaluation.scenario import derive_scenario, generate_scenarios
from usami.protocol import Operation
from usami.world.generator import GeneratorConfig, generate_world
from usami.world.models import WorldState
from usami.world.rules import ReferenceRules, check_invariants


def op(name, resource="sub_000", org="org_000", **params):
    return Operation(name=name, org_id=org, resource_id=resource, params=params)


def scenario(world, steps, authorized=True):
    return derive_scenario(world, "audit-test", steps, "test", authorized=authorized)


def analyze(world, steps, authorized=True):
    return analyze_scenario(world, scenario(world, steps, authorized), RuleObserver())


def renamed(world):
    data = world.model_dump(mode="json")
    mapping = {identifier: f"anonymous-{kind}-{n}" for kind in ("users", "organizations", "memberships", "subscriptions", "invoices")
               for n, identifier in enumerate(reversed(list(data[kind])))}
    mapping.update({p: f"plan-{n}" for n, p in enumerate(data["rules"]["plans"])})

    def rewrite(value):
        if isinstance(value, str):
            return mapping.get(value, value)
        if isinstance(value, list):
            return [rewrite(v) for v in value]
        if isinstance(value, dict):
            return {mapping.get(k, k): rewrite(v) for k, v in value.items()}
        return value

    data = rewrite(data)
    for kind in ("users", "organizations"):
        for obj in data[kind].values():
            obj["name"] = "new display label"
            if "email" in obj:
                obj["email"] = "renamed@example.invalid"
    for plan in data["rules"]["plans"].values():
        plan["name"] = "Plan display label"
    return WorldState.model_validate(data), mapping


def test_same_input_produces_identical_json_and_report():
    a = run_audit([0, 1], split_worlds=1, engine_checks=2)
    b = run_audit([0, 1], split_worlds=1, engine_checks=2)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert render_report(a) == render_report(b)
    assert render_report(a) == render_report(json.loads(json.dumps(a, sort_keys=True)))


@pytest.mark.parametrize("category", ["single", "permission", "refund", "multi_step_refund", "multi_step_plan", "multi_entity", "cross_organization"])
def test_signatures_ignore_ids_names_plan_labels_and_scenario_labels(category):
    world = generate_world(0)
    original = next(s for s in generate_scenarios(world) if s.category == category)
    other, mapping = renamed(world)
    steps = []
    for s in original.steps:
        operation = s.request.operation.model_dump()
        operation["org_id"] = mapping[operation["org_id"]]
        operation["resource_id"] = mapping[operation["resource_id"]]
        if "plan_id" in operation["params"]:
            operation["params"]["plan_id"] = mapping[operation["params"]["plan_id"]]
        steps.append((mapping[s.request.requester_id], Operation.model_validate(operation)))
    new = derive_scenario(other, "different-label", steps, "other-category",
                          authorized=original.steps[0].request.authorizes(original.steps[0].request.requester_id, original.steps[0].request.operation))
    a, b = analyze_scenario(world, original, RuleObserver()), analyze_scenario(other, new, RuleObserver())
    assert a["template_id"] == b["template_id"]
    assert a["semantic_id"] == b["semantic_id"]
    assert world_signature(world) == world_signature(other)
    assert world_signature(world, True) == world_signature(other, True)


def test_permission_topology_is_not_collapsed(world):
    owner = analyze(world, [("user_000", op("cancel_renewal"))])
    member = analyze(world, [("user_003", op("cancel_renewal"))])
    assert owner["template_id"] != member["template_id"]
    assert owner["semantic_id"] != member["semantic_id"]
    assert member["records"][0]["error_code"] == "PERMISSION_DENIED"


def test_business_prestate_and_noop_are_distinguished(world):
    steps = [("user_000", op("cancel_renewal"))]
    a = analyze(world, steps)
    other = world.clone()
    other.subscriptions["sub_000"].auto_renew = False
    b = analyze(other, steps)
    assert a["template_id"] == b["template_id"]
    assert a["semantic_id"] != b["semantic_id"]
    other.subscriptions["sub_000"].status = "expired"
    other.subscriptions["sub_000"].period_end = other.clock
    c = analyze(other, steps)
    assert c["records"][0]["error_code"] == "SUBSCRIPTION_NOT_ACTIVE"
    assert a["semantic_id"] != c["semantic_id"]


def test_irrelevant_amount_values_are_collapsed_but_full_vs_partial_is_not(world):
    inv = world.invoices["invoice_000_recent"]
    step = [("user_000", op("refund_invoice", inv.id, amount_cents=inv.amount_cents, payment_method=inv.payment_method))]
    a = analyze(world, step)
    other = world.clone()
    other.invoices[inv.id].amount_cents += 500
    other.invoices[inv.id].refundable_amount_cents += 500
    new_step = [("user_000", op("refund_invoice", inv.id, amount_cents=inv.amount_cents + 500, payment_method=inv.payment_method))]
    b = analyze(other, new_step)
    assert a["semantic_id"] == b["semantic_id"]
    c = analyze(world, [("user_000", op("refund_invoice", inv.id, amount_cents=inv.amount_cents - 1, payment_method=inv.payment_method))])
    assert a["semantic_id"] != c["semantic_id"]


def test_repeated_scenario_does_not_create_new_structure(world):
    s = scenario(world, [("user_000", op("cancel_renewal"))])
    census = Census(RuleObserver())
    census.add(world, s)
    census.add(world, deepcopy(s))
    result = census.result()
    for kind in ("template", "semantic"):
        assert result[kind]["total"] == 2
        assert result[kind]["unique"] == 1
        assert result[kind]["repetition_ratio"] == 0.5


def test_dependency_requires_real_legality_change(world):
    independent = analyze(world, [("user_000", op("get_subscription")), ("user_000", op("cancel_renewal"))])
    assert independent["dependency_edges"] == []
    dependent = analyze(world, [("user_000", op("change_plan", plan_id="team")), ("user_000", op("change_seats", seats=8))])
    assert dependent["dependency_edges"] == [[0, 1]]
    assert dependent["complexity"]["depth"] == 1
    assert dependent["complexity"]["decision_sensitive"]


def test_failed_repeat_refund_has_no_dependency_and_identical_reverse_is_not_order_sensitive(world):
    for identifier, dependent in (("invoice_000_stale", False), ("invoice_000_recent", True)):
        inv = world.invoices[identifier]
        operation = op("refund_invoice", identifier, amount_cents=inv.amount_cents, payment_method=inv.payment_method)
        result = analyze(world, [("user_000", operation), ("user_000", operation)])
        assert bool(result["dependency_edges"]) == dependent
        assert not result["complexity"]["decision_sensitive"]
        assert not result["complexity"]["final_state_sensitive"]


def test_guard_observation_matches_real_reference_and_restores_trace_hook(world):
    observer = RuleObserver()
    operations = [op("change_seats", seats=0), op("change_seats", seats=6), op("change_plan", plan_id="missing"),
                  op("change_member_role", "membership_000_user_003", role="billing_admin"), op("cancel_renewal")]
    previous = sys.gettrace()
    for actor in ("user_000", "user_002", "user_003", "missing"):
        for operation in operations:
            reference = ReferenceRules().decide(world, actor, operation, "e")
            decision, visits = observer.decide(world, actor, operation, "e")
            assert decision == reference
            vetoes = [gid for gid, veto in visits if veto]
            assert len(vetoes) == int(not reference.allowed)
            assert sys.gettrace() is previous
    assert len({g["id"] for g in observer.guards if g["error_code"] == "PERMISSION_DENIED"}) == 2


def test_auth_denial_does_not_become_business_permission_denial(world):
    a = analyze(world, [("user_000", op("cancel_renewal"))], authorized=False)
    record = a["records"][0]
    assert not record["allowed"] and record["policy_allowed"]
    assert record["error_code"] == "UNAUTHORIZED_OPERATION"
    stats = rule_statistics(RuleObserver(), a["records"])
    assert stats["rejection_codes"] == {"UNAUTHORIZED_OPERATION": 1}
    assert not any(g["vetoed"] for g in stats["implemented_guards"])


def test_masked_predicates_are_derived_and_not_counted_as_actual_veto(world):
    inv = world.invoices["invoice_000_stale"]
    a = analyze(world, [("user_000", op("refund_invoice", inv.id, amount_cents=inv.amount_cents - 1, payment_method="wrong"))])
    record = a["records"][0]
    assert record["error_code"] == "REFUND_WINDOW_EXPIRED"
    assert sum(veto for _, veto in record["guards"]) == 1
    assert sum(value is True for value in record["potential"].values()) == 3
    stats = rule_statistics(RuleObserver(), a["records"])
    assert sum(g["vetoed"] for g in stats["implemented_guards"]) == 1
    assert sum(stats["derived_multiple_veto_sets"].values()) == 1


def test_split_overlap_counts_classes_and_sample_mass_separately():
    a, b = {"a": 8, "b": 2}, {"a": 1, "c": 99}
    result = overlap(a, b)
    assert result["common"] == 1
    assert result["left_only"] == ["b"] and result["right_only"] == ["c"]
    assert result["jaccard"] == pytest.approx(1 / 3)
    assert result["right_structure_in_left_ratio"] == 0.5
    assert result["right_sample_in_left_ratio"] == 0.01
    assert overlap({}, {})["jaccard"] == 0


def test_split_analysis_is_deterministic_and_labels_demo():
    args = (RuleObserver(), 1, GeneratorConfig(), ROOT / "configs/validation-manifest.json")
    a, b = audit_splits(*args), audit_splits(*args)
    assert a == b
    assert all(s["visibility"] == "demo" for s in a["demo_manifests"].values())
    assert a["demo_manifests"]["train"]["heldout_worlds"] == 0
    assert a["demo_manifests"]["test"]["heldout_worlds"] == 1
    assert a["shipped_validation"]["template"]["total"] == 24
    assert a["shipped_private_test_task_manifest"] is None


def test_core_behavior_and_source_are_unchanged():
    before = source_hashes()
    original = generate_world(42)
    assert original.state_hash == "cbdc43984d682555281707a7431c7ed702f0c9bc4a3ebb3346b614f10936e0ec"
    for s in generate_scenarios(original):
        analyze_scenario(original, s, RuleObserver())
    assert generate_world(42) == original
    assert before == source_hashes()
    assert not check_invariants(original)


def test_large_world_canonicalization_explicitly_unknown():
    assert world_signature(generate_world(0, GeneratorConfig(organization_count=7))) is None


def test_signature_tampering_is_detected(world):
    s = scenario(world, [("user_000", op("cancel_renewal"))])
    s.steps[0].expected_allowed = False
    with pytest.raises(ValueError, match="differs"):
        analyze_scenario(world, s, RuleObserver())


@pytest.mark.parametrize("seeds", [[], [0, 0]])
def test_invalid_seed_population_is_rejected(seeds):
    with pytest.raises(ValueError, match="nonempty and unique"):
        run_audit(seeds)


def test_cli_small_real_audit_exports_json_and_markdown(tmp_path):
    result = subprocess.run([sys.executable, "-m", "usami.audit", "--count", "1", "--split-worlds", "1",
                             "--engine-checks", "1", "--output", str(tmp_path)], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    data = json.loads((tmp_path / "results.json").read_text())
    assert data["world"]["count"] == 1
    assert len(data["verification"]["engine_oracle_replay_checks"]) == 1
    assert "Business rule coverage" in (tmp_path / "REPORT.md").read_text()
    assert diversity({"one": 10})["effective_diversity"] == 1
