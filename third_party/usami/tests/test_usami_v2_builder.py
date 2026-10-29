"""Scene construction is real business data; task/interaction proofs stay separate."""
import json

import pytest

from usami_v2.commerce.invariants import invariants
from usami_v2.scenarios.schema import (
    Attribute, BuildBudget, IntegerRange, LineVariable, LinearExpression, LinearTerm,
    OrderVariable, RelationConstraint, ScenarioSpec, SKUVariable, StockConstraint, WarehouseVariable,
)
from usami_v2.scenarios.builder import build_world, verify_build
from usami_v2.scenarios.certification import verify_world_constraints


def fixed(value):
    return IntegerRange(low=value, high=value)


def term(alias, field, coefficient=1, warehouse=None):
    return LinearTerm(variable=alias, field=field, coefficient=coefficient, warehouse=warehouse)


def expression(*terms, constant=0):
    return LinearExpression(terms=terms, constant=constant)


def scene():
    skus = tuple(SKUVariable(
        alias=name, family="device" if name != "cable" else "cable",
        attributes=(Attribute(name="interface", value="USB_A" if name == "wrong" else "USB_C"),),
        price_cents=fixed(price))
        for name, price in (("original", 20000), ("wrong", 20000), ("late", 18000),
                            ("expensive", 24000), ("correct_choice_D", 22000),
                            ("correct_choice_E", 21000), ("cable", 2000)))
    return ScenarioSpec(
        scenario_id="business-world-test", provenance_kind="offline_fixture",
        customer_aliases=("customer",), skus=skus,
        warehouses=(WarehouseVariable(alias="fast", delivery_days=fixed(2), delivery_fee_cents=fixed(0)),
                    WarehouseVariable(alias="slow", delivery_days=fixed(6), delivery_fee_cents=fixed(0))),
        orders=(OrderVariable(alias="purchase", customer_alias="customer", delivered_days_ago=fixed(1),
                              lines=(LineVariable(alias="damaged", original_sku="original", quantity=fixed(1),
                                                  paid_cents=fixed(20000)),
                                     LineVariable(alias="accessory", original_sku="cable", quantity=fixed(1),
                                                  paid_cents=fixed(2000)))),),
        default_stock=fixed(0),
        stock=tuple(StockConstraint(sku_alias=name, warehouse_alias=warehouse, quantity=fixed(3))
                    for name, warehouse in (("wrong", "fast"), ("late", "slow"), ("expensive", "fast"),
                                             ("correct_choice_D", "fast"), ("correct_choice_E", "fast"))),
        constraints=(
            RelationConstraint(constraint_id="D_extra_under_budget",
                               left=expression(term("correct_choice_D", "price_cents"), term("damaged", "paid_cents", -1)),
                               operator="le", right=expression(constant=3000)),
            RelationConstraint(constraint_id="cheap_is_late", left=expression(term("slow", "delivery_days")),
                               operator="gt", right=expression(constant=3)),
        ))


def replace(row, **updates):
    data = row.model_dump(mode="json", exclude={"record_hash"})
    data.update(updates)
    return type(row).model_validate(data)


def test_real_world_build_is_frozen_reproducible_and_does_not_leak_designer_aliases():
    spec = scene()
    result = build_world(spec, 7)
    assert result.status == "SAT" and result == build_world(spec, 7)
    assert result.world_hash == result.world.state_hash and invariants(result.world) == []
    assert len(result.world.skus) == 7 and len(result.world.warehouses) == 2
    assert len(result.world.lines) == 2 and len(result.world.inventory) == 14
    assert "correct_choice" not in result.world.model_dump_json()
    assert not result.world.delegations and not result.world.consents
    proof = verify_build(spec, result)
    assert proof["verified"] and proof["world_constraint_certificate_hash"]
    assert not proof["task_reachability_certified"] and not proof["interaction_certified"]
    certificate = verify_world_constraints(spec, result)
    assert certificate.world_hash == result.world_hash


def test_interval_contradiction_is_proven_unsat_without_search():
    spec = scene()
    impossible = RelationConstraint(
        constraint_id="no_negative_price", left=expression(term("original", "price_cents")),
        operator="lt", right=expression(constant=0))
    spec = replace(spec, constraints=[impossible.model_dump(mode="json")])
    result = build_world(spec, 1)
    assert result.status == "UNSAT" and not result.attempts and result.world is None
    assert "interval contradiction" in result.reason
    assert verify_build(spec, result)["verified"]


def test_budget_exhaustion_is_unknown_not_a_fake_infeasibility_proof():
    spec = scene()
    skus = [replace(row, price_cents=IntegerRange(low=1, high=1000000).model_dump(mode="json"))
            if row.alias == "original" else row for row in spec.skus]
    # Individually feasible interval, deliberately too large for complete search.
    required = RelationConstraint(
        constraint_id="needle_price", left=expression(term("original", "price_cents")),
        operator="eq", right=expression(constant=777777))
    spec = replace(spec, skus=[row.model_dump(mode="json") for row in skus],
                   constraints=[required.model_dump(mode="json")])
    result = build_world(spec, 1, BuildBudget(max_nodes=1))
    assert result.status == "UNKNOWN" and len(result.attempts) == 1 and result.world is None
    assert result.attempts[0].rejected_constraints == ("needle_price",)
    assert verify_build(spec, result)["verified"]


def test_complete_small_search_can_prove_combined_constraints_unsat():
    spec = scene()
    skus = [replace(row, price_cents=IntegerRange(low=1, high=2).model_dump(mode="json"))
            if row.alias in ("original", "wrong") else row for row in spec.skus]
    relations = tuple(RelationConstraint(
        constraint_id=left + "_less_" + right, left=expression(term(left, "price_cents")),
        operator="lt", right=expression(term(right, "price_cents")))
        for left, right in (("original", "wrong"), ("wrong", "original")))
    spec = replace(spec, skus=[row.model_dump(mode="json") for row in skus],
                   constraints=[row.model_dump(mode="json") for row in relations])
    result = build_world(spec, 1, BuildBudget(max_nodes=4))
    assert result.status == "UNSAT" and len(result.attempts) == 4
    partial = build_world(spec, 1, BuildBudget(max_nodes=2))
    assert partial.status == "UNKNOWN"


def test_independent_world_inspector_detects_a_broken_constructor(monkeypatch):
    import usami_v2.scenarios.builder as builder
    original = builder._construct

    def broken(*args):
        world, aliases = original(*args)
        world.skus[dict(aliases)["original"]].price_cents = 90000
        return world, aliases

    monkeypatch.setattr(builder, "_construct", broken)
    with pytest.raises(ValueError, match="outside scenario interval"):
        builder.build_world(scene(), 1)


def test_resealed_assignment_or_proof_cannot_replace_actual_world_truth():
    spec = scene()
    result = build_world(spec, 7)
    assignment = list(result.assignment)
    assignment[0] = (*assignment[0][:3], assignment[0][3] + 1)
    forged = replace(result, assignment=assignment)
    with pytest.raises(ValueError, match="actual business data"):
        verify_world_constraints(spec, forged)
    with pytest.raises(ValueError, match="transcript differs"):
        verify_build(spec, forged)


def test_designer_cannot_supply_rules_unknown_fields_or_arbitrary_world():
    raw = scene().model_dump(mode="json", exclude={"record_hash"})
    raw["oracle"] = {"allow_all": True}
    with pytest.raises(ValueError, match="Extra inputs"):
        ScenarioSpec.model_validate(raw)
    bad = RelationConstraint(constraint_id="hidden", left=expression(term("customer", "price_cents")),
                             operator="eq", right=expression(constant=1))
    with pytest.raises(ValueError, match="unavailable business"):
        replace(scene(), constraints=[bad.model_dump(mode="json")])
    with pytest.raises(ValueError, match="distribution provenance"):
        replace(scene(), provenance_kind="designer_exploration")


def test_build_artifact_cli_preserves_source_attempts_and_replays_without_models(tmp_path):
    from usami_v2.cli import build_scenario_artifact, verify_scenario_artifact
    spec_path, output = tmp_path / "scenario.json", tmp_path / "world-run"
    spec_path.write_text(scene().model_dump_json(), encoding="utf-8")
    manifest = build_scenario_artifact(spec_path, 29, output, max_nodes=32)
    verified = verify_scenario_artifact(output)
    assert manifest["status"] == "SAT" and manifest["model_calls"] == 0
    assert verified["verified"] and not verified["task_reachability_certified"]
    build = json.loads((output / "build.json").read_text(encoding="utf-8"))
    assert build["world"] and build["attempts"]
    with pytest.raises(FileExistsError):
        build_scenario_artifact(spec_path, 29, output, max_nodes=32)
    manifest_path = output / "manifest.json"
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["world_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest integrity"):
        verify_scenario_artifact(output)
