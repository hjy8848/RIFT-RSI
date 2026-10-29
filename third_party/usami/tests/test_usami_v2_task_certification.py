"""Task reachability certificates use the native Reference, Oracle and Replay."""
from datetime import timedelta
import json

import pytest
from pydantic import ValidationError

from usami_harness.models import digest
from usami_v2.scenarios.builder import build_world
from usami_v2.scenarios.schema import BuildBudget
from usami_v2.scenarios.task_certification import (
    CustomerScenario, ProofBudget, ReviewPlan, TaskCertification, certify_task,
    verify_task_certificate,
)
from usami_v2.scenarios.tasks import RemedyGoal, TaskSpec
from test_usami_v2_builder import scene


def frozen_world(seed=13):
    result = build_world(scene(), seed, BuildBudget(max_nodes=128))
    assert result.status == "SAT"
    return result.world, dict(result.private_aliases)


def make_task(world, aliases, *, attributes=(), remedy=("exchange",), return_goal=False):
    line_id = aliases["damaged"]
    return TaskSpec(
        task_id="certified-refund-or-replacement",
        world_hash=world.state_hash,
        customer_id=aliases["customer"],
        customer_request="Please resolve the damaged item.",
        remedy_goals=(RemedyGoal(
            goal_id="damaged-line", line_id=line_id, quantity=1,
            allowed_remedies=remedy,
            required_attributes=attributes,
            maximum_extra_charge_cents=2000 if "exchange" in remedy else 0,
            latest_arrival=world.clock + timedelta(days=3) if "exchange" in remedy else None),),
        partition="train", source_distribution_hash=digest("offline task-certificate test"))


def customer_scenario(aliases, **updates):
    data = {
        "customer_id": aliases["customer"],
        "delegated_line_ids": (aliases["damaged"],),
        "maximum_extra_charge_cents": 2000,
        "latest_arrival": "2026-01-18T12:00:00+00:00",
    }
    data.update(updates)
    return CustomerScenario(**data)


def test_frozen_task_finds_two_independent_valid_skus_and_replays_certificates():
    world, aliases = frozen_world()
    task = make_task(world, aliases, attributes=({"name": "interface", "value": "USB_C"},))
    scenario = customer_scenario(aliases)
    budget = ProofBudget(max_plan_candidates=16, max_operation_steps=16, max_witnesses=2)
    certificate = certify_task(world, task, scenario, budget, minimum_witnesses=2)
    assert certificate.status == "CERTIFIED"
    assert certificate.business_reachability_certified
    assert certificate.full_task_reachability_certified
    assert certificate.minimum_witnesses_met and len(certificate.witness_hashes) == 2
    chosen = {row.candidates[0].target_sku_id for row in certificate.attempts if row.status == "WITNESS"}
    assert chosen == {aliases["correct_choice_D"], aliases["correct_choice_E"]}
    assert all(any(step.kind == "customer_consent" for step in row.steps)
               and any(step.kind == "operation" for step in row.steps)
               for row in certificate.attempts if row.status == "WITNESS")
    assert verify_task_certificate(world, task, scenario, certificate)["verified"]


def test_unreachable_task_and_budget_exhaustion_are_not_conflated():
    world, aliases = frozen_world()
    impossible = make_task(world, aliases, attributes=({"name": "interface", "value": "NOT_A_REAL_PORT"},))
    scenario = customer_scenario(aliases)
    unsat = certify_task(world, impossible, scenario)
    assert unsat.status == "UNSAT" and unsat.search_complete and not unsat.witness_hashes
    assert verify_task_certificate(world, impossible, scenario, unsat)["verified"]

    reachable = make_task(world, aliases, attributes=({"name": "interface", "value": "USB_C"},))
    unknown = certify_task(world, reachable, scenario,
                           ProofBudget(max_plan_candidates=8, max_operation_steps=1, max_witnesses=2))
    assert unknown.status == "UNKNOWN"
    assert "proof_operation_budget_exhausted" in unknown.blockers
    assert not unknown.business_reachability_certified


def test_tampered_private_witness_is_detected_even_if_outer_hash_is_resealed():
    world, aliases = frozen_world()
    task = make_task(world, aliases, attributes=({"name": "interface", "value": "USB_C"},))
    scenario = customer_scenario(aliases)
    certificate = certify_task(world, task, scenario)
    attempt = next(row for row in certificate.attempts if row.status == "WITNESS")
    step = attempt.steps[0]
    changed_step = step.model_copy(update={"after_state_hash": "0" * 64, "record_hash": ""})
    changed_attempt = attempt.model_copy(update={"steps": (changed_step,) + attempt.steps[1:], "record_hash": ""})
    raw = certificate.model_dump(mode="json", exclude={"record_hash"})
    raw["attempts"] = [changed_attempt.model_dump(mode="json") if row.record_hash == attempt.record_hash
                        else row.model_dump(mode="json") for row in certificate.attempts]
    raw["record_hash"] = ""
    resealed = TaskCertification.model_validate(raw)
    with pytest.raises(ValueError, match="certificate or private witness differs"):
        verify_task_certificate(world, task, scenario, resealed)


def test_return_required_refund_needs_frozen_return_event_and_second_consent():
    world, aliases = frozen_world()
    world = world.clone()
    original_id = world.lines[aliases["damaged"]].sku_id
    world.skus[original_id].return_required = True
    task = make_task(world, aliases, remedy=("refund",))
    scenario = customer_scenario(
        aliases, return_pickup_slot="weekday-evening", return_received_after_days=2,
        latest_arrival=None, maximum_extra_charge_cents=0)
    certificate = certify_task(world, task, scenario, ProofBudget(max_plan_candidates=8,
                                                                  max_operation_steps=16,
                                                                  max_witnesses=1))
    assert certificate.status == "CERTIFIED"
    witness = next(row for row in certificate.attempts if row.status == "WITNESS")
    assert sum(step.kind == "customer_consent" for step in witness.steps) == 2
    assert sum(step.kind == "external_event" and step.evidence.get("action") == "apply"
               for step in witness.steps) == 1
    assert verify_task_certificate(world, task, scenario, certificate)["verified"]

    missing_event = customer_scenario(aliases, return_pickup_slot=None,
                                      return_received_after_days=None, latest_arrival=None,
                                      maximum_extra_charge_cents=0)
    blocked = certify_task(world, task, missing_event)
    assert blocked.status == "UNKNOWN"
    assert "return_receipt_event_not_registered_in_frozen_customer_scenario" in blocked.blockers


def test_frozen_approval_event_is_used_only_after_reference_requests_review():
    world, aliases = frozen_world()
    world = world.clone()
    world.merchants[aliases["merchant"]].refund_approval_limit_cents = 1
    task = make_task(world, aliases, remedy=("refund",))
    scenario = customer_scenario(
        aliases, maximum_extra_charge_cents=0, latest_arrival=None,
        reviews=(ReviewPlan(line_id=aliases["damaged"], approved=True, result_after_days=2),))
    certificate = certify_task(world, task, scenario,
                               ProofBudget(max_plan_candidates=8, max_operation_steps=12, max_witnesses=1))
    assert certificate.status == "CERTIFIED"
    witness = next(row for row in certificate.attempts if row.status == "WITNESS")
    assert any(step.kind == "external_event" and step.evidence.get("action") == "apply"
               for step in witness.steps)
    assert any(step.kind == "operation"
               and step.evidence.get("request", {}).get("operation", {}).get("name") == "request_policy_review"
               for step in witness.steps)
    assert verify_task_certificate(world, task, scenario, certificate)["verified"]

    no_review = customer_scenario(aliases, maximum_extra_charge_cents=0, latest_arrival=None)
    blocked = certify_task(world, task, no_review)
    assert blocked.status == "UNKNOWN"
    assert "policy_review_event_not_registered_in_frozen_customer_scenario" in blocked.blockers


def test_task_certificate_cli_artifact_rebuilds_world_and_replays_private_witness(tmp_path):
    from usami_v2.cli import (
        build_scenario_artifact, certify_task_artifact, verify_task_certificate_artifact,
    )

    scenario_spec = scene()
    spec_path = tmp_path / "scenario.json"
    spec_path.write_text(scenario_spec.model_dump_json(), encoding="utf-8")
    world_directory = tmp_path / "world"
    build_scenario_artifact(spec_path, 13, world_directory, max_nodes=128)
    world, aliases = frozen_world()
    task = make_task(world, aliases, attributes=({"name": "interface", "value": "USB_C"},))
    customer = customer_scenario(aliases)
    task_path, customer_path = tmp_path / "task.json", tmp_path / "customer.json"
    task_path.write_text(task.model_dump_json(), encoding="utf-8")
    customer_path.write_text(customer.model_dump_json(), encoding="utf-8")
    artifact = tmp_path / "certificate"
    output = certify_task_artifact(world_directory, task_path, customer_path, artifact,
                                   max_plan_candidates=16, max_operation_steps=16, max_witnesses=2,
                                   minimum_witnesses=2)
    assert output["status"] == "CERTIFIED" and output["witness_count"] == 2
    verified = verify_task_certificate_artifact(world_directory, artifact)
    assert verified["verified"] and verified["model_calls"] == 0
    certificate_path = artifact / "certificate.json"
    altered = json.loads(certificate_path.read_text(encoding="utf-8"))
    altered["attempts"][0]["reason"] = "tampered"
    certificate_path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValidationError, match="content hash differs"):
        verify_task_certificate_artifact(world_directory, artifact)
