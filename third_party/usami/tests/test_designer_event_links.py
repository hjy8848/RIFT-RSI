"""Profile event visibility and independent Reference positive/negative controls."""
import copy
import json
from pathlib import Path

import pytest

from usami_benchmark.designer_exploration import (
    DesignerBudget, compile_proposal, inspect_world, verify_failed_journal,
)
from usami_benchmark.tasks import ConstructionBudget, verify_task
from usami_benchmark.worlds import FrozenWorld

RESULTS = Path(__file__).resolve().parents[1] / "experiments/vast_coevo/results"
CONTROL = RESULTS / "designer-exploration-v2-validation-001"


@pytest.fixture(scope="module")
def frozen():
    return FrozenWorld.model_validate_json((RESULTS / "aftersales-coevo-inferai-v41-001/world/frozen_world.json").read_text())


def test_profile_query_links_native_approval_and_return_events_without_broadening_scope(frozen):
    before = frozen.model_dump(mode="json")
    query = {"action": "read_profile", "profile_id": "profile_u1_order0_line1"}
    for related in (False, True):
        detail = inspect_world(frozen, query, include_related_objects=related)
        events = detail["registered_external_events"]
        assert {row["id"] for row in events} == {
            "manager_review_u1_order0_line1", "return_received_u1_order0_line1"}
        assert {row["kind"] for row in events} == {"manager_review", "return_received"}
        assert all(row == frozen.world["external_events"][row["id"]] for row in events)
        assert all(row["line_id"] == "u1_order0_line1" for row in events)
    assert inspect_world(frozen, query, include_profile_events=False)["registered_external_events"] == []
    assert frozen.model_dump(mode="json") == before


def test_native_five_step_refund_control_is_stable_and_replayable(frozen):
    raw = json.loads((CONTROL / "offline-positive-control-proposal.json").read_text())
    budget = ConstructionBudget(max_nodes=4096, max_depth=16, max_paths=1)
    _, first = compile_proposal(frozen, raw, budget)
    _, second = compile_proposal(frozen, raw, budget)
    assert first == second
    assert verify_task(first) is True
    witness = first.dialogue_witnesses[0]
    assert len(witness.trace.events) > 0
    assert sum(event.kind == "business" for event in witness.trace.events) == 5
    assert witness.certificate.task.external_schedule == raw["external_schedule"]


@pytest.mark.parametrize("schedule", [[], ["manager_review_u1_order0_line1"], ["return_received_u1_order0_line1"]])
def test_five_step_control_does_not_require_unnecessary_events(frozen, schedule):
    raw = json.loads((CONTROL / "offline-positive-control-proposal.json").read_text())
    raw["external_schedule"] = schedule
    _, task = compile_proposal(frozen, raw, ConstructionBudget(max_nodes=4096, max_depth=16))
    assert verify_task(task) is True
    assert not any(event.kind == "external" for event in task.dialogue_witnesses[0].trace.events)


@pytest.mark.parametrize("profile_id,event_id", [
    ("profile_u1_order1_line0", "manager_review_u1_order1_line0"),
    ("profile_u1_order0_line0", "return_received_u1_order0_line0"),
])
def test_actually_required_event_omission_is_not_silently_repaired(frozen, profile_id, event_id):
    line_id = frozen.world["profiles"][profile_id]["facts"]["line_id"]
    raw = {"profile_ids": [profile_id], "customer_request": "Please refund my damaged item.",
           "goals": [{"collection": "lines", "entity": line_id, "field": "status", "accepted_values": ["refunded"]}],
           "preference_requirements": [{"profile_id": profile_id, "preference": "resolution", "accepted_values": ["refund"]}],
           "external_schedule": [event_id]}
    before = frozen.model_dump(mode="json")
    budget = ConstructionBudget(max_nodes=4096, max_depth=16)
    _, task = compile_proposal(frozen, raw, budget)
    assert verify_task(task) is True
    assert event_id in [event.payload["event_id"] for event in task.dialogue_witnesses[0].trace.events if event.kind == "external"]
    raw["external_schedule"] = []
    with pytest.raises(ValueError, match="bounded Reference search did not prove"):
        compile_proposal(frozen, raw, budget)
    assert frozen.model_dump(mode="json") == before


def test_pre_fix_journal_keeps_its_original_query_semantics(frozen):
    journal = json.loads((RESULTS / "aftersales-coevo-inferai-v41-001/designer/task-1-journal.json").read_text())
    assert "include_profile_events" not in journal["budget"]
    assert "include_profile_events" not in DesignerBudget().model_dump(mode="json")
    assert verify_failed_journal(frozen, journal)["verified"]


def test_legacy_rulebook_text_is_frozen_without_changing_current_rules(frozen):
    from usami_benchmark.designer_exploration import LEGACY_PROTOCOL, PROTOCOL_V3, PROTOCOL
    from usami_benchmark.public_policy import catalogue
    from usami_harness.models import digest
    original = frozen.model_dump(mode="json")
    legacy = inspect_world(frozen, {"action": "read_rules"}, protocol=LEGACY_PROTOCOL)["policy"]
    assert len(legacy["business_rules"]) == 3
    assert legacy["catalogue_hash"] == digest({k: v for k, v in legacy.items() if k != "catalogue_hash"})
    current = catalogue(frozen.spec.domain, frozen.spec.rules_version)
    for protocol in (PROTOCOL_V3, PROTOCOL):
        assert inspect_world(frozen, {"action": "read_rules"}, protocol=protocol)["policy"] == current
    assert frozen.model_dump(mode="json") == original


def test_new_event_query_is_recorded_and_exactly_replayed(frozen):
    from test_vast_usami import ScriptedClient, SPEC, answer, call
    from usami_benchmark.designer import design_and_compile, verify_designer_execution
    raw = json.loads((CONTROL / "offline-positive-control-proposal.json").read_text())
    responses = [answer(calls=[
        call("read", "inspect_frozen_world", {"action": "read_profile", "profile_id": raw["profile_ids"][0]}),
        call("ready", "ready_to_submit", {}),
    ]), answer(calls=[call("submit", "submit_frozen_world_task", raw)])]
    client = ScriptedClient(SPEC, responses)
    result = design_and_compile(frozen, client, ConstructionBudget(max_nodes=4096, max_depth=16),
        designer_budget=DesignerBudget(include_profile_events=True))
    assert len(result.exchanges[0]["tool_results"][0]["data"]["registered_external_events"]) == 2
    assert verify_designer_execution(result)["actual_online_calls"] == 0
    forged = result.model_copy(deep=True)
    forged.exchanges[0]["tool_results"][0]["data"]["registered_external_events"][0]["outcome"] = "invented"
    forged.execution_hash = forged.computed_hash()
    with pytest.raises(ValueError):
        verify_designer_execution(forged)
