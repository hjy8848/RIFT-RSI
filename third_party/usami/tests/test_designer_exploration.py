import copy
import json

import pytest

from usami_benchmark.designer import design_and_compile, designer_view, verify_designer_execution
from usami_benchmark.designer_exploration import (
    PROTOCOL, PROTOCOL_V3, DialogueDesignerProposal, DesignerBudget, DesignerFailure,
    compile_proposal, inspect_world,
    verify_failed_journal,
)
from usami_benchmark.registry import default_spec
from usami_benchmark.tasks import ConstructionBudget
from usami_benchmark.worlds import freeze
from test_vast_usami import ScriptedClient, SPEC, answer, call
from test_vnext_open_compositions import multi_order_selection


@pytest.fixture(scope="module")
def frozen():
    return freeze(default_spec("aftersales", 0))


def proposal(frozen):
    raw = multi_order_selection(frozen).composition.model_dump(mode="json")
    raw.pop("protocol")
    raw.pop("frames")
    raw.pop("refusal_operations")
    return raw


def submit(args, identifier="submit"):
    return answer(calls=[call(identifier, "submit_frozen_world_task", args)])


def test_compact_view_and_bounded_read_tools_do_not_mutate_world(frozen):
    before = frozen.model_dump(mode="json")
    view = designer_view(frozen, protocol=PROTOCOL)
    assert "world" not in view and "customer_profiles_and_native_task_contracts" not in view
    assert designer_view(frozen) == view
    assert len(json.dumps(view)) < len(json.dumps(frozen.world)) / 10
    page = inspect_world(frozen, {"action": "list_profiles", "identity": "u0", "limit": 2})
    assert len(page["items"]) == 2 and page["total"] > 2
    detail = inspect_world(frozen, {"action": "read_profile", "profile_id": page["items"][0]["profile_id"]})
    assert detail["identity"] == "u0" and detail["facts"] and detail["related_objects"]
    assert detail["native_frame_contract"]
    v3_detail = inspect_world(frozen, {"action": "read_profile", "profile_id": page["items"][0]["profile_id"]},
                              protocol=PROTOCOL_V3)
    assert "native_frame_contract" not in v3_detail
    rules = inspect_world(frozen, {"action": "read_rules"})
    assert "refunded" in rules["native_field_enums"]["lines"]["status"]
    with pytest.raises(ValueError):
        inspect_world(frozen, {"action": "read_objects", "collection": "__dict__"})
    with pytest.raises(ValueError):
        inspect_world(frozen, {"action": "read_objects", "collection": "orders", "ids": ["invented"]})
    assert frozen.model_dump(mode="json") == before


def test_high_level_schema_derives_exact_native_scope_and_certifies_multi_order(frozen):
    raw = proposal(frozen)
    assert "frames" not in DialogueDesignerProposal.model_json_schema()["properties"]
    _, certified = compile_proposal(frozen, raw, ConstructionBudget(max_nodes=512, max_depth=16))
    old = multi_order_selection(frozen)
    assert certified.selection.composition.goals == old.composition.goals
    assert certified.selection.composition.frames == old.composition.frames
    assert len(certified.selection.composition.profile_ids) == 2
    raw["frames"] = []
    with pytest.raises(ValueError):
        compile_proposal(frozen, raw, ConstructionBudget())


def test_confirmation_requirements_are_optional_but_supplied_scope_stays_native(frozen):
    from usami_benchmark.designer_exploration import _tools

    schema = DialogueDesignerProposal.model_json_schema()
    assert "confirmation_requirements" not in schema["required"]
    assert "Optional evaluator assertion" in schema["properties"]["confirmation_requirements"]["description"]
    legacy_schema = _tools("submission")[0]["function"]["parameters"]
    guided_schema = _tools("submission", guide_optional_confirmations=True)[0]["function"]["parameters"]
    assert "description" not in legacy_schema["properties"]["confirmation_requirements"]
    assert "Optional evaluator assertion" in guided_schema["properties"]["confirmation_requirements"]["description"]


def test_submission_schema_requires_an_objective_and_types_native_refusal_params(frozen):
    from usami_benchmark.designer_exploration import _messages, _tools

    with pytest.raises(ValueError, match="information and preference requirements alone"):
        DialogueDesignerProposal.model_validate({
            "profile_ids": ["profile_u0_order0_line0"],
            "customer_request": "Please check what options are available.",
            "information_requirements": [{
                "profile_id": "profile_u0_order0_line0", "fact_keys": ["line_id"]
            }],
        })

    schema = _tools("submission", frozen=frozen)[0]["function"]["parameters"]
    assert {tuple(branch["required"]) for branch in schema["anyOf"]} == {
        ("goals",), ("refusal_operations",), ("created_goals",)
    }
    refusal_items = schema["properties"]["refusal_operations"]["items"]["anyOf"]
    quote = next(branch for branch in refusal_items
                 if branch["properties"]["name"].get("enum") == ["quote_resolution"])
    assert "params" in quote["required"]
    assert set(quote["properties"]["params"]["required"]) == {"case_id", "option"}
    # Confirmation operations are optional, but if supplied their structure
    # must be canonical too. This prevents line_id/case_id from being guessed
    # at the operation root or from silently accepting extra parameters.
    confirmation_items = schema["properties"]["confirmation_requirements"]["items"]["anyOf"]
    confirmation_quote = next(branch for branch in confirmation_items
                              if branch["properties"]["name"].get("enum") == ["quote_resolution"])
    assert set(confirmation_quote["required"]) == {"name", "scope_id", "resource_id", "params"}
    assert "case_id" not in confirmation_quote["properties"]
    assert set(confirmation_quote["properties"]["params"]["required"]) == {"case_id", "option"}
    assert "confirmation_requirements" not in schema["required"]
    assert "customer_operations" not in schema["properties"]["refusal_operations"]["description"]
    assert "Customer-authorized operation is not automatically a refusal" in schema["properties"]["refusal_operations"]["description"]
    assert "native Customer frame" in schema["$defs"]["Goal"]["description"]
    created = schema["$defs"]["CreatedObjectGoal"]
    assert created["properties"]["collection"]["enum"]
    assert created["properties"]["selectors"]["additionalProperties"]["anyOf"]

    v4_policy = _messages(frozen, {}, DesignerBudget(), protocol=PROTOCOL)[0]["content"]
    assert "customer_operations grounds Customer authority but does not mean an operation should be refused" in v4_policy
    assert "created-object goal" in v4_policy
    historical_v3_policy = _messages(frozen, {}, DesignerBudget(), protocol=PROTOCOL_V3)[0]["content"]
    assert "its presence does not mean the operation should be refused" in historical_v3_policy
    assert "created-object predicates are not supported yet" in historical_v3_policy
    legacy_policy = _messages(frozen, {}, DesignerBudget(),
                              protocol="frozen-world-designer-exploration-v2")[0]["content"]
    assert "its presence does not mean the operation should be refused" not in legacy_policy

    raw = proposal(frozen)
    raw.pop("confirmation_requirements", None)
    parsed = DialogueDesignerProposal.model_validate(raw)
    assert parsed.confirmation_requirements == []
    # An omitted evaluator assertion does not change native task certification.
    _, task = compile_proposal(frozen, raw, ConstructionBudget(max_nodes=512, max_depth=16))
    assert task.selection.composition.confirmation_requirements == []

    # Supplying a well-shaped operation for an object outside these selected
    # Customer profiles remains a hard native scope error.
    identities = {frozen.world["profiles"][profile_id]["identity"] for profile_id in raw["profile_ids"]}
    foreign_order = next(key for key in frozen.world["orders"] if key.split("_", 1)[0] not in identities)
    raw["confirmation_requirements"] = [{
        "name": "open_case", "scope_id": "m0", "resource_id": foreign_order,
        "params": {"line_id": "u0_order0_line0", "issue": "wrong_item"},
    }]
    with pytest.raises(ValueError, match="confirmation requirement is outside selected Customer authority"):
        compile_proposal(frozen, raw, ConstructionBudget(max_nodes=512, max_depth=16))


def test_v3_protocol_schema_and_execution_replay_remain_compatible(frozen):
    from usami_benchmark.designer_exploration import _tools

    schema = _tools("submission", frozen=frozen, protocol=PROTOCOL_V3)[0]["function"]["parameters"]
    assert {tuple(branch["required"]) for branch in schema["anyOf"]} == {
        ("goals",), ("refusal_operations",)
    }
    assert "CreatedObjectGoal" not in schema.get("$defs", {})
    client = ScriptedClient(SPEC, [submit(proposal(frozen))])
    result = design_and_compile(frozen, client, ConstructionBudget(max_nodes=512, max_depth=16),
        protocol=PROTOCOL_V3, designer_budget=DesignerBudget(max_exploration_calls=0))
    assert result.protocol == PROTOCOL_V3
    assert verify_designer_execution(result)["verified"]


def test_v4_designer_request_has_no_harness_output_cap_and_v3_keeps_its_config(frozen):
    configured = SPEC.model_copy(update={"options": {"max_tokens": 4096}})
    for protocol, capped in ((PROTOCOL, False), (PROTOCOL_V3, True)):
        client = ScriptedClient(configured, [submit(proposal(frozen))])
        result = design_and_compile(frozen, client, ConstructionBudget(max_nodes=512, max_depth=16),
            protocol=protocol, designer_budget=DesignerBudget(max_exploration_calls=0))
        assert ("max_tokens" in result.request) is capped


def test_v3_goal_feedback_explains_external_events_are_not_completion_targets(frozen):
    raw = proposal(frozen)
    event_id = next(iter(frozen.world["external_events"]))
    raw["goals"] = [{
        "collection": "external_events", "entity": event_id,
        "field": "outcome", "accepted_values": ["missing"],
    }]
    with pytest.raises(ValueError, match="external events belong in external_schedule, not goals"):
        compile_proposal(frozen, raw, ConstructionBudget(max_nodes=512, max_depth=16))


def test_proposal_report_separates_schema_constraints_and_reference(frozen):
    from experiments.vast_coevo.examples.designer_event_link_check import _proposal_validation

    valid = proposal(frozen)
    malformed = copy.deepcopy(valid)
    malformed.pop("profile_ids")
    # Build raw completion-response records from the same fixture helper used by the live journal.
    records = [
        {"stage": "submission", "response": submit(malformed)},
        {"stage": "submission", "response": submit(valid),
         "feedback": {"category": "invalid_proposal", "detail": "confirmation requirement is outside selected Customer authority"}},
        {"stage": "submission", "response": submit(valid),
         "feedback": {"category": "search_unproven", "detail": "bounded Reference search did not prove the open composition"}},
        {"stage": "submission", "response": submit(valid),
         "tool_results": [{"status": "certified", "task_hash": "fixture"}]},
    ]
    report = _proposal_validation({"exchanges": records})
    assert report["attempts"] == 4
    assert [row["proposal_schema_valid"] for row in report["stages"]] == [False, True, True, True]
    assert [row["native_constraints_valid"] for row in report["stages"]] == [None, False, True, True]
    assert [row["reference_certified"] for row in report["stages"]] == [None, None, False, True]

    provider_failure = _proposal_validation({"exchanges": [{"stage": "submission",
        "transport_error": {"type": "ServiceError", "message": "HTTP 502"}}]})
    assert provider_failure["stages"][0]["stage"] == "provider_error"
    assert provider_failure["stages"][0]["proposal_schema_valid"] is None
    assert provider_failure["stages"][0]["reference_certified"] is None


def test_cross_identity_and_invented_state_are_not_repaired(frozen):
    raw = proposal(frozen)
    raw["profile_ids"].append("profile_u1_order0_line1")
    with pytest.raises(ValueError, match="same authenticated identity"):
        compile_proposal(frozen, raw, ConstructionBudget())
    raw = proposal(frozen)
    raw["goals"][0]["accepted_values"] = ["resolved"]
    with pytest.raises(ValueError, match="native vocabulary"):
        compile_proposal(frozen, raw, ConstructionBudget())


def test_explore_reject_correct_and_exact_replay(frozen, tmp_path):
    good = proposal(frozen)
    bad = copy.deepcopy(good)
    bad["goals"][0]["accepted_values"] = ["resolved"]
    responses = [answer(calls=[
        call("profiles", "inspect_frozen_world", {"action": "list_profiles", "identity": "u0"}),
        call("rules", "inspect_frozen_world", {"action": "read_rules"}),
    ]), answer(calls=[
        call("profile", "inspect_frozen_world", {"action": "read_profile", "profile_id": good["profile_ids"][0]}),
        call("ready", "ready_to_submit", {}),
    ]), submit(bad, "bad"), submit(good, "good")]
    client = ScriptedClient(SPEC, responses)
    result = design_and_compile(frozen, client, ConstructionBudget(max_nodes=512, max_depth=16),
        prompt_context={"skill_snapshot_hash": "a" * 64, "skill_rendered": "An updated discovery skill."},
        protocol=PROTOCOL, designer_budget=DesignerBudget(max_exploration_calls=2),
        journal_path=tmp_path / "journal.json")
    assert len(client.requests) == 4
    assert client.requests[0]["tool_choice"] == "auto"
    assert all(request["tool_choice"]["function"]["name"] == "submit_frozen_world_task" for request in client.requests[2:])
    assert "An updated discovery skill." in result.request["messages"][0]["content"]
    assert any("native vocabulary" in (msg.get("content") or "") for msg in client.requests[-1]["messages"])
    assert result.metrics["valid_proposal_rate"] == 0.5
    assert result.metrics["reference_path_business_steps"] > 1
    assert verify_designer_execution(result)["verified"]
    assert verify_designer_execution(json.loads(result.model_dump_json()))["actual_online_calls"] == 0
    tampered = result.model_copy(deep=True)
    tampered.exchanges[0]["tool_results"][0]["data"]["items"][0]["identity"] = "fake"
    tampered.execution_hash = tampered.computed_hash()
    with pytest.raises(ValueError):
        verify_designer_execution(tampered)
    journal = json.loads((tmp_path / "journal.json").read_text())
    assert journal["proposal_attempts"] == 2 and journal["status"] == "CERTIFIED"


def test_at_most_three_corrections_and_preserved_failure(frozen, tmp_path):
    raw = proposal(frozen)
    raw["goals"][0]["accepted_values"] = ["invented"]
    client = ScriptedClient(SPEC, [submit(raw, f"bad-{i}") for i in range(4)])
    with pytest.raises(DesignerFailure, match="submission budget exhausted") as caught:
        design_and_compile(frozen, client, protocol=PROTOCOL,
            designer_budget=DesignerBudget(max_exploration_calls=0), journal_path=tmp_path / "journal.json")
    assert len(client.requests) == 4
    assert caught.value.journal["proposal_attempts"] == 4
    assert caught.value.journal["failure_category"] == "business_precondition_unmet"
    assert json.loads((tmp_path / "journal.json").read_text())["status"] == "BLOCKED"
    assert verify_failed_journal(frozen, caught.value.journal)["verified"]


def test_exploration_budget_and_read_errors_are_visible(frozen):
    good = proposal(frozen)
    client = ScriptedClient(SPEC, [answer(calls=[call("read", "inspect_frozen_world",
        {"action": "read_objects", "collection": "orders", "ids": ["missing"]})]), submit(good)])
    result = design_and_compile(frozen, client, ConstructionBudget(max_nodes=512, max_depth=16),
        protocol=PROTOCOL, designer_budget=DesignerBudget(max_exploration_calls=1))
    assert result.exchanges[0]["tool_results"][0]["status"] == "error"
    assert result.exchanges[0]["tool_results"][0]["category"] == "object_missing"
    assert len(client.requests) == 2 and result.metrics["proposal_attempts"] == 1


def test_transport_failure_is_not_retried(frozen, tmp_path):
    client = ScriptedClient(SPEC, [RuntimeError("offline provider")])
    with pytest.raises(DesignerFailure, match="no retry") as caught:
        design_and_compile(frozen, client, protocol=PROTOCOL, journal_path=tmp_path / "journal.json")
    assert len(client.requests) == 1
    assert caught.value.journal["budget"]["confirmation_guidance"] is True
    assert caught.value.journal["failure_category"] == "transport_error"
    assert "confirmation_requirements is optional" in client.requests[0]["messages"][0]["content"]
    assert caught.value.journal["exchanges"][0]["transport_error"]["type"] == "RuntimeError"
    assert verify_failed_journal(frozen, caught.value.journal)["verified"]


def test_search_feedback_is_unproven_not_an_impossibility_claim(frozen):
    raw = proposal(frozen)
    client = ScriptedClient(SPEC, [submit(raw, f"limited-{i}") for i in range(4)])
    with pytest.raises(DesignerFailure) as caught:
        design_and_compile(frozen, client, ConstructionBudget(max_nodes=1, max_depth=1),
            protocol=PROTOCOL, designer_budget=DesignerBudget(max_exploration_calls=0))
    feedback = caught.value.journal["exchanges"][0]["feedback"]
    assert feedback["category"] == "reference_unproven"
    assert "does not prove impossibility" in feedback["meaning"]
    assert verify_failed_journal(frozen, caught.value.journal)["verified"]


def test_v4_failure_categories_separate_proposal_layers():
    from pydantic import ValidationError
    from usami.service.models import ServiceError
    from usami_benchmark.designer_exploration import _designer_failure_category

    with pytest.raises(ValidationError) as schema_error:
        DialogueDesignerProposal.model_validate({"profile_ids": []})
    assert _designer_failure_category(schema_error.value, stage="submission") == "schema_invalid"
    assert _designer_failure_category(ValueError("requested object does not exist in the frozen collection"),
                                      stage="submission") == "object_missing"
    assert _designer_failure_category(ValueError("refusal request lies outside selected frozen Customer authority"),
                                      stage="submission") == "scope_invalid"
    assert _designer_failure_category(ValueError("confirmation requirement is outside selected Customer authority"),
                                      stage="submission") == "authorization_unmet"
    assert _designer_failure_category(ValueError("composed task references an unregistered external event"),
                                      stage="submission") == "external_event_missing"
    assert _designer_failure_category(ValueError("no native refusal task for this scope/operation"),
                                      stage="submission") == "business_precondition_unmet"
    assert _designer_failure_category(ValueError("bounded Reference search did not prove the open composition"),
                                      stage="submission") == "reference_unproven"
    assert _designer_failure_category(ValueError("duplicate JSON key"), stage="submission") == "model_output_invalid"
    assert _designer_failure_category(ServiceError("MODEL_OUTPUT_INVALID", "bad function call"),
                                      stage="submission") == "model_output_invalid"


def test_v3_failure_category_and_replay_keep_historical_semantics(frozen):
    raw = proposal(frozen)
    raw["goals"][0]["accepted_values"] = ["not-a-native-state"]
    client = ScriptedClient(SPEC, [submit(raw, f"v3-bad-{i}") for i in range(4)])
    with pytest.raises(DesignerFailure) as caught:
        design_and_compile(frozen, client, protocol=PROTOCOL_V3,
            designer_budget=DesignerBudget(max_exploration_calls=0))
    assert caught.value.journal["exchanges"][0]["feedback"]["category"] == "invalid_proposal"
    assert "failure_category" not in caught.value.journal
    assert verify_failed_journal(frozen, caught.value.journal)["verified"]


def test_incomplete_model_output_can_be_corrected_within_submission_budget(frozen):
    truncated = {"choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": "unfinished"}}]}
    client = ScriptedClient(SPEC, [truncated, submit(proposal(frozen))])
    result = design_and_compile(frozen, client, ConstructionBudget(max_nodes=512, max_depth=16),
        protocol=PROTOCOL, designer_budget=DesignerBudget(max_exploration_calls=0))
    assert result.metrics["proposal_attempts"] == 2
    assert verify_designer_execution(result)["verified"]


@pytest.mark.parametrize("domain", ["aftersales", "airline", "property", "education"])
def test_all_native_dialogue_domains_have_read_only_exploration(domain):
    world = freeze(default_spec(domain, 0))
    page = inspect_world(world, {"action": "list_profiles", "limit": 1})
    detail = inspect_world(world, {"action": "read_profile", "profile_id": page["items"][0]["profile_id"]})
    assert detail["facts"] and detail["related_objects"]
    assert inspect_world(world, {"action": "read_rules"})["native_field_enums"]


def test_reserved_exploration_supports_actual_corrections(frozen):
    raw = proposal(frozen)
    raw["goals"][0]["accepted_values"] = ["invented"]
    responses = [answer(calls=[call("first", "inspect_frozen_world", {"action": "read_rules"})]),
                 submit(raw, "bad-1"),
                 answer(calls=[call("inspect-more", "inspect_frozen_world", {"action": "list_profiles"})]),
                 submit(raw, "bad-2"), submit(raw, "bad-3"), submit(raw, "bad-4")]
    client = ScriptedClient(SPEC, responses)
    budget = DesignerBudget(max_exploration_calls=2, initial_exploration_calls=1, actionable_feedback=True)
    with pytest.raises(DesignerFailure) as caught:
        design_and_compile(frozen, client, protocol=PROTOCOL, designer_budget=budget)
    journal = caught.value.journal
    assert [row["stage"] for row in journal["exchanges"]] == ["exploration", "submission", "exploration", "submission", "submission", "submission"]
    assert journal["exchanges"][1]["feedback"]["remaining_exploration_calls"] == 1
    assert journal["exchanges"][3]["feedback"]["unchanged_submission"] is True
    assert journal["proposal_attempts"] == 4 and verify_failed_journal(frozen, journal)["verified"]
    # Old v2 journals omit these fields; their request and replay shape stays intact.
    assert set(DesignerBudget().model_dump(mode="json")) == {"max_exploration_calls", "max_revisions", "max_queries"}


def test_requested_profile_context_includes_immutable_order_and_evidence(frozen):
    raw = {"action": "read_profile", "profile_id": "profile_u0_order0_line0"}
    before = frozen.model_dump(mode="json")
    minimal = inspect_world(frozen, raw)
    linked = inspect_world(frozen, raw, include_related_objects=True)
    assert "orders" not in minimal["related_objects"]
    assert linked["related_objects"]["orders"]["u0_order0"] == frozen.world["orders"]["u0_order0"]
    assert linked["related_objects"]["evidence"]["photo_u0_order0_line0"] == frozen.world["evidence"]["photo_u0_order0_line0"]
    assert "u1_order0" not in linked["related_objects"]["orders"]
    assert sum(len(rows) for rows in linked["related_objects"].values()) <= 32
    assert designer_view(frozen, protocol=PROTOCOL)["clock"] == frozen.world["clock"]
    assert frozen.model_dump(mode="json") == before


def test_qwen_keeps_exploration_thinking_but_can_force_submission(frozen):
    from vast_coevo.models import ModelSpec
    spec = ModelSpec(model="dashscope/qwen3.7-plus", base_url="https://fixture.invalid/v1",
                     options={"enable_thinking": True})
    client = ScriptedClient(spec, [answer(calls=[call("ready", "ready_to_submit", {})]), submit(proposal(frozen))])
    result = design_and_compile(frozen, client, ConstructionBudget(max_nodes=512, max_depth=16), protocol=PROTOCOL)
    assert client.requests[0]["enable_thinking"] is True
    assert client.requests[1]["enable_thinking"] is False
    assert client.requests[1]["tool_choice"]["function"]["name"] == "submit_frozen_world_task"
    assert verify_designer_execution(result)["verified"]
