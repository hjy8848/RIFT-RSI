"""Direction prompts and deterministic provenance tests, not semantic scoring claims."""
import json
from pathlib import Path
import tarfile

import pytest

from vast_coevo.models import ModelCall
from vast_coevo.reflection.evidence import native_view, wire_view
from vast_coevo.reflection.reflector import CUSTOMER_OBJECTIVE, CustomerReflector, ReflectionInput, ServiceReflector
from vast_coevo.skills import SkillPatch, empty_snapshot
from test_vast_reflection import MODEL, Client, completion, evidence, reflection_input, noop
from test_vast_usami import adapter_for, libraries


def tool(name="get_invoice", resource="wrong"):
    return dict(id="tool_1", type="function", function=dict(name=name,
                arguments=json.dumps(dict(scope_id="scope", resource_id=resource, params={}))))


def with_tool(tmp_path, customer_resource=None):
    row = evidence(tmp_path)
    chosen = tool(name="pay_invoice" if customer_resource else "get_invoice")
    calls = [ModelCall(id="service_1", role="service", model=MODEL,
                base_request={"messages": [{"role": "user", "content": "Examine the invoice."}]},
                request={"messages": [{"role": "user", "content": "Examine the invoice."}]},
                response={"choices": [{"message": {"tool_calls": [chosen]}}]})]
    payload = row.model_dump(mode="json", exclude={"evidence_hash"})
    payload["model_calls"] = [call.model_dump(mode="json") for call in calls]
    row = type(row).model_validate(payload)
    declared = [dict(name=chosen["function"]["name"], scope_id="scope", resource_id=customer_resource, params={})] if customer_resource else []
    requests = [dict(turn_id="turn_1", message="Examine the invoice.", declared_write_intents=declared,
                     accepted_write_grants=[], authorization_source="native_session_input", source_refs=("/native_trace/inputs/0",))]
    records = [dict(call_id="tool_1", name=chosen["function"]["name"], arguments=chosen["function"]["arguments"],
                    response={"success": False, "error_code": "RESOURCE_NOT_FOUND"}, event_id="", turn_id="turn_1",
                    source_refs=("/native_trace/tool_records/0",))]
    return row, native_view(row, requests, records, (), ())


def test_customer_and_service_reflection_goals_are_separate_with_one_call(tmp_path):
    customer, service = reflection_input(tmp_path), reflection_input(tmp_path, "service")
    clients = [Client(completion(noop(value).model_dump_json())) for value in (customer, service)]
    c = CustomerReflector(clients[0], MODEL).reflect(customer)
    s = ServiceReflector(clients[1], MODEL).reflect(service)
    instruction = c.call.request["messages"][0]["content"]
    assert CUSTOMER_OBJECTIVE in instruction
    for check in ("Role Check", "Goal Check", "Evidence Check", "Action Check", "Transfer Check"):
        assert check in instruction
        assert check not in s.call.request["messages"][0]["content"]
    assert "Repair verified decision failures" in s.call.request["messages"][0]["content"]
    assert c.calls == s.calls == 1 and all(len(client.requests) == 1 for client in clients)
    assert c.semantic_review_status == "requires_experiment_review"
    assert s.semantic_review_status is None and s.customer_goal_version is None


def test_service_selected_wrong_resource_is_not_automatically_customer_fault(tmp_path):
    row, view = with_tool(tmp_path)
    action = view.service_tools[0]
    assert action.arguments["resource_id"] == "wrong"
    assert action.parameter_origin == "service_tool_call"
    assert action.fault_attribution == action.service_decision_correctness == "unknown"
    assert action.oracle_classification is None
    assert view.confirmed_business_failures == ()


def test_customer_supplied_wrong_resource_is_not_automatically_service_fault(tmp_path):
    row, view = with_tool(tmp_path, "wrong")
    action = view.service_tools[0]
    assert action.parameter_origin == "matches_customer_write_intent"
    assert action.fault_attribution == "unknown" and view.confirmed_business_failures == ()
    # No authorization is inferred from matching parameters.
    assert view.customer_requests[0].accepted_write_grants == ()


def test_new_input_separates_observation_from_oracle_and_omits_actor_prompts(tmp_path):
    row, view = with_tool(tmp_path)
    value = ReflectionInput(generation=0, role="customer", library=empty_snapshot("customer"), evidence=(row,), interaction_views=(view,))
    record = CustomerReflector(Client(completion(noop(value).model_dump_json())), MODEL).reflect(value)
    payload = json.loads(record.call.request["messages"][-1]["content"])
    data = payload["evidence"][0]
    assert "messages" not in data and "interaction_views" not in payload
    assert data["interaction"]["oracle_visibility"] == "post_run_verifier_only"
    assert data["interaction"]["service_tools"][0]["fault_attribution"] == "unknown"
    assert record.prepared_interaction_views == (view,)


@pytest.mark.parametrize("field", ["evidence_hash", "trace_sha256"])
def test_projection_cannot_be_bound_to_other_evidence(tmp_path, field):
    row, view = with_tool(tmp_path)
    raw = view.model_dump(mode="json");raw[field] = "a" * 64
    with pytest.raises(ValueError, match="bound"):
        ReflectionInput(generation=0, role="customer", library=empty_snapshot("customer"), evidence=(row,),
                        interaction_views=(type(view).model_validate(raw),))


@pytest.mark.parametrize("change", ["oracle", "arguments", "parsed_arguments", "tool_oracle"])
def test_projection_rejects_changed_verified_facts(tmp_path, change):
    row, view = with_tool(tmp_path)
    raw = view.model_dump(mode="json")
    if change == "oracle":
        raw["oracle_events"] = [{"operation": "invented"}]
    elif change == "arguments":
        raw["service_tools"][0]["arguments_raw"] = '{"resource_id":"invented"}'
    elif change == "parsed_arguments":
        raw["service_tools"][0]["arguments"]["resource_id"] = "invented"
    else:
        raw["service_tools"][0]["oracle_classification"] = "successful_business_violation"
    with pytest.raises(ValueError, match="changed|arguments differ"):
        ReflectionInput(generation=0, role="customer", library=empty_snapshot("customer"), evidence=(row,),
                        interaction_views=(type(view).model_validate(raw),))


def test_goal_semantics_are_reviewed_not_faked_by_keyword_checks(tmp_path):
    # A structurally valid business tutorial remains parsable. The code does not
    # pretend a keyword rule proves semantic alignment; experiments review it.
    from test_vast_evolution import customer_skill
    value = reflection_input(tmp_path)
    patch = SkillPatch(action="ADD", role="customer", base_snapshot_hash=value.library.snapshot_hash,
                       skill=customer_skill(description="Ordinary business operation advice."), rationale="CI counterexample.")
    record = CustomerReflector(Client(completion(patch.model_dump_json())), MODEL).reflect(value)
    assert record.patch == patch and record.semantic_review_status == "requires_experiment_review"
    schema = record.call.request["tools"][0]["function"]["parameters"]
    assert "Service decision-making" in schema["$defs"]["ExplorationSkill"]["properties"]["hypotheses"]["description"]


@pytest.mark.parametrize("domain", ["saas", "cloudops", "fulfillment", "stayops"])
def test_native_four_domain_projection_preserves_authority_and_runtime_visibility(tmp_path, domain):
    adapter, config, clients = adapter_for(tmp_path, domain)
    customer, service = libraries(domain)
    row = adapter.run_episode(config, customer, service, "projection." + domain)
    before = row.model_dump(mode="json")
    view = adapter.reflection_view(row)
    assert view.source == "verified_native" and len(view.service_tools) == row.consumption.tool_calls
    assert view.customer_requests and any(request.accepted_write_grants for request in view.customer_requests)
    assert all(request.authorization_source == "native_session_input" for request in view.customer_requests)
    assert any(tool.parameter_origin == "matches_customer_write_intent" for tool in view.service_tools)
    assert all(tool.fault_attribution == "unknown" for tool in view.service_tools)
    assert row.model_dump(mode="json") == before and adapter.replay(row, customer, service).verified
    for call in row.model_calls:
        assert "post_run_verifier_only" not in json.dumps(call.request)
        assert "confirmed_business_failures" not in json.dumps(call.request)
        assert call.request.get("tools") == call.base_request.get("tools")
    # Native projections can be audited and validated without another LLM call.
    ReflectionInput(generation=0, role="customer", library=customer, evidence=(row,), interaction_views=(view,))


def test_historical_invoice_error_has_service_provenance_and_no_confirmed_service_fault(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    audit = json.loads((repo / "experiments/vast_coevo/results/CUSTOMER_UPDATE_SCOPE_AUDIT.json").read_text())
    # Historical full source replay is tested separately against its archive.
    from vast_coevo.models import EpisodeEvidence
    from integrations.usami import UsamiAdapter
    with tarfile.open(repo / "experiments/vast_coevo/results/live-pilot-bundle.tar.gz") as bundle:
        name = next(name for name in bundle.getnames() if name.endswith("cloudops-live-002/evolution/g0.customer.json"))
        raw = json.load(bundle.extractfile(name))["record"]["reflection"]["input"]["evidence"][0]
        suffix = "/native/episodes/" + Path(raw["trace"]["path"]).parent.name + "/native.json"
        native = bundle.extractfile(next(name for name in bundle.getnames() if name.endswith(suffix))).read().decode()
    read_text = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda path, *args, **kwargs:
                        native if str(path) == raw["trace"]["path"] else read_text(path, *args, **kwargs))
    row = EpisodeEvidence.model_validate(raw)
    view = UsamiAdapter(tmp_path / "unused-projection", (), ()).reflection_view(row)
    error = next(tool for tool in view.service_tools if isinstance(tool.result, dict) and tool.result.get("error_code") == "RESOURCE_NOT_FOUND")
    assert error.parameter_origin == "service_tool_call"
    assert error.oracle_classification == "rejected_invalid_attempt" and error.oracle_operation_correct is True
    assert error.fault_attribution == error.service_decision_correctness == "unknown"
    assert view.confirmed_business_failures == ()
    assert audit["original_reasons"] == ["skill declares an unregistered validation domain"]
