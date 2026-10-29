"""Single-call reflection contracts and malformed-output audit behavior."""

import copy
import json

import pytest
from pydantic import ValidationError

from vast_coevo.models import ArtifactRef, Budget, Consumption, EpisodeConfig, EpisodeEvidence, ModelSpec, ReplayResult
from vast_coevo.reflection.reflector import CustomerReflector, ReflectionInput, ServiceReflector
from vast_coevo.reflection.scope import DomainPolicy
from vast_coevo.schema import atomic_json, digest
from vast_coevo.skills import ExplorationSkill, Scope, SkillPatch, Trigger, empty_snapshot


MODEL = ModelSpec(model="reflection-fixture", base_url="https://provider.invalid/v1", options={"temperature": 0})


def evidence(tmp_path, partition="development", verified=True):
    config = EpisodeConfig(id="world", config_id="configuration", environment="generic", domain="generic",
                           partition=partition, initial_world_hash=digest({"world": 1}), identity="actor",
                           customer_model=MODEL, service_model=MODEL, budget=Budget(), tools=("read_public",))
    native = {"episode_id": "observed", "public_messages": [{"role": "user", "content": "Inspect capacity."}]}
    path = tmp_path / "observed.json"
    atomic_json(path, native)
    return EpisodeEvidence(episode_id="observed", config=config, customer_snapshot_hash=empty_snapshot("customer").snapshot_hash,
                           service_snapshot_hash=empty_snapshot("service").snapshot_hash,
                           consumption=Consumption(customer_llm_calls=1, service_llm_calls=1, tool_calls=1), termination="finished",
                           trace=ArtifactRef(path=str(path), sha256=digest(native)), native_trace_hash=digest(native),
                           replay=ReplayResult(verified=verified, request_count=2))


def reflection_input(tmp_path, role="customer"):
    return ReflectionInput(generation=0, role=role, library=empty_snapshot(role), evidence=(evidence(tmp_path),),
                           allowed_tools=("read_public",), domain_policy=DomainPolicy(add_domains=("generic",)))


def noop(value):
    return SkillPatch(action="NOOP", role=value.role, base_snapshot_hash=value.library.snapshot_hash,
                      rationale="The observed evidence does not justify a reusable change.", evidence_ids=("observed",))


def completion(raw, via_tool=True):
    message = {"role": "assistant"}
    if via_tool:
        message["tool_calls"] = [{"id": "proposal", "type": "function", "function": {"name": "submit_skill_patch", "arguments": raw}}]
    else:
        message["content"] = raw
    return {"choices": [{"message": message}]}


class Client:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.requests = response, error, []

    def complete(self, request):
        self.requests.append(copy.deepcopy(request))
        if self.error:
            raise self.error
        return self.response


@pytest.mark.parametrize("role,via_tool", [("customer", True), ("service", True), ("customer", False)])
def test_one_reflection_call_records_exact_request_response_patch(tmp_path, role, via_tool):
    value = reflection_input(tmp_path, role)
    proposal = noop(value)
    response = completion(proposal.model_dump_json(), via_tool)
    client = Client(response)
    reflector = CustomerReflector(client, MODEL) if role == "customer" else ServiceReflector(client, MODEL)
    record = reflector.reflect(value)
    assert len(client.requests) == record.calls == 1
    assert record.patch == proposal and record.error is None
    assert record.input_hash == digest(value.model_dump(mode="json"))
    assert record.call.request == record.call.base_request == client.requests[0]
    assert record.call.response == response
    assert record.call.request_hash == digest(client.requests[0])
    assert record.call.response_hash == digest(response)
    assert client.requests[0]["tool_choice"]["function"]["name"] == "submit_skill_patch"
    payload = json.loads(client.requests[0]["messages"][-1]["content"])
    assert payload["library"]["snapshot_hash"] == value.library.snapshot_hash
    assert [item["episode_id"] for item in payload["evidence"]] == ["observed"]
    assert record.call.role == role + "_reflector"


def test_reflector_can_use_auto_tool_choice_for_thinking_mode_apis(tmp_path):
    value = reflection_input(tmp_path, "service")
    proposal = noop(value)
    client = Client(completion(proposal.model_dump_json(), via_tool=False))
    record = ServiceReflector(client, MODEL, tool_choice="auto").reflect(value)
    assert record.patch == proposal and record.error is None
    assert len(client.requests) == record.calls == 1
    assert client.requests[0]["tool_choice"] == "auto"
    assert record.call.request == client.requests[0]
    assert "case-insensitive substring" in client.requests[0]["messages"][0]["content"]


@pytest.mark.parametrize("response", [
    completion("not JSON"), completion('{"action":"NOOP","action":"ADD"}'),
    completion("[]"), {"choices": []}, {"choices": [{"message": {}}, {"message": {}}]},
    {"choices": [{"message": {"tool_calls": [{"function": {"name": "other", "arguments": "{}"}}]}}]},
    {"choices": [{"message": {"tool_calls": [{"function": {"name": "submit_skill_patch", "arguments": "{}"}},
                                              {"function": {"name": "submit_skill_patch", "arguments": "{}"}}]}}]},
    [], {"choices": [None]}, {"choices": [{"message": None}]},
])
def test_malformed_ambiguous_and_multiple_outputs_reject_without_retry(tmp_path, response):
    value = reflection_input(tmp_path)
    client = Client(response)
    record = CustomerReflector(client, MODEL).reflect(value)
    assert record.patch is None
    assert record.error.startswith("invalid_reflection:")
    assert record.calls == len(client.requests) == 1
    assert record.call.response == response


@pytest.mark.parametrize("change", [{"base_snapshot_hash": "a" * 64}, {"role": "service"}, {"evidence_ids": ("not-supplied",)}])
def test_reflector_rejects_wrong_parent_role_or_evidence_refs(tmp_path, change):
    value = reflection_input(tmp_path)
    data = noop(value).model_dump(mode="json")
    data.update(change)
    client = Client(completion(json.dumps(data)))
    result = CustomerReflector(client, MODEL).reflect(value)
    assert result.patch is None and result.error
    assert len(client.requests) == 1


def test_reflection_input_blocks_heldout_and_unverified_evidence(tmp_path):
    with pytest.raises(ValidationError, match="heldout"):
        ReflectionInput(generation=0, role="customer", library=empty_snapshot("customer"), evidence=(evidence(tmp_path, "heldout"),))
    with pytest.raises(ValidationError, match="replay"):
        ReflectionInput(generation=0, role="customer", library=empty_snapshot("customer"), evidence=(evidence(tmp_path, verified=False),))


def test_validation_evidence_must_be_explicitly_consumed_before_reflection(tmp_path):
    source = evidence(tmp_path, "validation")
    with pytest.raises(ValidationError, match="explicitly identified"):
        ReflectionInput(generation=1, role="service", library=empty_snapshot("service"), evidence=(source,))
    consumed = ReflectionInput(schema_version=2, generation=1, role="service",
        library=empty_snapshot("service"), evidence=(source,),
        consumed_validation_evidence_ids=(source.episode_id,))
    assert consumed.consumed_validation_evidence_ids == (source.episode_id,)
    with pytest.raises(ValidationError, match="schema v2"):
        ReflectionInput(generation=1, role="service", library=empty_snapshot("service"),
            evidence=(source,), consumed_validation_evidence_ids=(source.episode_id,))


def test_budget_and_provider_failure_are_audited_without_retry(tmp_path):
    value = reflection_input(tmp_path)
    client = Client(error=RuntimeError("provider unavailable"))
    bounded = CustomerReflector(client, MODEL, max_input_chars=1).reflect(value)
    assert bounded.calls == 0 and bounded.error == "reflection_input_budget_exceeded"
    assert client.requests == []
    failed = CustomerReflector(client, MODEL).reflect(value)
    assert failed.calls == 1 and failed.error == "reflection_client_error"
    assert failed.call.error["type"] == "RuntimeError"
    assert len(client.requests) == 1


def test_reflection_accepts_one_declarative_add(tmp_path):
    value = reflection_input(tmp_path)
    skill = ExplorationSkill(id="customer.capacity", scope=Scope(domains=("generic",), tools=("read_public",)),
                             trigger=Trigger(keywords=("capacity",)), hypotheses=("Capacity boundaries need exploration.",),
                             exploration=("Inspect permitted capacity observations.",), feedback=("Record public outcomes.",), stop=("Stop at the budget limit.",))
    proposal = SkillPatch(action="ADD", role="customer", base_snapshot_hash=value.library.snapshot_hash, skill=skill,
                          rationale="Learn a reusable boundary exploration method.", evidence_ids=("observed",))
    client = Client(completion(proposal.model_dump_json()))
    record = CustomerReflector(client, MODEL).reflect(value)
    assert record.patch == proposal
    assert len(client.requests) == 1


def test_reflector_revalidates_mutated_credentials_before_provider_call(tmp_path):
    value = reflection_input(tmp_path)
    model = ModelSpec(model="reflection-fixture", base_url="https://provider.invalid/v1", options={"temperature": 0})
    client = Client(completion(noop(value).model_dump_json()))
    reflector = CustomerReflector(client, model)
    reflector.model.options["api_key"] = "placeholder"
    with pytest.raises(ValidationError, match="credentials"):
        reflector.reflect(value)
    assert client.requests == []
