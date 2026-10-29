"""Actual skill request injection with a generic provider and native protocol."""

import copy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from vast_coevo.agents.runtime import RuntimeClient, request_context
from vast_coevo.models import Consumption, ModelCall, ModelSpec, ReplayResult
from vast_coevo.schema import digest
from vast_coevo.skills import DecisionSkill, ExplorationSkill, LibrarySnapshot, Scope, SkillContext, Trigger


MODEL = ModelSpec(model="frozen-generic", base_url="https://provider.invalid/v1", options={"temperature": 0})


def skills(role):
    common = dict(id=role + ".capacity", scope=Scope(domains=("generic",), tools=("inspect_capacity",)),
                  trigger=Trigger(keywords=("capacity",)))
    if role == "customer":
        skill = ExplorationSkill(**common, hypotheses=("Check boundary prerequisites.",),
                                 exploration=("Inspect permitted observations and request a useful action.",),
                                 feedback=("Record public outcomes.",), stop=("Stop at the budget limit.",))
    else:
        skill = DecisionSkill(**common, preconditions=("An exact write grant is present.",),
                              procedure=("Check observed capacity before executing.",),
                              postconditions=("The legal goal is complete.",), recovery=("Explain missing prerequisites.",))
    return LibrarySnapshot(role=role, skills=(skill,))


def native_request(text="Inspect available capacity."):
    return dict(
        model=MODEL.model,
        messages=[
            {"role": "system", "content": "Native identity and authorization instructions."},
            {"role": "system", "content": "Native response schema instructions."},
            {"role": "user", "content": text},
        ],
        tools=[{"type": "function", "function": {"name": "inspect_capacity", "parameters": {"type": "object", "properties": {}}}}],
        tool_choice="auto", response_format={"type": "json_object"}, temperature=0,
    )


class ProviderError(RuntimeError):
    code = "PROVIDER_FAILED"


class Provider:
    def __init__(self, error=None, mutate_wire=False):
        self.calls = []
        self.error = error
        self.mutate_wire = mutate_wire
        self.response = {"choices": [{"message": {"role": "assistant", "content": "public result"}}]}

    def complete(self, request):
        self.calls.append(copy.deepcopy(request))
        if self.mutate_wire:
            request["messages"][0]["content"] = "Provider changed its own request object."
        if self.error is not None:
            raise self.error
        return self.response


def runtime(provider, role="customer", **kwargs):
    return RuntimeClient(provider, role, skills(role), "generic", tools=("inspect_capacity",), model=MODEL, **kwargs)


@pytest.mark.parametrize("role", ["customer", "service"])
def test_runtime_injects_both_roles_and_preserves_native_protocol(role):
    provider = Provider()
    client = runtime(provider, role)
    request = native_request()
    original = copy.deepcopy(request)
    assert client.complete(request) == provider.response
    assert request == original
    assert len(provider.calls) == 1
    wire = provider.calls[0]
    assert wire["messages"][:2] == request["messages"][:2]
    assert "<VAST_SKILLS>" in wire["messages"][2]["content"]
    assert wire["messages"][3:] == request["messages"][2:]
    assert {key: value for key, value in wire.items() if key != "messages"} == {key: value for key, value in request.items() if key != "messages"}
    record = client.records[0]
    assert record.base_request == original
    assert record.request == wire
    assert record.request_hash == digest(wire)
    assert record.response_hash == digest(provider.response)
    injection = client.injections[0]
    assert injection.request_hash == record.request_hash
    assert injection.snapshot_hash == client.library.snapshot_hash
    assert [(hit.id, hit.version, hit.content_hash) for hit in injection.hits] == [(role + ".capacity", 1, client.library.skills[0].content_hash)]


def test_public_context_excludes_system_data_and_has_no_environment_input():
    request = native_request("No matching public topic.")
    request["messages"][0]["content"] = "capacity only appears in a system message"
    request["private_world"] = {"capacity": "hidden"}
    context = request_context(request, "generic", ("inspect_capacity",))
    assert "hidden" not in context.public_text
    assert "capacity only appears" not in context.public_text
    assert context.available_tools == ("inspect_capacity",)
    provider = Provider()
    client = runtime(provider)
    client.complete(request)
    assert provider.calls == [request]
    assert client.injections[0].hits == ()
    assert client.injections[0].rendered == ""
    assert client.records[0].request == client.records[0].base_request


def test_tool_and_assistant_public_feedback_can_trigger_retrieval():
    request = native_request("Read the latest public reply.")
    request["messages"].extend([
        {"role": "assistant", "content": "I will inspect capacity."},
        {"role": "tool", "tool_call_id": "read-1", "content": "Capacity is currently full."},
    ])
    client = runtime(Provider())
    client.complete(request)
    assert client.injections[0].hits
    assert client.records[0].request["messages"][-1] == request["messages"][-1]


def test_character_limit_uses_actual_injected_text():
    provider = Provider()
    client = runtime(provider, max_chars=1)
    request = native_request()
    client.complete(request)
    assert provider.calls == [request]
    assert client.injections[0].hits == ()


def test_requests_responses_and_started_observer_are_detached():
    provider = Provider(mutate_wire=True)

    def observer(state, record):
        if state == "started":
            record.request["messages"][0]["content"] = "Observer changed its own audit view."

    client = runtime(provider, observer=observer)
    request = native_request()
    original = copy.deepcopy(request)
    response = client.complete(request)
    request["messages"][0]["content"] = "Caller changed its request later."
    response["choices"][0]["message"]["content"] = "Caller changed provider response later."
    record = client.records[0]
    assert record.base_request == original
    assert record.request == provider.calls[0]
    assert record.response["choices"][0]["message"]["content"] == "public result"
    assert record.request_hash == digest(record.request)
    assert record.response_hash == digest(record.response)


def test_completed_observer_and_returned_audit_views_cannot_corrupt_journal():
    def observer(state, record):
        if state == "completed":
            record.request["messages"][0]["content"] = "Changed detached view."

    provider = Provider()
    client = runtime(provider, observer=observer)
    client.complete(native_request())
    view = client.records[0]
    view.request["messages"][0]["content"] = "Changed another detached view."
    assert client.records[0].request == provider.calls[0]
    assert client.records[0].request_hash == digest(provider.calls[0])


def test_provider_error_is_preserved_and_recorded_without_retry():
    failure = ProviderError("transport failed")
    provider = Provider(error=failure)
    seen = []
    client = runtime(provider, observer=lambda state, record: seen.append((state, record)))
    with pytest.raises(ProviderError) as raised:
        client.complete(native_request())
    assert raised.value is failure
    assert len(provider.calls) == 1
    assert len(client.records) == 1
    assert client.records[0].response is None
    assert client.records[0].error == {"type": "ProviderError", "code": "PROVIDER_FAILED", "message": "transport failed"}
    assert [state for state, _ in seen] == ["started", "completed"]
    assert len(client.injections) == 1


def test_transport_error_preserves_raw_response_body_and_keyboard_interrupt():
    failure = ProviderError("provider returned malformed body")
    failure.response = "malformed response body"
    provider = Provider(error=failure)
    client = runtime(provider)
    with pytest.raises(ProviderError):
        client.complete(native_request())
    assert client.records[0].response == "malformed response body"
    assert client.records[0].response_hash == digest("malformed response body")
    failure.response = "Changed exception later"
    assert client.records[0].response == "malformed response body"
    interrupted = runtime(Provider(error=KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        interrupted.complete(native_request())
    assert interrupted.records[0].error["type"] == "KeyboardInterrupt"


def test_malformed_provider_response_is_recorded_for_native_session_validation():
    provider = Provider()
    provider.response = ["a response outside the native completion schema"]
    client = runtime(provider)
    assert client.complete(native_request()) == provider.response
    assert len(provider.calls) == 1
    assert client.records[0].response == provider.response


@pytest.mark.parametrize("context", [
    SkillContext(domain="other", public_text="capacity", available_tools=("inspect_capacity",)),
    SkillContext(domain="generic", public_text="capacity", available_tools=("secret_tool",)),
])
def test_context_factory_cannot_expand_domain_or_tools(context):
    provider = Provider()
    client = runtime(provider, context_factory=lambda request: context)
    with pytest.raises(ValueError, match="exceeds"):
        client.complete(native_request())
    assert provider.calls == []
    assert client.records == ()


def test_changed_model_is_rejected_before_provider_call():
    provider = Provider()
    client = runtime(provider)
    request = native_request()
    request["model"] = "different-model"
    with pytest.raises(ValueError, match="frozen model"):
        client.complete(request)
    assert provider.calls == []


def test_runtime_can_read_native_settings_without_importing_an_environment():
    provider = Provider()
    provider.settings = SimpleNamespace(model=MODEL.model, base_url=MODEL.base_url,
                                        options={"temperature": 0}, timeout_seconds=60.0, trust_env=False)
    client = RuntimeClient(provider, "customer", skills("customer"), "generic", tools=("inspect_capacity",))
    provider.settings.options["temperature"] = 1
    assert client.model.options == {"temperature": 0}
    client.complete(native_request())
    assert len(provider.calls) == 1


@pytest.mark.parametrize("kwargs", [
    {"options": {"api_key": "placeholder"}}, {"options": {"headers": {"Authorization": "placeholder"}}},
    {"options": {"extra_body": {"tools": []}}},
    {"base_url": "https://user:placeholder@provider.invalid/v1"},
    {"base_url": "https://provider.invalid/v1?api_key=placeholder"},
    {"timeout_seconds": float("inf")},
])
def test_model_spec_rejects_credentials_overrides_and_invalid_timeout(kwargs):
    values = dict(model="generic", base_url="https://provider.invalid/v1")
    values.update(kwargs)
    with pytest.raises(ValidationError):
        ModelSpec(**values)


@pytest.mark.parametrize("value", [True, "1", 1.5, -1])
def test_audit_counts_reject_coercion_and_negative_values(value):
    with pytest.raises(ValidationError):
        Consumption(tool_calls=value)
    with pytest.raises(ValidationError):
        ReplayResult(verified=True, request_count=value)


def test_model_call_hashes_detect_nested_mutation_at_sealed_boundaries():
    request = native_request()
    call = ModelCall(id="one", role="customer", model=MODEL, base_request=request, request=request)
    request["messages"][0]["content"] = "Changed source object."
    assert call.request["messages"][0]["content"] != "Changed source object."
    changed = call.model_dump(mode="json")
    changed["request"]["messages"][0]["content"] = "Tampered journal."
    with pytest.raises(ValidationError, match="hash"):
        ModelCall.model_validate(changed)
    native_error = ModelCall(id="one", role="customer", model=MODEL, base_request={}, request={},
                             response={"error": "provider body"}, error={"type": "ProviderError"})
    assert native_error.response_hash == digest({"error": "provider body"})
