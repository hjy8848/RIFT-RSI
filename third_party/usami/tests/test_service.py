"""Offline deterministic contract tests. Scripted model fixtures are never the live runner."""
import copy
import json

import pytest

from usami import Operation, evaluate_trace, generate_world, replay_trace
from usami.service import CustomerInput, Experiment, ModelSettings, ServiceSession, replay_execution
from usami.service.models import ServiceError


def op(name="cancel_renewal", **params):
    return Operation(name=name, org_id="org_001", resource_id="sub_001", params=params)


def call(name, cid="call_1", **args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def operation_call(name="cancel_renewal", cid="call_1", **params):
    return call(name, cid, org_id="org_001", resource_id="sub_001", **params)


def answer(text="Done", calls=None):
    return {"id": "fixture_response", "model": "fixture_model", "choices": [{"finish_reason": "tool_calls" if calls else "stop",
        "message": {"role": "assistant", "content": None if calls else text, **({"tool_calls": calls} if calls else {})}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 30}}


class FixtureClient:
    settings = ModelSettings(model="fixture_model", base_url="https://example.invalid/v1")

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(copy.deepcopy(request))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return copy.deepcopy(response)


def experiment(authorizations=None, **kwargs):
    return Experiment(customer=CustomerInput(identity="user_001", message="Please cancel the renewal for Workspace 1.",
        authorizations=[op()] if authorizations is None else authorizations), **kwargs)


def run(responses, spec=None):
    client = FixtureClient(responses)
    return ServiceSession(spec or experiment(), client).run()


def codes(execution):
    return {v.code for r in execution.oracle.requests for v in r.violations}


def test_real_world_query_then_legal_write_and_no_world_or_authority_in_prompt():
    execution = run([answer(calls=[call("get_customer_context"), operation_call("get_subscription", "query")]),
                     answer(calls=[operation_call(cid="cancel")]), answer()])
    assert execution.status == "completed" and execution.oracle.status == "pass"
    assert execution.world_trace.initial_state == generate_world(42)
    assert not execution.world_trace.final_state.subscriptions["sub_001"].auto_renew
    request = execution.llm_records[0].request
    assert request["messages"][1]["content"] == execution.experiment.customer.message
    assert "authorizations" not in json.dumps(request) and "customer_000" not in json.dumps(request)
    assert "world_d7b758c82586402e848a" not in json.dumps(request)
    context = execution.tool_records[0].response["data"]
    assert context["customer"]["id"] == "user_001"
    assert all(m["user_id"] == "user_001" for m in context["memberships"])
    assert {"invoices", "subscriptions", "rules"}.isdisjoint(context)
    assert "operation_correct" not in json.dumps(execution.tool_records[1].response)
    assert evaluate_trace(execution.world_trace, execution.trusted_requests) == execution.oracle


def test_multistep_decisions_use_actual_evolving_state_not_fixed_script():
    upgrade, seats = op("change_plan", plan_id="team"), op("change_seats", seats=6)
    execution = run([answer(calls=[operation_call("get_subscription", "q")]),
        answer(calls=[operation_call("change_plan", "plan", plan_id="team"), operation_call("change_seats", "seats", seats=6)]),
        answer(calls=[operation_call("get_subscription", "verify")]), answer()], experiment([upgrade, seats]))
    final = execution.world_trace.final_state.subscriptions["sub_001"]
    assert final.plan_id == "team" and final.seats == 6 and execution.oracle.status == "pass"
    # Request is frozen before the LLM, activated against the actual upgraded pre-state.
    grant = next(r for r in execution.oracle.requests if r.request_id == "customer_001")
    assert grant.valid_request is True and grant.operation_correct is True
    query_data = json.loads(execution.messages[-2]["content"])["data"]
    assert query_data["plan_id"] == "team" and query_data["seats"] == 6


def test_service_can_choose_authorized_order_independently():
    spec = experiment([op("change_plan", plan_id="team"), op()])
    execution = run([answer(calls=[operation_call(cid="cancel"), operation_call("change_plan", "plan", plan_id="team")]), answer()], spec)
    operations = [e for e in execution.world_trace.events if e.kind == "operation"]
    assert [e.payload["request_id"] for e in operations] == ["customer_001", "customer_000"]
    assert execution.oracle.status == "pass"


@pytest.mark.parametrize("mode,success,violations", [("reference", False, 0), ("sandbox", True, 1)])
def test_permission_denial_and_actual_sandbox_violation(mode, success, violations):
    # Seed 42 has a real active Member in org_000 with an expired subscription.
    target = Operation(name="cancel_renewal", org_id="org_000", resource_id="sub_000")
    spec = Experiment(customer=CustomerInput(identity="user_007", message="Cancel Workspace 0 renewal", authorizations=[target]), mode=mode)
    execution = run([answer(calls=[call("cancel_renewal", org_id="org_000", resource_id="sub_000")]), answer()], spec)
    record = execution.tool_records[0]
    assert record.response["success"] is success
    assert execution.metrics["invalid_operation_attempts"] == 1
    assert execution.metrics["successful_business_violations"] == violations
    if mode == "reference":
        assert record.response["error_code"] == "PERMISSION_DENIED"
        assert execution.world_trace.final_state == execution.world_trace.initial_state
        assert execution.oracle.status == "pass"
    else:
        assert execution.oracle.status == "fail" and "PERMISSION_DENIED" in codes(execution)


@pytest.mark.parametrize("mode,success", [("reference", False), ("sandbox", True)])
def test_unrequested_tool_parameters_cannot_create_customer_authorization(mode, success):
    execution = run([answer(calls=[operation_call("change_seats", seats=4)]), answer()], experiment(mode=mode))
    assert execution.tool_records[0].response["success"] is success
    ungranted = execution.trusted_requests["ungranted_001"]
    assert ungranted.authorized_operations == []
    assert execution.trusted_requests["customer_000"].operation == op()
    assert execution.metrics["unfulfilled_authorizations"] == ["customer_000"]
    assert "INCOMPLETE_REQUEST" in codes(execution)
    if mode == "sandbox":
        assert "UNAUTHORIZED_OPERATION" in codes(execution)
        assert execution.metrics["successful_business_violations"] == 1
    else:
        assert execution.tool_records[0].response["error_code"] == "UNAUTHORIZED_OPERATION"
        assert execution.metrics["successful_business_violations"] == 0


def test_single_use_grants_cannot_be_reused():
    execution = run([answer(calls=[operation_call(cid="one"), operation_call(cid="two")]), answer()])
    assert execution.tool_records[0].response["success"]
    assert execution.tool_records[1].response["error_code"] == "UNAUTHORIZED_OPERATION"
    assert execution.trusted_requests["ungranted_002"].authorized_operations == []


def test_pure_text_success_cannot_hide_missing_customer_request():
    execution = run([answer("Everything is done")])
    assert execution.status == "completed" and execution.final_reply == "Everything is done"
    assert execution.oracle.status == "fail"
    assert "INCOMPLETE_REQUEST" in codes(execution)
    assert execution.metrics["engine_operations"] == 0
    assert execution.metrics["successful_business_violations"] == 0


@pytest.mark.parametrize("arguments", ["{", "[]", '{"org_id":"org_001","resource_id":"sub_001","seats":true}',
    '{"org_id":"org_001","resource_id":"sub_001","seats":2.5}',
    '{"org_id":"org_001","resource_id":"sub_001","seats":NaN}',
    '{"org_id":"org_001","org_id":"org_000","resource_id":"sub_001","seats":4}',
    '{"org_id":"org_001","resource_id":"sub_001","seats":4,"actor_id":"user_000"}'])
def test_invalid_tool_arguments_do_not_execute_and_model_can_continue(arguments):
    malformed = operation_call("change_seats")
    malformed["function"]["arguments"] = arguments
    execution = run([answer(calls=[malformed]), answer(calls=[operation_call(cid="correct")]), answer()])
    assert execution.tool_records[0].response["error_code"] == "INVALID_TOOL_ARGUMENTS"
    assert execution.tool_records[0].event_id == ""
    assert execution.metrics["engine_operations"] == 1 and execution.oracle.status == "pass"
    assert execution.metrics["dispatched_tool_calls"] == 2


def test_identity_and_tool_set_are_fixed():
    execution = run([answer(calls=[call("get_customer_context", identity="user_000"), call("get_world", "unknown")]),
                     answer(calls=[operation_call(cid="cancel")]), answer()])
    assert [r.response["error_code"] for r in execution.tool_records[:2]] == ["INVALID_TOOL_ARGUMENTS", "UNKNOWN_TOOL"]
    assert all(e.payload["actor_id"] == "user_001" for e in execution.world_trace.events if e.kind == "operation")
    with pytest.raises(ValueError):
        experiment(tools=["get_world"])


@pytest.mark.parametrize("response", [{}, {"choices": []}, answer(calls=[{"type": "function", "id": "a", "function": {"name": "cancel_renewal"}}]),
    {"choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": "partial"}}]},
    {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": None}}]},
    {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "done", "tool_calls": {}}}]},
    {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "done", "tool_calls": False}}]}])
def test_abnormal_model_outputs_fail_without_mutating_state(response):
    execution = run([response])
    assert execution.status == "model_output_error" and execution.final_reply is None
    assert execution.world_trace.final_state == execution.world_trace.initial_state
    assert execution.llm_records[0].response == response
    assert execution.llm_records[0].error["code"] == "MODEL_OUTPUT_INVALID"
    assert execution.metrics["engine_operations"] == 0
    assert replay_execution(execution) == execution


def test_duplicate_call_ids_in_batch_are_rejected_before_any_execution():
    execution = run([answer(calls=[operation_call(cid="same"), operation_call(cid="same")])])
    assert execution.status == "model_output_error" and not execution.tool_records


def test_reused_call_id_in_later_model_turn_cannot_execute_again():
    execution = run([answer(calls=[operation_call(cid="same")]), answer(calls=[operation_call(cid="same")])])
    assert execution.status == "model_output_error" and execution.metrics["engine_operations"] == 1
    assert execution.oracle.status == "pass" and replay_execution(execution) == execution


def test_provider_failure_still_exports_authority_and_oracle():
    execution = run([ServiceError("PROVIDER_TIMEOUT", "request timed out")])
    assert execution.status == "provider_error" and execution.oracle.status == "fail"
    assert execution.llm_records[0].error["code"] == "PROVIDER_TIMEOUT"
    assert replay_execution(execution) == execution


def test_interrupted_model_request_saves_a_replayable_failure():
    execution = run([KeyboardInterrupt()])
    assert execution.status == "interrupted" and execution.final_reply is None
    assert execution.llm_records[0].error["code"] == "PROVIDER_INTERRUPTED"
    assert replay_execution(execution) == execution


def test_model_budget_is_enforced_without_fabricating_final_reply():
    execution = run([answer(calls=[operation_call()])], experiment(max_llm_calls=1))
    assert execution.status == "llm_budget_exhausted" and execution.final_reply is None
    assert execution.oracle.status == "pass"  # actual business execution was correct
    assert execution.metrics["llm_calls"] == 1


def test_tool_budget_blocks_later_calls_in_same_batch():
    execution = run([answer(calls=[call("get_customer_context"), operation_call(cid="blocked")])], experiment(max_tool_calls=1))
    assert execution.status == "tool_budget_exhausted"
    assert execution.tool_records[1].response["error_code"] == "TOOL_BUDGET_EXHAUSTED"
    assert not execution.tool_records[1].dispatched
    assert execution.metrics["dispatched_tool_calls"] == 1
    assert execution.metrics["engine_operations"] == 0
    assert execution.world_trace.final_state == execution.world_trace.initial_state
    assert replay_execution(execution) == execution


def test_at_exact_tool_budget_model_can_still_return_a_final_reply():
    client = FixtureClient([answer(calls=[operation_call()]), answer()])
    execution = ServiceSession(experiment(max_tool_calls=1), client).run()
    assert execution.status == "completed" and execution.oracle.status == "pass"
    assert client.requests[-1]["tools"] == [] and client.requests[-1]["tool_choice"] == "none"


def test_discovery_is_scoped_and_real_query_permission_is_enforced():
    execution = run([answer(calls=[call("get_organization", "org", org_id="org_001", resource_id="org_001"),
        call("list_members", "members", org_id="org_001", resource_id="org_001"),
        call("get_invoice", "cross", org_id="org_001", resource_id="invoice_000_recent")]), answer()], experiment([]))
    assert execution.tool_records[0].response["discovery"]["invoice_ids"] == list(i.id for i in generate_world(42).invoices.values() if i.org_id == "org_001")
    assert all(u["id"] in {m.user_id for m in generate_world(42).memberships.values() if m.org_id == "org_001"}
               for u in execution.tool_records[1].response["discovery"]["users"])
    cross = execution.tool_records[2].response
    assert cross["error_code"] == "CROSS_ORG_RESOURCE" and cross["data"] == {} and "discovery" not in cross


def test_reset_and_replay_preserve_full_transcript_and_core_events(tmp_path):
    client = FixtureClient([answer(calls=[operation_call()]), answer()]*2)
    session = ServiceSession(experiment(), client)
    first = session.run()
    path = tmp_path / "execution.json"
    first.save(path)
    assert replay_execution(path, experiment()) == first
    assert replay_trace(first.world_trace).state == session.state
    assert session.reset() == generate_world(42)
    second = session.run()
    assert first == second
    with pytest.raises(ValueError, match="reset"):
        session.run()


def test_replay_rejects_forged_authorization_or_tool_evidence_even_with_rehashed_wrapper():
    execution = run([answer(calls=[operation_call()]), answer()])
    forged = execution.model_copy(deep=True)
    forged.tool_records[0].response["data"] = {"forged": True}
    forged.execution_hash = forged.computed_hash()
    with pytest.raises(ValueError, match="altered"):
        replay_execution(forged)
    wrong_customer = experiment()
    wrong_customer.customer.identity = "user_000"
    with pytest.raises(ValueError, match="external trusted"):
        replay_execution(execution, wrong_customer)
    execution.execution_hash = "forged"
    with pytest.raises(ValueError, match="corrupt"):
        replay_execution(execution)


def test_environment_configuration_required_without_model_defaults():
    from usami.service.client import settings_from_env, key_from_env
    with pytest.raises(ValueError, match="USAMI_LLM_MODEL"):
        settings_from_env({})
    with pytest.raises(ValueError, match="USAMI_LLM_API_KEY"):
        key_from_env({})
    settings = settings_from_env({"USAMI_LLM_MODEL": "provider/model", "USAMI_LLM_BASE_URL": "https://example.invalid/v1/",
                                  "USAMI_LLM_OPTIONS": '{"max_tokens":300,"temperature":0}'})
    assert settings.base_url == "https://example.invalid/v1"
    assert settings.options["temperature"] == 0
    assert key_from_env({"USAMI_LLM_API_KEY": "test-secret"}) == "test-secret"
    with pytest.raises(ValueError):
        ModelSettings(model="test", base_url="https://user:password@example.invalid/v1")
    with pytest.raises(ValueError):
        ModelSettings(model="test", base_url="https://example.invalid/v1", options={"messages": []})


def test_http_transport_uses_actual_function_call_protocol_and_never_exports_key():
    httpx = pytest.importorskip("httpx")
    from usami.service import OpenAICompatibleClient
    key = "test-credential-not-for-export"
    seen = []

    def endpoint(request):
        assert request.headers["Authorization"] == "Bearer " + key
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=answer(calls=[operation_call()]) if len(seen) == 1 else answer(key))

    client = OpenAICompatibleClient(FixtureClient.settings, key, transport=httpx.MockTransport(endpoint))
    try:
        execution = ServiceSession(experiment(), client).run()
    finally:
        client.close()
    assert execution.oracle.status == "pass" and len(seen) == 2
    assert seen[0]["tool_choice"] == "auto" and seen[0]["stream"] is False
    assert seen[1]["messages"][-1]["role"] == "tool"
    assert key not in execution.model_dump_json() and "[REDACTED]" in execution.final_reply


def test_http_transport_omits_empty_tool_calls_and_internal_message_hashes():
    httpx = pytest.importorskip("httpx")
    from usami.service import OpenAICompatibleClient
    tool_call = {"id": "call_1", "type": "function",
                 "function": {"name": "get_order", "arguments": "{}"}}
    request = {"model": "test", "messages": [
        {"role": "system", "content": "instructions", "record_hash": "internal-hash-1"},
        {"role": "user", "content": "help", "tool_calls": [], "record_hash": "internal-hash-2"},
        {"role": "assistant", "content": None, "tool_calls": [tool_call], "record_hash": "internal-hash-3"},
    ]}
    seen = []

    def endpoint(provider_request):
        seen.append(json.loads(provider_request.content))
        return httpx.Response(200, json=answer("OK"))

    client = OpenAICompatibleClient(FixtureClient.settings, "test-key",
                                   transport=httpx.MockTransport(endpoint))
    try:
        client.complete(request)
    finally:
        client.close()

    sent = seen[0]["messages"]
    assert "record_hash" not in sent[0] and "record_hash" not in sent[1]
    assert "tool_calls" not in sent[1]
    assert sent[2]["tool_calls"] == [tool_call]
    # Provider normalization must not alter the immutable/audited input.
    assert request["messages"][1]["tool_calls"] == []
    assert all("record_hash" in message for message in request["messages"])


@pytest.mark.parametrize("status", [401, 429, 500, 302])
def test_http_errors_do_not_retry_or_echo_sensitive_bodies(status):
    httpx = pytest.importorskip("httpx")
    from usami.service import OpenAICompatibleClient
    calls = []

    def endpoint(request):
        calls.append(request)
        return httpx.Response(status, json={"error": "gateway rejected forced tool_choice",
                                            "echo": "credential-secret"},
                              headers={"Location": "https://other.invalid"})

    client = OpenAICompatibleClient(FixtureClient.settings, "credential-secret", transport=httpx.MockTransport(endpoint))
    try:
        execution = ServiceSession(experiment(), client).run()
    finally:
        client.close()
    assert execution.status == "provider_error" and len(calls) == 1
    assert "credential-secret" not in execution.model_dump_json()
    assert "gateway rejected forced tool_choice" in execution.model_dump_json()
    assert execution.llm_records[0].error["code"] == "PROVIDER_HTTP_ERROR"
    assert execution.llm_records[0].response["status_code"] == status
    assert execution.llm_records[0].response["truncated"] is False


@pytest.mark.parametrize("name,resource_id,params", [
    ("refund_invoice", "invoice_001_recent", {"amount_cents": 48300, "payment_method": "card_mastercard"}),
    ("change_member_role", "membership_001_user_005", {"role": "member"}),
])
def test_invoice_and_role_tools_use_real_engine_transitions(name, resource_id, params):
    target = Operation(name=name, org_id="org_001", resource_id=resource_id, params=params)
    spec = Experiment(customer=CustomerInput(identity="user_001", message="Please perform this account update.", authorizations=[target]))
    execution = run([answer(calls=[call(name, org_id="org_001", resource_id=resource_id, **params)]), answer()], spec)
    assert execution.oracle.status == "pass" and execution.tool_records[0].response["success"]
    final = execution.world_trace.final_state
    if name == "refund_invoice":
        invoice = final.invoices[resource_id]
        assert invoice.status == "refunded" and len(invoice.refunds) == 1
        assert invoice.refunds[0].amount_cents == params["amount_cents"]
        assert invoice.refunds[0].payment_method == params["payment_method"]
    else:
        assert final.memberships[resource_id].role == "member"
        assert final.memberships["membership_001_user_001"].role == "owner"
    assert execution.tool_records[0].before_hash != execution.tool_records[0].after_hash
    assert replay_execution(execution) == execution


@pytest.mark.parametrize("fault", ["timeout", "nonjson", "duplicate_json"])
def test_transport_timeout_and_invalid_http_success_body_are_replayable(fault):
    httpx = pytest.importorskip("httpx")
    from usami.service import OpenAICompatibleClient
    attempts = []

    def endpoint(request):
        attempts.append(request)
        if fault == "timeout":
            raise httpx.ReadTimeout("sensitive-provider-detail", request=request)
        body = "not JSON sensitive-provider-detail" if fault == "nonjson" else '{"choices":[],"choices":[]}'
        return httpx.Response(200, text=body)

    client = OpenAICompatibleClient(FixtureClient.settings, "sensitive-provider-detail", transport=httpx.MockTransport(endpoint))
    try:
        execution = ServiceSession(experiment(), client).run()
    finally:
        client.close()
    assert len(attempts) == 1 and execution.metrics["engine_operations"] == 0
    assert execution.status == ("provider_error" if fault == "timeout" else "model_output_error")
    if fault == "timeout":
        assert execution.llm_records[0].response == {"timeout_phase": "ReadTimeout"}
    assert "sensitive-provider-detail" not in execution.model_dump_json()
    assert replay_execution(execution) == execution


def test_cli_run_exports_execution_and_replay_needs_no_model_client(tmp_path, monkeypatch, capsys):
    from usami.service.__main__ import main
    from usami.service import OpenAICompatibleClient, ServiceExecution
    spec, execution_path = tmp_path / "experiment.json", tmp_path / "execution.json"
    spec.write_text(experiment().model_dump_json(), encoding="utf-8")
    client = FixtureClient([answer(calls=[operation_call()]), answer("Renewal cancelled")])
    closed = []
    client.close = lambda: closed.append(True)
    monkeypatch.setattr(OpenAICompatibleClient, "from_env", lambda: client)
    assert main(["run", "--experiment", str(spec), "--out", str(execution_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["final_reply"] == "Renewal cancelled" and summary["oracle_status"] == "pass"
    assert closed == [True] and ServiceExecution.load(execution_path).status == "completed"

    def unexpected_client():
        pytest.fail("replay must not load provider configuration or credentials")

    monkeypatch.setattr(OpenAICompatibleClient, "from_env", unexpected_client)
    assert main(["replay", str(execution_path), "--experiment", str(spec)]) == 0
    replay = json.loads(capsys.readouterr().out)
    assert replay["replayed"] and replay["execution_hash"] == summary["execution_hash"]


def test_keychain_lookup_is_explicit_and_environment_key_takes_precedence(monkeypatch):
    from types import SimpleNamespace
    from usami.service.client import key_from_env
    lookups = []

    def lookup(args, **kwargs):
        lookups.append(args)
        assert kwargs == {"capture_output": True, "text": True}
        return SimpleNamespace(returncode=0, stdout="keychain-test-secret\n")

    monkeypatch.setattr("usami.service.client.subprocess.run", lookup)
    env = {"USAMI_LLM_KEYCHAIN_SERVICE": "configured-service", "USAMI_LLM_KEYCHAIN_ACCOUNT": "configured-account"}
    assert key_from_env(env) == "keychain-test-secret"
    assert lookups == [["security", "find-generic-password", "-s", "configured-service", "-a", "configured-account", "-w"]]
    assert key_from_env({**env, "USAMI_LLM_API_KEY": "environment-test-secret"}) == "environment-test-secret"
    assert len(lookups) == 1
    with pytest.raises(ValueError, match="KEYCHAIN_ACCOUNT"):
        key_from_env({"USAMI_LLM_KEYCHAIN_SERVICE": "configured-service"})
