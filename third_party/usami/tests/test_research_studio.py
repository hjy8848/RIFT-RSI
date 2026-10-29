"""Research Studio records native evidence and isolates candidate Skills."""
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from usami.service.models import ModelSettings, ServiceError
from usami_v2.runtime.agents import OpenAIServiceAgent, ServicePrompt
from usami_v2.scenarios.distribution import sample_schedule
from usami_v2.scenarios.designer import OpenAIDesignerProvider, ScenarioDraft
from vast_coevo.schema import digest
from vast_coevo.skills.library import LibrarySnapshot, empty_snapshot
from vast_coevo.skills.models import DecisionSkill, Scope, Trigger
from web.backend.human_customer import HUMAN_COOKIE, HumanSessionManager
from web.backend.research_studio import (
    ResearchRunCreate, ResearchRunManager, _designer_client, _designer_settings, _distribution,
    install_research_studio_routes,
)


class TextReplyClient:
    def __init__(self):
        self.settings = ModelSettings(model="fixed-test-model", base_url="http://127.0.0.1/v1",
                                      options={"max_tokens": 64})
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return {"choices": [{"message": {"role": "assistant", "content": "I can help."},
                              "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3}}


def test_research_distribution_reseals_difficulty_strata_and_samples():
    for name in ("mixed_after_sales", "multi_line", "inventory_and_deadline"):
        spec = _distribution(name, "train", "basic")
        checked = spec.checked()
        assert checked.record_hash == spec.record_hash
        assert all("small number of prerequisites" in row.contexts[-1] for row in spec.strata)
        assert len(sample_schedule(spec, seed=42, count=1)) == 1


def test_research_designer_uses_frozen_long_timeout_without_output_token_cap():
    settings = ModelSettings(model="fixed-test-model", base_url="http://127.0.0.1/v1",
                             timeout_seconds=60, options={"max_tokens": 2048, "temperature": 0})
    configured = _designer_settings(settings, 180)

    assert configured.timeout_seconds == 180
    assert configured.options == {"temperature": 0}
    assert ResearchRunCreate().designer_timeout_seconds == 180
    with pytest.raises(ValueError):
        ResearchRunCreate(designer_timeout_seconds=601)


def test_research_designer_http_transport_uses_frozen_timeout():
    settings = ModelSettings(model="fixed-test-model", base_url="http://127.0.0.1/v1",
                             timeout_seconds=180, options={})
    client = _designer_client(settings, {"USAMI_LLM_API_KEY": "test-key"})
    try:
        assert client.settings.timeout_seconds == 180
        assert client._http.timeout.read == 180
        assert client._http.timeout.connect == 180
    finally:
        client.close()


def test_designer_failure_record_keeps_only_redacted_provider_diagnostics():
    class FailingClient(TextReplyClient):
        def complete(self, request):
            raise ServiceError("PROVIDER_HTTP_ERROR", "HTTP 400", {
                "status_code": 400, "body": "bad request sk-secret-value"})

        def _redact(self, value):
            return json.loads(json.dumps(value).replace("sk-secret-value", "[REDACTED]"))

    provider = OpenAIDesignerProvider(FailingClient())
    with pytest.raises(ServiceError):
        provider.submit("world", [{"role": "user", "content": "test"}], ScenarioDraft)
    assert provider.last_call is not None
    error = provider.last_call.response["error"]
    assert error["code"] == "PROVIDER_HTTP_ERROR"
    assert error["provider_response"]["status_code"] == 400
    assert "sk-secret-value" not in json.dumps(provider.last_call.model_dump())


def test_qwen3_designer_uses_auto_but_still_requires_a_complete_function_call():
    client = TextReplyClient()
    client.settings = ModelSettings(model="dashscope/qwen3.7-plus", base_url="http://127.0.0.1/v1",
                                   options={"max_tokens": 64})
    provider = OpenAIDesignerProvider(client)
    with pytest.raises(ServiceError, match="exactly one complete structured function call"):
        provider.submit("world", [{"role": "user", "content": "test"}], ScenarioDraft)
    request = client.requests[0]
    assert request["tool_choice"] == "auto"
    assert request["parallel_tool_calls"] is False
    assert len(request["tools"]) == 1


def test_non_qwen_designer_keeps_forced_phase_submission_tool():
    client = TextReplyClient()
    provider = OpenAIDesignerProvider(client)
    with pytest.raises(ServiceError, match="exactly one complete structured function call"):
        provider.submit("task", [{"role": "user", "content": "test"}], ScenarioDraft)
    request = client.requests[0]
    assert request["tool_choice"] == {
        "type": "function", "function": {"name": "submit_customer_task"}}
    assert request["parallel_tool_calls"] is False


def test_candidate_skill_enters_only_the_s1_real_model_request():
    skill = DecisionSkill(
        id="transaction-consent-check", version=1,
        scope=Scope(domains=("aftersales",)),
        trigger=Trigger(keywords=("refund",)),
        description="Check transaction preconditions from observed case state.",
        preconditions=("Use the current tool results for the case.",),
        procedure=("Check prerequisites before committing a quote.",),
        postconditions=("Confirm the operation result before reporting completion.",),
        recovery=("If a prerequisite is absent, explain what remains needed.",),
    )
    candidate = LibrarySnapshot(role="service", skills=(skill,), provenance="test candidate")
    prompt = ServicePrompt(customer_id="customer-1", customer_request="Please refund these headphones.")
    s0_client, s1_client = TextReplyClient(), TextReplyClient()

    assert OpenAIServiceAgent(s0_client).decide(prompt).kind == "reply"
    s1 = OpenAIServiceAgent(s1_client, skill_snapshot=candidate)
    assert s1.decide(prompt).kind == "reply"

    s0_request, s1_request = s0_client.requests[0], s1_client.requests[0]
    assert s0_request["model"] == s1_request["model"]
    assert s0_request["tools"] == s1_request["tools"]
    assert s0_request["messages"][:2] == s1_request["messages"][:2]
    assert len(s0_request["messages"]) == 2
    assert len(s1_request["messages"]) == 3
    rendered = s1_request["messages"][-1]["content"]
    assert skill.id in rendered and skill.content_hash in rendered
    assert s1.last_skill_injection["snapshot_hash"] == candidate.snapshot_hash
    assert s1.last_skill_injection["rendered_hash"] == digest(rendered)
    assert s1.last_skill_injection["request_hash"] == digest(s1_request)
    assert len(s1.last_skill_injection["hits"]) == 1


def test_research_archive_hash_chain_detects_row_and_journal_tampering(tmp_path):
    manager = ResearchRunManager(object(), output_dir=tmp_path / "research")
    row = {"run_id": "rs-integrity-test", "status": "READY", "phases": {}, "journal": []}
    manager._persist(row, "created")
    folder = tmp_path / "research" / row["run_id"]
    saved = json.loads((folder / "current.json").read_text(encoding="utf-8"))

    assert ResearchRunManager._verify_persisted_row(saved, folder / "journal.jsonl")
    changed = dict(saved)
    changed["status"] = "FAILED"
    assert not ResearchRunManager._verify_persisted_row(changed, folder / "journal.jsonl")

    journal = folder / "journal.jsonl"
    journal.write_text(journal.read_text(encoding="utf-8").replace("created", "changed"), encoding="utf-8")
    assert not ResearchRunManager._verify_persisted_row(saved, journal)


def test_research_routes_bootstrap_shared_local_owner_cookie(tmp_path):
    app = FastAPI()
    human = HumanSessionManager(output_dir=tmp_path / "human")
    install_research_studio_routes(app, human, output_dir=tmp_path / "research")
    with TestClient(app, base_url="http://127.0.0.1:5173") as client:
        history = client.get("/api/research/runs")
        assert history.status_code == 200
        cookie = history.headers.get("set-cookie", "").lower()
        assert "httponly" in cookie and "samesite=strict" in cookie and "path=/api" in cookie
        assert "max-age=2592000" in cookie
        config = client.get("/api/research/config")
        assert config.status_code == 200
        assert "credential" not in json.dumps(config.json()).lower()


def test_local_run_can_be_recovered_after_owner_cookie_expires(tmp_path):
    app = FastAPI()
    human = HumanSessionManager(output_dir=tmp_path / "human")
    install_research_studio_routes(app, human, output_dir=tmp_path / "research")
    manager = app.state.research_runs
    run_id = "rs-expired-owner-recovery"
    row = {
        "run_id": run_id, "owner_hash": digest("previous-browser-owner"),
        "status": "READY_FOR_CUSTOMER", "created_at": "2026-10-23T00:00:00+00:00",
        "updated_at": "2026-10-23T00:00:00+00:00", "phases": {"human_episode": "COMPLETED"},
        "episodes": [{"session_id": "saved-episode-id", "arm": "S0"}], "journal": [],
    }
    manager.runs[run_id] = row
    manager._persist(row, "episode_archived")

    with TestClient(app, base_url="http://127.0.0.1:5173") as client:
        history = client.get("/api/research/runs")
        assert history.status_code == 200
        assert history.json()["runs"] == []
        new_owner = client.cookies.get(HUMAN_COOKIE)
        assert new_owner

        claim = client.post(f"/api/research/runs/{run_id}/claim", json={},
                            headers={"Origin": "http://127.0.0.1:5173"})
        assert claim.status_code == 200
        assert claim.json()["episodes"][0]["session_id"] == "saved-episode-id"
        assert manager.get(new_owner, run_id)["episodes"][0]["arm"] == "S0"

    restarted = ResearchRunManager(HumanSessionManager(output_dir=tmp_path / "human"),
                                   output_dir=tmp_path / "research")
    assert restarted.get(new_owner, run_id)["episodes"][0]["session_id"] == "saved-episode-id"
    saved = json.loads((tmp_path / "research" / run_id / "current.json").read_text())
    assert ResearchRunManager._verify_persisted_row(
        saved, tmp_path / "research" / run_id / "journal.jsonl")


def test_reflector_can_use_attached_episode_after_human_session_expires(tmp_path, monkeypatch):
    """The durable ResearchRun Episode, not the RAM session, is reflection input."""
    from test_usami_v2_runtime import (
        ScriptedCustomer, ScriptedRefundService, frozen, refund_profile, refund_task,
    )
    from usami_v2.runtime.runner import EpisodeBudget, replay_episode, run_episode
    from vast_coevo.models import ModelSpec
    import web.backend.research_studio as research_studio

    world, aliases = frozen(seed=912)
    record = run_episode(world, refund_task(world, aliases), refund_profile(aliases),
        ScriptedCustomer(), ScriptedRefundService(world, aliases),
        budget=EpisodeBudget(max_model_calls=0))
    assert replay_episode(record).verified
    run_id, session_id, owner_id = "rs-reflection-recovery", "session-expired-123456", "owner-test"
    scenario_id = "research-recovery-test"
    service_model = ModelSpec(model="historical-service-model", base_url="http://localhost/v1")
    report = {
        "customer_source": "human", "study_type": "qualitative_manual_pilot",
        "research_run_id": run_id, "comparison_arm": "S0",
        "service_skill_snapshot_hash": None,
        "scenario": {"scenario_id": scenario_id},
        "episode": record.model_dump(mode="json"),
        "outcome": record.outcome.model_dump(mode="json"),
        "service_model_spec": service_model.model_dump(mode="json"),
    }
    link = {
        "session_id": session_id, "arm": "S0", "customer_source": "human",
        "study_type": "qualitative_manual_pilot", "initial_world_hash": record.initial_world.state_hash,
        "task_hash": record.task.record_hash, "episode_hash": record.record_hash,
        "replay": replay_episode(record).model_dump(mode="json"),
        "explicitly_selected_for_reflection": False, "report": report,
    }
    root = tmp_path / "research"
    manager = ResearchRunManager(HumanSessionManager(output_dir=tmp_path / "human"), output_dir=root)
    row = {
        "protocol": "usami-research-studio-v1", "run_id": run_id,
        "owner_hash": digest(owner_id), "status": "READY_FOR_CUSTOMER",
        "created_at": "2026-10-23T00:00:00+00:00", "updated_at": "2026-10-23T00:00:00+00:00",
        "input": {"split": "train"},
        "scenario": {"scenario_id": scenario_id,
            "public": {"initial_world_hash": record.initial_world.state_hash},
            "task": {"record_hash": record.task.record_hash}},
        "episodes": [link], "reflection": None,
        "phases": {"reflection": "NOT_STARTED"}, "journal": [],
    }
    manager.runs[run_id] = row
    manager._persist(row, "episode_archived")

    # A newly constructed manager models a server restart: its HumanSession map is empty.
    restarted = ResearchRunManager(HumanSessionManager(output_dir=tmp_path / "human"), output_dir=root)
    assert not restarted.human.sessions
    selected = restarted.attach_episode(owner_id, run_id, session_id, "S0", use_for_reflection=True)
    assert selected["episodes"][0]["explicitly_selected_for_reflection"] is True

    monkeypatch.setattr(research_studio, "settings_from_env", lambda: ModelSettings(
        model="reflector-test-model", base_url="http://localhost/v1", options={}))
    captured = {}
    import threading
    done = threading.Event()

    def capture_worker(run_id, reflection_input, settings, episode_id):
        captured.update(run_id=run_id, reflection_input=reflection_input, settings=settings,
                        episode_id=episode_id)
        done.set()

    restarted._reflection_worker = capture_worker
    started = restarted.reflect(owner_id, run_id, session_id)
    assert started["status"] == "REFLECTOR_RUNNING"
    assert done.wait(2)
    assert captured["episode_id"] == session_id
    evidence = captured["reflection_input"].evidence[0]
    assert evidence.native_trace_hash == record.record_hash
    assert evidence.config.service_model.model == "historical-service-model"
    assert evidence.config.service_model.timeout_seconds == 60
    assert captured["settings"].timeout_seconds == 180
    assert evidence.consumption.tool_calls == len(record.tool_executions)


def test_saved_task_revalidation_preserves_raw_calls_and_runs_native_return_flow(tmp_path):
    from test_usami_v2_designer import distribution_and_slot, scenario_draft, task_draft, replace_sealed
    from usami_v2.scenarios.builder import build_world
    from usami_v2.scenarios.designer import DesignerCall, DesignerRun
    from usami_v2.runtime.agents import AgentCall, ServiceAction
    from usami_v2.runtime.customer import CustomerDecision
    from usami_v2.runtime.runner import EpisodeBudget, replay_episode, run_episode
    from usami_v2.commerce.models import Operation
    from web.backend.human_customer import _return_events

    distribution, slot = distribution_and_slot(42)
    world_draft = scenario_draft()
    world_draft = replace_sealed(world_draft, skus=tuple(
        replace_sealed(sku, return_required=True) if sku.alias == "original" else sku
        for sku in world_draft.skus))
    spec = world_draft.compile(distribution)
    build = build_world(spec, slot.seed)
    draft = task_draft()
    call = AgentCall(role="designer", request={"messages": []}, response={"choices": [{
        "finish_reason": "tool_calls", "message": {"role": "assistant", "tool_calls": [{
            "id": "saved-submission", "type": "function", "function": {
                "name": "submit_customer_task", "arguments": draft.model_dump_json()}}]}}]})
    source = DesignerRun(run_id="designer-saved-test", status="BLOCKED", sampling_slot=slot,
        distribution=distribution, scenario_spec=spec, build_result=build,
        calls=(DesignerCall(phase="task", attempt=1, call=call, parsed_output_hash="",
                            validation_feedback="original compiler rejected the proposal"),))
    manager = ResearchRunManager(HumanSessionManager(output_dir=tmp_path / "human"),
                                 output_dir=tmp_path / "research")
    row = {"run_id": "rs-saved-test", "owner_hash": digest("test-owner"), "status": "BLOCKED",
        "input": ResearchRunCreate().model_dump(mode="json"), "episodes": [], "journal": [],
        "phases": {"world": "COMPLETED", "task": "REJECTED"}, "designer_run": source.model_dump(mode="json")}
    manager.runs[row["run_id"]] = row
    manager._persist(row, "original_failed_generation")
    original = json.loads(json.dumps(row))

    restored = manager.revalidate_saved_task(row["run_id"])
    assert restored["status"] == "READY_FOR_CUSTOMER"
    assert restored["designer_run"]["calls"] == original["designer_run"]["calls"]
    assert restored["designer_run"]["build_result"] == original["designer_run"]["build_result"]
    assert restored["designer_run"]["task_draft"] == draft.model_dump(mode="json")
    report = restored["task_revalidations"][-1]
    assert report["model_calls"] == 0 and report["replay"]["verified"]
    archived = json.loads((manager.root / row["run_id"] / report["original_record_file"]).read_text())
    assert archived == original
    assert manager._verify_persisted_row(row, manager.root / row["run_id"] / "journal.jsonl")
    reloaded = ResearchRunManager(HumanSessionManager(output_dir=tmp_path / "human"), output_dir=manager.root)
    assert reloaded.runs[row["run_id"]]["status"] == "READY_FOR_CUSTOMER"
    scenario = reloaded.human.scenarios[restored["scenario"]["scenario_id"]]
    assert scenario.task.protocol == "commerce-task-v2"

    # Test-only actors follow the real Reference witness; authority, return
    # receipt, settlement, Oracle and Replay still run through the native Runner.
    actions = []
    witness = next(a for a in scenario.task_certificate.attempts if a.status == "WITNESS")
    for step in witness.steps:
        if step.kind == "customer_consent":
            consent = step.evidence["consent"]
            actions.append(ServiceAction(kind="reply", message="Please accept this exact replacement quote.",
                present_quote_id=consent["quote_id"], present_for=consent["operation"]))
        elif step.kind == "operation":
            actions.append(ServiceAction(kind="operation",
                operation=Operation.model_validate(step.evidence["request"]["operation"])))
    actions.append(ServiceAction(kind="reply", message="The replacement has been shipped."))

    class WitnessService:
        uses_model = False
        def __init__(self): self.actions = iter(actions)
        def decide(self, prompt): return next(self.actions)

    class AcceptingCustomer:
        uses_model = False
        def decide(self, prompt, profile):
            if prompt.offer:
                return CustomerDecision(message="I accept this quote.", accept_quote_id=prompt.offer.quote_id,
                                        accept_operation=prompt.offer.operation)
            return CustomerDecision(message="Thank you.", finish=True)

    events = _return_events(scenario)
    assert len(events) == 1
    unrelated_profile = replace_sealed(scenario.profile,
        delegation_request_spans={scenario.task.remedy_goals[0].line_id: "unrequested product"})
    with pytest.raises(ValueError, match="public request span or subject wording"):
        run_episode(scenario.world, scenario.task, unrelated_profile, AcceptingCustomer(), WitnessService())
    without_receipt = run_episode(scenario.world, scenario.task, scenario.profile,
        AcceptingCustomer(), WitnessService(), budget=EpisodeBudget(max_model_calls=0))
    assert without_receipt.outcome.business_goal_success is False
    assert replay_episode(without_receipt).verified
    record = run_episode(scenario.world, scenario.task, scenario.profile, AcceptingCustomer(), WitnessService(),
        budget=EpisodeBudget(max_customer_turns=3, max_service_decisions=20, max_service_tool_calls=16,
                             max_model_calls=0), external_events=events)
    assert record.outcome.business_goal_success is True
    assert any(event.kind == "external_event" for event in record.events)
    assert record.replay_verified and replay_episode(record).verified
    assert not record.model_calls
    from usami_v2.runtime.customer import HumanCustomerProfileV2
    from usami_v2.runtime.runner import EpisodeRecord
    for index, action in enumerate(actions):
        if action.operation and action.operation.name == "prepare_remedy_quote":
            operation = action.operation.model_copy(update={"params": {
                **action.operation.params, "pickup_slot": "2026-01-15T09:00:00Z/2026-01-15T17:00:00Z"}})
            actions[index] = ServiceAction(kind="operation", operation=operation)
    automated = run_episode(scenario.world, scenario.task, scenario.profile,
        AcceptingCustomer(), WitnessService(), external_events=events)
    assert automated.status == "customer_protocol_error"
    human_profile = HumanCustomerProfileV2.model_validate({
        **scenario.profile.model_dump(mode="json", exclude={"record_hash"}),
        "return_pickup_policy": "accept_presented_quote"})
    human = run_episode(scenario.world, scenario.task, human_profile,
        AcceptingCustomer(), WitnessService(), external_events=events)
    assert human.status == "customer_finished" and human.outcome.business_goal_success is True
    reloaded_episode = EpisodeRecord.model_validate(human.model_dump(mode="json"))
    assert reloaded_episode.profile.return_pickup_policy == "accept_presented_quote"
    assert replay_episode(reloaded_episode).verified
    with pytest.raises(ValueError, match="only a blocked run"):
        manager.revalidate_saved_task(row["run_id"])


@pytest.fixture
def native_process_error_episode():
    """A genuine blocked attempt, followed by a legal refund; no provider calls."""
    from test_usami_v2_runtime import (
        ScriptedCustomer, ScriptedRefundService, frozen, refund_profile, refund_task,
    )
    from usami_v2.commerce.models import Operation
    from usami_v2.runtime.agents import AgentCall, ServiceAction, service_tools
    from usami_v2.runtime.runner import EpisodeBudget, run_episode

    class RecoveringService(ScriptedRefundService):
        tried_early_quote = False

        def decide(self, prompt):
            if self.index == 1 and not self.tried_early_quote:
                self.tried_early_quote = True
                action = ServiceAction(kind="operation", operation=Operation(
                    name="prepare_remedy_quote", scope_id=next(iter(self.world.merchants)),
                    resource_id=self.case, params={"remedy": "refund", "target_sku_id": None,
                                                   "warehouse_id": None, "pickup_slot": None}))
            else:
                action = super().decide(prompt)
            if action.operation is None:
                return action
            operation = action.operation.model_dump(mode="json", exclude={"record_hash"})
            name = operation.pop("name")
            call_id = "fixture-tool-" + str(self.index) + "-" + name
            call = AgentCall(role="service", request={
                "messages": [{"role": "user", "content": prompt.customer_request}],
                "tools": service_tools()}, response={"choices": [{"message": {
                    "role": "assistant", "tool_calls": [{"id": call_id, "type": "function",
                    "function": {"name": name, "arguments": json.dumps(operation)}}]}}]})
            return ServiceAction.model_validate({
                **action.model_dump(mode="json", exclude={"record_hash"}),
                "call": call.model_dump(mode="json"), "call_id": call_id})

    world, aliases = frozen(seed=913)
    return run_episode(world, refund_task(world, aliases), refund_profile(aliases),
                       ScriptedCustomer(), RecoveringService(world, aliases),
                       budget=EpisodeBudget(max_model_calls=16))


def test_native_reflection_links_restarted_oracle_ids_to_exact_tool_execution(native_process_error_episode):
    from vast_coevo.models import ModelSpec
    from web.backend.research_studio import _native_evidence

    record = native_process_error_episode
    native_rows = [row for execution in record.tool_executions
                   for row in execution.oracle["operations"]]
    assert len({row["event_id"] for row in native_rows}) == 1
    evidence, view = _native_evidence(record, "session-collision-test", ModelSpec(
        model="fixture", base_url="http://localhost/v1"))
    assert len({event.event_id for event in evidence.events}) == len(native_rows)
    assert len(view.service_tools) == len(record.tool_executions)
    for index, (event, tool, native) in enumerate(zip(evidence.events, view.service_tools, native_rows)):
        assert event.event_id == tool.event_id
        assert event.classification == tool.oracle_classification == native["classification"]
        assert event.operation_correct == tool.oracle_operation_correct == native["operation_correct"]
        assert event.operation == tool.name == native["operation"]
        assert tool.source_refs == (f"/tool_executions/{index}",)
        assert tool.result == record.tool_executions[index].result
    assert evidence.failures == ()  # No fabricated business violation.
    assert view.confirmed_business_failures == ()


def test_reflector_receives_native_outcome_sources_and_can_still_choose_noop(native_process_error_episode):
    from test_vast_reflection import Client, completion
    from vast_coevo.models import ModelSpec
    from vast_coevo.reflection.reflector import ReflectionInput, ServiceReflector
    from vast_coevo.skills.models import SkillPatch
    from web.backend.research_studio import _native_evidence, NATIVE_REFLECTION_PROTOCOL

    record = native_process_error_episode
    model = ModelSpec(model="fixture", base_url="http://localhost/v1")
    evidence, view = _native_evidence(record, "session-assessment-test", model)
    assessment = evidence.native_assessment
    assert assessment["protocol"] == NATIVE_REFLECTION_PROTOCOL
    assert assessment["native_trace_hash"] == record.record_hash
    assert assessment["outcome"] == record.outcome.model_dump(mode="json")
    assert assessment["outcome"]["business_goal_success"] is True
    assert assessment["outcome"]["process_errors"]["status"] == "FAIL"
    assert assessment["outcome"]["safety_violations"]["status"] == "PASS"
    assert assessment["outcome"]["interaction_reliability"]["status"] == "PASS"
    assert {s["evidence_ref"] for s in assessment["evidence_sources"]} == set(
        record.outcome.process_errors.evidence_refs)
    raw_record = record.model_dump(mode="json")
    for source in assessment["evidence_sources"]:
        value = raw_record
        for part in source["source_ref"].strip("/").split("/"):
            value = value[int(part)] if isinstance(value, list) else value[part]
        assert value == source["native_value"]
        assert digest(value) == source["evidence_ref"]
    assert len(assessment["service_tool_contracts"]) == 1
    assert assessment["service_tool_contracts"][0]["tools"] == record.model_calls[0].request["tools"]
    previous = []
    raw_calls = [call for call in record.model_calls if call.role == "service"]
    for context, call in zip(view.customer_visible_context, raw_calls):
        assert context["encoding"] == "previous-request-prefix-v1"
        previous = previous[:context["unchanged_prefix_messages"]] + context["request_messages"]
        assert previous == call.request["messages"]
    assert len(view.customer_visible_context) == len(raw_calls)
    value = ReflectionInput(generation=0, role="service", library=empty_snapshot("service"),
                            evidence=(evidence,), interaction_views=(view,))
    patch = SkillPatch(action="NOOP", role="service", base_snapshot_hash=value.library.snapshot_hash,
                      evidence_ids=(evidence.episode_id,), rationale="No justified change in this fixture.")
    client = Client(completion(patch.model_dump_json()))
    result = ServiceReflector(client, model).reflect(value)
    assert result.patch == patch and result.calls == len(client.requests) == 1
    payload = json.loads(client.requests[0]["messages"][-1]["content"])
    assert payload["evidence"][0]["native_assessment"] == assessment
    assert payload["evidence"][0]["failures"] == []
    assert "Recovery does not erase" in client.requests[0]["messages"][0]["content"]
    assert "NOOP remains valid" in client.requests[0]["messages"][0]["content"]


def test_native_assessment_tampering_breaks_evidence_hash(native_process_error_episode):
    from vast_coevo.models import EpisodeEvidence, ModelSpec
    from web.backend.research_studio import _native_evidence

    evidence, _ = _native_evidence(native_process_error_episode, "session-tamper-test",
                                  ModelSpec(model="fixture", base_url="http://localhost/v1"))
    raw = evidence.model_dump(mode="json")
    raw["native_assessment"]["outcome"]["process_errors"]["status"] = "PASS"
    with pytest.raises(ValueError, match="evidence content hash mismatch"):
        EpisodeEvidence.model_validate(raw)


def test_old_evidence_hashes_do_not_gain_an_empty_native_assessment(tmp_path):
    from test_vast_reflection import evidence
    from vast_coevo.models import EpisodeEvidence

    original = evidence(tmp_path)
    raw = original.model_dump(mode="json")
    assert "native_assessment" not in raw
    assert original.evidence_hash == digest({k: v for k, v in raw.items() if k != "evidence_hash"})
    assert EpisodeEvidence.model_validate(raw).model_dump(mode="json") == raw


def test_native_tool_cannot_be_linked_to_another_legal_operation(native_process_error_episode):
    from vast_coevo.models import ModelSpec
    from vast_coevo.reflection.evidence import InteractionEvidenceView
    from vast_coevo.reflection.reflector import ReflectionInput
    from web.backend.research_studio import _native_evidence

    evidence, view = _native_evidence(native_process_error_episode, "session-wrong-link",
                                    ModelSpec(model="fixture", base_url="http://localhost/v1"))
    raw = view.model_dump(mode="json")
    # Both operations are legal. Classification checks alone would miss this swap.
    legal = [event for event in evidence.events if event.classification == "legal_execution"]
    assert legal[0].operation != legal[1].operation
    raw["service_tools"][0]["event_id"] = legal[1].event_id
    with pytest.raises(ValueError, match="Oracle linkage"):
        ReflectionInput(generation=0, role="service", library=empty_snapshot("service"),
                        evidence=(evidence,), interaction_views=(InteractionEvidenceView.model_validate(raw),))


def test_legacy_noop_requires_explicit_revision_and_preserves_history(
        native_process_error_episode, tmp_path, monkeypatch):
    import copy
    import threading
    from fastapi import HTTPException
    from vast_coevo.models import ModelSpec
    import web.backend.research_studio as studio

    record = native_process_error_episode
    model = ModelSpec(model="fixture", base_url="http://localhost/v1")
    manager = ResearchRunManager(HumanSessionManager(output_dir=tmp_path / "human"),
                                 output_dir=tmp_path / "research")
    run_id, session_id, owner = "rs-legacy-noop-test", "session-noop-123456", "owner-test"
    legacy = {"status": "NOOP", "input_hash": "legacy-input", "raw_call": {"response": "original"},
              "patch": {"action": "NOOP", "rationale": "original model decision"}}
    row = {"run_id": run_id, "owner_hash": digest(owner), "status": "REFLECTION_COMPLETE",
           "input": {"split": "train"}, "phases": {"reflection": "NOOP"}, "journal": [],
           "scenario": {"scenario_id": "frozen", "public": {"initial_world_hash": record.initial_world.state_hash},
                        "task": {"record_hash": record.task.record_hash}},
           "reflection": copy.deepcopy(legacy), "candidate_skill": None,
           "episodes": [{"session_id": session_id, "arm": "S0", "explicitly_selected_for_reflection": True,
               "episode_hash": record.record_hash, "initial_world_hash": record.initial_world.state_hash,
               "task_hash": record.task.record_hash, "replay": {"verified": True},
               "report": {"research_run_id": run_id, "customer_source": "human",
                   "study_type": "qualitative_manual_pilot", "comparison_arm": "S0",
                   "scenario": {"scenario_id": "frozen"}, "service_skill_snapshot_hash": None,
                   "episode": record.model_dump(mode="json"), "outcome": record.outcome.model_dump(mode="json"),
                   "service_model_spec": model.model_dump(mode="json")}}]}
    manager.runs[run_id] = row
    manager._persist(row, "original_noop_saved")
    monkeypatch.setattr(studio, "settings_from_env", lambda: ModelSettings(
        model="fixture", base_url="http://localhost/v1", options={}))
    done = threading.Event()
    captured = []

    def worker(run_id, reflection_input, settings, episode_id):
        captured.append(reflection_input)
        done.set()

    monkeypatch.setattr(manager, "_reflection_worker", worker)
    with pytest.raises(HTTPException) as blocked:
        manager.reflect(owner, run_id, session_id)
    assert blocked.value.detail["code"] == "REFLECTION_ALREADY_RUN"
    assert not captured and row["reflection"] == legacy
    started = manager.reflect(owner, run_id, session_id, replace_legacy_evidence=True)
    assert done.wait(2)
    assert started["reflection_history"] == [legacy]
    assert started["status"] == "REFLECTOR_RUNNING"
    assert len(captured) == 1
    assert captured[0].evidence[0].native_assessment["outcome"] == record.outcome.model_dump(mode="json")
    with pytest.raises(HTTPException) as running:
        manager.reflect(owner, run_id, session_id, replace_legacy_evidence=True)
    assert running.value.detail["code"] == "REFLECTION_RUNNING"
    row["reflection"] = {"status": "NOOP", "evidence_protocol": studio.NATIVE_REFLECTION_PROTOCOL}
    row["status"] = "REFLECTION_COMPLETE"
    manager._persist(row, "new_noop_saved")
    with pytest.raises(HTTPException) as blocked:
        manager.reflect(owner, run_id, session_id, replace_legacy_evidence=True)
    assert blocked.value.detail["code"] == "LEGACY_REFLECTION_REQUIRED"
    assert len(captured) == 1 and row["reflection_history"] == [legacy]
    saved = json.loads((manager.root / run_id / "current.json").read_text())
    assert saved["reflection_history"] == [legacy]
    assert manager._verify_persisted_row(saved, manager.root / run_id / "journal.jsonl")


@pytest.mark.parametrize('model_name,structured', [
    ('dashscope/qwen3.7-plus', True), ('another-model', True),
    ('dashscope/qwen3.7-plus', False),
])
def test_research_reflector_worker_respects_qwen_tool_choice_and_validates_output(
        tmp_path, monkeypatch, model_name, structured):
    import httpx
    from test_vast_reflection import completion, reflection_input, noop
    import web.backend.research_studio as studio

    value = reflection_input(tmp_path, 'service')
    response = (completion(noop(value).model_dump_json()) if structured else
                {'choices': [{'message': {'content': 'No structured patch submitted.'}}]})
    settings = ModelSettings(model=model_name, base_url='http://localhost/v1', options={}, timeout_seconds=240)
    requests = []

    def mock_provider(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response)

    client = studio.OpenAICompatibleClient(settings, 'fixture-key', transport=httpx.MockTransport(mock_provider))

    def configured_client(actual_settings, key):
        assert actual_settings == settings and key == 'fixture-key'
        assert client._http.timeout.read == actual_settings.timeout_seconds == 240
        return client

    monkeypatch.setattr(studio, 'OpenAICompatibleClient', configured_client)
    monkeypatch.setattr(studio, 'key_from_env', lambda: 'fixture-key')
    manager = ResearchRunManager(HumanSessionManager(output_dir=tmp_path / 'human'),
                                 output_dir=tmp_path / 'research')
    row = {'run_id': 'rs-qwen-reflector-worker', 'status': 'REFLECTOR_RUNNING',
           'phases': {'reflection': 'RUNNING'}, 'journal': []}
    manager.runs[row['run_id']] = row
    manager._reflection_worker(row['run_id'], value, settings, 'source-episode')
    assert len(requests) == 1 and client._http.is_closed
    expected = 'auto' if 'qwen3' in model_name else {
        'type': 'function', 'function': {'name': 'submit_skill_patch'}}
    assert requests[0]['tool_choice'] == expected
    result = row['reflection']
    assert result['calls'] == 1
    assert result['model']['timeout_seconds'] == 240
    if structured:
        assert result['status'] == 'NOOP' and result['patch']['action'] == 'NOOP'
    else:
        assert result['status'] == 'FAILED' and result['patch'] is None
        assert result['error'].startswith('invalid_reflection:')
    assert result['raw_call']['request'] == requests[0]


def test_reflection_timeout_default_and_bounds():
    from web.backend.research_studio import ReflectionCommand

    assert ReflectionCommand(session_id='saved-session-123456').timeout_seconds == 180
    assert ReflectionCommand(session_id='saved-session-123456', timeout_seconds=300).timeout_seconds == 300
    for invalid in (0, 601):
        with pytest.raises(ValueError):
            ReflectionCommand(session_id='saved-session-123456', timeout_seconds=invalid)


@pytest.mark.parametrize('citation_kind', ['episode', 'oracle_event', 'unavailable'])
def test_native_reflection_schema_enumerates_only_episode_citations(
        native_process_error_episode, citation_kind):
    from test_vast_reflection import Client, completion
    from vast_coevo.models import ModelSpec
    from vast_coevo.reflection.reflector import ReflectionInput, ServiceReflector
    from vast_coevo.skills.models import SkillPatch
    from web.backend.research_studio import _native_evidence

    model = ModelSpec(model='fixture', base_url='http://localhost/v1')
    evidence, view = _native_evidence(native_process_error_episode, 'session-citation-test', model)
    value = ReflectionInput(generation=0, role='service', library=empty_snapshot('service'),
                            evidence=(evidence,), interaction_views=(view,))
    reference = {'episode': evidence.episode_id, 'oracle_event': evidence.events[0].event_id,
                 'unavailable': 'not-a-supplied-training-episode'}[citation_kind]
    proposal = SkillPatch(action='NOOP', role='service', base_snapshot_hash=value.library.snapshot_hash,
                         evidence_ids=(reference,), rationale='Exact observed reference; never rewrite it.')
    response = completion(proposal.model_dump_json())
    client = Client(response)
    result = ServiceReflector(client, model).reflect(value)
    assert result.calls == len(client.requests) == 1
    request = client.requests[0]
    payload = json.loads(request['messages'][-1]['content'])
    assert payload['allowed_evidence_ids'] == [evidence.episode_id]
    citation_schema = request['tools'][0]['function']['parameters']['properties']['evidence_ids']
    assert citation_schema['items']['enum'] == [evidence.episode_id]
    assert 'Oracle event IDs' in citation_schema['description']
    assert result.call.response == response
    if citation_kind == 'episode':
        assert result.patch == proposal and result.error is None
    else:
        assert result.patch is None
        expected = 'Oracle event IDs were supplied instead' if citation_kind == 'oracle_event' else 'unavailable or heldout'
        assert expected in result.error
        # The event reference is never silently replaced by its parent Episode.
        raw = json.loads(result.call.response['choices'][0]['message']['tool_calls'][0]['function']['arguments'])
        assert raw['evidence_ids'] == [reference]
