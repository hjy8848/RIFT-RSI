"""Human Customer API runs the native Episode Runner and persists manual evidence."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from usami_v2.commerce.models import Operation, case_id, quote_id
from usami_v2.runtime.agents import ServiceAction
from usami_v2.runtime.runner import replay_episode
from vast_coevo.models import ModelSpec
from vast_coevo.reflection.reflector import ReflectionInput
from vast_coevo.reflection.scope import DomainPolicy
from vast_coevo.skills.library import empty_snapshot
from web.backend.research_studio import _native_evidence
from web.backend.app import create_app


class ScriptedRefundService:
    """Explicit test-only Service substitute; business effects use the native Engine."""
    uses_model = False
    last_call = None

    def __init__(self, scenario, *, commit_without_consent=False):
        self.world = scenario.world
        self.aliases = scenario.aliases
        self.index = 0
        self.merchant = next(iter(self.world.merchants))
        self.line = self.aliases["damaged"]
        self.case = case_id(self.line, (0,))
        self.quote = quote_id(self.case, 1)
        self.evidence = next(row.id for row in self.world.evidence.values() if row.line_id == self.line)
        self.commit_without_consent = commit_without_consent

    def decide(self, prompt):
        actions = [
            ServiceAction(kind="operation", operation=Operation(
                name="open_remedy_case", scope_id=self.merchant, resource_id=self.line,
                params={"quantity": 1, "issue": "damaged"})),
            ServiceAction(kind="operation", operation=Operation(
                name="verify_submitted_evidence", scope_id=self.merchant, resource_id=self.case,
                params={"evidence_id": self.evidence})),
            ServiceAction(kind="operation", operation=Operation(
                name="prepare_remedy_quote", scope_id=self.merchant, resource_id=self.case,
                params={"remedy": "refund", "target_sku_id": None,
                        "warehouse_id": None, "pickup_slot": None})),
            ServiceAction(kind="reply", message="Please review this exact refund offer.",
                          present_quote_id=self.quote, present_for="commit_accepted_quote"),
            ServiceAction(kind="operation", operation=Operation(
                name="commit_accepted_quote", scope_id=self.merchant, resource_id=self.quote, params={})),
            ServiceAction(kind="reply", message=self.line + ": refund status is released.",
                          subject_ids=(self.line,)),
        ]
        action = actions[self.index]
        self.index += 1
        return action


class IdleService:
    uses_model = False
    last_call = None

    def decide(self, prompt):
        return ServiceAction(kind="reply", message="I can help with this order.")


def client_with_service(tmp_path, factory):
    app = create_app()
    manager = app.state.human_sessions
    manager.output_dir = tmp_path
    manager.service_factory = factory(manager.scenarios)
    return app, TestClient(app)


def headers():
    return {"origin": "http://127.0.0.1:5173", "content-type": "application/json"}


def wait_for(client, session_id, status, timeout=5):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        response = client.get(f"/api/human/sessions/{session_id}")
        assert response.status_code == 200, response.text
        data = response.json()
        if data["status"] == status:
            return data
        time.sleep(0.02)
    raise AssertionError(f"session did not reach {status}; last={data['status']}")


def start(client):
    response = client.post("/api/human/sessions",
                           json={"scenario_id": "refund-damaged-headphones"}, headers=headers())
    assert response.status_code == 200, response.text
    cookie = response.headers.get("set-cookie", "").lower()
    assert "httponly" in cookie and "samesite=strict" in cookie and "path=/api" in cookie
    # Research Studio and Human Playground intentionally share one owner cookie.
    assert client.get("/api/research/runs").status_code == 200
    return response.json()


def test_human_reply_and_explicit_offer_accept_flow_through_runtime_engine_oracle_and_replay(tmp_path):
    app, client = client_with_service(tmp_path,
        lambda scenarios: lambda: (ScriptedRefundService(scenarios["refund-damaged-headphones"]), None))
    with client:
        initial = start(client)
        session_id = initial["session_id"]
        pending = wait_for(client, session_id, "awaiting_customer")
        assert pending["pending_turn"]["offer"]["remedy"] == "refund"
        assert pending["timeline"][-1]["kind"] == "service"
        assert pending["timeline"][-1]["offer"]["refund_cents"] == 20000

        turn = pending["pending_turn"]
        submitted = client.post(f"/api/human/sessions/{session_id}/turn", json={
            **turn, "intent": "accept_offer", "message": "spoofed quote id ignored"}, headers=headers())
        assert submitted.status_code == 200, submitted.text
        stale = client.post(f"/api/human/sessions/{session_id}/turn", json={
            **turn, "intent": "accept_offer", "message": ""}, headers=headers())
        assert stale.status_code == 409

        second_wait = wait_for(client, session_id, "awaiting_customer")
        assert second_wait["tool_calls"] == 4
        assert second_wait["pending_turn"]["offer"] is None
        finish_turn = second_wait["pending_turn"]
        response = client.post(f"/api/human/sessions/{session_id}/turn", json={
            **finish_turn, "intent": "finish", "message": "退款已处理，谢谢。"}, headers=headers())
        assert response.status_code == 200
        completed = wait_for(client, session_id, "completed")
        report = client.get(f"/api/human/sessions/{session_id}/report")
        assert report.status_code == 200, report.text
        data = report.json()
        episode = data["episode"]
        assert episode["profile"]["return_pickup_policy"] == "accept_presented_quote"
        assert episode["status"] == "customer_finished"
        assert episode["tool_executions"][-1]["operation"]["name"] == "commit_accepted_quote"
        assert episode["final_world"]["settlements"]
        assert data["outcome"]["business_goal_success"] is True
        assert data["outcome"]["safety_violations"]["status"] == "PASS"
        assert data["replay"]["verified"] is True
        assert data["customer_source"] == "human"
        assert data["study_type"] == "qualitative_manual_pilot"
        assert data["benchmark_eligible"] is False and data["training_evidence_eligible"] is False
        assert replay_episode(app.state.human_sessions.sessions[session_id].record).verified
        evidence, interaction = _native_evidence(
            app.state.human_sessions.sessions[session_id].record, session_id,
            ModelSpec(model="offline-test", base_url="http://127.0.0.1/v1"),
            service_snapshot=empty_snapshot("service"))
        reflected = ReflectionInput(
            generation=0, role="service", library=empty_snapshot("service"),
            evidence=(evidence,), interaction_views=(interaction,),
            allowed_tools=evidence.config.tools,
            domain_policy=DomainPolicy(add_domains=("aftersales",)),
            public_policy="manual evidence only", boundaries="no policy changes")
        assert reflected.evidence[0].replay.verified
        assert interaction.source == "verified_native"
        assert interaction.customer_requests[0].message == "Please refund my damaged headphones."
        assert interaction.oracle_events == tuple(row.model_dump(mode="json") for row in evidence.events)
        root = tmp_path / session_id
        assert json.loads((root / "manifest.json").read_text())["customer_source"] == "human"
        assert (root / "episode.json").is_file()


def test_plain_text_cannot_create_quote_consent_and_cancel_is_not_service_failure(tmp_path):
    app, client = client_with_service(tmp_path,
        lambda scenarios: lambda: (ScriptedRefundService(scenarios["refund-damaged-headphones"]), None))
    with client:
        session = start(client)
        pending = wait_for(client, session["session_id"], "awaiting_customer")
        turn = pending["pending_turn"]
        response = client.post(f"/api/human/sessions/{session['session_id']}/turn", json={
            **turn, "intent": "message", "message": "I accept this refund."}, headers=headers())
        assert response.status_code == 200
        second_wait = wait_for(client, session["session_id"], "awaiting_customer")
        detail = client.get(f"/api/human/sessions/{session['session_id']}").json()
        assert detail["status"] == "awaiting_customer"
        assert not any("accept_quote_id" in item for item in detail["timeline"] if item["kind"] == "customer")
        cancellation = client.post(f"/api/human/sessions/{session['session_id']}/cancel", json={}, headers=headers())
        assert cancellation.json()["status"] in {"cancelling", "cancelled"}
        cancelled = wait_for(client, session["session_id"], "cancelled")
        report = client.get(f"/api/human/sessions/{session['session_id']}/report")
        assert report.status_code == 200, report.text
        outcome = report.json()["outcome"]
        assert outcome["business_goal_success"] is None
        assert outcome["strict_task_success"] is None
        assert outcome["interaction_reliability"]["status"] == "UNRESOLVED"
        assert report.json()["replay"]["verified"] is True


def test_explicit_rejection_creates_no_consent_and_foreign_origin_is_blocked(tmp_path):
    app, client = client_with_service(tmp_path,
        lambda scenarios: lambda: (ScriptedRefundService(scenarios["refund-damaged-headphones"]), None))
    with client:
        scenario = client.get("/api/human/scenarios")
        assert scenario.status_code == 200
        started = start(client)
        pending = wait_for(client, started["session_id"], "awaiting_customer")
        other = TestClient(app)
        with other:
            foreign = other.get(f"/api/human/sessions/{started['session_id']}")
            assert foreign.status_code == 401
        rejected = client.post(f"/api/human/sessions/{started['session_id']}/turn", json={
            **pending["pending_turn"], "intent": "reject_offer", "message": ""}, headers=headers())
        assert rejected.status_code == 200
        wait_for(client, started["session_id"], "awaiting_customer")
        incomplete = client.get(f"/api/human/sessions/{started['session_id']}/report")
        assert incomplete.status_code == 409
        blocked = client.post("/api/human/sessions", json={
            "scenario_id": "refund-damaged-headphones"}, headers={"origin": "http://evil.example"})
        assert blocked.status_code == 403
        client.post(f"/api/human/sessions/{started['session_id']}/cancel", json={}, headers=headers())
        wait_for(client, started["session_id"], "cancelled")
        report = client.get(f"/api/human/sessions/{started['session_id']}/report")
        assert report.status_code == 200
        episode = report.json()["episode"]
        assert episode["final_world"]["consents"] == {}
        assert episode["final_world"]["settlements"] == {}


def test_model_initialization_failure_is_reported_without_fallback(tmp_path):
    def unavailable():
        raise RuntimeError("credentials not configured")
    app, client = client_with_service(tmp_path, lambda scenarios: unavailable)
    with client:
        response = client.post("/api/human/sessions", json={
            "scenario_id": "refund-damaged-headphones"}, headers=headers())
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "SERVICE_NOT_CONFIGURED"
