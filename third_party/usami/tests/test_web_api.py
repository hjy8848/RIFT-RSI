"""HTTP integration tests use real worlds, engine, reference rules and independent oracle."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from usami import generate_world, replay_trace, evaluate_trace
from usami.events import Trace
from web.backend.app import COOKIE, create_app


@pytest.fixture
def client():
    with TestClient(create_app()) as client:
        response = client.post("/api/sessions", json={"seed": 42})
        assert response.status_code == 200
        yield client


def world(client):
    response = client.get("/api/world")
    assert response.status_code == 200
    return response.json()


def execute(client, name="change_plan", org="org_001", actor="user_001", resource="sub_001", mode="reference", **params):
    response = client.post("/api/operations", json={"actor_id": actor, "mode": mode,
        "expected_state_hash": world(client)["state_hash"],
        "operation": {"name": name, "org_id": org, "resource_id": resource, "params": params}})
    assert response.status_code == 200, response.text
    return response.json()


def test_real_default_world_and_cookie(client):
    data = world(client)
    expected = generate_world(42)
    assert data["world"] == expected.model_dump(mode="json")
    assert data["initial_state_hash"] == expected.state_hash
    assert data["summary"] == {"organizations": 3, "users": 8, "active_subscriptions": 1, "invoices": 12}
    assert COOKIE in client.cookies
    response = client.post("/api/sessions", json={"seed": 42})
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "SameSite=strict" in response.headers["set-cookie"]


def test_two_browser_sessions_are_isolated():
    app = create_app()
    with TestClient(app) as a, TestClient(app) as b:
        a.post("/api/sessions", json={"seed": 42})
        b.post("/api/sessions", json={"seed": 42})
        assert a.cookies[COOKIE] != b.cookies[COOKIE]
        initial = world(b)
        execute(a, plan_id="team")
        assert world(a)["world"]["subscriptions"]["sub_001"]["plan_id"] == "team"
        assert world(b) == initial
        a.post("/api/world/generate", json={"seed": 71})
        assert world(b) == initial


def test_graph_is_exactly_the_actual_world(client):
    data = world(client)["world"]
    graph = client.get("/api/world/graph").json()
    expected = set(data["users"]) | set(data["organizations"]) | set(data["subscriptions"]) | set(data["invoices"])
    assert {n["id"] for n in graph["nodes"]} == expected
    membership_edges = {e["id"]: e for e in graph["edges"] if e["kind"] == "membership"}
    assert set(membership_edges) == set(data["memberships"])
    for mid, member in data["memberships"].items():
        edge = membership_edges[mid]
        assert edge["source"] == member["user_id"]
        assert edge["target"] == member["org_id"]
        assert edge["entity"] == member
    for edge in graph["edges"]:
        assert edge["source"] in expected and edge["target"] in expected
    focused = client.get("/api/world/graph?org_id=org_001").json()
    assert len([n for n in focused["nodes"] if n["kind"] == "organization"]) == 1
    assert all(e["target"] == "org_001" for e in focused["edges"] if e["kind"] == "membership")


def test_organization_relations(client):
    data = world(client)
    for org in data["organizations"]:
        assert org["owner"] == data["world"]["users"][org["owner_id"]]
        assert org["subscription"] == data["world"]["subscriptions"][org["subscription_id"]]
        assert all(m["org_id"] == org["id"] for m in org["memberships"])
        assert org["active_members"] == sum(m["status"] == "active" for m in org["memberships"])


def test_legal_reference_changes_and_real_oracle(client):
    result = execute(client, plan_id="team")
    op = result["operation"]
    assert op["event"]["payload"]["result"]["success"] is True
    assert op["evaluation"]["operation_correct"] is True
    assert op["evaluation"]["valid_request"] is True
    assert op["changes"] == [{"path": "subscriptions.sub_001.plan_id", "before": "starter", "after": "team"}]
    assert result["snapshot"]["world"]["subscriptions"]["sub_001"]["plan_id"] == "team"
    assert op["event"]["event_id"] in op["evaluation"]["evidence_event_ids"]


def test_reference_denial_does_not_modify_state(client):
    initial = world(client)
    result = execute(client, "change_seats", seats=6)
    assert result["operation"]["event"]["payload"]["result"]["error_code"] == "SEAT_LIMIT_EXCEEDED"
    assert result["operation"]["changes"] == []
    assert result["operation"]["evaluation"]["operation_correct"] is True
    assert result["operation"]["evaluation"]["valid_request"] is False
    assert result["snapshot"]["state_hash"] == initial["state_hash"]


def test_member_cancellation_and_sandbox_violation(client):
    execute(client, "change_member_role", resource="membership_001_user_005", role="member")
    initial = world(client)
    result = execute(client, "cancel_renewal", actor="user_005")
    assert result["operation"]["event"]["payload"]["result"]["error_code"] == "PERMISSION_DENIED"
    assert result["snapshot"]["world"] == initial["world"]
    result = execute(client, "cancel_renewal", actor="user_005", mode="sandbox")
    assert result["operation"]["event"]["payload"]["result"]["success"] is True
    assert result["operation"]["reference_allowed"] is False
    assert result["operation"]["evaluation"]["operation_correct"] is False
    assert "PERMISSION_DENIED" in {v["code"] for v in result["operation"]["evaluation"]["violations"]}
    assert result["snapshot"]["world"]["subscriptions"]["sub_001"]["auto_renew"] is False


def test_sandbox_invalid_world_and_reset(client):
    original = world(client)
    result = execute(client, "change_seats", mode="sandbox", seats=8)
    assert result["operation"]["evaluation"]["operation_correct"] is False
    assert result["snapshot"]["invariants"][0]["code"] == "SEAT_LIMIT_EXCEEDED"
    result = execute(client, "cancel_renewal")
    assert result["operation"]["reference_allowed"] is None
    assert result["operation"]["evaluation"]["valid_request"] is None
    # Unknown request legality does not erase an already proven state invariant violation.
    assert result["operation"]["evaluation"]["operation_correct"] is False
    assert result["operation"]["event"]["payload"]["result"]["error_code"] == "INVALID_WORLD"
    # Scenario catalog is still generated from the legal initial world.
    assert client.get("/api/scenarios").status_code == 200
    restored = client.post("/api/world/reset").json()
    assert restored["world"] == original["world"]
    assert restored["operations_used"] == 0 and len(restored["events"]) == 1


@pytest.mark.parametrize("category", ["multi_step_plan", "multi_entity", "multi_step_refund"])
def test_real_scenario_continuous_state(client, category):
    initial = world(client)
    scenarios = client.get("/api/scenarios").json()["scenarios"]
    scenario = next(s for s in scenarios if s["category"] == category)
    response = client.post("/api/scenarios/run", json={"scenario_id": scenario["scenario_id"], "expected_state_hash": initial["state_hash"]})
    assert response.status_code == 200, response.text
    result = response.json()
    assert len(result["steps"]) == len(scenario["steps"])
    assert result["snapshot"]["state_hash"] == scenario["steps"][-1]["expected_state_hash"]
    assert result["snapshot"]["evaluation"]["status"] == "pass"
    for previous, current in zip(result["steps"], result["steps"][1:]):
        assert previous["event"]["after"] == current["event"]["before"]
    # Scenario does not finalize/lock the live engine; a subsequent request works.
    execute(client, "cancel_renewal")
    assert client.post("/api/world/reset").json()["world"] == initial["world"]


def test_scenario_replaces_history_without_state_mix(client):
    execute(client, plan_id="team")
    catalog = client.get("/api/scenarios").json()
    scenario = next(s for s in catalog["scenarios"] if s["category"] == "multi_step_plan")
    result = client.post("/api/scenarios/run", json={"scenario_id": scenario["scenario_id"], "expected_state_hash": world(client)["state_hash"]}).json()
    assert result["steps"][0]["event"]["before"] == generate_world(42).model_dump(mode="json")
    assert result["snapshot"]["operations_used"] == len(scenario["steps"])


def test_generate_determinism_and_clean_session(client):
    initial = world(client)
    execute(client, plan_id="team")
    result = client.post("/api/world/generate", json={"seed": 42}).json()
    assert result["world"] == initial["world"] and result["state_hash"] == initial["state_hash"]
    assert result["events"] == initial["events"]
    other = client.post("/api/world/generate", json={"seed": 43, "config": {"user_count": 10, "organization_count": 4}}).json()
    assert other["world"]["world_id"] != initial["world"]["world_id"]
    assert other["summary"]["organizations"] == 4 and other["summary"]["users"] == 10
    assert other["operations_used"] == 0


def test_export_and_true_replay_leave_live_session_editable(client):
    execute(client, plan_id="team")
    initial = world(client)
    exported = client.get("/api/export/trace")
    assert "attachment" in exported.headers["content-disposition"]
    trace = Trace.model_validate(exported.json())
    assert replay_trace(trace).state.state_hash == initial["state_hash"]
    assert evaluate_trace(trace).status == "pass"
    replay = client.post("/api/replay", json={}).json()
    assert replay["replayed"] and replay["matches_current_state"]
    assert replay["final_state_hash"] == initial["state_hash"]
    assert world(client) == initial  # No replay-only events contaminate live history.
    execute(client, "cancel_renewal")
    assert client.get("/api/export/world").json() == world(client)["world"]


def test_sandbox_trace_replays_and_keeps_failure_score(client):
    execute(client, "change_seats", mode="sandbox", seats=8)
    replay = client.post("/api/replay", json={}).json()
    assert replay["replayed"] and replay["evaluation"]["status"] == "fail"


def test_no_browser_authority_or_oracle_injection(client):
    initial = world(client)
    command = {"actor_id": "user_001", "mode": "reference", "expected_state_hash": initial["state_hash"],
               "operation": {"name": "cancel_renewal", "org_id": "org_001", "resource_id": "sub_001", "params": {}},
               "oracle": {"operation_correct": True}, "request_context": {"authorized_operations": []}}
    assert client.post("/api/operations", json=command).status_code == 422
    assert world(client) == initial
    assert client.post("/api/scenarios/run", json={"scenario_id": "fake", "expected_state_hash": initial["state_hash"], "steps": []}).status_code == 422


def test_stale_operation_returns_conflict_without_new_events(client):
    initial = world(client)
    execute(client, plan_id="team")
    latest = world(client)
    command = {"actor_id": "user_001", "expected_state_hash": initial["state_hash"],
               "operation": {"name": "cancel_renewal", "org_id": "org_001", "resource_id": "sub_001", "params": {}}}
    result = client.post("/api/operations", json=command)
    assert result.status_code == 409 and result.json()["detail"]["code"] == "STALE_WORLD"
    assert world(client) == latest


def test_clear_http_errors_and_actual_event_detail(client):
    assert client.get("/api/world/graph?org_id=missing").status_code == 404
    assert client.get("/api/events/missing").status_code == 404
    assert client.post("/api/scenarios/run", json={"scenario_id": "missing", "expected_state_hash": world(client)["state_hash"]}).status_code == 404
    assert client.post("/api/world/generate", json={"seed": True}).status_code == 422
    assert client.post("/api/world/generate", json={"seed": 42, "config": {"organization_count": 1}}).status_code == 422
    result = execute(client, plan_id="team")
    event = result["operation"]["event"]
    assert client.get("/api/events/" + event["event_id"]).json() == event
    with TestClient(create_app()) as unauthenticated:
        assert unauthenticated.get("/api/world").status_code == 401


def test_operation_budget_is_enforced_before_registering_new_request(client):
    app = client.app
    session = app.state.sessions[client.cookies[COOKIE]]
    session.engine.operation_budget = 1
    session.engine.trace.operation_budget = 1
    execute(client, plan_id="team")
    previous = world(client)
    command = {"actor_id": "user_001", "expected_state_hash": previous["state_hash"],
               "operation": {"name": "cancel_renewal", "org_id": "org_001", "resource_id": "sub_001", "params": {}}}
    assert client.post("/api/operations", json=command).status_code == 409
    assert world(client) == previous
