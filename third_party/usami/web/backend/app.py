"""Trusted local research harness, backed entirely by the existing Usami engine."""
from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field

from usami import WorldEngine, evaluate_trace, generate_scenarios, generate_world, make_request, replay_trace
from usami.evaluation.scenario import Scenario, validate_scenario, derive_scenario
from usami.protocol import Operation, RequestContext
from usami.world.generator import GeneratorConfig
from usami.world.models import Model, WorldState
from usami.world.rules import ReferenceRules, check_invariants

COOKIE = "usami_explorer_session"
TTL_SECONDS = 6 * 60 * 60  # Transport/session lifetime only; never a business clock.

REASONS = {
    "PERMISSION_DENIED": "当前用户的组织角色无权执行此操作。普通成员不能管理订阅、账单或成员角色。",
    "CROSS_ORG_ACCESS": "执行主体不是目标组织的有效成员，不能跨组织访问资源。",
    "CROSS_ORG_RESOURCE": "目标资源属于另一个组织，与请求组织不一致。",
    "SEAT_LIMIT_EXCEEDED": "席位数量超过当前套餐或目标套餐的上限。",
    "SEATS_BELOW_MEMBERS": "席位数量不能少于当前有效成员数量，且至少需要一个席位。",
    "SUBSCRIPTION_NOT_ACTIVE": "当前订阅已经过期，不能变更套餐、席位或续费。",
    "OWNER_IMMUTABLE": "当前业务模型 不支持转移、降级 Owner 或创建第二个 Owner。",
    "MEMBERSHIP_NOT_ACTIVE": "只能修改有效成员的角色。",
    "INVOICE_NOT_PAID": "账单尚未支付，不能退款。",
    "REFUND_WINDOW_EXPIRED": "该账单已超过当前规则规定的退款期限。",
    "DUPLICATE_REFUND": "该账单已经退款，不能重复退款。",
    "REFUND_AMOUNT_MISMATCH": "退款金额必须与账单的全部可退金额一致。",
    "REFUND_METHOD_MISMATCH": "退款必须返回账单的原始支付渠道。",
    "UNAUTHORIZED_OPERATION": "实际执行主体、操作或参数不在明确授权范围内。",
    "INVALID_WORLD": "当前世界已有业务不变量违规。后续业务合法性无法确定，请查看证据或重置世界。",
    "INVALID_PLAN": "目标套餐不在该世界的规则配置中。",
    "UNKNOWN_ACTOR": "执行主体不存在。",
    "ORG_NOT_FOUND": "目标组织不存在。",
    "RESOURCE_NOT_FOUND": "目标资源不存在。",
    "FALSE_REJECTION": "明确授权且可完成的请求被错误拒绝。",
}


class GenerateCommand(Model):
    seed: int = Field(default=42, strict=True)
    config: GeneratorConfig = Field(default_factory=GeneratorConfig)


class ExecuteCommand(Model):
    actor_id: str
    operation: Operation
    mode: Literal["reference", "sandbox"] = "reference"
    expected_state_hash: str


class RunScenarioCommand(Model):
    scenario_id: str
    mode: Literal["reference", "sandbox"] = "reference"
    expected_state_hash: str


@dataclass
class Session:
    config: GeneratorConfig
    initial: WorldState
    engine: WorldEngine
    requests: Dict[str, RequestContext] = field(default_factory=dict)
    scenarios: Optional[Dict[str, Scenario]] = None
    lock: Any = field(default_factory=threading.RLock)
    touched: float = field(default_factory=time.monotonic)
    exploration_source: Optional[Dict[str, str]] = None


def changes(before: Any, after: Any, path: str = ""):
    """Generic JSON diff. No business rules or success labels are inferred here."""
    if before == after:
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(set(before) | set(after)):
            result.extend(changes(before.get(key), after.get(key), f"{path}.{key}" if path else key))
        return result
    return [{"path": path, "before": before, "after": after}]


def organization_views(world: WorldState):
    result = []
    for org in world.organizations.values():
        sub = world.subscriptions[org.subscription_id]
        memberships = [m for m in world.memberships.values() if m.org_id == org.id]
        plan = world.rules.plans.get(sub.plan_id)
        result.append({**org.model_dump(mode="json"), "owner": world.users[org.owner_id].model_dump(mode="json"),
                       "subscription": sub.model_dump(mode="json"), "plan": plan.model_dump(mode="json") if plan else None,
                       "active_members": sum(m.status == "active" for m in memberships),
                       "memberships": [m.model_dump(mode="json") for m in memberships],
                       "invoice_count": sum(i.org_id == org.id for i in world.invoices.values())})
    return result


def event_view(event):
    return {"event_id": event.event_id, "kind": event.kind, "payload": event.payload,
            "before_hash": event.before.state_hash if event.before else None,
            "after_hash": event.after.state_hash if event.after else None}


def snapshot(session: Session):
    world = session.engine.state
    trace = session.engine.trace
    report = evaluate_trace(trace, trusted_requests=session.requests)
    return {"world": world.model_dump(mode="json"), "config": session.config.model_dump(mode="json"),
            "initial_state_hash": session.initial.state_hash, "state_hash": world.state_hash,
            "exploration_source": session.exploration_source,
            "trace_initial_state_hash": trace.initial_state_hash,
            "summary": {"organizations": len(world.organizations), "users": len(world.users),
                        "active_subscriptions": sum(s.status == "active" for s in world.subscriptions.values()),
                        "invoices": len(world.invoices)},
            "organizations": organization_views(world), "evaluation": report.model_dump(mode="json"),
            "events": [event_view(e) for e in trace.events],
            "operations_used": sum(e.kind == "operation" for e in trace.events),
            "operation_budget": trace.operation_budget,
            "invariants": [v.model_dump(mode="json") for v in check_invariants(world)]}


def assert_current(session: Session, expected: str):
    if expected != session.engine.state.state_hash:
        raise HTTPException(409, detail={"code": "STALE_WORLD", "message": "世界已改变，请刷新后再执行。"})


def scenario_catalog(session: Session):
    if session.scenarios is None:
        items = generate_scenarios(session.initial)
        if session.config.workflow:
            from usami.benchmark.goals_v2 import build_task
            task = build_task(session.initial, session.config.workflow)
            paths = [w.operations for w in task.witnesses]
            if task.policy_certificate:
                paths = [[task.policy_certificate.operation]]
            for index, path in enumerate(paths):
                items.append(derive_scenario(session.initial, f"v2_{task.family}_{index}",
                    [(task.identity, op) for op in path], "business_witness"))
        session.scenarios = {s.scenario_id: s for s in items}
    return session.scenarios


def operation_view(event, report, decision):
    evaluation = next(r for r in report.requests if r.request_id == event.payload["request_id"])
    code = event.payload["result"]["error_code"] or decision.error_code
    reasons = REASONS
    if event.before.rules.version == "saas-rules-0.2.0":
        from usami.world.rules_v2 import public_policy_catalog
        reasons = {**REASONS, **public_policy_catalog()["policy"]}
    return {"event": event.model_dump(mode="json"),
            "changes": changes(event.before.model_dump(mode="json"), event.after.model_dump(mode="json")),
            "evaluation": evaluation.model_dump(mode="json"),
            "reference_allowed": None if decision.error_code in {"INVALID_WORLD", "UNSUPPORTED_RULES_VERSION"} else decision.allowed,
            "reference_error_code": decision.error_code,
            "reason": reasons.get(code, code or "操作满足当前业务规则，已按明确请求完成。")}


def create_app() -> FastAPI:
    app = FastAPI(title="Usami World Explorer", version="0.2.0")
    app.state.sessions = {}
    app.state.store_lock = threading.RLock()

    def current(request: Request) -> Session:
        key = request.cookies.get(COOKIE)
        with app.state.store_lock:
            session = app.state.sessions.get(key)
            if session is None or time.monotonic() - session.touched > TTL_SECONDS:
                app.state.sessions.pop(key, None)
                raise HTTPException(401, detail={"code": "SESSION_EXPIRED", "message": "请创建新的本地世界会话。"})
            session.touched = time.monotonic()
            return session

    @app.exception_handler(ValueError)
    def invalid_input(request, exc):
        return JSONResponse(status_code=422, content={"detail": {"code": "INVALID_INPUT", "message": str(exc)}})

    @app.get("/api/health")
    def health():
        return {"ok": True, "service": "Usami World Explorer"}

    @app.post("/api/sessions")
    def new_session(response: Response, request: Request, command: GenerateCommand = GenerateCommand()):
        world = generate_world(command.seed, command.config)
        session = Session(config=command.config.model_copy(deep=True), initial=world.clone(), engine=WorldEngine(world))
        key = secrets.token_urlsafe(32)
        with app.state.store_lock:
            expired = [k for k, v in app.state.sessions.items() if time.monotonic() - v.touched > TTL_SECONDS]
            for k in expired:
                del app.state.sessions[k]
            replacing = request.cookies.get(COOKIE)
            if len(app.state.sessions) >= 64 and replacing not in app.state.sessions:
                raise HTTPException(503, detail={"code": "SESSION_CAPACITY", "message": "本地会话已满，请重启服务或等待过期。"})
            app.state.sessions.pop(replacing, None)
            app.state.sessions[key] = session
        response.set_cookie(COOKIE, key, httponly=True, samesite="strict", max_age=TTL_SECONDS)
        return snapshot(session)

    @app.get("/api/world")
    def get_world(session: Session = Depends(current)):
        with session.lock:
            return snapshot(session)

    @app.post("/api/world/generate")
    def generate(command: GenerateCommand, session: Session = Depends(current)):
        world = generate_world(command.seed, command.config)
        with session.lock:
            session.config = command.config.model_copy(deep=True)
            session.initial, session.engine = world.clone(), WorldEngine(world)
            session.requests, session.scenarios, session.exploration_source = {}, None, None
            return snapshot(session)

    @app.post("/api/world/reset")
    def reset(session: Session = Depends(current)):
        with session.lock:
            # A scenario may start a new engine; reset always means the generated world.
            session.engine = WorldEngine(session.initial)
            session.requests = {}
            return snapshot(session)

    @app.get("/api/world/graph")
    def graph(org_id: Optional[str] = None, session: Session = Depends(current)):
        with session.lock:
            world = session.engine.state
        if org_id is not None and org_id not in world.organizations:
            raise HTTPException(404, detail={"code": "ORG_NOT_FOUND", "message": "组织不存在。"})
        orgs = [o for o in world.organizations.values() if org_id is None or o.id == org_id]
        org_ids = {o.id for o in orgs}
        memberships = [m for m in world.memberships.values() if m.org_id in org_ids]
        users = [u for u in world.users.values() if org_id is None or any(m.user_id == u.id for m in memberships)]
        nodes, edges = [], []
        for user in users:
            relations = [m.model_dump(mode="json") for m in world.memberships.values() if m.user_id == user.id]
            nodes.append({"id": user.id, "kind": "user", "label": user.name, "entity": user.model_dump(mode="json"), "memberships": relations})
        for org in orgs:
            nodes.append({"id": org.id, "kind": "organization", "label": org.name, "entity": org.model_dump(mode="json")})
        for sub in world.subscriptions.values():
            if sub.org_id in org_ids:
                nodes.append({"id": sub.id, "kind": "subscription", "label": sub.plan_id, "entity": sub.model_dump(mode="json")})
                edges.append({"id": "subscription_" + sub.id, "source": sub.org_id, "target": sub.id, "label": "subscription", "kind": "subscription"})
        for inv in world.invoices.values():
            if inv.org_id in org_ids:
                nodes.append({"id": inv.id, "kind": "invoice", "label": inv.id, "entity": inv.model_dump(mode="json")})
                edges.append({"id": "invoice_" + inv.id, "source": inv.org_id, "target": inv.id, "label": inv.status, "kind": "invoice"})
                if inv.subscription_id:
                    edges.append({"id": "funding_" + inv.id, "source": inv.id, "target": inv.subscription_id,
                                  "label": inv.purpose or "billing cycle", "kind": "funding"})
                for kind, records in ((("payment", inv.payments or []), ("refund", inv.refunds)) if world.rules.version == "saas-rules-0.2.0" else ()):
                    for record in records:
                        nodes.append({"id": record.id, "kind": kind, "label": record.id,
                                      "entity": {**record.model_dump(mode="json"), "org_id": inv.org_id, "invoice_id": inv.id}})
                        edges.append({"id": kind + "_" + record.id, "source": inv.id, "target": record.id,
                                      "label": kind, "kind": kind})
        for member in memberships:
            edges.append({"id": member.id, "source": member.user_id, "target": member.org_id,
                          "label": member.role.replace("_", " ") + (" · " + member.status if member.status != "active" else ""),
                          "kind": "membership", "entity": member.model_dump(mode="json")})
        return {"nodes": nodes, "edges": edges, "state_hash": world.state_hash}

    @app.post("/api/operations")
    def execute(command: ExecuteCommand, session: Session = Depends(current)):
        with session.lock:
            assert_current(session, command.expected_state_hash)
            if session.engine.trace.events[-1].kind == "final":
                raise HTTPException(409, detail={"code": "FINALIZED_TRACE", "message": "请先重置世界。"})
            if sum(e.kind == "operation" for e in session.engine.trace.events) >= session.engine.operation_budget:
                raise HTTPException(409, detail={"code": "BUDGET_EXHAUSTED", "message": "操作预算已用完，请重置世界。"})
            # Only the server creates authorization; incoming Oracle/context fields are forbidden.
            req = make_request("web_" + secrets.token_hex(12), command.actor_id, command.operation)
            decision = ReferenceRules().decide(session.engine.state, command.actor_id, command.operation)
            session.engine.register_request(req)
            session.requests[req.request_id] = req.model_copy(deep=True)
            event = session.engine.execute(command.actor_id, command.operation, req.request_id, command.mode)
            report = evaluate_trace(session.engine.trace, trusted_requests=session.requests)
            result = operation_view(event, report, decision)
            session.engine.trace.append("oracle", {"report": report.model_dump(mode="json")})
            return {"snapshot": snapshot(session), "operation": result}

    @app.get("/api/scenarios")
    def scenarios(session: Session = Depends(current)):
        with session.lock:
            return {"initial_state_hash": session.initial.state_hash,
                    "scenarios": [s.model_dump(mode="json") for s in scenario_catalog(session).values()]}

    @app.post("/api/scenarios/run")
    def run_scenario(command: RunScenarioCommand, session: Session = Depends(current)):
        with session.lock:
            assert_current(session, command.expected_state_hash)
            scenario = scenario_catalog(session).get(command.scenario_id)
            if scenario is None:
                raise HTTPException(404, detail={"code": "SCENARIO_NOT_FOUND", "message": "当前世界没有该场景。"})
            validate_scenario(session.initial, scenario)
            engine, authority, results = WorldEngine(session.initial, scenario.operation_budget), {}, []
            for step in scenario.steps:
                req = step.request.model_copy(deep=True)
                decision = ReferenceRules().decide(engine.state, req.requester_id, req.operation)
                engine.register_request(req)
                authority[req.request_id] = req
                event = engine.execute(req.requester_id, req.operation, req.request_id, command.mode)
                report = evaluate_trace(engine.trace, trusted_requests=authority)
                results.append(operation_view(event, report, decision))
            engine.trace.append("oracle", {"report": report.model_dump(mode="json")})
            session.engine, session.requests = engine, authority
            return {"snapshot": snapshot(session), "steps": results, "scenario_id": scenario.scenario_id}

    @app.get("/api/events/{event_id}")
    def event_detail(event_id: str, session: Session = Depends(current)):
        with session.lock:
            event = next((e for e in session.engine.trace.events if e.event_id == event_id), None)
            if event is None:
                raise HTTPException(404, detail={"code": "EVENT_NOT_FOUND", "message": "事件不存在。"})
            return event.model_dump(mode="json")

    @app.get("/api/export/world")
    def export_world(session: Session = Depends(current)):
        with session.lock:
            return JSONResponse(session.engine.state.model_dump(mode="json"), headers={"Content-Disposition": 'attachment; filename="usami-world.json"'})

    @app.get("/api/export/trace")
    def export_trace(session: Session = Depends(current)):
        with session.lock:
            trace = session.engine.clone().finish()
            return JSONResponse(trace.model_dump(mode="json"), headers={"Content-Disposition": 'attachment; filename="usami-trace.json"'})

    @app.post("/api/replay")
    def replay(session: Session = Depends(current)):
        with session.lock:
            trace = session.engine.clone().finish()
            replayed = replay_trace(trace)  # True deterministic execution, not animation.
            report = evaluate_trace(replayed.trace, trusted_requests=session.requests)
            return {"replayed": True, "event_count": len(trace.events), "final_state_hash": replayed.state.state_hash,
                    "matches_current_state": replayed.state == session.engine.state,
                    "evaluation": report.model_dump(mode="json")}

    from web.backend.release import install_release_routes
    install_release_routes(app, current, snapshot, assert_current)
    from web.backend.domains import install_domain_routes
    install_domain_routes(app)
    from web.backend.human_customer import install_human_customer_routes
    human_manager = install_human_customer_routes(app)
    from web.backend.research_studio import install_research_studio_routes
    install_research_studio_routes(app, human_manager)

    # Optional single-origin preview after `npm run build`; API routes take precedence.
    dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="explorer")
    return app


app = create_app()
