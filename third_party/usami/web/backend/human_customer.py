"""Local, qualitative Human Customer Playground over the native commerce Runner."""
from __future__ import annotations

import ast
import json
import os
import secrets
import threading
import time
from datetime import timedelta
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from vast_coevo.models import ModelSpec
from vast_coevo.schema import digest
from usami.service.client import OpenAICompatibleClient
from usami_v2.runtime.agents import CustomerPrompt, OpenAIServiceAgent, ServicePrompt
from usami_v2.runtime.customer import (
    CustomerDecision, CustomerProfile, CustomerProfileV2, HumanCustomerProfileV2, OfferPresentation,
)
from usami_v2.runtime.runner import (
    EpisodeBudget, EpisodeCancelled, EpisodeRecord, replay_episode, run_episode, save_episode,
)
from usami_v2.scenarios.builder import build_world, verify_build
from usami_v2.scenarios.certification import verify_world_constraints
from usami_v2.scenarios.schema import (
    BuildBudget, IntegerRange, LineVariable, OrderVariable, ScenarioSpec, SKUVariable,
    WarehouseVariable,
)
from usami_v2.scenarios.task_certification import (
    CustomerScenario, ProofBudget, certify_task, verify_task_certificate,
)
from usami_v2.scenarios.tasks import GroundedRemedyGoal, TaskSpecV2
from usami_v2.commerce.models import ExternalEvent, case_id

HUMAN_COOKIE = "usami_human_client"
HUMAN_TTL_SECONDS = 30 * 60
OWNER_COOKIE_TTL_SECONDS = 30 * 24 * 60 * 60
MAX_ACTIVE_EPISODES = 2
MAX_RETAINED_SESSIONS = 32
MAX_HUMAN_MESSAGE = 2000
MAX_SERVICE_MODEL_CALLS = 32


class HumanSessionCreate(BaseModel):
    scenario_id: str = Field(min_length=1, max_length=100)
    research_run_id: Optional[str] = Field(default=None, min_length=1, max_length=100)
    service_skill_snapshot_hash: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    comparison_arm: Optional[Literal["S0", "S1"]] = None


class HumanTurn(BaseModel):
    turn_id: str = Field(min_length=16, max_length=128)
    session_version: int = Field(ge=0, strict=True)
    intent: Literal["message", "accept_offer", "reject_offer", "finish"]
    message: str = Field(default="", max_length=MAX_HUMAN_MESSAGE)


@dataclass(frozen=True)
class CertifiedScenario:
    scenario_id: str
    title: str
    description: str
    request: str
    spec: ScenarioSpec
    build: Any
    world_certificate: Any
    task: TaskSpecV2
    profile: CustomerProfileV2
    proof_customer: CustomerScenario
    task_certificate: Any
    aliases: Dict[str, str]

    @property
    def world(self):
        return self.build.world


def _return_events(scenario: CertifiedScenario):
    """Freeze native return receipts before execution; Runner waits for a scheduled return."""
    days = scenario.proof_customer.return_received_after_days
    if days is None:
        return ()
    events = []
    for goal in scenario.task.remedy_goals:
        line = scenario.world.lines[goal.line_id]
        if not scenario.world.skus[line.sku_id].return_required:
            continue
        target = case_id(line.id, tuple(range(goal.quantity)))
        events.append(ExternalEvent(
            id="return_" + digest([scenario.proof_customer.record_hash, target])[:20],
            kind="return_received", target_id=target,
            at=scenario.world.clock + timedelta(days=days)))
    return tuple(events)


def _fixed(value: int) -> IntegerRange:
    return IntegerRange(low=value, high=value)


def certified_scenarios() -> Dict[str, CertifiedScenario]:
    """Build and independently certify the first production-facing scenario."""
    request = "Please refund my damaged headphones."
    spec = ScenarioSpec(
        scenario_id="human-refund-v1", provenance_kind="offline_fixture",
        customer_aliases=("customer",),
        skus=(SKUVariable(alias="headphones", family="headphones", price_cents=_fixed(20000),
                          return_days=30, return_required=False),),
        warehouses=(WarehouseVariable(alias="main", delivery_days=_fixed(2),
                                      delivery_fee_cents=_fixed(0)),),
        orders=(OrderVariable(alias="order", customer_alias="customer",
                              delivered_days_ago=_fixed(1),
                              lines=(LineVariable(alias="damaged", original_sku="headphones",
                                                  quantity=_fixed(1), paid_cents=_fixed(20000),
                                                  issue="damaged", evidence_valid=True),)),),
        default_stock=_fixed(0), refund_approval_limit_cents=50000)
    built = build_world(spec, 20261021, BuildBudget(max_nodes=64))
    if built.status != "SAT" or built.world is None:
        raise RuntimeError("registered Human Customer scenario did not build")
    build_check = verify_build(spec, built)
    if not build_check["verified"]:
        raise RuntimeError("registered Human Customer world failed independent verification")
    world_check = verify_world_constraints(spec, built)
    aliases = dict(built.private_aliases)
    line_id, customer_id = aliases["damaged"], aliases["customer"]
    task = TaskSpecV2(
        task_id="human-refund-v1", world_hash=built.world_hash, customer_id=customer_id,
        customer_request=request,
        remedy_goals=(GroundedRemedyGoal(
            goal_id="refund-damaged-headphones", line_id=line_id, quantity=1,
            allowed_remedies=("refund",), maximum_extra_charge_cents=0,
            public_request_span=request, subject_mention="damaged headphones"),),
        partition="train", source_distribution_hash=digest("usami-human-customer-playground-v1"))
    profile = CustomerProfileV2(
        customer_id=customer_id, delegated_line_ids=(line_id,),
        delegation_request_spans={line_id: request}, accepted_remedies=("refund",),
        maximum_extra_charge_cents=0)
    proof_customer = CustomerScenario(
        customer_id=customer_id, delegated_line_ids=(line_id,), accepted_remedies=("refund",),
        maximum_extra_charge_cents=0, latest_arrival=None)
    proof_budget = ProofBudget(max_plan_candidates=16, max_operation_steps=24, max_witnesses=1)
    task_certificate = certify_task(built.world, task, proof_customer, proof_budget)
    if task_certificate.status != "CERTIFIED" or not task_certificate.full_task_reachability_certified:
        raise RuntimeError("registered Human Customer task did not receive a full Reference certificate")
    if not verify_task_certificate(built.world, task, proof_customer, task_certificate)["verified"]:
        raise RuntimeError("registered Human Customer task certificate failed independent replay")
    scenario = CertifiedScenario(
        scenario_id="refund-damaged-headphones", title="Refund a damaged headset",
        description="A customer bought one pair of headphones, found it damaged, and requests a refund.",
        request=request, spec=spec, build=built, world_certificate=world_check, task=task,
        profile=profile, proof_customer=proof_customer, task_certificate=task_certificate,
        aliases=aliases)
    return {scenario.scenario_id: scenario}


def _offer_view(offer: Optional[OfferPresentation]) -> Optional[dict]:
    if offer is None:
        return None
    # `rendered_terms` is generated from the actual current Quote by render_offer.
    # Keep only customer-facing terms; quote identity and hashes stay server-side.
    prefix = "Offer terms: "
    try:
        terms = ast.literal_eval(offer.rendered_terms.removeprefix(prefix))
    except (ValueError, SyntaxError):
        terms = {}
    if not isinstance(terms, dict):
        terms = {}
    return {
        "remedy": terms.get("remedy"), "quantity": terms.get("quantity"),
        "refund_cents": terms.get("refund_cents"),
        "extra_charge_cents": terms.get("extra_charge_cents"),
        "promised_arrival": terms.get("promised_arrival"),
        "return_required": terms.get("return_required"),
        "pickup_slot": terms.get("pickup_slot"),
        "operation": offer.operation,
    }


def _public_scenario(scenario: CertifiedScenario) -> dict:
    world = scenario.world
    customer_id = scenario.task.customer_id
    orders = []
    for order in world.orders.values():
        if order.customer_id != customer_id:
            continue
        lines = []
        for line in world.lines.values():
            if line.order_id != order.id:
                continue
            sku = world.skus[line.sku_id]
            evidence = next((row for row in world.evidence.values() if row.line_id == line.id), None)
            lines.append({"line_id": line.id, "product": sku.product_family,
                          "attributes": sku.attributes, "quantity": line.quantity,
                          "paid_cents": line.allocated_paid_cents,
                          "issue": evidence.issue if evidence else "unknown"})
        orders.append({"order_id": order.id, "delivered_at": order.delivered_at.isoformat(),
                       "paid_cents": order.paid_cents, "lines": lines})
    return {"scenario_id": scenario.scenario_id, "title": scenario.title,
            "description": scenario.description, "customer_request": scenario.request,
            "customer": {"id": customer_id, "name": world.customers[customer_id].name},
            "orders": orders, "world_version": world.version,
            "initial_world_hash": world.state_hash,
            "world_build_verified": True, "task_certificate_status": scenario.task_certificate.status,
            "task_certificate_verified": True}


@dataclass
class HumanSession:
    session_id: str
    owner_id: str
    scenario: CertifiedScenario
    created_at: float = field(default_factory=time.monotonic)
    last_touched: float = field(default_factory=time.monotonic)
    status: str = "starting"
    session_version: int = 0
    timeline: list = field(default_factory=list)
    pending: Optional[dict] = None
    queued_decision: Optional[CustomerDecision] = None
    submitted_turn_ids: set = field(default_factory=set)
    cancel_requested: bool = False
    cancel_reason: str = "human_session_cancelled"
    record: Optional[EpisodeRecord] = None
    replay: Optional[dict] = None
    error: Optional[str] = None
    model_name: Optional[str] = None
    service_model_spec: Optional[Dict[str, Any]] = None
    output_path: Optional[str] = None
    research_run_id: Optional[str] = None
    service_skill_snapshot_hash: Optional[str] = None
    comparison_arm: Optional[str] = None
    service_skill_snapshot: Any = None
    skill_audit: Dict[str, Any] = field(default_factory=dict)
    condition: threading.Condition = field(default_factory=lambda: threading.Condition(threading.RLock()))
    agent: Any = None
    thread: Any = None


class HumanCustomerAgent:
    """A blocking CustomerAgent backed by exactly one authenticated browser session."""
    uses_model = False
    last_call = None

    def __init__(self, session: HumanSession):
        self.session = session

    def decide(self, prompt: CustomerPrompt, profile: CustomerProfile) -> CustomerDecision:
        session = self.session
        with session.condition:
            if session.cancel_requested:
                raise EpisodeCancelled(session.cancel_reason)
            turn_id = secrets.token_urlsafe(24)
            session.pending = {"turn_id": turn_id, "session_version": session.session_version,
                               "offer": prompt.offer.model_copy(deep=True) if prompt.offer else None}
            session.status = "awaiting_customer"
            session.last_touched = time.monotonic()
            session.condition.notify_all()
            while session.queued_decision is None and not session.cancel_requested:
                remaining = HUMAN_TTL_SECONDS - (time.monotonic() - session.created_at)
                if remaining <= 0:
                    session.cancel_requested = True
                    session.cancel_reason = "human_session_expired"
                    # The runner still needs to seal, replay, and persist the
                    # interrupted Episode before this becomes terminal.
                    session.status = "cancelling"
                    break
                session.condition.wait(timeout=min(remaining, 15.0))
            if session.cancel_requested:
                raise EpisodeCancelled(session.cancel_reason)
            decision = session.queued_decision
            session.queued_decision = None
            session.pending = None
            session.status = "service_running"
            session.last_touched = time.monotonic()
            return decision

    def cancel(self, reason="human_session_cancelled"):
        with self.session.condition:
            self.session.cancel_requested = True
            self.session.cancel_reason = reason
            self.session.condition.notify_all()


class HumanSessionManager:
    def __init__(self, *, service_factory: Optional[Callable[[], tuple]] = None,
                 output_dir: Optional[Path] = None):
        self.scenarios = certified_scenarios()
        self.sessions: Dict[str, HumanSession] = {}
        self.lock = threading.RLock()
        self.service_factory = service_factory or self._real_service
        self.skill_snapshots: Dict[str, Any] = {}
        self.output_dir = Path(output_dir or os.environ.get(
            "USAMI_HUMAN_OUTPUT_DIR",
            Path(__file__).resolve().parents[2] / "human_records"))

    @staticmethod
    def _real_service(skill_snapshot=None):
        client = OpenAICompatibleClient.from_env()
        return OpenAIServiceAgent(client, skill_snapshot=skill_snapshot), client

    def register_scenario(self, scenario: CertifiedScenario):
        """Register a native, independently certified scenario for local research."""
        if not isinstance(scenario, CertifiedScenario):
            raise TypeError("only a native CertifiedScenario can be registered")
        if scenario.world.state_hash != scenario.task.world_hash:
            raise ValueError("registered scenario task is not bound to its frozen world")
        if not verify_task_certificate(scenario.world, scenario.task, scenario.proof_customer,
                                      scenario.task_certificate)["verified"]:
            raise ValueError("registered scenario certificate failed independent verification")
        with self.lock:
            self.scenarios[scenario.scenario_id] = scenario

    def register_skill_snapshot(self, snapshot):
        from vast_coevo.skills.library import LibrarySnapshot
        checked = LibrarySnapshot.model_validate(snapshot.model_dump(mode="json"))
        if checked.role != "service":
            raise ValueError("only Service snapshots can be injected into Human Service sessions")
        with self.lock:
            self.skill_snapshots[checked.snapshot_hash] = checked

    def cleanup(self):
        now = time.monotonic()
        with self.lock:
            expired = [sid for sid, row in self.sessions.items()
                       if now - row.created_at > HUMAN_TTL_SECONDS]
            for sid in expired:
                row = self.sessions[sid]
                if row.status in {"completed", "failed", "cancelled"}:
                    self.sessions.pop(sid)
                else:
                    row.cancel_requested = True
                    row.cancel_reason = "human_session_expired"
                    row.status = "cancelling"
                    if row.agent:
                        row.agent.cancel(row.cancel_reason)

    def create(self, owner_id: str, scenario_id: str, *, research_run_id: Optional[str] = None,
               service_skill_snapshot_hash: Optional[str] = None,
               comparison_arm: Optional[str] = None) -> HumanSession:
        self.cleanup()
        if comparison_arm == "S0" and service_skill_snapshot_hash is not None:
            raise HTTPException(422, detail={"code": "S0_MUST_BE_SKILL_FREE",
                "message": "S0 is the no-candidate-Skill arm; a Skill Snapshot cannot be attached."})
        if comparison_arm == "S1" and (not research_run_id or not service_skill_snapshot_hash):
            raise HTTPException(422, detail={"code": "S1_REQUIRES_CANDIDATE",
                "message": "S1 needs a Research Run and a statically validated candidate Snapshot."})
        if service_skill_snapshot_hash and not research_run_id:
            raise HTTPException(422, detail={"code": "SKILL_RUN_BINDING_REQUIRED",
                "message": "A candidate Skill may only be injected from a bound Research Run."})
        scenario = self.scenarios.get(scenario_id)
        if scenario is None:
            raise HTTPException(404, detail={"code": "SCENARIO_NOT_FOUND", "message": "该认证场景不存在。"})
        with self.lock:
            owned_active = sum(row.owner_id == owner_id and row.status in
                               {"starting", "service_running", "awaiting_customer", "cancelling"}
                               for row in self.sessions.values())
            all_active = sum(row.status in {"starting", "service_running", "awaiting_customer", "cancelling"}
                             for row in self.sessions.values())
            if owned_active >= 1 or all_active >= MAX_ACTIVE_EPISODES:
                raise HTTPException(429, detail={"code": "EPISODE_LIMIT",
                    "message": "真人实验会话已达上限；每个浏览器最多一个进行中会话，全局最多两个。"})
            if len(self.sessions) >= MAX_RETAINED_SESSIONS:
                self.cleanup()
                if len(self.sessions) >= MAX_RETAINED_SESSIONS:
                    raise HTTPException(503, detail={"code": "SESSION_CAPACITY", "message": "本地会话存储已满。"})
        skill_snapshot = None
        if service_skill_snapshot_hash:
            with self.lock:
                skill_snapshot = self.skill_snapshots.get(service_skill_snapshot_hash)
            if skill_snapshot is None:
                raise HTTPException(404, detail={"code": "SKILL_SNAPSHOT_NOT_FOUND",
                    "message": "候选 Service Snapshot 不存在或尚未通过静态检查。"})
        try:
            if skill_snapshot is not None and getattr(self.service_factory, "__func__", None) is HumanSessionManager._real_service:
                service_agent, client = self.service_factory(skill_snapshot=skill_snapshot)
            elif skill_snapshot is not None:
                import inspect
                if "skill_snapshot" not in inspect.signature(self.service_factory).parameters:
                    raise ValueError("configured service factory cannot attest Skill injection")
                service_agent, client = self.service_factory(skill_snapshot=skill_snapshot)
            else:
                service_agent, client = self.service_factory()
        except Exception as exc:
            # Provider exceptions are sanitized by the client; never include env or headers.
            raise HTTPException(503, detail={"code": "SERVICE_NOT_CONFIGURED",
                "message": "真实 Service 尚未配置或初始化失败。请检查 USAMI_LLM_MODEL、USAMI_LLM_BASE_URL 和本机凭据配置；密钥不会由浏览器读取。"}) from exc
        sid = secrets.token_urlsafe(32)
        service_settings = getattr(service_agent, "settings", None)
        service_model_spec = None
        if service_settings is not None:
            service_model_spec = ModelSpec(
                model=service_settings.model, base_url=service_settings.base_url,
                options=service_settings.options, timeout_seconds=service_settings.timeout_seconds,
                trust_env=service_settings.trust_env).model_dump(mode="json")
        session = HumanSession(session_id=sid, owner_id=owner_id,
                               scenario=scenario, status="starting",
                               model_name=getattr(getattr(service_agent, "settings", None), "model", None),
                               service_model_spec=service_model_spec,
                               research_run_id=research_run_id,
                               service_skill_snapshot_hash=service_skill_snapshot_hash,
                               comparison_arm=comparison_arm,
                               service_skill_snapshot=skill_snapshot)
        human_agent = HumanCustomerAgent(session)
        session.agent = human_agent
        with self.lock:
            self.sessions[sid] = session

        def progress(event):
            with session.condition:
                if session.cancel_requested:
                    return
                kind = event.get("kind")
                if kind == "service_running":
                    session.status = "service_running"
                    session.timeline.append({"kind": "progress", "text": event.get("message", "Service 正在处理。")})
                elif kind == "service_tool":
                    operation = event.get("operation") or {}
                    result = event.get("result") or {}
                    session.timeline.append({
                        "kind": "tool", "operation": operation,
                        "success": result.get("success"),
                        "error_code": result.get("error_code"),
                        "text": "Service 已调用业务工具：" + str(operation.get("name", "unknown")),
                    })
                elif kind == "service_reply":
                    message = event.get("message") or {}
                    session.timeline.append({"kind": "service", "text": message.get("text", ""),
                                             "offer": _offer_view(OfferPresentation.model_validate(event["offer"]))
                                             if event.get("offer") else None})
                session.last_touched = time.monotonic()
                session.condition.notify_all()

        def cancelled():
            with session.condition:
                return session.cancel_requested

        def worker():
            try:
                with session.condition:
                    session.timeline.insert(0, {"kind": "customer", "source": "frozen_request",
                                                "text": scenario.task.customer_request})
                record = run_episode(
                    scenario.world.clone(), scenario.task.model_copy(deep=True),
                    HumanCustomerProfileV2.model_validate({
                        **scenario.profile.model_dump(mode="json", exclude={"record_hash"}),
                        "return_pickup_policy": "accept_presented_quote"}),
                    human_agent, service_agent,
                    budget=EpisodeBudget(max_customer_turns=8, max_service_decisions=MAX_SERVICE_MODEL_CALLS,
                                         max_service_tool_calls=MAX_SERVICE_MODEL_CALLS,
                                         max_model_calls=MAX_SERVICE_MODEL_CALLS),
                    mode="reference", external_events=_return_events(scenario),
                    on_progress=progress, cancel_check=cancelled)
                replay = replay_episode(record)
                skill_audit = self._skill_audit(record, service_agent, skill_snapshot)
                output_path = self._persist(session, record, replay.model_dump(mode="json"), skill_audit)
                with session.condition:
                    session.record = record
                    session.replay = replay.model_dump(mode="json")
                    session.skill_audit = skill_audit
                    session.output_path = str(output_path)
                    if record.status == "customer_finished":
                        session.status = "completed"
                    elif record.status == "cancelled":
                        session.status = "cancelled"
                    else:
                        session.status = "failed"
                        session.error = record.terminal_reason
                    session.last_touched = time.monotonic()
                    session.condition.notify_all()
            except Exception as exc:
                with session.condition:
                    # A cancellation is terminal only after an EpisodeRecord
                    # has been replayed and persisted. If that sealing path
                    # itself fails, expose a failure instead of claiming a
                    # replayable cancellation.
                    session.status = "failed"
                    session.error = "Episode runner failed: " + type(exc).__name__
                    session.last_touched = time.monotonic()
                    session.condition.notify_all()
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    close()

        thread = threading.Thread(target=worker, name="usami-human-" + sid[:10], daemon=True)
        session.thread = thread
        thread.start()
        return session

    @staticmethod
    def _skill_audit(record: EpisodeRecord, service_agent, snapshot) -> dict:
        from vast_coevo.schema import digest
        if snapshot is None:
            return {"snapshot_hash": None, "injected_request_count": 0,
                    "matched_skill_ids": [], "request_hashes": []}
        needles = {row.id: f'<VAST_SKILL id="{row.id}" version="{row.version}" type="{row.type}" content_hash="{row.content_hash}">'
                   for row in snapshot.skills}
        hashes, matched = [], set()
        injected = 0
        for call in record.model_calls:
            if call.role != "service":
                continue
            messages = call.request.get("messages", []) if isinstance(call.request, dict) else []
            content = "\n".join(str(row.get("content", "")) for row in messages if isinstance(row, dict))
            found = [skill_id for skill_id, needle in needles.items() if needle in content]
            if found:
                injected += 1
                matched.update(found)
                hashes.append(digest(call.request))
        return {"snapshot_hash": snapshot.snapshot_hash, "injected_request_count": injected,
                "matched_skill_ids": sorted(matched), "request_hashes": hashes,
                "last_retrieval_trace": getattr(service_agent, "last_skill_injection", None)}

    def _persist(self, session: HumanSession, record: EpisodeRecord, replay: dict,
                 skill_audit: Optional[dict] = None) -> Path:
        root = self.output_dir / session.session_id
        root.mkdir(parents=True, exist_ok=False, mode=0o700)
        os.chmod(root, 0o700)
        episode_path = root / "episode.json"
        save_episode(record, episode_path)
        os.chmod(episode_path, 0o600)
        manifest = {
            "protocol": "human-customer-playground-run-v1",
            "customer_source": "human", "study_type": "qualitative_manual_pilot",
            "benchmark_eligible": False, "training_evidence_eligible": False,
            "scenario_id": session.scenario.scenario_id,
            "scenario_spec_hash": session.scenario.spec.record_hash,
            "world_build_hash": session.scenario.build.record_hash,
            "initial_world_hash": session.scenario.world.state_hash,
            "world_certificate_hash": session.scenario.world_certificate.record_hash,
            "task_hash": session.scenario.task.record_hash,
            "task_certificate": session.scenario.task_certificate.model_dump(mode="json"),
            "profile_hash": record.profile.record_hash,
            "service_model": session.model_name,
            "service_model_spec": session.service_model_spec,
            "service_skill_snapshot_hash": session.service_skill_snapshot_hash,
            "comparison_arm": session.comparison_arm,
            "research_run_id": session.research_run_id,
            "skill_injection_audit": skill_audit or {},
            "episode_hash": record.record_hash,
            "replay": replay,
            "episode_file": "episode.json",
        }
        path = root / "manifest.json"
        raw = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        return root

    def owned(self, owner_id: str, session_id: str) -> HumanSession:
        self.cleanup()
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None or not secrets.compare_digest(session.owner_id, owner_id):
                raise HTTPException(404, detail={"code": "SESSION_NOT_FOUND", "message": "实验会话不存在。"})
            session.last_touched = time.monotonic()
            return session

    def public_session(self, session: HumanSession) -> dict:
        with session.condition:
            pending = None
            if session.pending:
                pending = {"turn_id": session.pending["turn_id"],
                           "session_version": session.pending["session_version"],
                           "offer": _offer_view(session.pending["offer"])}
            return {
                "session_id": session.session_id, "status": session.status,
                "session_version": session.session_version,
                "scenario": _public_scenario(session.scenario),
                "timeline": list(session.timeline), "pending_turn": pending,
                "model_name": session.model_name, "model_calls_limit": MAX_SERVICE_MODEL_CALLS,
                "service_model_spec": session.service_model_spec,
                "research_run_id": session.research_run_id,
                "service_skill_snapshot_hash": session.service_skill_snapshot_hash,
                "comparison_arm": session.comparison_arm,
                "skill_injection_audit": session.skill_audit,
                "tool_calls": sum(item.get("kind") == "tool" for item in session.timeline),
                "error": session.error,
                "study_type": "qualitative_manual_pilot", "customer_source": "human",
            }

    def report(self, session: HumanSession) -> dict:
        with session.condition:
            record = session.record
            if record is None:
                raise HTTPException(409, detail={"code": "REPORT_NOT_READY", "message": "Episode 尚未结束。"})
            return {
                "customer_source": "human", "study_type": "qualitative_manual_pilot",
                "benchmark_eligible": False, "training_evidence_eligible": False,
                "scenario": _public_scenario(session.scenario),
                "scenario_spec": session.scenario.spec.model_dump(mode="json"),
                "world_build": session.scenario.build.model_dump(mode="json"),
                "world_certificate": session.scenario.world_certificate.model_dump(mode="json"),
                "task_certificate": session.scenario.task_certificate.model_dump(mode="json"),
                "episode": record.model_dump(mode="json"),
                "outcome": record.outcome.model_dump(mode="json"),
                "replay": session.replay,
                "costs": record.costs,
                "model_name": session.model_name,
                "service_model_spec": session.service_model_spec,
                "service_skill_snapshot_hash": session.service_skill_snapshot_hash,
                "comparison_arm": session.comparison_arm,
                "research_run_id": session.research_run_id,
                "skill_injection_audit": session.skill_audit,
                "persisted_at": session.output_path,
            }


def _allowed_origins() -> set:
    raw = os.environ.get("USAMI_HUMAN_ALLOWED_ORIGINS")
    if raw:
        return {value.strip().rstrip("/") for value in raw.split(",") if value.strip()}
    return {"http://127.0.0.1:5173", "http://localhost:5173"}


def install_human_customer_routes(app: FastAPI):
    manager = HumanSessionManager()
    app.state.human_sessions = manager

    def local_browser(request: Request, *, mutation: bool = False):
        host = (request.url.hostname or "").lower()
        if host not in {"127.0.0.1", "localhost", "::1", "testserver"}:
            raise HTTPException(403, detail={"code": "LOCAL_ONLY", "message": "真人实验台只允许本机访问。"})
        if mutation:
            origin = request.headers.get("origin", "").rstrip("/")
            if origin not in _allowed_origins():
                raise HTTPException(403, detail={"code": "ORIGIN_REJECTED", "message": "请求来源不在本地同源白名单中。"})
            if request.headers.get("sec-fetch-site") == "cross-site":
                raise HTTPException(403, detail={"code": "CROSS_SITE_REJECTED", "message": "拒绝跨站请求。"})

    def owner(request: Request, response: Optional[Response] = None) -> str:
        value = request.cookies.get(HUMAN_COOKIE)
        if value and len(value) >= 32:
            return value
        if response is not None:
            value = secrets.token_urlsafe(32)
            response.set_cookie(HUMAN_COOKIE, value, httponly=True, samesite="strict",
                                secure=request.url.scheme == "https", max_age=OWNER_COOKIE_TTL_SECONDS,
                                path="/api")
            return value
        raise HTTPException(401, detail={"code": "HUMAN_COOKIE_REQUIRED", "message": "请重新打开本地实验台。"})

    @app.get("/api/human/config")
    def human_config(request: Request):
        local_browser(request)
        model = os.environ.get("USAMI_LLM_MODEL") or os.environ.get("OPENAI_MODEL", "")
        base_url = os.environ.get("USAMI_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "")
        credential_hint = bool(os.environ.get("USAMI_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
                               or (os.environ.get("USAMI_LLM_KEYCHAIN_SERVICE")
                                   and os.environ.get("USAMI_LLM_KEYCHAIN_ACCOUNT"))
                               or os.environ.get("USAMI_LLM_ALLOW_NO_AUTH") == "1")
        return {"configured": bool(model and base_url and credential_hint),
                "model": model or None, "service": "real OpenAIServiceAgent",
                "credential_source_configured": credential_hint,
                "message": "密钥仅由 Python 后端读取；浏览器不会接触凭据。"}

    @app.get("/api/human/scenarios")
    def human_scenarios(request: Request):
        local_browser(request)
        return {"scenarios": [_public_scenario(row) for row in manager.scenarios.values()]}

    @app.post("/api/human/sessions")
    def create_human_session(command: HumanSessionCreate, request: Request, response: Response):
        local_browser(request, mutation=True)
        owner_id = owner(request, response)
        session = manager.create(owner_id, command.scenario_id,
                                 research_run_id=command.research_run_id,
                                 service_skill_snapshot_hash=command.service_skill_snapshot_hash,
                                 comparison_arm=command.comparison_arm)
        # If an existing cookie was used, keep its remaining local lifetime fresh.
        if request.cookies.get(HUMAN_COOKIE):
            response.set_cookie(HUMAN_COOKIE, owner_id, httponly=True, samesite="strict",
                                secure=request.url.scheme == "https", max_age=OWNER_COOKIE_TTL_SECONDS,
                                path="/api")
        return manager.public_session(session)

    @app.get("/api/human/sessions/{session_id}")
    def get_human_session(session_id: str, request: Request):
        local_browser(request)
        session = manager.owned(owner(request), session_id)
        return manager.public_session(session)

    @app.post("/api/human/sessions/{session_id}/turn")
    def submit_human_turn(session_id: str, command: HumanTurn, request: Request):
        local_browser(request, mutation=True)
        session = manager.owned(owner(request), session_id)
        if command.intent in {"message", "finish"} and not command.message.strip():
            if command.intent != "finish":
                raise HTTPException(422, detail={"code": "EMPTY_MESSAGE", "message": "请输入顾客回复。"})
        with session.condition:
            pending = session.pending
            if session.status != "awaiting_customer" or pending is None:
                raise HTTPException(409, detail={"code": "NOT_AWAITING_CUSTOMER", "message": "客服当前没有等待顾客回复。"})
            if command.turn_id in session.submitted_turn_ids:
                raise HTTPException(409, detail={"code": "TURN_ALREADY_USED", "message": "此顾客回合已经提交。"})
            if (command.turn_id != pending["turn_id"]
                    or command.session_version != pending["session_version"]
                    or command.session_version != session.session_version):
                raise HTTPException(409, detail={"code": "STALE_TURN", "message": "顾客回合已更新，请刷新当前会话。"})
            offer = pending["offer"]
            if command.intent in {"accept_offer", "reject_offer"} and offer is None:
                raise HTTPException(409, detail={"code": "NO_CURRENT_OFFER", "message": "当前回合没有可确认的真实报价。"})
            if command.intent == "accept_offer":
                decision = CustomerDecision(message="我同意此报价。", accept_quote_id=offer.quote_id,
                                            accept_operation=offer.operation)
            elif command.intent == "reject_offer":
                decision = CustomerDecision(message="我不接受此报价。", reject_quote_id=offer.quote_id)
            elif command.intent == "finish":
                decision = CustomerDecision(message=command.message.strip() or "我结束本次沟通。", finish=True)
            else:
                decision = CustomerDecision(message=command.message.strip())
            session.queued_decision = decision
            session.submitted_turn_ids.add(command.turn_id)
            session.session_version += 1
            session.timeline.append({"kind": "customer", "source": "human",
                                     "text": decision.message, "intent": command.intent})
            session.pending = None
            session.status = "service_running"
            session.last_touched = time.monotonic()
            session.condition.notify_all()
        return manager.public_session(session)

    @app.post("/api/human/sessions/{session_id}/cancel")
    def cancel_human_session(session_id: str, request: Request):
        local_browser(request, mutation=True)
        session = manager.owned(owner(request), session_id)
        with session.condition:
            if session.status in {"completed", "failed", "cancelled", "cancelling"}:
                return manager.public_session(session)
            session.cancel_requested = True
            session.cancel_reason = "human_session_cancelled"
            session.status = "cancelling"
            if session.agent:
                session.agent.cancel(session.cancel_reason)
            session.condition.notify_all()
        return manager.public_session(session)

    @app.get("/api/human/sessions/{session_id}/report")
    def human_report(session_id: str, request: Request):
        local_browser(request)
        session = manager.owned(owner(request), session_id)
        return manager.report(session)

    @app.get("/api/human/sessions/{session_id}/export")
    def export_human_episode(session_id: str, request: Request):
        local_browser(request)
        session = manager.owned(owner(request), session_id)
        report = manager.report(session)
        filename = "usami-human-episode-" + session_id[:12] + ".json"
        return JSONResponse(report, headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    return manager
