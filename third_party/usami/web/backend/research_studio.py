"""Local Research Studio orchestration over the native Usami 2.0 modules.

This module stores an auditable human-pilot workflow. It does not publish Skill
versions, change business rules, or claim a performance gate that Usami 2.0
does not yet provide.
"""
from __future__ import annotations

import copy
import json
import os
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from usami.service.client import OpenAICompatibleClient, key_from_env, settings_from_env
from usami.service.models import ServiceError
from usami_harness.models import digest
from usami_v2.runtime.agents import service_tools
from usami_v2.runtime.runner import EpisodeRecord, replay_episode
from usami_v2.scenarios.builder import BuildBudget, verify_build
from usami_v2.scenarios.certification import verify_world_constraints
from usami_v2.scenarios.designer import (
    DesignerRun, OpenAIDesignerProvider, TaskDraft, compile_task, design_and_certify,
    task_design_capabilities,
)
from usami_v2.runtime.agents import _json_arguments, _one_tool_call
from usami_v2.scenarios.distribution import (
    BusinessSource, BusinessStratum, DistributionSpec, sample_schedule,
)
from usami_v2.scenarios.task_certification import (
    CustomerScenario, ProofBudget, certify_task, verify_task_certificate,
)
from usami_v2.runtime.customer import CustomerProfileV2
from usami_v2.scenarios.tasks import TaskSpecV2
from vast_coevo.models import (
    ArtifactRef, Budget as EvidenceBudget, Consumption, EpisodeConfig, EpisodeEvidence,
    InjectionRecord, ModelCall, ModelSpec, ReplayResult, VerifiedEvent,
)
from vast_coevo.reflection.evidence import InteractionEvidenceView, wire_view
from vast_coevo.reflection.evidence import native_view
from vast_coevo.reflection.reflector import ReflectionInput, ServiceReflector
from vast_coevo.reflection.scope import DomainPolicy
from vast_coevo.schema import atomic_json
from vast_coevo.skills.library import LibrarySnapshot, LibraryStore, apply_patch, empty_snapshot
from vast_coevo.skills.retriever import SkillContext, SkillRetriever
from vast_coevo.skills.renderer import render_skills
from vast_coevo.skills.validation import StaticSkillGate

from .human_customer import (
    HUMAN_COOKIE, OWNER_COOKIE_TTL_SECONDS, CertifiedScenario, HumanSessionManager, _allowed_origins,
)


class ResearchRunCreate(BaseModel):
    seed: int = Field(default=42, strict=True)
    split: Literal["train", "validation", "heldout"] = "train"
    distribution: Literal["mixed_after_sales", "multi_line", "inventory_and_deadline"] = "mixed_after_sales"
    difficulty_preference: Literal["basic", "complex", "multi_objective"] = "multi_objective"
    designer_model: Optional[str] = Field(default=None, min_length=1, max_length=120)
    designer_timeout_seconds: float = Field(default=180.0, gt=0, le=600)
    max_attempts: int = Field(default=2, ge=1, le=3, strict=True)
    build_nodes: int = Field(default=512, ge=1, le=100000, strict=True)
    proof_candidates: int = Field(default=512, ge=1, le=10000, strict=True)
    proof_steps: int = Field(default=64, ge=1, le=2000, strict=True)


class AttachEpisodeCommand(BaseModel):
    session_id: str = Field(min_length=16, max_length=128)
    arm: Literal["S0", "S1"]
    use_for_reflection: bool = False


class ReflectionCommand(BaseModel):
    session_id: str = Field(min_length=16, max_length=128)
    explicit_human_evidence_selection: bool = False
    replace_legacy_evidence: bool = False
    timeout_seconds: float = Field(default=180.0, gt=0, le=600)


NATIVE_REFLECTION_PROTOCOL = "usami-v2-reflection-evidence-v2"


class HumanAnnotationCommand(BaseModel):
    session_id: str = Field(min_length=16, max_length=128)
    event_index: int = Field(ge=0, strict=True)
    category: Literal["decision", "communication", "repetition", "other"]
    note: str = Field(min_length=1, max_length=1000)


class ResearchCancelled(Exception):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json_copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _distribution(name: str, split: str, difficulty: str = "multi_objective") -> DistributionSpec:
    source = BusinessSource(
        source_id="usami-synthetic-business-contract-v2", kind="synthetic_assumption",
        reference="Usami 2.0 native After-sales rules and generated commerce schema",
        version="usami-v2-commerce-1", supported_claims=(
            "synthetic business feasibility", "native commerce state invariants"),
        restrictions=("not an empirical population sample", "no Designer authority over Reference rules"))
    choices = {
        "mixed_after_sales": (
            BusinessStratum(stratum_id="mixed-lines", weight=2, weight_basis_source_id=source.source_id,
                            contexts=("Multiple customer-owned product lines may interact in one request.",
                                      "Consider distinct remedies and shared operational prerequisites.")),
            BusinessStratum(stratum_id="evidence-and-policy", weight=1, weight_basis_source_id=source.source_id,
                            contexts=("Vary evidence, delivery age, return policy and approval thresholds.",)),
            BusinessStratum(stratum_id="inventory-and-deadline", weight=2, weight_basis_source_id=source.source_id,
                            contexts=("Create meaningful alternatives involving inventory, attributes, cost and delivery.",)),
        ),
        "multi_line": (
            BusinessStratum(stratum_id="multi-customer-lines", weight=1, weight_basis_source_id=source.source_id,
                            contexts=("Prefer a request about multiple distinct lines owned by one customer.",
                                      "Every goal must be grounded in the public request.")),
        ),
        "inventory_and_deadline": (
            BusinessStratum(stratum_id="inventory-deadline-tradeoffs", weight=1, weight_basis_source_id=source.source_id,
                            contexts=("Create a legitimate replacement decision with inventory and timing tradeoffs.",)),
        ),
    }
    directions = {
        "basic": "Prefer one clear business objective with a small number of prerequisites; this is a generation preference, not a measured difficulty label.",
        "complex": "Prefer several interacting but independently understandable business constraints; this is a generation preference, not a measured difficulty label.",
        "multi_objective": "Prefer multiple grounded objectives or meaningful alternatives whose shared constraints require coordinated decisions; this is a generation preference, not a measured difficulty label.",
    }
    # Sealed contracts bind their content hash to every field. ``model_copy``
    # preserves the old hash, so rebuild each stratum through validation after
    # adding the difficulty preference. Otherwise the distribution fails its
    # own hash check before Designer is called.
    strata = tuple(BusinessStratum.model_validate({
        **row.model_dump(mode="json", exclude={"record_hash"}),
        "contexts": row.contexts + (directions[difficulty],),
    }) for row in choices[name])
    return DistributionSpec(
        distribution_id="research-studio-" + name, version=1,
        population_scope="synthetic After-sales commerce worlds generated by the native ScenarioSpec Builder",
        sources=(source,), strata=strata, split=split,
        max_generation_attempts=100,
        exclusions=("No production customer data", "No manually inserted orders or evidence",))


def _proof_customer(world, task: TaskSpecV2) -> CustomerScenario:
    lines = tuple(sorted({goal.line_id for goal in task.remedy_goals}))
    remedies = tuple(sorted({remedy for goal in task.remedy_goals for remedy in goal.allowed_remedies}))
    return CustomerScenario(
        customer_id=task.customer_id, delegated_line_ids=lines,
        accepted_remedies=remedies or ("refund",), maximum_extra_charge_cents=100_000_000,
        latest_arrival=None,
        return_pickup_slot="research-return-pickup" if any(
            world.skus[world.lines[line_id].sku_id].return_required for line_id in lines) else None,
        return_received_after_days=1 if any(
            world.skus[world.lines[line_id].sku_id].return_required for line_id in lines) else None,
    )


def _profile(world, task: TaskSpecV2, proof: CustomerScenario) -> CustomerProfileV2:
    spans = {}
    for goal in task.remedy_goals:
        spans.setdefault(goal.line_id, goal.subject_mention)
    return CustomerProfileV2(
        customer_id=task.customer_id, delegated_line_ids=proof.delegated_line_ids,
        delegation_request_spans=spans, accepted_remedies=proof.accepted_remedies,
        maximum_extra_charge_cents=proof.maximum_extra_charge_cents,
        latest_arrival_days=None, return_pickup_slot=proof.return_pickup_slot,
    )


def _certified_scenario(run: DesignerRun) -> CertifiedScenario:
    if (run.status != "CERTIFIED" or run.scenario_spec is None or run.build_result is None
            or run.task is None or run.certification is None or run.build_result.world is None):
        raise ValueError("Designer run does not contain a certified native task")
    spec, build, task, cert = run.scenario_spec, run.build_result, run.task, run.certification
    world = build.world
    build_check = verify_build(spec, build)
    if not build_check.get("verified"):
        raise ValueError("native Builder verification failed")
    # The native verifier raises on a mismatch and returns a WorldCertification
    # on success; it intentionally does not expose a `verified` boolean.
    world_cert = verify_world_constraints(spec, build)
    proof = _proof_customer(world, task)
    replayed_certificate = verify_task_certificate(world, task, proof, cert)
    if not replayed_certificate.get("verified"):
        # Keep the native certificate status visible; never repair an unmatched proof policy.
        raise ValueError("native Task Certification cannot be replayed against the frozen task")
    profile = _profile(world, task, proof)
    aliases = dict(build.private_aliases)
    return CertifiedScenario(
        scenario_id="research-" + run.run_id.removeprefix("designer-")[:32],
        title="Designer task · " + task.customer_request[:72],
        description="Designer-created synthetic After-sales scenario; human qualitative pilot.",
        request=task.customer_request, spec=spec, build=build, world_certificate=world_cert,
        task=task, profile=profile, proof_customer=proof, task_certificate=cert, aliases=aliases)


def _model_spec(settings) -> ModelSpec:
    return ModelSpec(model=settings.model, base_url=settings.base_url, options=settings.options,
                     timeout_seconds=settings.timeout_seconds, trust_env=settings.trust_env)


def _designer_settings(settings, timeout_seconds: float):
    """Use an explicit, frozen Designer timeout and honor the no-output-cap pilot setting."""
    options = {key: value for key, value in settings.options.items()
               if key not in {"max_tokens", "max_completion_tokens", "max_output_tokens"}}
    return settings.model_copy(update={"timeout_seconds": timeout_seconds, "options": options})


def _designer_client(settings, env=None):
    """Build the HTTP transport from the exact frozen settings used by the run."""
    return OpenAICompatibleClient(settings, key_from_env(env))


def _provider_failure(result: DesignerRun):
    for call in reversed(result.calls):
        response = call.call.response
        error = response.get("error") if isinstance(response, dict) else None
        code = error.get("code") if isinstance(error, dict) else None
        if isinstance(code, str) and code.startswith("PROVIDER_"):
            return call.phase, code
    return None, None


def _native_assessment(record):
    """Expose native scoring and resolve process/safety refs, without assigning blame."""
    outcome = record.outcome.model_dump(mode="json")
    required = set()
    for dimension in ("process_errors", "safety_violations"):
        if dimension in outcome:
            required.update(outcome[dimension]["evidence_refs"])
    sources = []

    def add(value, source_ref):
        fingerprint = digest(value)
        if fingerprint in required:
            sources.append({"evidence_ref": fingerprint, "source_ref": source_ref,
                            "native_value": value})

    for index, goal in enumerate(outcome["state_goals"]):
        add(goal, f"/outcome/state_goals/{index}")
    communication = record.communication.model_dump(mode="json")
    for field in ("claims", "obligations"):
        for index, value in enumerate(communication[field]):
            add(value, f"/communication/{field}/{index}")
    add({"communication_hash": record.communication.record_hash}, "/communication/record_hash")
    for index, execution in enumerate(record.tool_executions):
        for row_index, operation in enumerate(execution.oracle.get("operations", ())):
            add(operation, f"/tool_executions/{index}/oracle/operations/{row_index}")
    add({"episode_status": record.status}, "/status")
    if required != {source["evidence_ref"] for source in sources}:
        raise ValueError("native assessment contains unresolved process/safety evidence references")
    tool_contracts = []
    seen_contracts = set()
    for index, call in enumerate(record.model_calls):
        if call.role == "service" and "tools" in call.request:
            tools = call.request["tools"]
            fingerprint = digest(tools)
            if fingerprint not in seen_contracts:
                seen_contracts.add(fingerprint)
                tool_contracts.append({"source_ref": f"/model_calls/{index}/request/tools", "tools": tools})
    return {"protocol": NATIVE_REFLECTION_PROTOCOL, "native_trace_hash": record.record_hash,
            "outcome": outcome, "episode_status": record.status,
            "terminal_reason": record.terminal_reason, "evidence_sources": sources,
            "service_tool_contracts": tool_contracts}


def _native_evidence(record, episode_id: str, model_spec: ModelSpec,
                     service_snapshot=None) -> tuple[EpisodeEvidence, InteractionEvidenceView]:
    """Project actual V2 calls and Oracle rows without manufacturing failures."""
    if record is None:
        raise ValueError("a persisted human Episode is required")
    native_replay = replay_episode(record)
    if not native_replay.verified:
        raise ValueError("human Episode failed a fresh native Replay")
    from usami_v2.runtime.agents import service_tools
    available = tuple(sorted(row["function"]["name"] for row in service_tools()))
    context = SkillContext(domain="aftersales", public_text=record.task.customer_request,
                           available_tools=available)
    service_snapshot = service_snapshot or empty_snapshot("service")
    retrieval = SkillRetriever("service").retrieve(service_snapshot, context)
    rendered = render_skills(retrieval.skills)
    calls = []
    injections = []
    for index, call in enumerate(row for row in record.model_calls if row.role == "service"):
        model_call = ModelCall(
            id=f"{episode_id}.service.{index}", role="service", model=model_spec,
            base_request=copy.deepcopy(call.request), request=copy.deepcopy(call.request),
            response=copy.deepcopy(call.response),
            error=None if not isinstance(call.response, dict) or "error" not in call.response
            else copy.deepcopy(call.response["error"]))
        calls.append(model_call)
        if rendered and rendered in json.dumps(call.request, ensure_ascii=False):
            injections.append(InjectionRecord(
                call_id=model_call.id, role="service", snapshot_hash=service_snapshot.snapshot_hash,
                context_hash=digest(context), hits=retrieval.hits, rendered=rendered,
                request_hash=model_call.request_hash))
    events = []
    event_ids_by_execution = []
    event_classes = {"successful_business_violation", "rejected_invalid_attempt",
                     "authorization_only", "false_rejection", "legal_execution", "unscored"}
    for index, execution in enumerate(record.tool_executions):
        oracle = execution.oracle if isinstance(execution.oracle, dict) else {}
        event_ids = []
        for row_index, row in enumerate(oracle.get("operations", ())):
            # Native per-tool traces restart their local numbering. Scope it to
            # this Episode and execution before using it as a dictionary key.
            event_id = f"{episode_id}:tool:{index}:oracle:{row_index}:{row.get('event_id', '')}"
            event_ids.append(event_id)
            events.append(VerifiedEvent(
                event_id=event_id,
                operation=str(row.get("operation") or execution.operation.get("name", "unknown")),
                classification=(row.get("classification") if row.get("classification") in event_classes
                               else "unscored"),
                reference_codes=tuple(code for code in (row.get("reference_error_code"),
                    row.get("error_code")) if isinstance(code, str) and code),
                successful_execution=row.get("success") is True,
                authorized=row.get("authorized") if isinstance(row.get("authorized"), bool) else None,
                operation_correct=row.get("operation_correct") if isinstance(row.get("operation_correct"), bool) else None,
                before_hash=execution.before_state_hash, after_hash=execution.after_state_hash,
            ))
        event_ids_by_execution.append(event_ids)
    artifact_hash = digest(record.model_dump(mode="json"))
    service_hash = service_snapshot.snapshot_hash
    customer_hash = empty_snapshot("customer").snapshot_hash
    tools = tuple(sorted({name for call in calls for tool in call.request.get("tools", [])
                          if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
                          for name in [tool["function"].get("name")] if isinstance(name, str)}))
    partition = {"train": "development", "validation": "validation", "heldout": "heldout"}.get(
        record.task.partition, "development")
    cfg = EpisodeConfig(
        id=episode_id, config_id="research-studio-human-v1", environment="usami_v2",
        domain="aftersales", partition=partition, initial_world_hash=record.initial_world.state_hash,
        identity=record.task.customer_id, mode=record.mode,
        customer_model=ModelSpec(model="human-customer", base_url="http://localhost/human-customer"),
        service_model=model_spec,
        budget=EvidenceBudget(max_customer_turns=record.budget.max_customer_turns,
                              max_customer_llm_calls=0,
                              max_service_llm_calls=record.budget.max_service_decisions,
                              max_service_tool_calls=record.budget.max_service_tool_calls),
        tools=tools)
    outcome = record.outcome
    evidence = EpisodeEvidence(
        episode_id=episode_id, config=cfg, track="service_task", task_ref=record.task.record_hash,
        customer_snapshot_hash=customer_hash, service_snapshot_hash=service_hash,
        model_calls=tuple(calls), injections=tuple(injections), events=tuple(events), failures=(),
        native_assessment=_native_assessment(record),
        consumption=Consumption(customer_llm_calls=0, service_llm_calls=len(calls),
                                reflection_calls=0, tool_calls=len(record.tool_executions)),
        termination=record.status,
        trace=ArtifactRef(path=f"human-session/{episode_id}/episode.json", sha256=artifact_hash),
        native_trace_hash=record.record_hash,
        replay=ReplayResult(verified=True, request_count=len(calls), native_trace_hash=record.record_hash),
        task_success=outcome.strict_task_success,
        legal_task_success=outcome.business_goal_success,
        # Safety PASS does not mean the Service made a correct refusal. Preserve
        # the narrower fact as Oracle events; this legacy projection stays null.
        correct_refusal=None,
        goal_satisfied=outcome.business_goal_success,
        frame_preserved=None,
    )
    customer_requests = [{
        "turn_id": "initial-request", "message": record.task.customer_request,
        "declared_write_intents": (), "accepted_write_grants": (),
        "authorization_source": "native_session_input",
        "source_refs": ("/task/customer_request",),
    }]
    tool_records = []
    service_calls = [call for call in record.model_calls if call.role == "service"]
    tool_cursor = 0
    for event_index, event in enumerate(record.events):
        if event.kind == "customer_transition":
            source = event.payload.get("customer_event") or {}
            decision = event.payload.get("decision") or {}
            grants = event.payload.get("authority_proofs") or []
            customer_requests.append({
                "turn_id": str(source.get("event_id") or f"customer-transition-{event_index}"),
                "message": str(source.get("message") or ""),
                "declared_write_intents": (dict(decision),) if decision.get("accept_quote_id") else (),
                "accepted_write_grants": tuple(grants),
                "authorization_source": "native_session_input",
                "source_refs": (f"/events/{event_index}/payload/customer_event",),
            })
    for execution_index, execution in enumerate(record.tool_executions):
        event_ids = event_ids_by_execution[execution_index]
        # The native runner issues one Service operation per ToolExecution.
        matching = None
        raw_args = None
        while tool_cursor < len(service_calls) and matching is None:
            call = service_calls[tool_cursor]; tool_cursor += 1
            response = call.response if isinstance(call.response, dict) else {}
            choices = response.get("choices") or []
            message = choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
            selected = message.get("tool_calls") or []
            for selected_tool in selected:
                function = selected_tool.get("function", {}) if isinstance(selected_tool, dict) else {}
                if function.get("name") == execution.operation.get("name"):
                    matching = selected_tool
                    raw_args = function.get("arguments")
                    break
        if matching is None or not isinstance(raw_args, str):
            continue
        event_id = event_ids[0] if event_ids else None
        tool_records.append({
            "call_id": str(matching.get("id") or f"tool-call-{execution_index}"),
            "name": str(execution.operation.get("name") or "unknown"),
            "arguments": raw_args, "response": copy.deepcopy(execution.result),
            "turn_id": None, "event_id": event_id,
            "source_refs": (f"/tool_executions/{execution_index}",),
        })
    # Every growing request remains in the immutable Episode. Losslessly reuse
    # the preceding request's identical prefix instead of resending it N times.
    service_context = []
    previous_messages = []
    for native_index, call in enumerate(record.model_calls):
        if call.role != "service":
            continue
        messages = call.request.get("messages", [])
        shared = 0
        for before, current in zip(previous_messages, messages):
            if before != current:
                break
            shared += 1
        service_context.append({
            "model_call_id": f"{record.record_hash[:12]}-service-{native_index}",
            "encoding": "previous-request-prefix-v1", "unchanged_prefix_messages": shared,
            "request_messages": copy.deepcopy(messages[shared:]),
            "source_ref": f"/model_calls/{native_index}/request/messages",
        })
        previous_messages = messages
    replies = tuple({"model_call_index": index, "content": message.text,
                     "source_ref": f"/service_messages/{index}"}
                    for index, message in enumerate(record.service_messages))
    view = native_view(evidence, customer_requests, tool_records, replies, tuple(service_context))
    return evidence, view


class ResearchRunManager:
    MAX_ACTIVE = 2

    def __init__(self, human_manager: HumanSessionManager, output_dir: Optional[Path] = None):
        self.human = human_manager
        self.root = Path(output_dir or os.environ.get(
            "USAMI_RESEARCH_OUTPUT_DIR", Path(__file__).resolve().parents[2] / "research_records"))
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.library_store = LibraryStore(self.root / "libraries")
        self.lock = threading.RLock()
        self.runs: Dict[str, dict] = {}
        self.cancel_events: Dict[str, threading.Event] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.library_store.save(empty_snapshot("service"), publish=False)
        self._load_history()

    def _run_dir(self, run_id: str) -> Path:
        if not run_id.startswith("rs-") or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-" for ch in run_id):
            raise ValueError("invalid research run ID")
        return self.root / run_id

    def _load_history(self):
        for path in sorted(self.root.glob("rs-*/current.json")):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                if not self._verify_persisted_row(row, path.parent / "journal.jsonl"):
                    continue
                self.runs[row["run_id"]] = row
                run = row.get("designer_run")
                if run:
                    designer = DesignerRun.model_validate(run)
                    if designer.status == "CERTIFIED":
                        self.human.register_scenario(_certified_scenario(designer))
            except Exception:
                # Corrupt history remains on disk and is skipped, never rewritten.
                continue
        for hash_value in self.library_store.hashes(role="service"):
            try:
                self.human.register_skill_snapshot(self.library_store.load(hash_value))
            except Exception:
                continue

    @staticmethod
    def _verify_persisted_row(row: dict, journal_path: Optional[Path] = None) -> bool:
        record_hash = row.get("record_hash")
        content = {key: value for key, value in row.items() if key != "record_hash"}
        if not isinstance(record_hash, str) or digest(content) != record_hash:
            return False
        previous = "0" * 64
        for sequence, item in enumerate(row.get("journal", [])):
            payload = {key: value for key, value in item.items() if key != "event_hash"}
            if item.get("sequence") != sequence or item.get("previous_event_hash") != previous:
                return False
            expected = digest(payload)
            if item.get("event_hash") != expected:
                return False
            previous = expected
        if journal_path is not None:
            try:
                disk_journal = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
            except (OSError, json.JSONDecodeError):
                return False
            if disk_journal != row.get("journal", []):
                return False
        return True

    def _persist(self, row: dict, event: str):
        row["updated_at"] = _now()
        folder = self._run_dir(row["run_id"])
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(folder, 0o700)
        current = folder / "current.json"
        journal = folder / "journal.jsonl"
        item = {"at": row["updated_at"], "event": event, "status": row["status"],
                "phases": copy.deepcopy(row["phases"]), "sequence": len(row["journal"]),
                "previous_event_hash": row["journal"][-1]["event_hash"] if row["journal"] else "0" * 64}
        item["event_hash"] = digest(item)
        raw = (json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n").encode()
        fd = os.open(journal, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "ab") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        row["journal"].append(item)
        row.pop("record_hash", None)
        row["record_hash"] = digest(row)
        atomic_json(current, row)
        os.chmod(current, 0o600)

    @staticmethod
    def _view(row: dict):
        result = _json_copy(row)
        result.pop("owner_hash", None)
        return result

    def list(self, owner_id: str):
        owner_hash = digest(owner_id)
        with self.lock:
            rows = [row for row in self.runs.values() if row.get("owner_hash") == owner_hash]
            return [self._view(row) for row in sorted(rows, key=lambda r: r["created_at"], reverse=True)]

    def get(self, owner_id: str, run_id: str):
        with self.lock:
            row = self.runs.get(run_id)
            if row is None or row.get("owner_hash") != digest(owner_id):
                raise HTTPException(404, detail={"code": "RESEARCH_RUN_NOT_FOUND", "message": "研究运行不存在。"})
            return row

    def claim(self, owner_id: str, run_id: str):
        """Restore local access using the unguessable run ID in the saved URL."""
        with self.lock:
            row = self.runs.get(run_id)
            if row is None:
                raise HTTPException(404, detail={"code": "RESEARCH_RUN_NOT_FOUND", "message": "研究运行不存在。"})
            if row.get("owner_hash") != digest(owner_id):
                row["owner_hash"] = digest(owner_id)
                self._persist(row, "local_run_access_recovered_by_run_id")
            return self._view(row)

    @staticmethod
    def _stored_episode(row: dict, link: dict) -> tuple[EpisodeRecord, dict, Any]:
        """Load an attached Episode from its durable ResearchRun record.

        HumanSession is intentionally short-lived and in-memory; the attached
        report is the authenticated, content-addressed source used after that
        browser session expires or the backend restarts.
        """
        report = link.get("report")
        if not isinstance(report, dict) or not isinstance(report.get("episode"), dict):
            raise HTTPException(409, detail={"code": "EPISODE_ARCHIVE_MISSING",
                "message": "研究记录中没有完整的 Episode 归档。"})
        if (report.get("research_run_id") != row.get("run_id")
                or report.get("customer_source") != "human"
                or report.get("study_type") != "qualitative_manual_pilot"
                or report.get("comparison_arm") != link.get("arm")
                or (report.get("scenario") or {}).get("scenario_id")
                    != (row.get("scenario") or {}).get("scenario_id")):
            raise HTTPException(409, detail={"code": "EPISODE_ARCHIVE_BINDING_MISMATCH",
                "message": "Episode 归档与当前研究运行或实验组不匹配。"})
        try:
            record = EpisodeRecord.model_validate(report["episode"])
        except Exception as exc:
            raise HTTPException(422, detail={"code": "EPISODE_ARCHIVE_INVALID",
                "message": "保存的 Episode 记录无法通过原生结构校验。"}) from exc
        scenario = row.get("scenario") or {}
        public = scenario.get("public") or {}
        frozen_task = scenario.get("task") or {}
        replay_receipt = link.get("replay")
        expected_skill_hash = (None if link.get("arm") == "S0"
                               else (row.get("candidate_skill") or {}).get("snapshot_hash"))
        if (record.record_hash != link.get("episode_hash")
                or record.initial_world.state_hash != link.get("initial_world_hash")
                or record.task.record_hash != link.get("task_hash")
                or record.initial_world.state_hash != public.get("initial_world_hash")
                or record.task.record_hash != frozen_task.get("record_hash")
                or not isinstance(replay_receipt, dict) or replay_receipt.get("verified") is not True
                or report.get("service_skill_snapshot_hash") != expected_skill_hash
                or (report.get("outcome") != record.outcome.model_dump(mode="json"))):
            raise HTTPException(409, detail={"code": "EPISODE_ARCHIVE_HASH_MISMATCH",
                "message": "保存的 Episode 与已冻结世界、任务或结果摘要不一致。"})
        replay = replay_episode(record)
        if not replay.verified:
            raise HTTPException(422, detail={"code": "REPLAY_FAILED",
                "message": "保存的 Episode 未通过重新执行的原生 Replay。"})
        return record, report, replay

    def create(self, owner_id: str, command: ResearchRunCreate):
        with self.lock:
            active = sum(row["status"] in {"DESIGNER_QUEUED", "DESIGNER_RUNNING", "REFLECTOR_RUNNING"}
                         for row in self.runs.values())
            if active >= self.MAX_ACTIVE:
                raise HTTPException(429, detail={"code": "RESEARCH_CONCURRENCY_LIMIT",
                    "message": "Research Studio 同时最多运行两个耗时任务。"})
            run_id = "rs-" + secrets.token_urlsafe(12).replace("_", "a").replace("-", "b")
            distribution = _distribution(command.distribution, command.split,
                                         command.difficulty_preference)
            slot = sample_schedule(distribution, command.seed, 1)[0]
            row = {
                "protocol": "usami-research-studio-v1", "run_id": run_id,
                "owner_hash": digest(owner_id), "status": "DESIGNER_QUEUED",
                "created_at": _now(), "updated_at": _now(),
                "input": {**command.model_dump(mode="json"),
                          "distribution_spec": distribution.model_dump(mode="json"),
                          "sampling_slot": slot.model_dump(mode="json"),
                          "customer_source": "human", "study_type": "qualitative_manual_pilot",
                          "benchmark_eligible": False, "training_evidence_eligible": False},
                "model": None, "phases": {"world": "QUEUED", "task": "NOT_STARTED",
                    "human_episode": "NOT_STARTED", "oracle_replay": "NOT_STARTED",
                    "reflection": "NOT_STARTED", "skill_comparison": "NOT_STARTED",
                    "history": "ACTIVE"},
                "designer_run": None, "scenario": None, "episodes": [], "reflection": None,
                "candidate_skill": None, "comparisons": [], "human_annotations": [],
                "service_s0_snapshot": empty_snapshot("service").model_dump(mode="json"),
                "error": None, "cancel_requested": False,
                "journal": [],
            }
            self.runs[run_id] = row
            cancel = threading.Event(); self.cancel_events[run_id] = cancel
            self._persist(row, "research_run_registered")
            thread = threading.Thread(target=self._design_worker,
                args=(run_id, command, distribution, slot, cancel), name="usami-research-designer-" + run_id[-8:], daemon=True)
            self.threads[run_id] = thread
            thread.start()
            return self._view(row)

    def _design_worker(self, run_id, command, distribution, slot, cancel):
        client = None
        manager = self
        class ManagedProvider:
            def __init__(self, wrapped): self.wrapped = wrapped; self.last_call = None
            def submit(self, phase, messages, model_schema):
                if cancel.is_set():
                    raise ResearchCancelled("cancel requested")
                with manager.lock:
                    row = manager.runs[run_id]
                    row["status"] = "DESIGNER_RUNNING"
                    row["phases"][phase] = "RUNNING"
                    manager._persist(row, "designer_" + phase + "_request_started")
                try:
                    result = self.wrapped.submit(phase, messages, model_schema)
                finally:
                    # design_and_certify reads provider.last_call on failure,
                    # keeping the exact attempted request/response auditable.
                    self.last_call = self.wrapped.last_call
                with manager.lock:
                    row = manager.runs[run_id]
                    row["phases"][phase] = "SUBMITTED"
                    manager._persist(row, "designer_" + phase + "_response_received")
                return result
        try:
            settings = settings_from_env()
            if command.designer_model:
                settings = settings.model_copy(update={"model": command.designer_model})
            settings = _designer_settings(settings, command.designer_timeout_seconds)
            model_spec = _model_spec(settings)
            # Construct the transport only after applying the run-specific
            # timeout. Updating client.settings after from_env() would leave
            # the already-created httpx.Client at its environment default.
            client = _designer_client(settings)
            with self.lock:
                row = self.runs[run_id]; row["model"] = model_spec.model_dump(mode="json")
                self._persist(row, "designer_model_frozen")
            provider = ManagedProvider(OpenAIDesignerProvider(client))
            result = design_and_certify(
                provider, distribution, slot,
                build_budget=BuildBudget(max_nodes=command.build_nodes),
                proof_budget=ProofBudget(max_plan_candidates=command.proof_candidates,
                                         max_operation_steps=command.proof_steps, max_witnesses=1),
                customer_scenario_factory=_proof_customer, return_receipt_event_configured=True,
                max_attempts=command.max_attempts)
            with self.lock:
                row = self.runs[run_id]
                row["designer_run"] = result.model_dump(mode="json")
                failed_phase, provider_code = _provider_failure(result)
                if provider_code:
                    blocked_state = "BLOCKED_PROVIDER_TIMEOUT" if provider_code == "PROVIDER_TIMEOUT" else "BLOCKED_PROVIDER_ERROR"
                    row["phases"][failed_phase] = blocked_state
                    if failed_phase == "world":
                        row["phases"]["task"] = "NOT_STARTED"
                else:
                    row["phases"]["world"] = "COMPLETED" if result.build_result and result.build_result.status == "SAT" else (
                        result.build_result.status if result.build_result else "REJECTED")
                    row["phases"]["task"] = ("CERTIFIED" if result.status == "CERTIFIED" else
                                               "UNKNOWN" if result.status == "UNKNOWN" else "REJECTED")
                row["status"] = "CANCELLED" if cancel.is_set() else result.status
                if result.status == "CERTIFIED" and not cancel.is_set():
                    self._activate_certified(row, result)
                elif result.validation_errors:
                    row["error"] = {"code": provider_code or result.status, "messages": list(result.validation_errors)}
                self._persist(row, "designer_completed_" + result.status.lower())
        except Exception as exc:
            with self.lock:
                row = self.runs[run_id]
                row["status"] = "CANCELLED" if cancel.is_set() else "FAILED"
                row["error"] = {"code": getattr(exc, "code", "RESEARCH_DESIGNER_FAILED"),
                                 "type": type(exc).__name__,
                                 "message": "生成未完成；查看保留的 Designer 请求和验证反馈。"}
                row["phases"]["world"] = "CANCELLED" if cancel.is_set() else "FAILED"
                self._persist(row, "designer_failed_or_cancelled")
        finally:
            if client is not None:
                close = getattr(client, "close", None)
                if callable(close): close()

    def _activate_certified(self, row, result):
        scenario = _certified_scenario(result)
        self.human.register_scenario(scenario)
        row["scenario"] = {
            "scenario_id": scenario.scenario_id,
            "public": __import__("web.backend.human_customer", fromlist=["_public_scenario"])._public_scenario(scenario),
            "scenario_spec": scenario.spec.model_dump(mode="json"),
            "build_result": scenario.build.model_dump(mode="json"),
            "world_certificate": scenario.world_certificate.model_dump(mode="json"),
            "task": scenario.task.model_dump(mode="json"),
            "proof_customer": scenario.proof_customer.model_dump(mode="json"),
            "task_certificate": scenario.task_certificate.model_dump(mode="json"),
            "profile_hash": scenario.profile.record_hash,
            "researcher_hidden_goal_view": [goal.model_dump(mode="json") for goal in scenario.task.remedy_goals]
                + [goal.model_dump(mode="json") for goal in scenario.task.information_goals],
            "reference_witness_summary": {"witness_count": len(scenario.task_certificate.witness_hashes),
                "witness_hashes": list(scenario.task_certificate.witness_hashes),
                "business_reachability_certified": scenario.task_certificate.business_reachability_certified,
                "full_task_reachability_certified": scenario.task_certificate.full_task_reachability_certified,
                "interaction_certified": scenario.task_certificate.interaction_certified},
        }
        row["status"] = "READY_FOR_CUSTOMER"
        row["phases"]["human_episode"] = "READY"
        row["phases"]["oracle_replay"] = "WAITING_FOR_EPISODE"

    def revalidate_saved_task(self, run_id):
        """Offline recovery of the last raw Task proposal, with zero provider calls.

        Call with the server stopped. Keep the original sealed record beside
        the new journal entry; never rewrite the original model responses.
        """
        with self.lock:
            row = self.runs[run_id]
            if row["status"] != "BLOCKED" or row["episodes"]:
                raise ValueError("only a blocked run without Episodes can be revalidated")
            if not self._verify_persisted_row(row, self._run_dir(run_id) / "journal.jsonl"):
                raise ValueError("research record integrity check failed")
            source = DesignerRun.model_validate(row["designer_run"])
            call = next(item for item in reversed(source.calls) if item.phase == "task")
            function = _one_tool_call(call.call.response)["tool_calls"][0]["function"]
            if function["name"] != "submit_customer_task":
                raise ValueError("saved Task response called the wrong tool")
            draft = TaskDraft.model_validate(_json_arguments(function["arguments"]))
            spec, build = source.scenario_spec, source.build_result
            if not verify_build(spec, build)["verified"]:
                raise ValueError("saved world failed native Builder verification")
            capabilities = task_design_capabilities(spec, build, return_receipt_event_configured=True)
            task = compile_task(draft, spec, build, source.distribution, capabilities=capabilities)
            proof = _proof_customer(build.world, task)
            command = ResearchRunCreate.model_validate({
                key: value for key, value in row["input"].items() if key in ResearchRunCreate.model_fields})
            budget = ProofBudget(max_plan_candidates=command.proof_candidates,
                max_operation_steps=command.proof_steps, max_witnesses=1)
            certificate = certify_task(build.world, task, proof, budget, minimum_witnesses=1)
            if certificate.status != "CERTIFIED":
                raise ValueError("saved task remains uncertified: " + ", ".join(certificate.blockers))
            verified = verify_task_certificate(build.world, task, proof, certificate)
            if not verified["verified"]:
                raise ValueError("revalidated task certificate failed native Replay")
            result = DesignerRun.model_validate({
                **source.model_dump(mode="json", exclude={"record_hash"}),
                "status": "CERTIFIED", "task_draft": draft.model_dump(mode="json"),
                "task": task.model_dump(mode="json"),
                "certification": certificate.model_dump(mode="json"), "pre_run_quality": None})
            archive = self._run_dir(run_id) / ("before-revalidation-" + row["record_hash"] + ".json")
            atomic_json(archive, row)
            os.chmod(archive, 0o600)
            report = {"protocol": "research-saved-task-revalidation-v1",
                "original_record_hash": row["record_hash"], "original_record_file": archive.name,
                "source_call_hash": call.call.record_hash, "source_attempt": call.attempt,
                "world_hash": build.world_hash, "model_calls": 0,
                "capabilities": capabilities.model_dump(mode="json"),
                "return_received_after_days": proof.return_received_after_days,
                "certificate_hash": certificate.record_hash, "replay": verified}
            self._activate_certified(row, result)
            row["designer_run"] = result.model_dump(mode="json")
            row["task_revalidations"] = [*row.get("task_revalidations", []), report]
            row["phases"]["task"] = "CERTIFIED"
            row["error"] = None
            self._persist(row, "saved_task_revalidated_without_model_calls")
            return self._view(row)

    def cancel(self, owner_id, run_id):
        row = self.get(owner_id, run_id)
        if row["status"] not in {"DESIGNER_QUEUED", "DESIGNER_RUNNING", "REFLECTOR_RUNNING"}:
            raise HTTPException(409, detail={"code": "RUN_NOT_ACTIVE", "message": "当前运行已结束，不能取消。"})
        row["cancel_requested"] = True
        event = self.cancel_events.get(run_id)
        if event: event.set()
        with self.lock: self._persist(row, "cancel_requested")
        return self._view(row)

    def attach_episode(self, owner_id, run_id, session_id, arm, use_for_reflection=False):
        row = self.get(owner_id, run_id)
        if row["status"] not in {"READY_FOR_CUSTOMER", "READY_FOR_SKILL_TEST", "REFLECTION_COMPLETE"}:
            raise HTTPException(409, detail={"code": "TASK_NOT_READY", "message": "任务尚未完成独立认证。"})
        with self.lock:
            existing = next((item for item in row["episodes"] if item["session_id"] == session_id), None)
        if existing is not None:
            if existing.get("arm") != arm:
                raise HTTPException(409, detail={"code": "SERVICE_ARM_MISMATCH",
                    "message": "已关联 Episode 的实验组不能更改。"})
            record, _, _ = self._stored_episode(row, existing)
            if use_for_reflection:
                if arm != "S0":
                    raise HTTPException(422, detail={"code": "REFLECTION_BASELINE_ONLY",
                        "message": "本版 Reflector 仅消费用户明确选择的 S0 训练 Episode。"})
                if record.task.partition != "train":
                    raise HTTPException(422, detail={"code": "NON_TRAIN_EVIDENCE_FORBIDDEN",
                        "message": "只有 train Episode 可被明确选择进入本次 Reflection；validation / held-out 保持隔离。"})
                with self.lock:
                    existing["explicitly_selected_for_reflection"] = True
                    self._persist(row, "human_episode_explicitly_selected_for_reflection")
            return self._view(row)
        session = self.human.owned(owner_id, session_id)
        if session.research_run_id != run_id or session.scenario.scenario_id != row["scenario"]["scenario_id"]:
            raise HTTPException(409, detail={"code": "EPISODE_NOT_BOUND", "message": "Episode 与当前冻结研究任务不匹配。"})
        if session.status not in {"completed", "failed", "cancelled"} or session.record is None:
            raise HTTPException(409, detail={"code": "EPISODE_NOT_FINISHED", "message": "Episode 尚未封存。"})
        replay = replay_episode(session.record)
        if not replay.verified:
            raise HTTPException(422, detail={"code": "REPLAY_FAILED", "message": "原生 Replay 未验证，不能作为研究证据。"})
        if session.record.initial_world.state_hash != row["scenario"]["public"]["initial_world_hash"]:
            raise HTTPException(409, detail={"code": "WORLD_MISMATCH", "message": "Episode 初始世界与冻结世界不一致。"})
        if session.record.task.record_hash != row["scenario"]["task"]["record_hash"]:
            raise HTTPException(409, detail={"code": "TASK_MISMATCH", "message": "Episode TaskSpec 与冻结任务不一致。"})
        expected_hash = None if arm == "S0" else (row.get("candidate_skill") or {}).get("snapshot_hash")
        if arm == "S1" and not expected_hash:
            raise HTTPException(409, detail={"code": "CANDIDATE_REQUIRED", "message": "需要先有静态合法的候选 Skill。"})
        if session.service_skill_snapshot_hash != expected_hash:
            raise HTTPException(409, detail={"code": "SERVICE_ARM_MISMATCH", "message": "Service Skill 版本与所选实验组不匹配。"})
        if use_for_reflection and arm != "S0":
            raise HTTPException(422, detail={"code": "REFLECTION_BASELINE_ONLY", "message": "本版 Reflector 仅消费用户明确选择的 S0 训练 Episode。"})
        if use_for_reflection and session.record.task.partition != "train":
            raise HTTPException(422, detail={"code": "NON_TRAIN_EVIDENCE_FORBIDDEN",
                "message": "只有 train Episode 可被明确选择进入本次 Reflection；validation / held-out 保持隔离。"})
        with self.lock:
            report = self.human.report(session)
            link = {"session_id": session_id, "arm": arm,
                "customer_source": "human", "study_type": "qualitative_manual_pilot",
                "benchmark_eligible": False, "training_evidence_eligible": False,
                "explicitly_selected_for_reflection": bool(use_for_reflection),
                "initial_world_hash": session.record.initial_world.state_hash,
                "task_hash": session.record.task.record_hash,
                "episode_hash": session.record.record_hash,
                "replay": replay.model_dump(mode="json"),
                "skill_injection_audit": session.skill_audit,
                "outcome": session.record.outcome.model_dump(mode="json"),
                "costs": session.record.costs,
                "report": report}
            row["episodes"].append(link)
            row["phases"]["human_episode"] = "COMPLETED"
            row["phases"]["oracle_replay"] = "VERIFIED"
            if arm == "S1":
                audit = session.skill_audit
                candidate = row.get("candidate_skill") or {}
                if audit.get("injected_request_count", 0) > 0:
                    candidate["status"] = "INJECTED_FOR_TEST"
                    candidate["injection_evidence"] = audit
                else:
                    candidate["status"] = "STATIC_VALID"
                    candidate["injection_evidence"] = audit
                    candidate["injection_warning"] = "Snapshot 已加载，但本轮没有技能命中并进入 Service 请求。"
                row["candidate_skill"] = candidate
            pair = {item["arm"]: item for item in row["episodes"] if item["arm"] in {"S0", "S1"}}
            if "S0" in pair and "S1" in pair:
                row["comparisons"] = [{
                    "type": "same_world_human_qualitative_sanity_check",
                    "status": "INCONCLUSIVE",
                    "reason": "两次对话由真人分别控制；请求路径可能不同，且样本不足以做独立动态验收。",
                    "S0": _comparison_metrics(pair["S0"]), "S1": _comparison_metrics(pair["S1"]),
                    "same_initial_world": pair["S0"]["initial_world_hash"] == pair["S1"]["initial_world_hash"],
                    "same_task": pair["S0"]["task_hash"] == pair["S1"]["task_hash"],
                }]
                row["phases"]["skill_comparison"] = "INCONCLUSIVE"
            self._persist(row, "human_episode_attached_and_replayed_" + arm)
        return self._view(row)

    def reflect(self, owner_id, run_id, session_id, replace_legacy_evidence=False, timeout_seconds=180.0):
        row = self.get(owner_id, run_id)
        link = next((item for item in row["episodes"] if item["session_id"] == session_id), None)
        if not link or not link["explicitly_selected_for_reflection"]:
            raise HTTPException(409, detail={"code": "HUMAN_EVIDENCE_NOT_SELECTED",
                "message": "请先明确选择一条 Replay 验证的 S0 真人 Episode 作为本次反思证据。"})
        if link.get("arm") != "S0":
            raise HTTPException(422, detail={"code": "REFLECTION_BASELINE_ONLY",
                "message": "本版 Reflector 仅消费用户明确选择的 S0 训练 Episode。"})
        record, report, _ = self._stored_episode(row, link)
        settings = settings_from_env().model_copy(update={"timeout_seconds": timeout_seconds})
        model_spec = _model_spec(settings)
        if record.task.partition != "train":
            raise HTTPException(422, detail={"code": "NON_TRAIN_EVIDENCE_FORBIDDEN",
                "message": "只有 train Episode 可被明确选择进入本次 Reflection；validation / held-out 保持隔离。"})
        stored_model_spec = report.get("service_model_spec")
        if not isinstance(stored_model_spec, dict):
            raise HTTPException(409, detail={"code": "EPISODE_MODEL_SPEC_MISSING",
                "message": "保存的 Episode 缺少原始 Service 模型配置，不能准确归因反思证据。"})
        evidence_model = ModelSpec.model_validate(stored_model_spec)
        evidence, interaction_view = _native_evidence(
            record, session_id, evidence_model, service_snapshot=empty_snapshot("service"))
        parent = empty_snapshot("service")
        reflection_input = ReflectionInput(
            generation=0, role="service", library=parent, evidence=(evidence,),
            interaction_views=(interaction_view,),
            allowed_tools=evidence.config.tools,
            domain_policy=DomainPolicy(add_domains=("aftersales",)),
            public_policy="Use native After-sales operations only. Manual human pilots are qualitative evidence; do not infer hidden goals or claim a formal performance result.",
            boundaries="The Service may only act through existing tools and exact Customer authorization; do not alter Engine, Reference, Oracle, policy or permissions.")
        with self.lock:
            previous = row.get("reflection")
            if row["status"] == "REFLECTOR_RUNNING":
                raise HTTPException(409, detail={"code": "REFLECTION_RUNNING", "message": "本次反思仍在运行，请等待结果。"})
            legacy_noop = (previous is not None and previous.get("status") == "NOOP"
                           and previous.get("evidence_protocol") is None)
            if replace_legacy_evidence and not legacy_noop:
                raise HTTPException(409, detail={"code": "LEGACY_REFLECTION_REQUIRED",
                    "message": "仅旧版证据映射产生的 NOOP 可显式重做；新版结果不能重复抽样。"})
            if previous and previous.get("status") != "FAILED" and not (replace_legacy_evidence and legacy_noop):
                raise HTTPException(409, detail={"code": "REFLECTION_ALREADY_RUN",
                    "message": "此研究运行已完成 Reflection；旧版 NOOP 需明确选择使用修正后的证据再反思。"})
            if previous:
                row.setdefault("reflection_history", []).append(copy.deepcopy(previous))
            row["reflection"] = None
            row["status"] = "REFLECTOR_RUNNING"
            row["phases"]["reflection"] = "RUNNING"
            self._persist(row, "service_reflection_started")
        thread = threading.Thread(target=self._reflection_worker,
            args=(run_id, reflection_input, settings, session_id), name="usami-research-reflector-" + run_id[-8:], daemon=True)
        self.threads[run_id + ":reflection"] = thread
        thread.start()
        return self._view(row)

    def _reflection_worker(self, run_id, reflection_input, settings, episode_id):
        client = None
        try:
            # The transport must use this attempt's frozen Reflection settings,
            # rather than silently re-reading the Service's shorter timeout.
            client = OpenAICompatibleClient(settings, key_from_env())
            model = _model_spec(settings)
            # Match the existing Designer compatibility policy: Qwen3 thinking
            # endpoints reject forced function-name tool_choice before inference.
            choice = "auto" if "qwen3" in model.model.casefold() else None
            record = ServiceReflector(client, model, tool_choice=choice).reflect(reflection_input)
            reflection = {"record": record.model_dump(mode="json"),
                          "evidence_protocol": NATIVE_REFLECTION_PROTOCOL,
                          "source_episode_id": episode_id,
                          "input_hash": record.input_hash,
                          "input": reflection_input.model_dump(mode="json"),
                          "raw_call": record.call.model_dump(mode="json") if record.call else None,
                          "patch": record.patch.model_dump(mode="json") if record.patch else None,
                          "status": "NOOP" if record.patch and record.patch.action == "NOOP" else "PROPOSED",
                          "static_validation": None, "error": record.error,
                          "model": model.model_dump(mode="json"),
                          "calls": record.calls}
            candidate_info = None
            if record.patch and record.patch.action != "NOOP":
                scene = self.runs[run_id]["scenario"]
                world = DesignerRun.model_validate(self.runs[run_id]["designer_run"]).build_result.world
                identifiers = set(world.customers) | set(world.merchants) | set(world.orders) | set(world.lines) | set(world.skus) | set(world.warehouses) | set(world.evidence)
                gate = StaticSkillGate(allowed_tools=reflection_input.allowed_tools,
                                       forbidden_identifiers=identifiers,
                                       allowed_domains={"aftersales"})
                validation = gate.validate(empty_snapshot("service"), record.patch)
                reflection["static_validation"] = validation.model_dump(mode="json")
                if validation.passed:
                    candidate = apply_patch(empty_snapshot("service"), record.patch,
                                            "human_episode:" + episode_id)
                    self.library_store.save(candidate, publish=False)
                    self.human.register_skill_snapshot(candidate)
                    reflection["status"] = "STATIC_VALID"
                    candidate_info = {"status": "STATIC_VALID", "snapshot_hash": candidate.snapshot_hash,
                                      "parent_snapshot_hash": candidate.parent_hash,
                                      "snapshot": candidate.model_dump(mode="json"),
                                      "skill": record.patch.skill.model_dump(mode="json") if record.patch.skill else None,
                                      "patch_hash": record.patch.record_hash,
                                      "source_episode_id": episode_id,
                                      "evidence_ids": list(record.patch.evidence_ids),
                                      "durable_candidate_only": True}
                else:
                    reflection["status"] = "REJECTED"
            elif record.error:
                reflection["status"] = "FAILED"
            with self.lock:
                row = self.runs[run_id]
                row["reflection"] = reflection
                row["candidate_skill"] = candidate_info
                row["status"] = "READY_FOR_SKILL_TEST" if candidate_info else "REFLECTION_COMPLETE"
                row["phases"]["reflection"] = reflection["status"]
                row["phases"]["skill_comparison"] = "READY" if candidate_info else "NOT_AVAILABLE"
                self._persist(row, "service_reflection_completed_" + reflection["status"].lower())
        except Exception as exc:
            with self.lock:
                row = self.runs[run_id]
                row["status"] = "FAILED"
                row["phases"]["reflection"] = "FAILED"
                row["reflection"] = {"status": "FAILED", "error": {
                    "code": getattr(exc, "code", "REFLECTION_FAILED"), "type": type(exc).__name__,
                    "message": "Reflector 未能完成；原始请求或可用响应保留在本次运行记录中。"},
                    "evidence_protocol": NATIVE_REFLECTION_PROTOCOL, "source_episode_id": episode_id}
                self._persist(row, "service_reflection_failed")
        finally:
            if client is not None:
                close = getattr(client, "close", None)
                if callable(close): close()


def _comparison_metrics(link):
    report = link["report"]
    outcome = report["outcome"]
    return {"business_goal_success": outcome.get("business_goal_success"),
            "process_errors": outcome.get("process_errors", {}).get("status"),
            "safety_violations": outcome.get("safety_violations", {}).get("status"),
            "strict_task_success": outcome.get("strict_task_success"),
            "service_calls": link["costs"].get("service_calls"),
            "tool_calls": link["costs"].get("tool_calls"),
            "known_tokens": link["costs"].get("known_total_tokens"),
            "replay_verified": link["replay"].get("verified"),
            "skill_injected_requests": link.get("skill_injection_audit", {}).get("injected_request_count", 0)}


def _is_local_request(request: Request, mutation: bool):
    if (request.url.hostname or "").lower() not in {"127.0.0.1", "localhost", "::1", "testserver"}:
        raise HTTPException(403, detail={"code": "LOCAL_ONLY", "message": "Research Studio 只允许本机访问。"})
    if mutation:
        origin = request.headers.get("origin", "").rstrip("/")
        if origin not in _allowed_origins():
            raise HTTPException(403, detail={"code": "ORIGIN_REJECTED", "message": "请求来源不在本地同源白名单中。"})
        if request.headers.get("sec-fetch-site") == "cross-site":
            raise HTTPException(403, detail={"code": "CROSS_SITE_REJECTED", "message": "拒绝跨站请求。"})


def _owner(request: Request, response: Optional[Response] = None):
    value = request.cookies.get(HUMAN_COOKIE)
    if value and len(value) >= 32:
        if response is not None:
            response.set_cookie(HUMAN_COOKIE, value, httponly=True, samesite="strict",
                                secure=request.url.scheme == "https", max_age=OWNER_COOKIE_TTL_SECONDS,
                                path="/api")
        return value
    if response is None:
        raise HTTPException(401, detail={"code": "RESEARCH_COOKIE_REQUIRED", "message": "请重新打开本地 Research Studio。"})
    value = secrets.token_urlsafe(32)
    response.set_cookie(HUMAN_COOKIE, value, httponly=True, samesite="strict",
                        secure=request.url.scheme == "https", max_age=OWNER_COOKIE_TTL_SECONDS, path="/api")
    return value


def install_research_studio_routes(app: FastAPI, human_manager: HumanSessionManager,
                                   output_dir: Optional[Path] = None):
    manager = ResearchRunManager(human_manager, output_dir=output_dir)
    app.state.research_runs = manager

    @app.get("/api/research/config")
    def research_config(request: Request):
        _is_local_request(request, False)
        try:
            settings = settings_from_env()
            configured = True
            model = settings.model
        except Exception:
            configured, model = False, None
        return {"configured": configured, "model": model,
                "service": "native Usami 2.0 Designer / ServiceReflector / Human Runner",
                "max_parallel_runs": ResearchRunManager.MAX_ACTIVE,
                "message": "凭据只由 Python 后端读取，不会返回给浏览器。"}

    @app.get("/api/research/runs")
    def research_history(request: Request, response: Response):
        _is_local_request(request, False)
        return {"runs": manager.list(_owner(request, response))}

    @app.post("/api/research/runs")
    def create_research_run(command: ResearchRunCreate, request: Request, response: Response):
        _is_local_request(request, True)
        return manager.create(_owner(request, response), command)

    @app.get("/api/research/runs/{run_id}")
    def get_research_run(run_id: str, request: Request, response: Response):
        _is_local_request(request, False)
        return manager._view(manager.get(_owner(request, response), run_id))

    @app.post("/api/research/runs/{run_id}/claim")
    def claim_research_run(run_id: str, request: Request, response: Response):
        _is_local_request(request, True)
        return manager.claim(_owner(request, response), run_id)

    @app.post("/api/research/runs/{run_id}/cancel")
    def cancel_research_run(run_id: str, request: Request):
        _is_local_request(request, True)
        return manager.cancel(_owner(request), run_id)

    @app.post("/api/research/runs/{run_id}/attach-episode")
    def attach_research_episode(run_id: str, command: AttachEpisodeCommand, request: Request):
        _is_local_request(request, True)
        return manager.attach_episode(_owner(request), run_id, command.session_id,
                                      command.arm, command.use_for_reflection)

    @app.post("/api/research/runs/{run_id}/annotations")
    def add_human_annotation(run_id: str, command: HumanAnnotationCommand, request: Request):
        _is_local_request(request, True)
        owner_id = _owner(request)
        row = manager.get(owner_id, run_id)
        link = next((item for item in row["episodes"] if item["session_id"] == command.session_id), None)
        if link is None:
            raise HTTPException(404, detail={"code": "EPISODE_NOT_ATTACHED", "message": "Episode 尚未关联到当前研究运行。"})
        episode = (link.get("report") or {}).get("episode") or {}
        tool_executions = episode.get("tool_executions") or []
        if command.event_index >= len(tool_executions):
            raise HTTPException(422, detail={"code": "EVENT_NOT_FOUND", "message": "该时间线位置没有已记录的工具事件。"})
        annotation = {"annotation_id": "ha-" + secrets.token_urlsafe(10), "created_at": _now(),
                      "session_id": command.session_id, "event_index": command.event_index,
                      "category": command.category, "note": command.note.strip(),
                      "source": "human_researcher", "official_oracle_counted": False,
                      "reflection_eligible": False}
        with manager.lock:
            row.setdefault("human_annotations", []).append(annotation)
            manager._persist(row, "human_annotation_added")
        return manager._view(row)

    @app.post("/api/research/runs/{run_id}/reflect")
    def reflect_on_research_episode(run_id: str, command: ReflectionCommand, request: Request):
        _is_local_request(request, True)
        if command.explicit_human_evidence_selection is not True:
            raise HTTPException(422, detail={"code": "EXPLICIT_SELECTION_REQUIRED",
                "message": "人工 Episode 默认不进入反思；必须显式选择这条证据。"})
        return manager.reflect(_owner(request), run_id, command.session_id,
                               command.replace_legacy_evidence, command.timeout_seconds)

    @app.get("/api/research/runs/{run_id}/export")
    def export_research_run(run_id: str, request: Request, response: Response):
        _is_local_request(request, False)
        data = manager._view(manager.get(_owner(request, response), run_id))
        filename = f"usami-research-{run_id}.json"
        return JSONResponse(data, headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    return manager
