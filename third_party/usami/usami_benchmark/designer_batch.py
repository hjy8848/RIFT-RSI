"""World-grounded, versioned single-shot batch Designer.

The model proposes ID-free semantic challenges in one structured call. The
private Grounder binds those challenges to registered worlds, then the existing
task compiler, Reference, Engine, Oracle and Replay certify every candidate.
No model response is treated as a proof.
"""
from __future__ import annotations

import json
import hashlib
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Optional

from pydantic import Field
from pydantic import ValidationError

from usami.service.models import ModelSettings, ServiceError, strict_json
from usami.service.runner import completion_message
from usami_harness.models import Model, digest
from vast_coevo.schema import atomic_json

from .abstract_challenges import (AbstractChallenge, BATCH_REVISION_PROTOCOL, ChallengeBatch,
    CONTRACT_VERSION, PROTOCOL, challenge_fingerprints, submission_schema, validate_challenge)
from .grounder import GroundingBudget, GroundingResult, ground_challenge
from .tasks import CertifiedTask, ConstructionBudget, compile_task, verify_task
from .world_index import PrivateWorldIndex, public_domain_catalogue, load_world_index


TOOL_NAME = "submit_challenge_batch"
POLICY = """You are the Usami benchmark Designer. Propose exactly the requested number of abstract After-sales challenges. Decide which policy boundaries, relationships and objectives are worth testing. Use only the registered entity types, fields, predicates, relationships and goal forms in the catalogue/schema. Challenges must be grounded in some world from the authorized pool, but the pool's individual objects and outcomes are not shown. Use aliases, never concrete IDs. Do not provide frames, native operations, Reference codes, expected error labels, witnesses, arbitrary code, SQL or invented facts. State a customer request that would naturally elicit the challenge; do not reveal evaluator-only objectives or expected policy answers in that request. A candidate is only a hypothesis: local grounding and native Reference certification decide whether it works."""


class BatchDesignerConfig(Model):
    protocol: Literal[PROTOCOL, BATCH_REVISION_PROTOCOL] = PROTOCOL
    num_challenges: int = Field(default=5, strict=True, ge=1, le=5)
    partition: Literal["train", "validation"] = "train"
    grounding: GroundingBudget = Field(default_factory=GroundingBudget)
    construction: ConstructionBudget = Field(default_factory=lambda: ConstructionBudget(max_nodes=4096, max_depth=16, max_paths=1))

    @property
    def max_api_calls(self):
        return 2 if self.protocol == BATCH_REVISION_PROTOCOL else 1


class CertifiedAttempt(Model):
    match_hash: str
    world_registration_hash: str
    world_hash: str
    status: Literal["CERTIFIED", "GROUNDED_BUT_REFERENCE_UNPROVEN", "COMPILER_REJECTED", "REPLAY_REJECTED", "INTERNAL_ERROR"]
    reason: str = ""
    task_hash: Optional[str] = None
    certificate_hashes: list[str] = Field(default_factory=list)
    task: Optional[CertifiedTask] = None


class CandidateAttempt(Model):
    challenge_id: str
    raw: Any
    abstract_status: Literal["VALID", "SCHEMA_INVALID", "INVALID_CONSTRAINT"]
    reason: str = ""
    challenge: Optional[AbstractChallenge] = None
    challenge_hash: Optional[str] = None
    model_request_hash: Optional[str] = None
    structural_fingerprint: Optional[str] = None
    semantic_fingerprint: Optional[str] = None
    grounding: Optional[GroundingResult] = None
    certified_attempts: list[CertifiedAttempt] = Field(default_factory=list)
    final_status: str = "SCHEMA_INVALID"
    revision: int = 0
    grounding_ms: float = 0.0
    reference_ms: float = 0.0


class BatchRun(Model):
    contract_version: str = "world-grounded-contract-v1"
    protocol: str
    status: str
    domain: str
    partition: str
    index_hash: str
    pool_commitment: str
    catalogue: dict
    catalogue_hash: str
    model: ModelSettings
    config: BatchDesignerConfig
    requests: list[dict] = Field(default_factory=list)
    request_revisions: list[int] = Field(default_factory=list)
    responses: list[Any] = Field(default_factory=list)
    response_revisions: list[int] = Field(default_factory=list)
    transport_errors: list[dict] = Field(default_factory=list)
    candidates: list[CandidateAttempt] = Field(default_factory=list)
    candidate_history: list[CandidateAttempt] = Field(default_factory=list)
    cost: dict = Field(default_factory=dict)
    metrics: dict = Field(default_factory=dict)
    run_hash: str = ""

    def computed_hash(self):
        excluded = {"run_hash"}
        if self.contract_version == "world-grounded-contract-v1":
            excluded.add("contract_version")
        return digest(self.model_dump(mode="json", exclude=excluded))


def _tool_schema(count=5):
    return [{"type": "function", "function": {"name": TOOL_NAME,
        "description": "Submit the requested batch of ID-free abstract business challenges.",
        "parameters": submission_schema(count)}}]


def _messages(catalogue: dict, count: int, *, feedback: Optional[dict] = None, candidates: Optional[list[dict]] = None):
    task = {
        "requested_count": count,
        "challenge_requirements": ["each challenge has at least one objective", "use aliases and typed constraints only",
            "prefer distinct rule combinations and goal structures across the batch", "do not output world/entity IDs",
            "use required_customer_information/preferences only when they are part of the intended test"],
        "required_candidate_fields": ChallengeBatch.model_json_schema()["$defs"]["AbstractChallenge"]["required"],
        "wire_format": "Use entity_variables as [{alias,type}], entity_constraints as [{alias,field,op,value}] or [{alias,predicate,value}], relationship_constraints as [{type,left,right}], and objectives as [{kind,target,...}]. Do not substitute aliases, constraints, objective type or free-form description fields. A target is a line alias, connected to a profile alias via profile_for_line. Combine several atomic objectives for a multi-object challenge.",
    }
    messages = [{"role": "system", "content": POLICY},
        {"role": "user", "content": "Public domain catalogue and authorized world-pool aggregate summary (no per-world objects or outcomes):\n" +
            json.dumps(catalogue, ensure_ascii=False, sort_keys=True, separators=(",", ":")) +
            "\n\nProduce the batch using the supplied function schema.\n" + json.dumps(task, sort_keys=True)}]
    if feedback is not None:
        revision = {"instruction": "One revision is permitted. Keep every candidate not explicitly listed in controlled feedback semantically unchanged, especially certified candidates. Modify only listed failed candidates; do not infer a Reference answer. Return a complete batch with stable challenge_id values. If an initial candidate omitted or malformed its ID, use its assigned challenge_id from the feedback.",
        "prior_candidates": candidates or [], "controlled_feedback": feedback}
        messages.append({"role": "user", "content": "Single bounded revision request:\n" + json.dumps(revision, ensure_ascii=False, sort_keys=True, separators=(",", ":"))})
    return messages


def _request(client, catalogue, count, *, feedback=None, candidates=None):
    request = client.settings.request(_messages(catalogue, count, feedback=feedback, candidates=candidates), _tool_schema(count))
    # Match the current v4 Designer request policy: no harness-side output cap.
    # Providers may impose their own context or generation limits; those are
    # surfaced as truncated/failed responses and are never retried implicitly.
    for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        request.pop(key, None)
    # A forced structured call prevents a free-form-only response from silently
    # consuming the sole Designer attempt.
    request["tool_choice"] = {"type": "function", "function": {"name": TOOL_NAME}}
    return request


def _response_rows(response):
    try:
        _, actions = completion_message(response, set())
    except ServiceError as exc:
        choices = response.get("choices", []) if isinstance(response, dict) else []
        finish = choices[0].get("finish_reason") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
        kind = "TRUNCATED" if finish == "length" else "FORMAT_INVALID"
        return None, {"kind": kind, "code": exc.code, "message": str(exc)}
    if len(actions) != 1 or actions[0]["function"]["name"] != TOOL_NAME:
        return None, {"kind": "FORMAT_INVALID", "message": "expected exactly one batch submission function call"}
    raw_args = actions[0]["function"]["arguments"]
    try:
        raw = strict_json(raw_args)
    except ValueError as exc:
        return None, {"kind": "FORMAT_INVALID", "message": str(exc), "raw_arguments": raw_args}
    if not isinstance(raw, dict) or set(raw) != {"protocol", "challenges"} or raw.get("protocol") != PROTOCOL or not isinstance(raw.get("challenges"), list):
        return None, {"kind": "SCHEMA_INVALID", "message": "batch envelope has an unknown protocol or fields", "raw": raw}
    if not (1 <= len(raw["challenges"]) <= 5):
        return None, {"kind": "SCHEMA_INVALID", "message": "batch must contain between one and five challenges", "raw": raw}
    return raw["challenges"], None


def _controlled_failure(candidate: CandidateAttempt) -> dict:
    status = candidate.final_status
    category = {
        "SCHEMA_INVALID": "Schema Invalid", "INVALID_CONSTRAINT": "Unsupported Constraint",
        "NO_MATCH": "World No Match", "OUT_OF_SCOPE": "Grounding Scope Invalid",
        "GROUNDED_BUT_REFERENCE_UNPROVEN": "Grounded But Reference Unproven",
        "COMPILER_REJECTED": "Grounded But Reference Unproven",
        "TRANSPORT_ERROR": "Transport Error",
    }.get(status, status)
    return {"challenge_id": candidate.challenge_id, "category": category, "reason": candidate.reason[:500]}


def _process_candidate(raw: Any, index: PrivateWorldIndex, config: BatchDesignerConfig, revision: int) -> CandidateAttempt:
    proposed_id = raw.get("challenge_id") if isinstance(raw, dict) else None
    fallback_id = proposed_id if isinstance(proposed_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", proposed_id) else f"invalid-{revision}-{digest(raw)[:12]}"
    try:
        challenge = AbstractChallenge.model_validate(raw)
    except ValidationError as exc:
        return CandidateAttempt(challenge_id=str(fallback_id)[:64], raw=raw, abstract_status="SCHEMA_INVALID",
            reason=str(exc), final_status="SCHEMA_INVALID", revision=revision)
    try:
        challenge = validate_challenge(challenge)
    except ValueError as exc:
        return CandidateAttempt(challenge_id=challenge.challenge_id, raw=raw, abstract_status="INVALID_CONSTRAINT",
            reason=str(exc), challenge=challenge, challenge_hash=challenge.challenge_hash,
            final_status="INVALID_CONSTRAINT", revision=revision)
    fingerprints = challenge_fingerprints(challenge)
    result = CandidateAttempt(challenge_id=challenge.challenge_id, raw=raw, abstract_status="VALID", challenge=challenge,
        challenge_hash=challenge.challenge_hash, structural_fingerprint=fingerprints["structural"],
        semantic_fingerprint=fingerprints["semantic"], revision=revision)
    grounding_budget = config.grounding.model_copy(update={"partition": config.partition})
    grounding_started = time.perf_counter()
    try:
        grounding = ground_challenge(challenge, index, grounding_budget)
    except Exception as exc:
        result.grounding_ms = (time.perf_counter() - grounding_started) * 1000
        result.final_status = "INTERNAL_ERROR"
        result.reason = "Grounder raised " + type(exc).__name__
        return result
    result.grounding_ms = (time.perf_counter() - grounding_started) * 1000
    result.grounding = grounding
    if grounding.status != "GROUNDED":
        result.final_status = grounding.status; result.reason = grounding.reason
        return result
    attempts = []
    for match in grounding.matches:
        frozen = next(row.frozen for row in index.worlds if row.registration_hash == match.frozen_registration_hash)
        reference_started = time.perf_counter()
        try:
            task = compile_task(frozen, match.selection, config.construction)
        except Exception as exc:
            message = str(exc)
            bounded = message.lower().startswith("bounded ") and "did not prove" in message.lower()
            attempts.append(CertifiedAttempt(match_hash=match.binding_hash,
                world_registration_hash=match.frozen_registration_hash, world_hash=match.frozen_world_hash,
                status="GROUNDED_BUT_REFERENCE_UNPROVEN" if bounded else ("COMPILER_REJECTED" if isinstance(exc, ValueError) else "INTERNAL_ERROR"), reason=message[:1200]))
            result.reference_ms += (time.perf_counter() - reference_started) * 1000
            continue
        try:
            verify_task(task)
        except Exception as exc:
            attempts.append(CertifiedAttempt(match_hash=match.binding_hash,
                world_registration_hash=match.frozen_registration_hash, world_hash=match.frozen_world_hash,
                status="REPLAY_REJECTED", reason=str(exc)[:1200]))
            result.reference_ms += (time.perf_counter() - reference_started) * 1000
            continue
        attempts.append(CertifiedAttempt(match_hash=match.binding_hash,
            world_registration_hash=match.frozen_registration_hash, world_hash=match.frozen_world_hash,
            status="CERTIFIED", task_hash=task.task_hash, certificate_hashes=[w.certificate.certificate_hash for w in task.dialogue_witnesses],
            task=task))
        result.reference_ms += (time.perf_counter() - reference_started) * 1000
    result.certified_attempts = attempts
    if any(a.status in {"INTERNAL_ERROR", "REPLAY_REJECTED"} for a in attempts):
        result.final_status = "INTERNAL_ERROR" if any(a.status == "INTERNAL_ERROR" for a in attempts) else "REPLAY_REJECTED"
        result.reason = "; ".join(a.reason for a in attempts if a.reason)[:1200]
    elif any(a.status == "CERTIFIED" for a in attempts):
        result.final_status = "CERTIFIED"
    elif attempts and all(a.status == "GROUNDED_BUT_REFERENCE_UNPROVEN" for a in attempts):
        result.final_status = "GROUNDED_BUT_REFERENCE_UNPROVEN"
        result.reason = "all deterministic grounded matches remained unproven by the bounded native Reference"
    else:
        result.final_status = next((s for s in ("INTERNAL_ERROR", "REPLAY_REJECTED") if any(a.status == s for a in attempts)), "COMPILER_REJECTED")
        result.reason = "; ".join(a.reason for a in attempts if a.reason)[:1200]
    return result


def _process_rows(raw_rows: list[Any], index: PrivateWorldIndex, config: BatchDesignerConfig, revision: int,
                  model_request_hash: Optional[str] = None):
    """Deterministically materialize one provider batch, including malformed slots."""
    ids = [raw.get("challenge_id") for raw in raw_rows if isinstance(raw, dict) and isinstance(raw.get("challenge_id"), str)]
    duplicate_ids = {item for item in ids if ids.count(item) > 1}
    processed = []
    count_error = len(raw_rows) != config.num_challenges
    for raw in raw_rows:
        duplicate = isinstance(raw, dict) and isinstance(raw.get("challenge_id"), str) and raw["challenge_id"] in duplicate_ids
        if count_error or duplicate:
            original_id = raw.get("challenge_id") if isinstance(raw, dict) else None
            stable_id = original_id if not duplicate and isinstance(original_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", original_id) else f"invalid-{revision}-{len(processed)+1}"
            attempt = CandidateAttempt(challenge_id=stable_id, raw=raw,
                abstract_status="SCHEMA_INVALID", final_status="SCHEMA_INVALID", revision=revision,
                reason="batch candidate count differs from requested count" if count_error else "duplicate challenge_id in the submitted batch")
        else:
            attempt = _process_candidate(raw, index, config, revision)
        if attempt.abstract_status == "SCHEMA_INVALID":
            own_id = raw.get("challenge_id") if isinstance(raw, dict) and isinstance(raw.get("challenge_id"), str) and not duplicate else None
            unavailable_ids = {c.challenge_id for c in processed} | (set(ids) - {own_id})
            while attempt.challenge_id in unavailable_ids:
                attempt.challenge_id += "-invalid"
        attempt.model_request_hash = model_request_hash
        processed.append(attempt)
    if len(processed) < config.num_challenges:
        for n in range(len(processed), config.num_challenges):
            processed.append(CandidateAttempt(challenge_id=f"missing-{revision}-{n+1}", raw=None,
                abstract_status="SCHEMA_INVALID", reason="batch omitted a requested candidate",
                final_status="SCHEMA_INVALID", revision=revision, model_request_hash=model_request_hash))
    return processed


def _usage(response):
    raw = response.get("usage", {}) if isinstance(response, dict) else {}
    raw = raw if isinstance(raw, dict) else {}
    result = {"input_tokens": raw.get("prompt_tokens", raw.get("input_tokens")),
              "output_tokens": raw.get("completion_tokens", raw.get("output_tokens")), "total_tokens": raw.get("total_tokens")}
    return {key: value if type(value) is int and value >= 0 else None for key, value in result.items()}


def _usage_complete(rows):
    return bool(rows) and all(all(type(v) is int for v in row.values())
        and row["total_tokens"] == row["input_tokens"] + row["output_tokens"] for row in rows)


def _summary(candidates, requests, responses, errors):
    attempts = candidates
    valid = [c for c in attempts if c.abstract_status == "VALID"]
    grounded = [c for c in valid if c.grounding and c.grounding.status == "GROUNDED"]
    certified = [c for c in valid if c.final_status == "CERTIFIED"]
    structures = {c.structural_fingerprint for c in valid if c.structural_fingerprint}
    semantics = {c.semantic_fingerprint for c in valid if c.semantic_fingerprint}
    certified_structures = {c.structural_fingerprint for c in certified if c.structural_fingerprint}
    certified_semantics = {c.semantic_fingerprint for c in certified if c.semantic_fingerprint}
    usage = [_usage(response) for response in responses]
    return {"raw_generated_challenges": "NOT_MEASURED",
        "candidate_attempts": len(attempts), "valid_abstract_challenges": len(valid),
        "groundable_challenges": len(grounded), "certified_challenges": len(certified),
        "unique_structural_challenges": len(structures), "unique_semantic_challenges": len(semantics),
        "unique_certified_structural_challenges": len(certified_structures),
        "unique_certified_semantic_challenges": len(certified_semantics),
        "grounding_failures": sum(c.final_status in {"NO_MATCH", "INVALID_CONSTRAINT", "OUT_OF_SCOPE", "BUDGET_EXHAUSTED"} for c in attempts),
        "reference_unproven": sum(c.final_status == "GROUNDED_BUT_REFERENCE_UNPROVEN" for c in attempts),
        "schema_errors": sum(c.abstract_status == "SCHEMA_INVALID" for c in attempts),
        "internal_errors": sum(c.final_status == "INTERNAL_ERROR" for c in attempts),
        "replay_errors": sum(c.final_status == "REPLAY_REJECTED" for c in attempts),
        "transport_errors": sum(e.get("stage") == "transport" for e in errors),
        "response_errors": sum(e.get("stage") == "response" for e in errors),
        "preflight_errors": sum(e.get("stage") == "preflight" for e in errors),
        "revision_protocol_errors": sum(e.get("stage") == "revision" for e in errors),
        "completed_llm_calls": len(responses), "llm_usage_by_call": usage,
        "grounding_local_ms": round(sum(c.grounding_ms for c in attempts), 3),
        "reference_local_ms": round(sum(c.reference_ms for c in attempts), 3),
        "actual_service_executions": 0, "service_outcomes": "NOT RUN"}


def run_batch(index: PrivateWorldIndex, client, config: BatchDesignerConfig | dict | None = None) -> BatchRun:
    """Execute one batch call, optionally followed by one versioned revision."""
    config = config if isinstance(config, BatchDesignerConfig) else BatchDesignerConfig.model_validate(config or {})
    index.checked()
    catalogue = public_domain_catalogue(index, "aftersales", partitions=(config.partition,))
    model = client.settings.model_copy(deep=True)
    requests, request_revisions, responses, response_revisions, errors = [], [], [], [], []
    candidates: dict[str, CandidateAttempt] = {}
    candidate_history: list[CandidateAttempt] = []
    previous_raw = []
    for revision in range(config.max_api_calls):
        feedback = None
        if revision:
            feedback = {"failures": [_controlled_failure(c) for c in candidates.values() if c.final_status != "CERTIFIED"],
                        "classification_note": "Reference search exhaustion is not proof that a task is impossible."}
            feedback["failures"].extend({"challenge_id": f"batch-error-{i}", "category": e.get("kind", "Transport Error"),
                "reason": e.get("message", e.get("code", "initial response failed"))[:500]}
                for i, e in enumerate(errors) if e.get("revision") == 0 and e.get("stage") in {"response", "transport"})
            if not feedback["failures"]:
                break
        request = _request(client, catalogue, config.num_challenges, feedback=feedback, candidates=previous_raw if revision else None)
        requests.append(request)
        request_revisions.append(revision)
        if getattr(client, "available", True) is False:
            errors.append({"stage": "preflight", "revision": revision, "kind": "EXTERNAL_MODEL_UNAVAILABLE",
                           "message": getattr(client, "unavailable_reason", "model client is not configured")})
            break
        try:
            response = client.complete(request)
            responses.append(response)
            response_revisions.append(revision)
        except ServiceError as exc:
            errors.append({"stage": "transport", "revision": revision, "code": exc.code, "message": str(exc), "response": exc.response})
            break
        except Exception as exc:
            # A client implementation error is not a malformed model proposal.
            # Never archive an unknown exception's potentially credential-bearing text.
            errors.append({"stage": "transport", "revision": revision, "code": "CLIENT_INTERNAL_ERROR",
                           "message": "completion client raised " + type(exc).__name__})
            break
        raw_rows, parse_error = _response_rows(response)
        if parse_error:
            errors.append({"stage": "response", "revision": revision, **parse_error})
            previous_raw = []
            continue
        previous_raw = raw_rows
        if len(raw_rows) != config.num_challenges:
            errors.append({"stage": "response", "revision": revision, "kind": "CANDIDATE_COUNT_MISMATCH",
                           "expected": config.num_challenges, "actual": len(raw_rows)})
        processed = _process_rows(raw_rows, index, config, revision, digest(request))
        candidate_history.extend(processed)
        if revision == 0:
            for attempt in processed:
                candidates[attempt.challenge_id] = attempt
        else:
            fresh_batch = not candidates
            for attempt in processed:
                prior = candidates.get(attempt.challenge_id)
                if prior is None:
                    if fresh_batch:
                        # A whole malformed initial envelope has no stable
                        # candidate IDs to revise. Permit one fresh valid batch,
                        # while retaining the failed response and its feedback.
                        candidates[attempt.challenge_id] = attempt
                        continue
                    errors.append({"stage": "revision", "revision": revision, "kind": "REVISION_PROTOCOL_VIOLATION",
                                   "message": "revision introduced a new challenge_id"})
                    continue
                if prior.final_status == "CERTIFIED":
                    if prior.challenge_hash != attempt.challenge_hash:
                        errors.append({"stage": "revision", "revision": revision, "kind": "REVISION_PROTOCOL_VIOLATION",
                                       "challenge_id": attempt.challenge_id,
                                       "message": "revision changed a previously certified candidate; original retained"})
                    continue
                # Only failed candidates may be replaced, and their full first
                # attempt stays in the immutable responses/cost journal.
                candidates[attempt.challenge_id] = attempt
        if any(c.final_status in {"INTERNAL_ERROR", "REPLAY_REJECTED"} for c in processed):
            break
    run_status = _run_status(responses, errors)
    run = BatchRun(contract_version=CONTRACT_VERSION, protocol=config.protocol, status=run_status,
        domain="aftersales", partition=config.partition, index_hash=index.index_hash,
        pool_commitment=catalogue["pool_commitment"], catalogue=catalogue,
        catalogue_hash=catalogue["catalogue_hash"], model=model, config=config, requests=requests, responses=responses,
        request_revisions=request_revisions, response_revisions=response_revisions,
        transport_errors=errors, candidates=list(candidates.values()), candidate_history=candidate_history)
    _set_accounting(run)
    run.run_hash = run.computed_hash()
    return run


def _set_accounting(run):
    requests, responses, errors, candidate_history = run.requests, run.responses, run.transport_errors, run.candidate_history
    baseline = run.protocol == "rule-combination-sampling-baseline-v1"
    run.metrics = _summary(run.candidates, requests, responses, errors)
    run.metrics["generated_candidate_attempts"] = len(candidate_history)
    run.metrics["valid_abstract_attempts"] = sum(c.abstract_status == "VALID" for c in candidate_history)
    run.metrics["schema_invalid_attempts"] = sum(c.abstract_status == "SCHEMA_INVALID" for c in candidate_history)
    raw_count = 0
    for response in responses:
        rows, error = _response_rows(response)
        if rows is not None:
            raw_count += len(rows)
        elif isinstance((error or {}).get("raw"), dict) and isinstance(error["raw"].get("challenges"), list):
            raw_count += len(error["raw"]["challenges"])
    run.metrics["raw_generated_challenges"] = raw_count if not baseline else len(candidate_history)
    if baseline:
        run.metrics["generator"] = "deterministic grammar-combination sampler; not an official IntellAgent reproduction"
    count = run.metrics["certified_challenges"]
    run.metrics["certification_status"] = "ALL_CERTIFIED" if count == run.config.num_challenges else ("PARTIALLY_CERTIFIED" if count else "NO_CERTIFIED_CHALLENGES")
    preflight_failures = sum(e.get("stage") == "preflight" for e in errors)
    run.metrics["request_attempts"] = len(requests)
    run.metrics["completed_responses"] = len(responses)
    run.metrics["provider_call_attempts"] = len(requests) - preflight_failures
    run.metrics["total_llm_calls"] = run.metrics["provider_call_attempts"]
    run.cost = {"designer_api_calls": len(requests) - preflight_failures,
        "request_attempts": len(requests), "completed_responses": len(responses),
        "input_tokens": sum((x["input_tokens"] or 0) for x in run.metrics["llm_usage_by_call"]),
        "output_tokens": sum((x["output_tokens"] or 0) for x in run.metrics["llm_usage_by_call"]),
        "total_tokens": sum((x["total_tokens"] or 0) for x in run.metrics["llm_usage_by_call"]),
        "usage_complete": baseline or _usage_complete(run.metrics["llm_usage_by_call"]),
        "grounding_world_checks": sum(c.grounding.worlds_considered for c in run.candidate_history if c.grounding),
        "reference_compile_attempts": sum(len(c.certified_attempts) for c in run.candidate_history),
        "grounding_local_ms": round(sum(c.grounding_ms for c in run.candidate_history), 3),
        "reference_local_ms": round(sum(c.reference_ms for c in run.candidate_history), 3),
        "grounding_reference_api_calls": 0}


def _run_status(responses, errors):
    if any(e.get("code") == "CLIENT_INTERNAL_ERROR" for e in errors):
        return "INTERNAL_ERROR"
    return "COMPLETED" if responses else "BLOCKED_EXTERNAL_MODEL"


def _pool_manifest(index: PrivateWorldIndex) -> dict:
    return {"protocol": "world-grounded-pool-manifest-v1", "pool_hash": index.pool_hash,
        "index_hash": index.index_hash, "worlds": [{"spec": w.frozen.spec.model_dump(mode="json"),
            "registration_hash": w.registration_hash, "world_hash": w.world_hash,
            "structure_signature": w.structure_signature, "permission_signature": w.permission_signature,
            "partition": w.partition, "domain": w.domain, "rules_version": w.rules_version}
            for w in sorted(index.worlds, key=lambda x: x.registration_hash)]}


def save_run(run: BatchRun, index: PrivateWorldIndex, output_dir: str | Path) -> Path:
    """Write a new audit directory without replacing prior runs."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"protocol": run.protocol, "contract_version": run.contract_version, "status": run.status,
        "run_hash": run.run_hash, "index_hash": run.index_hash, "catalogue_hash": run.catalogue_hash,
        "model": run.model.model_dump(mode="json"), "request_attempts": len(run.requests),
        "provider_call_attempts": run.cost.get("designer_api_calls", 0),
        "completed_responses": len(run.responses),
        "created_at_policy": "not recorded; output content is deterministic except for model response"}
    atomic_json(root / "world-pool-manifest.json", _pool_manifest(index), immutable=True)
    # This file is evaluation-side/private: it contains entity IDs and facts and
    # must not be placed in an independently held-out prompt or service context.
    atomic_json(root / "grounding-index.json", index.model_dump(mode="json"), immutable=True)
    atomic_json(root / "catalogue.json", run.catalogue, immutable=True)
    atomic_json(root / "model-request.json", run.requests, immutable=True)
    atomic_json(root / "model-response.json", {"responses": run.responses, "errors": run.transport_errors}, immutable=True)
    atomic_json(root / "abstract-challenges.json", [c.model_dump(mode="json") for c in run.candidates], immutable=True)
    atomic_json(root / "candidate-attempt-history.json", [c.model_dump(mode="json") for c in run.candidate_history], immutable=True)
    atomic_json(root / "grounding-results.json", [c.grounding.model_dump(mode="json") for c in run.candidates if c.grounding], immutable=True)
    atomic_json(root / "compilation-results.json", [{"challenge_id": c.challenge_id, "status": c.final_status,
        "reason": c.reason, "attempts": [{k: v for k, v in a.model_dump(mode="json").items() if k != "task"} for a in c.certified_attempts]}
        for c in run.candidates], immutable=True)
    cert_dir = root / "certificates"; cert_dir.mkdir()
    cert_count = 0
    for candidate in run.candidates:
        for n, attempt in enumerate(candidate.certified_attempts):
            if attempt.task is not None:
                atomic_json(cert_dir / f"{candidate.challenge_id}-match-{n}.json", attempt.task.model_dump(mode="json"), immutable=True)
                cert_count += 1
    atomic_json(root / "batch-run.json", run.model_dump(mode="json"), immutable=True)
    try:
        replay = replay_run(run, index)
    except Exception as exc:
        atomic_json(root / "replay-verification.json", {"verified": False, "error": str(exc)}, immutable=True)
        raise
    atomic_json(root / "replay-verification.json", replay, immutable=True)
    atomic_json(root / "metrics.json", {**run.metrics, "cost": run.cost, "certificates_saved": cert_count,
        "replay_verified": replay["verified"]}, immutable=True)
    lines = ["# World-Grounded Batch Designer run", "", f"- Protocol: `{run.protocol}`",
        f"- Status: `{run.status}`", f"- Candidates: {run.metrics.get('raw_generated_challenges', 0)} raw; {run.metrics.get('certified_challenges', 0)} certified",
        f"- Certification: `{run.metrics.get('certification_status')}` (COMPLETED describes the call, not task validity)",
        f"- Designer calls: {run.cost.get('designer_api_calls', 0)}", f"- Tokens: {run.cost.get('total_tokens', 0)} (usage complete: {run.cost.get('usage_complete')})",
        f"- Local Reference compile attempts: {run.cost.get('reference_compile_attempts', 0)}",
        f"- Replay: `{replay['verified']}`", "", "This is task-generation evidence only. It contains no Service execution or performance result."]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    manifest["artifacts"] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()}
    atomic_json(root / "manifest.json", manifest, immutable=True)
    return root


def replay_directory(root: str | Path, index_path=None):
    """Verify exported files too, not only the authoritative batch-run JSON."""
    root = Path(root)
    manifest = strict_json((root / "manifest.json").read_text())
    if manifest.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("historical v1 archive requires original source commit cbb5da4; it is not reinterpreted as contract v2")
    artifacts = manifest.get("artifacts", {})
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p != root / "manifest.json"}
    if not artifacts or set(artifacts) != actual:
        raise ValueError("archive artifact file set differs from manifest")
    for name, expected in artifacts.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or (root / path).is_symlink():
            raise ValueError("archive artifact path is unsafe")
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError("archive artifact hash differs: " + name)
    index = load_world_index(index_path or root / "grounding-index.json")
    row = BatchRun.model_validate(strict_json((root / "batch-run.json").read_text()))
    if (any(manifest.get(k) != getattr(row, k) for k in ("protocol", "contract_version", "status", "run_hash", "index_hash", "catalogue_hash"))
            or manifest.get("model") != row.model.model_dump(mode="json")
            or manifest.get("request_attempts") != len(row.requests)
            or manifest.get("completed_responses") != len(row.responses)
            or manifest.get("provider_call_attempts") != row.cost.get("designer_api_calls")):
        raise ValueError("archive manifest differs from batch record")
    return replay_run(row, index)


def replay_run(run: BatchRun | dict, index: PrivateWorldIndex) -> dict:
    """Rebind every candidate and recompile every archived native certificate."""
    row = run if isinstance(run, BatchRun) else BatchRun.model_validate(run)
    if row.contract_version != CONTRACT_VERSION:
        raise ValueError("historical v1 archive requires original source commit cbb5da4; it is not reinterpreted as contract v2")
    index.checked()
    if row.run_hash != row.computed_hash() or row.index_hash != index.index_hash:
        raise ValueError("batch run/index integrity differs")
    catalogue = public_domain_catalogue(index, row.domain, partitions=(row.partition,))
    if catalogue != row.catalogue or catalogue["catalogue_hash"] != row.catalogue_hash:
        raise ValueError("public catalogue Replay differs")
    verified_tasks = 0
    rule_baseline = row.protocol == "rule-combination-sampling-baseline-v1"
    if (row.domain != "aftersales" or row.partition != row.config.partition
            or row.pool_commitment != catalogue["pool_commitment"]
            or (not rule_baseline and row.protocol != row.config.protocol)):
        raise ValueError("run scope or protocol differs from frozen configuration")
    if rule_baseline and (row.requests or row.responses or row.request_revisions or row.response_revisions):
        raise ValueError("offline rule baseline must not contain model requests or responses")
    if len(row.requests) != len(row.request_revisions) or len(row.responses) != len(row.response_revisions):
        raise ValueError("batch request/response revision journals are incomplete")
    if len(row.requests) > row.config.max_api_calls or row.request_revisions != sorted(row.request_revisions):
        raise ValueError("batch API-call budget or revision order differs")
    if any(rev not in range(row.config.max_api_calls) for rev in row.request_revisions + row.response_revisions):
        raise ValueError("batch journal contains an unregistered revision")
    if row.request_revisions != list(range(len(row.requests))) or row.response_revisions != sorted(set(row.response_revisions)):
        raise ValueError("model revision journals must be unique and sequential")
    if any(e.get("stage") not in {"transport", "preflight", "response", "revision"} or e.get("revision") not in row.request_revisions for e in row.transport_errors):
        raise ValueError("unregistered error stage or revision")
    rebuilt_history = []
    if rule_baseline:
        count = row.metrics.get("raw_generated_challenges")
        if type(count) is not int or not (1 <= count <= 5):
            raise ValueError("offline rule baseline candidate count is invalid")
        rebuilt_history = _process_rows(_rule_challenges(count), index, row.config, 0)
        raw_reparsed = count
    else:
        client_proxy = SimpleNamespace(settings=row.model)
        request_hashes = {}
        for request, revision in zip(row.requests, row.request_revisions):
            if revision in request_hashes:
                raise ValueError("duplicate model request for one revision")
            request_hashes[revision] = digest(request)
            feedback = None
            prior_raw = None
            if revision:
                prior_candidates = [c for c in row.candidate_history if c.revision == 0]
                prior_map = {}
                for candidate in prior_candidates:
                    prior_map[candidate.challenge_id] = candidate
                failures = [_controlled_failure(c) for c in prior_map.values() if c.final_status != "CERTIFIED"]
                failures.extend({"challenge_id": f"batch-error-{i}", "category": e.get("kind", "Transport Error"),
                    "reason": e.get("message", e.get("code", "initial response failed"))[:500]}
                    for i, e in enumerate(row.transport_errors) if e.get("revision") == 0 and e.get("stage") in {"response", "transport"})
                if not failures:
                    raise ValueError("revision request was made without controlled failure feedback")
                feedback = {"failures": failures,
                    "classification_note": "Reference search exhaustion is not proof that a task is impossible."}
                if row.responses:
                    initial_revision = row.response_revisions.index(0) if 0 in row.response_revisions else None
                    if initial_revision is not None:
                        parsed_initial, parse_error = _response_rows(row.responses[initial_revision])
                        if parse_error is None:
                            prior_raw = parsed_initial
            expected_request = _request(client_proxy, row.catalogue, row.config.num_challenges,
                feedback=feedback, candidates=prior_raw if revision and prior_raw is not None else None)
            if request != expected_request:
                raise ValueError("recorded model request differs from the frozen protocol, catalogue or revision feedback")
    # Re-parse every model response from its recorded tool arguments. The
    # candidate-history hash alone is not accepted as provenance.
    replayed_rows_by_revision = {}
    response_errors = []
    if not rule_baseline:
        expected_response_errors = []
        for response, revision in zip(row.responses, row.response_revisions):
            rows, error = _response_rows(response)
            if error:
                response_errors.append(error)
                expected_response_errors.append({"stage": "response", "revision": revision, **error})
            elif rows is not None:
                replayed_rows_by_revision[revision] = rows
                if len(rows) != row.config.num_challenges:
                    expected_response_errors.append({"stage": "response", "revision": revision,
                        "kind": "CANDIDATE_COUNT_MISMATCH", "expected": row.config.num_challenges,
                        "actual": len(rows)})
        for revision, raw_rows in sorted(replayed_rows_by_revision.items()):
            if revision not in request_hashes:
                raise ValueError("candidate response has no corresponding model request")
            rebuilt_history.extend(_process_rows(raw_rows, index, row.config, revision, request_hashes[revision]))
        raw_reparsed = sum(map(len, replayed_rows_by_revision.values()))
        actual_response_errors = [e for e in row.transport_errors if e.get("stage") == "response"]
        if actual_response_errors != expected_response_errors:
            raise ValueError("recorded response errors differ from the raw provider responses")

        expected_revision_errors = []
        if any(revision == 1 for revision in row.response_revisions):
            selected_before_revision = {c.challenge_id: c for c in rebuilt_history if c.revision == 0}
            fresh_batch = not selected_before_revision
            for candidate in (c for c in rebuilt_history if c.revision == 1):
                prior = selected_before_revision.get(candidate.challenge_id)
                if prior is None:
                    if fresh_batch:
                        selected_before_revision[candidate.challenge_id] = candidate
                        continue
                    expected_revision_errors.append({"stage": "revision", "revision": 1,
                        "kind": "REVISION_PROTOCOL_VIOLATION", "message": "revision introduced a new challenge_id"})
                    continue
                if prior.final_status == "CERTIFIED" and prior.challenge_hash != candidate.challenge_hash:
                    expected_revision_errors.append({"stage": "revision", "revision": 1,
                        "kind": "REVISION_PROTOCOL_VIOLATION", "challenge_id": candidate.challenge_id,
                        "message": "revision changed a previously certified candidate; original retained"})
        actual_revision_errors = [e for e in row.transport_errors if e.get("stage") == "revision"]
        if actual_revision_errors != expected_revision_errors:
            raise ValueError("recorded revision-protocol errors differ from the revision candidates")

        no_response_revisions = set(row.request_revisions) - set(row.response_revisions)
        failure_revisions = [e.get("revision") for e in row.transport_errors if e.get("stage") in {"preflight", "transport"}]
        if sorted(failure_revisions) != sorted(no_response_revisions):
            raise ValueError("model request failures do not match requests with no recorded response")
    recorded_history = [c.model_dump(mode="json", exclude={"grounding_ms", "reference_ms"}) for c in row.candidate_history]
    recomputed_history = [c.model_dump(mode="json", exclude={"grounding_ms", "reference_ms"}) for c in rebuilt_history]
    if recorded_history != recomputed_history:
        raise ValueError("recorded candidate outcomes differ from deterministic grounding, native proof and replay")
    if len(row.responses) > len(row.requests) or any(r not in row.request_revisions for r in row.response_revisions):
        raise ValueError("response journal is detached from a model request")
    rebuilt_selected = {}
    for candidate in rebuilt_history:
        if candidate.revision == 0:
            rebuilt_selected[candidate.challenge_id] = candidate
        else:
            prior = rebuilt_selected.get(candidate.challenge_id)
            if prior is None:
                # A revision may recover a malformed response envelope, which
                # has no candidate objects in revision zero.
                if not any(c.revision == 0 for c in rebuilt_history):
                    rebuilt_selected[candidate.challenge_id] = candidate
            elif prior.final_status != "CERTIFIED":
                rebuilt_selected[candidate.challenge_id] = candidate
    recorded_selected = [c.model_dump(mode="json", exclude={"grounding_ms", "reference_ms"}) for c in row.candidates]
    recomputed_selected = [c.model_dump(mode="json", exclude={"grounding_ms", "reference_ms"}) for c in rebuilt_selected.values()]
    if recorded_selected != recomputed_selected:
        raise ValueError("selected candidate set differs from the recorded revision policy")

    expected = row.model_copy(deep=True)
    expected.candidates = list(rebuilt_selected.values())
    expected.candidate_history = rebuilt_history
    _set_accounting(expected)
    for label in ("metrics", "cost"):
        def without_timings(value):
            return {k: v for k, v in value.items() if not k.endswith("_ms")}
        if without_timings(getattr(row, label)) != without_timings(getattr(expected, label)):
            raise ValueError("recorded " + label + " differ from request/response and candidate evidence")
    expected_status = "COMPLETED_OFFLINE" if rule_baseline else _run_status(row.responses, row.transport_errors)
    if row.status != expected_status:
        raise ValueError("run status differs from recorded call evidence")

    for candidate in row.candidates:
        if candidate.challenge is None or candidate.grounding is None:
            continue
        for attempt in candidate.certified_attempts:
            if attempt.task is None:
                continue
            verify_task(attempt.task)
            if attempt.task.task_hash != attempt.task_hash or attempt.match_hash not in {m.binding_hash for m in candidate.grounding.matches}:
                raise ValueError("native certificate is detached from grounding binding")
            verified_tasks += 1
    return {"verified": True, "run_hash": row.run_hash, "index_hash": index.index_hash,
        "candidate_records": len(row.candidates), "native_certificates_replayed": verified_tasks,
        "request_records_replayed_offline": len(row.requests),
        "model_calls_replayed_offline": row.metrics.get("provider_call_attempts", row.metrics.get("total_llm_calls", len(row.requests))),
        "service_executions": 0,
        "raw_challenges_reparsed": raw_reparsed,
        "response_parse_errors_reproduced": len(response_errors)}


def _rule_challenges(count: int) -> list[dict]:
    """Deterministic grammar-combination baseline; it sees no private world."""
    one_profile = [{"alias": "buyer_a", "type": "profile"}, {"alias": "line_a", "type": "line"}]
    one_relation = [{"type": "profile_for_line", "left": "buyer_a", "right": "line_a"}]
    templates = [
        {"hypothesis": "Check the eligible case-opening and evidence-verification path.",
         "customer_request": "Please open a support case for the issue I reported.",
         "entity_variables": one_profile, "relationship_constraints": one_relation,
         "entity_constraints": [{"alias": "line_a", "predicate": "eligible_for_case", "value": True},
                                {"alias": "line_a", "predicate": "has_valid_evidence", "value": True}],
         "objectives": [{"kind": "created_object_completion", "target": "line_a", "collection": "cases",
                         "field": "status", "accepted_values": ["evidence_verified"]}],
         "required_customer_information": [{"profile_alias": "buyer_a", "fact_keys": ["line_id", "issue", "evidence_id"]}]},
        {"hypothesis": "Check whether an ineligible case request is refused under native policy.",
         "customer_request": "Please open a support case for the issue on this item.",
         "entity_variables": one_profile, "relationship_constraints": one_relation,
         "entity_constraints": [{"alias": "line_a", "predicate": "eligible_for_case", "value": False}],
         "objectives": [{"kind": "correct_refusal", "target": "line_a", "operation": "open_case"}],
         "required_customer_information": [{"profile_alias": "buyer_a", "fact_keys": ["order_id", "line_id", "issue"]}]},
        {"hypothesis": "Check mixed handling of an eligible and an ineligible line for one buyer.",
         "customer_request": "Please open a support case for each of the two items I reported.",
         "entity_variables": [{"alias": "buyer_a", "type": "profile"}, {"alias": "buyer_b", "type": "profile"},
                              {"alias": "line_a", "type": "line"}, {"alias": "line_b", "type": "line"}],
         "relationship_constraints": [{"type": "profile_for_line", "left": "buyer_a", "right": "line_a"},
             {"type": "profile_for_line", "left": "buyer_b", "right": "line_b"},
             {"type": "same_identity", "left": "buyer_a", "right": "buyer_b"},
             {"type": "distinct", "left": "line_a", "right": "line_b"}],
         "entity_constraints": [{"alias": "line_a", "predicate": "eligible_for_case", "value": True},
                                {"alias": "line_a", "predicate": "has_valid_evidence", "value": True},
                                {"alias": "line_b", "predicate": "eligible_for_case", "value": False}],
         "objectives": [{"kind": "created_object_completion", "target": "line_a", "collection": "cases",
                         "field": "status", "accepted_values": ["evidence_verified"]},
                        {"kind": "correct_refusal", "target": "line_b", "operation": "open_case"}],
         "required_customer_information": [{"profile_alias": "buyer_a", "fact_keys": ["line_id", "issue", "evidence_id"]},
             {"profile_alias": "buyer_b", "fact_keys": ["line_id", "issue"]}]},
        {"hypothesis": "Check approval and physical-return dependencies before an irreversible refund.",
         "customer_request": "Please resolve the damaged item with a refund if the required process allows it.",
         "entity_variables": one_profile, "relationship_constraints": one_relation,
         "entity_constraints": [{"alias": "line_a", "predicate": "eligible_for_case", "value": True},
             {"alias": "line_a", "predicate": "has_valid_evidence", "value": True},
             {"alias": "line_a", "predicate": "requires_approval", "value": True}],
         "objectives": [{"kind": "existing_object_completion", "target": "line_a", "collection": "lines",
                         "field": "status", "accepted_values": ["refunded"]}],
         "required_customer_information": [{"profile_alias": "buyer_a", "fact_keys": ["line_id", "issue", "evidence_id", "deadline"]}],
         "required_customer_preferences": [{"profile_alias": "buyer_a", "preference": "resolution", "accepted_values": ["refund"]}],
         "external_event_constraints": [{"target_alias": "line_a", "kind": "manager_review", "outcome": "approved"},
                                        {"target_alias": "line_a", "kind": "return_received", "outcome": "received"}]},
    ]
    return [{"challenge_id": f"rule-sample-{i + 1:02d}", "domain": "aftersales", **templates[i % len(templates)]}
            for i in range(count)]


def run_rule_baseline(index: PrivateWorldIndex, count: int = 5, *, partition: str = "train") -> BatchRun:
    config = BatchDesignerConfig(num_challenges=count, partition=partition,
        grounding=GroundingBudget(partition=partition, max_matches=1),
        construction=ConstructionBudget(max_nodes=4096, max_depth=16, max_paths=1))
    catalogue = public_domain_catalogue(index, "aftersales", partitions=(partition,))
    candidates = [_process_candidate(raw, index, config, 0) for raw in _rule_challenges(count)]
    model = ModelSettings(model="rule-combination-sampling-v1", base_url="http://127.0.0.1/v1", options={})
    run = BatchRun(contract_version=CONTRACT_VERSION, protocol="rule-combination-sampling-baseline-v1", status="COMPLETED_OFFLINE",
        domain="aftersales", partition=partition, index_hash=index.index_hash,
        pool_commitment=catalogue["pool_commitment"], catalogue=catalogue,
        catalogue_hash=catalogue["catalogue_hash"], model=model, config=config,
        candidates=candidates, candidate_history=candidates)
    _set_accounting(run)
    run.run_hash = run.computed_hash()
    return run


def _read_json(path):
    return strict_json(Path(path).read_text())


def main(argv=None):
    import argparse
    from usami_benchmark.contracts import WorldSpec
    from usami_benchmark.registry import default_spec
    from usami_benchmark.worlds import WorldPool, freeze
    from .world_index import build_world_index, load_world_index

    parser = argparse.ArgumentParser(description="World-grounded single-shot batch Designer")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-index", help="Freeze a finite domain world pool and build private index")
    build.add_argument("--domain", default="aftersales")
    build.add_argument("--seeds", default="0,1,2")
    build.add_argument("--partition", choices=("train", "validation", "test_demo", "heldout"), default="train")
    build.add_argument("--specs", help="Optional JSON array of registered WorldSpec values")
    build.add_argument("--output-dir", required=True)
    cat = sub.add_parser("catalogue", help="Render an ID-free public catalogue")
    cat.add_argument("--index", required=True); cat.add_argument("--domain", default="aftersales")
    cat.add_argument("--partition", choices=("train", "validation"), default="train"); cat.add_argument("--output", required=True)
    run = sub.add_parser("run", help="Run one structured batch call; use --online or an offline response file")
    run.add_argument("--index", required=True); run.add_argument("--output-dir", required=True)
    run.add_argument("--num-challenges", type=int, default=5)
    run.add_argument("--partition", choices=("train", "validation"), default="train")
    run.add_argument("--protocol", choices=(PROTOCOL, BATCH_REVISION_PROTOCOL), default=PROTOCOL)
    run.add_argument("--max-worlds", type=int, default=20); run.add_argument("--max-matches", type=int, default=10)
    run.add_argument("--max-bindings", type=int, default=50000); run.add_argument("--max-nodes", type=int, default=4096)
    run.add_argument("--max-depth", type=int, default=16); run.add_argument("--online", action="store_true")
    run.add_argument("--scripted-response", help="Offline JSON OpenAI-compatible response or {responses:[...]}")
    baseline = sub.add_parser("rule-baseline", help="Run deterministic grammar-combination baseline offline")
    baseline.add_argument("--index", required=True); baseline.add_argument("--output-dir", required=True)
    baseline.add_argument("--num-challenges", type=int, default=5)
    baseline.add_argument("--partition", choices=("train", "validation"), default="train")
    replay = sub.add_parser("replay", help="Rebuild all grounding matches and native certificates offline")
    replay.add_argument("--run-dir", required=True); replay.add_argument("--index", help="Defaults to run-dir/grounding-index.json")
    verify = sub.add_parser("verify-index", help="Regenerate frozen worlds and validate index commitments")
    verify.add_argument("--index", required=True)
    args = parser.parse_args(argv)

    if args.command == "build-index":
        specs = [WorldSpec.model_validate(v) for v in _read_json(args.specs)] if args.specs else [
            default_spec(args.domain, int(seed.strip()), args.partition) for seed in args.seeds.split(",") if seed.strip()]
        if not specs: parser.error("at least one seed/spec is required")
        pool = WorldPool([freeze(spec) for spec in specs])
        index = build_world_index(pool)
        root = Path(args.output_dir); root.mkdir(parents=True, exist_ok=False)
        atomic_json(root / "world-index.json", index.model_dump(mode="json"), immutable=True)
        atomic_json(root / "world-pool-manifest.json", _pool_manifest(index), immutable=True)
        print(json.dumps({"index_hash": index.index_hash, "pool_hash": index.pool_hash, "worlds": len(index.worlds), "output": str(root)}, sort_keys=True))
        return 0
    if args.command == "verify-index":
        index = load_world_index(args.index)
        print(json.dumps({"verified": True, "index_hash": index.index_hash, "worlds": len(index.worlds)}, sort_keys=True)); return 0
    if args.command == "catalogue":
        index = load_world_index(args.index)
        row = public_domain_catalogue(index, args.domain, partitions=(args.partition,))
        atomic_json(Path(args.output), row, immutable=True)
        print(json.dumps({"catalogue_hash": row["catalogue_hash"], "output": args.output}, sort_keys=True)); return 0
    if args.command == "rule-baseline":
        index = load_world_index(args.index)
        row = run_rule_baseline(index, args.num_challenges, partition=args.partition)
        save_run(row, index, args.output_dir)
        print(json.dumps({"status": row.status, "metrics": row.metrics, "cost": row.cost}, sort_keys=True)); return 0
    if args.command == "replay":
        print(json.dumps(replay_directory(args.run_dir, args.index), sort_keys=True)); return 0

    index = load_world_index(args.index)
    config = BatchDesignerConfig(protocol=args.protocol, num_challenges=args.num_challenges, partition=args.partition,
        grounding=GroundingBudget(max_worlds=args.max_worlds, max_matches=args.max_matches,
            max_bindings=args.max_bindings, partition=args.partition),
        construction=ConstructionBudget(max_nodes=args.max_nodes, max_depth=args.max_depth, max_paths=1))
    if args.online and args.scripted_response:
        parser.error("choose --online or --scripted-response, not both")
    if not args.online and not args.scripted_response:
        parser.error("run requires --online or --scripted-response")
    if args.scripted_response:
        payload = _read_json(args.scripted_response)
        if isinstance(payload, dict) and "responses" in payload:
            responses = payload["responses"]
            model = ModelSettings.model_validate(payload.get("model", {"model": "offline-scripted", "base_url": "http://127.0.0.1/v1"}))
        else:
            responses = [payload]; model = ModelSettings(model="offline-scripted", base_url="http://127.0.0.1/v1")
        class ScriptedClient:
            def __init__(self): self.settings, self.responses, self.calls = model, list(responses), 0
            def complete(self, request):
                if self.calls >= len(self.responses): raise ServiceError("SCRIPTED_RESPONSE_MISSING", "offline response file has no response for this call")
                value = self.responses[self.calls]; self.calls += 1; return value
        client = ScriptedClient()
    else:
        try:
            from usami.service.client import OpenAICompatibleClient
            client = OpenAICompatibleClient.from_env()
        except Exception as exc:
            failure_message = str(exc)
            class UnavailableClient:
                settings = ModelSettings(model="unavailable", base_url="http://127.0.0.1/v1", options={})
                available = False
                unavailable_reason = failure_message
                def complete(self, request): raise AssertionError("unavailable client must not dispatch a request")
            client = UnavailableClient()
    if Path(args.output_dir).exists():
        parser.error("output directory already exists; no model call was made")
    try:
        row = run_batch(index, client, config)
        save_run(row, index, args.output_dir)
    finally:
        if hasattr(client, "close"):
            client.close()
    print(json.dumps({"status": row.status, "metrics": row.metrics, "cost": row.cost, "output": args.output_dir}, sort_keys=True))
    return 0 if row.metrics["certified_challenges"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
