"""One frozen-model call proposes one declarative modification. No retry."""
from __future__ import annotations
import copy
import json
from typing import Literal, Optional, Tuple

from pydantic import Field, model_validator
from ..models import EpisodeEvidence, ModelCall, ModelSpec
from ..schema import FrozenModel, digest
from ..skills.library import LibrarySnapshot
from ..skills.models import SkillPatch, SkillRole
from .scope import DomainPolicy, ScopeResolution, proposal_json_schema, resolve_scope
from .evidence import InteractionEvidenceView, wire_view


CUSTOMER_OBJECTIVE = (
    "Improve the Customer Agent's ability to examine, challenge, and evaluate the Service Agent's "
    "business decision-making through open-ended, authorized, evidence-grounded exploration."
)
CUSTOMER_SELF_CHECKS = (
    "Role Check: Am I improving Customer examination rather than Service business execution? "
    "Goal Check: How does this skill test Service decision correctness? "
    "Evidence Check: Separate actual observations, post-run Oracle findings, and unverified hypotheses. "
    "Action Check: Supply concrete, authorized, reusable probe methods. "
    "Transfer Check: Generalize test conditions; do not memorize fixed resource IDs."
)


class ReflectionInput(FrozenModel):
    schema_version: Literal[1, 2] = 1
    generation: int = Field(ge=0, strict=True)
    role: SkillRole
    library: LibrarySnapshot
    evidence: Tuple[EpisodeEvidence, ...]
    allowed_tools: Tuple[str, ...] = ()
    domain_policy: Optional[DomainPolicy] = Field(default=None, exclude_if=lambda value: value is None)
    interaction_views: Optional[Tuple[InteractionEvidenceView, ...]] = Field(default=None, exclude_if=lambda value: value is None)
    # Validation evidence may enter Reflection only after the pilot explicitly
    # consumes it as training data. Held-out evidence remains forbidden.
    consumed_validation_evidence_ids: Tuple[str, ...] = Field(default=(), exclude_if=lambda value: not value)
    public_policy: str = ""
    boundaries: str = "Use only existing tools; exact Customer authorization precedes every write."

    @model_validator(mode="after")
    def training_evidence_only(self):
        if self.library.role != self.role:
            raise ValueError("reflection library role mismatch")
        if any(e.config.partition == "heldout" for e in self.evidence):
            raise ValueError("heldout evidence cannot enter Reflection")
        validation_ids = tuple(e.episode_id for e in self.evidence if e.config.partition == "validation")
        consumed = self.consumed_validation_evidence_ids
        if len(set(consumed)) != len(consumed):
            raise ValueError("consumed validation evidence IDs must be unique")
        if validation_ids != consumed:
            raise ValueError("validation evidence must be explicitly identified as consumed training data")
        if validation_ids and self.schema_version != 2:
            raise ValueError("consumed validation evidence requires ReflectionInput schema v2")
        if any(not e.replay.verified for e in self.evidence):
            raise ValueError("reflection requires replay-verified evidence")
        if self.interaction_views is not None:
            if len(self.interaction_views) != len(self.evidence) or any(
                (view.episode_id, view.evidence_hash, view.trace_sha256) != (row.episode_id, row.evidence_hash, row.trace.sha256)
                for view, row in zip(self.interaction_views, self.evidence)
            ):
                raise ValueError("interaction view is not bound to its verified episode and artifact")
            for view, row in zip(self.interaction_views, self.evidence):
                if (list(view.oracle_events) != [event.model_dump(mode="json") for event in row.events] or
                        list(view.confirmed_business_failures) != [failure.model_dump(mode="json") for failure in row.failures]):
                    raise ValueError("interaction view changed independent Oracle evidence")
                selected = set()
                for call in row.model_calls:
                    if call.role != "service" or not isinstance(call.response, dict):
                        continue
                    for choice in call.response.get("choices") or []:
                        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
                            continue
                        for tool in choice["message"].get("tool_calls") or []:
                            if (isinstance(tool, dict) and isinstance(tool.get("function"), dict) and
                                    all(isinstance(value, str) for value in (tool.get("id"), tool["function"].get("name"), tool["function"].get("arguments")))):
                                selected.add((tool.get("id"), tool["function"].get("name"), tool["function"].get("arguments")))
                if any((tool.call_id, tool.name, tool.arguments_raw) not in selected for tool in view.service_tools):
                    raise ValueError("interaction view changed Service-selected tool arguments")
                events = {event.event_id: event for event in row.events}
                if row.native_assessment is not None and len(events) != len(row.events):
                    raise ValueError("native assessment requires unique Oracle event references")
                for tool in view.service_tools:
                    event = events.get(tool.event_id)
                    if (row.native_assessment is not None and tool.event_id is not None
                            and (event is None or event.operation != tool.name)):
                        raise ValueError("interaction view changed a tool's Oracle linkage")
                    if (tool.oracle_classification, tool.oracle_operation_correct) != (
                        event.classification if event else None, event.operation_correct if event else None
                    ):
                        raise ValueError("interaction view changed a tool's Oracle linkage")
        return self


class ReflectionRecord(FrozenModel):
    schema_version: Literal[1] = 1
    input: ReflectionInput
    input_hash: str
    call: Optional[ModelCall] = None
    patch: Optional[SkillPatch] = None
    error: Optional[str] = None
    calls: int = Field(ge=0, le=1, strict=True)
    proposal_protocol_version: Optional[Literal[2]] = Field(default=None, exclude_if=lambda value: value is None)
    scope_resolution: Optional[ScopeResolution] = Field(default=None, exclude_if=lambda value: value is None)
    customer_goal_version: Optional[Literal["service-examination-v1"]] = Field(default=None, exclude_if=lambda value: value is None)
    prepared_interaction_views: Optional[Tuple[InteractionEvidenceView, ...]] = Field(default=None, exclude_if=lambda value: value is None)
    semantic_review_status: Optional[Literal["requires_experiment_review"]] = Field(default=None, exclude_if=lambda value: value is None)


def _parse(response, value):
    if not isinstance(response, dict):
        raise ValueError("Reflection response must be a JSON object")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Reflection requires exactly one model choice")
    if not isinstance(choices[0], dict) or not isinstance(choices[0].get("message"), dict):
        raise ValueError("Reflection choice must contain a message object")
    message = choices[0]["message"]
    calls = message.get("tool_calls", [])
    if calls:
        if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict) or not isinstance(calls[0].get("function"), dict) or calls[0]["function"].get("name") != "submit_skill_patch":
            raise ValueError("Reflection requires exactly one submit_skill_patch")
        raw = calls[0]["function"]["arguments"]
    else:
        raw = message.get("content")
    if not isinstance(raw, str):
        raise ValueError("Reflection must return one JSON patch")
    # Duplicate keys would make the intended modification ambiguous.
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field: " + key)
            result[key] = value
        return result
    data, resolution = resolve_scope(json.loads(raw, object_pairs_hook=unique), value.library, value.domain_policy)
    return SkillPatch.model_validate(data), resolution


class Reflector:
    def __init__(self, role, client, model: ModelSpec, max_input_chars=400000,
                 tool_choice: Optional[Literal["auto"]] = None):
        self.role, self.client = role, client
        self.model = ModelSpec.model_validate(model.model_dump(mode="json"))
        self.model_hash = digest(self.model)
        self.max_input_chars = max_input_chars
        # Thinking-mode Qwen endpoints reject forced function-name tool_choice.
        # Keep the historical forced behavior as the default; pilots may opt in
        # to provider-compatible auto selection and still validate the response.
        self.tool_choice = tool_choice

    def reflect(self, value: ReflectionInput) -> ReflectionRecord:
        value = ReflectionInput.model_validate(value.model_dump(mode="json"))
        model = ModelSpec.model_validate(self.model.model_dump(mode="json"))
        if digest(model) != self.model_hash:
            raise ValueError("Reflector model changed after construction")
        if value.role != self.role:
            raise ValueError("Reflector role mismatch")
        input_hash = digest(value.model_dump(mode="json"))
        record_fields = dict(input=value, input_hash=input_hash, proposal_protocol_version=2)
        payload = value.model_dump(mode="json")
        native_assessment = any(e.native_assessment is not None for e in value.evidence)
        allowed_evidence_ids = [e.episode_id for e in value.evidence]
        if native_assessment:
            payload["allowed_evidence_ids"] = allowed_evidence_ids
        # Keep complete input in the record, but send compact actual messages and
        # verifier facts rather than duplicating every previous growing request.
        payload["evidence"] = [dict(episode_id=e.episode_id, config=e.config.model_dump(mode="json"),
                                    customer_snapshot_hash=e.customer_snapshot_hash, service_snapshot_hash=e.service_snapshot_hash,
                                    events=[row.model_dump(mode="json") for row in e.events],
                                    failures=[row.model_dump(mode="json") for row in e.failures],
                                    termination=e.termination, task_success=e.task_success,
                                    legal_task_success=e.legal_task_success, correct_refusal=e.correct_refusal,
                                    messages=[dict(role=c.role, request_messages=c.base_request.get("messages", []),
                                                   response=c.response, error=c.error) for c in e.model_calls]) for e in value.evidence]
        for row,e in zip(payload['evidence'],value.evidence):
            if e.task_decisions is not None:row['task_decisions']=[d.model_dump(mode='json') for d in e.task_decisions]
            if e.native_assessment is not None:
                row['native_assessment'] = e.native_assessment
        if self.role == "customer":
            views = value.interaction_views if value.interaction_views is not None else tuple(wire_view(row) for row in value.evidence)
            record_fields.update(customer_goal_version="service-examination-v1", prepared_interaction_views=views,
                                 semantic_review_status="requires_experiment_review")
            # Full raw inputs/outputs remain in the immutable input record. The
            # LLM gets separated committed actions and results, not repeated
            # actor system prompts or private provider reasoning_content.
            for row, view in zip(payload["evidence"], views):
                row.pop("messages")
                row["interaction"] = view.model_dump(mode="json")
            payload.pop("interaction_views", None)
        elif value.interaction_views is not None:
            # Same verified native projection for Service learning: actual
            # customer declarations, selected tools/results and Oracle facts.
            # Complete original wire records remain in input; no new judge.
            record_fields["prepared_interaction_views"] = value.interaction_views
            for row, view in zip(payload["evidence"], value.interaction_views):
                row.pop("messages")
                row["interaction"] = view.model_dump(mode="json")
            payload.pop("interaction_views", None)
        content = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if len(content) > self.max_input_chars:
            return ReflectionRecord(**record_fields, error="reflection_input_budget_exceeded", calls=0)
        objective = (
            CUSTOMER_OBJECTIVE + " "
            "Customer requests are probes; Service tool choices/parameters/order are the behavior under examination. "
            "Service aims to complete legal business requests correctly; Customer aims to test those decisions. "
            "Learn from unsuccessful trials by designing better tests, not by teaching Customer to operate the business system. "
            "RESOURCE_NOT_FOUND alone establishes neither Service fault nor Customer fault. Examine the exact Customer request, "
            "write intentions, Service-selected arguments and previous visible results. Unknown attribution must remain unknown. "
            "Oracle findings are independent post-run evidence; they were not Customer-visible during execution. "
            "Oracle operation_correct evaluates Engine policy compliance, not whether Service chose the best resource ID. "
            "Allow pending hypotheses and NOOP; do not require finding a vulnerability. "
            "A hypothesis is not a confirmed failure. Never turn a correct denial into a confirmed exploit. "
            "Map Testing Hypothesis to skill.hypotheses; Evidence to rationale/evidence_ids; Probe Strategy to exploration; "
            "Outcome Interpretation and conditional Next Exploration to feedback; budget/uncertainty limits to stop. "
            "If Service handles a probe correctly, specify how to vary a prerequisite, ambiguity or boundary legally next time. "
            + CUSTOMER_SELF_CHECKS
            if self.role == "customer" else
            "Repair verified decision failures while preserving legal completion and correct refusal. "
            "When evidence offers no justified change, emit NOOP. Never invent a verified failure."
        )
        if self.role == "service" and native_assessment:
            objective += (
                " Native assessments separate business goal completion, process errors, safety and interaction reliability. "
                "For customer_visible_context entries encoded as previous-request-prefix-v1, reconstruct the exact "
                "Service request by retaining unchanged_prefix_messages messages from the preceding reconstructed "
                "request, then appending request_messages. Source pointers identify the full original requests. "
                "Their evidence_sources point to the replay-verified native records. An empty failures array means no "
                "confirmed business violation in that field; it does not mean the process was error-free. "
                "Oracle operation_correct evaluates Engine enforcement, not Service decision quality: a correctly "
                "blocked invalid attempt may still reveal an avoidable Service error. Recovery does not erase an "
                "earlier error. Examine the exact arguments, preceding Service-visible observations and public tool "
                "contracts before attributing a mistake. An unmet goal or missing Customer runtime consent does not "
                "by itself prove Service fault. Keep uncertain attribution explicit. Propose a reusable Skill only "
                "when the actual evidence justifies it; otherwise NOOP remains valid."
            )
        if self.role == "customer" and value.interaction_views and any(
                isinstance(context, dict) and context.get("examiner_protocol") == "usami-examiner-v1"
                for view in value.interaction_views for context in view.customer_visible_context):
            objective += (" In examiner episodes, Customer globally constructs scenes before execution. "
                          "Learn reusable scene-construction and test-design strategies under fixed native invariants, "
                          "then freeze the world, execution identity, goal and exact capabilities. "
                          "Construction authority never changes Service permissions or the Oracle.")
        instruction = (
            "You are the " + self.role + " Skill Reflector. Frozen weights; one Reflection and one Patch only. "
            + objective + " Return exactly one ADD, UPDATE, REMOVE or NOOP patch via submit_skill_patch. "
            "ADD and UPDATE MUST include the complete skill object, including id, version, type, trigger and every required step array. "
            "Use the supplied base_snapshot_hash exactly. ADD starts at version 1; UPDATE increments once. "
            "Proposal protocol v2: omit scope.domains and content_hash; the program manages them. "
            "UPDATE inherits the original domains and, when omitted, tool requirements. "
            "ADD uses domain_policy.add_domains; domain scope changes require a separate experiment. "
            "Scope tools may only name supplied tools. Trigger has keywords and/or intents arrays. "
            "Skills are declarative text, never code, authority, hidden Oracle access or specific resource IDs. "
            "Runtime retrieval is deterministic: domain must match, scope.tools must be a subset of available tools, "
            "and a skill is retrieved only when a keyword is a case-insensitive substring of the initial public task text "
            "or a trigger intent exactly matches a supplied intent. Choose triggers from customer-visible natural language "
            "that is present when the skill must first apply; internal tool names do not match unless the customer text "
            "contains them. Do not rely on a future tool result to activate a skill. "
            "Cite only supplied episode IDs in evidence_ids. No retries or multiple alternatives."
        )
        parameters = proposal_json_schema(self.role)
        if native_assessment:
            citations = parameters["properties"]["evidence_ids"]
            citations["description"] = (
                "Cite whole Episode IDs from allowed_evidence_ids only. Copy an exact enum value; "
                "Oracle event IDs, source paths and content hashes are not Episode IDs. "
                "Describe event-level support in rationale."
            )
            citations["items"]["enum"] = allowed_evidence_ids
            instruction += (
                " evidence_ids must use exact values from allowed_evidence_ids and the tool schema enum. "
                "An Oracle event reference identifies a step inside an Episode; put it in rationale, "
                "not evidence_ids. Do not append tool/event suffixes to an Episode ID."
            )
        request = {"model": self.model.model, **copy.deepcopy(self.model.options),
                   "messages": [{"role": "system", "content": instruction}, {"role": "user", "content": content}],
                   "tools": [{"type": "function", "function": {"name": "submit_skill_patch", "description": "Propose exactly one declarative Skill modification.",
                               "parameters": parameters}}],
                   "tool_choice": self.tool_choice or {"type": "function", "function": {"name": "submit_skill_patch"}}}
        pending = dict(id="g" + str(value.generation) + "." + self.role + ".reflection", role=self.role + "_reflector",
                       model=self.model, base_request=copy.deepcopy(request), request=copy.deepcopy(request))
        try:
            response = self.client.complete(copy.deepcopy(request))
        except Exception as exc:
            call = ModelCall(**pending, response=getattr(exc, "response", None),
                             error={"type": getattr(exc, "original_type", type(exc).__name__),
                                    "message": str(exc), "code": getattr(exc, "code", "client_error")})
            return ReflectionRecord(**record_fields, call=call, error="reflection_client_error", calls=1)
        call = ModelCall(**pending, response=copy.deepcopy(response))
        try:
            patch, resolution = _parse(response, value)
            if patch.role != self.role or patch.base_snapshot_hash != value.library.snapshot_hash:
                raise ValueError("patch targets a different role or parent snapshot")
            unavailable = set(patch.evidence_ids) - set(allowed_evidence_ids)
            if unavailable:
                if native_assessment and unavailable.issubset({event.event_id for e in value.evidence for event in e.events}):
                    raise ValueError("evidence_ids requires supplied Episode IDs; Oracle event IDs were supplied instead. Use allowed_evidence_ids")
                raise ValueError("patch cites unavailable or heldout evidence")
        except (ValueError, TypeError, KeyError) as exc:
            return ReflectionRecord(**record_fields, call=call, error="invalid_reflection: " + str(exc), calls=1)
        return ReflectionRecord(**record_fields, call=call, patch=patch, scope_resolution=resolution, calls=1)


class CustomerReflector(Reflector):
    def __init__(self, client, model, **kwargs):
        super().__init__("customer", client, model, **kwargs)


class ServiceReflector(Reflector):
    def __init__(self, client, model, **kwargs):
        super().__init__("service", client, model, **kwargs)
