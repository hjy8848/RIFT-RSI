"""Versioned, bounded read/explore/submit protocol for frozen dialogue worlds.

Only native frame declarations are compiler-generated. Objectives, customer
choices and event selection remain Designer decisions; native certification is
unchanged. Recorded exchanges replay through this same deterministic program.
"""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, ValidationError, model_validator

from usami.service.models import ModelSettings, ServiceError, strict_json
from usami.service.runner import completion_message
from usami_dialogue.composition import _contract
from usami_dialogue.scope import load_profiles
from usami_dialogue.tasks import CreatedObjectGoal, Goal, InformationRequirement, PreferenceRequirement
from usami_harness.models import Model, digest
from vast_coevo.schema import atomic_json

from .contracts import TRACKS
from .public_policy import catalogue
from .splits import verified_world
from .tasks import CertifiedTask, ConstructionBudget, OpenDialogueComposition, TaskSelection, compile_task


LEGACY_PROTOCOL = "frozen-world-designer-exploration-v2"
PROTOCOL_V3 = "frozen-world-designer-exploration-v3"
PROTOCOL = "frozen-world-designer-exploration-v4"


class DesignerBudget(Model):
    max_exploration_calls: int = Field(default=5, ge=0, le=12)
    max_revisions: int = Field(default=3, ge=0, le=3)
    max_queries: int = Field(default=20, ge=1, le=48)
    initial_exploration_calls: Optional[int] = Field(default=None, ge=1, le=12, exclude_if=lambda value: value is None)
    actionable_feedback: bool = Field(default=False, exclude_if=lambda value: value is False)
    include_related_objects: bool = Field(default=False, exclude_if=lambda value: value is False)
    include_profile_events: bool = Field(default=False, exclude_if=lambda value: value is False)
    confirmation_guidance: bool = Field(default=False, exclude_if=lambda value: value is False)


class DialogueDesignerProposal(Model):
    """v4 proposal: goals may target existing or newly-created objects."""
    profile_ids: list[str] = Field(min_length=1, max_length=8)
    customer_request: str = Field(min_length=1, max_length=6000)
    goals: list[Goal] = Field(default_factory=list, max_length=32,
                               description=("Machine-checkable desired final states for existing, customer-scoped "
                                            "business objects and fields allowed by their native mutable frame. "
                                            "Do not target event records, evidence, immutable facts, or a state "
                                            "that already holds."))
    created_goals: list[CreatedObjectGoal] = Field(
        default_factory=list, max_length=32, exclude_if=lambda value: not value,
        description=("Goals for exactly one new object created through a selected Customer's native operation. "
                     "Name the selected profile, native collection, grounded immutable selector fields and desired "
                     "public field value; never provide a generated object ID."))
    refusal_operations: list[dict] = Field(
        default_factory=list, max_length=16,
        description=("Exact canonical business writes Service should correctly refuse. Include every native parameter; "
                     "never guess IDs or fields. A Customer-authorized operation is not automatically a refusal."))
    information_requirements: list[InformationRequirement] = Field(default_factory=list, max_length=64)
    preference_requirements: list[PreferenceRequirement] = Field(default_factory=list, max_length=64)
    confirmation_requirements: list[dict] = Field(
        default_factory=list,
        max_length=16,
        description=(
            "Optional evaluator assertion, not authorization; usually omit. Only include an exact "
            "canonical write copied from the selected Customer profile's customer_operations. "
            "Never guess operation parameters or invent case, quote, or resource IDs. Omitting this "
            "does not bypass live Customer consent. Supplied operations remain subject to native scope checks."
        ),
    )
    external_schedule: list[str] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def require_objective(self):
        if not self.goals and not self.created_goals and not self.refusal_operations:
            raise ValueError(
                "proposal needs at least one machine-checkable completion goal or exact correct-refusal operation; "
                "information and preference requirements alone are not task objectives"
            )
        return self


class _V3DialogueDesignerProposal(Model):
    """Frozen schema for v3 request and journal replay compatibility."""
    model_config = {"title": "DialogueDesignerProposal"}
    profile_ids: list[str] = Field(min_length=1, max_length=8)
    customer_request: str = Field(min_length=1, max_length=6000)
    goals: list[Goal] = Field(default_factory=list, max_length=32,
                               description=("Machine-checkable desired final states for existing, customer-scoped "
                                            "business objects and fields allowed by their native mutable frame. "
                                            "Do not target event records, evidence, immutable facts, or a state "
                                            "that already holds."))
    refusal_operations: list[dict] = Field(default_factory=list, max_length=16,
        description=("Exact canonical business writes Service should correctly refuse. Include every native parameter; "
                     "never guess IDs or fields. A Customer-authorized operation is not automatically a refusal."))
    information_requirements: list[InformationRequirement] = Field(default_factory=list, max_length=64)
    preference_requirements: list[PreferenceRequirement] = Field(default_factory=list, max_length=64)
    confirmation_requirements: list[dict] = Field(default_factory=list, max_length=16,
        description=("Optional evaluator assertion, not authorization; usually omit. Only include an exact "
                     "canonical write copied from the selected Customer profile's customer_operations. "
                     "Never guess operation parameters or invent case, quote, or resource IDs. Omitting this "
                     "does not bypass live Customer consent. Supplied operations remain subject to native scope checks."))
    external_schedule: list[str] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def require_objective(self):
        if not self.goals and not self.refusal_operations:
            raise ValueError("proposal needs at least one machine-checkable completion goal or exact correct-refusal operation; "
                             "information and preference requirements alone are not task objectives")
        return self


class _LegacyDialogueDesignerProposal(Model):
    model_config = {"title": "DialogueDesignerProposal"}
    profile_ids: list[str] = Field(min_length=1, max_length=8)
    customer_request: str = Field(min_length=1, max_length=6000)
    goals: list[Goal] = Field(default_factory=list, max_length=32)
    refusal_operations: list[dict] = Field(default_factory=list, max_length=16)
    information_requirements: list[InformationRequirement] = Field(default_factory=list, max_length=64)
    preference_requirements: list[PreferenceRequirement] = Field(default_factory=list, max_length=64)
    confirmation_requirements: list[dict] = Field(default_factory=list, max_length=16)
    external_schedule: list[str] = Field(default_factory=list, max_length=64)


class ReadQuery(Model):
    action: Literal["list_profiles", "read_profile", "read_objects", "read_rules"]
    identity: Optional[str] = None
    profile_id: Optional[str] = None
    collection: Optional[str] = None
    ids: list[str] = Field(default_factory=list, max_length=8)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=4, ge=1, le=8)


def _adapter_world(frozen):
    if TRACKS[frozen.spec.domain] != "B":
        raise ValueError("frozen-world exploration requires a native dialogue domain")
    adapter = importlib.import_module("usami_" + frozen.spec.domain + ".adapter")
    return adapter, adapter.World.model_validate(frozen.world)


def field_enums(adapter):
    schema = adapter.World.model_json_schema()
    result = {}
    for collection, raw in schema["properties"].items():
        additional = raw.get("additionalProperties", {})
        reference = additional.get("$ref", "") if isinstance(additional, dict) else ""
        entity = schema.get("$defs", {}).get(reference.rsplit("/", 1)[-1], {})
        values = {name: spec["enum"] for name, spec in entity.get("properties", {}).items() if "enum" in spec}
        if values:
            result[collection] = values
    return result


def exploration_view(frozen, *, include_context=True, protocol=PROTOCOL):
    """Compact catalogue, never the complete world or a selected task."""
    frozen = verified_world(frozen)
    _, world = _adapter_world(frozen)
    identities = {}
    for profile in world.profiles.values():
        identities[profile.identity] = identities.get(profile.identity, 0) + 1
    if protocol not in {LEGACY_PROTOCOL, PROTOCOL_V3, PROTOCOL}:
        raise ValueError("unknown frozen-world Designer exploration protocol")
    result = {"protocol": protocol, "world_hash": frozen.world_hash,
            "frozen_registration_hash": frozen.registration_hash, "domain": frozen.spec.domain,
            "rules_version": frozen.spec.rules_version,
            "identities": [{"identity": key, "profile_count": value} for key, value in sorted(identities.items())],
            "collections": {key: len(value) for key, value in frozen.world.items() if isinstance(value, dict)}}
    if include_context:
        result["clock"] = frozen.world["clock"]
    return result


def inspect_world(frozen, raw, *, include_related_objects=False, include_profile_events=True, protocol=PROTOCOL):
    """Read from registered world data only; no Engine writes or Reference calls."""
    query = ReadQuery.model_validate(raw)
    adapter, world = _adapter_world(frozen)
    if query.action == "read_rules":
        policy = catalogue(frozen.spec.domain, frozen.spec.rules_version)
        if protocol == LEGACY_PROTOCOL and frozen.spec.domain == "aftersales":
            # v2 journals predate the case-window wording clarification. Freeze
            # the model-visible prose, not the native eligibility decisions.
            policy["business_rules"] = [
                "Authenticate the order buyer and merchant/order/line scope. Ask for the actual order, issue, evidence and relevant customer deadline before committing a remedy.",
                "Delivery, product returnability, native return window, verified evidence, settlement state, merchant approval limit and replacement stock constrain refund, exchange and reship.",
                "Return-required refunds need actual receipt; pickup, current remedy choice and irreversible settlement require the applicable exact customer confirmation. Use the original payment channel and remaining paid value.",
            ]
            policy["catalogue_hash"] = digest({k: v for k, v in policy.items() if k != "catalogue_hash"})
        return {"policy": policy,
                "native_field_enums": field_enums(adapter),
                "operation_schema": adapter.Operation.model_json_schema()}
    if query.action == "list_profiles":
        rows = [{"profile_id": key, "identity": private.identity}
                for key, private in sorted(world.profiles.items())
                if query.identity is None or private.identity == query.identity]
        return {"items": rows[query.offset:query.offset + query.limit], "total": len(rows),
                "offset": query.offset, "limit": query.limit}
    if query.action == "read_profile":
        if query.profile_id not in world.profiles:
            raise ValueError("profile_id does not exist in the frozen world")
        private = adapter.private_state(world, query.profile_id)
        related = {}
        for rule in _contract(adapter, world, private):
            collection = frozen.world.get(rule.collection, {})
            rows = {key: row for key, row in collection.items() if isinstance(row, dict) and
                    (key == rule.selector_value if rule.selector_field == "$key"
                     else row.get(rule.selector_field) == rule.selector_value)}
            if rows:
                related.setdefault(rule.collection, {}).update(rows)
        if include_related_objects:
            # Mutation frames omit immutable order dates, evidence and owners.
            # Follow existing entity references from this profile, bounded to
            # 24 records / two edges; do not decide policy or choose a target.
            def references(value):
                if isinstance(value, str):
                    yield value
                elif isinstance(value, dict):
                    for item in value.values():
                        yield from references(item)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        yield from references(item)
            index = {}
            for collection, entities in frozen.world.items():
                if not isinstance(entities, dict):
                    continue
                for key, entity in entities.items():
                    if isinstance(entity, dict) and entity.get("id") == key:
                        index.setdefault(key, []).append((collection, entity))
            pending = set(private.protected_objects) | set(references(private.facts))
            seen = set()
            count = 0
            for _ in range(3):
                following = set()
                for key in sorted(pending - seen):
                    seen.add(key)
                    for collection, entity in index.get(key, []):
                        if count >= 24:
                            break
                        related.setdefault(collection, {})[key] = copy.deepcopy(entity)
                        following.update(references(entity))
                        count += 1
                pending = following
        def event_matches(event):
            if any(value in private.protected_objects for value in event.values()):
                return True
            # Native After-sales events bind to a line_id, while protected
            # objects contain the order_id. Match existing identifier fields
            # only; shared issue/outcome text does not establish a relation.
            return include_profile_events and any(
                name.endswith("_id") and isinstance(value, str) and event.get(name) == value
                for name, value in private.facts.items())

        result={"profile_id": query.profile_id, "identity": private.identity,
                "facts": private.facts, "preference_space": private.preference_space,
                "initial_preferences": private.initial_preferences,
                "confirmable_actions": list(private.confirmable_actions),
                "customer_operations": adapter.candidate_intents(world, private),
                "related_objects": related,
                "registered_external_events": [row.model_dump(mode="json") for row in world.external_events.values()
                    if event_matches(row.model_dump(mode="json"))]}
        if protocol==PROTOCOL:
            result["native_frame_contract"]=[frame.model_dump(mode="json") for frame in _contract(adapter,world,private)]
        return result
    collection = frozen.world.get(query.collection)
    if not isinstance(collection, dict):
        raise ValueError("collection must name a registered frozen collection")
    if query.ids:
        if any(key not in collection for key in query.ids):
            raise ValueError("requested object does not exist in the frozen collection")
        return {"items": {key: copy.deepcopy(collection[key]) for key in query.ids}}
    keys = sorted(collection)
    return {"items": {key: copy.deepcopy(collection[key]) for key in keys[query.offset:query.offset + query.limit]},
            "total": len(keys), "offset": query.offset, "limit": query.limit}


def compile_proposal(frozen, proposal, construction_budget, *, protocol=PROTOCOL):
    """Derive native scope without choosing, repairing or replacing objectives."""
    proposal_model = {LEGACY_PROTOCOL: _LegacyDialogueDesignerProposal,
                      PROTOCOL_V3: _V3DialogueDesignerProposal,
                      PROTOCOL: DialogueDesignerProposal}.get(protocol)
    if proposal_model is None:
        raise ValueError("unknown frozen-world Designer exploration protocol")
    proposal = proposal_model.model_validate(proposal)
    adapter, world = _adapter_world(frozen)
    profiles = load_profiles(adapter, world, proposal.profile_ids[0], proposal.profile_ids)
    frames = {}
    for _, private in profiles:
        for frame in _contract(adapter, world, private):
            frames.setdefault(digest(frame), frame)
    if protocol in {PROTOCOL_V3, PROTOCOL}:
        _validate_goal_targets(world, proposal.goals, frames)
    enums = field_enums(adapter)
    for goal in proposal.goals:
        values = enums.get(goal.collection, {}).get(goal.field)
        if values is not None and any(value not in values for value in goal.accepted_values):
            raise ValueError(f"invalid values for {goal.collection}.{goal.field}; native vocabulary is {values}")
    created_goals = getattr(proposal, "created_goals", [])
    composition_protocol = "open-dialogue-composition-v3" if created_goals else "open-dialogue-composition-v2"
    composition = OpenDialogueComposition(protocol=composition_protocol,
        **proposal.model_dump(mode="json"), frames=list(frames.values()))
    selection = TaskSelection(kind="dialogue", family="open_composition_v1",
                              profile_id=proposal.profile_ids[0], composition=composition)
    return selection, compile_task(frozen, selection, construction_budget)


def _validate_goal_targets(world, goals, frames):
    """Return actionable feedback before Reference search for unsupported targets."""
    objects = world.model_dump(mode="json")
    allowed = {}
    for frame in frames.values():
        rows = objects.get(frame.collection)
        if not isinstance(rows, dict):
            continue
        for key, row in rows.items():
            if not isinstance(row, dict):
                continue
            matches = (str(key) == str(frame.selector_value) if frame.selector_field == "$key"
                       else row.get(frame.selector_field) == frame.selector_value)
            if matches:
                allowed.setdefault((frame.collection, str(key)), set()).update(frame.mutable_fields)
    for goal in goals:
        fields = allowed.get((goal.collection, goal.entity))
        if fields is None:
            raise ValueError(
                f"goal target {goal.collection}.{goal.entity} is not an existing object in the selected "
                "Customer profiles' native task scope; external events belong in external_schedule, not goals"
            )
        if goal.field not in fields:
            raise ValueError(
                f"goal field {goal.collection}.{goal.entity}.{goal.field} is not mutable under the selected "
                f"Customer profiles' native frame; available fields for this object are {sorted(fields)}"
            )


POLICY_V3 = (
    "You are the autonomous Benchmark Task Designer in a read-only frozen business world. "
    "Use inspect_frozen_world to choose whom and what to investigate; you have a bounded exploration phase. "
    "Read the business rules and field vocabulary, inspect profiles and actual related objects. "
    "Select useful business challenges yourself, not a task supplied by the harness. Every submission MUST include "
    "at least one machine-checkable completion goal OR one exact correct-refusal operation; information and "
    "preference requirements alone do not define a task. A completion goal is the desired final business state: "
    "target an existing object and a field its native Customer frame permits changing. Do not use event records, "
    "evidence, immutable facts, already-true predicates, or registered event outcomes as completion goals; select "
    "external events separately in external_schedule. A refusal operation must be a complete canonical write with "
    "all native params and must be an operation the Customer requests but native business rules/state require the "
    "Service to refuse. customer_operations establishes Customer scope and parameter grounding; its presence does "
    "not mean the operation should be refused. Check read_rules and observed state before classifying it. Never "
    "guess missing case/quote IDs or params; if no observed operation supports a refusal, choose a grounded "
    "completion goal or inspect another profile. "
    "You may combine several "
    "profiles only for the same authenticated identity. Do not invent facts, permissions, entities, events, "
    "proofs or expected refusal codes. Current v3 completion goals address existing objects; created-object "
    "predicates are not supported yet. The compiler derives native frames; never write frames or selector fields. "
    "An external event occurs only if you select its registered ID in external_schedule. "
    "Call ready_to_submit when prepared. The next phase forces submit_frozen_world_task. "
    "On rejection, correct your own proposal using the feedback; do not claim search failure proves impossibility. "
    "Keep replies concise and use the tools, rather than writing an essay."
)

POLICY = (
    "You are the autonomous Benchmark Task Designer in a read-only frozen business world. "
    "Use inspect_frozen_world to choose whom and what to investigate; you have a bounded exploration phase. "
    "Read the business rules and field vocabulary, inspect profiles and actual related objects. "
    "Select useful business challenges yourself, not a task supplied by the harness. Every submission MUST include "
    "at least one existing-object completion goal, one created-object completion goal, or one exact correct-refusal "
    "operation. Information and preference requirements alone are not task objectives. Existing-object goals target "
    "an existing selected-Customer object and a mutable field. A created-object goal names a selected profile, a "
    "native create-enabled collection, immutable selector fields/values grounded in that profile's observed facts or "
    "customer_operations, and a public mutable field's accepted final value. The initial world must have no object "
    "matching those selectors; Reference requires exactly one matching object after native execution. Do not guess or "
    "supply the generated object's ID. Never use event records, evidence, private fields, immutable facts, or a state "
    "that already holds as a completion goal; select external events separately in external_schedule. A refusal "
    "operation must be a complete canonical write that the Customer requests but native rules require Service to "
    "refuse. customer_operations grounds Customer authority but does not mean an operation should be refused. Check "
    "read_rules and observed state before classifying it. Never guess missing operation IDs or params. You may combine "
    "profiles only for the same authenticated identity. Do not invent facts, permissions, entities, events, proofs, "
    "or expected refusal codes. The compiler derives native frames; never write frames or selector fields. An external "
    "event occurs only if you select its registered ID in external_schedule. Call ready_to_submit when prepared. On "
    "rejection, correct your own proposal using the feedback; a bounded search failure is not proof of impossibility. "
    "Keep replies concise and use the tools."
)

_LEGACY_POLICY = (
    "You are the autonomous Benchmark Task Designer in a read-only frozen business world. "
    "Use inspect_frozen_world to choose whom and what to investigate; you have a bounded exploration phase. "
    "Read the business rules and field vocabulary, inspect profiles and actual related objects. "
    "Select useful business challenges yourself, not a task supplied by the harness. You may combine several "
    "profiles only for the same authenticated identity. Do not invent facts, permissions, entities, events, "
    "proofs or expected refusal codes. Current v2 completion goals address existing objects; created-object "
    "predicates are not supported yet. The compiler derives native frames; never write frames or selector fields. "
    "An external event occurs only if you select its registered ID in external_schedule. "
    "Call ready_to_submit when prepared. The next phase forces submit_frozen_world_task. "
    "On rejection, correct your own proposal using the feedback; do not claim search failure proves impossibility. "
    "Keep replies concise and use the tools, rather than writing an essay."
)


def _tool(name, description, schema):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": schema}}


def _native_write_schemas(adapter):
    """Build strict schemas for canonical native writes used in proposals.

    Operation-shaped proposal values must not inherit ``dict``'s permissive
    JSON schema. Otherwise a model can put ``line_id`` or a guessed ``case_id``
    at the wrong level and discover the mistake only after submission.
    """
    alternatives = []
    for operation, params_model in adapter.Operation.parameters.items():
        if operation in adapter.Operation.queries:
            continue
        params_schema = params_model.model_json_schema()
        # Native operation dumps include defaults, and the composer checks
        # canonical operations, so require each declared parameter.
        if params_model.model_fields:
            params_schema["required"] = list(params_model.model_fields)
        alternatives.append({
            "type": "object",
            "properties": {
                "name": {"enum": [operation]},
                "scope_id": {"type": "string", "minLength": 1},
                "resource_id": {"type": "string", "minLength": 1},
                "params": params_schema,
            },
            "required": ["name", "scope_id", "resource_id", "params"],
            "additionalProperties": False,
        })
    return alternatives


_OPTIONAL_CONFIRMATION_GUIDANCE = (
    "confirmation_requirements is optional and normally omitted: it is an evaluator assertion, not Customer "
    "authorization. Include one only if you can copy the exact canonical write and all parameters from the "
    "selected profile's customer_operations; never guess case_id, quote_id, line_id, or parameters. Omitting "
    "it does not waive live Customer consent. Any supplied operation is still checked against native identity, "
    "object scope, and business rules."
)


def _proposal_schema(frozen=None, *, protocol=PROTOCOL):
    model = _V3DialogueDesignerProposal if protocol == PROTOCOL_V3 else DialogueDesignerProposal
    schema = model.model_json_schema()
    goal_definition = schema.get("$defs", {}).get("Goal", {})
    goal_definition["description"] = (
        "Desired final business state on an existing selected-Customer object. The target field must be mutable "
        "under the native Customer frame; event records, evidence and immutable facts are not completion goals."
    )
    for name, description in {
        "collection": "Native business-object collection covered by the selected Customer frame.",
        "entity": "Existing object ID covered by the selected Customer frame.",
        "field": "Mutable business-state field covered by the selected Customer frame.",
        "accepted_values": "Desired final values, different from the frozen initial value.",
    }.items():
        if name in goal_definition.get("properties", {}):
            goal_definition["properties"][name]["description"] = description
    created_definition = schema.get("$defs", {}).get("CreatedObjectGoal", {})
    if created_definition:
        created_definition["description"] = (
            "Target exactly one new native object without providing its generated ID. Selectors must be "
            "grounded in the named Customer profile, absent from the initial world, and covered by a native "
            "create-enabled frame."
        )
        for name, description in {
            "profile_id": "Selected frozen Customer profile whose facts ground the object selectors.",
            "collection": "Existing native object collection with a selected Customer create-enabled frame.",
            "selectors": "Immutable public field/value pairs from the selected Customer facts or customer_operations.",
            "field": "Public mutable native field whose value should hold on the uniquely created object.",
            "accepted_values": "Desired final values from the native object vocabulary.",
        }.items():
            if name in created_definition.get("properties", {}):
                created_definition["properties"][name]["description"] = description
        selector_schema=created_definition.get("properties",{}).get("selectors")
        if selector_schema is not None:
            selector_schema.update({"type":"object","minProperties":1,"maxProperties":16,
                "additionalProperties":{"anyOf":[{"type":"string"},{"type":"integer"},
                    {"type":"number"},{"type":"boolean"},{"type":"null"}]}})
    # Reject information-only proposals at the tool boundary as well as in the
    # runtime model, before they spend a submission or revision.
    schema["anyOf"] = [
        {"required": ["goals"], "properties": {"goals": {"minItems": 1}}},
        {"required": ["refusal_operations"],
         "properties": {"refusal_operations": {"minItems": 1}}},
    ]
    if protocol == PROTOCOL:
        schema["anyOf"].append({"required": ["created_goals"],
                                "properties": {"created_goals": {"minItems": 1}}})
    if frozen is not None:
        adapter, _ = _adapter_world(frozen)
        write_schemas = _native_write_schemas(adapter)
        schema["properties"]["refusal_operations"]["items"] = {"anyOf": write_schemas}
        schema["properties"]["confirmation_requirements"]["items"] = {"anyOf": write_schemas}
        if created_definition:
            _, world = _adapter_world(frozen)
            contracts=[frame for profile_id in sorted(world.profiles)
                       for frame in _contract(adapter,world,adapter.private_state(world,profile_id)) if frame.allow_create]
            collections=sorted({frame.collection for frame in contracts})
            fields=sorted({field for frame in contracts for field in frame.mutable_fields})
            created_definition["properties"]["profile_id"]["enum"]=sorted(world.profiles)
            created_definition["properties"]["collection"]["enum"]=collections
            created_definition["properties"]["field"]["enum"]=fields
    return schema


def _tools(stage, *, guide_optional_confirmations=False, frozen=None, protocol=PROTOCOL):
    if stage == "submission":
        if protocol in {PROTOCOL_V3, PROTOCOL}:
            schema = _proposal_schema(frozen, protocol=protocol)
        else:
            schema = _LegacyDialogueDesignerProposal.model_json_schema()
        if not guide_optional_confirmations:
            schema["properties"]["confirmation_requirements"].pop("description", None)
        return [_tool("submit_frozen_world_task", "Submit business targets and objectives; native scope is compiler-owned.",
                      schema)]
    return [_tool("inspect_frozen_world", "Bounded read-only inspection; never writes or evaluates a future answer.", ReadQuery.model_json_schema()),
            _tool("ready_to_submit", "Finish exploration and proceed to the forced submission phase.",
                  {"type": "object", "properties": {}, "additionalProperties": False})]


def _messages(frozen, context, budget, *, protocol=PROTOCOL):
    policy = (_LEGACY_POLICY if protocol == LEGACY_PROTOCOL else
              POLICY_V3 if protocol == PROTOCOL_V3 else POLICY)
    if budget.confirmation_guidance:
        policy += "\n" + _OPTIONAL_CONFIRMATION_GUIDANCE
    if context.get("skill_rendered"):
        policy += "\nApplied Designer Skill from snapshot " + str(context.get("skill_snapshot_hash")) + ":\n" + context["skill_rendered"]
    return [{"role": "system", "content": policy},
            {"role": "user", "content": json.dumps({"catalogue": exploration_view(
                frozen, include_context=budget.include_related_objects, protocol=protocol),
                "budget": budget.model_dump(mode="json"),
                "prior_evidence": {key: value for key, value in context.items() if key != "skill_rendered"}},
                ensure_ascii=False, sort_keys=True)}]


class ExplorationExecution(Model):
    protocol: Literal["frozen-world-designer-exploration-v2", "frozen-world-designer-exploration-v3",
                      "frozen-world-designer-exploration-v4"] = PROTOCOL
    frozen_hash: str
    model: ModelSettings
    budget: DesignerBudget
    construction_budget: ConstructionBudget
    prompt_context: dict
    exchanges: list[dict]
    request: dict
    response: dict
    selection: TaskSelection
    task: CertifiedTask
    cost: dict
    metrics: dict
    execution_hash: str = ""

    def computed_hash(self):
        return digest(self.model_dump(mode="json", exclude={"execution_hash"}))


class DesignerFailure(ValueError):
    def __init__(self, message, journal):
        super().__init__(message)
        self.journal = journal


_V4_FAILURE_MEANINGS = {
    "schema_invalid": "The proposal does not match the declared task schema. Correct its fields and native parameter shape.",
    "object_missing": "A referenced profile or object was not found in the frozen world. Query registered objects; do not invent identifiers.",
    "scope_invalid": "The target or operation is outside the selected Customer identity, object scope, or native frame.",
    "authorization_unmet": "A required Customer choice or confirmation is missing or does not match the selected profile's exact authority.",
    "external_event_missing": "The proposal references an event that is not registered for the selected frozen objects. Query registered events; do not invent one.",
    "business_precondition_unmet": "The proposal conflicts with a native business constraint. This rejects this proposal; it does not prove that no alternative task exists.",
    "reference_unproven": "A bounded search failure does not prove impossibility. The independent Reference did not prove this proposal within the configured budget; revise the task or report it as unproven.",
    "model_output_invalid": "The model response did not follow the required assistant/function-calling protocol. Submit one complete, valid tool call.",
    "transport_error": "The model provider request failed. No automatic retry was performed.",
    "verifier_error": "The native verifier encountered an internal consistency error. The run is stopped; do not revise the task to conceal it.",
}


def _designer_failure_category(exc, *, stage):
    """Classify v4 failures without turning bounded proof failure into impossibility.

    v2/v3 journals retain their historical two-category vocabulary in ``design``;
    this richer diagnostic is deliberately scoped to the current protocol.
    """
    message = str(exc).lower()
    if isinstance(exc, ValidationError):
        return "schema_invalid"
    if isinstance(exc, (ServiceError, json.JSONDecodeError)) or any(token in message for token in (
        "json", "function-call", "function call", "tool call", "assistant message",
        "submit_frozen_world_task call", "completion protocol", "duplicate key",
        "non-finite json",
    )):
        return "model_output_invalid"
    if any(token in message for token in (
        "unregistered external event", "unregistered external schedule",
        "external event is not registered", "external schedule entry",
    )):
        return "external_event_missing"
    if any(token in message for token in (
        "profile_id does not exist", "requested object does not exist",
        "does not exist in the frozen world", "not an existing frozen business object",
        "not an existing object in the selected", "registered frozen collection",
    )):
        return "object_missing"
    if any(token in message for token in (
        "no native refusal task", "outside frozen customer choices",
        "unregistered terminal outcome", "return window",
    )):
        return "business_precondition_unmet"
    if any(token in message for token in (
        "confirmation", "consent", "authorization", "customer choice",
        "preference requirement is outside", "not in the frozen customer choices",
    )):
        return "authorization_unmet"
    if any(token in message for token in (
        "outside selected customer", "outside selected frozen customer", "outside the selected customer",
        "outside the selected frozen customer", "outside customer authority",
        "same authenticated identity", "identity differs", "scope", "native frame",
        "out-of-scope", "outside the task",
    )):
        return "scope_invalid"
    if any(token in message for token in (
        "did not prove", "search exhausted", "reference search", "bounded native search",
        "bounded native dialogue search", "bounded reference search",
    )):
        return "reference_unproven"
    if any(token in message for token in (
        "engine disagrees", "certificate integrity", "oracle/replay", "task certificate does not reproduce",
        "task construction/certificate replay differs",
    )):
        return "verifier_error"
    if any(token in message for token in (
        "business", "policy", "native refusal", "terminal outcome", "return window",
        "outside frozen customer choices", "not established", "not covered",
    )):
        return "business_precondition_unmet"
    # Query-stage errors are failures to express a registered read, not an
    # invalid business proposal. Treat them as model output/schema errors.
    if stage == "exploration":
        return "model_output_invalid"
    return "business_precondition_unmet"


def design(frozen, client, construction_budget=None, prompt_context=None, *, budget=None, journal_path=None,
           protocol=PROTOCOL):
    frozen = verified_world(frozen)
    _adapter_world(frozen)
    if protocol not in {LEGACY_PROTOCOL, PROTOCOL_V3, PROTOCOL}:
        raise ValueError("unknown frozen-world Designer exploration protocol")
    # New sessions use the complete query context. Explicit old serialized
    # budgets retain their omitted/default flags for exact historical replay.
    budget = budget or DesignerBudget(initial_exploration_calls=3, actionable_feedback=True,
                                      include_related_objects=True, include_profile_events=True,
                                      confirmation_guidance=True)
    construction_budget = construction_budget or ConstructionBudget()
    context = copy.deepcopy(prompt_context or {})
    settings = client.settings.model_copy(deep=True)
    messages = _messages(frozen, context, budget, protocol=protocol)
    journal = {"protocol": protocol, "frozen_hash": frozen.registration_hash,
               "model": settings.model_dump(mode="json"), "budget": budget.model_dump(mode="json"),
               "construction_budget": construction_budget.model_dump(mode="json"), "prompt_context": context,
               "status": "RUNNING", "exchanges": [], "queries": 0, "proposal_attempts": 0}

    def save():
        journal["journal_hash"] = digest({key: value for key, value in journal.items() if key != "journal_hash"})
        if journal_path is not None:
            atomic_json(Path(journal_path), journal, immutable=False)

    def fail(message, *, category=None):
        journal.update(status="BLOCKED", failure=message)
        if protocol == PROTOCOL and category is not None:
            journal["failure_category"] = category
        save()
        raise DesignerFailure(message, copy.deepcopy(journal))

    save()
    seen = set()
    stage = "exploration" if budget.max_exploration_calls else "submission"
    exploration_calls = 0
    while True:
        request = settings.request(copy.deepcopy(messages), _tools(
            stage, guide_optional_confirmations=budget.confirmation_guidance,
            frozen=frozen if protocol in {PROTOCOL_V3, PROTOCOL} else None, protocol=protocol))
        if protocol == PROTOCOL:
            # v4 Designer submissions are schema-bounded; do not add a
            # harness-side output token cap. Providers may still enforce their
            # own limits. Historical v2/v3 requests remain byte-for-byte stable.
            for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
                request.pop(key, None)
        if stage == "submission":
            request["tool_choice"] = {"type": "function", "function": {"name": "submit_frozen_world_task"}}
            # DashScope Qwen supports thinking for exploration, but rejects a
            # forced function in thinking mode. Only an explicitly enabled
            # option is switched at this phase; old omitted-option journals
            # retain their original requests and failure replay.
            if settings.model.lower().startswith("dashscope/qwen") and settings.options.get("enable_thinking") is True:
                request["enable_thinking"] = False
            journal["proposal_attempts"] += 1
        else:
            exploration_calls += 1
        row = {"stage": stage, "request": copy.deepcopy(request), "tool_results": []}
        journal["exchanges"].append(row)
        save()
        try:
            response = client.complete(request)
        except Exception as exc:
            row["transport_error"] = {"type": type(exc).__name__, "message": str(exc)}
            fail("Designer transport failed; no retry performed: " + str(exc),
                 category="transport_error" if protocol == PROTOCOL else None)
        row["response"] = copy.deepcopy(response)
        save()
        parsed = None
        calls = []
        try:
            message, calls = completion_message(response, seen)
            messages.append(message)
            if stage == "submission":
                if len(calls) != 1 or calls[0]["function"]["name"] != "submit_frozen_world_task":
                    raise ValueError("submission requires exactly one submit_frozen_world_task call")
                raw = strict_json(calls[0]["function"]["arguments"])
                proposal_model = {LEGACY_PROTOCOL: _LegacyDialogueDesignerProposal,
                                  PROTOCOL_V3: _V3DialogueDesignerProposal,
                                  PROTOCOL: DialogueDesignerProposal}[protocol]
                parsed = proposal_model.model_validate(raw)
                selection, task = compile_proposal(frozen, parsed.model_dump(mode="json"),
                                                   construction_budget, protocol=protocol)
            else:
                ready = not calls
                for call in calls:
                    try:
                        name = call["function"]["name"]
                        args = strict_json(call["function"]["arguments"])
                        if name == "ready_to_submit":
                            if args != {}:
                                raise ValueError("ready_to_submit takes no arguments")
                            ready = True
                            result = {"status": "ready"}
                        elif name == "inspect_frozen_world":
                            journal["queries"] += 1
                            if journal["queries"] > budget.max_queries:
                                raise ValueError("read-only query budget exhausted")
                            result = {"status": "ok", "data": inspect_world(frozen, args,
                                include_related_objects=budget.include_related_objects,
                                include_profile_events=budget.include_profile_events, protocol=protocol)}
                        else:
                            raise ValueError("only registered read-only exploration tools are allowed")
                    except (ValueError, TypeError, KeyError) as exc:
                        result = {"status": "error", "detail": str(exc)}
                        if protocol == PROTOCOL:
                            result["category"] = _designer_failure_category(exc, stage="exploration")
                    row["tool_results"].append(result)
                    messages.append({"role": "tool", "tool_call_id": call["id"],
                                     "content": json.dumps(result, ensure_ascii=False, sort_keys=True)})
                initial_limit = (journal["proposal_attempts"] == 0 and budget.initial_exploration_calls is not None
                                 and exploration_calls >= budget.initial_exploration_calls)
                if ready or exploration_calls >= budget.max_exploration_calls or initial_limit:
                    stage = "submission"
                    messages.append({"role": "user", "content": "Exploration phase is finished. Submit your own business challenge now."})
                save()
                continue
        except (ValueError, TypeError, KeyError, ServiceError) as exc:
            if protocol == PROTOCOL:
                category = _designer_failure_category(exc, stage=stage)
                feedback = {"status": "rejected", "detail": str(exc), "category": category,
                            "meaning": _V4_FAILURE_MEANINGS[category]}
            else:
                category = "search_unproven" if "did not prove" in str(exc) else "invalid_proposal"
                feedback = {"status": "rejected", "detail": str(exc), "category": category,
                            "meaning": "A bounded search failure does not prove impossibility. No task or world has been repaired for you."}
            if category == "verifier_error":
                fail("Designer verifier consistency failure: " + str(exc), category=category)
            if budget.actionable_feedback:
                submitted = [exchange for exchange in journal["exchanges"] if exchange["stage"] == "submission"]
                def arguments(exchange):
                    choices = exchange.get("response", {}).get("choices", [{}])
                    tool_calls = choices[0].get("message", {}).get("tool_calls") or []
                    return [item.get("function", {}).get("arguments") for item in tool_calls]
                feedback.update(
                    remaining_exploration_calls=max(0, budget.max_exploration_calls - exploration_calls),
                    remaining_submissions=max(0, 1 + budget.max_revisions - journal["proposal_attempts"]),
                    construction_budget=construction_budget.model_dump(mode="json"),
                    unchanged_submission=len(submitted) > 1 and arguments(submitted[-1]) == arguments(submitted[-2]),
                    correction_guidance="Revise your own proposal; repeating an unchanged rejected proposal does not change deterministic certification. You may inspect another profile, choose other legitimate objectives, or revise optional requirements. Registered external events occur only when explicitly scheduled. Use only actual facts; no answer has been selected for you.")
            row["feedback"] = feedback
            # Preserve malformed raw responses in the journal, but never insert
            # incomplete assistant/tool messages into the provider conversation.
            if messages and messages[-1].get("role") == "assistant":
                for call in calls:
                    messages.append({"role": "tool", "tool_call_id": call["id"],
                                     "content": json.dumps(feedback, ensure_ascii=False, sort_keys=True)})
            messages.append({"role": "user", "content": json.dumps(feedback, ensure_ascii=False, sort_keys=True)})
            save()
            if stage == "submission" and journal["proposal_attempts"] >= 1 + budget.max_revisions:
                fail("Designer submission budget exhausted: " + str(exc),
                     category=category if protocol == PROTOCOL else None)
            stage = ("exploration" if exploration_calls < budget.max_exploration_calls
                     and journal["queries"] < budget.max_queries else "submission")
            continue

        row["tool_results"] = [{"status": "certified", "task_hash": task.task_hash}]
        journal["status"] = "CERTIFIED"
        save()
        usages = [exchange["response"].get("usage", {}) for exchange in journal["exchanges"]]
        cost = {"calls": len(usages), "usage_complete": all(all(type(usage.get(key)) is int
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")) for usage in usages)}
        for dest, source in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens"), ("total_tokens", "total_tokens")):
            cost[dest] = sum(usage.get(source, 0) or 0 for usage in usages)
        signature = {"goals": [{"collection": g.collection, "field": g.field, "values": g.accepted_values} for g in parsed.goals],
                     "refusal_operations": [op.get("name") for op in parsed.refusal_operations]}
        if getattr(parsed, "created_goals", None):
            signature["created_goals"] = [{"collection": goal.collection, "selectors": goal.selectors,
                "field": goal.field, "values": goal.accepted_values} for goal in parsed.created_goals]
        metrics = {"proposal_attempts": journal["proposal_attempts"], "certified_proposals": 1,
                   "valid_proposal_rate": 1 / journal["proposal_attempts"], "queries": journal["queries"],
                   "objective_signature": digest(signature), "selected_profiles": len(parsed.profile_ids),
                   "reference_path_business_steps": sum(event.kind == "business" for event in task.dialogue_witnesses[0].trace.events),
                   "service_execution_count": 0}
        execution = ExplorationExecution(protocol=protocol, frozen_hash=frozen.registration_hash,
            model=settings, budget=budget,
            construction_budget=construction_budget, prompt_context=context, exchanges=journal["exchanges"],
            request=request, response=response, selection=selection, task=task, cost=cost, metrics=metrics)
        return execution.model_copy(update={"execution_hash": execution.computed_hash()})


class _ReplayClient:
    def __init__(self, settings, exchanges):
        self.settings, self.exchanges, self.index = settings, exchanges, 0

    def complete(self, request):
        if self.index >= len(self.exchanges):
            raise ValueError("missing recorded Designer response")
        row = self.exchanges[self.index]
        self.index += 1
        if request != row["request"]:
            raise ValueError("Designer request, query results or phase differs from recorded input")
        if "transport_error" in row:
            error = row["transport_error"]
            # Reproduce an exception name/message without importing or executing
            # any exception type named by an untrusted journal.
            raise type(error["type"], (Exception,), {})(error["message"])
        return copy.deepcopy(row["response"])


def verify_execution(record):
    row = ExplorationExecution.model_validate(record.model_dump(mode="json") if isinstance(record, ExplorationExecution) else record)
    if row.execution_hash != row.computed_hash() or row.frozen_hash != row.task.frozen.registration_hash:
        raise ValueError("Designer exploration integrity differs")
    client = _ReplayClient(row.model, row.exchanges)
    rebuilt = design(row.task.frozen, client, row.construction_budget, row.prompt_context,
                     budget=row.budget, protocol=row.protocol)
    if rebuilt != row or client.index != len(row.exchanges):
        raise ValueError("Designer exploration, feedback, objective or certification Replay differs")
    return {"verified": True, "protocol": row.protocol, "model_calls": client.index,
            "actual_online_calls": 0, "task_hash": row.task.task_hash, "task_replay": True}


def verify_failed_journal(frozen, journal):
    """Reproduce a bounded failure, including query results and all feedback."""
    journal = copy.deepcopy(journal)
    expected_hash = digest({key: value for key, value in journal.items() if key != "journal_hash"})
    if journal.get("journal_hash") != expected_hash or journal.get("status") != "BLOCKED":
        raise ValueError("invalid failed Designer journal integrity/status")
    if journal["frozen_hash"] != frozen.registration_hash:
        raise ValueError("failed Designer journal belongs to another frozen world")
    client = _ReplayClient(ModelSettings.model_validate(journal["model"]), journal["exchanges"])
    try:
        design(frozen, client, ConstructionBudget(**journal["construction_budget"]), journal["prompt_context"],
               budget=DesignerBudget(**journal["budget"]),
               protocol=journal.get("protocol", LEGACY_PROTOCOL))
    except DesignerFailure as exc:
        if exc.journal != journal or client.index != len(journal["exchanges"]):
            raise ValueError("failed Designer query/feedback Replay differs") from exc
        return {"verified": True, "failure_reproduced": True, "model_calls": client.index, "actual_online_calls": 0}
    raise ValueError("recorded failed Designer run unexpectedly certified")
