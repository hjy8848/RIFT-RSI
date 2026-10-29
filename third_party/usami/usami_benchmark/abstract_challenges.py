"""Strict, ID-free challenge language for the world-grounded batch Designer.

The grammar is deliberately smaller than the native task language. A Designer
can choose predicates and goals; it cannot provide entity IDs, frames, native
operations, Reference outcomes, or executable code.
"""
from __future__ import annotations

import re
import copy
from datetime import datetime
from functools import lru_cache

from typing import Any, Literal, Optional, get_args
from pydantic import Field, model_validator

from usami_harness.models import Model, digest


PROTOCOL = "world-grounded-batch-designer-v1"
BATCH_REVISION_PROTOCOL = "world-grounded-batch-designer-one-revision-v1"
CONTRACT_VERSION = "world-grounded-contract-v2"

EntityType = Literal["profile", "customer", "order", "line", "product", "evidence", "external_event"]
Operator = Literal["eq", "neq", "in", "lt", "lte", "gt", "gte"]
Predicate = Literal["within_case_window", "eligible_for_case", "has_valid_evidence", "requires_approval", "registered_event_exists"]
Relation = Literal["distinct", "same_identity", "same_owner", "references", "profile_for_line", "event_for_line"]

ENTITY_COLLECTIONS = {"profile": "profiles", "customer": "users", "order": "orders", "line": "lines",
                      "product": "products", "evidence": "evidence", "external_event": "external_events"}
REFERENCE_FIELDS = {("line", "order"): "order_id", ("line", "product"): "product_id",
    ("evidence", "line"): "line_id", ("external_event", "line"): "line_id",
    ("profile", "line"): "facts.line_id", ("profile", "order"): "facts.order_id",
    ("profile", "evidence"): "facts.evidence_id", ("order", "customer"): "customer_id"}
RELATION_PAIRS = {
    "distinct": {(t, t) for t in get_args(EntityType)},
    "same_identity": {(l, r) for l in ("profile", "customer") for r in ("profile", "customer")},
    "same_owner": {(l, r) for l in ("profile", "line", "order") for r in ("profile", "line", "order")},
    "references": set(REFERENCE_FIELDS),
    "profile_for_line": {("profile", "line"), ("line", "profile")},
    "event_for_line": {("external_event", "line"), ("line", "external_event")},
}
EVENT_OUTCOMES = {"manager_review": {"approved", "denied"}, "return_received": {"received", "missing"}}
OBJECTIVE_CONTRACTS = {
    "existing_object_completion": {"collection": "lines", "field": "status",
        "accepted_values": ["return_pending", "refunded", "exchanged", "reshipped"]},
    "created_object_completion": {"collection": "cases", "field": "status",
        "accepted_values": ["open", "evidence_verified"]},
    "correct_refusal": {"operation": "open_case"},
}


class EntityVariable(Model):
    alias: str = Field(min_length=1, max_length=48, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    type: EntityType


class EntityConstraint(Model):
    alias: str = Field(min_length=1, max_length=48)
    field: Optional[str] = None
    op: Optional[Operator] = None
    value: Any = None
    predicate: Optional[Predicate] = None

    @model_validator(mode="after")
    def exactly_field_or_predicate(self):
        field_form = self.field is not None or self.op is not None
        predicate_form = self.predicate is not None
        if field_form == predicate_form:
            raise ValueError("constraint must define exactly one field comparison or registered predicate")
        if field_form and (self.field is None or self.op is None):
            raise ValueError("field comparisons need both field and operator")
        if predicate_form and self.op is not None:
            raise ValueError("registered predicates do not accept an operator")
        return self


class RelationshipConstraint(Model):
    type: Relation
    left: str = Field(min_length=1, max_length=48)
    right: str = Field(min_length=1, max_length=48)


class ChallengeObjective(Model):
    kind: Literal["existing_object_completion", "created_object_completion", "correct_refusal"]
    target: str = Field(min_length=1, max_length=48)
    collection: Optional[Literal["lines", "cases"]] = None
    field: Optional[Literal["status"]] = None
    accepted_values: list[str] = Field(default_factory=list, max_length=8)
    operation: Optional[Literal["open_case"]] = None

    @model_validator(mode="after")
    def validate_objective_shape(self):
        if self.kind == "existing_object_completion":
            if self.collection != "lines" or self.field != "status" or not self.accepted_values or self.operation:
                raise ValueError("existing completion currently targets a line status with accepted values")
            if set(self.accepted_values) - set(OBJECTIVE_CONTRACTS[self.kind]["accepted_values"]):
                raise ValueError("unregistered After-sales line completion value")
        elif self.kind == "created_object_completion":
            if self.collection != "cases" or self.field != "status" or not self.accepted_values or self.operation:
                raise ValueError("created completion currently targets a Case status")
            if set(self.accepted_values) - set(OBJECTIVE_CONTRACTS[self.kind]["accepted_values"]):
                raise ValueError("unregistered Case completion value")
        elif self.operation != "open_case" or any((self.collection, self.field, self.accepted_values)):
            raise ValueError("correct refusal currently targets the native open_case operation only")
        if len(self.accepted_values) != len(set(self.accepted_values)):
            raise ValueError("objective values must not repeat")
        return self


class CustomerInformationRequirement(Model):
    profile_alias: str = Field(min_length=1, max_length=48)
    fact_keys: list[Literal["order_id", "line_id", "issue", "evidence_id", "deadline", "pickup_slot"]] = Field(min_length=1, max_length=6)


class CustomerPreferenceRequirement(Model):
    profile_alias: str = Field(min_length=1, max_length=48)
    preference: Literal["resolution"] = "resolution"
    accepted_values: list[Literal["refund", "exchange", "reship"]] = Field(min_length=1, max_length=3)


class ExternalEventConstraint(Model):
    target_alias: str = Field(min_length=1, max_length=48)
    kind: Literal["manager_review", "return_received"]
    outcome: Literal["approved", "denied", "received", "missing"]
    required: bool = Field(default=True, strict=True)


class AbstractChallenge(Model):
    challenge_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    hypothesis: str = Field(min_length=8, max_length=1200)
    domain: Literal["aftersales", "airline", "property", "education", "saas", "cloudops", "fulfillment", "stayops"]
    customer_request: str = Field(min_length=8, max_length=1200)
    entity_variables: list[EntityVariable] = Field(min_length=1, max_length=10)
    entity_constraints: list[EntityConstraint] = Field(default_factory=list, max_length=32)
    relationship_constraints: list[RelationshipConstraint] = Field(default_factory=list, max_length=24)
    objectives: list[ChallengeObjective] = Field(min_length=1, max_length=8)
    required_customer_information: list[CustomerInformationRequirement] = Field(default_factory=list, max_length=10)
    required_customer_preferences: list[CustomerPreferenceRequirement] = Field(default_factory=list, max_length=10)
    external_event_constraints: list[ExternalEventConstraint] = Field(default_factory=list, max_length=12)
    difficulty_hypothesis: Optional[str] = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def unique_variables_and_objectives(self):
        aliases = [v.alias for v in self.entity_variables]
        if len(aliases) != len(set(aliases)):
            raise ValueError("entity aliases must be unique")
        if len({r.profile_alias for r in self.required_customer_information}) != len(self.required_customer_information):
            raise ValueError("at most one information declaration is allowed per profile alias")
        if len({r.profile_alias for r in self.required_customer_preferences}) != len(self.required_customer_preferences):
            raise ValueError("at most one preference declaration is allowed per profile alias")
        return self

    @property
    def challenge_hash(self) -> str:
        return digest(self)


class ChallengeBatch(Model):
    protocol: Literal[PROTOCOL] = PROTOCOL
    challenges: list[AbstractChallenge] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def unique_challenge_ids(self):
        ids = [c.challenge_id for c in self.challenges]
        if len(ids) != len(set(ids)):
            raise ValueError("challenge_id values must be unique in a batch")
        return self


FIELD_ALLOWLIST: dict[str, dict[str, set[str]]] = {
    "aftersales": {
        "profile": {"issue"},
        "customer": set(),
        "order": {"status", "paid_cents", "payment_channel", "delivered_at"},
        "line": {"status", "quantity", "unit_price_cents"},
        "product": {"stock", "returnable", "return_days", "return_required", "replacement_days"},
        "evidence": {"issue", "simulated_valid"},
        "external_event": {"kind", "outcome", "at"},
    }
}

PREDICATE_TYPES: dict[str, set[str]] = {
    "within_case_window": {"line"}, "eligible_for_case": {"line"},
    "has_valid_evidence": {"line"}, "requires_approval": {"line"},
    "registered_event_exists": {"line"},
}


@lru_cache(maxsize=64)
def _native_field_contract(entity_type: str, field: str) -> dict:
    """Publish native value types/enums, rather than a second handwritten rulebook."""
    from usami_aftersales import models as native
    models = {"profile": native.Evidence, "order": native.Order, "line": native.Line,
              "product": native.Product, "evidence": native.Evidence, "external_event": native.ExternalEvent}
    if field not in FIELD_ALLOWLIST["aftersales"].get(entity_type, set()):
        raise ValueError("unknown field for declared entity type")
    spec = copy.deepcopy(models[entity_type].model_json_schema()["properties"][field])
    spec.pop("title", None); spec.pop("default", None)
    # Operands are comparison thresholds, not proposed native field writes.
    for bound in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        spec.pop(bound, None)
    variants = spec.get("anyOf", [spec])
    ordered = any(v.get("type") == "integer" or v.get("format") == "date-time" for v in variants)
    return {"value_schema": spec, "operators": list(get_args(Operator)) if ordered else ["eq", "neq", "in"]}


def field_contract(entity_type: str, field: str) -> dict:
    return copy.deepcopy(_native_field_contract(entity_type, field))


def _aware_time(value):
    if not isinstance(value, str):
        raise ValueError("time comparison requires an ISO-8601 string")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("time comparison requires an ISO-8601 string") from None
    if stamp.tzinfo is None:
        raise ValueError("time comparison requires an explicit timezone")
    return stamp


def validate_field_value(entity_type, field, value, op):
    contract = field_contract(entity_type, field)
    if op not in contract["operators"]:
        raise ValueError("operator is not supported for this field type")
    if op == "in" and (not isinstance(value, list) or not value):
        raise ValueError("in predicate needs a nonempty value list")
    values = value if op == "in" else [value]
    spec = contract["value_schema"]
    variants = spec.get("anyOf", [spec])
    for item in values:
        valid = False
        for variant in variants:
            typ = variant.get("type")
            if typ == "null" and item is None and op in {"eq", "neq", "in"}:
                valid = True
            elif typ == "boolean" and type(item) is bool:
                valid = True
            elif typ == "integer" and type(item) is int:
                valid = True
            elif typ == "string" and isinstance(item, str):
                if variant.get("format") == "date-time":
                    _aware_time(item)
                valid = "enum" not in variant or item in variant["enum"]
        if not valid:
            raise ValueError("field comparison value has the wrong primitive type or unregistered enum value")


def challenge_grammar() -> dict:
    """One public vocabulary for the catalogue, schema and deterministic matcher."""
    return {
        "contract_version": CONTRACT_VERSION,
        "entity_types": {k: sorted(v) for k, v in FIELD_ALLOWLIST["aftersales"].items()},
        "field_contracts": {kind: {name: field_contract(kind, name) for name in sorted(fields)}
                            for kind, fields in FIELD_ALLOWLIST["aftersales"].items()},
        "relationships": list(get_args(Relation)),
        "relationship_pairs": {name: [list(pair) for pair in sorted(pairs)] for name, pairs in RELATION_PAIRS.items()},
        "constraint_grammar": {"field_operators": list(get_args(Operator)),
            "registered_predicates": {k: sorted(v) for k, v in PREDICATE_TYPES.items()}},
        "objective_types": list(get_args(ChallengeObjective.model_fields["kind"].annotation)),
        "objective_contracts": copy.deepcopy(OBJECTIVE_CONTRACTS),
        "composition": "Combine atomic objectives in the objectives array. mixed and multi_object are compositions, never kind values.",
        "scope_contract": "Every objective targets a line alias explicitly linked to a profile alias by profile_for_line. All bound customer, profile, order, line, evidence and event objects belong to that same authenticated identity. Products must be referenced by its lines. Information, preferences and scheduled events refer only to objective profiles/lines.",
        "event_outcomes": {k: sorted(v) for k, v in EVENT_OUTCOMES.items()},
        "event_selection": "Binding an external_event variable only inspects that registered event. Only external_event_constraints with required=true schedule it. required=false asserts that no matching registered event exists; it does not mean skip a present event.",
    }


def submission_schema(count: int) -> dict:
    """Inline local refs and express model-validator forms on the wire as well."""
    if type(count) is not int or not 1 <= count <= 5:
        raise ValueError("batch count must be between one and five")
    root = ChallengeBatch.model_json_schema()
    definitions = root.pop("$defs")
    scalar = {"type": ["string", "integer", "boolean", "null"]}
    constraint = definitions["EntityConstraint"]
    alias = constraint["properties"]["alias"]
    definitions["EntityConstraint"] = {"oneOf": [
        {"type": "object", "additionalProperties": False,
         "required": ["alias", "field", "op", "value"], "properties": {
             "alias": alias, "field": {"enum": sorted({f for fs in FIELD_ALLOWLIST["aftersales"].values() for f in fs})},
             "op": {"enum": [op for op in get_args(Operator) if op != "in"]}, "predicate": {"type": "null"}, "value": scalar}},
        {"type": "object", "additionalProperties": False,
         "required": ["alias", "field", "op", "value"], "properties": {
             "alias": alias, "field": {"enum": sorted({f for fs in FIELD_ALLOWLIST["aftersales"].values() for f in fs})},
             "op": {"const": "in"}, "predicate": {"type": "null"},
             "value": {"type": "array", "minItems": 1, "items": scalar}}},
        {"type": "object", "additionalProperties": False,
         "required": ["alias", "predicate", "value"], "properties": {
             "alias": alias, "predicate": {"enum": list(get_args(Predicate))},
             "value": {"type": "boolean"}, "field": {"type": "null"}, "op": {"type": "null"}}},
    ]}
    forms = []
    for kind, contract in OBJECTIVE_CONTRACTS.items():
        properties = {"kind": {"const": kind}, "target": definitions["ChallengeObjective"]["properties"]["target"]}
        if kind == "correct_refusal":
            properties.update({"operation": {"const": contract["operation"]},
                "collection": {"type": "null"}, "field": {"type": "null"},
                "accepted_values": {"type": "array", "maxItems": 0}})
            required = ["kind", "target", "operation"]
        else:
            properties.update({"collection": {"const": contract["collection"]}, "field": {"const": contract["field"]},
                "accepted_values": {"type": "array", "minItems": 1, "maxItems": 8, "uniqueItems": True,
                    "items": {"enum": contract["accepted_values"]}}, "operation": {"type": "null"}})
            required = ["kind", "target", "collection", "field", "accepted_values"]
        forms.append({"type": "object", "additionalProperties": False, "required": required, "properties": properties})
    definitions["ChallengeObjective"] = {"oneOf": forms}
    event_base = definitions["ExternalEventConstraint"]
    definitions["ExternalEventConstraint"] = {"oneOf": [
        {**event_base, "properties": {**event_base["properties"], "kind": {"const": kind}, "outcome": {"enum": sorted(outcomes)}}}
        for kind, outcomes in EVENT_OUTCOMES.items()]}
    definitions["AbstractChallenge"]["properties"]["domain"] = {"const": "aftersales"}
    definitions["CustomerInformationRequirement"]["properties"]["fact_keys"]["uniqueItems"] = True
    definitions["CustomerPreferenceRequirement"]["properties"]["accepted_values"]["uniqueItems"] = True

    def inline(node):
        if isinstance(node, list):
            return [inline(v) for v in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return inline({**definitions[node["$ref"].split("/")[-1]], **{k: v for k, v in node.items() if k != "$ref"}})
        return {k: inline(v) for k, v in node.items() if k not in {"title", "default"}}

    root["required"] = ["protocol", "challenges"]
    root["properties"]["challenges"].update({"minItems": count, "maxItems": count})
    return inline(root)


def validate_challenge(challenge: AbstractChallenge | dict) -> AbstractChallenge:
    """Validate typed references and fixed grammar before any world lookup."""
    row = challenge if isinstance(challenge, AbstractChallenge) else AbstractChallenge.model_validate(challenge)
    if row.domain != "aftersales":
        raise ValueError("grounder supports the registered aftersales grammar only")
    variables = {v.alias: v.type for v in row.entity_variables}
    if len({v.type for v in row.entity_variables}) > 7:
        raise ValueError("unsupported entity type combination")
    fields = FIELD_ALLOWLIST[row.domain]
    for constraint in row.entity_constraints:
        kind = variables.get(constraint.alias)
        if kind is None:
            raise ValueError("entity constraint references an unknown alias")
        if constraint.predicate:
            if kind not in PREDICATE_TYPES.get(constraint.predicate, set()):
                raise ValueError("registered predicate is unsupported for this entity type")
            if type(constraint.value) is not bool:
                raise ValueError("registered predicate value must be boolean")
        else:
            if constraint.field not in fields[kind]:
                raise ValueError("unknown field for declared entity type")
            if constraint.op == "in" and (not isinstance(constraint.value, list) or not constraint.value):
                raise ValueError("in predicate needs a nonempty value list")
            if constraint.op != "in" and isinstance(constraint.value, (dict, list)):
                raise ValueError("field comparison values must be scalar")
            validate_field_value(kind, constraint.field, constraint.value, constraint.op)
    for relation in row.relationship_constraints:
        left, right = variables.get(relation.left), variables.get(relation.right)
        if left is None or right is None:
            raise ValueError("relationship references an unknown alias")
        allowed = (left, right) in RELATION_PAIRS[relation.type]
        if not allowed:
            raise ValueError("relationship type is not registered for these entity types")
    relation_keys = [(r.type, *( [r.left, r.right] if r.type == "references" else sorted([r.left, r.right]) ))
                     for r in row.relationship_constraints]
    if len(set(relation_keys)) != len(relation_keys):
        raise ValueError("relationship constraints must not repeat, including symmetric reversals")
    if len({digest(c) for c in row.entity_constraints}) != len(row.entity_constraints):
        raise ValueError("entity constraints must not repeat")
    objective_lines = {o.target for o in row.objectives}
    objective_profiles = set()
    for relation in row.relationship_constraints:
        if relation.type == "profile_for_line":
            profile, line = ((relation.left, relation.right) if variables[relation.left] == "profile" else (relation.right, relation.left))
            if line in objective_lines:
                objective_profiles.add(profile)
    for objective in row.objectives:
        if variables.get(objective.target) != "line":
            raise ValueError("objective target must reference a declared line variable")
        related = any(r.type == "profile_for_line" and
                      ((variables.get(r.left) == "profile" and r.right == objective.target) or
                       (variables.get(r.right) == "profile" and r.left == objective.target))
                      for r in row.relationship_constraints)
        if not related:
            raise ValueError("each objective line needs an explicit profile_for_line relationship")
        profiles = {r.left if variables[r.left] == "profile" else r.right for r in row.relationship_constraints
                    if r.type == "profile_for_line" and objective.target in {r.left, r.right}}
        if len(profiles) != 1:
            raise ValueError("each objective line must select exactly one explicit Customer profile alias")
    if len({digest(o) for o in row.objectives}) != len(row.objectives):
        raise ValueError("objectives must not repeat")
    for requirement in row.required_customer_information:
        if len(set(requirement.fact_keys)) != len(requirement.fact_keys):
            raise ValueError("Customer information keys must not repeat")
        if variables.get(requirement.profile_alias) != "profile":
            raise ValueError("information requirement must reference a profile alias")
        if not any(r.type == "profile_for_line" and
                   ((r.left == requirement.profile_alias and r.right in objective_lines) or
                    (r.right == requirement.profile_alias and r.left in objective_lines))
                   for r in row.relationship_constraints):
            raise ValueError("information profile must be explicitly related to an objective line")
    for requirement in row.required_customer_preferences:
        if len(set(requirement.accepted_values)) != len(requirement.accepted_values):
            raise ValueError("Customer preference values must not repeat")
        if variables.get(requirement.profile_alias) != "profile":
            raise ValueError("preference requirement must reference a profile alias")
        if not any(r.type == "profile_for_line" and
                   ((r.left == requirement.profile_alias and r.right in objective_lines) or
                    (r.right == requirement.profile_alias and r.left in objective_lines))
                   for r in row.relationship_constraints):
            raise ValueError("preference profile must be explicitly related to an objective line")
    for event in row.external_event_constraints:
        if event.target_alias not in objective_lines:
            raise ValueError("external event constraint must reference an objective line alias")
        if event.kind == "manager_review" and event.outcome not in {"approved", "denied"}:
            raise ValueError("manager_review events must have approved or denied outcomes")
        if event.kind == "return_received" and event.outcome not in {"received", "missing"}:
            raise ValueError("return_received events must have received or missing outcomes")
    if any(kind == "profile" and alias not in objective_profiles for alias, kind in variables.items()):
        raise ValueError("every profile alias must be explicitly related to an objective line")
    event_conditions = {}
    for event in row.external_event_constraints:
        key = (event.target_alias, event.kind, event.outcome)
        if key in event_conditions:
            raise ValueError("external event conditions must not repeat or contradict")
        event_conditions[key] = event.required
    forbidden = re.compile(r"\b(?:profile_u\d+_order\d+_line\d+|u\d+_order\d+(?:_line\d+)?|photo_u\d+_order\d+_line\d+|(?:manager_review|return_received)_u\d+_order\d+_line\d+|case_[0-9a-f]{8,}|quote_[0-9a-f]{8,})\b")
    text = " ".join((row.customer_request, row.hypothesis, row.difficulty_hypothesis or ""))
    if forbidden.search(text):
        raise ValueError("abstract challenge text must not contain concrete generated-world identifiers")
    return row


def challenge_fingerprints(challenge: AbstractChallenge | dict) -> dict[str, str]:
    """Alias-independent graph fingerprints, including objective-to-entity links.

    These are declared coverage signatures, not a proof of semantic equivalence.
    """
    row = challenge if isinstance(challenge, AbstractChallenge) else AbstractChallenge.model_validate(challenge)
    variables = {v.alias: v.type for v in row.entity_variables}

    def fingerprint(semantic):
        attributes = {alias: {"type": kind, "objectives": []} for alias, kind in variables.items()}
        for objective in row.objectives:
            spec = objective.model_dump(mode="json", exclude={"target"})
            spec["accepted_values"] = sorted(spec["accepted_values"])
            attributes[objective.target]["objectives"].append(
                spec if semantic else objective.kind)
        if semantic:
            for c in row.entity_constraints:
                spec = c.model_dump(mode="json", exclude={"alias"})
                if c.op == "in":
                    spec["value"] = sorted(spec["value"], key=digest)
                attributes[c.alias].setdefault("constraints", []).append(spec)
            for req in row.required_customer_information:
                attributes[req.profile_alias].setdefault("information", []).append(sorted(req.fact_keys))
            for req in row.required_customer_preferences:
                attributes[req.profile_alias].setdefault("preferences", []).append([req.preference, sorted(req.accepted_values)])
            for event in row.external_event_constraints:
                attributes[event.target_alias].setdefault("events", []).append(event.model_dump(mode="json", exclude={"target_alias"}))
        for value in attributes.values():
            for key, items in value.items():
                if isinstance(items, list):
                    value[key] = sorted(items, key=digest)
        labels = {alias: digest(value) for alias, value in attributes.items()}
        for _ in range(len(variables)):
            labels = {alias: digest([labels[alias], sorted(
                [(r.type, "out" if r.type == "references" else "both", labels[r.right]) for r in row.relationship_constraints if r.left == alias]
                + [(r.type, "in" if r.type == "references" else "both", labels[r.left]) for r in row.relationship_constraints if r.right == alias])])
                for alias in variables}
        return digest({"nodes": sorted(labels.values()), "edges": sorted(
            (r.type, *( [labels[r.left], labels[r.right]] if r.type == "references" else sorted([labels[r.left], labels[r.right]]) ))
            for r in row.relationship_constraints)})

    return {"structural": fingerprint(False), "semantic": fingerprint(True)}
