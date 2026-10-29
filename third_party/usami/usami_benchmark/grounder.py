"""Deterministic, ID-grounded matching for the After-sales challenge grammar.

Grounding only binds typed aliases to facts in registered immutable worlds. It
does not consult a task witness or alter a challenge. A successful match is
still only an input proposal: native compilation and Replay certify it later.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Literal, Optional

from pydantic import Field

from usami_harness.models import Model, digest
from usami_benchmark.tasks import OpenDialogueComposition, TaskSelection
from usami_benchmark.splits import verified_world
from usami_aftersales import adapter as aftersales
from usami_aftersales import rules as aftersales_rules
from usami_aftersales import tasks as aftersales_tasks

from .abstract_challenges import (AbstractChallenge, ENTITY_COLLECTIONS, REFERENCE_FIELDS,
    _aware_time, field_contract, validate_challenge)
from .world_index import IndexedWorld, PrivateWorldIndex


class GroundingBudget(Model):
    max_worlds: int = Field(default=20, strict=True, ge=1, le=1000)
    max_matches: int = Field(default=10, strict=True, ge=1, le=100)
    max_bindings: int = Field(default=50000, strict=True, ge=1, le=1000000)
    partition: Literal["train", "validation"] = "train"
    selection_seed: int = Field(default=0, strict=True)


class GroundedMatch(Model):
    frozen_registration_hash: str
    frozen_world_hash: str
    binding: dict[str, str]
    binding_hash: str
    fact_sources: list[dict[str, str]]
    identity: str
    profile_ids: list[str]
    selection: TaskSelection


class GroundingResult(Model):
    protocol: Literal["world-grounded-grounding-result-v1"] = "world-grounded-grounding-result-v1"
    challenge_id: str
    challenge_hash: str
    index_hash: str
    status: Literal["GROUNDED", "NO_MATCH", "INVALID_CONSTRAINT", "OUT_OF_SCOPE",
                    "UNSUPPORTED_OBJECTIVE", "BUDGET_EXHAUSTED", "INTERNAL_ERROR"]
    reason: str = ""
    worlds_considered: int = 0
    bindings_explored: int = 0
    matches_found: int = 0
    search_truncated: bool = False
    matches: list[GroundedMatch] = Field(default_factory=list)
    selected_match_hash: Optional[str] = None
    result_hash: str = ""

    def computed_hash(self):
        return digest(self.model_dump(mode="json", exclude={"result_hash"}))


def _rows(frozen, entity_type):
    world = frozen.world
    collection = ENTITY_COLLECTIONS[entity_type]
    return [{"_key": key, **value} for key, value in sorted(world.get(collection, {}).items())]


def _field(entity_type, row, name):
    if entity_type == "profile":
        return row["identity"] if name == "identity" else row.get("facts", {}).get("issue")
    if entity_type == "customer" and name == "id":
        return row["_key"]
    return row.get(name)


def _compare(actual, op, expected):
    if op == "eq": return type(actual) is type(expected) and actual == expected
    if op == "neq": return type(actual) is not type(expected) or actual != expected
    if op == "in": return any(type(actual) is type(item) and actual == item for item in expected)
    if op in {"lt", "lte", "gt", "gte"}:
        if actual is None:
            return False
        if type(actual) is not type(expected):
            raise ValueError("ordered comparison requires the same primitive type")
        return {"lt": actual < expected, "lte": actual <= expected,
                "gt": actual > expected, "gte": actual >= expected}[op]
    raise ValueError("unregistered field operator")


def _predicate(frozen, entity_type, row, predicate):
    if entity_type != "line":
        raise ValueError("registered After-sales predicates currently target lines")
    world = aftersales.World.model_validate(frozen.world)
    line = world.lines[row["_key"]]
    order = world.orders[line.order_id]
    product = world.products[line.product_id]
    if predicate == "within_case_window":
        return order.status == "delivered" and order.delivered_at is not None and world.clock <= order.delivered_at + timedelta(days=product.return_days)
    if predicate == "eligible_for_case":
        return aftersales_rules.eligibility(world, order, line) is None
    if predicate == "has_valid_evidence":
        issue = next((p.facts.get("issue") for p in world.profiles.values()
                      if p.facts.get("line_id") == line.id and p.identity == order.customer_id), None)
        return any(e.line_id == line.id and e.order_id == order.id and e.holder_id == order.customer_id
                   and e.issue == issue and e.simulated_valid for e in world.evidence.values())
    if predicate == "requires_approval":
        return line.quantity * line.unit_price_cents > world.merchants[order.merchant_id].refund_approval_limit_cents
    if predicate == "registered_event_exists":
        return any(event.line_id == line.id for event in world.external_events.values())
    raise ValueError("unknown registered predicate")


def _relation(frozen, aliases, values, relation):
    ltype, rtype = aliases[relation.left], aliases[relation.right]
    left, right = values[relation.left], values[relation.right]
    world = frozen.world
    if relation.type == "distinct": return left["_key"] != right["_key"]
    def owner(kind, row):
        if kind == "profile": return row["identity"]
        if kind == "customer": return row["_key"]
        if kind == "order": return world["orders"][row["_key"]]["customer_id"]
        if kind == "line": return world["orders"][row["order_id"]]["customer_id"]
        return None
    if relation.type == "same_identity":
        return owner(ltype, left) == owner(rtype, right)
    if relation.type == "same_owner":
        return owner(ltype, left) is not None and owner(ltype, left) == owner(rtype, right)
    if relation.type == "profile_for_line":
        profile, line = (left, right) if ltype == "profile" else (right, left)
        return profile.get("facts", {}).get("line_id") == line["_key"]
    if relation.type in {"references", "event_for_line"}:
        if relation.type == "event_for_line":
            event, line = (left, right) if ltype == "external_event" else (right, left)
            return event.get("line_id") == line["_key"]
        field = REFERENCE_FIELDS.get((ltype, rtype))
        source, target = left, right
        if field is None: return False
        value = source
        for part in field.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        return value == target["_key"]
    return False


def _matches_constraints(frozen, challenge, aliases, assigned):
    for constraint in challenge.entity_constraints:
        if constraint.alias not in assigned:
            continue
        kind, row = aliases[constraint.alias], assigned[constraint.alias]
        if constraint.predicate:
            if _predicate(frozen, kind, row, constraint.predicate) is not constraint.value: return False
        else:
            actual, expected = _field(kind, row, constraint.field), constraint.value
            spec = field_contract(kind, constraint.field)["value_schema"]
            if any(v.get("format") == "date-time" for v in spec.get("anyOf", [spec])):
                actual = _aware_time(actual) if actual is not None else None
                expected = [_aware_time(v) if v is not None else None for v in expected] if constraint.op == "in" else (_aware_time(expected) if expected is not None else None)
            if not _compare(actual, constraint.op, expected): return False
    for relation in challenge.relationship_constraints:
        if relation.left in assigned and relation.right in assigned:
            if not _relation(frozen, aliases, assigned, relation): return False
    profiles = [row for alias, row in assigned.items() if aliases[alias] == "profile"]
    if len({p["identity"] for p in profiles}) > 1:
        return False
    if profiles:
        identity = profiles[0]["identity"]
        world = frozen.world
        for alias, item in assigned.items():
            kind = aliases[alias]
            if kind == "customer" and item["_key"] != identity: return False
            if kind == "order" and item["customer_id"] != identity: return False
            if kind in {"line", "evidence", "external_event"}:
                line = item if kind == "line" else world["lines"][item["line_id"]]
                if world["orders"][line["order_id"]]["customer_id"] != identity: return False
                if kind == "evidence" and item["holder_id"] != identity: return False
            if kind == "product" and len(assigned) == len(aliases):
                if not any(aliases[a] == "line" and v["product_id"] == item["_key"] for a, v in assigned.items()):
                    return False
    return True


def _source_facts(challenge, assigned):
    rows = []
    for alias in sorted(assigned):
        kind, row = next(v.type for v in challenge.entity_variables if v.alias == alias), assigned[alias]
        rows.append({"alias": alias, "entity_type": kind, "key": row["_key"], "source": f"world.{ENTITY_COLLECTIONS[kind]}/{row['_key']}"})
    return rows


def _proposal(frozen, challenge, aliases, assigned):
    world = aftersales.World.model_validate(frozen.world)
    profile_vars = {v.alias: assigned[v.alias] for v in challenge.entity_variables if v.type == "profile"}
    if not profile_vars:
        raise ValueError("challenge must bind a Customer profile for native scope")
    if len({p["identity"] for p in profile_vars.values()}) != 1:
        raise ValueError("selected profiles cross authenticated Customer identities")

    def profile_for_line(line_alias):
        line = assigned[line_alias]
        declared = {r.left if aliases[r.left] == "profile" else r.right for r in challenge.relationship_constraints
                    if r.type == "profile_for_line" and line_alias in {r.left, r.right}}
        matches = [pid for pid, profile in profile_vars.items() if pid in declared and profile.get("facts", {}).get("line_id") == line["_key"]]
        if not matches:
            raise ValueError("each objective line must have an explicitly grounded Customer profile")
        return sorted(matches)[0], profile_vars[sorted(matches)[0]]

    used_profiles: list[str] = []
    goals, created, refusals = [], [], []
    for objective in challenge.objectives:
        profile_alias, profile = profile_for_line(objective.target)
        profile_id = profile["_key"]
        if profile_id not in used_profiles: used_profiles.append(profile_id)
        line = assigned[objective.target]
        line_id = line["_key"]
        if objective.kind == "existing_object_completion":
            goals.append({"collection": "lines", "entity": line_id, "field": "status", "accepted_values": objective.accepted_values})
        elif objective.kind == "created_object_completion":
            issue = profile.get("facts", {}).get("issue")
            if issue is None: raise ValueError("created Case objective lacks a grounded issue fact")
            created.append({"profile_id": profile_id, "collection": "cases",
                "selectors": {"line_id": line_id, "issue": issue}, "field": "status", "accepted_values": objective.accepted_values})
        else:
            order = world.orders[line["order_id"]]
            op = aftersales.Operation(name="open_case", scope_id=order.merchant_id, resource_id=order.id,
                params={"line_id": line_id, "issue": profile["facts"]["issue"]})
            refusals.append(op.model_dump(mode="json"))
    information = []
    for req in challenge.required_customer_information:
        profile = assigned[req.profile_alias]
        if any(key not in profile.get("facts", {}) for key in req.fact_keys):
            raise ValueError("requested Customer fact is not present in the frozen profile")
        profile_id = profile["_key"]
        if profile_id not in used_profiles: used_profiles.append(profile_id)
        information.append({"profile_id": profile_id, "fact_keys": req.fact_keys})
    preferences = []
    for req in challenge.required_customer_preferences:
        profile = assigned[req.profile_alias]
        if set(req.accepted_values) - set(profile.get("preference_space", {}).get(req.preference, [])):
            raise ValueError("requested Customer preference is outside the frozen profile choices")
        profile_id = profile["_key"]
        if profile_id not in used_profiles: used_profiles.append(profile_id)
        preferences.append({"profile_id": profile_id, "preference": req.preference, "accepted_values": req.accepted_values})

    event_ids = []
    for req in challenge.external_event_constraints:
        line = assigned[req.target_alias]
        matching = [e.id for e in world.external_events.values() if e.line_id == line["_key"] and e.kind == req.kind and e.outcome == req.outcome]
        if req.required and not matching:
            raise ValueError("required registered external event does not exist for grounded line")
        if not req.required and matching:
            raise ValueError("forbidden registered external event exists for grounded line")
        event_ids.extend(matching)

    if not used_profiles:
        raise ValueError("challenge objectives do not select a Customer scope")
    if len({world.profiles[pid].identity for pid in used_profiles}) != 1:
        raise ValueError("selected native Customer profiles must share one authenticated identity")
    frames, frame_hashes = [], set()
    for profile_id in used_profiles:
        private = world.profiles[profile_id]
        for frame in aftersales_tasks.frame_contract(world, private):
            key = digest(frame)
            if key not in frame_hashes:
                frame_hashes.add(key); frames.append(frame)
    profile_ids = used_profiles
    composition = OpenDialogueComposition(protocol="open-dialogue-composition-v3" if created else "open-dialogue-composition-v2",
        profile_ids=profile_ids, customer_request=challenge.customer_request, goals=goals, created_goals=created,
        refusal_operations=refusals, frames=frames, external_schedule=sorted(set(event_ids)),
        information_requirements=information, preference_requirements=preferences)
    return TaskSelection(kind="dialogue", family="open_composition_v1", profile_id=profile_ids[0], composition=composition)


def ground_challenge(challenge: AbstractChallenge | dict, index: PrivateWorldIndex, budget: GroundingBudget | dict | None = None) -> GroundingResult:
    row = challenge if isinstance(challenge, AbstractChallenge) else AbstractChallenge.model_validate(challenge)
    budget = budget if isinstance(budget, GroundingBudget) else GroundingBudget.model_validate(budget or {})
    result = GroundingResult(challenge_id=row.challenge_id, challenge_hash=row.challenge_hash, index_hash=index.index_hash, status="INTERNAL_ERROR")
    try:
        index.checked()
    except ValueError as exc:
        result.status = "INTERNAL_ERROR"; result.reason = str(exc)
        result.result_hash = result.computed_hash(); return result
    if row.domain != "aftersales":
        result.status = "OUT_OF_SCOPE"; result.reason = "only After-sales native grounding is registered in this version"
        result.result_hash = result.computed_hash(); return result
    try:
        challenge = validate_challenge(row)
    except ValueError as exc:
        result.status = "INVALID_CONSTRAINT"; result.reason = str(exc)
        result.result_hash = result.computed_hash(); return result
    worlds = [w for w in sorted(index.worlds, key=lambda x: x.registration_hash)
              if w.domain == challenge.domain and w.partition == budget.partition]
    if not worlds:
        result.status = "OUT_OF_SCOPE"; result.reason = "no registered world in the authorized partition"
        result.result_hash = result.computed_hash(); return result
    if len(worlds) > budget.max_worlds:
        worlds = worlds[:budget.max_worlds]; world_budget_limited = True
    else:
        world_budget_limited = False
    aliases = {v.alias: v.type for v in challenge.entity_variables}
    unsupported = []
    for objective in challenge.objectives:
        if objective.kind == "created_object_completion" and objective.collection != "cases": unsupported.append(objective.kind)
        if objective.kind == "existing_object_completion" and objective.collection != "lines": unsupported.append(objective.kind)
        if objective.kind == "correct_refusal" and objective.operation != "open_case": unsupported.append(objective.kind)
    if unsupported:
        result.status = "UNSUPPORTED_OBJECTIVE"; result.reason = ", ".join(sorted(set(unsupported)))
        result.result_hash = result.computed_hash(); return result

    explored = 0; matches: list[GroundedMatch] = []; truncated = False
    try:
        for indexed in worlds:
            frozen = verified_world(indexed.frozen)
            result.worlds_considered += 1
            candidate_rows = {v.alias: _rows(frozen, v.type) for v in challenge.entity_variables}
            # Apply unary constraints before the bounded deterministic join.
            for variable in challenge.entity_variables:
                filtered = []
                for candidate in candidate_rows[variable.alias]:
                    try:
                        if _matches_constraints(frozen, challenge, aliases, {variable.alias: candidate}):
                            filtered.append(candidate)
                    except ValueError as exc:
                        result.status = "INVALID_CONSTRAINT"; result.reason = str(exc)
                        result.result_hash = result.computed_hash(); return result
                candidate_rows[variable.alias] = filtered
            order = sorted((v.alias for v in challenge.entity_variables), key=lambda a: (len(candidate_rows[a]), a))
            if any(not candidate_rows[a] for a in order):
                continue
            assigned = {}
            def search(pos):
                nonlocal explored, truncated
                if len(matches) >= budget.max_matches:
                    truncated = True; return
                if pos == len(order):
                    try:
                        if not _matches_constraints(frozen, challenge, aliases, assigned): return
                        selection = _proposal(frozen, challenge, aliases, assigned)
                    except ValueError:
                        return
                    binding = {alias: assigned[alias]["_key"] for alias in sorted(assigned)}
                    binding_hash = digest({"registration_hash": frozen.registration_hash, "binding": binding})
                    matches.append(GroundedMatch(frozen_registration_hash=frozen.registration_hash,
                        frozen_world_hash=frozen.world_hash, binding=binding, binding_hash=binding_hash,
                        fact_sources=_source_facts(challenge, assigned), identity=frozen.world["profiles"][selection.profile_id]["identity"],
                        profile_ids=list(selection.composition.profile_ids), selection=selection))
                    return
                alias = order[pos]
                for candidate in candidate_rows[alias]:
                    if explored >= budget.max_bindings:
                        truncated = True; return
                    explored += 1
                    assigned[alias] = candidate
                    try:
                        feasible = _matches_constraints(frozen, challenge, aliases, assigned)
                    except ValueError:
                        del assigned[alias]
                        raise
                    if feasible: search(pos + 1)
                    del assigned[alias]
                    if truncated and (explored >= budget.max_bindings or len(matches) >= budget.max_matches): return
            search(0)
            if truncated: break
    except ValueError as exc:
        result.status = "INVALID_CONSTRAINT"; result.reason = str(exc)
        result.bindings_explored = explored; result.result_hash = result.computed_hash(); return result
    result.bindings_explored = explored; result.matches_found = len(matches); result.search_truncated = truncated or world_budget_limited
    if not matches:
        result.status = "BUDGET_EXHAUSTED" if result.search_truncated else "NO_MATCH"
        result.reason = "grounding budget reached before a binding was found" if result.status == "BUDGET_EXHAUSTED" else "no same-world binding satisfies all typed constraints"
    else:
        # Deterministic seeded order. The seed changes only tie-breaking among
        # proven fact matches, never the challenge or predicate semantics.
        matches.sort(key=lambda m: digest([budget.selection_seed, challenge.challenge_hash, m.frozen_registration_hash, m.binding_hash]))
        result.matches = matches
        result.selected_match_hash = matches[0].binding_hash
        result.status = "GROUNDED"
        result.reason = "fact matching only; native Reference certification is still required"
    result.result_hash = result.computed_hash()
    return result
