"""Policy-relative signatures and reference counterfactuals (no LLM scoring)."""
from __future__ import annotations

from collections import Counter
from datetime import timedelta
from itertools import permutations
import json

from usami.world.models import digest
from usami.world.rules import ReferenceRules

SIGNATURE_VERSION = "usami-audit-signatures-1"
COLLECTIONS = ("users", "organizations", "subscriptions", "invoices", "memberships")


def cmp(a, b):
    return "lt" if a < b else "eq" if a == b else "gt"


def invoice_class(world, inv):
    if inv.status == "refunded" or inv.refunds:
        return "already_refunded"
    if inv.status != "paid":
        return "unpaid"
    return "stale" if world.clock - inv.paid_at > timedelta(days=world.rules.refund_window_days) else "eligible"


def resource_type(op):
    return ("organizations" if op.name in {"get_organization", "list_members", "invite_member"} else
            "invoices" if op.name in {"refund_invoice", "get_invoice", "pay_invoice"} else
            "memberships" if op.name in {"change_member_role", "activate_member", "deactivate_member", "remove_member"} else "subscriptions")


def facts(world, actor, op):
    if world.rules.version == "saas-rules-0.2.0":
        from usami.benchmark.audit_v2 import decision_facts
        return decision_facts(world, actor, op)
    membership = next((m for m in world.memberships.values() if m.user_id == actor and m.org_id == op.org_id), None)
    result = {"actor_membership": [membership.role, membership.status] if membership else "absent"}
    obj = getattr(world, resource_type(op)).get(op.resource_id)
    if obj is None:
        return dict(result, resource="missing")
    result["resource_scope_matches"] = (obj.id if resource_type(op) == "organizations" else obj.org_id) == op.org_id
    if resource_type(op) == "subscriptions":
        active = max(1, sum(m.org_id == obj.org_id and m.status == "active" for m in world.memberships.values()))
        cap = world.rules.plans[obj.plan_id].max_seats
        result["subscription"] = {"status": obj.status, "auto_renew": obj.auto_renew,
                                  "seats_vs_members": cmp(obj.seats, active), "seats_vs_capacity": cmp(obj.seats, cap)}
        if op.name == "change_plan":
            target = world.rules.plans.get(op.params["plan_id"])
            result["parameter"] = {"plan_exists": target is not None}
            if target:
                result["parameter"].update(capacity_vs_current=cmp(target.max_seats, cap),
                                           capacity_vs_seats=cmp(target.max_seats, obj.seats),
                                           capacity_vs_members=cmp(target.max_seats, active))
        elif op.name == "change_seats":
            result["parameter"] = {"seats_vs_members": cmp(op.params["seats"], active),
                                   "seats_vs_capacity": cmp(op.params["seats"], cap),
                                   "seats_vs_current": cmp(op.params["seats"], obj.seats)}
    elif resource_type(op) == "invoices":
        result["invoice"] = {"class": invoice_class(world, obj), "status": obj.status,
                             "history_present": bool(obj.refunds),
                             "payment_age_vs_window": None if obj.paid_at is None else
                             cmp(world.clock - obj.paid_at, timedelta(days=world.rules.refund_window_days))}
        if op.name == "refund_invoice":
            result["parameter"] = {"amount_vs_full": cmp(op.params["amount_cents"], obj.refundable_amount_cents),
                                   "method_matches": op.params["payment_method"] == obj.payment_method}
    elif resource_type(op) == "memberships":
        result["membership"] = {"role": obj.role, "status": obj.status, "actor_is_target": actor == obj.user_id}
        result["parameter"] = op.params["role"]
    return result


def transition(before, after, op):
    a = getattr(before, resource_type(op)).get(op.resource_id)
    b = getattr(after, resource_type(op)).get(op.resource_id)
    if a is None or b is None:
        return []
    changed = []
    for field in ("plan_id", "seats", "auto_renew", "role", "status", "refunds"):
        if hasattr(a, field) and getattr(a, field) != getattr(b, field):
            av, bv = getattr(a, field), getattr(b, field)
            if field == "plan_id":
                value = cmp(before.rules.plans[bv].max_seats, before.rules.plans[av].max_seats)
            elif field == "seats":
                value = cmp(bv, av)
            elif field == "refunds":
                value = "history_added"
            else:
                value = [av, bv]
            changed.append([field, value])
    return changed


def relation_graph(world, scenario):
    aliases, queue, nodes = {}, [], []

    def alias(kind, identifier):
        key = (kind, identifier)
        if key not in aliases:
            aliases[key] = f"{kind}:{sum(k[0] == kind for k in aliases)}"
            queue.append(key)
        return aliases[key]

    operations = []
    for step in scenario.steps:
        req, op = step.request, step.request.operation
        operations.append({"operation": op.name, "actor": alias("users", req.requester_id),
                           "scope": alias("organizations", op.org_id),
                           "resource": alias(resource_type(op), op.resource_id)})
    for kind, identifier in queue:
        obj = getattr(world, kind).get(identifier)
        node = {"entity": aliases[(kind, identifier)], "exists": obj is not None}
        if obj is not None:
            if kind == "organizations":
                node.update(owner=alias("users", obj.owner_id), subscription=alias("subscriptions", obj.subscription_id))
            elif kind in {"subscriptions", "invoices", "memberships"}:
                node["organization"] = alias("organizations", obj.org_id)
                if kind == "memberships":
                    node["user"] = alias("users", obj.user_id)
        nodes.append(node)
    # Preserve actor roles in every involved organization, including foreign scope.
    actor_ids = list(dict.fromkeys(s.request.requester_id for s in scenario.steps))
    organizations = [key for key in aliases if key[0] == "organizations"]
    roles = []
    for actor in actor_ids:
        for _, org in organizations:
            membership = next((m for m in world.memberships.values() if m.user_id == actor and m.org_id == org), None)
            roles.append([alias("users", actor), alias("organizations", org),
                          [membership.role, membership.status] if membership else "absent"])
    return {"operations": operations, "entities": nodes, "actor_relations": roles}


def run_steps(world, steps, omitted=None):
    state, outcomes, changes = world, [], []
    rules = ReferenceRules()
    for index, step in enumerate(steps):
        if index == omitted:
            outcomes.append(None)
            changes.append(False)
            continue
        req = step.request
        decision = rules.decide(state, req.requester_id, req.operation, f"event_{2 * index + 2:06d}")
        allowed = decision.allowed and req.authorizes(req.requester_id, req.operation)
        error = None if allowed else decision.error_code if req.authorizes(req.requester_id, req.operation) else "UNAUTHORIZED_OPERATION"
        after = decision.expected_state if allowed else state
        outcomes.append([allowed, error])
        changes.append(after != state)
        state = after
    return state, outcomes, changes


def dependencies(world, scenario, outcomes, changes):
    edges = []
    for i in range(len(scenario.steps) - 1):
        if not changes[i]:
            continue
        _, other, _ = run_steps(world, scenario.steps, omitted=i)
        edges.extend([i, j] for j in range(i + 1, len(other)) if other[j] != outcomes[j])
    depths = [0] * len(outcomes)
    for i, j in edges:
        depths[j] = max(depths[j], depths[i] + 1)
    return edges, max(depths, default=0)


def business_snapshot(world):
    """Same world/order comparison only; remove execution-nonce refund IDs."""
    data = world.model_dump(mode="json", exclude={"world_id", "seed", "config_hash", "generator_version"})
    for invoice in data["invoices"].values():
        for refund in invoice["refunds"]:
            refund.pop("id")
    return data


def outcome_bag(steps, outcomes):
    return Counter(json.dumps([s.request.requester_id, s.request.operation.model_dump(mode="json"), o], sort_keys=True)
                   for s, o in zip(steps, outcomes))


def analyze_scenario(world, scenario, observer):
    state, records, semantic_steps, outcomes, changes = world, [], [], [], []
    for i, step in enumerate(scenario.steps):
        req, op = step.request, step.request.operation
        decision, guards = observer.decide(state, req.requester_id, op, f"event_{2 * i + 2:06d}")
        authorized = req.authorizes(req.requester_id, op)
        allowed = authorized and decision.allowed
        error = None if allowed else decision.error_code if authorized else "UNAUTHORIZED_OPERATION"
        after = decision.expected_state if allowed else state
        if (allowed != step.expected_allowed or error != step.expected_error_code or after.state_hash != step.expected_state_hash):
            raise ValueError("sample label/transition differs from reference")
        record = dict(operation=op.name, allowed=allowed, policy_allowed=decision.allowed, error_code=error,
                      guards=guards, potential=observer.potential_vetoes(state, req.requester_id, op))
        records.append(record)
        semantic_steps.append({"before": facts(state, req.requester_id, op), "authorized": authorized,
                               "outcome": [allowed, error], "policy_outcome": [decision.allowed, decision.error_code],
                               "guards": guards, "transition": transition(state, after, op),
                               "after": facts(after, req.requester_id, op)})
        outcomes.append([allowed, error])
        changes.append(after != state)
        state = after
    edges, depth = dependencies(world, scenario, outcomes, changes)
    graph = relation_graph(world, scenario)
    template = dict(version=SIGNATURE_VERSION, relations=graph, legality_dependencies=edges)
    semantic = dict(template=template, steps=semantic_steps)
    order = {"decision_sensitive": False, "final_state_sensitive": False}
    if len(scenario.steps) > 1:
        reversed_steps = list(reversed(scenario.steps))
        reversed_state, reversed_outcomes, _ = run_steps(world, reversed_steps)
        order = {"decision_sensitive": outcome_bag(scenario.steps, outcomes) != outcome_bag(reversed_steps, reversed_outcomes),
                 "final_state_sensitive": business_snapshot(state) != business_snapshot(reversed_state)}
    orgs = {s.request.operation.org_id for s in scenario.steps}
    for s in scenario.steps:
        op = s.request.operation
        obj = getattr(world, resource_type(op)).get(op.resource_id)
        if obj is not None:
            orgs.add(obj.id if resource_type(op) == "organizations" else obj.org_id)
    subsystem = {"subscriptions": "subscription", "memberships": "membership", "invoices": "billing", "organizations": "organization"}
    complexity = {"steps": len(records), "distinct_operations": len({r["operation"] for r in records}),
                  "related_entities": len(graph["entities"]), "organizations": len(orgs),
                  "subsystems": len({subsystem[resource_type(s.request.operation)] for s in scenario.steps}),
                  "business_guards_visited": len({g for r in records for g, _ in r["guards"] if int(g[1:]) > 7}),
                  "dependencies": len(edges), "depth": depth, "cross_organization": len(orgs) > 1,
                  "mixed_allowed_rejected": any(o[0] for o in outcomes) and not all(o[0] for o in outcomes), **order}
    return {"template": template, "semantic": semantic, "template_id": digest(template), "semantic_id": digest(semantic),
            "records": records, "complexity": complexity, "dependency_edges": edges}


def world_signature(world, include_state=False):
    """Exact ID-free bipartite incidence canonicalization for <=6 organizations.

    Anonymous user rows are sorted; enumerate org permutations. Policy state is
    coarsened. Larger graphs are explicitly Unknown rather than heuristically exact.
    """
    if len(world.organizations) > 6:
        return None
    membership = {(m.user_id, m.org_id): [m.role, m.status] for m in world.memberships.values()}
    candidates = []
    for order in permutations(world.organizations):
        rows = sorted([membership.get((u, org), ["absent", "absent"]) for org in order] for u in world.users)
        data = {"memberships": rows}
        if include_state:
            states = []
            for org in order:
                sub = world.subscriptions[world.organizations[org].subscription_id]
                active = max(1, sum(m.org_id == org and m.status == "active" for m in world.memberships.values()))
                states.append({"status": sub.status, "capacity": world.rules.plans[sub.plan_id].max_seats,
                               "auto_renew": sub.auto_renew, "seats_vs_members": cmp(sub.seats, active),
                               "seats_vs_capacity": cmp(sub.seats, world.rules.plans[sub.plan_id].max_seats),
                               "invoices": dict(sorted(Counter(invoice_class(world, i) for i in world.invoices.values() if i.org_id == org).items()))})
            data["organization_states"] = states
        candidates.append(json.dumps(data, sort_keys=True, separators=(",", ":")))
    return digest(min(candidates))
