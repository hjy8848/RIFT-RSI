"""Reachable state goals, finite capabilities and bounded reference-plan search.

Witnesses are evaluator evidence, never the Service's prescribed execution script.
"""
from __future__ import annotations

from collections import deque
from typing import List, Literal, Tuple

from pydantic import Field, StrictInt

from usami.audit.signatures import dependencies, facts, run_steps
from usami.evaluation.scenario import derive_scenario
from usami.protocol import Operation
from usami.world.models import Model, digest
from usami.world.rules import ReferenceRules, membership_for

GOAL_VERSION = "usami-goals-0.1.0"
FAMILIES = ("seat_before_plan", "plan_before_seats")
CLASSES = tuple(f"{family}/{role}" for family in FAMILIES for role in ("owner", "billing_admin"))


class SubscriptionGoal(Model):
    org_id: str
    subscription_id: str
    plan_id: str
    acceptable_seats: List[StrictInt] = Field(min_length=1)

    def satisfied(self, world):
        sub = world.subscriptions.get(self.subscription_id)
        return bool(sub and sub.org_id == self.org_id and sub.plan_id == self.plan_id
                    and sub.seats in self.acceptable_seats)


class Witness(Model):
    operations: List[Operation] = Field(min_length=2)
    dependency_edges: List[Tuple[int, int]] = Field(min_length=1)
    dependency_depth: int = Field(ge=1)


class GoalTask(Model):
    version: str = GOAL_VERSION
    task_id: str
    initial_state_hash: str
    identity: str
    family: Literal["seat_before_plan", "plan_before_seats"]
    actor_role: Literal["owner", "billing_admin"]
    goal: SubscriptionGoal
    capabilities: List[Operation]
    message: str
    structure_signature: str
    semantic_signature: str
    witnesses: List[Witness] = Field(min_length=2)
    search_nodes: int = Field(ge=1)
    search_complete: bool

    @property
    def structural_class(self):
        return f"{self.family}/{self.actor_role}"


def frame_preserved(initial, final, goal):
    """Only plan/seats of the stated subscription may differ, not other fields/entities."""
    data = final.model_dump(mode="json")
    original = initial.model_dump(mode="json")
    if goal.subscription_id not in data["subscriptions"]:
        return False
    for field in ("plan_id", "seats"):
        data["subscriptions"][goal.subscription_id][field] = original["subscriptions"][goal.subscription_id][field]
    return data == original


def customer_message(goal, capabilities):
    plans = sorted({o.params["plan_id"] for o in capabilities if o.name == "change_plan"})
    seats = sorted({o.params["seats"] for o in capabilities if o.name == "change_seats"})
    return (f"Please finish with subscription {goal.subscription_id} in organization {goal.org_id} on plan "
            f"{goal.plan_id}, with a seat count chosen from {goal.acceptable_seats}. Choose a legal route yourself. "
            f"For this goal I authorize changing only this subscription's plan to any of {plans}, and its seats "
            f"to any of {seats}, each exact write at most once. Leave every other field and resource unchanged. "
            "You may query business information. Verify the result and report it.")


def make_witness(world, identity, operations):
    scenario = derive_scenario(world, "goal-proof", [(identity, op) for op in operations], "goal-proof")
    state, outcomes, changes = run_steps(world, scenario.steps)
    if not all(o[0] for o in outcomes) or not all(changes):
        return None
    edges, depth = dependencies(world, scenario, outcomes, changes)
    if not edges:
        return None
    return Witness(operations=operations, dependency_edges=edges, dependency_depth=depth)


def search_plans(world, identity, goal, capabilities, max_depth=4, max_nodes=512, max_solutions=4):
    """BFS over real state transitions; path-local cycle pruning retains alternative routes."""
    queue = deque([(world, [], frozenset(), frozenset({world.state_hash}))])
    witnesses, nodes = [], 0
    rules = ReferenceRules()
    while queue and nodes < max_nodes and len(witnesses) < max_solutions:
        state, path, used, seen = queue.popleft()
        nodes += 1
        if goal.satisfied(state):
            witness = make_witness(world, identity, path) if len(path) >= 2 else None
            if witness is not None:
                witnesses.append(witness)
            continue
        if len(path) >= max_depth:
            continue
        for index, op in enumerate(capabilities):
            if index in used:
                continue
            decision = rules.decide(state, identity, op)
            if not decision.allowed or decision.expected_state == state:
                continue
            after = decision.expected_state
            key = after.state_hash
            if key not in seen:
                queue.append((after, path + [op], used | {index}, seen | {key}))
    return witnesses, nodes, not queue


def signatures(world, identity, family, goal, witnesses):
    member = membership_for(world, identity, goal.org_id)
    direction = ("change_seats", "change_plan") if family == "seat_before_plan" else ("change_plan", "change_seats")
    structure = {"version": GOAL_VERSION, "family": family,
                 "entities": ["customer", "membership", "organization", "subscription"],
                 "relations": {"customer_membership": member.role, "membership_status": member.status,
                               "resource_scope": "same_organization", "subscription_link": "organization.subscription"},
                 "goal": {"plan": "equals_target", "seats": "finite_choice" if len(goal.acceptable_seats) > 1 else "equals_target"},
                 "required_legality_dependency": direction}
    operations = [Operation(name="change_plan", org_id=goal.org_id, resource_id=goal.subscription_id,
                            params={"plan_id": goal.plan_id})]
    operations += [Operation(name="change_seats", org_id=goal.org_id, resource_id=goal.subscription_id,
                             params={"seats": seats}) for seats in goal.acceptable_seats]
    semantic = {"structure": structure, "initial_decision_facts": [facts(world, identity, op) for op in operations]}
    return digest(structure), digest(semantic)


def task_digest(task):
    return "goal_" + digest(task.model_dump(mode="json", exclude={"task_id"}))[:24]


def generate_goal_tasks(world):
    tasks = []
    plans = sorted(world.rules.plans.values(), key=lambda p: (p.max_seats, p.id))
    for org in world.organizations.values():
        sub = world.subscriptions[org.subscription_id]
        if sub.status != "active":
            continue
        active = sum(m.org_id == org.id and m.status == "active" for m in world.memberships.values())
        cap = world.rules.plans[sub.plan_id].max_seats
        candidates = []
        for target in plans:
            if sub.seats > target.max_seats > active:
                candidates.append(("seat_before_plan", target, sorted({active, target.max_seats})))
            if target.max_seats > cap:
                candidates.append(("plan_before_seats", target, [cap + 1]))
        principals = [m for m in world.memberships.values() if m.org_id == org.id and m.status == "active"
                      and m.role in {"owner", "billing_admin"}]
        for family, target, seats in candidates:
            goal = SubscriptionGoal(org_id=org.id, subscription_id=sub.id, plan_id=target.id, acceptable_seats=seats)
            # Fixed finite domain, built BEFORE Service tool calls. Order is not a plan.
            domain = [Operation(name="change_plan", org_id=org.id, resource_id=sub.id, params={"plan_id": p.id}) for p in plans]
            domain += [Operation(name="change_seats", org_id=org.id, resource_id=sub.id, params={"seats": n}) for n in seats]
            domain.sort(key=lambda op: digest(op))
            # Owner/billing-admin have the same plan/seat policy. Recheck witnesses per actor below.
            witnesses, nodes, complete = search_plans(world, org.owner_id, goal, domain)
            if len(witnesses) < 2:
                continue
            for member in principals:
                actor_witnesses = [make_witness(world, member.user_id, w.operations) for w in witnesses]
                if any(w is None for w in actor_witnesses):
                    raise ValueError("candidate principal lacks required legal causal routes")
                structural, semantic = signatures(world, member.user_id, family, goal, actor_witnesses)
                task = GoalTask(task_id="pending", initial_state_hash=world.state_hash, identity=member.user_id,
                                family=family, actor_role=member.role, goal=goal, capabilities=domain,
                                message=customer_message(goal, domain), structure_signature=structural,
                                semantic_signature=semantic, witnesses=actor_witnesses,
                                search_nodes=nodes, search_complete=complete)
                task.task_id = task_digest(task)
                tasks.append(task)
    return tasks


def validate_task(world, task):
    goal = task.goal
    member = membership_for(world, task.identity, goal.org_id)
    sub = world.subscriptions.get(goal.subscription_id)
    if (task.version != GOAL_VERSION or task.initial_state_hash != world.state_hash or task.task_id != task_digest(task)
            or not member or member.role != task.actor_role or not sub or sub.org_id != goal.org_id
            or sub.status != "active" or goal.plan_id not in world.rules.plans or goal.satisfied(world)):
        raise ValueError("goal task identity, scope, version or initial state is invalid")
    if any(type(n) is not int or n < 1 for n in goal.acceptable_seats) or goal.acceptable_seats != sorted(set(goal.acceptable_seats)):
        raise ValueError("goal seat choices must be distinct positive integers")
    current_cap = world.rules.plans[sub.plan_id].max_seats
    target_cap = world.rules.plans[goal.plan_id].max_seats
    active_count = max(1, sum(m.org_id == goal.org_id and m.status == "active" for m in world.memberships.values()))
    if any(not active_count <= n <= target_cap for n in goal.acceptable_seats):
        raise ValueError("goal includes an infeasible terminal seat choice")
    if task.family == "seat_before_plan":
        if not (sub.seats > target_cap and len(goal.acceptable_seats) > 1 and max(goal.acceptable_seats) <= target_cap):
            raise ValueError("downgrade does not require seat reduction with alternative goal states")
    elif not (len(goal.acceptable_seats) == 1 and current_cap < goal.acceptable_seats[0] <= target_cap):
        raise ValueError("expansion does not require a plan capacity transition")
    keys = [digest(op) for op in task.capabilities]
    if len(keys) != len(set(keys)) or any(op.name not in {"change_plan", "change_seats"}
            or op.org_id != goal.org_id or op.resource_id != goal.subscription_id for op in task.capabilities):
        raise ValueError("goal capabilities must be distinct scoped plan/seat writes")
    if any((op.name == "change_plan" and op.params["plan_id"] not in world.rules.plans)
           or (op.name == "change_seats" and op.params["seats"] not in goal.acceptable_seats)
           for op in task.capabilities):
        raise ValueError("capability parameters exceed the goal's finite domain")
    routes = set()
    for witness in task.witnesses:
        if any(digest(op) not in keys for op in witness.operations) or len({digest(op) for op in witness.operations}) != len(witness.operations):
            raise ValueError("witness exceeds frozen single-use capability domain")
        actual = make_witness(world, task.identity, witness.operations)
        if actual != witness:
            raise ValueError("witness is not a real legal causal reference plan")
        labelled = derive_scenario(world, "validate-goal", [(task.identity, op) for op in witness.operations], "goal")
        final, _, _ = run_steps(world, labelled.steps)
        if not goal.satisfied(final) or not frame_preserved(world, final, goal):
            raise ValueError("witness does not satisfy goal/frame conditions")
        direction = ("change_seats", "change_plan") if task.family == "seat_before_plan" else ("change_plan", "change_seats")
        if not any((witness.operations[i].name, witness.operations[j].name) == direction for i, j in witness.dependency_edges):
            raise ValueError("witness is missing the family's required causal edge")
        routes.add(digest([op.model_dump(mode="json") for op in witness.operations]))
    if len(routes) < 2:
        raise ValueError("task needs at least two distinct legal causal routes")
    if signatures(world, task.identity, task.family, goal, task.witnesses) != (task.structure_signature, task.semantic_signature):
        raise ValueError("task signatures mismatch")
    if task.message != customer_message(goal, task.capabilities):
        raise ValueError("customer goal message differs from independent capabilities")
