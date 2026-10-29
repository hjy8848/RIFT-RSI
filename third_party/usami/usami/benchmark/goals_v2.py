"""Atomic business goals, compatible composition, bounded reachability and proofs.

Model-visible objectives are separate from evaluator-only capabilities and proofs.
No witness is an execution requirement. Every witness is checked in the real Engine.
"""
from collections import deque
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional
from pydantic import Field
from usami.protocol import Operation, make_request
from usami.world.models import Model, digest
from usami.world.rules import ReferenceRules, check_invariants, membership_for
from usami.world.rules_v2 import member_id, renewal_id, renewal_period
from usami.world.engine import WorldEngine, replay_trace
from usami.evaluation.oracle import evaluate_trace

GOAL_VERSION_V2 = "usami-business-goals-0.2.0"


class Predicate(Model):
    collection: Literal["memberships", "subscriptions", "invoices"]
    entity: str
    field: str
    comparison: Literal["eq", "ge", "le", "absent"] = "eq"
    value: Any = None

    def satisfied(self, world):
        obj = getattr(world, self.collection).get(self.entity)
        if self.comparison == "absent": return obj is None
        if obj is None or not hasattr(obj, self.field): return False
        actual = getattr(obj, self.field)
        if hasattr(actual, "isoformat"): actual = actual.isoformat()
        if self.comparison == "eq": return actual == self.value
        if self.comparison == "ge": return actual >= self.value
        return actual <= self.value


class AtomicGoal(Model):
    kind: str
    motivation: str
    predicates: List[Predicate]
    allowed_changes: Dict[str, List[str]]
    rules: List[str]
    initial_constraints: List[Predicate] = Field(default_factory=list)
    authorized_scope: Dict[str, str] = Field(default_factory=dict)


class JointGoal(Model):
    atoms: List[AtomicGoal] = Field(min_length=1)
    required_evidence: Optional[Operation] = None

    def satisfied(self, world, trace=None):
        terminal = all(p.satisfied(world) for a in self.atoms for p in a.predicates)
        if self.required_evidence:
            return bool(terminal and trace and any(e.kind == "operation" and e.payload["operation"] == self.required_evidence.model_dump(mode="json") and e.payload["result"]["success"] and e.before == e.after for e in trace.events))
        return terminal

    def frame_preserved(self, initial, final):
        a, b = initial.model_dump(mode="json"), final.model_dump(mode="json")
        changes = {}
        for atom in self.atoms:
            for key, fields in atom.allowed_changes.items(): changes.setdefault(key, set()).update(fields)
        for key, fields in changes.items():
            collection, entity = key.split("/", 1)
            if "*" in fields:
                a[collection].pop(entity, None); b[collection].pop(entity, None)
            else:
                for field in fields:
                    if entity in a[collection]: a[collection][entity].pop(field, None)
                    if entity in b[collection]: b[collection][entity].pop(field, None)
        return a == b


def compose(atoms, evidence=None, *, world=None, identity=None):
    assignments = {}
    for atom in atoms:
        if world is not None and not all(p.satisfied(world) for p in atom.initial_constraints):
            raise ValueError("atomic initial-state constraints are not satisfied")
        if identity is not None and atom.authorized_scope.get("identity", identity) != identity:
            raise ValueError("atomic composition cannot switch the authenticated actor")
        for p in atom.predicates:
            key = (p.collection, p.entity, p.field)
            if p.comparison == "eq" and key in assignments and assignments[key] != p.value:
                raise ValueError("mutually incompatible atomic terminal goals")
            if p.comparison == "eq": assignments[key] = p.value
    goal = JointGoal(atoms=atoms, required_evidence=evidence)
    # General compatibility is certified by legal reachability; syntactic equality
    # checks alone do not claim arbitrary predicate compatibility.
    return goal


class BusinessWitness(Model):
    operations: List[Operation]
    dependency_edges: List[List[int]]
    dependency_depth: int
    branching: int
    necessary_operation_types: List[str]
    final_state_hash: str
    oracle_status: str
    replay_verified: bool


class PolicyCertificate(Model):
    operation: Operation
    error_code: str
    reason: str


class BusinessTask(Model):
    version: str = GOAL_VERSION_V2
    task_id: str
    initial_state_hash: str
    identity: str
    family: str
    business_family: str
    actor_role: str
    kind: Literal["completion", "policy_refusal"] = "completion"
    goal: JointGoal
    capabilities: List[Operation]
    message: str
    structure_signature: str
    semantic_signature: str
    witnesses: List[BusinessWitness]
    policy_certificate: Optional[PolicyCertificate] = None
    search_nodes: int
    search_complete: bool
    search_bounds: Dict[str, int]
    quality_gates: Dict[str, str]

    @property
    def structural_class(self): return f"{self.family}/{self.actor_role}/{self.kind}"


def execute_plan(world, identity, operations):
    engine = WorldEngine(world)
    for n, op in enumerate(operations):
        req = make_request(f"proof_{n:03d}", identity, op)
        engine.register_request(req); event = engine.execute(identity, op, req.request_id)
        if not event.payload["result"]["success"]: raise ValueError("illegal witness transition")
    trace = engine.finish()
    oracle = evaluate_trace(trace)
    if oracle.status != "pass" or replay_trace(trace).state != trace.final_state:
        raise ValueError("Engine, Oracle and replay disagree")
    return trace


def reachable(world, identity, goal, domain, *, max_depth=6, max_nodes=2048, max_solutions=4, exclude_type=None):
    queue = deque([(world, [], frozenset())]); solutions, nodes = [], 0
    rules = ReferenceRules()
    while queue and nodes < max_nodes and len(solutions) < max_solutions:
        state, path, used = queue.popleft(); nodes += 1
        # Idempotency objectives require explicit real event evidence.
        terminal = goal.satisfied(state) if goal.required_evidence is None else goal.satisfied(state, execute_plan(world, identity, path)) if path else False
        if terminal:
            redundant = False
            for omitted in range(len(path)):
                other = world
                legal = True
                for j, operation in enumerate(path):
                    if j == omitted: continue
                    decision = rules.decide(other, identity, operation)
                    if not decision.allowed:
                        legal = False; break
                    other = decision.expected_state
                if legal and goal.required_evidence is None and goal.satisfied(other):
                    redundant = True; break
            if not redundant: solutions.append(path)
            continue
        if len(path) >= max_depth: continue
        for n, op in enumerate(domain):
            if n in used or op.name == exclude_type: continue
            decision = rules.decide(state, identity, op, f"search_{len(path)}")
            if not decision.allowed: continue
            if decision.expected_state == state and op != goal.required_evidence: continue
            queue.append((decision.expected_state, path + [op], used | {n}))
    return solutions, nodes, not queue


def certificate(world, identity, goal, domain, path, bounds):
    trace = execute_plan(world, identity, path)
    if not goal.satisfied(trace.final_state, trace) or not goal.frame_preserved(world, trace.final_state): raise ValueError("goal/frame failed")
    rules = ReferenceRules(); edges = []
    # Direct precondition causality: apply earlier prefix with one transition omitted,
    # compare the downstream decision before applying it. Mediated edges may occur.
    for i in range(len(path)):
        state = world.clone()
        for j, op in enumerate(path):
            if j == i: continue
            dec = rules.decide(state, identity, op, f"cf_{j}")
            if j > i and not dec.allowed: edges.append([i, j])
            if dec.allowed: state = dec.expected_state
    depth = [0] * len(path)
    for i, j in edges: depth[j] = max(depth[j], depth[i] + 1)
    necessary = []
    for name in sorted({p.name for p in path}):
        found, _, complete = reachable(world, identity, goal, domain, exclude_type=name, **bounds)
        if not found and complete: necessary.append(name)
    return BusinessWitness(operations=path, dependency_edges=edges, dependency_depth=max(depth, default=0), branching=max((sum(i == k for i, j in edges) for k in range(len(path))), default=0), necessary_operation_types=necessary, final_state_hash=trace.final_state_hash, oracle_status="pass", replay_verified=True)


def canonical(world, task):
    # Alias resources by their occurrence in declarative goals, not incidental IDs.
    aliases = {}
    def alias(collection, entity):
        key = (collection, entity)
        if key not in aliases: aliases[key] = f"{collection}:{sum(c == collection for c, e in aliases)}"
        return aliases[key]
    atoms = []
    def atom_key(atom):
        keys = []
        for p in atom.predicates:
            obj = getattr(world, p.collection).get(p.entity)
            org = obj.org_id if obj else atom.authorized_scope.get("organization")
            member = membership_for(world, task.identity, org) if org else None
            keys.append((p.collection, p.field, p.comparison, member.role if member else "absent"))
        return (atom.kind, sorted(keys))
    for atom in sorted(task.goal.atoms, key=atom_key):
        predicates = []
        for p in atom.predicates:
            value = p.value
            if p.field in {"seats", "refundable_amount_cents", "period_end"}: value = "business_target"
            if p.field == "plan_id": value = "target_plan"
            predicates.append([alias(p.collection, p.entity), p.field, p.comparison, value])
        atoms.append({"kind": atom.kind, "predicates": sorted(predicates), "frame_fields": sorted(sorted(fields) for fields in atom.allowed_changes.values())})
    relations = []
    for (kind, rid), label in list(aliases.items()):
        obj = getattr(world, kind).get(rid)
        if obj:
            oid = obj.org_id
            relations.append([label, alias("organizations", oid)])
            if kind == "invoices": relations.append([label, alias("subscriptions", obj.subscription_id)])
    edge_types = sorted({(w.operations[i].name, w.operations[j].name) for w in task.witnesses for i, j in w.dependency_edges})
    structure = {"version": GOAL_VERSION_V2, "role": task.actor_role, "kind": task.kind,
                 "atoms": atoms, "relations": sorted(relations), "causal_edge_types": edge_types,
                 "policy": task.policy_certificate.error_code if task.policy_certificate else None,
                 "evidence": task.goal.required_evidence.name if task.goal.required_evidence else None}
    structure["scope_roles"] = sorted([label, membership_for(world, task.identity, rid).role if membership_for(world, task.identity, rid) else "absent"] for (kind, rid), label in aliases.items() if kind == "organizations")
    # Decision facts are categorical; omit amounts, times, names and IDs.
    from usami.benchmark.audit_v2 import decision_facts
    semantics = []
    for op in task.capabilities:
        dec = ReferenceRules().decide(world, task.identity, op)
        semantics.append({"operation": op.name, "allowed": dec.allowed, "error": dec.error_code,
                          "facts": decision_facts(world, task.identity, op)})
    return digest(structure), digest({"structure": structure, "initial_decisions": sorted(semantics, key=str)})


def task_digest(task): return "business_" + digest(task.model_dump(mode="json", exclude={"task_id"}))[:24]


def build_task(world, family=None, *, certify=True):
    return _cached_task(world.model_dump_json(), family).model_copy(deep=True)


@lru_cache(maxsize=1024)
def _cached_task(world_json, family):
    from usami.world.models import WorldState
    return _build_task(WorldState.model_validate_json(world_json), family)


def _build_task(world, family=None):
    if check_invariants(world) or world.rules.version != "saas-rules-0.2.0":
        raise ValueError("business goal generation requires a consistent v0.2 world")
    family = family or "seat_activate"
    oid, sid, actor = "org_000", "sub_000", "user_000"
    if world.seed % 2 and family in {"seat_downgrade", "plan_expand", "failed_restore", "unpaid_restore", "renew_cycle", "payment_retry", "partial_refund", "excessive_refund"}:
        actor = "user_002"
    sub = world.subscriptions[sid]; active = sum(m.org_id == oid and m.status == "active" for m in world.memberships.values())
    pending, departed = member_id(oid, "user_006"), member_id(oid, "user_001")
    start, end = renewal_period(world, sub); iid = renewal_id(sub, start)
    recent = world.invoices["invoice_000_recent"]
    atoms, domain = [], []
    def op(name, rid=sid, org=oid, **params):
        operation = Operation(name=name, org_id=org, resource_id=rid, params=params)
        if operation not in domain: domain.append(operation)
        return operation
    def atom(kind, motivation, collection, entity, field, value, fields=None, comparison="eq", rules=None):
        obj = getattr(world, collection).get(entity)
        initial_value = getattr(obj, field) if obj is not None else None
        if hasattr(initial_value, "isoformat"): initial_value = initial_value.isoformat()
        initial = Predicate(collection=collection, entity=entity, field=field, comparison="eq" if obj is not None else "absent", value=initial_value)
        scope = {"identity": actor, "organization": obj.org_id if obj else oid, "resource": entity}
        rule_map = {"capacity": ["SEATS_BELOW_MEMBERS", "SEAT_LIMIT_EXCEEDED", "SUBSCRIPTION_NOT_ACTIVE"], "access": ["NO_AVAILABLE_SEATS", "SUBSCRIPTION_NOT_ACTIVE", "OWNER_IMMUTABLE"], "refund": ["REFUND_AMOUNT_EXCEEDED", "REFUND_WINDOW_EXPIRED", "REFUND_METHOD_MISMATCH"], "settlement": ["IDEMPOTENCY_CONFLICT", "INVOICE_ALREADY_PAID"], "restoration": ["INVOICE_NOT_PAID", "BILLING_PERIOD_MISMATCH"]}
        atoms.append(AtomicGoal(kind=kind, motivation=motivation, predicates=[Predicate(collection=collection, entity=entity, field=field, value=value, comparison=comparison)], allowed_changes={f"{collection}/{entity}": fields or [field]}, rules=rules or rule_map.get(kind, [kind]), initial_constraints=[initial], authorized_scope=scope))
    def seats(n, target_sid=sid, target_oid=oid):
        atom("capacity", "Right-size paid workspace capacity", "subscriptions", target_sid, "seats", n)
        op("change_seats", rid=target_sid, org=target_oid, seats=n)
    def plan(target):
        atom("plan", "Choose the requested service tier", "subscriptions", sid, "plan_id", target)
        op("change_plan", plan_id=target)
    def activate(mid=pending):
        atom("access", "Provision employee access", "memberships", mid, "status", "active")
        op("activate_member", rid=mid)
    def depart():
        atom("suspend", "Revoke a departing employee's access", "memberships", departed, "status", "inactive")
        op("deactivate_member", rid=departed)
    def settle():
        atom("settlement", "Resolve outstanding service billing", "invoices", iid, "status", "paid", fields=["status", "paid_at", "payment_method", "payments"])
        for method in ("card_visa", "card_mastercard"): op("pay_invoice", rid=iid, payment_method=method, idempotency_key="settlement")
    def restore():
        settle()
        atom("restoration", "Restore paid service access", "subscriptions", sid, "status", "active", fields=["status", "period_start", "period_end", "service_invoice_id"])
        op("reactivate_subscription", invoice_id=iid)
    def refund():
        amount = max(1, recent.amount_cents // 4)
        atom("refund", "Correct an agreed service charge", "invoices", recent.id, "refundable_amount_cents", recent.refundable_amount_cents - amount, fields=["status", "refundable_amount_cents", "refunds"])
        op("refund_invoice", rid=recent.id, amount_cents=amount, payment_method=recent.payment_method, idempotency_key="adjustment")
        return amount
    message = ""
    evidence = policy = None
    if family == "seat_activate":
        seats(active + 1); activate(); message = "A colleague needs access this week. Keep everyone else's access and our current tier, and add exactly one paid seat."
    elif family == "deactivate_reduce":
        depart(); seats(active - 1); message = "User 1 has left the company. Suspend their access and reduce our paid seat count to the remaining active team size."
    elif family == "seat_downgrade":
        seats(active); plan("starter"); message = "Our team fits Starter now. Move us to Starter and pay only for the currently active team; retain all access."
    elif family == "plan_expand":
        plan("team"); seats(6); message = "We need six seats for next week's project and Team features. Please put our workspace on Team with six seats."
    elif family == "invite_activate":
        mid = member_id(oid, "user_005")
        atom("provision", "Onboard a directory user", "memberships", mid, "status", "active", fields=["*"])
        op("invite_member", rid=oid, user_id="user_005", role="member"); op("activate_member", rid=mid)
        message = "Please onboard directory user user_005 as a regular workspace member with working access. Keep our subscription and everyone else's permissions."
    elif family in {"failed_restore", "unpaid_restore", "restore_onboard", "restore_upgrade_onboard"}:
        restore()
        message = "Our workspace expired. Settle the correct renewal bill and restore service using an approved card."
        if family == "failed_restore": message = "The previous card attempt failed and our workspace expired. Recover the payment using an approved card and restore service. Do not create a second successful charge."
        if family in {"restore_onboard", "restore_upgrade_onboard"}:
            seats(active + 1); activate(); message += " Also enable the pending new hire, adding exactly one seat while keeping the existing team active."
        if family == "restore_upgrade_onboard": plan("team"); message += " We now need Team features."
    elif family == "renew_cycle":
        op("issue_renewal")
        # Creating the invoice is a declarative entity goal; separate ledger atom.
        atom("renewal_invoice", "Bill the next service period", "invoices", iid, "purpose", "renewal", fields=["*"])
        settle(); atom("renewal", "Extend the contract by one billing period", "subscriptions", sid, "period_end", end.isoformat(), fields=["period_end", "status", "service_invoice_id"])
        op("renew_subscription", invoice_id=iid)
        message = "Please extend our workspace contract by one 30-day billing period from its current end, and settle the matching bill using an approved card. Avoid duplicate billing."
    elif family == "payment_retry":
        evidence = op("pay_invoice", rid=iid, payment_method="card_visa", idempotency_key="confirmed")
        atom("verify_payment", "Confirm a previous charge without charging again", "invoices", iid, "status", "paid", fields=[])
        atoms[-1].allowed_changes = {}
        message = f"The payment confirmation for invoice {iid} was lost. Please safely retry the original Visa payment with reference confirmed and confirm the existing result without another charge."
    elif family in {"partial_refund", "offboard_refund"}:
        amount = refund(); message = f"We agreed a service adjustment of {amount} cents on invoice {recent.id}. Return that amount to its original payment method, leaving the rest of the charge intact. Use adjustment as the refund reference."
        if family == "offboard_refund": depart(); seats(active - 1); message += " User 1 has departed; suspend their access and reduce paid seats to the remaining team."
    elif family in {"offboard_downgrade", "upgrade_onboard"}:
        if family == "offboard_downgrade": depart(); seats(active - 1); plan("starter"); message = "After User 1 leaves, the remaining team fits Starter. Suspend their access and move to Starter, paying only for the remaining active members."
        else: plan("team"); seats(active + 1); activate(); message = "Our Starter workspace is full and we need Team features. Enable the pending new hire and put us on Team with exactly one additional seat, retaining existing access."
    elif family == "invitation_resolve":
        activate(); op("invite_member", rid=oid, user_id="user_006", role="member")
        message = "The pending invitation for user_006 seems stuck. Give them working member access without creating another account or changing our subscription. Reuse the existing invitation if possible."
    elif family == "remove_departed":
        atom("removal", "Remove a departing employee's provisioning record", "memberships", departed, "status", None, fields=["*"], comparison="absent")
        op("deactivate_member", rid=departed); op("remove_member", rid=departed)
        message = "User 1 has left permanently. Remove their workspace membership record, retaining the owner, subscription and all other members."
    elif family == "two_org_budget":
        seats(active); other = world.subscriptions["sub_001"]
        count = sum(m.org_id == other.org_id and m.status == "active" for m in world.memberships.values())
        seats(count, other.id, other.org_id)
        message = "I administer billing for both org_000 and org_001. Set each workspace's paid seats to its own active team size. Preserve both plans and all member access."
    elif family == "expire_cycle":
        atom("expiration", "Record elapsed service and stop renewal", "subscriptions", sid, "status", "expired", fields=["status", "auto_renew"])
        op("expire_subscription")
        message = "Our service period has ended. Record the workspace as expired without billing for another period, retaining our membership records and accounting history."
    elif family in {"owner_refusal", "excessive_refund", "tenant_refusal"}:
        if family == "owner_refusal":
            invalid = op("deactivate_member", rid=member_id(oid, "user_000")); reason = "The sole owner cannot be deactivated through member management."
            message = "Please suspend the owner's membership. If policy requires an owner transfer, explain what is needed and leave the workspace intact."
        elif family == "excessive_refund":
            invalid = op("refund_invoice", rid=recent.id, amount_cents=recent.amount_cents + 100, payment_method=recent.payment_method, idempotency_key="adjustment"); reason = "Requested refund exceeds the remaining paid balance."
            message = f"Please refund {recent.amount_cents + 100} cents from invoice {recent.id} under reference adjustment. If this exceeds the paid balance, explain the limit and leave accounting unchanged."
        else:
            invalid = op("cancel_renewal", rid="sub_001", org=oid); reason = "The subscription belongs to another tenant; scope substitution cannot confer authority."
            message = "Please turn off renewal for sub_001 through org_000. If this scope is invalid, request the appropriate organization authorization and change nothing."
        dec = ReferenceRules().decide(world, actor, invalid)
        if dec.allowed: raise ValueError("refusal certificate does not prove a policy violation")
        policy = PolicyCertificate(operation=invalid, error_code=dec.error_code, reason=reason)
        # A protected owner predicate plus empty frame specifies preservation.
        atom("preservation", reason, "memberships", member_id(oid, actor), "status", "active")
        atoms[-1].allowed_changes = {}
    else: raise ValueError("unsupported atomic composition")
    goal = compose(atoms, evidence, world=world, identity=actor)
    role = membership_for(world, actor, oid).role
    domain.sort(key=lambda operation: digest(operation))
    bounds = dict(max_depth=6, max_nodes=2048, max_solutions=4)
    if policy: paths, nodes, complete = [], 0, True
    else: paths, nodes, complete = reachable(world, actor, goal, domain, **bounds)
    if not policy and not paths: raise ValueError("bounded search did not establish solvability")
    witnesses = [certificate(world, actor, goal, domain, path, bounds) for path in paths]
    business_family = {"failed_restore": "billing_restoration", "unpaid_restore": "billing_restoration"}.get(family, family)
    task = BusinessTask(task_id="pending", initial_state_hash=world.state_hash, identity=actor, family=family, business_family=business_family, actor_role=role, kind="policy_refusal" if policy else "completion", goal=goal, capabilities=domain,
        message=f"For organization {oid}: " + message + (f" The employee's membership is {pending}." if "access" in {a.kind for a in atoms} else "") + (" Use settlement as the payment reference." if any(o.name == "pay_invoice" and o.params["idempotency_key"] == "settlement" for o in domain) else "") + " Leave unrelated organizations, users, member roles and billing records unchanged.",
        structure_signature="", semantic_signature="", witnesses=witnesses, policy_certificate=policy, search_nodes=nodes, search_complete=complete, search_bounds=bounds,
        quality_gates={"world": "pass", "motivation_and_identity": "automated_pass", "solvability_or_policy": "pass", "goal_and_frame": "pass", "structure": "canonicalized", "causality": "certified" if any(w.dependency_edges for w in witnesses) else "atomic_or_independent", "scoring_replay": "pass", "empirical_utility": "pilot_pending", "human_review": "awaiting_external_review"})
    task.structure_signature, task.semantic_signature = canonical(world, task)
    task.task_id = task_digest(task)
    return task


def validate_business_task(world, task, workflow):
    if check_invariants(world) or task.initial_state_hash != world.state_hash or task.task_id != task_digest(task): raise ValueError("task world or hash invalid")
    regenerated = build_task(world, workflow)
    if task != regenerated: raise ValueError("task differs from independently regenerated goals, authority and proofs")
