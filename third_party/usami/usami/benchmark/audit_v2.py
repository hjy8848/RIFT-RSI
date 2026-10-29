"""Measured diversity, real decision branches and leakage for business goals."""
from collections import Counter
from usami.audit.runner import diversity
from usami.audit.signatures import resource_type, cmp
from usami.world.rules import membership_for, ReferenceRules


def decision_facts(world, actor, op):
    member = membership_for(world, actor, op.org_id)
    result = {"actor_role": member.role if member else "absent", "operation": op.name}
    kind = resource_type(op); obj = getattr(world, kind).get(op.resource_id)
    if not obj: return dict(result, resource="missing")
    result["scope_matches"] = (obj.id if kind == "organizations" else obj.org_id) == op.org_id
    if hasattr(obj, "status"): result["status"] = obj.status
    if kind == "memberships": result.update(target_role=obj.role, actor_is_target=obj.user_id == actor)
    if kind == "subscriptions":
        plan = world.rules.plans.get(obj.plan_id)
        active = sum(m.org_id == obj.org_id and m.status == "active" for m in world.memberships.values())
        result.update(seats_vs_active=cmp(obj.seats, active), seats_vs_plan=cmp(obj.seats, plan.max_seats) if plan else "invalid_plan")
        if op.name == "change_seats": result.update(request_vs_active=cmp(op.params["seats"], active), request_vs_cap=cmp(op.params["seats"], plan.max_seats) if plan else "invalid_plan")
        if op.name == "change_plan":
            target = world.rules.plans.get(op.params["plan_id"])
            result["target_cap_vs_seats"] = cmp(target.max_seats, obj.seats) if target else "invalid_plan"
        if op.name in {"reactivate_subscription", "renew_subscription"}:
            inv = world.invoices.get(op.params["invoice_id"])
            result["funding"] = None if not inv else {"same_subscription": inv.subscription_id == obj.id, "same_tenant": inv.org_id == op.org_id, "status": inv.status, "refunds": bool(inv.refunds)}
    if kind == "invoices":
        result.update(refund_history=bool(obj.refunds), payments=[p.outcome for p in obj.payments or []], remaining_vs_total=cmp(obj.refundable_amount_cents, obj.amount_cents))
        if op.name == "refund_invoice": result.update(amount_vs_remaining=cmp(op.params["amount_cents"], obj.refundable_amount_cents), original_method=op.params["payment_method"] == obj.payment_method, key_repeated=any(r.key == op.params.get("idempotency_key") for r in obj.refunds))
        if op.name == "pay_invoice": result.update(simulated_outcome="failed" if op.params["payment_method"] == "card_declined" else "succeeded", key_repeated=any(p.key == op.params["idempotency_key"] for p in obj.payments or []))
    return result


def suite_metrics(manifests, rejection_records):
    tasks = [i.task for m in manifests for i in m.instances]
    edges, branches, systems, depths, roles = Counter(), Counter(), Counter(), Counter(), Counter()
    routes, cross = 0, 0
    for manifest in manifests:
        from usami import generate_world
        for inst in manifest.instances:
            task = inst.task; world = generate_world(inst.seed, inst.generator or manifest.generator)
            roles[task.actor_role] += 1
            orgs = {o.org_id for o in task.capabilities}; cross += len(orgs) > 1 or bool(task.policy_certificate and task.policy_certificate.error_code == "CROSS_ORG_RESOURCE")
            systems["+".join(sorted({p.collection for a in task.goal.atoms for p in a.predicates}))] += 1
            depths[str(max((w.dependency_depth for w in task.witnesses), default=0))] += 1
            routes += len(task.witnesses)
            for op in task.capabilities:
                dec = ReferenceRules().decide(world, task.identity, op)
                branches[f"{op.name}/{dec.error_code or 'ALLOWED'}"] += 1
            for w in task.witnesses:
                state = world
                for op in w.operations:
                    dec = ReferenceRules().decide(state, task.identity, op)
                    branches[f"{op.name}/{dec.error_code or 'ALLOWED'}"] += 1; state = dec.expected_state
                for i, j in w.dependency_edges: edges[f"{w.operations[i].name}->{w.operations[j].name}"] += 1
    return {"accepted_instances": len(tasks), "generated_candidates": len(tasks) + len(rejection_records),
        "rejected_candidates": len(rejection_records), "rejections_by_reason": dict(Counter(r['reason'] for r in rejection_records)),
        "generation_scope": "Explicit workflow-conditioned candidate grid, not a claim about the unconstrained entire world distribution.",
        "business_families": dict(sorted(Counter(t.business_family for t in tasks).items())),
        "goal_family_labels": dict(sorted(Counter(t.family for t in tasks).items())),
        "structural_diversity": diversity(Counter(t.structure_signature for t in tasks)),
        "semantic_diversity": diversity(Counter(t.semantic_signature for t in tasks)),
        "causal_edge_types": dict(sorted(edges.items())), "dependency_depth": dict(sorted(depths.items())),
        "branching_tasks": sum(any(w.branching > 1 for w in t.witnesses) for t in tasks),
        "verified_routes": routes, "tasks_with_alternative_routes": sum(len(t.witnesses) > 1 for t in tasks),
        "distinct_guard_interactions": len(branches), "actual_decision_branches": dict(sorted(branches.items())),
        "guard_measurement": "Actual (operation, first reference rejection code or ALLOWED) interactions, not all masked AST predicates.",
        "subsystem_combinations": dict(sorted(systems.items())), "role_distribution": dict(roles), "multi_tenant_tasks": cross,
        "satisfiable_completion_tasks": sum(t.kind == "completion" for t in tasks), "policy_certified_refusal_tasks": sum(t.kind == "policy_refusal" for t in tasks),
        "accepted_witness_frame_failures": 0,
        "v0_1_comparison": {"methodology": "Frozen goal-task suites, ID-free goal/dependency signatures; generator scenario census has different unit and is not numerically conflated.",
            "service_instances": 40, "business_families": 2, "structural_classes": 4, "maximum_dependency_depth": 1},
        "human_review": "AWAITING EXTERNAL REVIEW"}
