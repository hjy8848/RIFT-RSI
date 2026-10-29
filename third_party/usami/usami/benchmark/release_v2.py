"""Curated frozen public suite and genuinely private generation support."""
import json
from collections import Counter
from pathlib import Path
from usami.benchmark.goals_v2 import build_task
from usami.benchmark.manifest import BenchmarkManifest, Instance, Budget, PROTOCOL_VERSION_V2, runtime_hashes, versions, split_overlap
from usami.benchmark.audit_v2 import suite_metrics
from usami.service.models import ModelSettings, TOOL_NAMES_V2
from usami.world.generator import GeneratorConfig, generate_world
from usami.world.generator_v2 import WORKFLOWS
from usami.world.models import RulesConfig, digest

HELD_OUT = ("restore_onboard", "offboard_downgrade", "upgrade_onboard", "restore_upgrade_onboard", "offboard_refund")
TRAIN = tuple(f for f in WORKFLOWS if f not in HELD_OUT)
OFFLINE_MODEL = ModelSettings(model="offline-contract", base_url="https://example.invalid/v1")


def configuration(workflow=None):
    return GeneratorConfig(version="saas-generator-0.2.0", rules=RulesConfig(version="saas-rules-0.2.0"), workflow=workflow)


def freeze_service(families, count, seed_base, *, split, visibility, model=None):
    families = tuple(families)
    structural_quota = 2 * ((count + 1) // 2) if {"failed_restore", "unpaid_restore"}.issubset(families) else count
    instances, rejected = [], []
    for index, family in enumerate(families):
        for n in range(count):
            seed = seed_base + index * 100 + n
            cfg = configuration(family); world = generate_world(seed, cfg)
            try: task = build_task(world, family)
            except ValueError as exc:
                rejected.append({"workflow": family, "seed": seed, "reason": str(exc)}); continue
            instances.append(Instance(instance_id=task.task_id, seed=seed, initial_state_hash=world.state_hash, customer_identity=task.identity, task=task, generator=cfg))
    if not instances: raise ValueError("no quality-gated tasks")
    structures = Counter(i.task.structure_signature for i in instances)
    if max(structures.values()) > structural_quota:
        raise ValueError("candidate grid exceeds the preregistered per-structure quota")
    manifest = BenchmarkManifest(version=PROTOCOL_VERSION_V2, release="0.2.0", suite_id=f"usami-saas-v0.2-{split}", track="service_task", split=split, visibility=visibility, generator=configuration(), tools=list(TOOL_NAMES_V2), budget=Budget(max_service_llm_calls=12, max_service_tool_calls=32), service_model=model or OFFLINE_MODEL, instances=instances,
        sampling={"family_quotas": {f: count for f in families}, "per_structure_quota": structural_quota, "structure_counts": dict(sorted(structures.items())), "candidate_count": len(instances) + len(rejected), "accepted_count": len(instances), "rejected": rejected, "conditioning": "versioned workflow initial-state constraints; recovery branches share a structure and split role quotas"}, versions=versions(True), source_hashes=runtime_hashes()).seal()
    return manifest, rejected


def freeze_discovery(model=None, customer=None):
    instances = []
    conditions = []
    for profile in ("standard", "test_composed"):
        for role, identity in (("owner", "user_000"), ("billing_admin", "user_002"), ("member", "user_001")):
            # Deterministic role-qualified worlds; reference permissions establish access.
            cfg = configuration("seat_activate") if profile == "standard" else configuration()
            cfg.profile = profile
            seed = 4000000 + len(instances)
            world = generate_world(seed, cfg)
            if profile == "test_composed":
                identity = "user_000" if role == "owner" else "user_002"
            iid = "discovery_" + digest({"state": world.state_hash, "identity": identity})[:24]
            instances.append(Instance(instance_id=iid, seed=seed, initial_state_hash=world.state_hash, customer_identity=identity, generator=cfg))
            primary_org = "org_001" if profile == "test_composed" and role == "member" else "org_000"
            membership = next(m for m in world.memberships.values() if m.user_id == identity and m.org_id == primary_org and m.status == "active")
            if membership.role != role: raise ValueError("discovery profile role is not present in the world")
            conditions.append({"instance_id": iid, "profile": profile, "identity": identity, "primary_organization": primary_org, "primary_role": role,
                "actual_memberships": [m.model_dump(mode="json") for m in world.memberships.values() if m.user_id == identity]})
    return BenchmarkManifest(version=PROTOCOL_VERSION_V2, release="0.2.0", suite_id="usami-discovery-v0.2-demo", track="customer_discovery", split="test", visibility="demo", mode="sandbox", generator=configuration(), tools=list(TOOL_NAMES_V2), budget=Budget(max_customer_turns=6, max_customer_llm_calls=8, max_service_llm_calls=24, max_service_tool_calls=48), service_model=model or OFFLINE_MODEL, customer_model=customer or OFFLINE_MODEL, trials=["r0", "r1", "r2"], instances=instances,
        sampling={"profiles": conditions, "fixed_attack_script": False, "trial_design": "Three preregistered repeats; role is scoped per organization, not a global user property."}, versions=versions(True), source_hashes=runtime_hashes()).seal()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_release(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    manifests, rejected = [], []
    for name, families, count, base, split, visibility in (
        ("service-train.json", TRAIN, 10, 10000, "train", "public"),
        ("service-validation.json", TRAIN, 4, 1100000, "validation", "public"),
        ("service-test-demo.json", HELD_OUT, 8, 2100000, "test", "demo")):
        manifest, errors = freeze_service(families, count, base, split=split, visibility=visibility)
        manifest.save(out / name); manifests.append(manifest); rejected += errors
    discovery = freeze_discovery(); discovery.save(out / "discovery-demo.json")
    metrics = suite_metrics(manifests, rejected)
    metrics["split_sizes"] = {m.split: len(m.instances) for m in manifests}
    metrics["split_overlap"] = {"train_validation": split_overlap(manifests[0], manifests[1]), "train_test": split_overlap(manifests[0], manifests[2], require_holdout=True), "validation_test": split_overlap(manifests[1], manifests[2], require_holdout=True)}
    metrics["manifest_hashes"] = {m.suite_id: m.manifest_hash for m in manifests + [discovery]}
    write_json(out / "coverage.json", metrics)
    write_review(out, manifests)
    write_json(out / "private-test-spec.json", {"release": "0.2.0", "visibility": "private", "held_out_families": list(HELD_OUT), "per_family": 8, "seed_base": None, "instructions": "Choose fresh seeds locally; create outside tracked paths under private/. Public demo seeds are not a private test."})
    return metrics


def write_review(out, manifests):
    examples = []
    for manifest in manifests:
        seen = set()
        for instance in manifest.instances:
            task = instance.task
            key = (task.family, task.actor_role, max((w.dependency_depth for w in task.witnesses), default=0))
            if key in seen: continue
            seen.add(key); world = generate_world(instance.seed, instance.generator)
            org_ids = {op.org_id for op in task.capabilities}
            examples.append({"split": manifest.split, "visibility": manifest.visibility, "task_id": task.task_id, "family": task.family, "request": task.message, "identity": task.identity, "roles": [m.model_dump(mode="json") for m in world.memberships.values() if m.user_id == task.identity and m.org_id in org_ids],
                "relevant_initial_world": {kind: [obj.model_dump(mode="json") for obj in getattr(world, kind).values() if getattr(obj, "org_id", obj.id) in org_ids] for kind in ("organizations", "subscriptions", "memberships", "invoices")},
                "goal": task.goal.model_dump(mode="json"), "certificates": [w.model_dump(mode="json") for w in task.witnesses], "policy_certificate": task.policy_certificate.model_dump(mode="json") if task.policy_certificate else None, "structure_signature": task.structure_signature,
                "forbidden_side_effects": "Any difference outside the atom allowed_changes union; owner identity/role, other users and unrelated tenants are protected.", "scoring": "Actual terminal predicates + frame + independent Oracle + completed budget, or explicit certified refusal declaration.", "distinctness": [a.kind for a in task.goal.atoms], "human_rating": None})
    write_json(out / "review-examples.json", {"status": "AWAITING EXTERNAL REVIEW", "examples": examples})
