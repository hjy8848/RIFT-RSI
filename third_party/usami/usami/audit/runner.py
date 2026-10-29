"""Reproducible census over actual generator candidates, plus labeled demo splits."""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path
import platform
from typing import Callable, Optional

import pydantic

from usami.evaluation.oracle import evaluate_trace
from usami.evaluation.scenario import SCENARIO_VERSION, generate_scenarios, simulate_scenario
from usami.evaluation.splits import business_fingerprint, create_manifest, load_manifest, validate_splits
from usami.world.engine import replay_trace
from usami.world.generator import GeneratorConfig, composition_signature, generate_world
from usami.world.models import digest
from usami.audit.rules import RuleObserver, rule_statistics
from usami.audit.signatures import SIGNATURE_VERSION, analyze_scenario, cmp, invoice_class, world_signature

AUDIT_VERSION = "usami-diversity-audit-1"
ROOT = Path(__file__).resolve().parents[2]
CORE_FILES = ["usami/world/models.py", "usami/world/generator.py", "usami/world/rules.py",
              "usami/world/engine.py", "usami/protocol.py", "usami/evaluation/scenario.py",
              "usami/evaluation/splits.py", "usami/evaluation/oracle.py", "usami/events.py", "usami/randomness.py"]


def source_hashes():
    files = CORE_FILES + [str(p.relative_to(ROOT)) for p in sorted((ROOT / "usami/audit").glob("*.py"))]
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files}


def diversity(counter):
    total = sum(counter.values())
    entropy = -sum((n / total) * math.log2(n / total) for n in counter.values()) if total else 0.0
    ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return {"total": total, "unique": len(counter), "repetitions": total - len(counter),
            "repetition_ratio": (total - len(counter)) / total if total else 0.0,
            "singletons": sum(n == 1 for n in counter.values()), "entropy_bits": entropy,
            "effective_diversity": 2 ** entropy if total else 0.0,
            "top10_share": sum(n for _, n in ordered[:10]) / total if total else 0.0,
            "top10": [{"signature": key, "count": n} for key, n in ordered[:10]],
            "frequencies": dict(sorted(counter.items()))}


def overlap(left, right):
    a, b = set(left), set(right)
    common = sorted(a & b)
    return {"left_unique": len(a), "right_unique": len(b), "common": len(common),
            "left_only": sorted(a - b), "right_only": sorted(b - a), "common_signatures": common,
            "jaccard": len(common) / len(a | b) if a | b else 0.0,
            "right_structure_in_left_ratio": len(common) / len(b) if b else 0.0,
            "right_sample_in_left_ratio": sum(n for k, n in right.items() if k in a) / sum(right.values()) if right else 0.0}


class Census:
    def __init__(self, observer):
        self.observer = observer
        self.templates, self.semantics, self.categories = Counter(), Counter(), Counter()
        self.examples, self.records, self.complexity, self.dependencies = {}, [], {}, Counter()

    def add(self, world, scenario):
        analysis = analyze_scenario(world, scenario, self.observer)
        self.templates[analysis["template_id"]] += 1
        self.semantics[analysis["semantic_id"]] += 1
        self.categories[scenario.category] += 1
        for kind in ("template", "semantic"):
            key = analysis[kind + "_id"]
            if key not in self.examples:
                self.examples[key] = {"seed": world.seed, "scenario_id": scenario.scenario_id,
                                      "category": scenario.category, "signature_payload": analysis[kind]}
        self.records.extend(analysis["records"])
        for metric, value in analysis["complexity"].items():
            self.complexity.setdefault(metric, Counter())[str(value)] += 1
        for i, j in analysis["dependency_edges"]:
            pair = f"{scenario.steps[i].request.operation.name}->{scenario.steps[j].request.operation.name}"
            self.dependencies[pair] += 1
        return analysis

    def result(self, with_examples=True):
        result = {"template": diversity(self.templates), "semantic": diversity(self.semantics),
                  "categories": dict(sorted(self.categories.items())),
                  "complexity": {k: dict(sorted(v.items())) for k, v in sorted(self.complexity.items())},
                  "dependency_operation_pairs": dict(sorted(self.dependencies.items())),
                  "rules": rule_statistics(self.observer, self.records)}
        if with_examples:
            result["signature_examples"] = dict(sorted(self.examples.items()))
        return result


def world_metrics(world, hist, unique):
    def count(key, value):
        hist.setdefault(key, Counter())[str(value)] += 1
    for name in ("users", "organizations", "memberships", "subscriptions", "invoices"):
        count(name + "_count", len(getattr(world, name)))
    degrees, active_degrees, different, connected, multi = [], [], 0, 0, 0
    for user in world.users:
        memberships = [m for m in world.memberships.values() if m.user_id == user]
        count("user_organization_degree", len(memberships))
        count("user_active_organization_degree", sum(m.status == "active" for m in memberships))
        connected += bool(memberships)
        multi += len(memberships) > 1
        different += len({m.role for m in memberships}) > 1
    for m in world.memberships.values():
        count("membership_role_status", m.role + "/" + m.status)
    for org in world.organizations:
        memberships = [m for m in world.memberships.values() if m.org_id == org]
        count("organization_members", len(memberships))
        count("organization_active_members", sum(m.status == "active" for m in memberships))
    for sub in world.subscriptions.values():
        count("subscription_plan", sub.plan_id)
        count("subscription_status", sub.status)
        count("subscription_seats", sub.seats)
        count("subscription_auto_renew", sub.auto_renew)
        cap = world.rules.plans[sub.plan_id].max_seats
        active = max(1, sum(m.org_id == sub.org_id and m.status == "active" for m in world.memberships.values()))
        count("subscription_seats_vs_capacity", cmp(sub.seats, cap))
        count("subscription_seats_vs_members", cmp(sub.seats, active))
        count("subscription_capacity_slack", cap - sub.seats)
        count("subscription_member_slack", sub.seats - active)
        count("subscription_joint_state", f"{sub.status}/{sub.auto_renew}/{cmp(sub.seats, active)}/{cmp(sub.seats, cap)}")
    for inv in world.invoices.values():
        count("invoice_status", inv.status)
        count("invoice_refund_class", invoice_class(world, inv))
        count("invoice_payment_method", inv.payment_method)
        count("invoice_amount_cents", inv.amount_cents)
        count("invoice_full_refund_only", inv.amount_cents == inv.refundable_amount_cents)
        count("invoice_history_length", len(inv.refunds))
    count("heldout_world_combination", composition_signature(world))
    unique["business_instance"].add(business_fingerprint(world))
    for key, include_state in (("role_topology", False), ("policy_state_topology", True)):
        signature = world_signature(world, include_state)
        if signature is not None:
            unique[key].add(signature)
        else:
            count(key + "_unknown", True)
    return different, connected, multi


def audit_splits(observer, count, config, validation_path):
    if count < 1:
        raise ValueError("split world count must be positive")
    manifests = {}
    for name, start, profile in (("train", 0, "train_single"), ("validation", 1000000, "standard"), ("test", 2000000, "test_composed")):
        cfg = config.model_copy(deep=True)
        cfg.profile = profile
        manifests[name] = create_manifest(name, range(start, start + count), cfg, visibility="demo")
    validate_splits(*manifests.values())
    summaries, censuses = {}, {}
    for name, manifest in manifests.items():
        census = Census(observer)
        for entry in manifest.entries:
            world = generate_world(entry.seed, manifest.config)
            for scenario in entry.scenarios:
                census.add(world, scenario)
        summaries[name] = {"manifest_hash": manifest.manifest_hash, "visibility": "demo", "worlds": len(manifest.entries),
                           "seeds": [e.seed for e in manifest.entries], "config": manifest.config.model_dump(mode="json"),
                           "heldout_worlds": sum(composition_signature(generate_world(e.seed, manifest.config)) for e in manifest.entries),
                           **census.result(with_examples=False)}
        censuses[name] = census
    pairs = {}
    for left, right in (("train", "test"), ("train", "validation"), ("validation", "test")):
        pairs[left + "_vs_" + right] = {kind: overlap(getattr(censuses[left], plural), getattr(censuses[right], plural))
                                        for kind, plural in (("template", "templates"), ("semantic", "semantics"))}
        left_rules = set(summaries[left]["rules"]["visited_business_guard_sets"])
        right_rules = set(summaries[right]["rules"]["visited_business_guard_sets"])
        pairs[left + "_vs_" + right]["visited_rule_sets_right_only"] = sorted(right_rules - left_rules)
        pairs[left + "_vs_" + right]["dependency_pairs_right_only"] = sorted(set(censuses[right].dependencies) - set(censuses[left].dependencies))
    shipped = None
    if validation_path is not None:
        manifest = load_manifest(validation_path)
        census = Census(observer)
        for entry in manifest.entries:
            world = generate_world(entry.seed, manifest.config)
            for scenario in entry.scenarios:
                census.add(world, scenario)
        shipped = {"path": str(Path(validation_path).name), "manifest_hash": manifest.manifest_hash,
                   "worlds": len(manifest.entries), "seeds": [e.seed for e in manifest.entries],
                   **census.result(with_examples=False)}
    return {"kind": "audit-derived public demo; not an official or blind task test set", "demo_manifests": summaries,
            "pairwise": pairs, "shipped_validation": shipped,
            "shipped_train_task_manifest": None, "shipped_private_test_task_manifest": None}


def run_audit(seeds, config=None, split_worlds=100, validation_path=None, engine_checks=32,
              progress: Optional[Callable[[str], None]] = None):
    seeds = list(seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("audit seeds must be nonempty and unique")
    if engine_checks < 0:
        raise ValueError("engine checks cannot be negative")
    cfg = (config or GeneratorConfig()).model_copy(deep=True)
    hashes_before = source_hashes()
    observer, hist = RuleObserver(), {}
    census, unique = Census(observer), {k: set() for k in ("business_instance", "role_topology", "policy_state_topology")}
    growth, world_evidence, verification = [], [], []
    roles_different = connected = multi = 0
    checked_templates = set()
    for index, seed in enumerate(seeds, 1):
        world = generate_world(seed, cfg)
        different, conn, mult = world_metrics(world, hist, unique)
        roles_different += different
        connected += conn
        multi += mult
        candidates = generate_scenarios(world)
        world_evidence.append({"seed": seed, "state_hash": world.state_hash, "scenarios": len(candidates)})
        for scenario in candidates:
            analysis = census.add(world, scenario)
            if len(verification) < engine_checks and analysis["template_id"] not in checked_templates:
                trace = simulate_scenario(world, scenario)
                trusted = {s.request.request_id: s.request for s in scenario.steps}
                report = evaluate_trace(trace, trusted_requests=trusted)
                replayed = replay_trace(trace)
                if report.status != "pass" or replayed.state != trace.final_state:
                    raise ValueError("engine/oracle/replay disagrees with sampled reference")
                verification.append({"seed": seed, "scenario_id": scenario.scenario_id,
                                     "template": analysis["template_id"], "oracle": report.status,
                                     "trace_hash": trace.events[-1].event_hash, "replay_final_hash": replayed.state.state_hash})
                checked_templates.add(analysis["template_id"])
        if index in {1, 10, 25, 50, 100, 200, 500, 1000} or index == len(seeds):
            growth.append({"worlds": index, "scenarios": sum(census.templates.values()),
                           "templates": len(census.templates), "semantics": len(census.semantics)})
        if progress and (index % 25 == 0 or index == len(seeds)):
            progress(f"worlds {index}/{len(seeds)}; scenarios {sum(census.templates.values())}; templates {len(census.templates)}; semantic {len(census.semantics)}")
    if progress:
        progress(f"auditing demo splits: {split_worlds} worlds each, plus shipped validation")
    splits = audit_splits(observer, split_worlds, cfg, validation_path)
    if hashes_before != source_hashes():
        raise ValueError("source changed during audit; rerun from stable source")
    return {"version": AUDIT_VERSION, "signature_version": SIGNATURE_VERSION,
            "provenance": {"generator_version": cfg.version, "rules_version": cfg.rules.version,
                           "scenario_version": SCENARIO_VERSION, "config": cfg.model_dump(mode="json"),
                           "config_hash": digest(cfg), "rules_hash": digest(cfg.rules), "seeds": seeds,
                           "python": platform.python_version(), "pydantic": pydantic.__version__,
                           "source_hashes": hashes_before, "world_evidence": world_evidence},
            "world": {"count": len(seeds), "histograms": {k: dict(sorted(v.items())) for k, v in sorted(hist.items())},
                      "unique": {k: len(v) for k, v in unique.items()},
                      "cross_role_users": {"different_roles": roles_different, "all_users": len(seeds) * cfg.user_count,
                                           "connected_users": connected, "multi_organization_users": multi,
                                           "ratio_all": roles_different / (len(seeds) * cfg.user_count),
                                           "ratio_connected": roles_different / connected if connected else 0,
                                           "ratio_multi_organization": roles_different / multi if multi else 0}},
            "scenario": census.result(), "growth": growth, "splits": splits,
            "verification": {"all_scenario_labels_and_state_hashes_matched_reference": True,
                             "reference_steps": len(census.records), "engine_oracle_replay_checks": verification,
                             "all_dependency_cases_known_under_prefix_omission": True}}
