from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from usami import generate_world, generate_scenarios, evaluate_trace, replay_trace
from usami.events import Trace
from usami.evaluation.scenario import Scenario, simulate_scenario
from usami.evaluation.splits import (create_manifest, load_manifest, validate_manifest,
                                     validate_splits)
from usami.world.generator import GeneratorConfig
from usami.world.models import WorldState, digest
from usami.world.rules import check_invariants


def save_model(value, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(value.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_world(path):
    world = WorldState.model_validate_json(Path(path).read_text(encoding="utf-8"))
    if check_invariants(world):
        raise ValueError("world violates business invariants")
    return world


def parser():
    p = argparse.ArgumentParser(description="Usami: deterministic compositional SaaS business worlds")
    commands = p.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="generate lawful initial worlds")
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--count", type=int, default=10)
    generate.add_argument("--config", type=Path)
    generate.add_argument("--out", type=Path, required=True)
    inspect = commands.add_parser("inspect", help="show entities, relationships and rules")
    inspect.add_argument("world", type=Path)
    scenarios = commands.add_parser("scenarios", help="derive structured scenarios using reference rules")
    scenarios.add_argument("--world", type=Path, required=True)
    scenarios.add_argument("--out", type=Path, required=True)
    simulate = commands.add_parser("simulate", help="execute one structured scenario")
    simulate.add_argument("--world", type=Path, required=True)
    simulate.add_argument("--scenario", type=Path, required=True)
    simulate.add_argument("--mode", choices=["reference", "sandbox", "refuse"], default="reference")
    simulate.add_argument("--out", type=Path, default=Path("outputs/trace.json"))
    evaluate = commands.add_parser("evaluate", help="independently audit a saved trace")
    evaluate.add_argument("trace", type=Path)
    evaluate.add_argument("--manifest", type=Path, help="external trusted source of request contexts")
    evaluate.add_argument("--scenario-id", help="scenario in the trusted manifest")
    evaluate.add_argument("--out", type=Path)
    replay = commands.add_parser("replay", help="deterministically re-execute a saved event log")
    replay.add_argument("trace", type=Path)
    replay.add_argument("--out", type=Path)
    manifest = commands.add_parser("manifest", help="freeze a versioned split manifest")
    manifest.add_argument("--split", choices=["train", "validation", "test"], required=True)
    manifest.add_argument("--seed", type=int)
    manifest.add_argument("--count", type=int, default=2)
    manifest.add_argument("--scenarios-per-world", type=int, default=12)
    manifest.add_argument("--budget", type=int, default=20)
    manifest.add_argument("--config", type=Path)
    manifest.add_argument("--demo", action="store_true", help="label a public test manifest as a non-blind demo")
    manifest.add_argument("--out", type=Path, required=True)
    validate = commands.add_parser("validate", help="batch-validate generators, rules and split manifests")
    validate.add_argument("--seeds", type=int, default=100)
    validate.add_argument("--manifest", type=Path, action="append", default=[])
    benchmark = commands.add_parser("run-manifest", help="run all frozen scenarios independently")
    benchmark.add_argument("manifest", type=Path)
    benchmark.add_argument("--mode", choices=["reference", "sandbox", "refuse"], default="reference")
    benchmark.add_argument("--out", type=Path, required=True)
    return p


def run(args):
    if args.command == "generate":
        if args.count < 1:
            raise ValueError("count must be positive")
        cfg = GeneratorConfig.model_validate_json(args.config.read_text()) if args.config else GeneratorConfig()
        for n in range(args.count):
            world = generate_world(args.seed + n, cfg)
            save_model(world, args.out / f"world_{n + 1:04d}.json")
        print(json.dumps({"worlds": args.count, "out": str(args.out), "config_hash": digest(cfg)}))
    elif args.command == "inspect":
        world = read_world(args.world)
        print(f"{world.world_id} | seed={world.seed} | clock={world.clock.isoformat()} | hash={world.state_hash}")
        print(f"Rules {world.rules.version}: full refund within {world.rules.refund_window_days} days, original channel")
        for plan in world.rules.plans.values():
            print(f"  {plan.name}: at most {plan.max_seats} seats")
        for org in world.organizations.values():
            sub = world.subscriptions[org.subscription_id]
            print(f"\n{org.id} {org.name} | owner={org.owner_id} | subscription={sub.id}")
            print(f"  {sub.plan_id} seats={sub.seats} status={sub.status} auto_renew={sub.auto_renew} until={sub.period_end.isoformat()}")
            for m in world.memberships.values():
                if m.org_id == org.id:
                    print(f"  {m.id}: user={m.user_id} role={m.role} status={m.status}")
            for i in world.invoices.values():
                if i.org_id == org.id:
                    print(f"  {i.id}: amount={i.amount_cents} cents status={i.status} paid_at={i.paid_at} method={i.payment_method} refunds={len(i.refunds)}")
    elif args.command == "scenarios":
        scenarios = generate_scenarios(read_world(args.world))
        args.out.mkdir(parents=True, exist_ok=True)
        for n, scenario in enumerate(scenarios):
            save_model(scenario, args.out / f"scenario_{n + 1:04d}.json")
        print(json.dumps({"scenarios": len(scenarios), "categories": sorted({s.category for s in scenarios})}))
    elif args.command == "simulate":
        trace = simulate_scenario(read_world(args.world), Scenario.model_validate_json(args.scenario.read_text()), args.mode)
        trace.save(args.out)
        report = evaluate_trace(trace)
        print(report.model_dump_json(indent=2))
    elif args.command == "evaluate":
        trace = Trace.load(args.trace)
        trusted = None
        if args.manifest:
            manifest = load_manifest(args.manifest)
            scenarios = [s for e in manifest.entries for s in e.scenarios if s.scenario_id == args.scenario_id]
            if len(scenarios) != 1:
                raise ValueError("--scenario-id must identify exactly one trusted manifest scenario")
            scenario = scenarios[0]
            if scenario.initial_state_hash != trace.initial_state_hash or scenario.operation_budget != trace.operation_budget:
                raise ValueError("trace differs from manifest initial state/budget")
            trusted = {s.request.request_id: s.request for s in scenario.steps}
        report = evaluate_trace(trace, trusted)
        if args.out:
            save_model(report, args.out)
        print(report.model_dump_json(indent=2))
    elif args.command == "replay":
        original = Trace.load(args.trace)
        replayed = replay_trace(original)
        if args.out:
            replayed.export_snapshot(args.out)
        print(json.dumps({"replayed": True, "events": len(original.events), "final_state_hash": replayed.state.state_hash}))
    elif args.command == "manifest":
        if args.count < 1:
            raise ValueError("count must be positive")
        seed = args.seed if args.seed is not None else {"train": 0, "validation": 1000000, "test": 2000000}[args.split]
        cfg = GeneratorConfig.model_validate_json(args.config.read_text()) if args.config else None
        manifest = create_manifest(args.split, range(seed, seed + args.count), cfg,
                                   args.scenarios_per_world, args.budget, "demo" if args.demo else None)
        manifest.save(args.out)
        print(json.dumps({"manifest_hash": manifest.manifest_hash, "split": manifest.split,
                          "visibility": manifest.visibility, "worlds": len(manifest.entries)}))
    elif args.command == "validate":
        if args.seeds < 1:
            raise ValueError("seed count must be positive")
        for profile in ("standard", "train_single", "test_composed"):
            cfg = GeneratorConfig(profile=profile)
            for seed in range(args.seeds):
                world = generate_world(seed, cfg)
                if world != generate_world(seed, cfg) or check_invariants(world):
                    raise ValueError(f"generation failure: {profile}, {seed}")
        manifests = [load_manifest(path) for path in args.manifest] if args.manifest else [
            create_manifest("train", [0, 1]), create_manifest("validation", [1000000, 1000001]),
            create_manifest("test", [2000000, 2000001], visibility="demo")]
        validate_splits(*manifests)
        scenarios_checked = 0
        for manifest in manifests:
            for entry in manifest.entries:
                world = generate_world(entry.seed, manifest.config)
                for scenario in entry.scenarios:
                    trace = simulate_scenario(world, scenario)
                    if evaluate_trace(trace).status != "pass":
                        raise ValueError("reference scenario failed independent audit")
                    replay_trace(trace)
                    scenarios_checked += 1
        print(json.dumps({"valid": True, "generated_worlds": args.seeds * 3, "manifests": len(manifests),
                          "reference_scenarios_and_replays": scenarios_checked}))
    elif args.command == "run-manifest":
        manifest = load_manifest(args.manifest)
        reports = []
        for entry in manifest.entries:
            world = generate_world(entry.seed, manifest.config)
            for scenario in entry.scenarios:
                trace = simulate_scenario(world, scenario, args.mode)
                trace.save(args.out / f"{scenario.scenario_id}.trace.json")
                contexts = {s.request.request_id: s.request for s in scenario.steps}
                report = evaluate_trace(trace, contexts)
                save_model(report, args.out / f"{scenario.scenario_id}.report.json")
                reports.append({"scenario_id": scenario.scenario_id, "status": report.status,
                                "metrics": report.metrics})
        summary = {"manifest_hash": manifest.manifest_hash, "mode": args.mode,
                   "operation_budget": manifest.operation_budget, "scenarios": len(reports),
                   "passed": sum(r["status"] == "pass" for r in reports), "results": reports}
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({k: v for k, v in summary.items() if k != "results"}))
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError, ValidationError) as exc:
        print(f"usami: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
