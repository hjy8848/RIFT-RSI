"""Concrete research CLI; all Usami imports remain outside VAST-CoEvo Core."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Literal, Optional, Tuple

from pydantic import Field, model_validator
from vast_coevo.models import Budget, EpisodeEvidence, ModelSpec
from vast_coevo.schema import FrozenModel, atomic_json, digest
from vast_coevo.skills.library import LibrarySnapshot, LibraryStore
from vast_coevo.evidence.archive import EvidenceArchive
from vast_coevo.evolution.models import GatePolicy, RunConfig, ValidationCase
from vast_coevo.evolution.controller import CoEvolutionController
from vast_coevo.storage.runs import RunStore
from vast_coevo.reflection.reflector import CustomerReflector, ServiceReflector, ReflectionInput
from vast_coevo.reflection.scope import DomainPolicy
from .adapter import UsamiAdapter
from .domains import UsamiProfile, ServiceTaskProfile, model_settings, native_task


class TaskCase(FrozenModel):
    task_ref: str
    purpose: Literal["legal_completion", "correct_refusal"]


class ExperimentPlan(FrozenModel):
    schema_version: Literal[1] = 1
    profiles: Tuple[UsamiProfile, ...] = Field(min_length=1)
    service_tasks: Tuple[ServiceTaskProfile, ...] = ()
    exploration: Tuple[str, ...] = Field(min_length=1)
    validation: Tuple[str, ...] = Field(min_length=1)
    regressions: Tuple[TaskCase, ...] = ()
    customer_library: str
    service_library: str
    customer_model: Optional[ModelSpec] = None
    service_model: Optional[ModelSpec] = None
    customer_reflector_model: Optional[ModelSpec] = None
    service_reflector_model: Optional[ModelSpec] = None
    budget: Budget = Field(default_factory=Budget)
    task_budget: Budget = Field(default_factory=lambda: Budget(max_customer_turns=1, max_customer_llm_calls=0,
                                                              max_service_llm_calls=14, max_service_tool_calls=24))
    generations: int = Field(default=1, ge=1, strict=True)
    max_llm_calls: int = Field(default=500, ge=2, strict=True)
    max_tool_calls: int = Field(default=1000, ge=0, strict=True)
    policy: GatePolicy = Field(default_factory=GatePolicy)
    customer_gate_version: Optional[Literal['legacy-v1', 'quality-gated-discovery-v1', 'quality-gated-discovery-v2']] = Field(
        default=None, exclude_if=lambda value: value is None)
    training_protocol: Optional[Literal['recursive-training-v1']] = Field(default=None, exclude_if=lambda value: value is None)
    customer_selection: Optional[Literal['customer-static-only-v1']] = Field(
        default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def registry(self):
        self.check_gate_selection(require_explicit=False)
        ids = {p.config_id for p in self.profiles}
        tasks = {t.task_ref for t in self.service_tasks}
        if set(self.exploration + self.validation) - ids or {r.task_ref for r in self.regressions} - tasks:
            raise ValueError("experiment plan references an unregistered case")
        if any(p.split == "heldout" for p in self.profiles if p.config_id in self.exploration + self.validation):
            raise ValueError("heldout profiles cannot be selected for development")
        return self

    def check_gate_selection(self, require_explicit=True):
        if self.customer_gate_version is None:
            if require_explicit: raise ValueError('new experiments must explicitly choose customer_gate_version')
            return
        quality = self.policy.customer_quality
        if self.customer_gate_version == 'legacy-v1':
            if quality is not None: raise ValueError('legacy gate conflicts with customer_quality configuration')
        elif quality is None or quality.protocol != self.customer_gate_version:
            raise ValueError('declared Customer gate version requires matching customer_quality configuration')
        if self.customer_gate_version == 'quality-gated-discovery-v2' and any(
                profile.read_probe_protocol != 'cloudops-read-probes-v1' for profile in self.profiles
                if profile.config_id in self.validation + self.exploration):
            raise ValueError('QGD v2 requires structured read probe protocol on exploration and validation profiles')


def model_from_env(role, fallback=None):
    """Pin actual configuration once; secret lookup remains in native transport."""
    from usami.service.client import settings_from_env
    env = dict(os.environ)
    prefix = "VAST_" + role.upper() + "_"
    configured = False
    for suffix in ("MODEL", "BASE_URL", "OPTIONS", "TIMEOUT", "TRUST_ENV"):
        if prefix + suffix in env:
            env["USAMI_LLM_" + suffix] = env[prefix + suffix]
            configured = True
    if fallback is not None and not configured:
        return fallback
    settings = settings_from_env(env)
    return ModelSpec(model=settings.model, base_url=settings.base_url, options=settings.options,
                     timeout_seconds=settings.timeout_seconds, trust_env=settings.trust_env)


def context(plan_path, root, publish_libraries=True):
    plan_path, root = Path(plan_path).resolve(), Path(root).resolve()
    plan = ExperimentPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    customer = LibrarySnapshot.model_validate_json((plan_path.parent / plan.customer_library).read_text())
    service = LibrarySnapshot.model_validate_json((plan_path.parent / plan.service_library).read_text())
    store = LibraryStore(root / "libraries")
    if publish_libraries:
        store.save(customer)
        store.save(service)
    adapter = UsamiAdapter(root / "native", plan.profiles, plan.service_tasks)
    archive = EvidenceArchive(root / "evidence")
    runs = RunStore(root / "runs")
    specs = {role: model_from_env(role, getattr(plan, role + "_model"))
             for role in ("customer", "service", "customer_reflector", "service_reflector")}
    return plan, adapter, store, archive, runs, customer, service, specs


def run_config(plan, adapter, customer, service, specs, run_id, generations=None):
    def case(config_id, task_ref=None, purpose=None):
        config = adapter.config_for(config_id, specs["customer"], specs["service"],
                                    plan.task_budget if task_ref else plan.budget, task_ref=task_ref)
        return ValidationCase(id=task_ref or config_id, config=config, task_ref=task_ref, purpose=purpose)
    profiles = {p.config_id: p for p in plan.profiles}
    tasks = {t.task_ref: t for t in plan.service_tasks}
    task_cases = []
    for regression in plan.regressions:
        task = tasks[regression.task_ref]
        _, actual, _ = native_task(profiles[task.config_id], task)
        expected = "correct_refusal" if actual.kind == "policy_refusal" else "legal_completion"
        if regression.purpose != expected:
            raise ValueError("declared regression purpose differs from native task kind")
        task_cases.append(case(task.config_id, task.task_ref, regression.purpose))
    policy = plan.policy
    if policy.customer_quality is not None and policy.customer_quality.protocol == 'quality-gated-discovery-v2':
        if adapter.service_scope_risks(service):
            raise ValueError('domain-specific Service guidance spans unverified domains; split or scope a new candidate first')
    if policy.customer_quality is not None:
        from .domains import initial_world
        excluded = set(policy.customer_excluded_world_hashes or ())
        excluded.update(initial_world(profile).state_hash for profile in plan.profiles
                        if profile.split == "heldout" or profile.config_id in plan.exploration)
        from .customer_probes import business_shape
        shapes = set(policy.customer_excluded_business_shapes or ())
        shapes.update(business_shape(initial_world(profile), profile.identity) for profile in plan.profiles
                      if profile.domain == "cloudops" and (profile.split == "heldout" or profile.config_id in plan.exploration))
        policy = policy.model_copy(update={"customer_excluded_world_hashes": tuple(sorted(excluded)),
                                           "customer_excluded_business_shapes": tuple(sorted(shapes))})
    policy = policy.model_copy(update={"service_task_regression_protocol": "paired-task-v1"})
    return RunConfig(schema_version=4, run_id=run_id, generations=generations or plan.generations,
                     exploration_cases=tuple(case(key) for key in plan.exploration),
                     validation_cases=tuple(case(key) for key in plan.validation), service_task_cases=tuple(task_cases),
                     customer_reflector_model=specs["customer_reflector"], service_reflector_model=specs["service_reflector"],
                     max_llm_calls=plan.max_llm_calls, max_tool_calls=plan.max_tool_calls, policy=policy,
                     initial_customer_hash=customer.snapshot_hash, initial_service_hash=service.snapshot_hash)


def transport(spec):
    from usami.service.client import OpenAICompatibleClient, key_from_env
    return OpenAICompatibleClient(model_settings(spec), key_from_env())


def _evidence(value, archive):
    path = Path(value)
    if path.is_file():
        row = EpisodeEvidence.model_validate_json(path.read_text())
        EvidenceArchive.verify_artifact(row.trace)
        return row
    return archive.load(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description="VAST-CoEvo on the unchanged Usami business worlds")
    parser.add_argument("--root", default="artifacts/vast-coevo", help="persistent run/evidence/library directory")
    parser.add_argument("--plan", help="versioned experiment plan (required for online execution and replay)")
    commands = parser.add_subparsers(dest="command", required=True)
    library = commands.add_parser("library", help="inspect immutable skill snapshots")
    library.add_argument("--hash")
    episode = commands.add_parser("episode", help="run one autonomous static Skill interaction")
    episode.add_argument("--profile", required=True)
    episode.add_argument("--id", required=True)
    reflect = commands.add_parser("reflect", help="one real Reflection, one patch; no automatic retry")
    reflect.add_argument("--role", choices=("customer", "service"), required=True)
    reflect.add_argument("--evidence", action="append", required=True)
    reflect.add_argument("--generation", type=int, default=0)
    reflect.add_argument("--output", required=True)
    run = commands.add_parser("run", help="run or resume one or multiple alternating generations")
    run.add_argument("--id", required=True)
    run.add_argument("--generations", type=int)
    cross = commands.add_parser("evaluate-cross", help="independent read-only Customer/Service Snapshot comparisons; no Reflection")
    cross.add_argument("--pair", action="append", required=True, help="LABEL:CUSTOMER_HASH:SERVICE_HASH; hashes are printed by history")
    cross.add_argument("--profile", action="append", required=True)
    cross.add_argument("--training-run", action="append", required=True, help="completed source run; excludes its training/Gate worlds")
    cross.add_argument("--purpose", choices=("development-diagnostic", "frozen-heldout"), required=True)
    cross.add_argument("--output", required=True, help="new independent log directory")
    cross.add_argument("--id", required=True)
    history = commands.add_parser("history", help="inspect saved generation history")
    history.add_argument("--id", required=True)
    restore = commands.add_parser("restore", help="branch a historical snapshot into a new run")
    restore.add_argument("--source", required=True)
    restore.add_argument("--id", required=True)
    restore.add_argument("--generation", type=int)
    restore.add_argument("--generations", type=int)
    replay = commands.add_parser("replay", help="offline exact skill/wire/native/oracle replay")
    replay.add_argument("--evidence", required=True)
    diagnose = commands.add_parser("diagnose-customer", help="optional paired Customer diagnostics; spends model budget, never selects skills")
    diagnose.add_argument("--id", required=True)
    diagnose.add_argument("--baseline", required=True, help="exact Customer snapshot hash")
    diagnose.add_argument("--candidate", required=True, help="exact Customer snapshot hash")
    diagnose.add_argument("--service", required=True, help="fixed Service snapshot hash")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    store, archive, runs = LibraryStore(root / "libraries"), EvidenceArchive(root / "evidence"), RunStore(root / "runs")
    try:
        if args.command == "library":
            output = store.load(args.hash).model_dump(mode="json") if args.hash else {"hashes": store.hashes()}
        elif args.command == "history":
            output = {"generations": [g.model_dump(mode="json") for g in runs.history(args.id)],
                      "report": runs.load_report(args.id).model_dump(mode="json")}
        elif args.command == "restore":
            output = runs.branch_config(args.source, args.id, args.generation, args.generations).model_dump(mode="json")
        else:
            if not args.plan:
                raise ValueError("--plan is required")
            if args.command == "evaluate-cross":
                from vast_coevo.evaluation.cross import evaluate_cross
                plan = ExperimentPlan.model_validate_json(Path(args.plan).read_text())
                destination = Path(args.output).resolve()
                if destination == root or any(destination == root/name or root/name in destination.parents
                        for name in ('runs','libraries','evidence','native')):
                    raise ValueError('evaluation must use an independent output directory')
                excluded=set(); model_pairs=set()
                for ident in args.training_run:
                    source=runs.load_config(ident); report=runs.load_report(ident)
                    if report.status!='completed' or len(report.generations)!=source.generations or runs.history(ident)!=report.generations:
                        raise ValueError('evaluation requires completed source training')
                    cases=source.exploration_cases+source.validation_cases+source.service_task_cases
                    excluded.update(c.config.initial_world_hash for c in cases)
                    model_pairs.update((digest(c.config.customer_model),digest(c.config.service_model)) for c in cases)
                if len(model_pairs)!=1:raise ValueError('source training models differ')
                if plan.customer_model is None or plan.service_model is None or (digest(plan.customer_model),digest(plan.service_model)) not in model_pairs:
                    raise ValueError('evaluation plan must pin the same Agent models as training')
                profiles=[p for p in plan.profiles if p.config_id in args.profile]
                if {p.config_id for p in profiles}!=set(args.profile):raise ValueError('unregistered evaluation profile')
                pairs={}
                for choice in args.pair:
                    label,ch,sh=choice.split(':')
                    if label in pairs:raise ValueError('duplicate pairing label')
                    pairs[label]=(store.load(ch),store.load(sh))
                evaluation_adapter=UsamiAdapter(destination/'native',profiles)
                configs=tuple(evaluation_adapter.config_for(p.config_id,plan.customer_model,plan.service_model,plan.budget) for p in profiles)
                output=evaluate_cross(evaluation_adapter,pairs,configs,destination,evaluation_id=args.id,
                    purpose=args.purpose,excluded_world_hashes=excluded,max_llm_calls=plan.max_llm_calls,max_tool_calls=plan.max_tool_calls)
                print(json.dumps(output,ensure_ascii=False,indent=2));return 0
            if args.command == "replay":
                # Offline replay does not need model environment or credentials.
                plan = ExperimentPlan.model_validate_json(Path(args.plan).read_text())
                adapter = UsamiAdapter(root / "native", plan.profiles, plan.service_tasks)
                evidence = _evidence(args.evidence, archive)
                output = adapter.replay(evidence, store.load(evidence.customer_snapshot_hash),
                                        store.load(evidence.service_snapshot_hash)).model_dump(mode="json")
                print(json.dumps(output, ensure_ascii=False, indent=2))
                return 0 if output["verified"] else 2
            if args.command == "reflect" and (Path(args.output).exists() or Path(args.output + ".started.json").exists()):
                raise ValueError("Reflection output or in-flight marker already exists; no duplicate call performed")
            if args.command == 'run':
                preflight = ExperimentPlan.model_validate_json(Path(args.plan).read_text())
                saved = root / 'runs' / args.id / 'config.json'
                if saved.exists():
                    existing = runs.load_config(args.id)
                    if existing.schema_version not in (3, 4):
                        raise ValueError('historical training protocol requires its archived code; use a new recursive run ID')
                    if preflight.customer_selection != existing.customer_selection:
                        raise ValueError('resume Customer selection differs from frozen run')
                elif preflight.training_protocol != 'recursive-training-v1':
                    raise ValueError('new runs must explicitly choose training_protocol=recursive-training-v1')
                elif preflight.customer_selection != 'customer-static-only-v1':
                    raise ValueError('new experiments must explicitly choose customer_selection=customer-static-only-v1')
            plan, adapter, store, archive, runs, customer, service, specs = (
                context(args.plan, root, False) if args.command == "diagnose-customer" else context(args.plan, root))
            if args.command == "diagnose-customer":
                # Deliberately separate from evolution: no publish, selection,
                # Reflection or rollback API is available to this execution.
                from vast_coevo.evolution.gates import DynamicGate
                from vast_coevo.evolution.controller import _JournalAdapter
                from vast_coevo.storage.runs import RunHalt
                from .adapter import source_hashes
                baseline = store.load(args.baseline)
                candidate = store.load(args.candidate)
                fixed_service = store.load(args.service)
                if (baseline.role, candidate.role, fixed_service.role) != ('customer', 'customer', 'service'):
                    raise ValueError('diagnostic snapshot roles do not match')
                config = run_config(plan, adapter, baseline, fixed_service, specs, args.id)
                diagnostic_store = RunStore(root / 'diagnostics')
                diagnostic_store.initialize(config)
                destination = diagnostic_store.run_path(args.id) / 'result.json'
                with diagnostic_store.lock(args.id):
                    if destination.exists():
                        raise ValueError('diagnostic already recorded; inspect its result or replay evidence, no repeat performed')
                    metadata = dict(schema_version=1, purpose='customer-diagnostics-only', selection_authority=False,
                        protocol=plan.policy.customer_quality.protocol if plan.policy.customer_quality else 'legacy-v1',
                        baseline_hash=baseline.snapshot_hash, candidate_hash=candidate.snapshot_hash,
                        service_hash=fixed_service.snapshot_hash, config_hash=digest(config), source_hashes=source_hashes())
                    atomic_json(destination.parent / 'manifest.json', metadata, immutable=True)
                    journal = _JournalAdapter(adapter, diagnostic_store, archive, config)
                    try:
                        result = DynamicGate(journal, archive, config.policy).customer(
                            baseline, candidate, fixed_service, config.validation_cases, 'diagnostic.' + args.id)
                    except (Exception, RunHalt) as exc:
                        output = dict(**metadata, status='failed', error=type(exc).__name__ + ': ' + str(exc),
                            consumption=diagnostic_store.consumption(args.id).model_dump(mode='json'),
                            pending_operations=diagnostic_store.pending(args.id))
                        atomic_json(destination, output, immutable=True)
                        raise ValueError('diagnostic failed; record saved at ' + str(destination)) from exc
                    output = dict(**metadata, status='completed', result=result.model_dump(mode='json'),
                        consumption=diagnostic_store.consumption(args.id).model_dump(mode='json'))
                    atomic_json(destination, output, immutable=True)
            elif args.command == "episode":
                config = adapter.config_for(args.profile, specs["customer"], specs["service"], plan.budget)
                evidence = adapter.run_episode(config, customer, service, args.id)
                archive.save(evidence)
                output = evidence.model_dump(mode="json")
            elif args.command == "replay":
                evidence = _evidence(args.evidence, archive)
                customer = store.load(evidence.customer_snapshot_hash)
                service = store.load(evidence.service_snapshot_hash)
                output = adapter.replay(evidence, customer, service).model_dump(mode="json")
            elif args.command == "reflect":
                rows = tuple(_evidence(item, archive) for item in args.evidence)
                for row in rows:
                    checked = adapter.replay(row, store.load(row.customer_snapshot_hash), store.load(row.service_snapshot_hash))
                    if not checked.verified:
                        raise ValueError("Reflection evidence failed native verification: " + checked.reason)
                snapshot = customer if args.role == "customer" else service
                client = transport(specs[args.role + "_reflector"])
                try:
                    klass = CustomerReflector if args.role == "customer" else ServiceReflector
                    profiles = {profile.config_id: profile for profile in plan.profiles}
                    tasks = {task.task_ref: task for task in plan.service_tasks}
                    scope_profiles = set(plan.validation) | {tasks[row.task_ref].config_id for row in plan.regressions}
                    value = ReflectionInput(generation=args.generation, role=args.role, library=snapshot, evidence=rows,
                                            interaction_views=tuple(adapter.reflection_view(row) for row in rows) if args.role == "customer" else None,
                                            domain_policy=DomainPolicy(add_domains=tuple(sorted({profiles[key].domain for key in scope_profiles}))),
                                            allowed_tools=tuple(sorted({t for row in rows for t in row.config.tools})),
                                            public_policy=json.dumps([adapter.public_policy(row.config) for row in rows], sort_keys=True))
                    atomic_json(Path(args.output + ".started.json"), {"schema_version": 1,
                                "input_hash": digest(value), "model": specs[args.role + "_reflector"].model_dump(mode="json")}, immutable=True)
                    result = klass(client, specs[args.role + "_reflector"]).reflect(value)
                finally:
                    client.close()
                atomic_json(Path(args.output), result.model_dump(mode="json"), immutable=True)
                output = result.model_dump(mode="json")
            else:
                config = runs.load_config(args.id) if (root / "runs" / args.id / "config.json").exists() else run_config(
                    plan, adapter, customer, service, specs, args.id, args.generations)
                clients = [transport(config.customer_reflector_model), transport(config.service_reflector_model)]
                try:
                    output = CoEvolutionController(adapter, *clients, store, archive, runs).run(config).model_dump(mode="json")
                finally:
                    for client in clients:
                        client.close()
        print(json.dumps(output, ensure_ascii=False, indent=2))
        if args.command == "replay" and not output["verified"]:
            return 2
        if args.command == "run" and output["status"] != "completed":
            return 2
        return 0
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, "VAST-CoEvo: " + str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
