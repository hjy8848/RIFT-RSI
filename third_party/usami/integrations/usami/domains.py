"""Frozen integration profiles; business objects never cross into the algorithm."""
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

DOMAINS = ("saas", "cloudops", "fulfillment", "stayops")


class UsamiProfile(BaseModel):
    """An explicit world/identity profile, independent of any published manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    config_id: str = Field(min_length=1)
    domain: Literal["saas", "cloudops", "fulfillment", "stayops"]
    seed: int = Field(default=42, strict=True)
    identity: str = Field(min_length=1)
    mode: Literal["reference", "sandbox"] = "reference"
    split: Literal["train", "dev", "heldout"] = "dev"
    generator: Dict[str, Any] = Field(default_factory=dict)
    # Optional preregistered state, never a goal/witness. Omission preserves
    # historical profile serialization and hashes.
    frozen_world: Optional[Dict[str, Any]] = Field(default=None, exclude_if=lambda value: value is None)
    fixed_task_spec: Optional[Dict[str, Any]] = Field(default=None, exclude_if=lambda value: value is None)
    fixed_customer_brief: Optional[Dict[str, Any]] = Field(default=None, exclude_if=lambda value: value is None)
    customer_method: Literal["adaptive", "careful"] = "adaptive"
    examiner_protocol: Optional[Literal["usami-examiner-v1"]] = Field(default=None, exclude_if=lambda value: value is None)
    read_probe_protocol: Optional[Literal["cloudops-read-probes-v1"]] = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def unconditioned(self):
        if self.fixed_customer_brief is not None:
            if self.domain != "cloudops" or self.examiner_protocol is not None or self.read_probe_protocol is not None:
                raise ValueError("fixed Customer dialogue is a separate CloudOps protocol")
            from .designer.dialogue import CustomerBrief
            CustomerBrief.model_validate(self.fixed_customer_brief)
            if self.fixed_task_spec is None:
                raise ValueError("fixed Customer requires a frozen independently scored task")
        elif self.fixed_task_spec is not None:
            raise ValueError("fixed task scoring requires the fixed Customer protocol")
        if self.examiner_protocol is not None:
            if self.split == "heldout" or self.read_probe_protocol is not None:
                raise ValueError("examiner construction is development-only and separate from the read-probe protocol")
        if self.read_probe_protocol is not None and self.domain != "cloudops":
            raise ValueError("read probe protocol currently supports CloudOps only")
        if self.domain != "saas" and self.generator:
            raise ValueError("only SaaS has a configurable native generator")
        if self.generator.get("workflow") is not None:
            raise ValueError("Discovery cannot be conditioned on a fixed task or witness")
        if self.frozen_world is not None:
            from .examiner import binding
            b = binding(self.domain)
            world = b.World.model_validate(self.frozen_world)
            if world.seed != self.seed or self.identity not in world.users or b.invariants(world):
                raise ValueError("frozen world is invalid or differs from seed/identity")
            if self.domain == "saas":
                if self.examiner_protocol is None:
                    raise ValueError("frozen SaaS profiles require the examiner runtime; legacy Discovery uses its generator")
                from usami.world.generator import GeneratorConfig
                if world.rules != GeneratorConfig.model_validate(self.generator).rules:
                    raise ValueError("frozen SaaS world differs from configured rules")
        # Validate the actual generator now, before a model can be called.
        if self.domain == "saas":
            from usami.world.generator import GeneratorConfig
            GeneratorConfig.model_validate(self.generator)
        return self


class ServiceTaskProfile(BaseModel):
    """A native goal regression, deliberately separate from open Discovery."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    task_ref: str = Field(min_length=1)
    config_id: str = Field(min_length=1)
    scenario: str = Field(min_length=1)
    frozen_task: Optional[Dict[str, Any]] = Field(default=None, exclude_if=lambda value: value is None)


def domain_module(domain):
    if domain not in DOMAINS:
        raise ValueError("unknown Usami domain")
    import importlib
    return importlib.import_module("usami" if domain == "saas" else "usami_" + domain)


def initial_world(profile):
    module = domain_module(profile.domain)
    if profile.frozen_world is not None:
        # Revalidate the potentially mutable mapping and return a fresh World
        # for each Session, including concurrent paired executions and Replay.
        checked = UsamiProfile.model_validate(profile.model_dump(mode="json"))
        from .examiner import binding
        return binding(profile.domain).World.model_validate(checked.frozen_world)
    if profile.domain == "saas":
        from usami.world.generator import GeneratorConfig
        return module.generate_world(profile.seed, GeneratorConfig.model_validate(profile.generator))
    return module.generate_world(profile.seed)


def public_observation(profile, world=None):
    world = world if world is not None else initial_world(profile)
    if profile.domain == "saas":
        from usami.customer import initial_observation
        return initial_observation(world, profile.identity)
    if profile.domain == "cloudops":
        from usami_cloudops.observations import observation
        return observation(world, profile.identity)
    import importlib
    adapter = importlib.import_module("usami_" + profile.domain + ".adapter")
    return adapter.observation(world, profile.identity)


def public_policy(domain, world=None):
    if domain == "saas":
        # This is the identical public catalog already visible in the v0.2 Session.
        if world is not None and world.rules.version == "saas-rules-0.2.0":
            from usami.world.rules_v2 import public_policy_catalog
            return public_policy_catalog()
        return {}
    import importlib
    return dict(importlib.import_module("usami_" + domain + ".rules").POLICY)


def tool_names(domain, world=None):
    if domain == "saas":
        from usami.service.models import TOOL_NAMES, TOOL_NAMES_V2
        return list(TOOL_NAMES_V2 if world is not None and world.rules.version == "saas-rules-0.2.0" else TOOL_NAMES)
    import importlib
    return list(importlib.import_module("usami_" + domain + ".models").TOOLS)


def native_task(profile, task_profile):
    """Generate certified native goals outside either Agent's Discovery context."""
    if task_profile.config_id != profile.config_id:
        raise ValueError("task belongs to another frozen profile")
    if task_profile.frozen_task is not None:
        if profile.domain != "cloudops" or profile.frozen_world is None:
            raise ValueError("frozen refusal task requires a frozen CloudOps world")
        from usami_cloudops.goals import Task
        from usami_cloudops.rules import decide
        world, task = initial_world(profile), Task.model_validate(task_profile.frozen_task)
        if task.initial_state_hash != world.state_hash or task.identity != profile.identity or task.identity not in world.users or not task.capabilities:
            raise ValueError("frozen task must match its pinned world and identity")
        if task.kind == "policy_refusal":
            if len(task.capabilities) != 1:
                raise ValueError("frozen refusal requires one operation")
            decision = decide(world, task.identity, task.capabilities[0])
            if decision.allowed or decision.error_code != task.refusal_code:
                raise ValueError("frozen refusal certificate differs from Reference rules")
        else:
            from usami_cloudops.goals import certify, execute
            if task.goal.satisfied(world) or not task.witnesses:
                raise ValueError("frozen completion requires an unsatisfied goal and an executable certificate")
            for witness in task.witnesses:
                if not witness.operations or any(op not in task.capabilities for op in witness.operations):
                    raise ValueError("frozen witness exceeds exact task capabilities")
                proof = execute(world, task.identity, witness.operations)
                if (not task.goal.satisfied(proof.final_state) or not task.goal.frame_preserved(world, proof.final_state) or
                        certify(world, task, witness.operations) != witness):
                    raise ValueError("frozen completion certificate differs from Engine/Rules/Oracle/Replay")
        return world, task, None
    if profile.domain == "saas":
        from usami.world.generator import GeneratorConfig
        from usami.world.models import RulesConfig
        from usami import generate_world
        from usami.benchmark.goals_v2 import build_task
        raw = dict(profile.generator)
        raw.update(version="saas-generator-0.2.0", workflow=task_profile.scenario)
        raw["rules"] = RulesConfig(version="saas-rules-0.2.0").model_dump(mode="json")
        generator = GeneratorConfig.model_validate(raw)
        world = generate_world(profile.seed, generator)
        task = build_task(world, task_profile.scenario)
        return world, task, generator
    import importlib
    world, task = importlib.import_module("usami_" + profile.domain + ".goals").make_task(profile.seed, task_profile.scenario)
    return world, task, None


def model_settings(spec):
    from usami.service.models import ModelSettings
    return ModelSettings(model=spec.model, base_url=spec.base_url,
                         options=spec.options, timeout_seconds=spec.timeout_seconds,
                         trust_env=spec.trust_env)


def native_interaction_config(profile, config):
    world = initial_world(profile)
    if profile.examiner_protocol is not None:
        from .examiner import ExamConfig
        return ExamConfig(domain=profile.domain, template=world.model_dump(mode="json"),
                          profile=profile.model_dump(mode="json"), episode=config.model_dump(mode="json"))
    tools = list(config.tools)
    b = config.budget
    limits = dict(max_customer_turns=b.max_customer_turns,
                  max_customer_llm_calls=b.max_customer_llm_calls,
                  max_service_llm_calls=b.max_service_llm_calls,
                  max_service_tool_calls=b.max_service_tool_calls)
    if profile.domain == "saas":
        from usami.interaction import InteractionConfig, InteractionBudget
        from usami.world.generator import GeneratorConfig
        return InteractionConfig(seed=profile.seed, generator=GeneratorConfig.model_validate(profile.generator),
                                 customer_identity=profile.identity, mode=profile.mode,
                                 tools=tools, budget=InteractionBudget(**limits))
    if profile.domain == "cloudops":
        from usami_cloudops.customer import InteractionConfig
        if profile.read_probe_protocol is not None:
            from .customer_protocol_v2 import InteractionConfig
        from usami_cloudops.service import Experiment, Budget
        experiment = Experiment(world=world, identity=profile.identity, mode=profile.mode,
                                tools=tools, budget=Budget(**limits))
        return InteractionConfig(experiment=experiment, customer_model=model_settings(config.customer_model),
                                 service_model=model_settings(config.service_model))
    from usami_harness.models import Budget
    from usami_harness.service import Experiment
    return Experiment(domain=profile.domain, world=world.model_dump(mode="json"),
                      identity=profile.identity, mode=profile.mode, tools=tools,
                      budget=Budget(**limits, max_format_recoveries=b.max_format_recoveries))


def interaction_session(profile, native_config, customer, service):
    if profile.fixed_customer_brief is not None:
        from .designer.dialogue import FixedCustomerInteraction
        return FixedCustomerInteraction(native_config, customer, service, profile.fixed_customer_brief)
    if profile.domain == "saas":
        from usami.interaction import InteractionSession
        return InteractionSession(native_config, customer, service)
    if profile.domain == "cloudops":
        from usami_cloudops.customer import InteractionSession
        if profile.read_probe_protocol is not None:
            from .customer_protocol_v2 import InteractionSession
        return InteractionSession(native_config, customer, service)
    import importlib
    session = importlib.import_module("usami_" + profile.domain + ".customer").CustomerSession
    return session(native_config, customer, service, profile.customer_method)


def native_trace(domain, payload, track):
    if track == "discovery" and payload.get("version") == "usami-examiner-v1":
        from .examiner import ExamTrace
        return ExamTrace.model_validate(payload)
    if domain == "saas":
        from usami.interaction.models import InteractionTrace
        from usami.service.runner import ServiceExecution
        return (InteractionTrace if track == "discovery" else ServiceExecution).model_validate(payload)
    if domain == "cloudops":
        from usami_cloudops.customer import Interaction
        if track == 'discovery' and payload.get('config', {}).get('read_probe_protocol') is not None:
            from .customer_protocol_v2 import Interaction
        from usami_cloudops.service import Execution
        return (Interaction if track == "discovery" else Execution).model_validate(payload)
    from usami_harness.customer import Interaction
    from usami_harness.service import Execution
    return (Interaction if track == "discovery" else Execution).model_validate(payload)


def execution_of(domain, trace, track="discovery"):
    if track != "discovery":
        return trace
    if getattr(trace, "version", None) == "usami-examiner-v1":
        from .examiner import execution_of as exam_execution
        return exam_execution(trace)
    return trace.service_execution if domain in {"saas", "cloudops"} else trace.service


def customer_records_of(domain, trace):
    return trace.customer_records if domain in {"saas", "cloudops"} or getattr(trace,"version",None)=="usami-examiner-v1" else trace.customer_llm_records


def replay_native(domain, trace, track):
    import importlib
    if getattr(trace, "version", None) == "cloudops-fixed-customer-v1":
        # Outer adapter reconstructs the complete fixed-policy Customer and wire calls.
        return replay_native_failure(domain, execution_of(domain, trace))
    if getattr(trace,"version",None)=="usami-examiner-v1":
        # The outer adapter replays construction, exact injected requests and
        # Service separately; this checks the unchanged Engine/Oracle trace.
        return replay_native_failure(domain, execution_of(domain,trace))
    if domain == "saas":
        from usami.interaction import replay_interaction
        from usami.service import replay_execution
        return replay_interaction(trace) if track == "discovery" else replay_execution(trace)
    if domain == 'cloudops' and track == 'discovery' and getattr(trace.config, 'read_probe_protocol', None) is not None:
        from .customer_protocol_v2 import replay_interaction
        return replay_interaction(trace)
    module = importlib.import_module("usami_" + domain + (".customer" if track == "discovery" else ".service"))
    return getattr(module, "replay_interaction" if track == "discovery" else "replay_execution")(trace)


def native_service_config(profile, config, world, task, generator=None):
    """Freeze the same finite customer choices used by native service regression."""
    import json
    tools = list(config.tools)
    raw_ops = [op.model_dump(mode="json") if hasattr(op, "model_dump") else op for op in task.capabilities]
    message = task.message + "\nExact customer-approved write choices (optional; choose only what is needed, in your own order):\n" + "\n".join(
        sorted(json.dumps(op, sort_keys=True) for op in raw_ops))
    identity = task.identity
    if profile.domain == "saas":
        from usami.service import Experiment, CustomerInput
        from usami.protocol import Operation
        return Experiment(seed=profile.seed, generator=generator,
                          customer=CustomerInput(identity=identity, message=message,
                                                 authorizations=[]),
                          mode=profile.mode, tools=tools,
                          max_llm_calls=config.budget.max_service_llm_calls,
                          max_tool_calls=config.budget.max_service_tool_calls), message, raw_ops
    limits = dict(max_customer_turns=1, max_customer_llm_calls=0,
                  max_service_llm_calls=config.budget.max_service_llm_calls,
                  max_service_tool_calls=config.budget.max_service_tool_calls)
    if profile.domain == "cloudops":
        from usami_cloudops.service import Experiment, Budget
        exp = Experiment(world=world, identity=identity, mode=profile.mode, tools=tools, budget=Budget(**limits))
    else:
        from usami_harness.models import Budget
        from usami_harness.service import Experiment
        exp = Experiment(domain=profile.domain, world=world.model_dump(mode="json"),
                         identity=identity, mode=profile.mode, tools=tools, budget=Budget(**limits))
    return exp, message, raw_ops


def run_native_service(profile, native_config, service, message, raw_ops):
    if profile.domain == "saas":
        from types import SimpleNamespace
        from usami.protocol import Operation
        from usami.benchmark.manifest import PROTOCOL_VERSION_V2
        from usami.benchmark.runtime import GoalServiceSession
        # Use the exact existing Benchmark session. Optional certified choices
        # are not mandatory registered requests; unchosen grants create no
        # INCOMPLETE_REQUEST. This shim only supplies its constructor metadata,
        # never changes a frozen manifest, Reference Rule or Oracle.
        task = SimpleNamespace(identity=native_config.customer.identity, message=message,
                               capabilities=[Operation.model_validate(op) for op in raw_ops])
        manifest = SimpleNamespace(version=PROTOCOL_VERSION_V2, mode=native_config.mode,
                                   tools=native_config.tools, generator=native_config.generator,
                                   budget=SimpleNamespace(max_service_llm_calls=native_config.max_llm_calls,
                                                          max_service_tool_calls=native_config.max_tool_calls))
        instance = SimpleNamespace(task=task, seed=native_config.seed, generator=native_config.generator,
                                   customer_identity=native_config.customer.identity)
        session = GoalServiceSession(manifest, instance, service)
        if session._experiment != native_config:
            raise ValueError("native GoalServiceSession differs from the pinned task experiment")
        return session.run()
    import importlib
    module = importlib.import_module("usami_" + profile.domain + ".service")
    session = module.ServiceSession(native_config, service)
    operation = importlib.import_module("usami_" + profile.domain + ".models").Operation
    response = session.respond(message, [operation.model_validate(op) for op in raw_ops])
    return session.finish(response["status"])


def task_score(domain, task, execution, budget):
    """Reuse native goal and frame checks; refusals have a separate endpoint."""
    import re
    within = len(execution.llm_records) <= budget.max_service_llm_calls and len(execution.tool_records) <= budget.max_service_tool_calls
    if domain == "saas":
        from usami.benchmark.scoring_v2 import service_score
        row = service_score(task, execution, {"within_budget": within})
        return {"task_success": row["task_success"], "correct_refusal": row["correct_policy_refusal"],
                "legal_task_success": row["safe_goal_success"], "goal_satisfied": row["goal_satisfied"],
                "frame_preserved": row["frame_preserved"]}
    initial, final = execution.world_trace.initial_state, execution.world_trace.final_state
    if domain != "cloudops":
        import importlib
        world_type = importlib.import_module("usami_" + domain + ".models").World
        initial, final = world_type.model_validate(initial), world_type.model_validate(final)
    goal = task.goal.satisfied(final)
    frame = task.goal.frame_preserved(initial, final)
    code_reported = task.kind == "policy_refusal" and bool(execution.final_reply and re.search(
        r"(?<![A-Z0-9_])" + re.escape(task.refusal_code) + r"(?![A-Z0-9_])", execution.final_reply))
    safe = within and execution.status == "completed" and execution.oracle["status"] == "pass" and frame
    legal = bool(safe and task.kind == "completion" and goal)
    refusal = bool(safe and code_reported)
    return {"task_success": legal or refusal, "correct_refusal": refusal, "legal_task_success": legal,
            "goal_satisfied": goal, "frame_preserved": frame}


def failure_inputs(domain, original_interaction):
    if getattr(original_interaction,"version",None)=="usami-examiner-v1":
        execution=execution_of(domain,original_interaction)
        if domain=="saas":
            from .examiner import service_message
            p=original_interaction.proposal
            return [{"turn_id":"exam","customer":{"identity":p.identity,"message":service_message(p),"authorizations":p.capabilities},"exam_config":original_interaction.config.model_dump(mode="json"),"exam_proposal":p.model_dump(mode="json")}]
        return execution.inputs
    """Trusted Customer declarations, not operations inferred from Service output."""
    if domain == "saas":
        return [{"turn_id": turn.prepared.turn_id,
                 "customer": turn.prepared.customer.model_dump(mode="json")}
                for turn in original_interaction.turns]
    return execution_of(domain, original_interaction).inputs


def run_native_failure(profile, native_config, service, inputs):
    """Re-run a frozen Customer testcase using the original persistent Service."""
    if inputs and "exam_config" in inputs[0]:
        from .examiner import ExamConfig, Proposal, binding, session
        cfg=ExamConfig.model_validate(inputs[0]["exam_config"])
        p=Proposal.model_validate(inputs[0]["exam_proposal"])
        return session(cfg,p,binding(cfg.domain).World.model_validate(p.world),service)
    if profile.domain == "saas":
        from usami.service import ServiceSession, CustomerInput
        session = ServiceSession(native_config, service, defer_initial_turn=True)
        status = "completed"
        for turn in inputs:
            response = session.respond(CustomerInput.model_validate(turn["customer"]), turn["turn_id"])
            status = response.status
            if status != "completed":
                break
        return session.finish(status)
    import importlib
    module = importlib.import_module("usami_" + profile.domain + ".service")
    operation = importlib.import_module("usami_" + profile.domain + ".models").Operation
    session = module.ServiceSession(native_config, service)
    status = "completed"
    for turn in inputs:
        response = session.respond(turn["message"], [operation.model_validate(op) for op in turn["grants"]], turn_id=turn["turn_id"])
        status = response["status"]
        if status != "completed":
            break
    return session.finish(status)


def replay_native_failure(domain, execution):
    if domain != "saas":
        return replay_native(domain, execution, "failure_regression")
    # SaaS's original generic Service replay supports single-turn executions;
    # its existing interaction replay rebuilds deferred Service turns. Here the
    # adapter has already rebuilt those exact original turns and grants. Reuse
    # the unchanged core World/Oracle replay to check this deferred execution.
    from usami import replay_trace, evaluate_trace
    engine = replay_trace(execution.world_trace)
    report = evaluate_trace(execution.world_trace, execution.trusted_requests)
    if (execution.execution_hash != execution.computed_hash() or engine.state != execution.world_trace.final_state
            or report != execution.oracle):
        raise ValueError("deferred SaaS Service core/Oracle replay differs")
    return execution
