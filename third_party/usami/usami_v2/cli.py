"""Implemented Usami 2.0 commands only; importing never looks up credentials."""
import argparse
import json
from pathlib import Path

from vast_coevo.schema import atomic_json, digest
from usami_v2 import PROTOCOL
from usami_v2.contracts import CommunicationObligation, FactEvidence, PublicRequest, ServiceMessage
from usami_v2.commerce.models import Operation, PARAMETERS, Quote, World
from usami_v2.evaluation.checkpoints import DecisionCheckpoint
from usami_v2.evaluation.challenge_quality import ChallengeDescriptor, PreRunQuality, PreRunQualityV1
from usami_v2.evaluation.readiness import ReadinessPolicy, ReadinessReport
from usami_v2.runtime.observations import WorkingState
from usami_v2.scenarios.builder import BuildResult, build_world, verify_build
from usami_v2.scenarios.distribution import DistributionSpec
from usami_v2.scenarios.schema import BuildBudget, ScenarioSpec
from usami_v2.scenarios.task_certification import (
    CustomerScenario, CustomerTaskScope, PlanCandidate, ProofAttempt, ProofBudget, ProofStep, ReviewPlan,
    TaskDesignCapabilities,
    TaskCertification, certify_task, verify_task_certificate,
)
from usami_v2.scenarios.tasks import (
    CustomerRequestBundle, InformationObjective, RemedyGoal, StructuredInformationObjective,
    TaskSpec, TaskSpecV3, parse_task_spec,
)


def contract_schemas():
    from usami_v2.runtime.agents import AgentCall, AgentMessage, CustomerPrompt, ServiceAction, ServicePrompt
    from usami_v2.runtime.customer import (
        AuthorityProof, CustomerDecision, CustomerEvent, CustomerProfile, CustomerProfileV2,
        CustomerProfileV3, CustomerTransition,
        OfferPresentation, PrivateCustomerFact,
    )
    from usami_v2.runtime.runner import EpisodeBudget, EpisodeEvent, EpisodeRecord, EpisodeReplay, ToolExecution
    from usami_v2.scenarios.designer import (
        CandidateConsequenceDiagnostic, ChallengeQualityVector, ChallengeQualityVectorV1,
        DesignerCall, DesignerRun, DesignerTrainingEvidence, PostRunQualityVector,
        ScenarioDraft, TaskDesignAttempt, TaskDraft, WorldDesignAttempt,
    )
    from usami_v2.scenarios.distribution import SamplingSlot
    from usami_v2.scenarios.tasks import GroundedRemedyGoal, TaskSpecV2
    from usami_v2.evaluation.outcomes import OutcomeV2

    models = (CommunicationObligation, FactEvidence, PublicRequest, ServiceMessage,
              Quote, World, Operation, DecisionCheckpoint, ChallengeDescriptor,
              PreRunQuality, PreRunQualityV1,
              ReadinessPolicy, ReadinessReport, WorkingState, DistributionSpec,
              ScenarioSpec, BuildBudget, BuildResult, RemedyGoal,
              InformationObjective, TaskSpec, CustomerScenario, ReviewPlan,
              ProofBudget, ProofStep, PlanCandidate, ProofAttempt, TaskCertification,
              CustomerTaskScope, TaskDesignCapabilities,
              CustomerDecision, CustomerProfile, PrivateCustomerFact, OfferPresentation,
              CustomerProfileV2, CustomerProfileV3, GroundedRemedyGoal, TaskSpecV2, TaskSpecV3,
              CustomerRequestBundle, StructuredInformationObjective, OutcomeV2,
              AuthorityProof, CustomerEvent, CustomerTransition,
              AgentMessage, AgentCall, CustomerPrompt, ServicePrompt, ServiceAction,
              EpisodeBudget, EpisodeEvent, EpisodeRecord, EpisodeReplay, ToolExecution,
              SamplingSlot, ScenarioDraft, TaskDraft, DesignerCall, DesignerRun,
              TaskDesignAttempt, WorldDesignAttempt, DesignerTrainingEvidence,
              CandidateConsequenceDiagnostic, ChallengeQualityVectorV1, ChallengeQualityVector,
              PostRunQualityVector)
    schemas = {model.__name__: model.model_json_schema() for model in models}
    schemas["BusinessToolParameters"] = {
        "protocol": "commerce-tool-parameters-v1",
        "tools": {name: model.model_json_schema() for name, model in sorted(PARAMETERS.items())},
    }
    return schemas


def export_schemas(root):
    root = Path(root)
    # Failure leaves any partial output visible. Never replace another attempt.
    root.mkdir(parents=True, exist_ok=False)
    schemas = contract_schemas()
    for name, schema in schemas.items():
        atomic_json(root / (name + ".json"), schema, immutable=True)
    manifest = {
        "protocol": "commerce-schema-bundle-v1", "research_protocol": PROTOCOL,
        "schema_hashes": {name: digest(schema) for name, schema in schemas.items()},
        "model_calls": 0,
        "claim_boundary": "Contract schemas only; no business task, real-model run or readiness PASS.",
    }
    manifest["manifest_hash"] = digest(manifest)
    atomic_json(root / "manifest.json", manifest, immutable=True)
    return manifest


def verify_schemas(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected_hash = manifest.get("manifest_hash")
    if expected_hash != digest({k: v for k, v in manifest.items() if k != "manifest_hash"}):
        raise ValueError("schema manifest integrity differs")
    if manifest.get("protocol") != "commerce-schema-bundle-v1" or manifest.get("model_calls") != 0:
        raise ValueError("unrecognized or falsely attributed contract artifact")
    expected = contract_schemas()
    if manifest.get("schema_hashes") != {name: digest(schema) for name, schema in expected.items()}:
        raise ValueError("schema bundle differs from this version of the runtime")
    if {p.name for p in root.iterdir()} != {name + ".json" for name in expected} | {"manifest.json"}:
        raise ValueError("schema bundle file set differs")
    for name, schema in expected.items():
        if json.loads((root / (name + ".json")).read_text(encoding="utf-8")) != schema:
            raise ValueError("schema content was changed: " + name)
    return {"verified": True, "schema_count": len(expected), "model_calls": 0,
            "manifest_hash": expected_hash}


def build_scenario_artifact(scenario_path, seed, output_path, max_nodes=256):
    """Build and freeze one world. SAT, UNSAT and UNKNOWN all retain their evidence."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(str(output_path))
    raw = json.loads(Path(scenario_path).read_text(encoding="utf-8"))
    scenario = ScenarioSpec.model_validate(raw)
    budget = BuildBudget(max_nodes=max_nodes)
    result = build_world(scenario, seed, budget)
    output_path.mkdir(parents=True, exist_ok=False)
    atomic_json(output_path / "scenario.json", scenario, immutable=True)
    atomic_json(output_path / "build.json", result, immutable=True)
    manifest = {
        "protocol": "commerce-world-artifact-v1",
        "research_protocol": PROTOCOL,
        "scenario_hash": scenario.record_hash,
        "build_hash": result.record_hash,
        "world_hash": result.world_hash,
        "status": result.status,
        "model_calls": 0,
        "artifact_files": ["scenario.json", "build.json"],
    }
    manifest["manifest_hash"] = digest(manifest)
    atomic_json(output_path / "manifest.json", manifest, immutable=True)
    return manifest


def verify_scenario_artifact(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("manifest_hash") != digest({k: v for k, v in manifest.items() if k != "manifest_hash"}):
        raise ValueError("world artifact manifest integrity differs")
    if manifest.get("protocol") != "commerce-world-artifact-v1" or manifest.get("model_calls") != 0:
        raise ValueError("unrecognized or falsely attributed world artifact")
    if manifest.get("artifact_files") != ["scenario.json", "build.json"]:
        raise ValueError("world artifact manifest file list differs")
    if set(p.name for p in root.iterdir()) != {"scenario.json", "build.json", "manifest.json"}:
        raise ValueError("world artifact file set differs")
    scenario = ScenarioSpec.model_validate(json.loads((root / "scenario.json").read_text(encoding="utf-8")))
    result = BuildResult.model_validate(json.loads((root / "build.json").read_text(encoding="utf-8")))
    if (manifest.get("scenario_hash") != scenario.record_hash or manifest.get("build_hash") != result.record_hash
            or manifest.get("world_hash") != result.world_hash or manifest.get("status") != result.status):
        raise ValueError("world artifact content differs from manifest")
    proof = verify_build(scenario, result)
    return {"verified": True, "model_calls": 0, **proof, "manifest_hash": manifest["manifest_hash"]}


def certify_task_artifact(world_directory, task_path, customer_scenario_path, output_path,
                          *, max_plan_candidates=256, max_operation_steps=64,
                          max_witnesses=2, minimum_witnesses=1):
    """Create a private, offline Reference certificate without a model call."""
    world_directory, output_path = Path(world_directory), Path(output_path)
    if output_path.exists():
        raise FileExistsError(str(output_path))
    verify_scenario_artifact(world_directory)
    build = BuildResult.model_validate(json.loads((world_directory / "build.json").read_text(encoding="utf-8")))
    if build.status != "SAT" or build.world is None:
        raise ValueError("task certification requires a previously verified SAT world artifact")
    task = parse_task_spec(json.loads(Path(task_path).read_text(encoding="utf-8")))
    customer_scenario = CustomerScenario.model_validate(
        json.loads(Path(customer_scenario_path).read_text(encoding="utf-8")))
    budget = ProofBudget(max_plan_candidates=max_plan_candidates,
                         max_operation_steps=max_operation_steps, max_witnesses=max_witnesses)
    certificate = certify_task(build.world, task, customer_scenario, budget,
                               minimum_witnesses=minimum_witnesses)
    output_path.mkdir(parents=True, exist_ok=False)
    atomic_json(output_path / "task.json", task, immutable=True)
    atomic_json(output_path / "customer-scenario.private.json", customer_scenario, immutable=True)
    atomic_json(output_path / "certificate.json", certificate, immutable=True)
    manifest = {
        "protocol": "commerce-task-certificate-artifact-v1",
        "research_protocol": PROTOCOL,
        "world_hash": build.world_hash,
        "task_hash": task.record_hash,
        "customer_scenario_hash": customer_scenario.record_hash,
        "proof_budget_hash": budget.record_hash,
        "certificate_hash": certificate.record_hash,
        "status": certificate.status,
        "model_calls": 0,
        "artifact_files": ["task.json", "customer-scenario.private.json", "certificate.json"],
        "privacy_boundary": "CustomerScenario and proof witnesses are private validator artifacts; never inject into Service context.",
        "claim_boundary": "Reference business reachability only; no natural Customer interaction or model learning is certified.",
    }
    manifest["manifest_hash"] = digest(manifest)
    atomic_json(output_path / "manifest.json", manifest, immutable=True)
    return {"status": certificate.status, "certificate_hash": certificate.record_hash,
            "witness_count": len(certificate.witness_hashes), "model_calls": 0,
            "artifact_path": str(output_path)}


def verify_task_certificate_artifact(world_directory, certificate_directory):
    world_directory, certificate_directory = Path(world_directory), Path(certificate_directory)
    verify_scenario_artifact(world_directory)
    manifest = json.loads((certificate_directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("manifest_hash") != digest({k: v for k, v in manifest.items() if k != "manifest_hash"}):
        raise ValueError("task certificate manifest integrity differs")
    expected_files = {"manifest.json", "task.json", "customer-scenario.private.json", "certificate.json"}
    if set(path.name for path in certificate_directory.iterdir()) != expected_files:
        raise ValueError("task certificate artifact file set differs")
    if (manifest.get("protocol") != "commerce-task-certificate-artifact-v1"
            or manifest.get("model_calls") != 0
            or manifest.get("artifact_files") != ["task.json", "customer-scenario.private.json", "certificate.json"]):
        raise ValueError("task certificate artifact protocol differs")
    build = BuildResult.model_validate(json.loads((world_directory / "build.json").read_text(encoding="utf-8")))
    task = parse_task_spec(json.loads((certificate_directory / "task.json").read_text(encoding="utf-8")))
    customer_scenario = CustomerScenario.model_validate(
        json.loads((certificate_directory / "customer-scenario.private.json").read_text(encoding="utf-8")))
    certificate = TaskCertification.model_validate(
        json.loads((certificate_directory / "certificate.json").read_text(encoding="utf-8")))
    bindings = (manifest.get("world_hash") == build.world_hash
                and manifest.get("task_hash") == task.record_hash
                and manifest.get("customer_scenario_hash") == customer_scenario.record_hash
                and manifest.get("proof_budget_hash") == certificate.budget_hash
                and manifest.get("certificate_hash") == certificate.record_hash
                and manifest.get("status") == certificate.status)
    if not bindings:
        raise ValueError("task certificate manifest binding differs")
    result = verify_task_certificate(build.world, task, customer_scenario, certificate)
    return {**result, "model_calls": 0, "manifest_hash": manifest["manifest_hash"]}


def verify_episode_artifact(path):
    """Offline replay only. Episode artifacts contain private Customer/model context."""
    from usami_v2.runtime.runner import verify_episode_artifact as verify
    result = verify(path)
    if not result["verified"]:
        raise ValueError("Episode Replay failed: " + str(result["replay"].get("errors", ())))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=PROTOCOL)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("schema", help="export implemented contracts; zero model requests")
    export.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify-schemas", help="offline integrity and current-schema verification")
    verify.add_argument("--directory", type=Path, required=True)
    build = commands.add_parser("build", help="build one constrained world; preserves SAT/UNSAT/UNKNOWN evidence")
    build.add_argument("--scenario", type=Path, required=True)
    build.add_argument("--seed", type=int, required=True)
    build.add_argument("--max-nodes", type=int, default=256)
    build.add_argument("--output", type=Path, required=True)
    verify_build_parser = commands.add_parser("verify-build", help="reconstruct and verify a frozen world artifact")
    verify_build_parser.add_argument("--directory", type=Path, required=True)
    certify = commands.add_parser("certify-task", help="bounded offline Reference search for a frozen task")
    certify.add_argument("--world-directory", type=Path, required=True)
    certify.add_argument("--task", type=Path, required=True)
    certify.add_argument("--customer-scenario", type=Path, required=True)
    certify.add_argument("--max-plan-candidates", type=int, default=256)
    certify.add_argument("--max-operation-steps", type=int, default=64)
    certify.add_argument("--max-witnesses", type=int, default=2)
    certify.add_argument("--minimum-witnesses", type=int, default=1)
    certify.add_argument("--output", type=Path, required=True)
    verify_task = commands.add_parser("verify-task-certificate", help="replay a sealed task proof artifact")
    verify_task.add_argument("--world-directory", type=Path, required=True)
    verify_task.add_argument("--directory", type=Path, required=True)
    verify_episode = commands.add_parser("verify-episode", help="offline Engine/Oracle/communication Replay of a private Episode artifact")
    verify_episode.add_argument("--path", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "schema":
        result = export_schemas(args.output)
    elif args.command == "verify-schemas":
        result = verify_schemas(args.directory)
    elif args.command == "build":
        result = build_scenario_artifact(args.scenario, args.seed, args.output, args.max_nodes)
    elif args.command == "certify-task":
        result = certify_task_artifact(
            args.world_directory, args.task, args.customer_scenario, args.output,
            max_plan_candidates=args.max_plan_candidates,
            max_operation_steps=args.max_operation_steps,
            max_witnesses=args.max_witnesses, minimum_witnesses=args.minimum_witnesses)
    elif args.command == "verify-task-certificate":
        result = verify_task_certificate_artifact(args.world_directory, args.directory)
    elif args.command == "verify-episode":
        result = verify_episode_artifact(args.path)
    else:
        result = verify_scenario_artifact(args.directory)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
