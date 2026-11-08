"""Compile and certify one Designer-authored retail plan."""
from hashlib import sha256
import json

from rift.schemas import DataPurpose, Identity, Split
from rift.verl.retail_registry import CertifiedRetailTask
from .format import PlanError
from .retail_format import parse_retail_plan, retail_plan_scene


def certify_retail_plan(world_kit, text, plan_id, world_seed, *, signatures=False):
    from usami_v2.scenarios.retail_scene import SceneError, compile_scene, scene_task
    from usami_v2.scenarios.retail_signature import retail_signature
    from usami_v2.scenarios.retail_tasks import certify_retail_task
    row = {"plan_id": plan_id, "world_seed": world_seed, "task_refs": [], "answers": [],
           "actions": [], "reasons": [], "signatures": []}
    try:
        plan = parse_retail_plan(text)
        scene, tasks = retail_plan_scene(plan)
    except PlanError as exc:
        return {**row, "status": "format_rejected", "reasons": [exc.code], "message": str(exc)}
    row["plan"] = plan.model_dump(mode="json")
    try:
        world, ids = compile_scene(scene, world_seed)
    except (SceneError, ValueError) as exc:
        return {**row, "status": "world_rejected", "reasons": [getattr(exc, "code", "WORLD_BUILD_FAILED")],
                "message": str(exc)}
    plan_hash = sha256(json.dumps(row["plan"], sort_keys=True, ensure_ascii=False,
                                  separators=(",", ":")).encode("utf-8")).hexdigest()
    # The legacy registry and split registry form a pair, even for a retail-only run.
    if not world_kit.tasks_path.exists():
        world_kit.save()
    for i, description in enumerate(tasks):
        try:
            task = scene_task(world, ids, description, task_id=f"{plan_id}:{i}", scene=scene)
            certificate = certify_retail_task(world, task)
            lineage = json.dumps({"plan_id": plan_id, "world_seed": world_seed, "plan_hash": plan_hash},
                                 sort_keys=True, separators=(",", ":"))
            identity = Identity(world.state_hash, task.record_hash, "retail:designer", lineage,
                                Split.TRAIN, DataPurpose.TRAIN)
            world_kit.retail_registry.register(CertifiedRetailTask(world, task, certificate, identity))
            row["task_refs"].append(task.record_hash)
            row["answers"].append("success")
            row["actions"].append([request.type for request in description.requests])
            if signatures:
                row["signatures"].append(retail_signature(world, task))
        except ValueError as exc:
            reason = str(exc).split(":", 1)[0]
            row["reasons"].append(reason if reason in ("AMBIGUOUS_TARGET", "TASK_NO_STATE_CHANGE",
                                                        "ENGINE_REFERENCE_MISMATCH") else "TASK_REJECTED")
    row["status"] = ("certified" if len(row["task_refs"]) == len(tasks) else
                     "partly_certified" if row["task_refs"] else "tasks_rejected")
    return row
