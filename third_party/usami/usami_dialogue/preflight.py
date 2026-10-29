"""Cheap, offline Customer/task contract checks before any model call.

This preflight is deliberately distinct from native Reference reachability. It
checks that the registered Customer information and follow-up protocol can be
represented to the selected simulator; it does not predict Service behavior.
"""

from usami_harness.models import digest
from .user_scenario import scenario_from_task, validate_scenario


def customer_interaction_preflight(adapter, world, task, scenario=None):
    if task.composition is None:
        return {"protocol": "customer-interaction-preflight-v1", "status": "not_applicable",
                "reason": "task has no explicit open-composition Customer contract"}
    scenario = scenario or scenario_from_task(adapter, world, task)
    validate_scenario(adapter, world, task, scenario)
    profile = adapter.private_state(world, task.profile_id)
    requirements = [row for row in task.composition.information_requirements
                    if row.profile_id == task.profile_id]
    required_facts = sorted({key for row in requirements for key in row.fact_keys})
    missing = [key for key in required_facts if key not in profile.facts or key not in scenario.known_info]
    terminal_to_option = {"refunded": "refund", "exchanged": "exchange", "reshipped": "reship"}
    resolution_values = {terminal_to_option[value] for goal in task.goals
                         if goal.collection == "lines" and goal.entity == profile.facts.get("line_id")
                         for value in goal.accepted_values if value in terminal_to_option}
    deadline_required = bool(resolution_values & {"exchange", "reship"})
    if deadline_required and ("deadline" not in profile.facts or "deadline" not in scenario.known_info):
        missing.append("deadline")
    if (deadline_required and task.composition.protocol == "open-dialogue-composition-v4" and
            "deadline" not in required_facts):
        missing.append("deadline_information_requirement")
    followups = task.composition.follow_up_requests
    versioned_followup_ok = (task.composition.protocol != "open-dialogue-composition-v4" or
                             (len(followups) == len(task.refusals) and
                              all(row.message in scenario.follow_up_requests for row in followups)))
    checks = {
        "single_frozen_identity": len(task.composition.profile_ids) == 1 and
                                  profile.identity == scenario.identity,
        "required_customer_facts_available": not missing,
        "deadline_available_for_delivery_remedies": not deadline_required or (
            "deadline" in scenario.known_info and
            (task.composition.protocol != "open-dialogue-composition-v4" or "deadline" in required_facts)),
        "versioned_followups_represented_publicly": versioned_followup_ok,
        "followup_action_metadata_hidden_from_scenario": "requested_action" not in scenario.model_dump(mode="json"),
    }
    status = "pass" if all(checks.values()) else "fail"
    result = {
        "protocol": "customer-interaction-preflight-v1",
        "status": status,
        "task_hash": task.task_hash,
        "world_hash": world.state_hash,
        "scenario_hash": scenario.scenario_hash,
        "customer_contract_compatible": status == "pass",
        "required_disclosure_keys": required_facts,
        "missing_customer_facts": sorted(set(missing)),
        "followup_count": len(followups),
        "checks": checks,
        "claim_boundary": "Checks information and protocol compatibility only; native certificate remains the reachability proof and runtime remains the interaction proof.",
    }
    result["preflight_hash"] = digest(result)
    return result
