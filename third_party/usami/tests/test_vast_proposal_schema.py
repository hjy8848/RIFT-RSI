"""Validate actual tool parameters, not just prompt text or schema keywords."""
import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from vast_coevo.reflection.scope import proposal_json_schema
from vast_coevo.skills import SkillPatch, empty_snapshot
from test_vast_evolution import customer_skill, service_skill, proposal


def validator(role=None, wire=True):
    schema = proposal_json_schema(role) if wire else SkillPatch.model_json_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def mutation(role, action):
    skill = customer_skill() if role == "customer" else service_skill()
    raw = proposal(empty_snapshot(role), action, skill=skill).model_dump(mode="json")
    # Generated metadata may be absent on the wire, never the learned content.
    raw["skill"].pop("scope")
    raw["skill"].pop("content_hash")
    return raw


@pytest.mark.parametrize("role", ["customer", "service"])
@pytest.mark.parametrize("action", ["ADD", "UPDATE"])
def test_mutations_require_complete_non_null_role_specific_skill(role, action):
    check = validator(role)
    raw = mutation(role, action)
    assert check.is_valid(raw)
    for missing in ("skill", "null", "id", "trigger", "type", "steps", "empty_steps", "empty_trigger", "wrong_type"):
        invalid = copy.deepcopy(raw)
        steps = "exploration" if role == "customer" else "procedure"
        if missing == "skill":
            invalid.pop("skill")
        elif missing == "null":
            invalid["skill"] = None
        elif missing == "steps":
            invalid["skill"].pop(steps)
        elif missing == "empty_steps":
            invalid["skill"][steps] = []
        elif missing == "empty_trigger":
            invalid["skill"]["trigger"] = {"keywords": [], "intents": []}
        elif missing == "wrong_type":
            invalid["skill"] = mutation("service" if role == "customer" else "customer", action)["skill"]
        else:
            invalid["skill"].pop(missing)
        assert not check.is_valid(invalid), missing
    raw["role"] = "service" if role == "customer" else "customer"
    assert not check.is_valid(raw)


@pytest.mark.parametrize("wire", [True, False])
def test_remove_and_noop_exclude_conflicting_payloads(wire):
    check = validator(wire=wire)
    parent = empty_snapshot("customer")
    remove = proposal(parent, "REMOVE", skill_id="customer.example").model_dump(mode="json")
    assert check.is_valid(remove)
    for value in (None, "", 42):
        invalid = dict(remove, skill_id=value)
        assert not check.is_valid(invalid)
    invalid = dict(remove)
    invalid.pop("skill_id")
    assert not check.is_valid(invalid)
    noop = proposal(parent, "NOOP").model_dump(mode="json")
    assert check.is_valid(noop)
    for raw in (remove, noop):
        omitted = {key: value for key, value in raw.items() if value is not None}
        assert check.is_valid(omitted)
        assert not check.is_valid(dict(raw, skill=customer_skill().model_dump(mode="json")))
    assert not check.is_valid(dict(noop, skill_id="customer.example"))


@pytest.mark.parametrize("role", ["customer", "service"])
def test_stored_patch_schema_retains_required_scope_and_serialization(role):
    raw = mutation(role, "ADD")
    assert validator(role).is_valid(raw)
    assert not validator(wire=False).is_valid(raw)
    skill = customer_skill() if role == "customer" else service_skill()
    patch = proposal(empty_snapshot(role), skill=skill)
    complete = patch.model_dump(mode="json")
    assert validator(wire=False).is_valid(complete)
    assert SkillPatch.model_validate(complete).model_dump(mode="json") == complete


def test_actual_previous_omitted_skill_output_is_rejected_by_new_tool_schema():
    repo = Path(__file__).resolve().parents[1]
    old = json.loads((repo / "experiments/vast_coevo/results/customer-goal-history-001/new-reflection.json").read_text())
    raw = json.loads(old["call"]["response"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert raw["action"] == "ADD" and "skill" not in raw
    assert Draft202012Validator(old["call"]["request"]["tools"][0]["function"]["parameters"]).is_valid(raw)
    assert not validator("customer").is_valid(raw)


def test_unknown_proposal_role_rejected():
    with pytest.raises(ValueError, match="unknown proposal role"):
        proposal_json_schema("unknown")
