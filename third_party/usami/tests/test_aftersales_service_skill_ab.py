import json

import pytest

from experiments.vast_coevo.examples.aftersales_coevo_live import (
    _initial_world,
    _make_task_config,
)
from experiments.vast_coevo.examples.aftersales_coevo_smoke import _single_task
from experiments.vast_coevo.examples.aftersales_service_skill_ab import (
    _base_config,
    _skill_changes,
    _validate_world_disjointness,
    load_certified_task,
)
from experiments.vast_coevo.examples.aftersales_coevo_support import initial_skill_libraries
from usami_benchmark.registry import default_spec
from usami_benchmark.tasks import ConstructionBudget, TaskSelection, compile_task
from usami_benchmark.worlds import freeze
from usami_aftersales import adapter as aftersales


def _first_certifiable_refund_task(frozen):
    world = aftersales.World.model_validate(frozen.world)
    for profile_id in world.profiles:
        selection = TaskSelection(kind="dialogue", family="dispute_resolution",
                                  profile_id=profile_id, parameters={"resolutions": ["refund"]})
        try:
            return compile_task(frozen, selection,
                ConstructionBudget(max_nodes=2048, max_depth=16, max_paths=1))
        except ValueError:
            continue
    raise AssertionError("no native refundable After-sales validation task found")
from vast_coevo.skills import DecisionSkill, LibrarySnapshot


def _candidate(s0):
    old = s0.skills[0].model_dump(mode="json")
    old["version"] += 1
    old["description"] = "Updated guidance for the paired-evaluation fixture."
    old["content_hash"] = ""
    changed = DecisionSkill.model_validate(old)
    return LibrarySnapshot(role="service", skills=(changed,), parent_hash=s0.snapshot_hash,
                           provenance="paired-evaluation-test-candidate")


def test_paired_evaluator_accepts_only_validation_partition_task_files(tmp_path):
    frozen = _initial_world(1)
    record = _single_task(frozen)
    path = tmp_path / "task.json"
    path.write_text(json.dumps(record.model_dump(mode="json")), encoding="utf-8")
    with pytest.raises(ValueError, match="validation partition"):
        load_certified_task(path)


def test_validation_label_alone_does_not_prove_world_independence():
    train = freeze(default_spec("aftersales", 1, partition="train"))
    validation_same_seed = freeze(default_spec("aftersales", 1, partition="validation"))
    validation_new_seed = freeze(default_spec("aftersales", 2, partition="validation"))
    assert train.registration_hash != validation_same_seed.registration_hash
    assert train.world_hash == validation_same_seed.world_hash
    assert train.structure_signature == validation_same_seed.structure_signature
    record = _single_task(validation_same_seed)

    with pytest.raises(ValueError, match="excluded training world hash"):
        _validate_world_disjointness([record], [train])
    _validate_world_disjointness([_first_certifiable_refund_task(validation_new_seed)], [train])


def test_s0_s1_task_configs_differ_only_in_service_skill_context():
    frozen = _initial_world(1)
    task = _single_task(frozen)
    _, s0, _ = initial_skill_libraries()
    s1 = _candidate(s0)
    config0 = _make_task_config(frozen, task, s0, turns=4, mode="sandbox")
    config1 = _make_task_config(frozen, task, s1, turns=4, mode="sandbox")

    assert _base_config(config0) == _base_config(config1)
    assert config0.skill_contexts["service"].snapshot_hash == s0.snapshot_hash
    assert config1.skill_contexts["service"].snapshot_hash == s1.snapshot_hash
    assert config0.skill_contexts["service"].rendered != config1.skill_contexts["service"].rendered


def test_pair_evaluator_rejects_multiple_skill_changes():
    _, s0, _ = initial_skill_libraries()
    s1 = _candidate(s0)
    assert _skill_changes(s0, s1) == ["aftersales.service-decision"]
    assert _skill_changes(s0, s0) == []


def test_paired_evaluator_runs_two_isolated_replayable_conditions_without_reflection(tmp_path, monkeypatch):
    from experiments.vast_coevo.examples import aftersales_service_skill_ab as evaluator
    from experiments.vast_coevo.examples.aftersales_coevo_smoke import (
        CUSTOMER_SETTINGS,
        MODEL_SETTINGS,
        ServiceScriptClient,
        _single_task,
        _task_customer_client,
    )
    from usami_aftersales import adapter

    frozen = freeze(default_spec("aftersales", 1, partition="validation"))
    task = _single_task(frozen)
    excluded = freeze(default_spec("aftersales", 2, partition="train"))
    _, s0, _ = initial_skill_libraries()
    s1 = _candidate(s0)
    task_path, excluded_path = tmp_path / "task.json", tmp_path / "training-world.json"
    s0_path, s1_path = tmp_path / "s0.json", tmp_path / "s1.json"
    task_path.write_text(json.dumps(task.model_dump(mode="json")), encoding="utf-8")
    excluded_path.write_text(json.dumps(excluded.model_dump(mode="json")), encoding="utf-8")
    s0_path.write_text(json.dumps(s0.model_dump(mode="json")), encoding="utf-8")
    s1_path.write_text(json.dumps(s1.model_dump(mode="json")), encoding="utf-8")

    expected_settings = {"customer": CUSTOMER_SETTINGS, "service": MODEL_SETTINGS}
    monkeypatch.setattr(evaluator, "_role_settings", lambda role: expected_settings[role])
    monkeypatch.setattr(evaluator, "key_from_env", lambda: "fixture-key")
    clients = []
    for _ in range(2):
        clients.extend([
            _task_customer_client(1, adapter.World.model_validate(frozen.world), "refund"),
            ServiceScriptClient(task_number=1),
        ])

    class CloseableFixtureClient:
        def __init__(self, delegate):
            self.delegate = delegate
            self.settings = delegate.settings

        def complete(self, request):
            return self.delegate.complete(request)

        def close(self):
            return None

    def client_factory(settings, _key):
        delegate = clients.pop(0)
        assert delegate.settings == settings
        return CloseableFixtureClient(delegate)

    monkeypatch.setattr(evaluator, "OpenAICompatibleClient", client_factory)
    result = evaluator.run_paired_evaluation(
        tmp_path / "paired", task_paths=[task_path], exclude_world_paths=[excluded_path],
        s0_path=s0_path, s1_path=s1_path, mode="reference", repeats=1,
        turns=5, service_calls=9, service_tool_calls=16, max_model_calls=64,
    )

    assert result["status"] == "EXECUTED"
    assert result["reflection_or_skill_selection"] == "disabled"
    assert result["model_calls"] > 0
    assert len(result["episodes"]) == 2
    assert all(row["task_success"] is True for row in result["episodes"])
    assert all(row["replay"]["verified"] is True for row in result["episodes"])
    assert all(row["service_skill_injection"]["all_service_requests_contained_context"]
               for row in result["episodes"])
    assert not clients
