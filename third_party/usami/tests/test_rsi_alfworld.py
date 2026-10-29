import json
from pathlib import Path

from integrations.alfworld.adapter import AlfWorldAdapter, _at_zero
from integrations.alfworld.packages import AlfWorldChallengeBridge, build_task_banks
from integrations.alfworld.tasks import (AlfWorldTask, AlfWorldTaskRegistry, DOMAIN_ID,
                                         discover_tasks)
from experiments.rsi.seeds.M0_alfworld import next_step as alfworld_research_step
from rsi_core.archive import Archive
from rsi_core.budget import Ledger
from rsi_core.programs import ExperienceArtifact, ProgramArtifact, ServiceSnapshot
from rsi_core.protocol import Completion, FrozenModels, Limits, ModelSnapshot


def _task(root, name="task", split="train", task_type="pick_and_place_simple"):
    Path(root).mkdir(parents=True, exist_ok=True)
    game = Path(root) / "game.tw-pddl"
    game.write_text(json.dumps({"solvable": True, "fixture": name}), encoding="utf-8")
    trajectory = game.with_name("traj_data.json")
    trajectory.write_text(json.dumps({"task_type": task_type, "fixture": name}), encoding="utf-8")
    from integrations.alfworld.tasks import _sha256
    from rsi_core.protocol import digest
    game_hash, trajectory_hash = _sha256(game), _sha256(trajectory)
    origin = digest({"benchmark": "alfworld-json-2.1.1",
                     "game_sha256": game_hash, "trajectory_sha256": trajectory_hash})
    task_ref = "alfworld-" + digest({"origin_ref": origin, "split": split})[:32]
    return AlfWorldTask(task_ref, split, task_type, str(game),
                        game.name, game_hash, trajectory_hash, origin)


class FakeAlfWorld:
    def __init__(self, *, fail_reset=False, never_finish=False):
        self.fail_reset, self.never_finish = fail_reset, never_finish
        self.state = 0
        self.actions = []

    def reset(self):
        if self.fail_reset:
            raise OSError("synthetic environment failure")
        self.state = 0
        self.actions = []
        return {"observation": "Goal: put the apple in the fridge. State 0.",
                "admissible_actions": ("take apple", "open fridge")}

    def step(self, action):
        self.actions.append(action)
        if action not in {"take apple", "open fridge", "put apple in fridge"}:
            return {"observation": "Nothing happens.", "admissible_actions": ("look",),
                    "score": 0.0, "done": False, "won": False}
        self.state += 1
        won = action == "put apple in fridge" and not self.never_finish
        return {"observation": "State " + str(self.state),
                "admissible_actions": ("put apple in fridge", "look"),
                "score": float(won), "done": bool(won), "won": bool(won)}

    def close(self):
        pass


class FakeWorker:
    def __init__(self, actions):
        self.actions = list(actions)

    def invoke(self, program, view, state, ledger, scope):
        return {"kind": "ACTION", "state": state, "action": self.actions.pop(0)}


class ModelWorker:
    def invoke(self, program, view, state, ledger, scope):
        return {"kind": "MODEL", "state": state,
                "messages": [{"role": "user", "content": view["observation"]}]}


class FakeModel:
    def __init__(self, actions):
        self.actions = list(actions)

    def complete(self, snapshot, request, *, max_tokens, seed):
        action = self.actions.pop(0)
        return Completion({"choices": [{"message": {"tool_calls": [{
            "function": {"name": "choose_alfworld_action",
                         "arguments": json.dumps({"action": action})}}]}}]}, 3, 2)


class FailingModel:
    def complete(self, snapshot, request, *, max_tokens, seed):
        raise TimeoutError("synthetic model timeout")


def _adapter(tmp_path, task, factory, *, max_steps=4, model=None):
    return AlfWorldAdapter({task.task_ref: task}, tmp_path / "episodes", factory,
                           max_steps=max_steps, model=model)


def _models():
    base = ModelSnapshot("fixture-model", "fixed")
    return FrozenModels(base, base, base)


def _ledger():
    return Ledger(Limits(model_calls=30, model_tokens=1000, episodes=20,
                         tool_steps=100, research_steps=100, cpu_seconds=100,
                         wall_seconds=100))


def test_batch_observations_are_unwrapped_without_splitting_text_or_actions():
    class Vector:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return "Goal: put the apple in the fridge." if index == 0 else None

    assert _at_zero(Vector()) == "Goal: put the apple in the fridge."
    assert _at_zero("observation") == "observation"
    assert _at_zero(["take apple", "look"]) == ["take apple", "look"]
    assert _at_zero([["take apple", "look"]]) == ["take apple", "look"]


def test_native_action_and_won_score_replay(tmp_path):
    task = _task(tmp_path / "game")
    envs = []
    def factory(task, seed):
        env = FakeAlfWorld()
        envs.append(env)
        return env
    adapter = _adapter(tmp_path, task, factory)
    program = ProgramArtifact("P", "def step(view, state):\n    return {'kind':'ACTION','state':state,'action':'x'}\n")
    result = adapter.run_task(program, task.task_ref, _models(), _ledger(),
                              FakeWorker(["take apple", "put apple in fridge"]), seed=7)
    assert result.status == "VERIFIED"
    assert result.score == 1.0
    assert result.tool_steps == 2
    assert envs[0].actions == ["take apple", "put apple in fridge"]
    assert adapter.replay(result.run_ref)["verified"] is True


def test_model_action_is_sent_to_native_environment_without_host_language_gate(tmp_path):
    task = _task(tmp_path / "game")
    envs = []
    def factory(task, seed):
        env = FakeAlfWorld()
        envs.append(env)
        return env
    adapter = _adapter(tmp_path, task, factory, max_steps=1,
                       model=FakeModel(["invented action"]))
    program = ProgramArtifact("P", "def step(view, state):\n    return {'kind':'MODEL','state':state,'messages':[]}\n")
    result = adapter.run_task(program, task.task_ref, _models(), _ledger(), ModelWorker(), seed=9)
    assert result.status == "VERIFIED"
    assert result.score == 0.0
    assert envs[0].actions == ["invented action"]
    assert adapter.replay(result.run_ref)["verified"] is True


def test_completed_but_unsolved_step_limit_is_verified_zero(tmp_path):
    task = _task(tmp_path / "game")
    adapter = _adapter(tmp_path, task, lambda *_: FakeAlfWorld(never_finish=True), max_steps=2)
    program = ProgramArtifact("P", "def step(view, state):\n    return {'kind':'ACTION','state':state,'action':'look'}\n")
    result = adapter.run_task(program, task.task_ref, _models(), _ledger(),
                              FakeWorker(["take apple", "open fridge"]), seed=1)
    assert result.status == "VERIFIED"
    assert result.score == 0.0
    assert result.tool_steps == 2
    assert adapter.replay(result.run_ref)["verified"] is True


def test_reproducible_environment_failure_is_unknown_not_success(tmp_path):
    task = _task(tmp_path / "game")
    adapter = _adapter(tmp_path, task, lambda *_: FakeAlfWorld(fail_reset=True))
    program = ProgramArtifact("P", "def step(view, state):\n    return {'kind':'ACTION','state':state,'action':'look'}\n")
    result = adapter.run_task(program, task.task_ref, _models(), _ledger(), FakeWorker([]), seed=3)
    assert result.status == "UNKNOWN"
    assert result.score == 0.0
    assert result.replay_verified is True
    assert adapter.replay(result.run_ref)["verified"] is True


def test_reproducible_environment_factory_failure_is_archived_and_replayed(tmp_path):
    task = _task(tmp_path / "game")

    def factory(*_):
        raise ImportError("synthetic ALFWorld import failure")

    adapter = _adapter(tmp_path, task, factory)
    program = ProgramArtifact("P", "def step(view, state):\n    return {'kind':'ACTION','state':state,'action':'look'}\n")
    result = adapter.run_task(program, task.task_ref, _models(), _ledger(), FakeWorker([]), seed=6)
    record = json.loads((adapter.root / (result.run_ref + ".json")).read_text())
    assert result.status == "UNKNOWN"
    assert record["terminal"]["error_type"] == "ImportError"
    assert adapter.replay(result.run_ref)["verified"] is True


def test_model_provider_failure_is_recorded_as_unknown_and_environment_replayable(tmp_path):
    task = _task(tmp_path / "game")
    adapter = _adapter(tmp_path, task, lambda *_: FakeAlfWorld(), model=FailingModel())
    program = ProgramArtifact("P", "def step(view, state):\n    return {'kind':'MODEL','state':state,'messages':[]}\n")
    result = adapter.run_task(program, task.task_ref, _models(), _ledger(), ModelWorker(), seed=4)
    record = json.loads((adapter.root / (result.run_ref + ".json")).read_text())
    assert result.status == "UNKNOWN"
    assert result.score == 0.0
    assert result.model_calls == 1
    assert record["terminal"]["kind"] == "model_error"
    assert record["model_events"][0]["status"] == "ERROR"
    replay = adapter.replay(result.run_ref)
    assert replay["verified"] is True
    assert replay["replay_scope"] == "native environment actions; model calls are not reissued"


def test_replay_detects_episode_record_tampering(tmp_path):
    task = _task(tmp_path / "game")
    adapter = _adapter(tmp_path, task, lambda *_: FakeAlfWorld(), max_steps=1)
    program = ProgramArtifact("P", "def step(view, state):\n    return {'kind':'ACTION','state':state,'action':'x'}\n")
    result = adapter.run_task(program, task.task_ref, _models(), _ledger(),
                              FakeWorker(["take apple"]), seed=5)
    path = adapter.root / (result.run_ref + ".json")
    record = json.loads(path.read_text())
    record["steps"][0]["action"] = "put apple in fridge"
    path.write_text(json.dumps(record))
    assert adapter.replay(result.run_ref) == {
        "verified": False, "reason": "episode record hash mismatch"}


def test_task_registry_and_period_banks_keep_train_and_heldout_separate(tmp_path):
    root = tmp_path / "alfworld-data"
    train_types = list(__import__("integrations.alfworld.tasks", fromlist=["TASK_TYPE_IDS"]).TASK_TYPE_IDS)
    tasks = []
    for index in range(10):
        path = root / "json_2.1.1" / "train" / ("game-" + str(index))
        path.mkdir(parents=True)
        tasks.append(_task(path, "game-" + str(index), task_type=train_types[index % len(train_types)]))
    test_path = root / "json_2.1.1" / "valid_unseen" / "game-test"
    test_path.mkdir(parents=True)
    heldout = _task(test_path, "heldout", split="valid_unseen", task_type=train_types[0])
    archive = Archive(tmp_path / "archive")
    registry = AlfWorldTaskRegistry(root, archive, tasks + [heldout])
    assert registry.verify_task(DOMAIN_ID, registry.proof(tasks[0].task_ref)) is True
    snapshot = ModelSnapshot("fixed", "fixed")
    service = ServiceSnapshot(snapshot.ref,
                              ProgramArtifact("P", "def step(view,state):\n    return {'kind':'ACTION','state':state,'action':'look'}\n"),
                              ExperienceArtifact())
    banks = build_task_banks(registry, archive, service)
    refs = [ref for task_set in banks.task_sets.values() for ref in task_set.task_refs]
    assert len(refs) == len(set(refs)) == 3
    assert banks.audit_task_set.partition == "meta-test"
    assert banks.audit_task_set.task_refs == (heldout.task_ref,)
    assert banks.challenge_tasks
    challenge = AlfWorldChallengeBridge(registry, banks.challenge_tasks)
    task_type = banks.challenge_tasks[0].task_type
    receipt = challenge.certify({"task_type": task_type, "hypothesis": "Compare this task family."})
    assert receipt["status"] == "CERTIFIED"
    assert receipt["task_ref"] in {row.task_ref for row in banks.challenge_tasks}


def test_discovery_matches_alfworld_game_metadata_and_certificates_source(tmp_path):
    root = tmp_path / "alfworld-data"
    task_type = "pick_and_place_simple"
    normal_dir = root / "json_2.1.1" / "train" / "normal"
    normal = _task(normal_dir, "normal", task_type=task_type)
    movable_dir = root / "json_2.1.1" / "train" / "movable" / "variation"
    _task(movable_dir, "movable", task_type=task_type)
    unsolvable_dir = root / "json_2.1.1" / "train" / "unsolvable"
    unsolvable = _task(unsolvable_dir, "unsolvable", task_type=task_type)
    Path(unsolvable.game_file).write_text(json.dumps({"solvable": False}), encoding="utf-8")

    discovered = discover_tasks(root, splits=("train",))
    assert [task.game_file for task in discovered] == [normal.game_file]
    archive = Archive(tmp_path / "archive")
    registry = AlfWorldTaskRegistry(root, archive, discovered)
    registered = registry.task(discovered[0].task_ref)
    assert registry.verify_task(DOMAIN_ID, registry.proof(registered.task_ref)) is True
    Path(registered.game_file).write_text(json.dumps({"solvable": False}), encoding="utf-8")
    assert registry.verify_task(DOMAIN_ID, registry.proof(registered.task_ref)) is False


def test_task_origin_is_content_bound_across_splits(tmp_path):
    root = tmp_path / "alfworld-data"
    task_types = list(__import__("integrations.alfworld.tasks", fromlist=["TASK_TYPE_IDS"]).TASK_TYPE_IDS)
    train = []
    for index in range(10):
        path = root / "json_2.1.1" / "train" / ("game-" + str(index))
        path.mkdir(parents=True)
        train.append(_task(path, "shared" if index == 0 else "train-" + str(index),
                           task_type=task_types[index % len(task_types)]))
    heldout_path = root / "json_2.1.1" / "valid_unseen" / "duplicate"
    heldout_path.mkdir(parents=True)
    heldout = _task(heldout_path, "shared", split="valid_unseen", task_type=task_types[0])
    assert heldout.task_ref != train[0].task_ref
    assert heldout.origin_ref == train[0].origin_ref
    archive = Archive(tmp_path / "archive")
    registry = AlfWorldTaskRegistry(root, archive, train + [heldout])
    snapshot = ModelSnapshot("fixed", "fixed")
    service = ServiceSnapshot(snapshot.ref,
                              ProgramArtifact("P", "def step(view,state):\n    return {'kind':'ACTION','state':state,'action':'look'}\n"),
                              ExperienceArtifact())
    try:
        build_task_banks(registry, archive, service)
    except ValueError as exc:
        assert "overlaps the train split" in str(exc)
    else:
        raise AssertionError("identical source content crossed train and held-out splits")


def test_alfworld_designer_supports_core_independent_adaptation_mode():
    view = {"mode": "EVALUATE_ADAPT", "target": {
        "ref": "parent", "policy": {"source": "def step(view,state): return {}"},
        "experience": {}}, "observations": [],
        "capabilities": {"probe_refs": ["probe-task"]},
        "environment_capabilities": {}, "prior_service_research": []}
    model_step = alfworld_research_step(view, {})
    assert model_step["action"] == "MODEL"
    proposal = {"policy_source": "def step(view,state): return {}", "experience": {}}
    stage = alfworld_research_step({**view, "observations": [
        {"kind": "MODEL", "output": proposal}]}, model_step["state"])
    assert stage["action"] == "STAGE" and stage["kind"] == "SERVICE_BUNDLE"
    state = {**stage["state"], "refs": ["parent"], "cursor": 0, "scores": {}}
    observations = [{"kind": "MODEL", "output": proposal},
                    {"kind": "STAGE", "status": "VALID", "candidate_ref": "candidate"}]
    first_probe = alfworld_research_step({**view, "observations": observations}, state)
    assert (first_probe["candidate_ref"], first_probe["probe_ref"]) == ("parent", "probe-task")
    observations.append({"kind": "PROBE", "candidate_ref": "parent",
                         "result": {"score": 0.0}})
    second_probe = alfworld_research_step({**view, "observations": observations}, first_probe["state"])
    assert (second_probe["candidate_ref"], second_probe["probe_ref"]) == ("candidate", "probe-task")
    observations.append({"kind": "PROBE", "candidate_ref": "candidate",
                         "result": {"score": 1.0}})
    submit = alfworld_research_step({**view, "observations": observations}, second_probe["state"])
    assert submit == {"action": "SUBMIT", "state": {}, "candidate_ref": "candidate"}
