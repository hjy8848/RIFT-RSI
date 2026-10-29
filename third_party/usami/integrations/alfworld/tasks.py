"""ALFWorld task discovery and hash-bound source certificates."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from rsi_core.protocol import digest


DOMAIN_ID = "alfworld-textworld"
DATA_VERSION = "alfworld-json-2.1.1"
SPLITS = ("train", "valid_seen", "valid_unseen")
TASK_TYPE_IDS = {
    "pick_and_place_simple": 1,
    "look_at_obj_in_light": 2,
    "pick_clean_then_place_in_recep": 3,
    "pick_heat_then_place_in_recep": 4,
    "pick_cool_then_place_in_recep": 5,
    "pick_two_obj_and_place": 6,
}


def _sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


@dataclass(frozen=True)
class AlfWorldTask:
    task_ref: str
    split: str
    task_type: str
    game_file: str
    relative_game_file: str
    game_sha256: str
    trajectory_sha256: str
    origin_ref: str
    certificate_ref: str = ""


def discover_tasks(data_root, splits=("train", "valid_seen", "valid_unseen")):
    """Discover the solvable TextWorld games ALFWorld itself registers.

    ``data_root`` is the directory containing ``json_2.1.1`` (normally
    ``$ALFWORLD_DATA``). Candidate files use the same traj_data.json,
    game.tw-pddl, task_type, and solvable fields as AlfredTWEnv.
    """
    root = Path(data_root).expanduser().resolve()
    dataset_root = root / "json_2.1.1"
    if not dataset_root.is_dir():
        raise FileNotFoundError("ALFWorld dataset directory not found: " + str(dataset_root))
    tasks = []
    for split in splits:
        if split not in SPLITS:
            raise ValueError("unsupported ALFWorld split: " + str(split))
        split_root = dataset_root / split
        if not split_root.is_dir():
            raise FileNotFoundError("ALFWorld split not found: " + str(split_root))
        for trajectory_file in sorted(split_root.rglob("traj_data.json")):
            path_text = trajectory_file.as_posix().casefold()
            if "movable" in path_text or "sliced" in path_text:
                continue
            game_file = trajectory_file.with_name("game.tw-pddl")
            if not game_file.is_file():
                continue
            trajectory = json.loads(trajectory_file.read_text(encoding="utf-8"))
            game = json.loads(game_file.read_text(encoding="utf-8"))
            task_type = trajectory.get("task_type")
            if task_type not in TASK_TYPE_IDS or game.get("solvable") is not True:
                continue
            game_hash, trajectory_hash = _sha256(game_file), _sha256(trajectory_file)
            origin_ref = digest({"benchmark": DATA_VERSION,
                                 "game_sha256": game_hash,
                                 "trajectory_sha256": trajectory_hash})
            task_ref = "alfworld-" + digest({"origin_ref": origin_ref, "split": split})[:32]
            tasks.append(AlfWorldTask(
                task_ref=task_ref, split=split, task_type=task_type,
                game_file=str(game_file.resolve()),
                relative_game_file=game_file.resolve().relative_to(root).as_posix(),
                game_sha256=game_hash, trajectory_sha256=trajectory_hash,
                origin_ref=origin_ref))
    refs = [task.task_ref for task in tasks]
    if len(refs) != len(set(refs)):
        raise ValueError("ALFWorld task source is duplicated across registered splits")
    return tuple(tasks)


class AlfWorldTaskRegistry:
    """Host-only map of split-bound ALFWorld tasks and source certificates."""

    def __init__(self, data_root, archive, tasks=None):
        self.data_root = Path(data_root).expanduser().resolve()
        self.archive = archive
        self.tasks = {task.task_ref: task for task in (
            discover_tasks(self.data_root) if tasks is None else tasks)}
        if not self.tasks:
            raise ValueError("ALFWorld registry contains no supported solvable tasks")
        certified = {}
        for task in self.tasks.values():
            certificate = {
                "kind": "alfworld_task_source_certificate_v1",
                "data_version": DATA_VERSION,
                "task_ref": task.task_ref,
                "origin_ref": task.origin_ref,
                "split": task.split,
                "task_type": task.task_type,
                "relative_game_file": task.relative_game_file,
                "game_sha256": task.game_sha256,
                "trajectory_sha256": task.trajectory_sha256,
            }
            certificate_ref = archive.save(certificate)
            certified[task.task_ref] = AlfWorldTask(**{
                **asdict(task), "certificate_ref": certificate_ref})
        self.tasks = certified

    def task(self, task_ref):
        return self.tasks[task_ref]

    def by_split(self, split):
        return tuple(task for task in self.tasks.values() if task.split == split)

    def proof(self, task_ref):
        from rsi_core.packages import TaskProof
        task = self.task(task_ref)
        return TaskProof(task.task_ref, task.origin_ref, task.certificate_ref)

    def verify_task(self, domain_id, proof):
        if domain_id != DOMAIN_ID:
            return False
        task = self.tasks.get(proof.task_ref)
        if task is None or (proof.origin_ref, proof.certificate_ref) != (
                task.origin_ref, task.certificate_ref):
            return False
        certificate = self.archive.load(proof.certificate_ref)
        expected = {
            "kind": "alfworld_task_source_certificate_v1",
            "data_version": DATA_VERSION,
            "task_ref": task.task_ref,
            "origin_ref": task.origin_ref,
            "split": task.split,
            "task_type": task.task_type,
            "relative_game_file": task.relative_game_file,
            "game_sha256": task.game_sha256,
            "trajectory_sha256": task.trajectory_sha256,
        }
        if certificate != expected:
            return False
        game_file = Path(task.game_file)
        trajectory_file = game_file.with_name("traj_data.json")
        return (_sha256(game_file) == task.game_sha256
                and _sha256(trajectory_file) == task.trajectory_sha256)
