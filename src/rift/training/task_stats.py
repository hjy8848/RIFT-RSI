"""Observed GRPO task outcomes for benchmark selection; no extra rollouts."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

from rift.schemas import Checkpoint


@dataclass(frozen=True)
class TaskObservation:
    task_ref: str
    checkpoint: Checkpoint
    round_index: int
    successes: int
    group_size: int

    def __post_init__(self):
        if (not self.task_ref or not isinstance(self.checkpoint, Checkpoint)
                or type(self.round_index) is not int or self.round_index < 0
                or type(self.group_size) is not int or self.group_size < 2
                or type(self.successes) is not int or not 0 <= self.successes <= self.group_size):
            raise ValueError('invalid observed task group')


class TaskStatsStore:
    """Append-only JSONL; repeated recovery of an identical observation is harmless."""
    def __init__(self, path):
        self.path = Path(path)

    def read(self):
        if not self.path.exists():
            return ()
        result = []
        with self.path.open(encoding='utf-8') as stream:
            for line in stream:
                data = json.loads(line)
                data['checkpoint'] = Checkpoint(**data['checkpoint'])
                result.append(TaskObservation(**data))
        return tuple(result)

    def append(self, observation):
        if not isinstance(observation, TaskObservation):
            raise TypeError('TaskObservation required')
        key = (observation.task_ref, observation.checkpoint, observation.round_index)
        for old in self.read():
            if (old.task_ref, old.checkpoint, old.round_index) == key:
                if old != observation:
                    raise ValueError('conflicting task observation for the same round and checkpoint')
                return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(asdict(observation), sort_keys=True, separators=(',', ':')) + '\n')
            stream.flush()

    def record_request(self, package, request, round_index):
        # Reuse the package's source/group/mix checks before persisting outcomes.
        package.record_service_request(request)
        grouped = {}
        for row in request.rollouts:
            if row.reward not in (0, 1):
                raise ValueError('binary Service outcomes required')
            grouped.setdefault(row.task_id, []).append(row)
        for task_ref, rows in grouped.items():
            self.append(TaskObservation(task_ref, request.policy_before, round_index,
                sum(int(row.reward) for row in rows), len(rows)))

    def select(self, task_refs, count, *, round_index, seed, low=0.2, high=0.8,
               recalibrate_after=10):
        """Prefer the observed band, reserving one place for unknown/stale tasks."""
        refs = tuple(sorted(task_refs))
        if (len(set(refs)) != len(refs) or type(count) is not int or not 0 <= count <= len(refs)
                or type(round_index) is not int or round_index < 0
                or type(seed) is not int or seed < 0 or not 0 <= low <= high <= 1
                or type(recalibrate_after) is not int or recalibrate_after < 1):
            raise ValueError('invalid benchmark selection parameters')
        latest = {}
        for obs in self.read():
            # Exclude observations from this round on resume: selection must be
            # reproducible even after sampling has already been persisted.
            if obs.round_index >= round_index or obs.task_ref not in refs:
                continue
            old = latest.get(obs.task_ref)
            if old is None or obs.round_index > old.round_index:
                latest[obs.task_ref] = obs
        in_band, explore, outside = [], [], []
        for ref in refs:
            obs = latest.get(ref)
            if obs is None or round_index - obs.round_index >= recalibrate_after:
                explore.append(ref)
            elif low <= obs.successes / obs.group_size <= high:
                in_band.append(ref)
            else:
                outside.append(ref)
        rng = random.Random(seed)
        for bucket in (in_band, explore, outside):
            rng.shuffle(bucket)
        chosen = []
        if explore and count:
            chosen.append(explore.pop())
        return tuple((chosen + in_band + explore + outside)[:count])
