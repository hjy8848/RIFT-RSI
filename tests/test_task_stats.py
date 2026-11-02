from dataclasses import replace

import pytest

from rift.schemas import Checkpoint
from rift.training.task_stats import TaskObservation, TaskStatsStore


def test_selection_explores_and_recalibrates_without_resume_leakage(tmp_path):
    store = TaskStatsStore(tmp_path / 'stats.jsonl')
    cp = Checkpoint('service', '0', 'tokenizer', 'weights')
    store.append(TaskObservation('in-band', cp, 1, 2, 4))
    store.append(TaskObservation('too-easy', cp, 1, 4, 4))
    store.append(TaskObservation('stale', cp, 0, 0, 4))
    refs = ('in-band', 'too-easy', 'unseen', 'stale')
    selected = store.select(refs, 2, round_index=2, seed=3, recalibrate_after=2)
    assert 'in-band' in selected and selected[0] in {'unseen', 'stale'}
    assert selected == store.select(refs, 2, round_index=2, seed=3, recalibrate_after=2)
    store.append(TaskObservation('unseen', cp, 2, 0, 4))
    assert selected == store.select(refs, 2, round_index=2, seed=3, recalibrate_after=2)


def test_observation_append_is_idempotent_and_rejects_conflicts(tmp_path):
    store = TaskStatsStore(tmp_path / 'stats.jsonl')
    row = TaskObservation('task', Checkpoint('service', '0', 't', 'w'), 1, 2, 4)
    store.append(row)
    store.append(row)
    assert store.read() == (row,)
    with pytest.raises(ValueError, match='conflicting'):
        store.append(replace(row, successes=3))
