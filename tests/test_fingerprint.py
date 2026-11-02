from dataclasses import replace
from math import inf

from rift.probes.fingerprint import TaskFingerprint, fingerprint_distance, task_fingerprint


def test_fingerprint_distance_only_compares_same_discrete_attributes():
    baseline = TaskFingerprint((('refund', 'device'),), ((1, 2, 3),))
    same = TaskFingerprint((('refund', 'device'),), ((1, 2, 3),))
    adjacent = replace(same, numeric=((2, 2, 3),))
    different = replace(same, discrete=(('exchange', 'device'),))
    assert fingerprint_distance(baseline, same) == 0
    assert fingerprint_distance(baseline, adjacent) == 1
    assert fingerprint_distance(baseline, different) == inf


def test_fingerprint_of_real_compiled_task_ignores_task_id():
    # The native fixture provides a certified task and its actual frozen World.
    from test_training_package import _native_fixture, _package
    native_fixture = _native_fixture()
    package = _package(native_fixture)
    registry = native_fixture[3]
    record = registry.resolve_record(package.designer_task_refs[0])
    task = record.task
    changed = task.model_copy(update={'task_id': 'only-the-id-changed'})
    assert task_fingerprint(record.world, task) == task_fingerprint(record.world, changed)
