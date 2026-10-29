"""Task grouping needs real native proofs; labels cannot manufacture coverage."""
import copy
import json

import pytest

from usami_harness.models import digest
from usami_benchmark.registry import default_spec
from usami_benchmark.worlds import freeze
from usami_benchmark.tasks import TaskSelection, ConstructionBudget, CertifiedTask, compile_task
from usami_benchmark.task_audit import (
    describe_task, anonymous_task_signature, audit_tasks, verify_audit, audit_archive,
)
from test_vnext_tool_tasks import single


def tool_record(domain='cloudops', refusal=False):
    frozen, task = single(domain, refusal)
    return compile_task(frozen, TaskSelection(
        kind='tool', family=task.family, identity=task.identity, message=task.message,
        goal=task.goal, capabilities=task.capabilities, refusal_operations=task.refusal_operations,
    ), ConstructionBudget(max_nodes=32, max_depth=3))


@pytest.fixture(scope='module')
def records():
    rows = [tool_record(d) for d in ('saas', 'cloudops', 'fulfillment', 'stayops')]
    from test_vnext_aftersales import prepared
    session, _, _ = prepared()
    for domain, seed, profile, family, params in [
        ('aftersales', session.initial.seed, session.profile_id, 'settlement_with_duplicate_refusal', {'mixed': True}),
        ('airline', 0, 'profile_u0_booking1', 'refund', {'mixed': True}),
        ('property', 1, 'profile_p0_building0_unit1_asset1', 'repair', {'slot_ids': ['p0_tech0_slot2'], 'mixed': True}),
        ('education', 0, 'profile_u1_enrollment1', 'refund', {'mixed': True}),
    ]:
        rows.append(compile_task(freeze(default_spec(domain, seed)), TaskSelection(
            kind='dialogue', profile_id=profile, family=family, parameters=params,
        ), ConstructionBudget(max_nodes=256)))
    return rows


@pytest.fixture(scope='module')
def report(records):
    return audit_tasks(records, per_domain_target=40)


def test_all_eight_domains_have_native_evidence_but_no_invented_release_counts(report):
    assert report['task_count'] == 8 and report['model_experiments'] == 'NOT RUN'
    assert all(item['tasks'] == 1 and item['shortfall'] == 39 for item in report['domains'].values())
    assert set(report['domain_weight'].values()) == {1 / 8}
    assert report['split_status'] == 'NOT_TESTED_SINGLE_PARTITION'
    assert report['world_and_task_structure_isolation'] is None
    for domain, row in report['domains'].items():
        assert row['matrix'] and row['capabilities']['identity_authorization'] == 1
        assert row['capabilities']['side_effect_consistency'] == 1
        if domain in ('aftersales', 'airline', 'property', 'education'):
            assert row['capabilities']['information_resolution'] == 1
            assert row['capabilities']['explicit_consent'] == 1
            assert row['outcomes'] == {'mixed': 1}
            assert row['difficulty'] == {'unclassified': 1}


def test_roundtrip_and_resealed_false_coverage_are_independently_checked(records, report):
    loaded = json.loads(json.dumps(report, sort_keys=True))
    assert verify_audit(records, loaded)
    bad = copy.deepcopy(loaded)
    bad['domains']['cloudops']['tasks'] = 40
    bad['report_hash'] = digest({k: v for k, v in bad.items() if k != 'report_hash'})
    with pytest.raises(ValueError, match='does not reproduce'):
        verify_audit(records, bad)


def test_labels_and_natural_language_paraphrases_do_not_add_task_groups():
    first = tool_record()
    changed = compile_task(first.frozen, first.selection.model_copy(update={
        'family': 'invented_new_family', 'message': 'Please carry out that same requested change.',
    }), first.construction_budget)
    assert first.task_hash != changed.task_hash
    assert anonymous_task_signature(first) == anonymous_task_signature(changed)
    report = audit_tasks([first, changed], per_domain_target=40)
    assert report['domains']['cloudops']['task_structure_groups'] == 1
    assert report['domains']['cloudops']['decision_groups'] == 1
    assert report['domains']['cloudops']['duplicate_group_fraction'] == .5
    assert report['domains']['cloudops']['shortfall'] == 38
    assert report['domains']['cloudops']['structural_group_shortfall'] == 39


def test_id_alpha_renaming_and_display_names_do_not_change_rooted_grouping():
    record = tool_record()
    raw = record.model_dump(mode='json')
    names = {key: 'renamed_' + key for collection in raw['frozen']['world'].values()
             if isinstance(collection, dict) for key, row in collection.items()
             if isinstance(row, dict) and row.get('id') == key}

    def renamed(value):
        if isinstance(value, dict):
            return {renamed(k): ('changed display name' if k == 'name' and isinstance(v, str)
                                 and k not in {'operation'} and v not in {o['name'] for o in record.task.capabilities}
                                 else renamed(v)) for k, v in value.items()}
        if isinstance(value, list):
            return [renamed(v) for v in value]
        if isinstance(value, str):
            if '/' in value:
                left, right = value.split('/', 1)
                return left + '/' + names.get(right, right)
            return names.get(value, value)
        return value

    changed = CertifiedTask.model_validate(renamed(raw))
    # This calls the pure grouping primitive only. A fabricated world registration
    # is still forbidden from entering an actual coverage report below.
    assert anonymous_task_signature(record) == anonymous_task_signature(changed)
    with pytest.raises(ValueError):
        describe_task(changed)


def test_refusal_boundaries_are_actual_native_codes_not_designer_labels():
    completion = describe_task(tool_record())
    refusal = describe_task(tool_record(refusal=True))
    assert refusal['outcome'] == 'refusal'
    assert refusal['difficulty'] == 'unclassified'
    assert any(outcome == 'refused' and code != 'OK' for _, outcome, code in refusal['decision_shape']['boundaries'])
    assert completion['decision_signature'] != refusal['decision_signature']
    assert refusal['capabilities']['refusal_decisions']['evidence']


def test_missing_capability_evidence_is_not_na_or_claimed_from_long_path(report):
    for row in report['rows']:
        for key in ('time_deadlines', 'capacity_resources', 'entity_isolation'):
            assert row['capabilities'][key] == {'status': 'NOT_AUDITED', 'evidence': []}
    # Dialogue has real information/consent proofs, but that is not a native
    # business-operation omission proof and is not called causal depth.
    for row in report['rows'][4:]:
        assert row['paths'][0]['causal_depth'] is None


def test_relabelling_the_same_training_world_as_test_is_detected():
    first = tool_record()
    frozen = freeze(first.frozen.spec.model_copy(update={'partition': 'test_demo'}))
    second = compile_task(frozen, first.selection, first.construction_budget)
    report = audit_tasks([first, second])
    overlap = report['split_overlaps']['test_demo / train']
    assert overlap['world_hash'] and overlap['world_structure'] and overlap['task_structure_signature']
    assert overlap['decision_signature'] and overlap['family']
    assert not report['world_and_task_structure_isolation']


def test_resealed_certificate_tampering_never_enters_coverage():
    bad = tool_record().model_copy(deep=True)
    bad.tool_certificate.witnesses[0]['omissions'][0]['objective_after_remaining_legal_steps'] = True
    bad.tool_certificate.certificate_hash = bad.tool_certificate.computed_hash()
    bad.record_hash = bad.computed_hash()
    with pytest.raises(ValueError, match='Replay differs'):
        audit_tasks([bad])


def test_target_and_duplicate_records_are_strict():
    record = tool_record()
    with pytest.raises(ValueError, match='distinct'):
        audit_tasks([record, record])
    with pytest.raises(ValueError, match='positive'):
        audit_tasks([record], per_domain_target=0)


def test_different_proven_mechanisms_and_actual_causal_omissions_are_reported():
    from usami_benchmark.tool_tasks import load_world, binding
    from usami_harness.goals import JointGoal
    selected = None
    for seed in range(16):
        frozen = freeze(default_spec('cloudops', seed))
        world = load_world(frozen); native, _ = binding('cloudops', world)
        for target in world.instances.values():
            if target.status != 'stopped' or len(target.dependencies) != 1:
                continue
            parent = world.instances[target.dependencies[0]]
            if parent.status != 'stopped' or parent.dependencies:
                continue
            for identity in world.users:
                start = native.Operation(name='start_instance', project_id=target.project_id, resource_id=target.id)
                upstream = native.Operation(name='start_instance', project_id=target.project_id, resource_id=parent.id)
                detach = native.Operation(name='detach_dependency', project_id=target.project_id, resource_id=target.id, params={'dependency_id': parent.id})
                a = native.decide(world, identity, upstream); b = native.decide(world, identity, detach)
                if not a.allowed or not b.allowed or not native.decide(a.expected_state, identity, start).allowed or not native.decide(b.expected_state, identity, start).allowed:
                    continue
                goal = JointGoal(atoms=[{'collection': 'instances', 'entity': target.id, 'field': 'status', 'value': 'running'}],
                                 allowed_changes={'instances/' + target.id: ['status', 'dependencies'], 'instances/' + parent.id: ['status']})
                selected = compile_task(frozen, TaskSelection(kind='tool', family='dependency_recovery', identity=identity,
                    message='Start the application; you may start or detach this explicitly optional dependency.', goal=goal,
                    capabilities=[op.model_dump(mode='json') for op in (start, upstream, detach)]),
                    ConstructionBudget(max_depth=3, max_nodes=64, max_paths=2))
                break
            if selected:
                break
        if selected:
            break
    assert selected
    row = describe_task(selected)
    assert row['mechanisms'] == 2 and row['difficulty'] == 'medium'
    assert row['capabilities']['alternative_paths']['evidence']
    assert row['capabilities']['causal_dependencies']['evidence']
    assert all(path['causal_depth'] == 1 and path['all_writes_omission_necessary'] for path in row['paths'])


def test_immutable_cli_audit_retains_source_archive_and_never_reads_credentials(tmp_path, monkeypatch):
    from usami_benchmark.artifacts import save_archive
    from usami_benchmark.__main__ import main
    import usami.service.client as client
    monkeypatch.setattr(client, 'key_from_env', lambda: (_ for _ in ()).throw(AssertionError('offline only')))
    archive = tmp_path / 'archive'
    record = tool_record()
    save_archive(archive, [record])
    before = (archive / 'manifest.json').read_bytes()
    output = tmp_path / 'audit'
    assert main(['audit-tasks', str(archive), '--output', str(output), '--per-domain-target', '40']) == 0
    assert json.loads((output / 'verification.json').read_text())['model_calls'] == 0
    assert verify_audit([record], json.loads((output / 'coverage.json').read_text()))
    assert (archive / 'manifest.json').read_bytes() == before
    with pytest.raises(FileExistsError):
        audit_archive(archive, output)


def test_failed_archive_verification_has_retained_failure_and_registration(tmp_path):
    from usami_benchmark.artifacts import save_archive
    archive = tmp_path / 'archive'
    saved = save_archive(archive, [tool_record()])
    case = archive / saved['cases'][0]['path']
    # Valid gzip/JSON but changed bytes: no need to execute a corrupted task.
    case.write_bytes(case.read_bytes() + b'\0')
    output = tmp_path / 'failed-audit'
    with pytest.raises(ValueError, match='compressed'):
        audit_archive(archive, output)
    assert (output / 'registration.json').exists() and (output / 'source-snapshot.zip').exists()
    assert json.loads((output / 'failure.json').read_text())['retry'] == 'not attempted'
    assert not (output / 'coverage.json').exists()


def test_private_heldout_audit_cannot_publish_task_information_in_repo(tmp_path):
    from pathlib import Path
    from usami_benchmark.artifacts import save_archive
    record = tool_record()
    private = freeze(record.frozen.spec.model_copy(update={'partition': 'heldout'}))
    record = compile_task(private, record.selection, record.construction_budget)
    archive = tmp_path / 'private-archive'
    save_archive(archive, [record])
    forbidden = Path(__file__).resolve().parents[1] / 'benchmarks' / 'forbidden-private-audit-test'
    assert not forbidden.exists()
    with pytest.raises(ValueError, match='Held-out'):
        audit_archive(archive, forbidden)
    assert not forbidden.exists()
