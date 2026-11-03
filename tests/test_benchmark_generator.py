import pytest

from rift.benchmark.generator import generate_composite, generate_instance
from usami_v2.scenarios.builder import verify_build
from usami_v2.scenarios.tasks import check_task_scope


@pytest.mark.parametrize('family', [f'F{i}' for i in range(1, 12)])
def test_generated_families_are_reproducible_and_native_verified(family):
    first = generate_instance(family, 3)
    second = generate_instance(family, 3)
    assert first.spec == second.spec and first.build == second.build and first.task == second.task
    assert verify_build(first.spec, first.build)['verified']
    assert check_task_scope(first.build.world, first.task) == first.task
    assert first.task.request_bundle.renderer_version == 'commerce-request-renderer-v2'


def test_exchange_generator_produces_catalog_grid():
    result = generate_instance('F6', 4)
    line = result.build.world.lines[result.task.request_bundle.requests[0].line_id]
    source = result.build.world.skus[line.sku_id]
    options = [sku for sku in result.build.world.skus.values()
               if sku.product_family == source.product_family and sku.id != source.id]
    assert 3 <= len(options) <= 8
    assert all('color' in sku.attributes and 'storage' in sku.attributes for sku in options)
    assert len({(sku.attributes['color'], sku.attributes['storage']) for sku in options}) > 1


def test_near_distractors_include_same_sku_and_engine_recorded_history():
    result = generate_instance('F1', 3, near_distractors=3)
    intent = result.task.request_bundle.requests[0]
    target = result.build.world.lines[intent.line_id]
    same_sku = [line for line in result.build.world.lines.values()
                if line.sku_id == target.sku_id and line.id != target.id]
    assert len(same_sku) >= 2
    assert intent.reference == 'delivery_age'
    assert result.spec.history and result.build.world.settlements
    assert verify_build(result.spec, result.build)['verified']


@pytest.mark.parametrize('coupling', ['none', 'same_line', 'shared_inventory'])
def test_composite_generation_includes_requested_coupling(coupling):
    result = generate_composite(3, count=3, coupling=coupling, with_history=True)
    assert len(result.task.request_bundle.requests) == 3
    assert result.spec.history and verify_build(result.spec, result.build)['verified']
    if coupling == 'same_line':
        assert result.task.protocol == 'commerce-task-v4'
    if coupling == 'shared_inventory':
        assert result.task.request_bundle.sequential
        assert result.task.refusal_goals[-1].reason_code == 'INSUFFICIENT_STOCK'
        from rift.benchmark.usami import default_customer_scenario
        from usami_v2.scenarios.refusal_certification import certify_refusal
        scenario = default_customer_scenario(result.task, result.spec, result.build)
        assert certify_refusal(result.build.world, result.task, scenario).full_task_reachability_certified


@pytest.mark.parametrize('family', ['F6', 'F7'])
@pytest.mark.parametrize('mode,expected', [('none', 0), ('one', 1), ('multiple', 2)])
def test_controlled_exchange_grid_has_the_requested_feasible_count(family, mode, expected):
    from rift.probes.fingerprint import decision_fingerprint
    result = generate_instance(family, 4, grid_mode=mode)
    signature = decision_fingerprint(result.spec, result.build, result.task)
    assert signature.goals[0].feasible_candidates == expected
    if mode == 'none':
        assert result.task.refusal_goals[0].reason_code == 'INSUFFICIENT_STOCK'
