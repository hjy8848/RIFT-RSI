import pytest

from usami_v2.commerce.invariants import invariants
from usami_v2.scenarios.retail_scene import (RetailScene, SceneChange, SceneError, SceneItem,
    SceneOrder, SceneRequest, SceneTask, SceneVariant, compile_scene, scene_task, validate_scene)
from usami_v2.scenarios.retail_tasks import certify_retail_task, generate_retail_task
from usami_v2.scenarios.retail_signature import retail_signature
from rift.designer.retail_format import parse_retail_plan, plan_from_scene, retail_plan_scene
from rift.designer.retail_prompt import example_retail_plans, retail_copy_score
from rift.designer.retail_stage0 import certify_retail_plan
from rift.runtime.curriculum_run import WorldKit
from rift.designer import stage0
from rift.designer.failures import failure_cases
from rift.schemas import DataPurpose, Identity, Split
from rift.verl.retail_registry import CertifiedRetailTask


def sample():
    return RetailScene(gift_card_balance_cents=100, orders=(SceneOrder(ref="A", status="unshipped",
        pays_with="credit_card", items=(SceneItem(ref="A1", product="lamp", price_cents=500),)),),
        variants=(SceneVariant(ref="X", of="A1", price_cents=900, in_stock=True),))


def test_scene_compiles_deterministically_and_certifies():
    scene = sample()
    task = SceneTask(requests=(SceneRequest(type="change_items", order="A",
        changes=(SceneChange(item="A1", want=("X",)),), pay_with=("gift_card", "paypal")),))
    validate_scene(scene, (task,))
    world, ids = compile_scene(scene, 42)
    world2, ids2 = compile_scene(scene, 42)
    assert world == world2 and ids == ids2 and not invariants(world)
    compiled = scene_task(world, ids, task, task_id="test", scene=scene)
    assert certify_retail_task(world, compiled)["changes_state"]


def test_scene_reports_reference_errors():
    scene = sample()
    bad = SceneTask(requests=(SceneRequest(type="change_items", order="A",
        changes=(SceneChange(item="A1", want=("missing",)),)),))
    with pytest.raises(SceneError, match="UNKNOWN_REF"):
        validate_scene(scene, (bad,))


@pytest.mark.parametrize('case, expected', [
    ('duplicate', 'DUPLICATE_REF'),
    ('unknown', 'UNKNOWN_REF'),
    ('outside', 'ITEM_NOT_IN_ORDER'),
    ('variant', 'VARIANT_OF_OTHER_ITEM'),
    ('address', 'ADDRESS_SOURCE_INVALID'),
    ('empty', 'EMPTY_REQUEST'),
    ('limit', 'LIMIT'),
])
def test_scene_validation_uses_stable_error_codes(case, expected):
    base = RetailScene(gift_card_balance_cents=100, orders=(
        SceneOrder(ref='A', status='unshipped', pays_with='credit_card', items=(
            SceneItem(ref='A1', product='lamp', price_cents=500),)),
        SceneOrder(ref='B', status='delivered', pays_with='paypal', items=(
            SceneItem(ref='B1', product='kettle', price_cents=700),))),
        variants=(SceneVariant(ref='X', of='A1', price_cents=600, in_stock=True),
                  SceneVariant(ref='Y', of='B1', price_cents=800, in_stock=True)))
    req = SceneRequest(type='change_items', order='A',
        changes=(SceneChange(item='A1', want=('X',)),))
    scene = base
    if case == 'duplicate':
        scene = base.model_copy(update={'variants': (SceneVariant(ref='A1', of='A1', price_cents=600, in_stock=True),)})
        task = SceneTask(requests=(req,))
    elif case == 'unknown':
        task = SceneTask(requests=(SceneRequest(type='set_order_address', order='missing', to='new'),))
    elif case == 'outside':
        task = SceneTask(requests=(SceneRequest(type='change_items', order='A',
            changes=(SceneChange(item='B1', want=('Y',)),)),))
    elif case == 'variant':
        task = SceneTask(requests=(SceneRequest(type='change_items', order='A',
            changes=(SceneChange(item='A1', want=('Y',)),)),))
    elif case == 'address':
        task = SceneTask(requests=(SceneRequest(type='set_order_address', order='A', to='nowhere'),))
    elif case == 'empty':
        task = SceneTask(requests=(SceneRequest(type='change_items', order='A'),))
    else:
        scene = base.model_copy(update={'orders': base.orders + (SceneOrder(ref='C', status='delivered',
            pays_with='credit_card', items=(SceneItem(ref='C1', product='chair', price_cents=500),)),
            SceneOrder(ref='D', status='delivered', pays_with='credit_card',
            items=(SceneItem(ref='D1', product='chair', price_cents=500),)),
            SceneOrder(ref='E', status='delivered', pays_with='credit_card',
            items=(SceneItem(ref='E1', product='chair', price_cents=500),)))})
        task = SceneTask(requests=(req,))
    with pytest.raises(SceneError) as error:
        validate_scene(scene, (task,))
    assert error.value.code == expected


def test_scene_references_cannot_overwrite_internal_bindings():
    from usami_v2.scenarios.retail_scene import SceneIds
    scene = RetailScene(gift_card_balance_cents=10, orders=(SceneOrder(ref='me', status='unshipped',
        pays_with='credit_card', items=(SceneItem(ref='new_address', product='lamp', price_cents=500),)),),
        variants=(SceneVariant(ref='credit_card', of='new_address', price_cents=600, in_stock=True),))
    world, ids = compile_scene(scene, 17)
    assert isinstance(ids, SceneIds) and ids.customer_id in world.customers
    assert ids['me'] in world.orders and ids['new_address'] in world.lines
    assert ids.new_address != ids['new_address'] and ids.item_skus['new_address'] in world.skus


def test_plan_scene_round_trip():
    scene = sample()
    tasks = [SceneTask(requests=(SceneRequest(type="change_items", order="A",
        changes=(SceneChange(item="A1", want=("X",)),)),))]
    plan = plan_from_scene(scene, tasks, "A stock preference")
    assert retail_plan_scene(parse_retail_plan(plan.model_dump_json())) == (scene, tasks)


def test_signature_tracks_decisions_not_random_names():
    a, ta, _ = generate_retail_task(1, "items_stock_fallback")
    b, tb, _ = generate_retail_task(2, "items_stock_fallback")
    c, tc, _ = generate_retail_task(1, "items_simple")
    assert retail_signature(a, ta) == retail_signature(b, tb)
    assert retail_signature(a, ta) != retail_signature(c, tc)


def test_certification_rejects_noop_task_and_codes_engine_mismatch(monkeypatch):
    from types import SimpleNamespace
    import usami_v2.scenarios.retail_tasks as retail_tasks
    scene = RetailScene(gift_card_balance_cents=100, orders=(SceneOrder(ref='A', status='delivered',
        pays_with='credit_card', items=(SceneItem(ref='A1', product='lamp', price_cents=500),)),))
    description = SceneTask(requests=(SceneRequest(type='set_order_address', order='A', to='new'),))
    world, ids = compile_scene(scene, 12)
    task = scene_task(world, ids, description, task_id='noop', scene=scene)
    with pytest.raises(ValueError, match='^TASK_NO_STATE_CHANGE:'):
        certify_retail_task(world, task)

    scene = sample()
    description = SceneTask(requests=(SceneRequest(type='change_items', order='A',
        changes=(SceneChange(item='A1', want=('X',)),)),))
    world, ids = compile_scene(scene, 12)
    task = scene_task(world, ids, description, task_id='mismatch', scene=scene)
    monkeypatch.setattr(retail_tasks, 'apply', lambda state, *args: (state, SimpleNamespace(success=False)))
    with pytest.raises(ValueError, match='^ENGINE_REFERENCE_MISMATCH:'):
        certify_retail_task(world, task)


def test_generated_examples_compile_and_certify():
    for plan in example_retail_plans(11, 13):
        scene, tasks = retail_plan_scene(plan)
        world, ids = compile_scene(scene, 11)
        task = scene_task(world, ids, tasks[0], task_id="test", scene=scene)
        assert certify_retail_task(world, task)["certified"]
        assert retail_copy_score(plan.model_dump(mode="json"), [plan]) == 1.0


def test_designer_certification_and_registry_roundtrip(tmp_path):
    kit = WorldKit(tmp_path)
    plan = example_retail_plans(4, 1)[0]
    row = certify_retail_plan(kit, plan.model_dump(mode="json"), "p1", 100, signatures=True)
    assert row["status"] == "certified" and len(row["signatures"]) == 1
    ref = row["task_refs"][0]
    world, task, profile, events = kit.task_resolver.resolve(ref)
    assert profile is None and events == ()
    assert kit.task_resolver.verify_task(ref, world, task, None)
    kit.save()
    reloaded = WorldKit(tmp_path)
    assert reloaded.task_resolver.resolve(ref) == (world, task, None, ())
    target = reloaded.retail_tasks_path
    data = target.read_text(encoding="utf-8").replace('"certified":true', '"certified":false')
    target.write_text(data, encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        WorldKit(tmp_path)


def test_retail_specs_generate_cache_and_reload(tmp_path):
    kit = WorldKit(tmp_path)
    spec = [{"retail_family": "items_simple", "instances": 2, "seed": 25}]
    first = kit.certify_specs(spec)
    assert len(first["retail:items_simple"]) == 2
    kit.save()
    loaded = WorldKit(tmp_path)
    assert loaded.certify_specs(spec) == first
    assert all(loaded.task_resolver.resolve(ref)[1].family == "items_simple"
               for ref in first["retail:items_simple"])


def test_stage0_retail_plans_can_be_reloaded_for_training(tmp_path):
    kit = WorldKit(tmp_path)
    texts = stage0.dry_run_generate([None, None], seed=77, plan_format='retail')
    report = stage0.run(kit, lambda prompts: texts, [], samples=2, seed=77, out_dir=tmp_path,
                        plan_format='retail')
    assert report['certified_tasks'] == 2
    loaded = stage0.certify_saved_plans(kit, tmp_path / 'designer-plans.jsonl')
    assert len(loaded) == 2 and set(loaded) == set(kit.retail_registry.task_refs)


def test_retail_failure_cases_include_derived_hidden_answers(tmp_path):
    scene = sample()
    description = SceneTask(requests=(SceneRequest(type='change_items', order='A',
        changes=(SceneChange(item='A1', want=('X',)),)),))
    world, ids = compile_scene(scene, 7)
    task = scene_task(world, ids, description, task_id='failure-task', scene=scene)
    certificate = certify_retail_task(world, task)
    identity = Identity(world.state_hash, task.record_hash, 'retail:gen:items_simple',
                        '{"family":"items_simple","seed":7}', Split.TRAIN, DataPurpose.TRAIN)
    kit = WorldKit(tmp_path)
    kit.retail_registry.register(CertifiedRetailTask(world, task, certificate, identity))
    from usami_v2.runtime.retail_runner import run_retail_episode
    class Idle:
        uses_model = False
        def decide(self, prompt):
            from usami_v2.runtime.agents import ServiceAction
            return ServiceAction(kind='reply', message='Done.')
    episode = run_retail_episode(world, task, Idle()).record
    class Store:
        def resolve_episode(self, ref):
            assert ref == 'episode-1'
            return episode
    rows = failure_cases([{'task_ref': task.record_hash, 'outcome': 'failure', 'episode_id': 'episode-1'}],
                         kit.task_resolver, Store())
    assert rows and rows[0]['goals'][0]['action'] == 'change items'
    assert rows[0]['goals'][0]['answer'].startswith('change to')
