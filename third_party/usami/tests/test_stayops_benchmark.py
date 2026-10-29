"""Hotel goals, true interval dependencies and frozen protocol integrity."""
import pytest
from usami_harness.models import Budget,digest
from usami_harness.goals import certify,AtomicGoal,JointGoal
from usami_stayops import adapter,generate_world
from usami_stayops.goals import make_task,progress_task
from usami_stayops.release import TRAIN
from usami_stayops.benchmark import Instance,Manifest,validate,runtime_hashes,run_instance,replay_run,score,assert_comparable
from test_fulfillment_agents import Client,call,reply


def profile(scenario='calendar_reopening'):
    world,task=make_task(42,scenario)
    return Manifest(domain='stayops',domain_version=adapter.version,suite_id='offline-hotel',track='service_task',split='train',
        tools=list(adapter.TOOLS),budget=Budget(max_customer_turns=1,max_customer_llm_calls=0),service_model=Client.settings,
        instances=[Instance(id=task.id,world=world.model_dump(mode='json'),initial_hash=world.state_hash,identity=task.identity,
            task=task,generation_recipe=scenario)],source_hashes=runtime_hashes()).seal()


@pytest.mark.parametrize('scenario',['guest_check_in','booking_date_change','booking_upgrade','booking_downgrade',
    'calendar_reopening','calendar_capacity','maintenance_recovery','room_reassignment','complete_stay',
    'rate_date_requote:capacity','arrival_turnover_rebooking','booking_reallocation','expired_hold_reallocation'])
def test_actual_hotel_joint_goals_and_omission_certificates(scenario):
    world,task=make_task(42,scenario)
    assert task.witnesses
    assert len({digest(w.operations) for w in task.witnesses})==len(task.witnesses)
    for witness in task.witnesses:
        path=[adapter.Operation.model_validate(o) for o in witness.operations]
        assert certify(adapter,world,task,path)==witness
    if scenario=='arrival_turnover_rebooking':assert max(w.causal_depth for w in task.witnesses)>=8


def test_room_interval_capacity_is_a_causal_business_dependency():
    world,task=make_task(42,'maintenance_recovery')
    witness=task.witnesses[0]
    unlock=next(i for i,o in enumerate(witness.operations) if o['name']=='set_room_maintenance')
    edge=next(e for e in witness.direct_edges if e['from']==unlock)
    assert witness.operations[edge['to']]['name']=='create_hold' and edge['error_code']=='ROOM_TYPE_UNAVAILABLE'
    assert witness.causal_depth>=4


def test_partial_progress_is_real_date_inventory_not_renamed_seed():
    world,task=make_task(42,'guest_check_in')
    initial,remaining=progress_task(world,task,0,2)
    assert initial!=world and remaining.initial_state_hash==initial.state_hash
    assert make_task(42,'guest_check_in@0@2')==(initial,remaining)
    _,copy=make_task(99,'guest_check_in');assert copy.structure_signature==task.structure_signature
    assert copy.semantic_signature==task.semantic_signature


def test_all_hotel_refusal_cases_rebuild_independently():
    for name in TRAIN:
        if '_refusal' not in name:continue
        world,task=make_task(42,name)
        assert task.kind=='policy_refusal' and not task.witnesses and not task.goal.allowed_changes


def test_goal_frame_cannot_mutate_guest_or_property_identity():
    world,task=make_task(42,'guest_check_in')
    task.goal.allowed_changes['reservations/b0']=['guest_id']
    with pytest.raises(ValueError):task.goal.validate_for(adapter,world)
    bad=JointGoal(atoms=[AtomicGoal(collection='rooms',entity='p0_standard_0',field='current_stay_id',value='stay_fabricated')],
                  allowed_changes={'rooms/p0_standard_0':['*']})
    with pytest.raises(ValueError):bad.validate_for(adapter,world)


def test_hotel_frozen_model_world_authority_and_budget_replay():
    manifest=profile();instance=manifest.instances[0];responses=[]
    for index,operation in enumerate(instance.task.witnesses[0].operations):
        responses.append(reply(calls=[call(str(index),operation['name'],{k:v for k,v in operation.items() if k!='name'})]))
    responses.append(reply('Confirmed after reopening both nights.'))
    run=run_instance(manifest,instance,Client(responses))
    assert score(manifest,run)['task_success'] and replay_run(manifest,run)==run
    tampered=manifest.model_copy(deep=True);tampered.budget.max_service_tool_calls+=1;tampered.seal()
    with pytest.raises(ValueError):replay_run(tampered,run)
    changed=manifest.model_copy(deep=True);changed.instances[0].task.witnesses[0].causal_depth=99;changed.seal()
    with pytest.raises(ValueError):validate(changed)


def test_discovery_is_unconditioned_and_customer_comparison_pins_victim():
    world=generate_world()
    manifest=Manifest(domain='stayops',domain_version=adapter.version,suite_id='offline-open-hotel',track='customer_discovery',split='discovery',
        mode='sandbox',tools=list(adapter.TOOLS),budget=Budget(),service_model=Client.settings,customer_model=Client.settings,
        customer_method='adaptive',trials=['r0','r1','r2'],instances=[Instance(id='world',world=world.model_dump(mode='json'),
        initial_hash=world.state_hash,identity='u0')],source_hashes=runtime_hashes()).seal()
    assert validate(manifest)==manifest
    other=manifest.model_copy(deep=True);other.customer_method='careful';other.seal();assert_comparable(manifest,other)
    other.service_model.options['temperature']=1;other.seal()
    with pytest.raises(ValueError):assert_comparable(manifest,other)
    manifest.instances[0].world=make_task(42,'guest_check_in')[0].model_dump(mode='json')
    # This base world is intentionally unconditioned: use a genuinely progressed
    # state to demonstrate that arbitrary discovery tasks cannot condition it.
    initial,task=make_task(42,'maintenance_recovery');manifest.instances[0].world=initial.model_dump(mode='json')
    manifest.instances[0].initial_hash=initial.state_hash;manifest.seal()
    with pytest.raises(ValueError):validate(manifest)
