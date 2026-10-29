"""Independent proofs, frozen authority/budget tamper and task-free Discovery."""
import copy
import pytest
from usami_harness.models import Budget, digest
from usami_harness.goals import AtomicGoal, JointGoal, execute, certify
from usami_fulfillment import adapter,generate_world
from usami_fulfillment.goals import make_task,progress_task
from usami_fulfillment.benchmark import (
    Instance,Manifest,validate,runtime_hashes,run_instance,replay_run,score,summarize,assert_comparable)
from usami_fulfillment.release import TRAIN,HELDOUT
from test_fulfillment_agents import Client,call,reply


def profile(scenario='return_restock'):
    world,task=make_task(42,scenario)
    return Manifest(domain='fulfillment',domain_version=adapter.version,suite_id='offline-test',track='service_task',
        split='train',tools=list(adapter.TOOLS),budget=Budget(max_customer_turns=1,max_customer_llm_calls=0),
        service_model=Client.settings,instances=[Instance(id=task.id,world=world.model_dump(mode='json'),
        initial_hash=world.state_hash,identity=task.identity,task=task,generation_recipe=scenario)],source_hashes=runtime_hashes()).seal()


def discovery():
    world=generate_world(42)
    return Manifest(domain='fulfillment',domain_version=adapter.version,suite_id='open-test',track='customer_discovery',
        split='discovery',mode='sandbox',tools=list(adapter.TOOLS),budget=Budget(),service_model=Client.settings,
        customer_model=Client.settings,customer_method='adaptive',trials=['r0','r1','r2'],instances=[Instance(
            id='world',world=world.model_dump(mode='json'),initial_hash=world.state_hash,identity='u0')],source_hashes=runtime_hashes()).seal()


@pytest.mark.parametrize('scenario',['paid_fulfillment','regional_transfer','packed_cancellation:refund',
    'return_replacement','fulfillment_return','draft_amendment:fulfill','supply_replenishment','quality_release',
    'split_fulfillment','multi_line_fulfillment','partial_cancel_fulfillment:refund'])
def test_business_goals_have_independent_actual_witnesses(scenario):
    world,task=make_task(42,scenario)
    assert task.witnesses
    for witness in task.witnesses:
        path=[adapter.Operation.model_validate(op) for op in witness.operations]
        assert certify(adapter,world,task,path)==witness
        assert all(not op.is_query for op in path)
    if scenario=='fulfillment_return':
        assert max(w.causal_depth for w in task.witnesses)>=8


def test_phase_cases_are_real_distinct_progress_not_label_replication():
    world,task=make_task(42,'paid_fulfillment')
    initial,phase=progress_task(world,task,0,2)
    assert world.orders['order0'].status=='draft' and initial.orders['order0'].status=='confirmed'
    assert phase.initial_state_hash==initial.state_hash
    assert all(len(w.operations)<len(task.witnesses[0].operations) for w in phase.witnesses)
    assert make_task(42,'paid_fulfillment@0@2')==(initial,phase)
    assert world.orders['order0'].status=='draft' # no shared mutable state


def test_ids_names_seed_and_amounts_do_not_create_structures():
    world,task=make_task(42,'paid_fulfillment')
    other,replica=make_task(99,'paid_fulfillment')
    assert world!=other
    assert task.structure_signature==replica.structure_signature
    assert task.semantic_signature==replica.semantic_signature
    assert task.structure_signature!=make_task(42,'fulfillment_return')[1].structure_signature


def test_goal_compatibility_and_field_whitelist_reject_bad_compositions():
    world=generate_world()
    bad=JointGoal(atoms=[AtomicGoal(collection='orders',entity='order0',field='status',value='draft'),
                        AtomicGoal(collection='orders',entity='order0',field='status',value='confirmed')],allowed_changes={})
    with pytest.raises(ValueError):bad.validate_for(adapter,world)
    bad=JointGoal(atoms=[AtomicGoal(collection='orders',entity='order0',field='customer_id',value='u5')],allowed_changes={})
    with pytest.raises(ValueError):bad.validate_for(adapter,world)
    good=make_task(42,'paid_fulfillment')[1].goal
    good.allowed_changes['organizations/o0']=['owner_id']
    with pytest.raises(ValueError):good.validate_for(adapter,world)


def test_causal_omissions_separate_first_failure_from_mediated_failures():
    world,task=make_task(42,'paid_fulfillment')
    witness=task.witnesses[0]
    pay=next(i for i,op in enumerate(witness.operations) if op['name']=='capture_payment')
    edge=next(e for e in witness.direct_edges if e['from']==pay)
    assert witness.operations[edge['to']]['name']=='confirm_order'
    assert edge['error_code']=='PAYMENT_REQUIRED'
    assert any(e['from']==pay for e in witness.mediated_failures)
    assert witness.causal_depth>=4


def test_frame_preservation_protects_unrelated_warehouse_and_bindings():
    world,task=make_task(42,'paid_fulfillment')
    trace=execute(adapter,world,task.identity,task.witnesses[0].operations)
    state=adapter.World.model_validate(trace.final_state)
    assert task.goal.frame_preserved(world,state)
    state.warehouses['w2'].capacity_volume+=1
    assert not task.goal.frame_preserved(world,state)


def test_all_refusal_certificates_rebuild_and_preserve_state():
    for scenario in TRAIN:
        if '_refusal' not in scenario:continue
        world,task=make_task(42,scenario)
        assert task.kind=='policy_refusal' and task.refusal_code and not task.witnesses
        assert not task.goal.allowed_changes and task.search_exhausted


def test_manifest_certificate_and_budget_tamper_are_detected():
    manifest=profile();assert validate(manifest)==manifest
    changed=manifest.model_copy(deep=True);changed.budget.max_service_tool_calls+=1
    with pytest.raises(ValueError):validate(changed)
    changed=manifest.model_copy(deep=True);changed.instances[0].task.witnesses[0].causal_depth=500;changed.seal()
    with pytest.raises(ValueError):validate(changed)
    changed=manifest.model_copy(deep=True);changed.instances[0].world['lots']['lot0']['on_hand']+=1;changed.seal()
    with pytest.raises(ValueError):validate(changed)


def test_service_evidence_replays_exact_frozen_independent_authority():
    manifest=profile();instance=manifest.instances[0]
    responses=[]
    for index,operation in enumerate(instance.task.witnesses[0].operations):
        responses.append(reply(calls=[call(str(index),operation['name'],{k:v for k,v in operation.items() if k!='name'})]))
    responses.append(reply('Returned goods have been restocked.'))
    run=run_instance(manifest,instance,Client(responses))
    assert score(manifest,run)['task_success']
    assert replay_run(manifest,run)==run
    assert summarize(manifest,[run])['success_rate']==1
    tampered=run.model_copy(deep=True)
    tampered.execution.inputs[0]['grants']=[]
    tampered.execution.execution_hash=tampered.execution.computed_hash();tampered.run_hash=tampered.computed_hash()
    with pytest.raises(ValueError):replay_run(manifest,tampered)


def test_partial_summary_cannot_claim_complete_score():
    manifest=profile();manifest.trials=['r0','r1'];manifest.seal()
    run=run_instance(manifest,manifest.instances[0],Client([reply('I did nothing.')]),trial='r0')
    summary=summarize(manifest,[run])
    assert not summary['complete'] and summary['success_rate'] is None
    assert summary['missing']==[[manifest.instances[0].id,'r1']]


def test_discovery_rejects_conditioned_task_world_even_if_resealed():
    manifest=discovery();assert validate(manifest)==manifest
    manifest.instances[0].world=make_task(42,'return_restock')[0].model_dump(mode='json')
    manifest.instances[0].initial_hash=digest(manifest.instances[0].world);manifest.seal()
    with pytest.raises(ValueError):validate(manifest)


def test_open_discovery_protocol_has_no_task_or_recipe_and_compares_victim_pins():
    manifest=discovery();other=manifest.model_copy(deep=True);other.customer_method='careful';other.seal()
    assert_comparable(manifest,other)
    other.budget.max_service_llm_calls+=1;other.seal()
    with pytest.raises(ValueError):assert_comparable(manifest,other)
    for forbidden in ['task','generation_recipe']:
        changed=manifest.model_copy(deep=True)
        if forbidden=='task':changed.instances[0].task=make_task(42,'paid_fulfillment')[1]
        else:changed.instances[0].generation_recipe='paid_fulfillment'
        with pytest.raises(ValueError):Manifest.model_validate(changed.model_dump())


def test_stop_fixture_is_recorded_as_offline_protocol_evidence_only():
    manifest=discovery()
    customer=Client([reply(calls=[call('stop','stop_exploration',{'reason':'Offline protocol test.'})])])
    run=run_instance(manifest,manifest.instances[0],Client([]),customer)
    assert replay_run(manifest,run)==run
    result=score(manifest,run)
    assert result['customer_llm_calls']==1 and result['service_llm_calls']==0 and not result['discovery_success']
