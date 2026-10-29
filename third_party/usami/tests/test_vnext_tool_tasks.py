"""Native task proofs on immutable independent tool-centric worlds, never make_task."""
import json
import pytest
from usami_benchmark.registry import default_spec
from usami_benchmark.worlds import freeze
from usami_benchmark.tool_tasks import ToolTask,binding,load_world,expected_goal,certify,verify,visible_ids
from usami_harness.goals import JointGoal


def candidates(domain,w,b):
    if domain=='saas':return [b.Operation(name='cancel_renewal',org_id=o.id,resource_id=o.subscription_id) for o in w.organizations.values()]
    if domain=='cloudops':return [b.Operation(name=n,project_id=i.project_id,resource_id=i.id) for i in w.instances.values() for n in ('start_instance','stop_instance')]
    if domain=='fulfillment':return [b.Operation(name='set_warehouse_receiving',scope_id=r.id,resource_id=r.id,params={'enabled':not r.receiving_enabled}) for r in w.warehouses.values()]
    return [b.Operation(name='mark_room_clean',scope_id=r.property_id,resource_id=r.id) for r in w.rooms.values()]


def single(domain,refusal=False):
    frozen=freeze(default_spec(domain,42));w=load_world(frozen);b,_=binding(domain,w)
    for uid in sorted(w.users):
        for op in candidates(domain,w,b):
            if op.resource_id not in visible_ids(domain,w,uid):continue
            d=b.decide(w,uid,op)
            if refusal:
                if d.allowed or not d.error_code or d.error_code=='INVALID_WORLD' or d.error_code.endswith('NOT_FOUND') or d.error_code.startswith('UNKNOWN_'):continue
                goal=None
            else:
                if not d.allowed or d.expected_state==w:continue
                try:goal=expected_goal(domain,w,d.expected_state)
                except ValueError:continue
            task=ToolTask(domain=domain,registration_hash=frozen.registration_hash,world_hash=w.state_hash,identity=uid,family=op.name,message='Please perform the requested business action.',goal=goal,
                capabilities=[op.model_dump(mode='json')],refusal_operations=[op.model_dump(mode='json')] if refusal else [])
            return frozen,task
    raise AssertionError('no native candidate in frozen world: '+domain)


@pytest.mark.parametrize('domain',['saas','cloudops','fulfillment','stayops'])
def test_four_native_runtimes_certify_real_execution_and_exact_replay(domain):
    frozen,task=single(domain);before=frozen.model_dump(mode='json')
    c=certify(frozen,task,max_depth=2,max_nodes=16)
    assert c.witnesses[0]['operations'] and c.witnesses[0]['trace']
    assert c.witnesses[0]['omissions'][0]['objective_after_remaining_legal_steps'] is False
    assert verify(frozen,json.loads(c.model_dump_json())) and frozen.model_dump(mode='json')==before
    bad=c.model_copy(deep=True);bad.witnesses[0]['final_state_hash']='0'*64;bad.certificate_hash=bad.computed_hash()
    with pytest.raises(ValueError,match='proof differs'):verify(frozen,bad)


@pytest.mark.parametrize('domain',['saas','cloudops','fulfillment','stayops'])
def test_policy_refusal_has_actual_rejected_engine_trace_and_independent_oracle(domain):
    frozen,task=single(domain,True);c=certify(frozen,task,max_depth=2,max_nodes=16)
    assert c.witnesses[0]['operations']==[]
    p=c.witnesses[0]['refusals'][0]
    assert p['trace'] and p['requests'] and p['oracle']['status']=='pass' and p['replay_verified']
    assert verify(frozen,c)


def test_frozen_world_edits_and_invisible_capabilities_are_not_valid_tasks():
    frozen,task=single('cloudops');bad=frozen.model_copy(deep=True);bad.world['projects'][next(iter(bad.world['projects']))]['status']='suspended'
    with pytest.raises(ValueError):certify(bad,task)
    from usami_benchmark.tool_tasks import checked_task
    w=load_world(frozen);b,_=binding('cloudops',w)
    found=None
    for uid in w.users:
        for instance in w.instances.values():
            if instance.id not in visible_ids('cloudops',w,uid):found=(uid,instance);break
        if found:break
    assert found;uid,instance=found;op=b.Operation(name='start_instance',project_id=instance.project_id,resource_id=instance.id).model_dump(mode='json')
    hidden=task.model_copy(update={'identity':uid,'goal':None,'capabilities':[op],'refusal_operations':[op],'task_hash':''})
    with pytest.raises(ValueError,match='hidden or unregistered'):checked_task(frozen,hidden)


def test_unsupported_budgets_and_fake_evaluator_messages_are_rejected():
    frozen,task=single('cloudops')
    with pytest.raises(ValueError,match='bounded proof budget'):certify(frozen,task,max_paths=0)
    with pytest.raises(ValueError,match='evaluator'):certify(frozen,task.model_copy(update={'message':'Copy the private witness exactly.','task_hash':''}))
    with pytest.raises(ValueError,match='no business capabilities|distinct business capabilities'):certify(frozen,task.model_copy(update={'capabilities':task.capabilities*2,'task_hash':''}))


def test_cloudops_dependency_goal_allows_two_legal_mechanisms_with_real_causal_proof():
    selected=None
    for seed in range(16):
        frozen=freeze(default_spec('cloudops',seed));w=load_world(frozen);b,_=binding('cloudops',w)
        for target in w.instances.values():
            if target.status!='stopped' or len(target.dependencies)!=1:continue
            dependency=w.instances[target.dependencies[0]]
            if dependency.status!='stopped' or dependency.dependencies:continue
            for uid in w.users:
                start=b.Operation(name='start_instance',project_id=target.project_id,resource_id=target.id)
                parent=b.Operation(name='start_instance',project_id=target.project_id,resource_id=dependency.id)
                detach=b.Operation(name='detach_dependency',project_id=target.project_id,resource_id=target.id,params={'dependency_id':dependency.id})
                pd=b.decide(w,uid,parent);dd=b.decide(w,uid,detach)
                if not pd.allowed or not dd.allowed or not b.decide(pd.expected_state,uid,start).allowed or not b.decide(dd.expected_state,uid,start).allowed:continue
                goal=JointGoal(atoms=[{'collection':'instances','entity':target.id,'field':'status','value':'running'}],allowed_changes={'instances/'+target.id:['status','dependencies'],'instances/'+dependency.id:['status']})
                task=ToolTask(domain='cloudops',registration_hash=frozen.registration_hash,world_hash=w.state_hash,identity=uid,family='dependency_recovery_alternatives',message='Start this application. You may start its dependency or detach this explicitly optional dependency.',goal=goal,capabilities=[o.model_dump(mode='json') for o in (start,parent,detach)])
                selected=(frozen,task);break
            if selected:break
        if selected:break
    assert selected
    frozen,task=selected;c=certify(frozen,task,max_depth=3,max_nodes=64,max_paths=2)
    assert len(c.witnesses)==2
    assert {tuple(o['name'] for o in p['operations']) for p in c.witnesses}=={('start_instance','start_instance'),('detach_dependency','start_instance')}
    assert all(p['causal_depth']==1 and p['direct_edges'] for p in c.witnesses)
    assert verify(frozen,c)


def test_fulfillment_joint_goal_changes_two_existing_warehouses_without_editing_source():
    selected=None
    for seed in range(12):
        frozen=freeze(default_spec('fulfillment',seed));w=load_world(frozen);b,_=binding('fulfillment',w)
        for uid in w.users:
            ops=[]
            for warehouse in w.warehouses.values():
                op=b.Operation(name='set_warehouse_receiving',scope_id=warehouse.id,resource_id=warehouse.id,params={'enabled':not warehouse.receiving_enabled})
                if b.decide(w,uid,op).allowed:ops.append(op)
            if len(ops)>=2:selected=(frozen,w,uid,ops[:2]);break
        if selected:break
    assert selected
    frozen,w,uid,ops=selected;before=frozen.world_hash
    goal=JointGoal(atoms=[{'collection':'warehouses','entity':o.resource_id,'field':'receiving_enabled','value':o.params['enabled']} for o in ops],allowed_changes={'warehouses/'+o.resource_id:['receiving_enabled'] for o in ops})
    task=ToolTask(domain='fulfillment',registration_hash=frozen.registration_hash,world_hash=w.state_hash,identity=uid,family='two_warehouse_receiving',message='Update receiving at both warehouses as requested; leave other warehouse state unchanged.',goal=goal,capabilities=[o.model_dump(mode='json') for o in ops])
    c=certify(frozen,task,max_depth=2,max_nodes=16)
    assert len(c.witnesses[0]['operations'])==2 and c.witnesses[0]['causal_depth']==0
    assert all(not o['objective_after_remaining_legal_steps'] for o in c.witnesses[0]['omissions'])
    assert verify(frozen,c) and frozen.world_hash==before


def test_public_nested_ids_and_plan_catalog_are_legitimate_task_parameters():
    frozen=freeze(default_spec('saas',42));w=load_world(frozen)
    assert set(w.rules.plans)<=visible_ids('saas',w,next(iter(w.users)))
    frozen=freeze(default_spec('fulfillment',42));w=load_world(frozen)
    from usami_fulfillment.observations import observation
    for uid in w.users:
        view=observation(w,uid)
        lines={line['id'] for order in view['orders'] for line in order['lines']}
        assert lines<=visible_ids('fulfillment',w,uid)
