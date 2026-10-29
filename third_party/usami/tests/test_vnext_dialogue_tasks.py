"""Native Reference certificates, objective scoring and actual multi-path replay."""
import pytest
from usami_dialogue.tasks import score,verify_certificate,checked_task
from usami_aftersales.tasks import certified_tasks,execute_witness
from test_vnext_aftersales import prepared


def test_frozen_profile_reachability_mixed_completion_and_real_refusal():
    s,order,line=prepared();initial_hash=s.initial.state_hash
    task,result,certificates=certified_tasks(s.initial,s.profile_id,mixed=True,max_nodes=256,max_paths=2)
    assert result['nodes']<=256 and len(certificates)==2
    assert s.initial.state_hash==initial_hash
    final_statuses=set()
    for row in certificates:
        trace=row['trace'];c=row['certificate']
        assert c.evaluation['success'] and c.evaluation['completion_subgoals']==[True]
        assert c.evaluation['refusal_subgoals']==[True] and c.evaluation['metrics']['correct_refusal']==1
        assert c.dependencies and verify_certificate(s.adapter,s.initial,c,trace)
        import json
        from usami_dialogue.engine import ExecutionTrace
        serialized=ExecutionTrace.model_validate(json.loads(json.dumps(trace.model_dump(mode='json'),sort_keys=True)))
        assert verify_certificate(s.adapter,s.initial,json.loads(json.dumps(c.model_dump(mode='json'),sort_keys=True)),serialized)
        final_statuses.add(trace.final_world['lines'][line.id]['status'])
    assert len(final_statuses)==2  # Goal accepts genuinely distinct business resolutions.


def test_goal_and_frame_not_path_equality_define_success():
    s,order,line=prepared()
    task,result,certificates=certified_tasks(s.initial,s.profile_id,resolutions=['refund'],max_nodes=256,max_paths=1)
    assert len(certificates)==1
    trace=certificates[0]['trace'];assert score(s.adapter,s.initial,task,trace)['success']
    # Adding a legal scoped read changes the path but not business correctness.
    path=list(result['paths'][0])
    path.append({'kind':'business','operation':{'name':'get_order','scope_id':order.merchant_id,'resource_id':order.id,'params':{}}})
    alternate=execute_witness(s.initial,task,path)
    assert alternate.trace_hash!=trace.trace_hash and score(s.adapter,s.initial,task,alternate)['success']
    # A completion trace cannot pass the mixed task without a true refusal decision.
    mixed=task.model_copy(deep=True)
    from usami_aftersales.tasks import compose
    mixed=compose(s.initial,s.profile_id,resolutions=['refund'],mixed=True)
    assert not score(s.adapter,s.initial,mixed,trace)['success']


def test_task_source_frame_and_certificate_tampering_rejected():
    s,order,line=prepared()
    task,result,rows=certified_tasks(s.initial,s.profile_id,resolutions=['refund'],max_nodes=256,max_paths=1)
    bad=task.model_copy(deep=True);bad.frames[0].mutable_fields.append('unit_price_cents');bad.task_hash=''
    with pytest.raises(ValueError,match='frame'):checked_task(s.adapter,s.initial,bad)
    bad=task.model_copy(deep=True);bad.goals[0].entity=next(k for k in s.initial.lines if k!=line.id);bad.task_hash=''
    with pytest.raises(ValueError,match='scope'):checked_task(s.adapter,s.initial,bad)
    c=rows[0]['certificate'].model_copy(deep=True);c.evaluation['success']=False;c.certificate_hash=c.computed_hash()
    with pytest.raises(ValueError,match='reproduce'):verify_certificate(s.adapter,s.initial,c,rows[0]['trace'])


def test_bounded_exhaustion_is_reported_not_fabricated_as_unreachable():
    s,order,line=prepared()
    task,result,rows=certified_tasks(s.initial,s.profile_id,max_nodes=1,max_paths=2)
    assert not rows and not result['exhausted'] and result['nodes']==1
