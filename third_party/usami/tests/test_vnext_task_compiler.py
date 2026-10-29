"""Common compiler keeps world bytes frozen and never trusts authored answers."""
import json
import pytest
from usami_benchmark.registry import default_spec
from usami_benchmark.worlds import freeze
from usami_benchmark.tasks import TaskSelection,ConstructionBudget,compile_task,verify_task
from test_vnext_tool_tasks import single


@pytest.mark.parametrize('domain',['saas','cloudops','fulfillment','stayops'])
def test_tool_domain_compiler_uses_native_certificates_on_exact_world(domain):
    frozen,t=single(domain);selection=TaskSelection(kind='tool',family=t.family,identity=t.identity,message=t.message,goal=t.goal,capabilities=t.capabilities)
    record=compile_task(frozen,selection,ConstructionBudget(max_nodes=32,max_depth=3))
    assert record.track=='A' and record.tool_certificate and record.frozen==frozen
    assert verify_task(json.loads(record.model_dump_json()))


@pytest.mark.parametrize('domain,seed,pid,family,params',[
 ('airline',0,'profile_u0_booking1','refund',{'mixed':True}),
 ('property',1,'profile_p0_building0_unit1_asset1','repair',{'slot_ids':['p0_tech0_slot2'],'mixed':True}),
 ('education',0,'profile_u1_enrollment1','refund',{'mixed':True}),
])
def test_dialogue_compiler_reuses_native_goal_and_process_proofs(domain,seed,pid,family,params):
    frozen=freeze(default_spec(domain,seed));selection=TaskSelection(kind='dialogue',profile_id=pid,family=family,parameters=params)
    record=compile_task(frozen,selection,ConstructionBudget(max_nodes=256))
    assert record.track=='B' and record.dialogue_witnesses[0].certificate.evaluation['success']
    assert record.frozen==frozen and verify_task(json.loads(record.model_dump_json()))


def test_aftersales_compiler_keeps_two_real_outcomes_as_alternatives():
    from test_vnext_aftersales import prepared
    s,order,line=prepared();frozen=freeze(default_spec('aftersales',s.initial.seed))
    record=compile_task(frozen,TaskSelection(kind='dialogue',profile_id=s.profile_id,family='dispute_resolution'),ConstructionBudget(max_nodes=256,max_paths=2))
    assert len(record.dialogue_witnesses)==2
    assert len({r.trace.final_world['lines'][line.id]['status'] for r in record.dialogue_witnesses})==2


def test_refusal_reason_is_discovered_by_reference_not_supplied_by_designer():
    frozen=freeze(default_spec('education',0))
    record=compile_task(frozen,TaskSelection(kind='dialogue',profile_id='profile_u1_enrollment0',family='refuse_refund'),ConstructionBudget(max_nodes=64))
    assert record.task.refusals[0].code=='REFUND_WINDOW_EXPIRED'
    assert record.dialogue_witnesses[0].certificate.evaluation['refusal_subgoals']==[True]
    assert not record.dialogue_witnesses[0].trace.final_world['quotes']
    assert verify_task(record)
    with pytest.raises(ValueError,match='unregistered native task parameters'):
        compile_task(frozen,record.selection.model_copy(update={'parameters':{'refusal_code':'FAKE_EXPECTED_ANSWER'}}))


def test_no_world_edit_or_hidden_certificate_parameter_is_in_selection_schema():
    frozen=freeze(default_spec('education',0))
    raw={'kind':'dialogue','profile_id':'profile_u1_enrollment0','family':'refund','world':frozen.world}
    with pytest.raises(ValueError):TaskSelection.model_validate(raw)
    with pytest.raises(ValueError,match='kind differs'):compile_task(frozen,TaskSelection(kind='tool',family='fake',identity='u1',message='x'))
    selection=TaskSelection(kind='dialogue',profile_id='profile_u1_enrollment0',family='refund',parameters={'world':frozen.world})
    with pytest.raises(ValueError,match='unregistered'):compile_task(frozen,selection)
