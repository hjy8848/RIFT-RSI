"""One frozen API dispatches to actual existing engines and real dialogue agents."""
import json
import pytest
from test_vast_usami import ScriptedClient,SPEC
from test_vnext_tool_tasks import single
from test_vnext_tool_agent import tool_call,final
from usami_benchmark.tasks import TaskSelection,ConstructionBudget,compile_task
from usami_benchmark.episodes import EpisodeConfig,run,replay
from usami_benchmark.registry import default_spec
from usami_benchmark.worlds import freeze
from usami_harness.models import Budget


@pytest.mark.parametrize('domain',['saas','cloudops','fulfillment','stayops'])
def test_one_api_runs_tool_tracks_without_exposing_private_proofs(domain):
    frozen,t=single(domain);record=compile_task(frozen,TaskSelection(kind='tool',identity=t.identity,family=t.family,message=t.message,goal=t.goal,capabilities=t.capabilities),ConstructionBudget(max_nodes=32,max_depth=3))
    customer=ScriptedClient(SPEC,[]);service=ScriptedClient(SPEC,[tool_call('op',record.task.capabilities[0],domain),final()])
    config=EpisodeConfig(task=record,service_model=service.settings,customer_model=customer.settings,evidence_kind='scripted_transport_fixture',budget=Budget(max_service_llm_calls=4,max_service_tool_calls=4))
    result=run(config,customer,service)
    assert result.metrics['task_success'] and result.metrics['replay_verification'] and result.metrics['legal_completion']
    assert result.cost['customer_calls']==0 and not customer.requests
    assert replay(json.loads(json.dumps(result.model_dump(mode='json'),sort_keys=True)))['verified']


def test_unified_dialogue_runs_fixed_customer_with_public_rules_and_exact_model_replay():
    from test_vnext_education_agents import fixture_execution
    w,native,old=fixture_execution();frozen=freeze(default_spec('education',w.seed))
    record=compile_task(frozen,TaskSelection(kind='dialogue',profile_id=native.profile_id,family='refund'),ConstructionBudget(max_nodes=128))
    customer=ScriptedClient(SPEC,[c.record.response for c in old.calls if c.role=='customer'])
    service=ScriptedClient(SPEC,[c.record.response for c in old.calls if c.role=='service'])
    config=EpisodeConfig(task=record,service_model=service.settings,customer_model=customer.settings,evidence_kind='scripted_transport_fixture',budget=native.budget)
    result=run(config,customer,service)
    assert result.metrics['task_success'] and result.metrics['verified_dialogue_dependency']
    assert result.metrics['required_information_resolution'] and result.metrics['consent_compliance']
    assert result.cost['customer_calls']==5 and result.cost['service_calls']==7
    assert result.cost['customer_input_tokens'] is None and result.cost['service_input_tokens'] is None
    assert replay(json.loads(json.dumps(result.model_dump(mode='json'),sort_keys=True)))['verified']
    bad=result.model_copy(deep=True);bad.metrics['task_success']=False;bad.execution_hash=bad.computed_hash()
    with pytest.raises(ValueError,match='score/cost'):replay(bad)


def test_model_or_interface_changes_are_not_silently_replayed():
    frozen,t=single('cloudops');record=compile_task(frozen,TaskSelection(kind='tool',identity=t.identity,family=t.family,message=t.message,goal=t.goal,capabilities=t.capabilities))
    customer=ScriptedClient(SPEC,[]);service=ScriptedClient(SPEC,[final()])
    c=EpisodeConfig(task=record,service_model=service.settings,customer_model=customer.settings,evidence_kind='scripted_transport_fixture')
    result=run(c,customer,service);bad=result.model_copy(deep=True);bad.interface_hash='0'*64;bad.execution_hash=bad.computed_hash()
    with pytest.raises(ValueError,match='interface'):replay(bad)


def scripted_dialogue_clients(record):
    # Private witness -> explicit offline transport fixture. This is protocol QA,
    # never represented as autonomous real-model behavior or research evidence.
    from test_vast_usami import answer,call
    events=record.dialogue_witnesses[0].trace.events;c=[];s=[]
    turns=[i for i,e in enumerate(events) if e.kind=='customer' and not e.payload.get('setup')]
    for position,start in enumerate(turns):
        event=events[start];c.append(answer(calls=[call('c'+str(position),'submit_customer_turn',event.payload['intent'])]))
        end=turns[position+1] if position+1<len(turns) else len(events);segment=events[start+1:end];replied=False
        for e in segment:
            if e.kind=='business':
                op=next(x['payload']['operation'] for x in e.payload['trace']['events'] if x['kind']=='operation')
                s.append(tool_call('s'+str(len(s)),op,record.frozen.spec.domain));replied=False
            elif e.kind=='service':
                p=e.payload;s.append(answer(calls=[call('s'+str(len(s)),'respond_to_customer',{k:p[k] for k in ('message','offered_operation','refused_operation','refusal_code')})]));replied=True
        if not replied:s.append(answer(calls=[call('s'+str(len(s)),'respond_to_customer',{'message':'Please tell me your next actual business need.','offered_operation':None})]))
    c.append(answer(calls=[call('finish','submit_customer_turn',{'message':'My request is resolved.','finish':True})]))
    return ScriptedClient(SPEC,c),ScriptedClient(SPEC,s)


@pytest.mark.parametrize('domain,seed,pid,family,params',[
 ('aftersales',None,None,'dispute_resolution',{'resolutions':['refund']}),
 ('airline',0,'profile_u0_booking1','refund',{'mixed':True}),
 ('property',1,'profile_p0_building0_unit1_asset1','repair',{'slot_ids':['p0_tech0_slot2'],'mixed':True}),
])
def test_unified_runner_covers_each_other_native_dialogue_domain(domain,seed,pid,family,params):
    if seed is None:
        from test_vnext_aftersales import prepared
        source,_,_=prepared();seed=source.initial.seed;pid=source.profile_id
    frozen=freeze(default_spec(domain,seed));record=compile_task(frozen,TaskSelection(kind='dialogue',profile_id=pid,family=family,parameters=params),ConstructionBudget(max_nodes=256))
    customer,service=scripted_dialogue_clients(record)
    config=EpisodeConfig(task=record,customer_model=customer.settings,service_model=service.settings,evidence_kind='scripted_transport_fixture',budget=Budget(max_customer_turns=30,max_customer_llm_calls=32,max_service_llm_calls=64,max_service_tool_calls=64))
    result=run(config,customer,service)
    assert result.metrics['task_success'],(result.score,result.dialogue_execution.errors)
    assert result.metrics['replay_verification'] and replay(json.loads(result.model_dump_json()))['verified']


def test_only_replayed_results_with_matching_interfaces_enter_evaluation_plan():
    from usami_benchmark.episodes import evaluation_result
    from usami_benchmark.evaluation import EvaluationPlan,aggregate
    frozen,t=single('cloudops');record=compile_task(frozen,TaskSelection(kind='tool',identity=t.identity,family=t.family,message=t.message,goal=t.goal,capabilities=t.capabilities))
    customer=ScriptedClient(SPEC,[]);service=ScriptedClient(SPEC,[tool_call('op',record.task.capabilities[0],'cloudops'),final()])
    config=EpisodeConfig(task=record,service_model=service.settings,customer_model=customer.settings,evidence_kind='scripted_transport_fixture');episode=run(config,customer,service)
    plan=EvaluationPlan(partition=frozen.spec.partition,evidence_kind=config.evidence_kind,model_configuration_hash=episode.config.model_configuration_hash,tasks={'cloudops':[record.task_hash]},interface_hashes={'cloudops':episode.interface_hash})
    result=evaluation_result(plan,episode)
    assert result.metrics['task_success'] and aggregate(plan,[result])['domains']['cloudops']['metrics']['task_success']['rate']==1
    assert aggregate(plan,[result])['macro']['task_success'] is None
    bad=episode.model_copy(deep=True);bad.metrics['task_success']=False;bad.execution_hash=bad.computed_hash()
    with pytest.raises(ValueError,match='score/cost'):evaluation_result(plan,bad)


def test_certified_result_strata_require_native_evidence_instead_of_caller_labels():
    from usami_benchmark.episodes import certified_evaluation_result
    from usami_benchmark.evaluation import EvaluationPlan
    frozen,t=single('cloudops')
    record=compile_task(frozen,TaskSelection(kind='tool',identity=t.identity,family=t.family,message=t.message,goal=t.goal,capabilities=t.capabilities))
    customer=ScriptedClient(SPEC,[])
    service=ScriptedClient(SPEC,[tool_call('op',record.task.capabilities[0],'cloudops'),final()])
    config=EpisodeConfig(task=record,service_model=service.settings,customer_model=customer.settings,evidence_kind='scripted_transport_fixture')
    episode=run(config,customer,service)
    plan=EvaluationPlan(partition=frozen.spec.partition,evidence_kind=config.evidence_kind,
        model_configuration_hash=episode.config.model_configuration_hash,tasks={'cloudops':[record.task_hash]},interface_hashes={'cloudops':episode.interface_hash})
    result=certified_evaluation_result(plan,episode)
    assert result.difficulty=='basic'
    assert set(result.capabilities)=={'identity_authorization','side_effect_consistency'}
    assert result.evidence_kind=='scripted_transport_fixture' and result.metrics['task_success']
    with pytest.raises(TypeError):
        certified_evaluation_result(plan,episode,capabilities=['capacity_resources'])
