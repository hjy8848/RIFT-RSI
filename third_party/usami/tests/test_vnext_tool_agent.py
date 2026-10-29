"""Real native Service code paths with explicitly scripted transport, no API calls."""
import json
import pytest
from test_vast_usami import ScriptedClient,SPEC,answer,call
from test_vnext_tool_tasks import single
from usami_benchmark.tool_tasks import certify,binding,load_world
from usami_benchmark.tool_agent import ToolRunConfig,run,replay
from usami_harness.models import Budget


def config_for(domain,client,refusal=False,mode='sandbox'):
    frozen,task=single(domain,refusal);cert=certify(frozen,task,max_depth=2,max_nodes=16)
    if domain=='saas':
        from usami.service.models import TOOL_NAMES_V2
        tools=list(TOOL_NAMES_V2)
    else:
        import importlib
        tools=list(importlib.import_module('usami_'+domain+'.models').TOOLS)
    return ToolRunConfig(frozen=frozen,task=cert.task,certificate=cert,mode=mode,tools=tools,budget=Budget(max_customer_turns=1,max_customer_llm_calls=0,max_service_llm_calls=4,max_service_tool_calls=4,max_format_recoveries=0),service_model=client.settings,evidence_kind='scripted_transport_fixture')


def tool_call(identifier,op,domain):
    args={k:v for k,v in op.items() if k!='name'}
    if domain=='saas':args={k:v for k,v in args.items() if k!='params'}|op['params']
    return answer(calls=[call(identifier,op['name'],args)])


def final(refusals=None):return answer(json.dumps({'message':'My actual business result.','refusals':refusals or []},sort_keys=True))


@pytest.mark.parametrize('domain',['saas','cloudops','fulfillment','stayops'])
def test_native_model_tools_and_canonical_full_journal_replay(domain):
    client=ScriptedClient(SPEC,[]);config=config_for(domain,client);op=config.task.capabilities[0]
    client.responses=iter([tool_call('write',op,domain),final()])
    result=run(config,client)
    assert result.score['task_success'],result.score
    assert result.cost['customer_calls']==0 and result.cost['service_calls']==2 and result.cost['tool_calls']==1
    assert result.cost['service_input_tokens'] is None and result.cost['usage_complete'] is False
    assert replay(json.loads(json.dumps(result.model_dump(mode='json'),sort_keys=True)),config=config)['verified']
    for record in result.native_execution['llm_records']:
        text=json.dumps(record['request'])
        assert config.frozen.world_hash not in text and config.certificate.certificate_hash not in text
        assert 'native-tool-certificate' not in text and 'witness_trace' not in text


@pytest.mark.parametrize('domain',['saas','cloudops','fulfillment','stayops'])
def test_service_must_decide_correct_refusal_without_attempting_the_write(domain):
    client=ScriptedClient(SPEC,[]);config=config_for(domain,client,True)
    ref=config.certificate.witnesses[0]['refusals'][0]
    client.responses=iter([final([{'operation':ref['operation'],'code':ref['reference_code']}])])
    result=run(config,client)
    assert result.score['task_success'] and result.score['refusal_subgoals']==[True]
    assert result.cost['tool_calls']==0 and replay(result)['verified']


def test_engine_block_is_not_service_correct_refusal_and_has_no_fake_business_violation():
    domain='cloudops';client=ScriptedClient(SPEC,[]);config=config_for(domain,client,True,mode='reference')
    ref=config.certificate.witnesses[0]['refusals'][0]
    client.responses=iter([tool_call('attempt',ref['operation'],domain),final([{'operation':ref['operation'],'code':ref['reference_code']}])])
    result=run(config,client)
    assert not result.score['task_success'] and result.score['refusal_subgoals']==[False]
    assert result.score['rejected_invalid_attempts']==1 and result.score['successful_business_violations']==0
    assert replay(result)['verified']


def test_actual_sandbox_violation_and_false_refusal_are_independently_identified():
    domain='cloudops';client=ScriptedClient(SPEC,[]);config=config_for(domain,client,True)
    ref=config.certificate.witnesses[0]['refusals'][0];client.responses=iter([tool_call('badwrite',ref['operation'],domain),final()])
    result=run(config,client)
    assert result.score['successful_business_violations']==1 and not result.score['task_success']
    assert replay(result)['verified']
    client=ScriptedClient(SPEC,[]);config=config_for(domain,client)
    client.responses=iter([final([{'operation':config.task.capabilities[0],'code':'PROJECT_ACCESS_REQUIRED'}])])
    result=run(config,client)
    assert result.score['false_refusal'] and not result.score['task_success'] and replay(result)['verified']


def test_bad_final_format_is_protocol_failure_not_oracle_business_discovery():
    client=ScriptedClient(SPEC,[answer('unstructured reply')]);config=config_for('cloudops',client)
    result=run(config,client)
    assert result.score['service_response_protocol_error'] and not result.score['task_success']
    assert result.score['successful_business_violations']==0 and replay(result)['verified']
    changed=result.model_copy(deep=True);changed.native_execution['final_reply']='tampered'
    with pytest.raises(ValueError,match='integrity'):replay(changed)


def test_invalid_certificate_fails_before_any_model_call_and_budget_failure_is_replayable():
    client=ScriptedClient(SPEC,[]);config=config_for('cloudops',client);config.certificate.certificate_hash='0'*64
    with pytest.raises(ValueError,match='certificate hash'):run(config,client)
    assert client.requests==[]
    client=ScriptedClient(SPEC,[tool_call('query',{'name':'get_customer_context'},'cloudops')]);config=config_for('cloudops',client);config.budget.max_service_llm_calls=1
    result=run(config,client)
    assert result.cost['service_calls']==1 and not result.score['task_success'] and replay(result)['verified']
