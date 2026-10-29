import json
import pytest
from usami.service.models import ModelSettings, ServiceError
from usami_cloudops import generate_world
from usami_cloudops.baselines import completion
from usami_cloudops.goals import make_task
from usami_cloudops.service import Budget, Experiment, ServiceSession,replay_execution
from usami_cloudops.models import Operation

SETTINGS=ModelSettings(model='offline-test',base_url='https://example.invalid/v1')
class Client:
    settings=SETTINGS
    def __init__(self,responses):self.responses=iter(responses)
    def complete(self,request):return next(self.responses)


def test_legitimate_autonomous_function_calls_and_external_authority():
    w,t=make_task(42,'billing_recovery')
    client=Client([completion(operations=t.witnesses[0].operations),completion('Restored project')])
    session=ServiceSession(Experiment(world=w,identity=t.identity),client)
    result=session.respond(t.message,t.capabilities);execution=session.finish(result['status'])
    assert execution.oracle['status']=='pass' and t.goal.satisfied(execution.world_trace.final_state)
    assert replay_execution(execution)==execution
    changed=execution.model_copy(deep=True);changed.inputs[0]['grants']=[];changed.execution_hash=changed.computed_hash()
    with pytest.raises(ValueError):replay_execution(changed)
    prompt=execution.messages[0]['content']
    assert 'witnesses' not in prompt and 'search_nodes' not in prompt and 'bill2' not in prompt


def test_reference_permission_refusal_and_ungranted_write():
    operation=Operation(name='start_instance',project_id='p0',resource_id='p0_db')
    session=ServiceSession(Experiment(world=generate_world(),identity='u2'),Client([completion(operations=[operation]),completion('PROJECT_ACCESS_REQUIRED')]))
    response=session.respond('Start production', [operation]);execution=session.finish(response['status'])
    assert execution.oracle['operations'][0]['classification']=='rejected_invalid_attempt'
    session=ServiceSession(Experiment(world=generate_world(),identity='u0'),Client([completion(operations=[operation]),completion('Unauthorized')]))
    session.respond('Read project info',[]);e=session.finish()
    assert e.tool_records[0].response['error_code']=='UNAUTHORIZED_OPERATION' and e.world_trace.final_state==e.experiment.world


def test_malformed_batch_and_error_parameters_never_mutate():
    good=completion(operations=[Operation(name='start_instance',project_id='p0',resource_id='p0_db')]);bad=json.loads(json.dumps(good));bad['choices'][0]['message']['tool_calls'].append({'id':'bad','type':'nonsense'})
    session=ServiceSession(Experiment(world=generate_world(),identity='u0'),Client([bad]));r=session.respond('Start',[])
    assert r['status']=='model_output_error' and not session.tool_records and session.state==generate_world()
    good['choices'][0]['message']['tool_calls'][0]['function']['arguments']='{"project_id":"p0","resource_id":"p0_db","params":{"surprise":true}}'
    session=ServiceSession(Experiment(world=generate_world(),identity='u0'),Client([good,completion('Bad parameters')]));session.respond('Start',[])
    assert session.tool_records[0].response['error_code']=='TOOL_ARGUMENT_INVALID' and session.state==generate_world()
    assert replay_execution(session.finish()).oracle['status']=='pass'


def test_llm_and_tool_budgets_reset_and_identity():
    w,t=make_task(42,'billing_recovery')
    session=ServiceSession(Experiment(world=w,identity=t.identity,budget=Budget(max_service_llm_calls=1)),Client([completion(operations=t.witnesses[0].operations)]))
    r=session.respond(t.message,t.capabilities);assert r['status']=='llm_budget_exhausted'
    e=session.finish(r['status']);assert replay_execution(e)==e
    session.reset();assert session.state==w and not session.authority
    session=ServiceSession(Experiment(world=w,identity=t.identity,budget=Budget(max_service_tool_calls=1)),Client([completion(operations=t.witnesses[0].operations)]))
    assert session.respond(t.message,t.capabilities)['status']=='tool_budget_exhausted' and len(session.tool_records)==1
    with pytest.raises(ValueError):Experiment(world=w,identity='someone-else')


@pytest.mark.parametrize('error,status', [
    (ServiceError('PROVIDER_TIMEOUT','Timed out'),'provider_error'),
    (ServiceError('MODEL_OUTPUT_INVALID','Invalid provider JSON'),'model_output_error'),
    (KeyboardInterrupt(),'interrupted'),
    (RuntimeError('private transport details'),'provider_error')])
def test_partial_provider_attempt_is_retained_and_replays_without_retry(error,status):
    class FailingClient(Client):
        def complete(self,request):raise error
    cfg=Experiment(world=generate_world(),identity='u0')
    session=ServiceSession(cfg,FailingClient([]));response=session.respond('Inspect my project')
    execution=session.finish(response['status'])
    assert response['status']==status and len(execution.llm_records)==1
    assert not execution.tool_records and execution.world_trace.final_state==cfg.world
    assert 'private transport details' not in execution.model_dump_json()
    assert replay_execution(execution,cfg)==execution
