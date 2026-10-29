"""Same production Function Calling runner; scripted transport is protocol evidence only."""
import json
from test_vnext_dialogue_agents import CheckedClient,intent,reply,service_tool
from test_vast_usami import SPEC
from usami_airline import adapter
from usami_airline.rules import quote_key
from usami_dialogue.agents import RunConfig,run,replay
from usami_harness.models import Budget


def test_airline_fixed_customer_clarification_actual_quote_consent_and_model_replay():
    world=adapter.generate_world(0);pid='profile_u0_booking1';private=world.profiles[pid];b=world.bookings[private.facts['booking_id']]
    def op(name,**p):return adapter.Operation(name=name,scope_id=b.carrier_id,resource_id=b.id,params=p)
    verify=op('verify_passenger',document_token=private.facts['document_token']);quote=op('quote_refund')
    commit=op('commit_quote',quote_id=quote_key(b,'refund',b.flights,b.cabin,0,1,world.clock))
    customer=CheckedClient(SPEC,[intent('c0'),intent('c1',preferences={'service':'refund'},requested_operations=[verify.model_dump(mode='json'),quote.model_dump(mode='json')]),
        intent('c2',confirm_operation=commit.model_dump(mode='json')),intent('c3',finish=True)])
    service=CheckedClient(SPEC,[reply('r0','Which booking and passenger document do you hold?'),service_tool('s0',verify),service_tool('s1',quote),
        reply('r1','Here are the actual refund amount and fare fee. Do you consent?',commit),service_tool('s2',commit),reply('r2','The real simulator refund has been issued.')])
    config=RunConfig(profile_id=pid,initial_world_hash=world.state_hash,tools=list(adapter.TOOLS),budget=Budget(max_customer_turns=4,max_customer_llm_calls=4,max_service_llm_calls=8,max_service_tool_calls=8))
    result=run(adapter,world,config,customer,service)
    assert result.status=='customer_finished' and result.oracle['status']=='pass'
    assert result.trace.final_world['bookings'][b.id]['status']=='cancelled'
    assert result.cost['customer_calls']==4 and result.cost['service_calls']==6
    assert result.cost['by_role']['customer']['calls']==4 and result.cost['by_role']['service']['calls']==6
    for row in result.calls:
        if row.role=='service':
            wire=json.dumps(row.record.request)
            assert 'simulated_identity_valid' not in wire and 'private_hash' not in wire and 'preference_space' not in wire
    assert replay(adapter,result,world=world,config=config)['verified']
    assert replay(adapter,json.loads(json.dumps(result.model_dump(mode='json'),sort_keys=True)),world=world)['verified']


def test_certified_task_goal_injection_excludes_hidden_answers_and_old_protocol_still_works():
    from usami_airline.tasks import certified_tasks
    from usami_dialogue.agents import TaskRunConfig,AgentExecution
    world=adapter.generate_world(0)
    task,_,rows=certified_tasks(world,'profile_u0_booking1',family='refund',max_nodes=128,max_paths=1)
    config=TaskRunConfig(task=task,certificate=rows[0]['certificate'],witness=rows[0]['trace'],profile_id=task.profile_id,initial_world_hash=world.state_hash,
        tools=list(adapter.TOOLS),external_schedule=task.external_schedule,budget=Budget(max_customer_turns=2,max_customer_llm_calls=2,max_service_llm_calls=2))
    customer=CheckedClient(SPEC,[intent('c0'),intent('c1',finish=True)])
    service=CheckedClient(SPEC,[reply('r0','Let us review the actual booking.')])
    result=run(adapter,world,config,customer,service)
    assert result.protocol=='business-dialogue-task-agents-v2'
    first=result.calls[0].record.request['messages'][0]['content']
    assert 'Your chosen business request:' in first and 'You want a refund for this booking.' in first
    assert 'certificate_hash' not in first and 'expected_changed_code' not in first and 'direct_edges' not in first
    for row in result.calls:
        if row.role=='service':assert 'Your chosen business request:' not in json.dumps(row.record.request)
    serialized=AgentExecution.model_validate_json(result.model_dump_json())
    assert replay(adapter,serialized,world=world,config=config)['verified']
    assert isinstance(serialized.config,TaskRunConfig)


def test_invalid_task_certificate_is_rejected_before_any_model_call():
    import pytest
    from usami_airline.tasks import certified_tasks
    from usami_dialogue.agents import TaskRunConfig
    world=adapter.generate_world(0);task,_,rows=certified_tasks(world,'profile_u0_booking1',family='refund',max_nodes=128,max_paths=1)
    c=rows[0]['certificate'].model_copy(deep=True);c.evaluation['success']=False;c.certificate_hash=c.computed_hash()
    config=TaskRunConfig(task=task,certificate=c,witness=rows[0]['trace'],profile_id=task.profile_id,initial_world_hash=world.state_hash,tools=list(adapter.TOOLS))
    class NoCalls:
        settings=SPEC
        def complete(self,request):raise AssertionError('invalid certificate must not incur a model call')
    with pytest.raises(ValueError,match='reproduce'):run(adapter,world,config,NoCalls(),NoCalls())
