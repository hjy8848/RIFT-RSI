"""Canonical two-agent journal through native education tools; scripted transport only."""
import json
from test_vnext_dialogue_agents import CheckedClient,intent,reply,service_tool
from test_vast_usami import SPEC
from usami_education import adapter
from usami_education.tasks import certified_tasks
from usami_education.rules import case_key
from usami_dialogue.agents import TaskRunConfig,run,replay
from usami_harness.models import Budget


def fixture_execution():
    w=adapter.generate_world(0);pid='profile_u1_enrollment1';p=w.profiles[pid];e=w.enrollments[p.facts['enrollment_id']]
    task,search,rows=certified_tasks(w,pid,family='refund',max_nodes=128,max_paths=1)
    assert rows
    config=TaskRunConfig(task=task,certificate=rows[0]['certificate'],witness=rows[0]['trace'],profile_id=pid,initial_world_hash=w.state_hash,tools=list(adapter.TOOLS),external_schedule=task.external_schedule,
        budget=Budget(max_customer_turns=5,max_customer_llm_calls=5,max_service_llm_calls=8,max_service_tool_calls=5))
    def op(name,**params):return adapter.Operation(name=name,scope_id=e.school_id,resource_id=e.id,params=params)
    opening=op('open_request',request_id=p.facts['request_id'],reason=p.facts['reason']);quote=op('quote_refund',case_id=case_key(e.id,p.facts['request_id']))
    q=next(iter(rows[0]['trace'].final_world['quotes'].values()));commit=op('commit_quote',quote_id=q['id'],payment_channel=e.payment_channel)
    customer=CheckedClient(SPEC,[intent('c0'),intent('c1',disclosures=p.facts,preferences={'resolution':'refund'},requested_operations=[opening.model_dump(mode='json')]),
        intent('c_quote',requested_operations=[quote.model_dump(mode='json')]),
        intent('c2',confirm_operation=commit.model_dump(mode='json')),intent('c3',finish=True)])
    service=CheckedClient(SPEC,[reply('r0','Which enrollment and why do you need help?'),service_tool('s0',opening),reply('r_quote','May I calculate the contractual unused credit refund?'),service_tool('s1',quote),
        reply('r1','Confirm the unused credit refund and contractual fee to the original channel.',commit),service_tool('s2',commit),reply('r2','The original-channel refund has been issued.')])
    result=run(adapter,w,config,customer,service)
    assert result.status=='customer_finished' and result.oracle['status']=='pass',(result.status,result.errors)
    assert result.trace.final_world['enrollments'][e.id]['status']=='refunded'
    assert result.cost['customer_calls']==5 and result.cost['service_calls']==7 and result.cost['tool_calls']==3
    for call in result.calls:
        if call.role=='service':
            wire=json.dumps(call.record.request)
            assert 'private_student_' not in wire and 'external_events' not in wire and 'private_hash' not in wire
    encoded=json.loads(json.dumps(result.model_dump(mode='json'),sort_keys=True))
    assert replay(adapter,encoded,world=w,config=config)['verified']
    return w,config,result


def test_native_refund_runs_through_production_agents_and_exact_serialized_replay():
    fixture_execution()
