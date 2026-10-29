"""Production Function Calling path, with explicitly scripted offline model transport."""
import json
from test_vnext_dialogue_agents import CheckedClient,intent,reply,service_tool
from test_vast_usami import SPEC
from test_vnext_property import PID,SLOT
from usami_property import adapter
from usami_property.rules import ticket_key
from usami_property.tasks import certified_tasks
from usami_dialogue.agents import TaskRunConfig,run,replay
from usami_harness.models import Budget,digest


def configuration():
    w=adapter.generate_world(1);task,result,rows=certified_tasks(w,PID,family='repair',slot_ids=[SLOT],max_nodes=256,max_paths=1)
    assert rows
    config=TaskRunConfig(task=task,certificate=rows[0]['certificate'],witness=rows[0]['trace'],profile_id=PID,initial_world_hash=w.state_hash,tools=list(adapter.TOOLS),external_schedule=task.external_schedule,
        budget=Budget(max_customer_turns=6,max_customer_llm_calls=7,max_service_llm_calls=16,max_service_tool_calls=12))
    return w,config


def fixture_execution():
    w,config=configuration();p=w.profiles[PID];u=w.units[p.facts['unit_id']];asset=w.assets[p.facts['asset_id']];slot=w.slots[SLOT];tid=ticket_key(asset.id)
    def op(name,**params):return adapter.Operation(name=name,scope_id=u.property_id,resource_id=u.id,params=params)
    opening=op('open_ticket',asset_id=asset.id,symptom=p.facts['symptom']);triage=op('triage_ticket',ticket_id=tid,severity='urgent')
    quote=op('quote_visit',ticket_id=tid,technician_id=slot.technician_id,slot_id=slot.id)
    qid='plan_'+digest([tid,1,0,slot.technician_id,slot.id,1,w.clock.isoformat()])[:20]
    booking=op('book_visit',plan_id=qid);read=op('get_ticket',ticket_id=tid);close=op('close_ticket',ticket_id=tid)
    c=CheckedClient(SPEC,[intent('c0'),intent('c1',disclosures=p.facts,preferences={'slot_id':SLOT,'feedback':'satisfied'},requested_operations=[opening.model_dump(mode='json')]),
        intent('c2',requested_operations=[triage.model_dump(mode='json'),quote.model_dump(mode='json')]),intent('c3',confirm_operation=booking.model_dump(mode='json')),
        intent('c4',confirm_operation=close.model_dump(mode='json')),intent('c5',finish=True)])
    s=CheckedClient(SPEC,[reply('r0','Which unit, asset and observed fault?'),service_tool('s0',opening),reply('r1','Please authorize assessment and an actual visit quote.'),
        service_tool('s1',triage),service_tool('s2',quote),reply('r2','Confirm the real visit time and charge.',booking),service_tool('s3',booking),service_tool('s4',read),
        reply('r3','The actual technician receipt reports repair. Are you satisfied and willing to close?',close),service_tool('s5',close),reply('r4','Your verified work order is closed.')])
    result=run(adapter,w,config,c,s)
    assert result.status=='customer_finished' and result.oracle['status']=='pass',(result.status,result.errors,result.tools)
    assert result.trace.final_world['tickets'][tid]['status']=='closed'
    assert len(result.oracle['consents'])==2
    assert result.cost['customer_calls']==6 and result.cost['service_calls']==11
    for call in result.calls:
        if call.role=='service':
            wire=json.dumps(call.record.request)
            assert 'private_contact_' not in wire and 'simulated_symptom' not in wire and 'simulated_hazard' not in wire and 'private_hash' not in wire
    assert replay(adapter,result,world=w,config=config)['verified']
    # Real archive writers sort object keys; exact model requests must survive it.
    encoded=json.loads(json.dumps(result.model_dump(mode='json'),sort_keys=True))
    assert replay(adapter,encoded,world=w,config=config)['verified']
    return w,config,result


def test_real_work_receipt_and_two_consents_cross_actual_agent_tool_boundary():
    fixture_execution()


def test_customer_task_drift_is_recorded_without_blame_or_calls_to_service():
    w,config=configuration();config.budget.max_format_recoveries=0
    other=next(s for s in w.profiles[PID].preference_space['slot_id'] if s!=SLOT)
    c=CheckedClient(SPEC,[intent('c0',preferences={'slot_id':other})]);s=CheckedClient(SPEC,[])
    result=run(adapter,w,config,c,s)
    assert result.status=='customer_protocol_error' and result.cost['service_calls']==0
    assert result.errors[0]['role']=='customer' and 'selected task' in result.errors[0]['message']
    assert not result.oracle['operations'] and result.oracle['metrics']['successful_business_violation']==0
    assert replay(adapter,result,world=w,config=config)['verified']
