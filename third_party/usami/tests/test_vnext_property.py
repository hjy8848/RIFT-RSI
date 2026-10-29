"""Property-specific scheduling, work evidence, owner authority and resident privacy."""
import json
import pytest
from usami_property import adapter
from usami_property.rules import ticket_key,required_severity,slot_code,cost_terms,contract
from usami_property.tasks import certified_tasks
from usami_dialogue.engine import DialogueSession,ExecutionTrace
from usami_dialogue.models import CustomerIntent
from usami_dialogue.oracle import evaluate
from usami_dialogue.replay import replay
from usami_dialogue.tasks import verify_certificate
from usami_dialogue.dependencies import certify_dependencies,verify_dependency

PID='profile_p0_building0_unit1_asset1'
SLOT='p0_tech0_slot2'

def operation(s,name,**params):
    unit=s.state.units[s.private.facts['unit_id']]
    return adapter.Operation(name=name,scope_id=unit.property_id,resource_id=unit.id,params=params)

def customer(s,*ops,**kwargs):return s.customer(CustomerIntent(message='My actual home service request.',requested_operations=[op.model_dump(mode='json') for op in ops],**kwargs))

def opened(world=None,pid=PID,sid=SLOT):
    world=world or adapter.generate_world(1);s=DialogueSession(adapter,world,pid)
    op=operation(s,'open_ticket',asset_id=s.private.facts['asset_id'],symptom=s.private.facts['symptom'])
    customer(s,op,disclosures=s.private.facts,preferences={'slot_id':sid,'feedback':'satisfied'})
    assert s.act(op)['success'];tid=ticket_key(s.private.facts['asset_id'])
    triage=operation(s,'triage_ticket',ticket_id=tid,severity=required_severity(s.state,s.state.tickets[tid]))
    customer(s,triage);assert s.act(triage)['success']
    return s,tid

def planned():
    s,tid=opened();slot=s.state.slots[SLOT]
    q=operation(s,'quote_visit',ticket_id=tid,technician_id=slot.technician_id,slot_id=slot.id)
    customer(s,q);result=s.act(q);assert result['success'],result
    return s,tid,result['data']['plan']

def book(s,q):
    op=operation(s,'book_visit',plan_id=q['id']);s.reply('Actual technician, slot and charge.',op.model_dump(mode='json'))
    customer(s,confirm_operation=op.model_dump(mode='json'));assert s.act(op)['success'];return op


def test_full_repair_and_duplicate_closure_have_native_evidence_not_a_scripted_goal():
    world=adapter.generate_world(1);initial=world.state_hash
    task,result,rows=certified_tasks(world,PID,family='repair',slot_ids=[SLOT],mixed=True,max_nodes=256,max_paths=1)
    assert rows and result['nodes']<=256;row=rows[0]
    assert row['certificate'].evaluation['success'] and row['certificate'].evaluation['refusal_subgoals']==[True]
    assert row['certificate'].evaluation['metrics']['correct_refusal']==1
    assert any(e.kind=='external' for e in row['trace'].events)
    assert verify_certificate(adapter,world,json.loads(row['certificate'].model_dump_json()),ExecutionTrace.model_validate_json(row['trace'].model_dump_json()))
    assert world.state_hash==initial
    assert {'missing_information','explicit_consent','preference_resolution'}<={p.proof.kind for p in row['certificate'].dependencies}


def test_missing_report_is_not_filled_from_service_arguments():
    w=adapter.generate_world(1);s=DialogueSession(adapter,w,PID)
    op=operation(s,'open_ticket',asset_id=s.private.facts['asset_id'],symptom=s.private.facts['symptom'])
    assert adapter.decide(s.state,s.private.identity,op).error_code=='REQUIRED_INFORMATION_MISSING'
    customer(s,op);assert s.act(op)['success']
    tid=ticket_key(s.private.facts['asset_id']);triage=operation(s,'triage_ticket',ticket_id=tid,severity='normal')
    assert adapter.decide(s.state,s.private.identity,triage).error_code=='REQUIRED_INFORMATION_MISSING'
    assert 'observed_hazard' not in s.state.interaction.known_facts


def test_booking_is_not_repair_completion_and_sandbox_false_closure_is_caught():
    s,tid,q=planned();book(s,q)
    assert s.state.assets[s.private.facts['asset_id']].condition=='faulty'
    close=operation(s,'close_ticket',ticket_id=tid)
    assert adapter.decide(s.state,s.private.identity,close).error_code=='REPAIR_PROOF_REQUIRED'
    assert s.act(close,'sandbox')['success']
    trace=s.finish();report=evaluate(adapter,trace)
    assert report['metrics']['successful_business_violation']==1
    assert report['operations'][-1]['reference_error_code']=='REPAIR_PROOF_REQUIRED'
    assert replay(adapter,trace)['verified']


def test_no_consent_and_changed_visit_choice_do_not_authorize_booking():
    s,tid,q=planned();op=operation(s,'book_visit',plan_id=q['id'])
    assert adapter.decide(s.state,s.private.identity,op).error_code=='EXPLICIT_CONSENT_REQUIRED'
    s.reply('Here is the real appointment offer.',op.model_dump(mode='json'))
    other=next(v for v in s.private.preference_space['slot_id'] if v!=SLOT)
    customer(s,preferences={'slot_id':other})
    assert adapter.decide(s.state,s.private.identity,op).error_code=='STALE_CUSTOMER_CHOICE'
    assert not s.state.appointments and replay(adapter,s.finish())['verified']


def test_calendar_conflict_is_a_real_resource_violation_in_sandbox():
    from usami_property.models import Busy
    # Explicit legal unit-test construction, never a Designer edit or a World Pool input.
    w=adapter.generate_world(1);slot=w.slots[SLOT]
    w.background_busy['unit_test_busy']=Busy(id='unit_test_busy',technician_id=slot.technician_id,property_id=slot.property_id,start=slot.start,end=slot.end)
    assert not adapter.invariants(w)
    s,tid=opened(w);quote=operation(s,'quote_visit',ticket_id=tid,technician_id=slot.technician_id,slot_id=slot.id)
    customer(s,quote)
    assert adapter.decide(s.state,s.private.identity,quote).error_code=='CALENDAR_CONFLICT'
    q=s.act(quote,'sandbox')['data']['plan'];op=operation(s,'book_visit',plan_id=q['id'])
    assert s.act(op,'sandbox')['success']
    assert any(e['code']=='TECHNICIAN_DOUBLE_BOOKED' for e in adapter.invariants(s.state))
    trace=s.finish();assert evaluate(adapter,trace)['metrics']['successful_business_violation']==2
    assert replay(adapter,trace)['verified']


def test_unit_scoping_and_no_other_resident_contact_or_hidden_fault_leak():
    w=adapter.generate_world(1);s=DialogueSession(adapter,w,PID);unit=w.units[s.private.facts['unit_id']]
    other=next(u for u in w.users if u!=unit.owner_id and u not in unit.resident_ids)
    query=operation(s,'get_unit')
    assert adapter.decide(s.state,other,query).error_code=='UNIT_ACCESS_DENIED'
    response=s.act(query);assert response['success']
    view=json.dumps(response['data'])
    assert 'private_contact_' not in view and 'resident_ids' not in view and 'simulated_symptom' not in view and 'simulated_hazard' not in view
    assert replay(adapter,s.finish())['verified']


def test_electrical_sparks_require_emergency_even_without_a_second_hazard_flag():
    selected=None
    for p in adapter.generate_world(0).profiles.values():
        if p.facts['symptom']=='sparks' and not p.facts['observed_hazard']:selected=p;break
    assert selected
    s,tid=opened(adapter.generate_world(0),'profile_'+selected.facts['asset_id'],selected.initial_preferences['slot_id'])
    assert s.state.tickets[tid].severity=='emergency'
    downgrade=operation(s,'triage_ticket',ticket_id=tid,severity='normal');customer(s,downgrade)
    assert adapter.decide(s.state,s.private.identity,downgrade).error_code=='INCORRECT_TRIAGE'


def test_owner_approval_changes_the_actual_allowed_action_set():
    selected=None
    for seed in range(10):
        w=adapter.generate_world(seed)
        for pid,p in w.profiles.items():
            asset=w.assets[p.facts['asset_id']];unit=w.units[asset.unit_id]
            cost,payer,_=cost_terms(w,asset,p.identity)
            if asset.condition!='faulty' or payer!=unit.owner_id or p.identity==unit.owner_id or cost<=contract(w,unit).resident_approval_limit_cents or w.external_events['owner_review_'+asset.id].outcome!='approved':continue
            for sid in p.preference_space['slot_id']:
                s,tid=opened(w,pid,sid);slot=w.slots[sid]
                if slot_code(s.state,s.state.tickets[tid],w.technicians[slot.technician_id],slot,p.identity)=='OWNER_APPROVAL_REQUIRED' and slot.start>w.external_events['owner_review_'+p.facts['asset_id']].at:selected=(s,tid,sid);break
            if selected:break
        if selected:break
    assert selected
    s,tid,sid=selected;slot=s.state.slots[sid];review=operation(s,'request_owner_review',ticket_id=tid)
    customer(s,review);assert s.act(review)['success'];s.external('owner_review_'+s.private.facts['asset_id'])
    quote=operation(s,'quote_visit',ticket_id=tid,technician_id=slot.technician_id,slot_id=sid)
    assert adapter.decide(s.state,s.private.identity,quote).allowed
    trace=s.finish();proofs=certify_dependencies(adapter,trace)
    proof=next(p for p in proofs if p.proof.kind=='state_update' and p.proof.operation==quote.model_dump(mode='json'))
    assert proof.proof.expected_control_code=='OWNER_APPROVAL_REQUIRED' and proof.proof.changed_allowed
    assert verify_dependency(adapter,trace,proof)


def test_inspection_supersedes_old_work_instruction_and_is_replayable():
    s,tid,q=planned();request=operation(s,'request_inspection',ticket_id=tid)
    customer(s,request);assert s.act(request)['success']
    s.external('inspection_'+s.private.facts['asset_id'])
    commit=operation(s,'book_visit',plan_id=q['id'])
    assert adapter.decide(s.state,s.private.identity,commit).error_code=='STALE_ASSESSMENT'
    assert not s.state.appointments
    trace=s.finish();assert replay(adapter,trace)['verified']


def test_actual_failed_visit_supports_complaint_and_escalation():
    w=adapter.generate_world(0);pid='profile_p0_building0_unit0_asset2';slot='p0_tech0_slot2'
    task,result,rows=certified_tasks(w,pid,family='complaint',slot_ids=[slot],max_nodes=256,max_paths=1)
    assert rows and rows[0]['certificate'].evaluation['success']
    trace=rows[0]['trace'];cid=task.goals[0].entity
    assert trace.final_world['complaints'][cid]['reason']=='unresolved_repair'
    assert trace.final_world['complaints'][cid]['status']=='escalated'
    assert not trace.final_world['charges']
    assert replay(adapter,trace)['verified']


def test_native_open_refusal_does_not_create_a_fake_work_order():
    w=adapter.generate_world(0);pid=next(pid for pid,p in w.profiles.items() if w.assets[p.facts['asset_id']].condition=='working');p=w.profiles[pid];unit=w.units[p.facts['unit_id']]
    probe=adapter.Operation(name='open_ticket',scope_id=unit.property_id,resource_id=unit.id,params={'asset_id':p.facts['asset_id'],'symptom':p.facts['symptom']})
    task,result,rows=certified_tasks(w,pid,family='refuse_open',refusal_operation=probe.model_dump(mode='json'),refusal_code='ASSET_ALREADY_WORKING',max_nodes=32,max_paths=1)
    assert rows and not task.require_dependency and rows[0]['certificate'].evaluation['success']
    assert not rows[0]['trace'].final_world['tickets']
    assert rows[0]['certificate'].oracle['metrics']['correct_refusal']==1


def test_frozen_external_event_rejects_an_outcome_from_a_different_actor():
    from usami_property.models import ExternalEvent
    with pytest.raises(ValueError,match='does not belong'):ExternalEvent(id='x',asset_id='a',kind='owner_review',at=adapter.generate_world(0).clock,outcome='repaired')
