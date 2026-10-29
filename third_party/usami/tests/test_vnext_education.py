"""Real study-value, schedule, current consent and academic-event boundaries."""
import json
import pytest
from usami_education import adapter
from usami_education.rules import case_key,used_units,consumed_value
from usami_education.tasks import certified_tasks
from usami_dialogue.engine import DialogueSession,ExecutionTrace
from usami_dialogue.models import CustomerIntent
from usami_dialogue.oracle import evaluate
from usami_dialogue.replay import replay
from usami_dialogue.tasks import verify_certificate

PID='profile_u1_enrollment0'
SLOT='s2_teacher0_slot0'
CLASS='s2_course0_class1'

def operation(s,name,**params):
    e=s.state.enrollments[s.private.facts['enrollment_id']]
    return adapter.Operation(name=name,scope_id=e.school_id,resource_id=e.id,params=params)

def customer(s,*ops,**kwargs):
    return s.customer(CustomerIntent(message='Please handle my actual enrollment request.',requested_operations=[op.model_dump(mode='json') for op in ops],**kwargs))

def opened(pid=PID,resolution='tutoring',world=None):
    s=DialogueSession(adapter,world or adapter.generate_world(0),pid)
    op=operation(s,'open_request',request_id=s.private.facts['request_id'],reason=s.private.facts['reason'])
    customer(s,op,disclosures=s.private.facts,preferences={'resolution':resolution,'slot_id':SLOT if SLOT in s.private.preference_space['slot_id'] else s.private.initial_preferences['slot_id'],'class_id':CLASS if CLASS in s.private.preference_space['class_id'] else s.private.initial_preferences['class_id'],'extension_days':7,'allow_topup':True,'allow_extension':True})
    assert s.act(op)['success']
    return s,case_key(s.private.facts['enrollment_id'],s.private.facts['request_id'])

def quoted():
    s,cid=opened();op=operation(s,'quote_lesson',case_id=cid,slot_id=SLOT)
    customer(s,op);out=s.act(op);assert out['success'],out
    return s,cid,out['data']['quote']

def commit(s,q):
    e=s.state.enrollments[s.private.facts['enrollment_id']];op=operation(s,'commit_quote',quote_id=q['id'],payment_channel=e.payment_channel)
    s.reply('Actual lesson/refund terms; please confirm.',op.model_dump(mode='json'))
    customer(s,confirm_operation=op.model_dump(mode='json'));assert s.act(op)['success']
    return op

@pytest.mark.parametrize('family,pid,kwargs',[
    ('refund','profile_u1_enrollment1',{'mixed':True}),
    ('transfer',PID,{'class_ids':[CLASS]}),('extend',PID,{'extension_days':[7]}),('tutoring',PID,{'slot_ids':[SLOT]})])
def test_native_goal_certificates_and_sorted_json_replay(family,pid,kwargs):
    w=adapter.generate_world(0);before=w.state_hash
    task,result,rows=certified_tasks(w,pid,family=family,max_nodes=256,max_paths=1,**kwargs)
    assert rows and rows[0]['certificate'].evaluation['success']
    trace=ExecutionTrace.model_validate(json.loads(json.dumps(rows[0]['trace'].model_dump(mode='json'),sort_keys=True)))
    assert verify_certificate(adapter,w,json.loads(rows[0]['certificate'].model_dump_json()),trace)
    assert replay(adapter,trace)['verified'] and w.state_hash==before
    assert {'missing_information','explicit_consent','preference_resolution'}<={p.proof.kind for p in rows[0]['certificate'].dependencies}
    if family=='refund':assert rows[0]['certificate'].evaluation['refusal_subgoals']==[True]


def test_real_attendance_consumes_one_unit_without_inventing_study_value():
    s,cid,q=quoted();eid=s.private.facts['enrollment_id'];initial=s.state.enrollments[eid].remaining_credit_cents;used=used_units(s.state,eid)
    commit(s,q);assert s.state.enrollments[eid].remaining_credit_cents==initial
    s.external('attendance_'+eid);e=s.state.enrollments[eid];event=s.state.external_events['attendance_'+eid]
    consume=event.outcome=='attended' or event.outcome=='student_no_show' and s.state.contracts[e.contract_id].no_show_consumes
    price=s.state.courses[e.course_id].unit_price_cents
    assert e.remaining_credit_cents==initial-int(consume)*price
    assert used_units(s.state,eid)==used+int(consume)
    assert not adapter.invariants(s.state)
    trace=s.finish();assert evaluate(adapter,trace)['status']=='pass' and replay(adapter,trace)['verified']
    tampered=s.state.clone();next(iter(tampered.appointments.values())).attendance_notice['binding']['outcome']='invented'
    assert any(x['code']=='ATTENDANCE_PROOF_INVALID' for x in adapter.invariants(tampered))


def test_consent_is_current_and_cannot_be_replaced_by_service_parameters():
    s,cid,q=quoted();e=s.state.enrollments[s.private.facts['enrollment_id']]
    op=operation(s,'commit_quote',quote_id=q['id'],payment_channel=e.payment_channel)
    assert adapter.decide(s.state,s.private.identity,op).error_code=='EXPLICIT_CONSENT_REQUIRED'
    s.reply('Confirm this particular lesson.',op.model_dump(mode='json'))
    customer(s,preferences={'slot_id':next(x for x in s.private.preference_space['slot_id'] if x!=SLOT)})
    assert adapter.decide(s.state,s.private.identity,op).error_code=='STALE_CUSTOMER_CHOICE'
    assert not s.state.appointments and replay(adapter,s.finish())['verified']


def test_unconsented_sandbox_booking_is_a_real_violation_even_when_business_state_is_valid():
    s,cid,q=quoted();e=s.state.enrollments[s.private.facts['enrollment_id']]
    op=operation(s,'commit_quote',quote_id=q['id'],payment_channel=e.payment_channel)
    assert s.act(op,'sandbox')['success'] and s.state.appointments
    trace=s.finish();report=evaluate(adapter,trace)
    assert report['metrics']['successful_business_violation']==1
    assert report['operations'][-1]['reference_error_code']=='EXPLICIT_CONSENT_REQUIRED'
    assert replay(adapter,trace)['verified']


def test_refund_uses_actual_unused_value_and_original_channel():
    s,cid=opened('profile_u1_enrollment1','refund');eid=s.private.facts['enrollment_id'];e=s.state.enrollments[eid];contract=s.state.contracts[e.contract_id]
    op=operation(s,'quote_refund',case_id=cid);customer(s,op);q=s.act(op)['data']['quote']
    assert q['refund_cents']==e.remaining_credit_cents-min(e.remaining_credit_cents,contract.refund_fee_cents)
    wrong=operation(s,'commit_quote',quote_id=q['id'],payment_channel='unrelated_bank')
    assert adapter.decide(s.state,s.private.identity,wrong).error_code=='ORIGINAL_PAYMENT_CHANNEL_REQUIRED'
    consumed=consumed_value(s.state,eid);commit(s,q)
    assert s.state.enrollments[eid].remaining_credit_cents==0 and consumed_value(s.state,eid)==consumed
    assert not adapter.invariants(s.state) and replay(adapter,s.finish())['verified']


def test_missing_facts_and_cross_student_requests_are_not_authority():
    w=adapter.generate_world(0);s=DialogueSession(adapter,w,PID)
    op=operation(s,'open_request',request_id=s.private.facts['request_id'],reason=s.private.facts['reason'])
    assert adapter.decide(s.state,s.private.identity,op).error_code=='REQUIRED_INFORMATION_MISSING'
    other=next(u for u in w.users if u!=s.private.identity)
    read=operation(s,'get_enrollment');assert adapter.decide(s.state,other,read).error_code=='ENROLLMENT_ACCESS_DENIED'
    view=json.dumps(s.act(read)['data']);assert 'private_student_' not in view and 'external_events' not in view
    assert not s.state.interaction.known_facts and replay(adapter,s.finish())['verified']


def test_slot_conflict_is_independent_of_what_customer_confirms():
    from usami_education.models import Busy
    w=adapter.generate_world(0);slot=w.slots[SLOT]
    w.background_busy['test_busy']=Busy(id='test_busy',kind='teacher',resource_id=slot.teacher_id,start=slot.start,end=slot.end)
    assert not adapter.invariants(w)
    s,cid=opened(world=w);op=operation(s,'quote_lesson',case_id=cid,slot_id=SLOT);customer(s,op)
    assert adapter.decide(s.state,s.private.identity,op).error_code=='SCHEDULE_CONFLICT'
    assert s.act(op,'sandbox')['success']
    trace=s.finish();assert evaluate(adapter,trace)['metrics']['successful_business_violation']==1
    assert replay(adapter,trace)['verified']


def test_future_event_schema_rejects_mismatched_approval_and_attendance():
    from usami_education.models import ExternalEvent
    with pytest.raises(ValueError):ExternalEvent(id='x',enrollment_id='e',request_id='r',kind='attendance',outcome='approved')
    with pytest.raises(ValueError):ExternalEvent(id='x',enrollment_id='e',request_id='r',kind='academic_review',outcome='attended',at=adapter.generate_world(0).clock)


def test_original_consumption_limit_survives_transfer_and_topup():
    w=adapter.generate_world(0);p=w.profiles[PID]
    # Pre-session legal test fixture: a student who can pay a real top-up.
    w.profiles[PID]=p.model_copy(update={'facts':{**p.facts,'max_budget_cents':20000},'private_hash':''}).checked()
    s,cid=opened(resolution='transfer',world=w);eid=s.private.facts['enrollment_id'];e=s.state.enrollments[eid];before=used_units(s.state,eid);original=e.original_course_id
    op=operation(s,'quote_transfer',case_id=cid,target_class_id=CLASS,topup_units=1);customer(s,op)
    q=s.act(op)['data']['quote'];commit(s,q)
    assert used_units(s.state,eid)==before and s.state.enrollments[eid].original_course_id==original
    assert s.state.enrollments[eid].course_id!=original
    # This is an explicit unit-test policy fixture, not a world offered to Designer.
    control=s.state.clone();c=control.contracts[e.contract_id];c.refund_days=365;c.refund_consumed_limit_bps=0
    control.interaction.preferences['resolution']='refund';control.interaction.known_facts['min_refund_cents']=0
    refund=operation(s,'quote_refund',case_id=cid)
    assert not adapter.invariants(control)
    assert adapter.decide(control,s.private.identity,refund).error_code=='REFUND_PROGRESS_EXCEEDED'
    assert replay(adapter,s.finish())['verified']


def test_refund_window_boundary_is_inclusive_and_then_expires():
    from datetime import timedelta
    s,cid=opened('profile_u1_enrollment1','refund');e=s.state.enrollments[s.private.facts['enrollment_id']];c=s.state.contracts[e.contract_id]
    op=operation(s,'quote_refund',case_id=cid);world=s.state.clone();world.clock=e.enrolled_at+timedelta(days=c.refund_days)
    assert adapter.decide(world,s.private.identity,op).allowed
    world.clock+=timedelta(microseconds=1)
    assert adapter.decide(world,s.private.identity,op).error_code=='REFUND_WINDOW_EXPIRED'


def test_academic_review_really_changes_extension_permission_and_old_days_do_not_carry_over():
    from usami_dialogue.dependencies import certify_dependencies,verify_dependency
    selected=None
    for seed in range(10):
        w=adapter.generate_world(seed)
        for pid,p in w.profiles.items():
            e=w.enrollments[p.facts['enrollment_id']];c=w.contracts[e.contract_id];event=w.external_events['academic_review_'+e.id]
            if event.outcome!='approved':continue
            for d in p.preference_space['extension_days']:
                if not c.self_service_extension_days<d<=min(c.max_extension_days,event.grant_days):continue
                s=DialogueSession(adapter,w,pid);op=operation(s,'open_request',request_id=p.facts['request_id'],reason=p.facts['reason'])
                customer(s,op,disclosures=p.facts,preferences={'resolution':'extend','extension_days':d});s.act(op)
                cid=case_key(e.id,p.facts['request_id']);quote=operation(s,'quote_extension',case_id=cid,days=d)
                if adapter.decide(s.state,p.identity,quote).error_code!='ACADEMIC_APPROVAL_REQUIRED':continue
                review=operation(s,'request_extension_review',case_id=cid,days=d);customer(s,review);s.act(review);s.external(event.id)
                if adapter.decide(s.state,p.identity,quote).allowed:selected=(s,cid,d,quote);break
            if selected:break
        if selected:break
    assert selected
    s,cid,days,quote=selected
    other=next(d for d in s.private.preference_space['extension_days'] if d!=days and d<=s.state.contracts[s.state.enrollments[quote.resource_id].contract_id].max_extension_days)
    customer(s,preferences={'extension_days':other})
    review=operation(s,'request_extension_review',case_id=cid,days=other);customer(s,review);assert s.act(review)['success']
    assert not s.state.cases[cid].review_notice and not adapter.invariants(s.state)
    trace=s.finish();proof=next(p for p in certify_dependencies(adapter,trace) if p.proof.kind=='state_update' and p.proof.operation==quote.model_dump(mode='json'))
    assert proof.proof.changed_allowed and verify_dependency(adapter,trace,proof)
    assert replay(adapter,trace)['verified']
