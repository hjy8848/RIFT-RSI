"""Independent education contracts and expected state transitions."""
from datetime import datetime,timedelta
from usami_harness.models import Decision,digest,signed_notice,notice_valid
from usami_dialogue.process import consent_code
from .models import Case,Quote,Appointment,Ledger
POLICY={
 'ENROLLMENT_ACCESS_DENIED':'Only the authenticated student may change their school-scoped enrollment.',
 'REQUIRED_INFORMATION_MISSING':'Clarify the enrollment, actual problem and relevant budget/time/credit requirements.',
 'REFUND_WINDOW_EXPIRED':'Refunds must be inside the signed contract window.',
 'REFUND_PROGRESS_EXCEEDED':'Cumulative consumed units use the original contract limit; transfer/top-up never reset progress.',
 'PENDING_LESSONS':'Cancel outstanding confirmed lessons before refund or changing curriculum.',
 'COURSE_EXPIRED':'Renew eligibility before reserving a new place or lesson.',
 'COURSE_NOT_IN_PACKAGE':'Transfers stay inside the contracted course package and school.',
 'ENTRY_LEVEL_NOT_MET':'A new course must fit the verified student placement level.',
 'CLASS_CAPACITY_EXCEEDED':'A class has a finite number of active seats.',
 'SCHEDULE_CONFLICT':'A student or teacher cannot be booked across overlapping commitments.',
 'STUDY_CREDIT_INSUFFICIENT':'Consumed and reserved study value cannot be invented or double-spent.',
 'ACADEMIC_APPROVAL_REQUIRED':'Extension beyond self-service authority requires an actual academic review for these days/version.',
 'EXPLICIT_CONSENT_REQUIRED':'Charges, refund, course changes and lesson reservations require exact current quote consent.'}


def case_key(eid,request):return 'case_'+digest([eid,request])[:20]
def used_units(w,eid):return sum(r.units for r in w.learning_records.values() if r.enrollment_id==eid)
def consumed_value(w,eid):return sum(r.units*r.unit_price_cents for r in w.learning_records.values() if r.enrollment_id==eid)
def occupancy(w,cid,exclude=None):return sum(e.status=='active' and e.class_id==cid and e.id!=exclude for e in w.enrollments.values())
def overlaps(a,b):return a.start<b.end and b.start<a.end

def reservation_value(w,eid):return sum(a.unit_price_cents for a in w.appointments.values() if a.enrollment_id==eid and a.status=='scheduled')
def case_spend(w,cid):return sum(x.amount_cents for x in w.ledger.values() if x.case_id==cid and x.kind=='charge')
def quote_hash(q):return digest(q.model_dump(mode='json',exclude={'quote_hash','status'}))
def quote_key(e,case,action,params,revision,clock):return 'quote_'+digest([e.id,case.id,action,params,e.version,revision,clock.isoformat()])[:20]
def record_binding(r):return {'record_id':r.id,'enrollment_id':r.enrollment_id,'course_id':r.course_id,'units':r.units,'unit_price_cents':r.unit_price_cents,'origin':r.origin,'appointment_id':r.appointment_id}

def student_busy(w,uid,interval,exclude_enrollment=None):
    if any(b.kind=='student' and b.resource_id==uid and overlaps(interval,b) for b in w.background_busy.values()):return True
    if any(e.status=='active' and e.student_id==uid and e.id!=exclude_enrollment and any(overlaps(interval,m) for m in w.classes[e.class_id].meetings) for e in w.enrollments.values()):return True
    return any(a.status=='scheduled' and w.enrollments[a.enrollment_id].student_id==uid and overlaps(interval,w.slots[a.slot_id]) for a in w.appointments.values())

def teacher_busy(w,tid,interval):
    return (any(b.kind=='teacher' and b.resource_id==tid and overlaps(interval,b) for b in w.background_busy.values()) or
        any(c.teacher_id==tid and any(overlaps(interval,m) for m in c.meetings) for c in w.classes.values()) or
        any(a.status=='scheduled' and w.slots[a.slot_id].teacher_id==tid and overlaps(interval,w.slots[a.slot_id]) for a in w.appointments.values()))


def invariants(w):
    errors=[]
    def add(code,path):errors.append({'code':code,'path':path})
    for name in ('users','schools','courses','packages','teachers','classes','contracts','enrollments','learning_records','slots','background_busy','cases','quotes','appointments','ledger','external_events'):
        for key,row in getattr(w,name).items():
            if row.id!=key:add('ENTITY_KEY_MISMATCH',name+'/'+key)
    for c in w.courses.values():
        if c.school_id not in w.schools:add('COURSE_REFERENCE_INVALID','courses/'+c.id)
    for p in w.packages.values():
        if p.school_id not in w.schools or len(set(p.course_ids))!=len(p.course_ids) or any(cid not in w.courses or w.courses[cid].school_id!=p.school_id for cid in p.course_ids):add('PACKAGE_REFERENCE_INVALID','packages/'+p.id)
    for t in w.teachers.values():
        if t.school_id not in w.schools or not t.subjects or set(t.subjects)-{'math','language','coding'}:add('TEACHER_REFERENCE_INVALID','teachers/'+t.id)
    intervals=[]
    for c in w.classes.values():
        course=w.courses.get(c.course_id);teacher=w.teachers.get(c.teacher_id)
        if not course or not teacher or c.school_id!=course.school_id or teacher.school_id!=c.school_id or course.subject not in teacher.subjects or course.level>teacher.max_level:add('CLASS_REFERENCE_INVALID','classes/'+c.id)
        if occupancy(w,c.id)>c.capacity:add('CLASS_OVERBOOKED','classes/'+c.id)
        for m in c.meetings:
            if m.start.tzinfo is None or m.end.tzinfo is None or m.start>=m.end:add('MEETING_INVALID','classes/'+c.id)
            intervals.append((c.teacher_id,m))
    for i,(tid,a) in enumerate(intervals):
        if any(other==tid and overlaps(a,b) for other,b in intervals[i+1:]):add('TEACHER_CLASS_CONFLICT','classes')
        if any(b.kind=='teacher' and b.resource_id==tid and overlaps(a,b) for b in w.background_busy.values()):add('TEACHER_CLASS_CONFLICT','classes')
    for c in w.contracts.values():
        p=w.packages.get(c.package_id)
        if not p or p.school_id!=c.school_id or c.self_service_extension_days>c.max_extension_days:add('CONTRACT_REFERENCE_INVALID','contracts/'+c.id)
    for e in w.enrollments.values():
        c=w.contracts.get(e.contract_id);co=w.courses.get(e.course_id);original=w.courses.get(e.original_course_id);cl=w.classes.get(e.class_id)
        if not c or not co or not original or not cl or e.student_id not in w.users or any(s!=e.school_id for s in (c.school_id,co.school_id,original.school_id,cl.school_id)) or cl.course_id!=e.course_id:add('ENROLLMENT_REFERENCE_INVALID','enrollments/'+e.id);continue
        if co.id not in w.packages[c.package_id].course_ids or co.level>w.users[e.student_id].levels.get(co.subject,0):add('ENROLLMENT_QUALIFICATION_INVALID','enrollments/'+e.id)
        if e.enrolled_at.tzinfo is None or e.expires_at.tzinfo is None or e.enrolled_at>w.clock or e.expires_at<=e.enrolled_at:add('ENROLLMENT_DATES_INVALID','enrollments/'+e.id)
        if e.initial_paid_cents!=c.total_units*original.unit_price_cents:add('OPENING_PAYMENT_INVALID','enrollments/'+e.id)
        net=e.initial_paid_cents+sum((1 if x.kind=='charge' else -1)*x.amount_cents for x in w.ledger.values() if x.enrollment_id==e.id)
        if net!=consumed_value(w,e.id)+e.remaining_credit_cents+e.retained_fees_cents:add('STUDY_VALUE_NOT_CONSERVED','enrollments/'+e.id)
        if reservation_value(w,e.id)>e.remaining_credit_cents:add('STUDY_CREDIT_DOUBLE_RESERVED','enrollments/'+e.id)
        if e.status=='refunded' and (e.remaining_credit_cents or any(a.enrollment_id==e.id and a.status=='scheduled' for a in w.appointments.values())):add('REFUNDED_ENROLLMENT_ACTIVE','enrollments/'+e.id)
        if e.status=='active':
            for m in cl.meetings:
                if any(b.kind=='student' and b.resource_id==e.student_id and overlaps(m,b) for b in w.background_busy.values()):add('STUDENT_CLASS_CONFLICT','enrollments/'+e.id)
                if any(other.id!=e.id and other.status=='active' and other.student_id==e.student_id and other.class_id in w.classes and any(overlaps(m,n) for n in w.classes[other.class_id].meetings) for other in w.enrollments.values()):add('STUDENT_CLASS_CONFLICT','enrollments/'+e.id)
    for r in w.learning_records.values():
        e=w.enrollments.get(r.enrollment_id);c=w.courses.get(r.course_id)
        if not e or not c or c.school_id!=e.school_id or r.unit_price_cents!=c.unit_price_cents or not notice_valid(r.notice) or r.notice.get('domain')!=w.domain or r.notice.get('source')!='education-progress-simulator-v0.1' or r.notice.get('binding')!=record_binding(r):add('LEARNING_RECORD_INVALID','learning_records/'+r.id)
        if r.origin=='attendance' and (r.appointment_id not in w.appointments or w.appointments[r.appointment_id].enrollment_id!=r.enrollment_id or w.appointments[r.appointment_id].status not in {'attended','missed'}):add('ATTENDANCE_REFERENCE_INVALID','learning_records/'+r.id)
    for s in w.slots.values():
        t=w.teachers.get(s.teacher_id)
        if not t or t.school_id!=s.school_id or s.start.tzinfo is None or s.end.tzinfo is None or s.start>=s.end:add('SLOT_REFERENCE_INVALID','slots/'+s.id)
    for b in w.background_busy.values():
        if (b.kind=='teacher' and b.resource_id not in w.teachers) or (b.kind=='student' and b.resource_id not in w.users) or b.start>=b.end:add('BUSY_REFERENCE_INVALID','background_busy/'+b.id)
    for c in w.cases.values():
        e=w.enrollments.get(c.enrollment_id)
        if not e or c.requester_id!=e.student_id or c.id!=case_key(e.id,c.request_id):add('CASE_REFERENCE_INVALID','cases/'+c.id)
        if c.last_slot_id is not None and not any(a.case_id==c.id and a.slot_id==c.last_slot_id for a in w.appointments.values()):add('CASE_LESSON_MISMATCH','cases/'+c.id)
        if c.last_appointment_id is not None:
            last=w.appointments.get(c.last_appointment_id)
            if not last or last.case_id!=c.id or last.slot_id!=c.last_slot_id or last.status!=c.last_lesson_status:add('CASE_LESSON_MISMATCH','cases/'+c.id)
        attended=sum(r.units for r in w.learning_records.values() if r.appointment_id in w.appointments and w.appointments[r.appointment_id].case_id==c.id and w.appointments[r.appointment_id].status=='attended')
        if c.attended_units!=attended:add('CASE_PROGRESS_MISMATCH','cases/'+c.id)
        if c.review_notice:
            proof=c.review_notice;b=proof.get('binding',{});event=w.external_events.get(b.get('event_id'))
            if not e or not event or event.kind!='academic_review' or event.enrollment_id!=e.id or event.request_id!=c.request_id or event.id not in w.applied_events or not notice_valid(proof) or proof.get('source')!='education-academic-simulator-v0.1' or proof.get('domain')!=w.domain or b!={'case_id':c.id,'enrollment_id':e.id,'days':c.pending_extension_days,'version':c.review_version,'approved':event.outcome=='approved' and (c.pending_extension_days or 0)<=event.grant_days,'event_id':event.id}:add('ACADEMIC_APPROVAL_INVALID','cases/'+c.id)
    for q in w.quotes.values():
        c=w.cases.get(q.case_id)
        if not c or c.enrollment_id!=q.enrollment_id or q.quote_hash!=quote_hash(q):add('QUOTE_INTEGRITY','quotes/'+q.id)
    scheduled=[]
    for a in w.appointments.values():
        e=w.enrollments.get(a.enrollment_id);q=w.quotes.get(a.quote_id);slot=w.slots.get(a.slot_id)
        course=w.courses.get(a.course_id)
        if not e or not q or not slot or not course or q.enrollment_id!=e.id or q.case_id!=a.case_id or q.action!='tutoring' or q.status!='committed' or a.id!='lesson_'+q.id or a.unit_price_cents!=course.unit_price_cents:add('LESSON_REFERENCE_INVALID','appointments/'+a.id);continue
        if a.attendance_notice:
            proof=a.attendance_notice;b=proof.get('binding',{});event=w.external_events.get(b.get('event_id'))
            consume=bool(event and (event.outcome=='attended' or event.outcome=='student_no_show' and w.contracts[e.contract_id].no_show_consumes))
            expected_status={'attended':'attended','student_no_show':'missed','teacher_cancelled':'cancelled'}
            expected_binding={'appointment_id':a.id,'enrollment_id':e.id,'outcome':event.outcome if event else None,'consumed':consume,'at':slot.end.isoformat(),'event_id':event.id if event else None}
            records=[r for r in w.learning_records.values() if r.appointment_id==a.id]
            if not event or event.kind!='attendance' or event.enrollment_id!=e.id or event.request_id!=w.cases[a.case_id].request_id or event.id not in w.applied_events or not notice_valid(proof) or proof.get('domain')!=w.domain or proof.get('source')!='education-attendance-simulator-v0.1' or b!=expected_binding or a.status!=expected_status.get(event.outcome) or sum(r.units for r in records)!=int(consume) or any(r.course_id!=a.course_id or r.unit_price_cents!=a.unit_price_cents for r in records):add('ATTENDANCE_PROOF_INVALID','appointments/'+a.id)
        elif a.status in {'attended','missed'}:add('ATTENDANCE_PROOF_INVALID','appointments/'+a.id)
        if a.status=='scheduled':
            if a.course_id!=e.course_id or teacher_busy_without_appointments(w,slot.teacher_id,slot) or student_busy_without_appointments(w,e.student_id,slot):add('LESSON_SCHEDULE_INVALID','appointments/'+a.id)
            scheduled.append(a)
    for i,a in enumerate(scheduled):
        for b in scheduled[i+1:]:
            if overlaps(w.slots[a.slot_id],w.slots[b.slot_id]) and (w.slots[a.slot_id].teacher_id==w.slots[b.slot_id].teacher_id or w.enrollments[a.enrollment_id].student_id==w.enrollments[b.enrollment_id].student_id):add('LESSON_DOUBLE_BOOKED','appointments')
    for x in w.ledger.values():
        q=w.quotes.get(x.quote_id);e=w.enrollments.get(x.enrollment_id)
        binding={'enrollment_id':x.enrollment_id,'case_id':x.case_id,'quote_id':x.quote_id,'kind':x.kind,'amount_cents':x.amount_cents,'channel':x.channel}
        expected=q.charge_cents if q and x.kind=='charge' else q.refund_cents if q else -1
        if not q or not e or q.status!='committed' or q.case_id!=x.case_id or q.enrollment_id!=e.id or x.id!=x.kind+'_'+q.id or x.amount_cents!=expected or x.channel!=e.payment_channel or not notice_valid(x.notice) or x.notice.get('domain')!=w.domain or x.notice.get('source')!='education-payment-simulator-v0.1' or x.notice.get('binding')!=binding:add('EDUCATION_PAYMENT_INVALID','ledger/'+x.id)
    for p in w.profiles.values():
        try:p.checked()
        except ValueError:add('PRIVATE_PROFILE_INTEGRITY','profiles')
        e=w.enrollments.get(p.facts.get('enrollment_id'))
        if not e or e.student_id!=p.identity or p.protected_objects!=(e.id,):add('PRIVATE_PROFILE_REFERENCE','profiles')
    for event in w.external_events.values():
        if event.enrollment_id not in w.enrollments:add('EXTERNAL_EVENT_INVALID','external_events/'+event.id)
    if len(set(w.applied_events))!=len(w.applied_events) or set(w.applied_events)-set(w.external_events):add('EXTERNAL_EVENT_INVALID','applied_events')
    return errors


def teacher_busy_without_appointments(w,tid,s):return any(b.kind=='teacher' and b.resource_id==tid and overlaps(s,b) for b in w.background_busy.values()) or any(c.teacher_id==tid and any(overlaps(s,m) for m in c.meetings) for c in w.classes.values())
def student_busy_without_appointments(w,uid,s):return any(b.kind=='student' and b.resource_id==uid and overlaps(s,b) for b in w.background_busy.values()) or any(e.status=='active' and e.student_id==uid and any(overlaps(s,m) for m in w.classes[e.class_id].meetings) for e in w.enrollments.values())


def proposal(w,e,case,action,p):
    ctx=w.interaction;c=w.contracts[e.contract_id];course=w.courses[e.course_id]
    if e.status!='active':return None,'ENROLLMENT_REFUNDED'
    resolution=ctx.preferences.get('resolution')
    if resolution!=action and not (action=='extend' and resolution in {'transfer','tutoring'} and ctx.preferences.get('allow_extension') is True):return None,'PREFERENCE_RESOLUTION_REQUIRED'
    if 'max_budget_cents' not in ctx.known_facts:return None,'REQUIRED_INFORMATION_MISSING'
    charge=refund=fee=days=topup=0;target=slot=None;credit=e.remaining_credit_cents;expiry=e.expires_at
    if action=='refund':
        if w.clock>e.enrolled_at+timedelta(days=c.refund_days):return None,'REFUND_WINDOW_EXPIRED'
        if used_units(w,e.id)*10000>c.total_units*c.refund_consumed_limit_bps:return None,'REFUND_PROGRESS_EXCEEDED'
        if reservation_value(w,e.id):return None,'PENDING_LESSONS'
        if not credit:return None,'NO_REFUNDABLE_CREDIT'
        fee=min(credit,c.refund_fee_cents);refund=credit-fee;credit=0
        if 'min_refund_cents' not in ctx.known_facts:return None,'REQUIRED_INFORMATION_MISSING'
        if refund<ctx.known_facts['min_refund_cents']:return None,'CUSTOMER_CONSTRAINT_UNMET'
    elif action=='transfer':
        if not c.transferable:return None,'TRANSFER_FORBIDDEN'
        if w.clock>=e.expires_at:return None,'COURSE_EXPIRED'
        if reservation_value(w,e.id):return None,'PENDING_LESSONS'
        cl=w.classes.get(p['target_class_id'])
        if not cl or cl.school_id!=e.school_id or cl.course_id not in w.packages[c.package_id].course_ids:return None,'COURSE_NOT_IN_PACKAGE'
        if cl.id==e.class_id:return None,'ALREADY_IN_CLASS'
        if ctx.preferences.get('class_id')!=cl.id:return None,'PREFERENCE_RESOLUTION_REQUIRED'
        target_course=w.courses[cl.course_id]
        if w.users[e.student_id].levels.get(target_course.subject,0)<target_course.level:return None,'ENTRY_LEVEL_NOT_MET'
        if occupancy(w,cl.id,e.id)>=cl.capacity:return None,'CLASS_CAPACITY_EXCEEDED'
        if any(m.start<=w.clock or m.end>e.expires_at for m in cl.meetings):return None,'CLASS_AFTER_EXPIRY'
        if any(student_busy(w,e.student_id,m,e.id) for m in cl.meetings):return None,'SCHEDULE_CONFLICT'
        if any(k not in ctx.known_facts for k in ('available_start_hour','available_end_hour','min_remaining_units')):return None,'REQUIRED_INFORMATION_MISSING'
        if any(m.start.hour<ctx.known_facts['available_start_hour'] or m.end.hour>ctx.known_facts['available_end_hour'] or (m.end.hour==ctx.known_facts['available_end_hour'] and m.end.minute>0) for m in cl.meetings):return None,'CUSTOMER_CONSTRAINT_UNMET'
        topup=p['topup_units']
        if topup and (ctx.preferences.get('allow_topup') is not True or topup>ctx.known_facts.get('max_topup_units',0)):return None,'TOPUP_NOT_AUTHORIZED'
        fee=c.transfer_fee_cents;charge=fee+topup*target_course.unit_price_cents;credit+=topup*target_course.unit_price_cents
        if credit//target_course.unit_price_cents<ctx.known_facts['min_remaining_units']:return None,'STUDY_CREDIT_INSUFFICIENT'
        target=cl.id
    elif action=='extend':
        days=p['days']
        if ctx.preferences.get('extension_days')!=days:return None,'PREFERENCE_RESOLUTION_REQUIRED'
        if case.reason not in c.extension_reasons:return None,'EXTENSION_REASON_INELIGIBLE'
        if e.extended_days+days>c.max_extension_days:return None,'EXTENSION_LIMIT_EXCEEDED'
        if w.clock>e.expires_at+timedelta(days=c.extension_grace_days):return None,'EXTENSION_GRACE_EXPIRED'
        if days>c.self_service_extension_days:
            b=case.review_notice.get('binding',{})
            if b.get('approved') is not True or b.get('days')!=days or b.get('version')!=e.version:return None,'ACADEMIC_APPROVAL_REQUIRED'
        expiry=e.expires_at+timedelta(days=days)
        if 'min_valid_until' not in ctx.known_facts:return None,'REQUIRED_INFORMATION_MISSING'
        if expiry<=w.clock or expiry<datetime.fromisoformat(ctx.known_facts['min_valid_until']):return None,'EXTENSION_TOO_SHORT'
        fee=days*c.extension_fee_per_day_cents;charge=fee
    else:
        if w.clock>=e.expires_at:return None,'COURSE_EXPIRED'
        s=w.slots.get(p['slot_id'])
        if not s or s.school_id!=e.school_id:return None,'LESSON_SCOPE_MISMATCH'
        teacher=w.teachers[s.teacher_id]
        if course.subject not in teacher.subjects or teacher.max_level<course.level:return None,'TEACHER_QUALIFICATION_REQUIRED'
        if s.start<=w.clock or s.end>e.expires_at:return None,'LESSON_OUTSIDE_VALIDITY'
        if (s.end-s.start).total_seconds()<course.unit_minutes*60:return None,'LESSON_TOO_SHORT'
        if teacher_busy(w,s.teacher_id,s) or student_busy(w,e.student_id,s):return None,'SCHEDULE_CONFLICT'
        if ctx.preferences.get('slot_id')!=s.id:return None,'PREFERENCE_RESOLUTION_REQUIRED'
        if any(k not in ctx.known_facts for k in ('earliest_lesson','latest_lesson')):return None,'REQUIRED_INFORMATION_MISSING'
        if s.start<datetime.fromisoformat(ctx.known_facts['earliest_lesson']) or s.end>datetime.fromisoformat(ctx.known_facts['latest_lesson']):return None,'CUSTOMER_CONSTRAINT_UNMET'
        if credit-reservation_value(w,e.id)<course.unit_price_cents:return None,'STUDY_CREDIT_INSUFFICIENT'
        slot=s.id
    if case_spend(w,case.id)+charge>ctx.known_facts['max_budget_cents']:return None,'CUSTOMER_CONSTRAINT_UNMET'
    params={'target_class_id':target,'slot_id':slot,'extension_days':days,'topup_units':topup}
    q=Quote(id=quote_key(e,case,action,params,ctx.revision,w.clock),enrollment_id=e.id,case_id=case.id,action=action,**params,
        charge_cents=charge,refund_cents=refund,fee_cents=fee,resulting_credit_cents=credit,resulting_expiry=expiry,revision=ctx.revision,enrollment_version=e.version,
        expires_at=w.clock+timedelta(hours=1),quote_hash='')
    q.quote_hash=quote_hash(q);return q,None


def decide(w,actor,op):
    deny=lambda code:Decision(allowed=False,error_code=code)
    if invariants(w):return deny('INVALID_WORLD')
    if actor not in w.users:return deny('UNKNOWN_ACTOR')
    if op.scope_id not in w.schools:return deny('SCHOOL_NOT_FOUND')
    if op.name=='get_enrollments':
        if op.resource_id!=op.scope_id:return deny('SCHOOL_SCOPE_MISMATCH')
        from .observations import query_data
        return Decision(allowed=True,expected_state=w.clone(),data=query_data(w,actor,op))
    e=w.enrollments.get(op.resource_id)
    if not e:return deny('ENROLLMENT_NOT_FOUND')
    if e.school_id!=op.scope_id:return deny('SCHOOL_SCOPE_MISMATCH')
    if e.student_id!=actor:return deny('ENROLLMENT_ACCESS_DENIED')
    p=op.params;case=w.cases.get(p.get('case_id',''));q=w.quotes.get(p.get('quote_id',''));a=w.appointments.get(p.get('appointment_id',''))
    if 'case_id' in p and (not case or case.enrollment_id!=e.id):return deny('CASE_SCOPE_MISMATCH')
    if op.name=='commit_quote' and (not q or q.enrollment_id!=e.id):return deny('QUOTE_SCOPE_MISMATCH')
    if op.name=='cancel_lesson' and (not a or a.enrollment_id!=e.id):return deny('LESSON_SCOPE_MISMATCH')
    if op.is_query:
        from .observations import query_data
        return Decision(allowed=True,expected_state=w.clone(),data=query_data(w,actor,op))
    ctx=w.interaction
    if ctx.identity!=actor or ctx.known_facts.get('enrollment_id')!=e.id or 'request_id' not in ctx.known_facts:return deny('REQUIRED_INFORMATION_MISSING')
    if q:case=w.cases[q.case_id]
    if a:case=w.cases[a.case_id]
    if case and case.request_id!=ctx.known_facts['request_id']:return deny('CASE_SCOPE_MISMATCH')
    new=w.clone();data={}
    if op.name=='open_request':
        if ctx.known_facts.get('request_id')!=p['request_id'] or ctx.known_facts.get('reason')!=p['reason']:return deny('REQUIRED_INFORMATION_MISSING')
        cid=case_key(e.id,p['request_id'])
        if cid not in new.cases:new.cases[cid]=Case(id=cid,enrollment_id=e.id,request_id=p['request_id'],requester_id=actor,reason=p['reason'],created_at=w.clock)
        data={'case_id':cid,'reason':new.cases[cid].reason}
    elif op.name=='request_extension_review':
        c=w.contracts[e.contract_id]
        if e.status!='active':return deny('ENROLLMENT_REFUNDED')
        if p['days']!=ctx.preferences.get('extension_days') or e.extended_days+p['days']>c.max_extension_days:return deny('EXTENSION_LIMIT_EXCEEDED')
        if case.pending_extension_days!=p['days'] or case.review_version!=e.version:new.cases[case.id].review_notice={}
        new.cases[case.id].pending_extension_days=p['days'];new.cases[case.id].review_version=e.version;data={'case_id':case.id,'status':'pending'}
    elif op.name=='cancel_lesson':
        if a.status!='scheduled':return deny('LESSON_NOT_SCHEDULED')
        if w.clock>=w.slots[a.slot_id].start:return deny('LESSON_CANCELLATION_CLOSED')
        code=consent_code(ctx,op,digest(a))
        if code:return deny(code)
        new.appointments[a.id].status='cancelled'
        if case.last_appointment_id==a.id:new.cases[case.id].last_lesson_status='cancelled'
        new.enrollments[e.id].version+=1;data={'appointment_id':a.id,'status':'cancelled'}
    elif op.name=='commit_quote':
        if q.status!='offered' or q.enrollment_version!=e.version:return deny('STALE_ENROLLMENT_QUOTE')
        if q.revision!=ctx.revision:return deny('STALE_CUSTOMER_CHOICE')
        if w.clock>q.expires_at:return deny('QUOTE_EXPIRED')
        if p['payment_channel']!=e.payment_channel:return deny('ORIGINAL_PAYMENT_CHANNEL_REQUIRED')
        params={'target_class_id':q.target_class_id,'topup_units':q.topup_units,'days':q.extension_days,'slot_id':q.slot_id}
        expected,code=proposal(w,e,case,q.action,params)
        if code:return deny(code)
        if any(getattr(expected,k)!=getattr(q,k) for k in ('target_class_id','slot_id','extension_days','topup_units','charge_cents','refund_cents','fee_cents','resulting_credit_cents','resulting_expiry')):return deny('QUOTE_TERMS_CHANGED')
        code=consent_code(ctx,op,q.quote_hash)
        if code:return deny(code)
        en=new.enrollments[e.id];en.version+=1;new.quotes[q.id].status='committed'
        en.remaining_credit_cents=q.resulting_credit_cents;en.retained_fees_cents+=q.fee_cents;en.expires_at=q.resulting_expiry
        if q.action=='refund':en.status='refunded'
        elif q.action=='transfer':en.class_id=q.target_class_id;en.course_id=w.classes[q.target_class_id].course_id
        elif q.action=='extend':en.extended_days+=q.extension_days
        else:
            aid='lesson_'+q.id;new.appointments[aid]=Appointment(id=aid,enrollment_id=e.id,case_id=case.id,quote_id=q.id,course_id=e.course_id,slot_id=q.slot_id,unit_price_cents=w.courses[e.course_id].unit_price_cents);new.cases[case.id].last_slot_id=q.slot_id;new.cases[case.id].last_appointment_id=aid;new.cases[case.id].last_lesson_status='scheduled'
        for kind,amount in [('charge',q.charge_cents),('refund',q.refund_cents)]:
            if amount:
                lid=kind+'_'+q.id;b={'enrollment_id':e.id,'case_id':case.id,'quote_id':q.id,'kind':kind,'amount_cents':amount,'channel':e.payment_channel}
                new.ledger[lid]=Ledger(id=lid,enrollment_id=e.id,case_id=case.id,quote_id=q.id,kind=kind,amount_cents=amount,channel=e.payment_channel,notice=signed_notice(w.domain,'education-payment',b,w.clock))
        data={'enrollment_id':e.id,'action':q.action,'status':en.status,'remaining_credit_cents':en.remaining_credit_cents,'charge_cents':q.charge_cents,'refund_cents':q.refund_cents}
    else:
        action={'quote_refund':'refund','quote_transfer':'transfer','quote_extension':'extend','quote_lesson':'tutoring'}[op.name]
        q,code=proposal(w,e,case,action,p)
        if code:return deny(code)
        if q.id not in new.quotes:new.quotes[q.id]=q
        data={'quote':new.quotes[q.id].model_dump(mode='json')}
    return Decision(allowed=True,expected_state=new,data=data)
