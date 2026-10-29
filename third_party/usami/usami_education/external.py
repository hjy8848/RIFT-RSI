"""Frozen academic decisions and actual lesson attendance, not invented progress."""
from usami_harness.models import signed_notice
from .models import LearningRecord


def expected(w,event_id):
    event=w.external_events.get(event_id)
    if not event or event_id in w.applied_events:raise ValueError('unknown or consumed education event')
    case=next((c for c in w.cases.values() if c.enrollment_id==event.enrollment_id and c.request_id==event.request_id),None)
    if not case:raise ValueError('event requires the actual student request')
    new=w.clone();e=new.enrollments[event.enrollment_id];c=new.cases[case.id]
    if event.kind=='academic_review':
        if case.pending_extension_days is None or case.review_version!=e.version:raise ValueError('review request absent or enrollment version changed')
        new.clock=max(w.clock,event.at)
        binding={'case_id':c.id,'enrollment_id':e.id,'days':c.pending_extension_days,'version':c.review_version,'approved':event.outcome=='approved' and c.pending_extension_days<=event.grant_days,'event_id':event.id}
        c.review_notice=signed_notice(w.domain,'education-academic',binding,new.clock)
    else:
        rows=sorted((a for a in w.appointments.values() if a.case_id==case.id and a.status=='scheduled'),key=lambda a:(w.slots[a.slot_id].end,a.id))
        if not rows:raise ValueError('attendance requires an actually booked lesson')
        original=rows[0];a=new.appointments[original.id];slot=w.slots[a.slot_id];new.clock=max(w.clock,slot.end)
        outcome=event.outcome;consume=outcome=='attended' or (outcome=='student_no_show' and w.contracts[e.contract_id].no_show_consumes)
        a.status='attended' if outcome=='attended' else 'missed' if outcome=='student_no_show' else 'cancelled';e.version+=1
        binding={'appointment_id':a.id,'enrollment_id':e.id,'outcome':outcome,'consumed':consume,'at':slot.end.isoformat(),'event_id':event.id}
        a.attendance_notice=signed_notice(w.domain,'education-attendance',binding,new.clock)
        if consume:
            e.remaining_credit_cents-=a.unit_price_cents;rid='record_'+a.id
            record=LearningRecord(id=rid,enrollment_id=e.id,course_id=a.course_id,units=1,unit_price_cents=a.unit_price_cents,origin='attendance',appointment_id=a.id,notice={})
            proof={'record_id':record.id,'enrollment_id':record.enrollment_id,'course_id':record.course_id,'units':1,'unit_price_cents':record.unit_price_cents,'origin':'attendance','appointment_id':a.id}
            record.notice=signed_notice(w.domain,'education-progress',proof,new.clock);new.learning_records[rid]=record
        if c.last_appointment_id==a.id:c.last_lesson_status=a.status
        if outcome=='attended':c.attended_units+=1
    new.applied_events.append(event.id);return new


def apply(w,event_id):
    if event_id not in w.external_events or event_id in w.applied_events:raise ValueError('unknown or consumed education event')
    new=w.clone();event=new.external_events[event_id]
    cases=[c for c in new.cases.values() if c.enrollment_id==event.enrollment_id and c.request_id==event.request_id]
    if len(cases)!=1:raise ValueError('missing or ambiguous student case')
    case=cases[0];enrollment=new.enrollments[event.enrollment_id]
    if event.kind=='academic_review':
        if case.pending_extension_days is None or case.review_version!=enrollment.version:raise ValueError('stale or absent review request')
        new.clock=max(event.at,new.clock);ok=event.outcome=='approved' and case.pending_extension_days<=event.grant_days
        payload={'case_id':case.id,'enrollment_id':enrollment.id,'days':case.pending_extension_days,'version':case.review_version,'approved':ok,'event_id':event_id}
        case.review_notice=signed_notice(new.domain,'education-academic',payload,new.clock)
    else:
        appointments=[a for a in new.appointments.values() if a.case_id==case.id and a.status=='scheduled']
        appointments.sort(key=lambda a:(new.slots[a.slot_id].end,a.id))
        if not appointments:raise ValueError('no booked lesson to attend')
        a=appointments[0];slot=new.slots[a.slot_id];new.clock=max(slot.end,new.clock);enrollment.version+=1
        charged=event.outcome=='attended' or event.outcome=='student_no_show' and new.contracts[enrollment.contract_id].no_show_consumes
        a.status={'attended':'attended','student_no_show':'missed','teacher_cancelled':'cancelled'}[event.outcome]
        details={'appointment_id':a.id,'enrollment_id':enrollment.id,'outcome':event.outcome,'consumed':charged,'at':slot.end.isoformat(),'event_id':event_id}
        a.attendance_notice=signed_notice(new.domain,'education-attendance',details,new.clock)
        if charged:
            enrollment.remaining_credit_cents=enrollment.remaining_credit_cents-a.unit_price_cents
            key='record_'+a.id;proof={'record_id':key,'enrollment_id':enrollment.id,'course_id':a.course_id,'units':1,'unit_price_cents':a.unit_price_cents,'origin':'attendance','appointment_id':a.id}
            new.learning_records[key]=LearningRecord(id=key,enrollment_id=enrollment.id,course_id=a.course_id,units=1,unit_price_cents=a.unit_price_cents,origin='attendance',appointment_id=a.id,notice=signed_notice(new.domain,'education-progress',proof,new.clock))
        if case.last_appointment_id==a.id:case.last_lesson_status=a.status
        if event.outcome=='attended':case.attended_units+=1
    new.applied_events.append(event_id);return new
