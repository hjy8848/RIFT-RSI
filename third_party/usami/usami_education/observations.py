"""Scoped contract/progress facts and public classes, never another student's details."""
from usami_dialogue.process import public_context


def observation(w,actor):return {'domain':w.domain,'identity':{'id':actor,'name':w.users[actor].name},'clock':w.clock.isoformat(),
 'enrollments':[e.model_dump(mode='json') for e in sorted(w.enrollments.values(),key=lambda r:r.id) if e.student_id==actor],'customer_provided':public_context(w.interaction)}
def visible_ids(view):return {r['id'] for values in view.values() if isinstance(values,list) for r in values if isinstance(r,dict) and 'id' in r}

def query_data(w,actor,op):
    if op.name=='get_enrollments':return {'enrollments':[e.model_dump(mode='json') for e in sorted(w.enrollments.values(),key=lambda r:r.id) if e.student_id==actor and e.school_id==op.scope_id]}
    e=w.enrollments[op.resource_id]
    if op.name=='get_enrollment':return {**e.model_dump(mode='json'),'course':w.courses[e.course_id].model_dump(mode='json'),'class':w.classes[e.class_id].model_dump(mode='json'),'contract':w.contracts[e.contract_id].model_dump(mode='json'),'verified_levels':dict(w.users[actor].levels)}
    if op.name=='get_progress':
        rows=[r.model_dump(mode='json',exclude={'notice'}) for r in sorted(w.learning_records.values(),key=lambda r:r.id) if r.enrollment_id==e.id]
        return {'enrollment_id':e.id,'learning_records':rows,'consumed_units':sum(r['units'] for r in rows),'consumed_value_cents':sum(r['units']*r['unit_price_cents'] for r in rows),
            'remaining_credit_cents':e.remaining_credit_cents,'reserved_credit_cents':sum(a.unit_price_cents for a in w.appointments.values() if a.enrollment_id==e.id and a.status=='scheduled')}
    if op.name=='get_catalogue':return {'package':w.packages[w.contracts[e.contract_id].package_id].model_dump(mode='json'),
        'courses':[c.model_dump(mode='json') for c in sorted(w.courses.values(),key=lambda r:r.id) if c.school_id==e.school_id],
        'classes':[{**c.model_dump(mode='json'),'available_seats':c.capacity-sum(x.status=='active' and x.class_id==c.id for x in w.enrollments.values())} for c in sorted(w.classes.values(),key=lambda r:r.id) if c.school_id==e.school_id]}
    if op.name=='get_lesson_slots':return {'teachers':[t.model_dump(mode='json') for t in sorted(w.teachers.values(),key=lambda r:r.id) if t.school_id==e.school_id],
        'slots':[s.model_dump(mode='json') for s in sorted(w.slots.values(),key=lambda r:r.id) if s.school_id==e.school_id],
        'teacher_busy':[{'teacher_id':b.resource_id,'start':b.start.isoformat(),'end':b.end.isoformat()} for b in sorted(w.background_busy.values(),key=lambda r:r.id) if b.kind=='teacher' and w.teachers[b.resource_id].school_id==e.school_id],
        'teaching_schedule':[{'teacher_id':c.teacher_id,'class_id':c.id,'meetings':[m.model_dump(mode='json') for m in c.meetings]} for c in sorted(w.classes.values(),key=lambda r:r.id) if c.school_id==e.school_id],
        'your_class_meetings':[{'class_id':x.class_id,'meetings':[m.model_dump(mode='json') for m in w.classes[x.class_id].meetings]} for x in sorted(w.enrollments.values(),key=lambda r:r.id) if x.student_id==actor and x.status=='active'],
        'your_busy':[b.model_dump(mode='json') for b in sorted(w.background_busy.values(),key=lambda r:r.id) if b.kind=='student' and b.resource_id==actor]}
    c=w.cases[op.params['case_id']]
    return {**c.model_dump(mode='json',exclude={'review_notice'}),'academic_review':c.review_notice.get('binding') or None,
        'quotes':[q.model_dump(mode='json') for q in sorted(w.quotes.values(),key=lambda r:r.id) if q.case_id==c.id],
        'lessons':[a.model_dump(mode='json',exclude={'attendance_notice'})|{'attendance':a.attendance_notice.get('binding') or None} for a in sorted(w.appointments.values(),key=lambda r:r.id) if a.case_id==c.id]}
