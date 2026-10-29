"""Independent school/cohort/contract worlds with factual opening consumption ledgers."""
from datetime import timedelta
from usami.randomness import StableRandom
from usami_harness.models import digest,signed_notice
from usami_benchmark.contracts import GeneratorConfig
from usami_dialogue.models import CustomerPrivateState
from .models import World,User,School,Course,Package,Teacher,Meeting,Cohort,Contract,Enrollment,LearningRecord,Slot,Busy,ExternalEvent
GENERATOR_VERSION='education-generator-v1'
RULES_VERSION='education-0.1.0'
SUBJECTS=['math','language','coding']

def overlap(a,b):return a.start<b.end and b.start<a.end


def generate_world(seed=42,config=None,*,diagnostics=None):
    cfg=GeneratorConfig.model_validate(config or {});rng=StableRandom(digest({'version':GENERATOR_VERSION,'seed':seed,'config':cfg.model_dump(mode='json')}))
    users={f'u{i}':User(id=f'u{i}',name=f'Student {i}',contact=f'private_student_{i}@example.invalid',levels={s:rng.randint(1,3) for s in SUBJECTS}) for i in range(6)}
    schools={f's{i}':School(id=f's{i}',name=f'School {i}') for i in range(rng.randint(cfg.min_scopes,cfg.max_scopes))}
    w=World(seed=seed,users=users,schools=schools,courses={},packages={},teachers={},classes={},contracts={},enrollments={},learning_records={},slots={},background_busy={},profiles={})
    for sid in schools:
        for i in range(rng.randint(2,4)):
            tid=sid+'_teacher'+str(i);w.teachers[tid]=Teacher(id=tid,school_id=sid,name=f'Teacher {i}',subjects=list(SUBJECTS) if not i else rng.sample(SUBJECTS,rng.randint(1,2)),max_level=3 if not i else rng.randint(1,3))
        for i in range(3+rng.randint(1,cfg.max_resources)):
            cid=sid+'_course'+str(i);w.courses[cid]=Course(id=cid,school_id=sid,subject=SUBJECTS[i] if i<3 else rng.choice(SUBJECTS),level=1 if i<3 else rng.randint(1,3),unit_price_cents=rng.choice([500,1000,2000,3000]),unit_minutes=rng.choice([45,60]))
        ids=[c.id for c in w.courses.values() if c.school_id==sid]
        for i in range(2):
            pid=sid+'_package'+str(i);selected=ids[:3]+[c for c in ids[3:] if rng.random()<.6]
            w.packages[pid]=Package(id=pid,school_id=sid,course_ids=selected)
        for i in range(rng.randint(2,4)):
            cid=sid+'_contract'+str(i);maximum=rng.choice([14,30,60])
            w.contracts[cid]=Contract(id=cid,school_id=sid,package_id=rng.choice([p.id for p in w.packages.values() if p.school_id==sid]),total_units=rng.choice([12,24,40]),
                refund_days=rng.choice([7,14,30]),refund_consumed_limit_bps=rng.choice([0,2500,5000,10000]),refund_fee_cents=rng.choice([0,500,1500]),transferable=rng.random()<.8,transfer_fee_cents=rng.choice([0,500,2000]),
                max_extension_days=maximum,self_service_extension_days=min(maximum,rng.choice([7,14])),extension_fee_per_day_cents=rng.choice([0,100,300]),extension_grace_days=rng.choice([7,30]),
                extension_reasons=['too_busy','illness']+(['level_mismatch'] if rng.random()<.3 else []),no_show_consumes=rng.random()<.6)
        teacher_counts={tid:0 for tid,t in w.teachers.items() if t.school_id==sid}
        for course in [w.courses[cid] for cid in ids]:
            qualified=[t for t in w.teachers.values() if t.school_id==sid and course.subject in t.subjects and course.level<=t.max_level]
            for j in range(rng.randint(1,2)):
                teacher=rng.choice(qualified);index=teacher_counts[teacher.id];teacher_counts[teacher.id]+=1
                start=(w.clock+timedelta(days=index*2+1)).replace(hour=9 if index%2==0 else 14,minute=0)
                cid=course.id+'_class'+str(j)
                meetings=[Meeting(id=cid+'_meeting'+str(k),start=start+timedelta(days=k*7),end=start+timedelta(days=k*7,hours=2)) for k in range(2)]
                w.classes[cid]=Cohort(id=cid,school_id=sid,course_id=course.id,teacher_id=teacher.id,capacity=1000,meetings=meetings)
        for teacher in [t for t in w.teachers.values() if t.school_id==sid]:
            for j in range(rng.randint(cfg.min_resources,cfg.max_resources)):
                start=(w.clock+timedelta(days=rng.choice([1,2,4]))).replace(hour=rng.choice([10,13,16,19]),minute=rng.choice([0,30]));key=teacher.id+'_slot'+str(j)
                w.slots[key]=Slot(id=key,school_id=sid,teacher_id=teacher.id,start=start,end=start+timedelta(hours=1))
            if rng.random()<.4:
                start=(w.clock+timedelta(days=2)).replace(hour=13,minute=0);end=start+timedelta(hours=2)
                if not any(c.teacher_id==teacher.id and any(start<m.end and m.start<end for m in c.meetings) for c in w.classes.values()):
                    key='busy_'+teacher.id;w.background_busy[key]=Busy(id=key,kind='teacher',resource_id=teacher.id,start=start,end=end)
    for uid in users:
        if rng.random()<.5:
            start=(w.clock+timedelta(days=rng.choice([1,2,4]))).replace(hour=rng.choice([9,14,19]),minute=0);key='busy_'+uid
            w.background_busy[key]=Busy(id=key,kind='student',resource_id=uid,start=start,end=start+timedelta(hours=2))
        for j in range(rng.randint(1,3)):
            candidates=[]
            for cl in w.classes.values():
                course=w.courses[cl.course_id]
                if course.level>users[uid].levels[course.subject]:continue
                if any(b.kind=='student' and b.resource_id==uid and any(overlap(b,m) for m in cl.meetings) for b in w.background_busy.values()):continue
                existing=[m for e in w.enrollments.values() if e.student_id==uid for m in w.classes[e.class_id].meetings]
                if any(overlap(a,b) for a in cl.meetings for b in existing):continue
                candidates.append(cl)
            if not candidates:break
            cl=rng.choice(candidates);course=w.courses[cl.course_id];contracts=[c for c in w.contracts.values() if c.school_id==cl.school_id and course.id in w.packages[c.package_id].course_ids]
            if not contracts:continue
            contract=rng.choice(contracts);eid=f'{uid}_enrollment{j}';age=rng.choice([1,5,12,30,60]);enrolled=w.clock-timedelta(days=age);initial=contract.total_units*course.unit_price_cents
            used=rng.choice([0,1,contract.total_units//4,contract.total_units//2,contract.total_units]);credit=initial-used*course.unit_price_cents
            w.enrollments[eid]=Enrollment(id=eid,school_id=cl.school_id,student_id=uid,course_id=course.id,original_course_id=course.id,class_id=cl.id,contract_id=contract.id,enrolled_at=enrolled,expires_at=enrolled+timedelta(days=rng.choice([14,45,90])),initial_paid_cents=initial,remaining_credit_cents=credit,payment_channel='original_'+uid)
            if used:
                rid='opening_'+eid;record=LearningRecord(id=rid,enrollment_id=eid,course_id=course.id,units=used,unit_price_cents=course.unit_price_cents,origin='opening',notice={})
                binding={'record_id':rid,'enrollment_id':eid,'course_id':course.id,'units':used,'unit_price_cents':course.unit_price_cents,'origin':'opening','appointment_id':None}
                record.notice=signed_notice(w.domain,'education-progress',binding,w.clock);w.learning_records[rid]=record
            req='request_'+eid;classes=[c.id for c in w.classes.values() if c.school_id==cl.school_id];slots=[s.id for s in w.slots.values() if s.school_id==cl.school_id]
            facts={'enrollment_id':eid,'request_id':req,'reason':rng.choice(['too_busy','illness','level_mismatch','teaching_quality']),
                'initial_expires_at':w.enrollments[eid].expires_at.isoformat(),'max_budget_cents':rng.choice([0,1000,5000,20000]),'min_refund_cents':rng.choice([0,1000]),'min_remaining_units':rng.choice([1,3,6]),'max_topup_units':rng.choice([1,2]),
                'available_start_hour':rng.choice([9,12,14]),'available_end_hour':rng.choice([18,21]),'min_valid_until':(w.clock+timedelta(days=rng.choice([3,10,21]))).isoformat(),
                'earliest_lesson':(w.clock+timedelta(hours=12)).isoformat(),'latest_lesson':(w.clock+timedelta(days=rng.choice([3,5,10]))).isoformat()}
            w.profiles['profile_'+eid]=CustomerPrivateState(identity=uid,opening_request='I am not satisfied with my course and need help.',facts=facts,
                preference_space={'resolution':['refund','transfer','extend','tutoring'],'class_id':classes,'slot_id':slots,'extension_days':[7,14,30],'allow_topup':[False,True],'allow_extension':[False,True]},
                initial_preferences={'resolution':rng.choice(['refund','transfer','extend','tutoring']),'class_id':rng.choice(classes),'slot_id':rng.choice(slots),'extension_days':rng.choice([7,14,30]),'allow_topup':False,'allow_extension':True},
                confirmable_actions=('open_request','request_extension_review','quote_refund','quote_transfer','quote_extension','quote_lesson','commit_quote','cancel_lesson'),protected_objects=(eid,)).checked()
            for kind,outcome in [('academic_review',rng.choice(['approved','approved','denied'])),('attendance',rng.choice(['attended','attended','student_no_show','teacher_cancelled']))]:
                key=kind+'_'+eid;w.external_events[key]=ExternalEvent(id=key,enrollment_id=eid,request_id=req,kind=kind,outcome=outcome,at=w.clock+timedelta(hours=1) if kind=='academic_review' else None,grant_days=rng.choice([7,14,30]))
    for cl in w.classes.values():
        used=sum(e.class_id==cl.id and e.status=='active' for e in w.enrollments.values());cl.capacity=max(1,used+rng.choice([0,1,2,5]))
    from .rules import invariants
    errors=invariants(w)
    if errors:raise ValueError(str(errors))
    return w
