"""Goal-based education remedies; no required Service tool path is scored."""
from datetime import timedelta
from usami_dialogue.tasks import DialogueTask,Goal,FrameRule,RefusalGoal,checked_task
from .customer import private_state
from .rules import case_key
from .models import Operation


def frames(w,p,refusal_only=False):
    eid=p.facts['enrollment_id'];cid=case_key(eid,p.facts['request_id'])
    rows=[FrameRule(collection='cases',selector_value=cid,mutable_fields=['pending_extension_days','review_version','review_notice','last_slot_id','last_appointment_id','last_lesson_status','attended_units'],allow_create=True)]
    if not refusal_only:rows += [FrameRule(collection='enrollments',selector_value=eid,mutable_fields=['course_id','class_id','status','expires_at','remaining_credit_cents','retained_fees_cents','extended_days','version']),
        FrameRule(collection='quotes',selector_field='case_id',selector_value=cid,mutable_fields=['status'],allow_create=True),
        FrameRule(collection='appointments',selector_field='case_id',selector_value=cid,mutable_fields=['status','attendance_notice'],allow_create=True),
        FrameRule(collection='ledger',selector_field='case_id',selector_value=cid,allow_create=True),
        FrameRule(collection='learning_records',selector_field='enrollment_id',selector_value=eid,allow_create=True)]
    return rows


def compose(w,pid,*,family='refund',class_ids=None,slot_ids=None,extension_days=None,mixed=False,academic_review=True,refusal_operation=None,refusal_code=None):
    from . import adapter
    p=private_state(w,pid);e=w.enrollments[p.facts['enrollment_id']];cid=case_key(e.id,p.facts['request_id']);pure=family.startswith('refuse_');refusals=[];goals=[]
    if pure:
        if mixed or refusal_operation is None or refusal_code is None:raise ValueError('refusal needs one exact native request and reason')
        refusals=[RefusalGoal(operation=refusal_operation,code=refusal_code)]
    elif family=='refund':goals=[Goal(collection='enrollments',entity=e.id,field='status',accepted_values=['refunded'])]
    elif family=='transfer':
        targets=list(class_ids) if class_ids is not None else [x for x in p.preference_space['class_id'] if x!=e.class_id]
        goals=[Goal(collection='enrollments',entity=e.id,field='class_id',accepted_values=targets)]
    elif family=='extend':
        days=list(extension_days) if extension_days is not None else [p.initial_preferences['extension_days']]
        goals=[Goal(collection='enrollments',entity=e.id,field='expires_at',accepted_values=[(e.expires_at+timedelta(days=d)).isoformat().replace('+00:00','Z') for d in days])]
    elif family in {'tutoring','attend_lesson'}:
        targets=list(slot_ids) if slot_ids is not None else [p.initial_preferences['slot_id']]
        goals=[Goal(collection='cases',entity=cid,field='last_slot_id',accepted_values=targets),Goal(collection='cases',entity=cid,field='last_lesson_status',accepted_values=['attended' if family=='attend_lesson' else 'scheduled'])]
        if family=='attend_lesson':goals.append(Goal(collection='cases',entity=cid,field='attended_units',accepted_values=[1]))
    else:raise ValueError('unregistered education family')
    if mixed:
        if family!='refund':raise ValueError('only refund has an unconditional duplicate-remedy refusal')
        probe=Operation(name='quote_refund',scope_id=e.school_id,resource_id=e.id,params={'case_id':cid});refusals=[RefusalGoal(operation=probe.model_dump(mode='json'),code='ENROLLMENT_REFUNDED')]
    scheduled=[]
    if academic_review and family in {'extend','transfer','tutoring','attend_lesson','refuse_extension'}:scheduled.append('academic_review_'+e.id)
    if family=='attend_lesson':scheduled.append('attendance_'+e.id)
    task=DialogueTask(domain='education',rules_version=w.version,world_hash=w.state_hash,profile_id=pid,private_hash=p.private_hash,family=family,goals=goals,refusals=refusals,
        frames=frames(w,p,refusal_only=pure),external_schedule=scheduled)
    return checked_task(adapter,w,task)


def validate_task(w,p,t):
    e=w.enrollments[p.facts['enrollment_id']];cid=case_key(e.id,p.facts['request_id']);pure=t.family.startswith('refuse_')
    if t.family not in {'refund','transfer','extend','tutoring','attend_lesson','refuse_refund','refuse_transfer','refuse_extension','refuse_lesson'}:raise ValueError('unregistered education family')
    if t.frames!=frames(w,p,refusal_only=pure):raise ValueError('task frame exceeds native student request')
    if any(w.external_events[x].enrollment_id!=e.id or w.external_events[x].request_id!=p.facts['request_id'] for x in t.external_schedule if x in w.external_events):raise ValueError('event outside this frozen request')
    if not t.require_dependency:raise ValueError('education tasks require actual information/consent dependency')
    if pure:
        if t.goals or len(t.refusals)!=1:raise ValueError('native refusal requires one probe')
        r=t.refusals[0];op=Operation.model_validate(r.operation)
        expected={'refuse_refund':'quote_refund','refuse_transfer':'quote_transfer','refuse_extension':'quote_extension','refuse_lesson':'quote_lesson'}[t.family]
        if op.name!=expected or op.scope_id!=e.school_id or op.resource_id!=e.id or op.params.get('case_id')!=cid:raise ValueError('refusal scope differs')
        if op.name=='quote_transfer' and (op.params['target_class_id'] not in p.preference_space['class_id'] or op.params['topup_units']>p.facts['max_topup_units']):raise ValueError('refusal class/topup outside student scope')
        if op.name=='quote_lesson' and op.params['slot_id'] not in p.preference_space['slot_id']:raise ValueError('refusal lesson outside student scope')
        if op.name=='quote_extension' and op.params['days'] not in p.preference_space['extension_days']:raise ValueError('refusal extension outside student scope')
        allowed={'REFUND_WINDOW_EXPIRED','REFUND_PROGRESS_EXCEEDED','NO_REFUNDABLE_CREDIT','TRANSFER_FORBIDDEN','COURSE_EXPIRED','COURSE_NOT_IN_PACKAGE','ENTRY_LEVEL_NOT_MET','CLASS_CAPACITY_EXCEEDED','CLASS_AFTER_EXPIRY','SCHEDULE_CONFLICT','CUSTOMER_CONSTRAINT_UNMET','STUDY_CREDIT_INSUFFICIENT','EXTENSION_REASON_INELIGIBLE','EXTENSION_LIMIT_EXCEEDED','EXTENSION_GRACE_EXPIRED','EXTENSION_TOO_SHORT','TEACHER_QUALIFICATION_REQUIRED','LESSON_OUTSIDE_VALIDITY','LESSON_TOO_SHORT'}
        if r.code not in allowed:raise ValueError('unregistered native refusal boundary')
        return
    if t.family=='refund':expected=[Goal(collection='enrollments',entity=e.id,field='status',accepted_values=['refunded'])]
    elif t.family=='transfer':
        if len(t.goals)!=1:raise ValueError('one native course-transfer goal required')
        values=t.goals[0].accepted_values
        if not values or e.class_id in values or set(values)-set(p.preference_space['class_id']):raise ValueError('transfer goals outside actual alternative classes')
        expected=[Goal(collection='enrollments',entity=e.id,field='class_id',accepted_values=values)]
    elif t.family=='extend':
        if len(t.goals)!=1:raise ValueError('one native expiry goal required')
        allowed={(e.expires_at+timedelta(days=d)).isoformat().replace('+00:00','Z') for d in p.preference_space['extension_days']};values=t.goals[0].accepted_values
        if not values or set(values)-allowed:raise ValueError('expiry goal outside actual extension choices')
        expected=[Goal(collection='enrollments',entity=e.id,field='expires_at',accepted_values=values)]
    else:
        if not t.goals:raise ValueError('a real lesson choice is required')
        values=t.goals[0].accepted_values
        if not values or set(values)-set(p.preference_space['slot_id']):raise ValueError('lesson goal outside actual slot choices')
        expected=[Goal(collection='cases',entity=cid,field='last_slot_id',accepted_values=values),Goal(collection='cases',entity=cid,field='last_lesson_status',accepted_values=['attended' if t.family=='attend_lesson' else 'scheduled'])]
        if t.family=='attend_lesson':expected.append(Goal(collection='cases',entity=cid,field='attended_units',accepted_values=[1]))
    if t.goals!=expected:raise ValueError('education goal differs from domain contract')
    if t.refusals:
        op=Operation(name='quote_refund',scope_id=e.school_id,resource_id=e.id,params={'case_id':cid})
        if t.family!='refund' or t.refusals!=[RefusalGoal(operation=op.model_dump(mode='json'),code='ENROLLMENT_REFUNDED')]:raise ValueError('unregistered mixed education refusal')


def preference_options(w,p,t):
    e=w.enrollments[p.facts['enrollment_id']];resolution=t.family.removeprefix('refuse_');resolution='tutoring' if resolution in {'lesson','attend_lesson'} else 'extend' if resolution=='extension' else resolution
    classes=t.goals[0].accepted_values if t.family=='transfer' else [t.refusals[0].operation['params']['target_class_id']] if t.family=='refuse_transfer' else [p.initial_preferences['class_id']]
    slots=t.goals[0].accepted_values if t.family in {'tutoring','attend_lesson'} else [t.refusals[0].operation['params']['slot_id']] if t.family=='refuse_lesson' else [p.initial_preferences['slot_id']]
    days=[d for d in p.preference_space['extension_days'] if (e.expires_at+timedelta(days=d)).isoformat().replace('+00:00','Z') in t.goals[0].accepted_values] if t.family=='extend' else [t.refusals[0].operation['params']['days']] if t.family=='refuse_extension' else [p.initial_preferences['extension_days']]
    return [{'resolution':resolution,'class_id':cid,'slot_id':sid,'extension_days':d,'allow_topup':True,'allow_extension':True} for cid in classes for sid in slots for d in days]


def search(w,t,*,max_nodes=1024,max_depth=12,max_paths=2):
    from . import adapter
    from usami_dialogue.planner import search as native_search
    return native_search(adapter,w,t,preference_options(w,private_state(w,t.profile_id),t),max_nodes=max_nodes,max_depth=max_depth,max_paths=max_paths)


def certified_tasks(w,pid,*,max_nodes=1024,max_paths=2,**kwargs):
    from . import adapter
    from usami_dialogue.planner import certified_paths
    task=compose(w,pid,**kwargs);result=search(w,task,max_nodes=max_nodes,max_paths=max_paths)
    return task,result,certified_paths(adapter,w,task,result)


def customer_task_brief(w,p,t):
    import json
    prefix='Your enrollment is '+p.facts['enrollment_id']+'. Clarify your actual reason and consider your frozen budget and time needs. '
    if t.family.startswith('refuse_'):
        op=t.refusals[0].operation;params={k:v for k,v in op['params'].items() if k!='case_id' or v in w.cases}
        return prefix+'You want this particular service: '+json.dumps({'operation':op['name'],'parameters':params},sort_keys=True)+'. Respond to real explanations without inventing progress, approval or a different goal.'
    if t.family=='refund':text='Seek a refund based on the actual consumed/unused study value and contract.'
    elif t.family=='transfer':text='Seek a course change to one of these actual classes: '+json.dumps(t.goals[0].accepted_values)+'. You may agree to a necessary legitimate extension or top-up within your frozen constraints.'
    elif t.family=='extend':text='Seek study validity extension to one of these dates: '+json.dumps(t.goals[0].accepted_values)+'.'
    else:text='Arrange one real teaching lesson in one of these slots: '+json.dumps(t.goals[0].accepted_values)+'.'+(' Check the actual attendance record afterward.' if t.family=='attend_lesson' else '')
    if t.refusals:text+=' You also want to request another refund after it has already been issued.'
    return prefix+text


def task_intent_allowed(p,t,intent):
    # Scope only: no Reference decision, world truth or hidden expected outcome.
    desired=t.family.removeprefix('refuse_');desired='tutoring' if desired in {'lesson','attend_lesson'} else 'extend' if desired=='extension' else desired
    if 'resolution' in intent.preferences and intent.preferences['resolution']!=desired:return False
    if t.family=='transfer' and 'class_id' in intent.preferences and intent.preferences['class_id'] not in t.goals[0].accepted_values:return False
    if t.family in {'tutoring','attend_lesson'} and 'slot_id' in intent.preferences and intent.preferences['slot_id'] not in t.goals[0].accepted_values:return False
    if t.family=='extend' and 'extension_days' in intent.preferences:
        from datetime import datetime
        resulting=(datetime.fromisoformat(p.facts['initial_expires_at'])+timedelta(days=intent.preferences['extension_days'])).isoformat().replace('+00:00','Z')
        if resulting not in t.goals[0].accepted_values:return False
    mapping={'quote_refund':'refund','quote_transfer':'transfer','quote_extension':'extend','quote_lesson':'tutoring'}
    for op in intent.requested_operations:
        kind=mapping.get(op.get('name'))
        if kind and kind!=desired and not(kind=='extend' and desired in {'transfer','tutoring'}):return False
    return True
