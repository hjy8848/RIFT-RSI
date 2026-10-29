"""Student facts and current remedy/offer fit; no hidden legality filter."""
from datetime import datetime
from usami_harness.models import digest
from .models import Operation


def private_state(w,pid):return w.profiles[pid].checked()
def validate_private(w,private):
    private=private.checked()
    if not any(p==private for p in w.profiles.values()):raise ValueError('student profile is not independently frozen')
    return private

def intent_allowed(w,p,raw):
    op=Operation.model_validate(raw);e=w.enrollments.get(op.resource_id)
    if op.is_query or not e or e.id not in p.protected_objects or e.school_id!=op.scope_id or e.student_id!=p.identity or op.name not in p.confirmable_actions:return False
    a=op.params
    if op.name=='open_request':return a=={'request_id':p.facts['request_id'],'reason':p.facts['reason']}
    if 'quote_id' in a:
        q=w.quotes.get(a['quote_id']);case=w.cases.get(q.case_id) if q else None
    elif 'appointment_id' in a:
        lesson=w.appointments.get(a['appointment_id']);case=w.cases.get(lesson.case_id) if lesson else None
    else:case=w.cases.get(a.get('case_id',''))
    if not case or case.enrollment_id!=e.id or case.request_id!=p.facts['request_id']:return False
    if op.name=='quote_transfer':return a['target_class_id'] in p.preference_space['class_id'] and a['topup_units']<=p.facts['max_topup_units']
    if op.name=='quote_lesson':return a['slot_id'] in p.preference_space['slot_id']
    if op.name in {'quote_extension','request_extension_review'}:return a['days'] in p.preference_space['extension_days']
    return True

def normalize_intent(p,intent):
    from usami_dialogue.models import CustomerIntent
    data=intent.model_dump(mode='json');facts=data['disclosures']
    for op in data['requested_operations']+([data['confirm_operation']] if data['confirm_operation'] else []):
        if op.get('resource_id')==p.facts['enrollment_id']:facts.setdefault('enrollment_id',p.facts['enrollment_id'])
        for key in ('request_id','reason'):
            if op.get('params',{}).get(key)==p.facts[key]:facts.setdefault(key,p.facts[key])
    data['disclosures']=facts;return CustomerIntent.model_validate(data)

def offer_for(w,raw):
    if raw is None:return None
    op=Operation.model_validate(raw)
    if op.name=='cancel_lesson':
        a=w.appointments.get(op.params['appointment_id'])
        if not a or a.enrollment_id!=op.resource_id:return None
        return {'operation':op.model_dump(mode='json'),'quote_hash':digest(a),'world_hash':w.state_hash,'revision':w.interaction.revision,'action':'cancel','commitments':{'appointment_id':a.id,'cancellation_fee_cents':0}}
    if op.name!='commit_quote':return None
    q=w.quotes.get(op.params['quote_id'])
    if not q or q.enrollment_id!=op.resource_id:return None
    terms={'action':q.action,'class_id':q.target_class_id,'slot_id':q.slot_id,'extension_days':q.extension_days,'topup_units':q.topup_units,
        'charge_cents':q.charge_cents,'refund_cents':q.refund_cents,'case_spend_cents':sum(x.amount_cents for x in w.ledger.values() if x.case_id==q.case_id and x.kind=='charge'),
        'resulting_credit_cents':q.resulting_credit_cents,'resulting_expiry':q.resulting_expiry.isoformat()}
    if q.slot_id:terms.update(start=w.slots[q.slot_id].start.isoformat(),end=w.slots[q.slot_id].end.isoformat())
    return {'operation':op.model_dump(mode='json'),'quote_hash':q.quote_hash,'world_hash':w.state_hash,'revision':w.interaction.revision,'commitments':terms,**terms}

def consent_allowed(private,ctx,raw,offer):
    if raw['resource_id'] not in private.protected_objects:return False
    if offer['action']=='cancel':return True
    action=offer['action'];resolution=ctx.preferences.get('resolution')
    if resolution!=action and not(action=='extend' and resolution in {'transfer','tutoring'} and ctx.preferences.get('allow_extension') is True):return False
    if offer['case_spend_cents']+offer['charge_cents']>private.facts['max_budget_cents']:return False
    if action=='refund':return offer['refund_cents']>=private.facts['min_refund_cents']
    if action=='extend':return offer['extension_days']==ctx.preferences.get('extension_days') and datetime.fromisoformat(offer['resulting_expiry'])>=datetime.fromisoformat(private.facts['min_valid_until'])
    if action=='transfer':return offer['class_id']==ctx.preferences.get('class_id')
    return offer['slot_id']==ctx.preferences.get('slot_id') and datetime.fromisoformat(offer['start'])>=datetime.fromisoformat(private.facts['earliest_lesson']) and datetime.fromisoformat(offer['end'])<=datetime.fromisoformat(private.facts['latest_lesson'])

def candidate_intents(w,p):
    e=w.enrollments[p.facts['enrollment_id']];base={'scope_id':e.school_id,'resource_id':e.id};items=[('open_request',{'request_id':p.facts['request_id'],'reason':p.facts['reason']})]
    for c in w.cases.values():
        if c.enrollment_id!=e.id or c.request_id!=p.facts['request_id']:continue
        items.append(('quote_refund',{'case_id':c.id}))
        items += [(name,{'case_id':c.id,'days':d}) for name in ('request_extension_review','quote_extension') for d in p.preference_space['extension_days']]
        items += [('quote_transfer',{'case_id':c.id,'target_class_id':cid,'topup_units':n}) for cid in p.preference_space['class_id'] for n in sorted({0,p.facts['max_topup_units']})]
        items += [('quote_lesson',{'case_id':c.id,'slot_id':sid}) for sid in p.preference_space['slot_id']]
    items += [('commit_quote',{'quote_id':q.id,'payment_channel':e.payment_channel}) for q in w.quotes.values() if q.enrollment_id==e.id and w.cases[q.case_id].request_id==p.facts['request_id']]
    items += [('cancel_lesson',{'appointment_id':a.id}) for a in w.appointments.values() if a.enrollment_id==e.id and w.cases[a.case_id].request_id==p.facts['request_id']]
    return sorted([Operation(name=n,**base,params=a).model_dump(mode='json') for n,a in items],key=digest)

def external_ready(w,event_id):
    event=w.external_events[event_id]
    if event_id in w.applied_events:return False
    c=next((c for c in w.cases.values() if c.enrollment_id==event.enrollment_id and c.request_id==event.request_id),None)
    if not c:return False
    if event.kind=='academic_review':return c.pending_extension_days is not None and c.review_version==w.enrollments[event.enrollment_id].version
    return any(a.case_id==c.id and a.status=='scheduled' for a in w.appointments.values())
