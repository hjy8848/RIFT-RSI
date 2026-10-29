"""Frozen resident facts and ordinary quote/appointment consent."""
from datetime import datetime
from usami_harness.models import digest
from .models import Operation


def private_state(world,profile_id):return world.profiles[profile_id].checked()
def validate_private(world,private):
    private=private.checked()
    if not any(p==private for p in world.profiles.values()):raise ValueError('resident profile differs from frozen source')
    return private

def intent_allowed(world,private,operation):
    op=Operation.model_validate(operation);u=world.units.get(op.resource_id)
    if op.is_query or not u or u.id not in private.protected_objects or u.property_id!=op.scope_id or op.name not in private.confirmable_actions:return False
    p=op.params
    if op.name=='quote_visit':
        slot=world.slots.get(p['slot_id']);tech=world.technicians.get(p['technician_id'])
        if not slot or slot.id not in private.preference_space['slot_id'] or not tech or tech.property_id!=u.property_id:return False
    if op.name=='open_ticket':return p['asset_id']==private.facts['asset_id'] and p['symptom']==private.facts['symptom']
    if 'plan_id' in p:
        q=world.plans.get(p['plan_id'])
        if not q or q.requester_id!=private.identity:return False
        t=world.tickets.get(q.ticket_id)
    elif 'complaint_id' in p:
        c=world.complaints.get(p['complaint_id'])
        if not c or c.reporter_id!=private.identity:return False
        t=world.tickets.get(c.ticket_id)
    else:t=world.tickets.get(p.get('ticket_id',''))
    return bool(t and t.asset_id==private.facts['asset_id'])

def normalize_intent(private,intent):
    from usami_dialogue.models import CustomerIntent
    data=intent.model_dump(mode='json');facts=data['disclosures']
    for op in data['requested_operations']+([data['confirm_operation']] if data['confirm_operation'] else []):
        if op.get('resource_id')==private.facts['unit_id']:facts.setdefault('unit_id',private.facts['unit_id'])
        for key in ('asset_id','symptom'):
            if op.get('params',{}).get(key)==private.facts.get(key):facts.setdefault(key,private.facts[key])
    return CustomerIntent.model_validate(data)

def offer_for(world,operation):
    if operation is None:return None
    op=Operation.model_validate(operation);p=op.params;ctx=world.interaction
    if op.name=='book_visit':
        q=world.plans.get(p['plan_id'])
        if not q:return None
        t=world.tickets[q.ticket_id];slot=world.slots[q.slot_id]
        terms={'kind':'appointment','slot_id':q.slot_id,'start':slot.start.isoformat(),'end':slot.end.isoformat(),'charge_cents':q.customer_charge_cents,'total_cost_cents':q.cost_cents}
        binding=q.quote_hash
    elif op.name in {'cancel_visit','close_ticket'}:
        t=world.tickets.get(p['ticket_id'])
        if not t:return None
        if op.name=='close_ticket':
            if not t.repair_notice:return None
            binding=digest({'ticket_id':t.id,'repair_signature':t.repair_notice.get('signature'),'ticket_version':t.version});terms={'kind':'closure','visit_result':t.repair_notice.get('binding',{}).get('outcome')}
        else:
            a=world.appointments.get(t.appointment_id or '')
            if not a:return None
            binding=digest(a);terms={'kind':'cancellation','slot_id':a.slot_id,'cancellation_fee_cents':0}
    else:return None
    if t.unit_id!=op.resource_id:return None
    return {'operation':op.model_dump(mode='json'),'quote_hash':binding,'world_hash':world.state_hash,'revision':ctx.revision,
        'commitments':{'ticket_id':t.id,**terms},**terms}

def consent_allowed(private,context,operation,offer):
    if operation['resource_id'] not in private.protected_objects:return False
    if offer['kind']=='closure':return context.preferences.get('feedback')=='satisfied'
    if offer['kind']=='cancellation':return True
    return (offer['slot_id']==context.preferences.get('slot_id') and offer['charge_cents']<=private.facts['max_budget_cents'] and
        datetime.fromisoformat(offer['start'])>=datetime.fromisoformat(private.facts['earliest_access']) and datetime.fromisoformat(offer['end'])<=datetime.fromisoformat(private.facts['latest_access']))

def candidate_intents(world,private):
    unit=world.units[private.facts['unit_id']];base={'scope_id':unit.property_id,'resource_id':unit.id};asset=private.facts['asset_id']
    inputs=[('open_ticket',{'asset_id':asset,'symptom':private.facts['symptom']})]
    for t in world.tickets.values():
        if t.asset_id!=asset:continue
        inputs += [('triage_ticket',{'ticket_id':t.id,'severity':s}) for s in ('normal','urgent','emergency')]
        inputs += [(name,{'ticket_id':t.id}) for name in ('request_owner_review','request_inspection','cancel_visit','close_ticket')]
        inputs += [('quote_visit',{'ticket_id':t.id,'technician_id':world.slots[sid].technician_id,'slot_id':sid}) for sid in private.preference_space['slot_id']]
        inputs += [('file_complaint',{'ticket_id':t.id,'reason':r}) for r in ('unresolved_repair','missed_visit','service_quality')]
    inputs += [('book_visit',{'plan_id':q.id}) for q in world.plans.values() if q.requester_id==private.identity and world.tickets[q.ticket_id].asset_id==asset]
    inputs += [('escalate_complaint',{'complaint_id':c.id}) for c in world.complaints.values() if c.reporter_id==private.identity and world.tickets[c.ticket_id].asset_id==asset]
    return sorted([Operation(name=n,**base,params=p).model_dump(mode='json') for n,p in inputs],key=digest)

def external_ready(world,event_id):
    e=world.external_events[event_id]
    if event_id in world.applied_events:return False
    t=next((t for t in world.tickets.values() if t.asset_id==e.asset_id),None)
    if not t:return False
    if e.kind=='owner_review':return t.approval_requested
    if e.kind=='inspection':return t.inspection_requested and t.status not in {'work_verified','closed'}
    if e.kind=='technician_visit':
        a=world.appointments.get(t.appointment_id or '')
        return bool(a and a.status=='scheduled' and world.plans[a.plan_id].asset_revision==world.assets[t.asset_id].revision)
    return any(c.ticket_id==t.id and c.status=='escalated' for c in world.complaints.values())
