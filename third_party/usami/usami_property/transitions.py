"""Actual property mutations. No Reference or policy helper is imported here."""
from datetime import timedelta
from usami_harness.models import Result,digest,signed_notice
from .models import Ticket,Plan,Appointment,Complaint


def _mutate(world,actor,op):
    new=world.clone();p=op.params
    if op.is_query:
        from .observations import query_data
        return new,query_data(world,actor,op)
    unit=new.units[op.resource_id];t=new.tickets.get(p.get('ticket_id',''));ctx=new.interaction
    if op.name=='open_ticket':
        asset=new.assets[p['asset_id']];key='ticket_'+digest(asset.id)[:18]
        if key not in new.tickets:new.tickets[key]=Ticket(id=key,unit_id=unit.id,asset_id=asset.id,reporter_id=actor,symptom=p['symptom'],created_at=new.clock)
        return new,{'ticket_id':key,'status':new.tickets[key].status}
    if op.name=='triage_ticket':
        asset=new.assets[t.asset_id]
        if t.assessment_version!=asset.revision or t.severity!=p['severity']:t.version+=1
        t.severity=p['severity'];t.assessment_version=asset.revision
        if t.status=='open':t.status='triaged'
        return new,{'ticket_id':t.id,'severity':t.severity,'assessment_version':asset.revision}
    if op.name in {'request_owner_review','request_inspection'}:
        setattr(t,'approval_requested' if op.name=='request_owner_review' else 'inspection_requested',True)
        return new,{'ticket_id':t.id,'status':'pending'}
    if op.name=='quote_visit':
        asset=new.assets[t.asset_id];c=next(c for c in new.contracts.values() if c.property_id==unit.property_id)
        price=max(0,asset.repair_cost_cents-(c.coverage_cents if asset.category in c.covered_categories else 0))
        payer=unit.owner_id if asset.responsibility=='owner' else actor if asset.responsibility=='resident' else unit.property_id
        key='plan_'+digest([t.id,t.version,asset.revision,p['technician_id'],p['slot_id'],ctx.revision,new.clock.isoformat()])[:20]
        if key not in new.plans:
            q=Plan(id=key,ticket_id=t.id,technician_id=p['technician_id'],slot_id=p['slot_id'],requester_id=actor,cost_cents=price,customer_charge_cents=price if payer==actor else 0,payer_id=payer,
                revision=ctx.revision,ticket_version=t.version,asset_revision=asset.revision,expires_at=new.clock+timedelta(hours=1),quote_hash='')
            q.quote_hash=digest(q.model_dump(mode='json',exclude={'quote_hash','status'}));new.plans[key]=q
        return new,{'plan':new.plans[key].model_dump(mode='json')}
    if op.name=='book_visit':
        q=new.plans[p['plan_id']];t=new.tickets[q.ticket_id];key='appointment_'+q.id
        new.appointments[key]=Appointment(id=key,ticket_id=t.id,plan_id=q.id,technician_id=q.technician_id,slot_id=q.slot_id)
        t.appointment_id=key;t.booked_slot_id=q.slot_id;t.status='scheduled';t.version+=1;q.status='committed'
        return new,{'ticket_id':t.id,'appointment_id':key,'status':'scheduled'}
    if op.name=='cancel_visit':
        appointment=new.appointments[t.appointment_id];appointment.status='cancelled';t.status='triaged';t.version+=1
        return new,{'ticket_id':t.id,'status':'cancelled'}
    if op.name=='close_ticket':
        consent=next((c for c in ctx.consents if c.operation==op.model_dump(mode='json') and c.revision==ctx.revision),None)
        binding={'ticket_id':t.id,'repair_signature':t.repair_notice.get('signature'),'consent_hash':digest(consent) if consent else None,'actor':actor}
        t.closure_notice=signed_notice(new.domain,'property-closure',binding,new.clock);t.status='closed';t.version+=1
        return new,{'ticket_id':t.id,'status':'closed'}
    if op.name=='file_complaint':
        key='complaint_'+digest([t.id,p['reason'],actor])[:18]
        if key not in new.complaints:new.complaints[key]=Complaint(id=key,ticket_id=t.id,reporter_id=actor,reason=p['reason'])
        return new,{'complaint_id':key,'status':new.complaints[key].status}
    if op.name=='escalate_complaint':
        complaint=new.complaints[p['complaint_id']];complaint.status='escalated'
        return new,{'complaint_id':complaint.id,'status':'escalated'}
    raise ValueError('unknown native mutation')


def apply(world,actor,op,event_index=0):
    try:
        new,data=_mutate(world,actor,op);return new,Result(success=True,data=data)
    except (KeyError,ValueError,TypeError,AttributeError):return world.clone(),Result(success=False,error_code='MUTATION_NOT_REPRESENTABLE')
