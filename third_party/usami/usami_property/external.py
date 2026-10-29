"""Frozen owner/inspector/technician outcomes with independent expected and actual updates."""
from usami_harness.models import signed_notice
from .models import Charge


def _target(world,e):
    ticket=next((t for t in world.tickets.values() if t.asset_id==e.asset_id),None)
    if not ticket:raise ValueError('external event requires a real ticket')
    return ticket


def expected(world,event_id):
    e=world.external_events.get(event_id)
    if not e or event_id in world.applied_events:raise ValueError('unknown or consumed property event')
    t=_target(world,e);new=world.clone();asset=new.assets[e.asset_id];ticket=new.tickets[t.id]
    if e.timing=='fixed':new.clock=max(world.clock,e.at)
    if e.kind=='owner_review':
        if not t.approval_requested:raise ValueError('owner review not requested')
        binding={'asset_id':asset.id,'approved':e.outcome=='approved','event_id':e.id}
        new.approvals[asset.id]={'approved':binding['approved'],'notice':signed_notice(world.domain,'property-owner',binding,new.clock)}
    elif e.kind=='inspection':
        if not t.inspection_requested or t.status in {'work_verified','closed'}:raise ValueError('inspection not pending for unresolved work')
        asset.revision+=1;asset.simulated_hazard=e.outcome=='hazard'
        asset.inspection_notice=signed_notice(world.domain,'property-inspection',{'asset_id':asset.id,'hazard':asset.simulated_hazard,'event_id':e.id},new.clock)
    elif e.kind=='technician_visit':
        a=world.appointments.get(t.appointment_id or '')
        if not a or a.status!='scheduled':raise ValueError('technician cannot visit an unscheduled appointment')
        if world.plans[a.plan_id].asset_revision!=world.assets[t.asset_id].revision:raise ValueError('physical work order is superseded by a new inspection')
        slot=world.slots[a.slot_id];new.clock=max(world.clock,slot.end)
        binding={'ticket_id':t.id,'appointment_id':a.id,'outcome':e.outcome,'visited_at':slot.end.isoformat(),'event_id':e.id}
        ticket.repair_notice=signed_notice(world.domain,'property-technician',binding,new.clock)
        new.appointments[a.id].status='completed' if e.outcome=='repaired' else 'failed';ticket.status='work_verified' if e.outcome=='repaired' else 'awaiting_repair';ticket.version+=1
        if e.outcome=='repaired':
            asset.condition='working';q=world.plans[a.plan_id]
            if q.cost_cents:
                cid='charge_'+q.id;binding={'ticket_id':t.id,'plan_id':q.id,'payer_id':q.payer_id,'amount_cents':q.cost_cents}
                new.charges[cid]=Charge(id=cid,ticket_id=t.id,plan_id=q.id,payer_id=q.payer_id,amount_cents=q.cost_cents,notice=signed_notice(world.domain,'property-billing',binding,new.clock))
    else:
        rows=[c for c in world.complaints.values() if c.ticket_id==t.id and c.status=='escalated']
        if not rows:raise ValueError('no actual escalated complaint')
        for c in rows:
            new.complaints[c.id].status=e.outcome;binding={'complaint_id':c.id,'outcome':e.outcome,'event_id':e.id}
            new.complaints[c.id].review_notice=signed_notice(world.domain,'property-manager',binding,new.clock)
    new.applied_events.append(e.id);return new


def apply(world,event_id):
    # Separate simulator mutation implementation. Reference never calls this.
    if event_id not in world.external_events or event_id in world.applied_events:raise ValueError('unknown or consumed property event')
    new=world.clone();e=new.external_events[event_id];ticket=_target(new,e);asset=new.assets[e.asset_id]
    if e.timing=='fixed':new.clock=max(e.at,new.clock)
    if e.kind=='owner_review':
        if not ticket.approval_requested:raise ValueError('owner review not requested')
        approved=e.outcome=='approved';proof={'asset_id':asset.id,'approved':approved,'event_id':event_id}
        new.approvals[asset.id]={'approved':approved,'notice':signed_notice(new.domain,'property-owner',proof,new.clock)}
    elif e.kind=='inspection':
        if not ticket.inspection_requested or ticket.status in {'work_verified','closed'}:raise ValueError('inspection not pending for unresolved work')
        result=e.outcome=='hazard';asset.revision+=1;asset.simulated_hazard=result
        asset.inspection_notice=signed_notice(new.domain,'property-inspection',{'asset_id':asset.id,'hazard':result,'event_id':event_id},new.clock)
    elif e.kind=='technician_visit':
        appointment=new.appointments.get(ticket.appointment_id or '')
        if not appointment or appointment.status!='scheduled':raise ValueError('no booked appointment for physical visit')
        if new.plans[appointment.plan_id].asset_revision!=asset.revision:raise ValueError('physical work order is superseded by a new inspection')
        slot=new.slots[appointment.slot_id];new.clock=max(slot.end,new.clock)
        receipt={'ticket_id':ticket.id,'appointment_id':appointment.id,'outcome':e.outcome,'visited_at':slot.end.isoformat(),'event_id':event_id}
        ticket.repair_notice=signed_notice(new.domain,'property-technician',receipt,new.clock);ticket.version+=1
        if e.outcome=='repaired':
            appointment.status='completed';ticket.status='work_verified';asset.condition='working'
            plan=new.plans[appointment.plan_id]
            if plan.cost_cents>0:
                key='charge_'+plan.id;payload={'ticket_id':ticket.id,'plan_id':plan.id,'payer_id':plan.payer_id,'amount_cents':plan.cost_cents}
                new.charges[key]=Charge(id=key,ticket_id=ticket.id,plan_id=plan.id,payer_id=plan.payer_id,amount_cents=plan.cost_cents,notice=signed_notice(new.domain,'property-billing',payload,new.clock))
        else:appointment.status='failed';ticket.status='awaiting_repair'
    else:
        complaints=[c for c in new.complaints.values() if c.ticket_id==ticket.id and c.status=='escalated']
        if not complaints:raise ValueError('complaint not escalated')
        for c in complaints:
            c.status=e.outcome;c.review_notice=signed_notice(new.domain,'property-manager',{'complaint_id':c.id,'outcome':e.outcome,'event_id':event_id},new.clock)
    new.applied_events.append(event_id);return new
