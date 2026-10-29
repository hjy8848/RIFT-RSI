"""Independent property policy and expected transitions; no actual mutation import."""
from datetime import datetime,timedelta
from usami_harness.models import Decision,digest,signed_notice,notice_valid
from usami_dialogue.process import consent_code
from .models import Ticket,Plan,Appointment,Complaint

POLICY={
 'UNIT_ACCESS_DENIED':'The authenticated owner or current resident can service only their own unit.',
 'REQUIRED_INFORMATION_MISSING':'Clarify the unit, asset, observed fault and hazard before classification.',
 'INCORRECT_TRIAGE':'Electrical sparks or a reported/inspected active hazard are emergencies; leaks and stuck locks are urgent; other non-hazard faults are routine.',
 'TRIAGE_REQUIRED':'A current assessment must classify the actual report; inspection changes invalidate old assessments.',
 'TECHNICIAN_SKILL_REQUIRED':'Assign a technician licensed for the asset category in this property.',
 'CALENDAR_CONFLICT':'A technician cannot serve overlapping appointments or existing commitments.',
 'OWNER_APPROVAL_REQUIRED':'A resident cannot commit owner-paid work above the contract limit without actual owner review.',
 'CUSTOMER_CONSTRAINT_UNMET':'Respect the chosen slot, access interval, budget and quoted charge.',
 'RESPONSE_DEADLINE_EXCEEDED':'Emergency and urgent work must start within the contract response SLA.',
 'OUTSIDE_WORKING_HOURS':'Routine work stays within contract hours; emergency response requires an on-call technician.',
 'EXPLICIT_CONSENT_REQUIRED':'Entry/work scheduling, cancellation and closure bind the actual current offer.',
 'REPAIR_PROOF_REQUIRED':'Only an attested completed repair can be closed; an agent cannot invent a technician visit.'}


def ticket_key(asset):return 'ticket_'+digest(asset)[:18]
def complaint_key(ticket,reason,actor):return 'complaint_'+digest([ticket,reason,actor])[:18]
def plan_hash(q):return digest(q.model_dump(mode='json',exclude={'quote_hash','status'}))
def plan_key(t,asset,tech,slot,revision,clock):return 'plan_'+digest([t.id,t.version,asset.revision,tech,slot,revision,clock.isoformat()])[:20]
def can_access(world,actor,unit):return actor==unit.owner_id or actor in unit.resident_ids
def contract(world,unit):return next(c for c in world.contracts.values() if c.property_id==unit.property_id)
def overlaps(a,b):return a.start<b.end and b.start<a.end

def cost_terms(world,asset,actor):
    unit=world.units[asset.unit_id];c=contract(world,unit)
    cost=max(0,asset.repair_cost_cents-(c.coverage_cents if asset.category in c.covered_categories else 0))
    payer=unit.owner_id if asset.responsibility=='owner' else actor if asset.responsibility=='resident' else unit.property_id
    return cost,payer,cost if payer==actor else 0

def required_severity(world,t):
    asset=world.assets[t.asset_id]
    hazard=asset.inspection_notice.get('binding',{}).get('hazard',world.interaction.known_facts.get('observed_hazard'))
    return 'emergency' if hazard or t.symptom=='sparks' else 'urgent' if t.symptom in {'leak','stuck'} else 'normal'

def consent_binding(world,t,kind):
    if kind=='close_ticket':return digest({'ticket_id':t.id,'repair_signature':t.repair_notice.get('signature'),'ticket_version':t.version})
    a=world.appointments.get(t.appointment_id or '')
    return digest(a) if a else None

def slot_code(world,t,technician,slot,actor):
    asset=world.assets[t.asset_id];unit=world.units[t.unit_id];c=contract(world,unit);ctx=world.interaction
    if not technician or not slot or technician.property_id!=unit.property_id or slot.property_id!=unit.property_id or slot.technician_id!=technician.id:return 'TECHNICIAN_SCOPE_MISMATCH'
    if asset.category not in technician.skills:return 'TECHNICIAN_SKILL_REQUIRED'
    if slot.start<=world.clock or (slot.end-slot.start).total_seconds()<asset.duration_minutes*60:return 'SLOT_UNAVAILABLE'
    if t.severity=='emergency' and not technician.on_call:return 'EMERGENCY_RESPONDER_REQUIRED'
    if t.severity!='emergency' and (slot.start.hour<c.normal_start_hour or slot.end.date()!=slot.start.date() or slot.end.hour>c.normal_end_hour or (slot.end.hour==c.normal_end_hour and slot.end.minute>0)):return 'OUTSIDE_WORKING_HOURS'
    limit=c.emergency_response_hours if t.severity=='emergency' else c.urgent_response_hours if t.severity=='urgent' else None
    if limit is not None and slot.start>t.created_at+timedelta(hours=limit):return 'RESPONSE_DEADLINE_EXCEEDED'
    if any(x.technician_id==technician.id and overlaps(slot,x) for x in world.background_busy.values()):return 'CALENDAR_CONFLICT'
    if any(a.status=='scheduled' and a.technician_id==technician.id and a.ticket_id!=t.id and overlaps(slot,world.slots[a.slot_id]) for a in world.appointments.values()):return 'CALENDAR_CONFLICT'
    if any(k not in ctx.known_facts for k in ('earliest_access','latest_access','max_budget_cents')):return 'REQUIRED_INFORMATION_MISSING'
    if ctx.preferences.get('slot_id')!=slot.id:return 'PREFERENCE_RESOLUTION_REQUIRED'
    if slot.start<datetime.fromisoformat(ctx.known_facts['earliest_access']) or slot.end>datetime.fromisoformat(ctx.known_facts['latest_access']):return 'CUSTOMER_CONSTRAINT_UNMET'
    cost,payer,charge=cost_terms(world,asset,actor)
    if charge>ctx.known_facts['max_budget_cents']:return 'CUSTOMER_CONSTRAINT_UNMET'
    if payer==unit.owner_id and actor!=unit.owner_id and cost>c.resident_approval_limit_cents and world.approvals.get(asset.id,{}).get('approved') is not True:return 'OWNER_APPROVAL_REQUIRED'
    return None


def invariants(world):
    errors=[]
    def add(code,path):errors.append({'code':code,'path':path})
    collections=('users','properties','buildings','units','contracts','assets','technicians','slots','background_busy','tickets','plans','appointments','complaints','charges','external_events')
    for name in collections:
        for key,row in getattr(world,name).items():
            if row.id!=key:add('ENTITY_KEY_MISMATCH',name+'/'+key)
    for b in world.buildings.values():
        if b.property_id not in world.properties:add('BUILDING_REFERENCE_INVALID','buildings/'+b.id)
    for u in world.units.values():
        b=world.buildings.get(u.building_id)
        if not b or b.property_id!=u.property_id or u.owner_id not in world.users or set(u.resident_ids)-set(world.users):add('UNIT_REFERENCE_INVALID','units/'+u.id)
    for c in world.contracts.values():
        if c.property_id not in world.properties or set(c.covered_categories)-{'plumbing','electrical','hvac','lock'} or c.emergency_response_hours>c.urgent_response_hours:add('CONTRACT_INVALID','contracts/'+c.id)
    for pid in world.properties:
        cs=[c for c in world.contracts.values() if c.property_id==pid]
        if len(cs)!=1 or cs[0].normal_start_hour>=cs[0].normal_end_hour:add('CONTRACT_INVALID','properties/'+pid)
    for a in world.assets.values():
        u=world.units.get(a.unit_id)
        if not u or u.property_id!=a.property_id:add('ASSET_REFERENCE_INVALID','assets/'+a.id)
        if a.inspection_notice and (not notice_valid(a.inspection_notice) or a.inspection_notice.get('domain')!=world.domain or a.inspection_notice.get('source')!='property-inspection-simulator-v0.1' or a.inspection_notice.get('binding',{}).get('asset_id')!=a.id):add('INSPECTION_ATTESTATION_INVALID','assets/'+a.id)
    for tech in world.technicians.values():
        if tech.property_id not in world.properties or not tech.skills or set(tech.skills)-{'plumbing','electrical','hvac','lock'}:add('TECHNICIAN_REFERENCE_INVALID','technicians/'+tech.id)
    intervals=[]
    for name in ('slots','background_busy'):
        for s in getattr(world,name).values():
            tech=world.technicians.get(s.technician_id)
            if not tech or tech.property_id!=s.property_id or s.start.tzinfo is None or s.end.tzinfo is None or s.start>=s.end:add('CALENDAR_REFERENCE_INVALID',name+'/'+s.id)
            if name=='background_busy':intervals.append(s)
    for t in world.tickets.values():
        a=world.assets.get(t.asset_id)
        if not a or a.unit_id!=t.unit_id or t.reporter_id not in world.users or t.id!=ticket_key(t.asset_id):add('TICKET_REFERENCE_INVALID','tickets/'+t.id);continue
        if t.appointment_id is not None and (t.appointment_id not in world.appointments or world.appointments[t.appointment_id].slot_id!=t.booked_slot_id):add('APPOINTMENT_REFERENCE_INVALID','tickets/'+t.id)
        if t.status=='scheduled' and (t.appointment_id not in world.appointments or world.appointments[t.appointment_id].status!='scheduled'):add('TICKET_SCHEDULE_MISMATCH','tickets/'+t.id)
        if t.repair_notice:
            notice=t.repair_notice;binding=notice.get('binding',{});event=world.external_events.get(binding.get('event_id'));app=world.appointments.get(binding.get('appointment_id'))
            if not event or event.kind!='technician_visit' or event.asset_id!=t.asset_id or event.id not in world.applied_events or binding.get('outcome')!=event.outcome or not app or app.ticket_id!=t.id or binding.get('visited_at')!=world.slots[app.slot_id].end.isoformat() or not notice_valid(notice) or notice.get('source')!='property-technician-simulator-v0.1' or notice.get('domain')!=world.domain:add('REPAIR_ATTESTATION_INVALID','tickets/'+t.id)
        if t.status in {'work_verified','closed'}:
            proof=t.repair_notice;binding=proof.get('binding',{});app=world.appointments.get(t.appointment_id or '')
            if not app or app.status!='completed' or a.condition!='working' or not notice_valid(proof) or proof.get('source')!='property-technician-simulator-v0.1' or proof.get('domain')!=world.domain or binding.get('ticket_id')!=t.id or binding.get('appointment_id')!=app.id or binding.get('outcome')!='repaired':add('REPAIR_ATTESTATION_INVALID','tickets/'+t.id)
        if t.status=='closed' and (t.closure_notice.get('domain')!=world.domain or t.closure_notice.get('source')!='property-closure-simulator-v0.1' or t.closure_notice.get('binding',{}).get('ticket_id')!=t.id or not notice_valid(t.closure_notice) or t.closure_notice.get('binding',{}).get('repair_signature')!=t.repair_notice.get('signature') or not t.closure_notice.get('binding',{}).get('consent_hash')):add('CLOSURE_ATTESTATION_INVALID','tickets/'+t.id)
    for q in world.plans.values():
        t=world.tickets.get(q.ticket_id);s=world.slots.get(q.slot_id)
        requester=world.users.get(q.requester_id)
        if not t or not s or not requester or not can_access(world,q.requester_id,world.units[t.unit_id]) or q.technician_id!=s.technician_id or q.quote_hash!=plan_hash(q):add('PLAN_INTEGRITY','plans/'+q.id)
    for a in world.appointments.values():
        t=world.tickets.get(a.ticket_id);q=world.plans.get(a.plan_id);s=world.slots.get(a.slot_id)
        if not t or not q or not s or q.ticket_id!=t.id or q.status!='committed' or a.technician_id!=s.technician_id or a.slot_id!=q.slot_id or a.id!='appointment_'+q.id:add('APPOINTMENT_REFERENCE_INVALID','appointments/'+a.id);continue
        if a.status=='scheduled':intervals.append(s)
    for i,a in enumerate(intervals):
        if any(a.technician_id==b.technician_id and overlaps(a,b) for b in intervals[i+1:]):add('TECHNICIAN_DOUBLE_BOOKED','calendar')
    for c in world.complaints.values():
        if c.ticket_id not in world.tickets or c.reporter_id not in world.users or c.id!=complaint_key(c.ticket_id,c.reason,c.reporter_id):add('COMPLAINT_REFERENCE_INVALID','complaints/'+c.id)
        if c.status in {'reviewed','dismissed'}:
            proof=c.review_notice;binding=proof.get('binding',{});event=world.external_events.get(binding.get('event_id'))
            if not event or event.kind!='complaint_review' or event.id not in world.applied_events or world.tickets.get(c.ticket_id) is None or event.asset_id!=world.tickets[c.ticket_id].asset_id or event.outcome!=c.status or binding!={'complaint_id':c.id,'outcome':c.status,'event_id':event.id} or not notice_valid(proof) or proof.get('domain')!=world.domain or proof.get('source')!='property-manager-simulator-v0.1':add('COMPLAINT_ATTESTATION_INVALID','complaints/'+c.id)
    for c in world.charges.values():
        t=world.tickets.get(c.ticket_id);q=world.plans.get(c.plan_id)
        binding={'ticket_id':c.ticket_id,'plan_id':c.plan_id,'payer_id':c.payer_id,'amount_cents':c.amount_cents}
        if not t or not q or t.status not in {'work_verified','closed'} or c.id!='charge_'+q.id or c.amount_cents!=q.cost_cents or c.payer_id!=q.payer_id or not notice_valid(c.notice) or c.notice.get('source')!='property-billing-simulator-v0.1' or c.notice.get('domain')!=world.domain or c.notice.get('binding')!=binding:add('REPAIR_ACCOUNTING_INVALID','charges/'+c.id)
    for p in world.profiles.values():
        try:p.checked()
        except ValueError:add('PRIVATE_PROFILE_INTEGRITY','profiles')
        a=world.assets.get(p.facts.get('asset_id'));u=world.units.get(a.unit_id) if a else None
        if not u or not can_access(world,p.identity,u) or p.protected_objects!=(u.id,) or p.facts.get('unit_id')!=u.id:add('PRIVATE_PROFILE_REFERENCE','profiles')
    for e in world.external_events.values():
        if e.asset_id not in world.assets or (e.timing=='fixed' and (e.at is None or e.at.tzinfo is None)) or (e.kind=='technician_visit' and e.timing!='appointment_end'):add('EXTERNAL_EVENT_INVALID','external_events/'+e.id)
    if len(set(world.applied_events))!=len(world.applied_events) or set(world.applied_events)-set(world.external_events):add('EXTERNAL_EVENT_INVALID','applied_events')
    for aid,approval in world.approvals.items():
        proof=approval.get('notice',{});binding=proof.get('binding',{});event=world.external_events.get(binding.get('event_id'))
        if aid not in world.assets or not event or event.kind!='owner_review' or event.asset_id!=aid or event.id not in world.applied_events or approval.get('approved')!=(event.outcome=='approved') or not notice_valid(proof) or proof.get('domain')!=world.domain or proof.get('source')!='property-owner-simulator-v0.1' or binding!={'asset_id':aid,'approved':approval.get('approved'),'event_id':event.id}:add('OWNER_APPROVAL_ATTESTATION_INVALID','approvals/'+aid)
    return errors


def decide(world,actor,op):
    deny=lambda code:Decision(allowed=False,error_code=code)
    if invariants(world):return deny('INVALID_WORLD')
    if actor not in world.users:return deny('UNKNOWN_ACTOR')
    if op.scope_id not in world.properties:return deny('PROPERTY_NOT_FOUND')
    if op.name=='get_units':
        if op.resource_id!=op.scope_id:return deny('PROPERTY_SCOPE_MISMATCH')
        from .observations import query_data
        return Decision(allowed=True,expected_state=world.clone(),data=query_data(world,actor,op))
    unit=world.units.get(op.resource_id)
    if not unit:return deny('UNIT_NOT_FOUND')
    if unit.property_id!=op.scope_id:return deny('PROPERTY_SCOPE_MISMATCH')
    if not can_access(world,actor,unit):return deny('UNIT_ACCESS_DENIED')
    p=op.params;t=world.tickets.get(p.get('ticket_id',''))
    if op.name in {'get_ticket','triage_ticket','request_owner_review','request_inspection','quote_visit','cancel_visit','close_ticket','file_complaint'} and (not t or t.unit_id!=unit.id):return deny('TICKET_SCOPE_MISMATCH')
    q=world.plans.get(p.get('plan_id',''))
    if op.name in {'get_plan','book_visit'} and (not q or q.requester_id!=actor or world.tickets[q.ticket_id].unit_id!=unit.id):return deny('PLAN_SCOPE_MISMATCH')
    if op.name in {'get_asset','open_ticket'} and (p['asset_id'] not in world.assets or world.assets[p['asset_id']].unit_id!=unit.id):return deny('ASSET_SCOPE_MISMATCH')
    complaint=world.complaints.get(p.get('complaint_id',''))
    if op.name=='escalate_complaint' and (not complaint or world.tickets[complaint.ticket_id].unit_id!=unit.id or complaint.reporter_id!=actor):return deny('COMPLAINT_SCOPE_MISMATCH')
    if op.is_query:
        from .observations import query_data
        return Decision(allowed=True,expected_state=world.clone(),data=query_data(world,actor,op))
    ctx=world.interaction
    if ctx.identity!=actor or ctx.known_facts.get('unit_id')!=unit.id:return deny('REQUIRED_INFORMATION_MISSING')
    new=world.clone();data={}
    if op.name=='open_ticket':
        asset=world.assets[p['asset_id']]
        if ctx.known_facts.get('asset_id')!=asset.id or ctx.known_facts.get('symptom')!=p['symptom']:return deny('REQUIRED_INFORMATION_MISSING')
        if asset.condition=='working':return deny('ASSET_ALREADY_WORKING')
        tid=ticket_key(asset.id)
        if tid not in new.tickets:new.tickets[tid]=Ticket(id=tid,unit_id=unit.id,asset_id=asset.id,reporter_id=actor,symptom=p['symptom'],created_at=world.clock)
        data={'ticket_id':tid,'status':new.tickets[tid].status}
    elif op.name=='triage_ticket':
        if 'observed_hazard' not in ctx.known_facts:return deny('REQUIRED_INFORMATION_MISSING')
        if t.status in {'closed','work_verified'}:return deny('TICKET_NOT_OPEN')
        asset=world.assets[t.asset_id];severity=required_severity(world,t)
        if p['severity']!=severity:return deny('INCORRECT_TRIAGE')
        updated=new.tickets[t.id]
        if updated.assessment_version!=asset.revision or updated.severity!=severity:updated.version+=1
        updated.severity=severity;updated.assessment_version=asset.revision
        if updated.status=='open':updated.status='triaged'
        data={'ticket_id':t.id,'severity':severity,'assessment_version':asset.revision}
    elif op.name in {'request_owner_review','request_inspection'}:
        if t.status=='closed':return deny('TICKET_CLOSED')
        if op.name=='request_inspection' and t.status=='work_verified':return deny('INSPECTION_WINDOW_CLOSED')
        setattr(new.tickets[t.id],'approval_requested' if op.name=='request_owner_review' else 'inspection_requested',True)
        data={'ticket_id':t.id,'status':'pending'}
    elif op.name=='quote_visit':
        asset=world.assets[t.asset_id]
        if t.status not in {'triaged','scheduled','awaiting_repair'} or t.severity is None or t.assessment_version!=asset.revision:return deny('TRIAGE_REQUIRED')
        tech=world.technicians.get(p['technician_id']);slot=world.slots.get(p['slot_id']);code=slot_code(world,t,tech,slot,actor)
        if code:return deny(code)
        cost,payer,charge=cost_terms(world,asset,actor);qid=plan_key(t,asset,tech.id,slot.id,ctx.revision,world.clock)
        if qid not in new.plans:
            plan=Plan(id=qid,ticket_id=t.id,technician_id=tech.id,slot_id=slot.id,requester_id=actor,cost_cents=cost,customer_charge_cents=charge,payer_id=payer,revision=ctx.revision,ticket_version=t.version,asset_revision=asset.revision,expires_at=world.clock+timedelta(hours=1),quote_hash='')
            plan.quote_hash=plan_hash(plan);new.plans[qid]=plan
        data={'plan':new.plans[qid].model_dump(mode='json')}
    elif op.name=='book_visit':
        t=world.tickets[q.ticket_id];asset=world.assets[t.asset_id]
        if t.status=='scheduled':return deny('CANCEL_ACTIVE_APPOINTMENT_FIRST')
        if q.status!='offered' or q.ticket_version!=t.version or q.asset_revision!=asset.revision or t.assessment_version!=asset.revision:return deny('STALE_ASSESSMENT')
        if q.revision!=ctx.revision:return deny('STALE_CUSTOMER_CHOICE')
        if world.clock>q.expires_at:return deny('PLAN_EXPIRED')
        code=slot_code(world,t,world.technicians[q.technician_id],world.slots[q.slot_id],actor)
        if code:return deny(code)
        code=consent_code(ctx,op,q.quote_hash)
        if code:return deny(code)
        aid='appointment_'+q.id;new.appointments[aid]=Appointment(id=aid,ticket_id=t.id,plan_id=q.id,technician_id=q.technician_id,slot_id=q.slot_id)
        updated=new.tickets[t.id];updated.appointment_id=aid;updated.booked_slot_id=q.slot_id;updated.status='scheduled';updated.version+=1;new.plans[q.id].status='committed'
        data={'ticket_id':t.id,'appointment_id':aid,'status':'scheduled'}
    elif op.name=='cancel_visit':
        appointment=world.appointments.get(t.appointment_id or '')
        if not appointment or appointment.status!='scheduled':return deny('NO_ACTIVE_APPOINTMENT')
        if world.clock>=world.slots[appointment.slot_id].start:return deny('CANCELLATION_WINDOW_CLOSED')
        code=consent_code(ctx,op,consent_binding(world,t,'cancel_visit'))
        if code:return deny(code)
        new.appointments[appointment.id].status='cancelled';new.tickets[t.id].status='triaged';new.tickets[t.id].version+=1
        data={'ticket_id':t.id,'status':'cancelled'}
    elif op.name=='close_ticket':
        if t.status=='closed':return deny('TICKET_ALREADY_CLOSED')
        if t.status!='work_verified' or t.repair_notice.get('binding',{}).get('outcome')!='repaired':return deny('REPAIR_PROOF_REQUIRED')
        if ctx.preferences.get('feedback')!='satisfied':return deny('CUSTOMER_FEEDBACK_REQUIRED')
        code=consent_code(ctx,op,consent_binding(world,t,'close_ticket'))
        if code:return deny(code)
        acknowledgment=next(c for c in ctx.consents if c.operation==op.model_dump(mode='json') and c.revision==ctx.revision)
        binding={'ticket_id':t.id,'repair_signature':t.repair_notice['signature'],'consent_hash':digest(acknowledgment),'actor':actor}
        new.tickets[t.id].closure_notice=signed_notice(world.domain,'property-closure',binding,world.clock);new.tickets[t.id].status='closed';new.tickets[t.id].version+=1
        data={'ticket_id':t.id,'status':'closed'}
    elif op.name=='file_complaint':
        reason=p['reason'];outcome=t.repair_notice.get('binding',{}).get('outcome')
        if reason=='unresolved_repair' and outcome!='unresolved':return deny('COMPLAINT_BASIS_REQUIRED')
        if reason=='missed_visit' and outcome!='no_show':return deny('COMPLAINT_BASIS_REQUIRED')
        if reason=='service_quality' and (ctx.preferences.get('feedback')!='dissatisfied' or not t.appointment_id):return deny('COMPLAINT_BASIS_REQUIRED')
        cid=complaint_key(t.id,reason,actor)
        if cid not in new.complaints:new.complaints[cid]=Complaint(id=cid,ticket_id=t.id,reporter_id=actor,reason=reason)
        data={'complaint_id':cid,'status':new.complaints[cid].status}
    elif op.name=='escalate_complaint':
        if complaint.status not in {'open','escalated'}:return deny('COMPLAINT_REVIEWED')
        new.complaints[complaint.id].status='escalated';data={'complaint_id':complaint.id,'status':'escalated'}
    return Decision(allowed=True,expected_state=new,data=data)
