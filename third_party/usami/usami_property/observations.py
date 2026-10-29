"""Unit-scoped service information; other residents' contacts and simulator outcomes stay hidden."""
from usami_dialogue.process import public_context


def _unit(u,actor):return {'id':u.id,'property_id':u.property_id,'building_id':u.building_id,'relationship':'owner' if actor==u.owner_id else 'resident'}
def observation(world,actor):return {'domain':world.domain,'identity':{'id':actor,'name':world.users[actor].name},'clock':world.clock.isoformat(),
 'units':[_unit(u,actor) for u in sorted(world.units.values(),key=lambda row:row.id) if actor==u.owner_id or actor in u.resident_ids], 'customer_provided':public_context(world.interaction)}
def visible_ids(view):return {r['id'] for values in view.values() if isinstance(values,list) for r in values if isinstance(r,dict) and 'id' in r}

def _asset(a):return a.model_dump(mode='json',exclude={'simulated_symptom','simulated_hazard','inspection_notice'})|{'inspection':a.inspection_notice.get('binding') or None}

def query_data(world,actor,op):
    if op.name=='get_units':return {'units':[_unit(u,actor) for u in sorted(world.units.values(),key=lambda row:row.id) if u.property_id==op.scope_id and (actor==u.owner_id or actor in u.resident_ids)]}
    unit=world.units[op.resource_id]
    if op.name=='get_unit':return {**_unit(unit,actor),'assets':[_asset(a) for a in sorted(world.assets.values(),key=lambda row:row.id) if a.unit_id==unit.id],
        'contract':next(c.model_dump(mode='json') for c in sorted(world.contracts.values(),key=lambda row:row.id) if c.property_id==unit.property_id)}
    if op.name=='get_asset':return {'asset':_asset(world.assets[op.params['asset_id']]),'owner_review':world.approvals.get(op.params['asset_id'],{}).get('approved')}
    if op.name=='get_technicians':return {'technicians':[t.model_dump(mode='json') for t in sorted(world.technicians.values(),key=lambda row:row.id) if t.property_id==unit.property_id],
        'slots':[s.model_dump(mode='json') for s in sorted(world.slots.values(),key=lambda row:row.id) if s.property_id==unit.property_id],
        'busy':[{'technician_id':s.technician_id,'start':s.start.isoformat(),'end':s.end.isoformat()} for s in sorted(world.background_busy.values(),key=lambda row:row.id) if s.property_id==unit.property_id]+
            [{'technician_id':a.technician_id,'start':world.slots[a.slot_id].start.isoformat(),'end':world.slots[a.slot_id].end.isoformat()} for a in sorted(world.appointments.values(),key=lambda row:row.id) if a.status=='scheduled' and world.slots[a.slot_id].property_id==unit.property_id]}
    if op.name=='get_plan':return {'plan':world.plans[op.params['plan_id']].model_dump(mode='json')}
    t=world.tickets[op.params['ticket_id']]
    return {**t.model_dump(mode='json',exclude={'reporter_id','repair_notice','closure_notice'}),'visit_result':t.repair_notice.get('binding') or None,
        'plans':[q.model_dump(mode='json') for q in sorted(world.plans.values(),key=lambda row:row.id) if q.ticket_id==t.id and q.requester_id==actor],
        'complaints':[c.model_dump(mode='json',exclude={'reporter_id','review_notice'}) for c in sorted(world.complaints.values(),key=lambda row:row.id) if c.ticket_id==t.id and c.reporter_id==actor]}
