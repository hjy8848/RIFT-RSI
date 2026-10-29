"""Anonymous interval/workflow structures and independent decision branches."""
import re
from datetime import date
from usami_harness.models import digest
from .rules import role,decide,net_paid,available,cancellation_fee


def task_signatures(world,task):
    entities={key:collection for collection in world.__class__.model_fields if isinstance(getattr(world,collection),dict) for key in getattr(world,collection)}
    for atom in task.goal.atoms:entities.setdefault(atom.entity,atom.collection)
    for op in task.capabilities:
        if op['name']=='create_hold':entities.setdefault(op['params']['reservation_id'],'reservations')
        for prefix,collection in [('assign_','assignments'),('stay_','stays'),('identity_','identity_checks'),('cancel_','cancellations')]:
            entities.setdefault(prefix+op['resource_id'],collection)
    aliases,counters={},{}
    def symbolic(value,key=''):
        if isinstance(value,dict):return {k:symbolic(v,k) for k,v in sorted(value.items())}
        if isinstance(value,list):return [symbolic(v,key) for v in value]
        if isinstance(value,str) and value in entities:
            if value not in aliases:
                collection=entities[value];counters[collection]=counters.get(collection,0)+1;aliases[value]=collection+':'+str(counters[collection])
            return aliases[value]
        if isinstance(value,bool) or value is None:return value
        if isinstance(value,int):return '<integer>'
        if isinstance(value,str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}',value):return '<date>'
        if key=='key':return '<key>'
        return value
    goals=[symbolic(a.model_dump(mode='json')) for a in task.goal.atoms]
    ops=[symbolic(o) for o in task.capabilities];shapes={digest(raw):shape for raw,shape in zip(task.capabilities,ops)}
    edges=set()
    for witness in task.witnesses:
        for e in witness.direct_edges:edges.add((digest(shapes[digest(witness.operations[e['from']])]),digest(shapes[digest(witness.operations[e['to']])]),e['error_code']))
    used={digest(shapes[digest(op)]):shapes[digest(op)] for w in task.witnesses for op in w.operations} if task.kind=='completion' else {digest(op):op for op in ops}
    structure={'kind':task.kind,'goals':sorted(goals,key=digest) if task.kind=='completion' else ['total_frame_preservation'],
               'operations':sorted(used.values(),key=digest),'causal_edges':sorted(edges)}
    features=[]
    for raw in task.capabilities:
        from .models import Operation
        op=Operation.model_validate(raw);booking=world.reservations.get(op.resource_id);p=op.params
        tid=p.get('room_type_id',booking.room_type_id if booking else None);planid=p.get('rate_plan_id',booking.rate_plan_id if booking else None)
        room=world.rooms.get(p.get('room_id',op.resource_id));plan=world.rate_plans.get(planid)
        arrival=date.fromisoformat(p['arrival']) if 'arrival' in p else booking.arrival if booking else None
        departure=date.fromisoformat(p['departure']) if 'departure' in p else booking.departure if booking else None
        guest=world.guests.get(p.get('guest_id',booking.guest_id if booking else None))
        features.append({'operation':op.name,'role':role(world,task.identity,op.scope_id),
            'owner':world.properties.get(op.scope_id) and world.properties[op.scope_id].owner_id==task.identity,
            'guest_owner':bool(guest and guest.user_id==task.identity),'initial_decision':decide(world,task.identity,op).error_code or 'ALLOWED',
            'booking_status':booking.status if booking else None,
            'hold_time':'expired' if booking and world.clock>=booking.hold_expires_at else 'valid',
            'payment':('none' if not net_paid(world,booking.id) else 'full' if net_paid(world,booking.id)>=booking.total_cents else 'deposit' if net_paid(world,booking.id)>=booking.deposit_cents else 'partial') if booking else None,
            'deposit_policy':'full_prepay' if plan and plan.deposit_bps==10000 else 'partial_deposit' if plan else None,
            'cancellation_policy':plan.late_fee if plan else None,
            'cancellation_window':('free' if cancellation_fee(world,booking,world.clock)==0 else 'penalty') if booking else None,
            'arrival_relation':'today' if arrival==world.clock.date() else 'future' if arrival and arrival>world.clock.date() else 'past' if arrival else None,
            'available':available(world,tid,arrival,departure,booking.id if booking else None) if tid in world.room_types and arrival and departure else None,
            'room_cleanliness':room.housekeeping if room else None,'room_maintenance':room.maintenance if room else None,
            'room_occupied':bool(room and room.current_stay_id),
            'identity_simulator_branch':guest.simulated_identity_result if guest and op.name in {'verify_guest','check_in'} else None})
    return digest(structure),digest({'structure':structure,'branches':sorted(features,key=digest),'refusal':task.refusal_code})
