"""Scoped public hotel catalog and private operational views."""
from datetime import date
from .rules import role,days,occupied,calendar_capacity,net_paid,refund_limit,cancellation_fee


def guest_view(guest):return guest.model_dump(mode='json',exclude={'simulated_identity_result'})


def reservation_view(world,reservation):
    return {**reservation.model_dump(mode='json'),'net_paid_cents':net_paid(world,reservation.id),
            'refundable_cents':refund_limit(world,reservation),'cancellation_fee_now_cents':cancellation_fee(world,reservation,world.clock)}


def observation(world,identity):
    if identity not in world.users:raise ValueError('unknown customer identity')
    staff={p for p in world.properties if role(world,identity,p)}
    own={r.id for r in world.reservations.values() if world.guests[r.guest_id].user_id==identity}
    bookings={r.id for r in world.reservations.values() if r.property_id in staff}|own
    financial={p for p in staff if role(world,identity,p) in {'manager','finance'}}
    own_guests={g.id for g in world.guests.values() if g.user_id==identity}
    return {'domain':world.domain,'version':world.version,'clock':world.clock.isoformat(),
        'identity':world.users[identity].model_dump(mode='json'),
        'properties':[p.model_dump(mode='json',exclude={'owner_id'}) for p in world.properties.values()],
        'access':[a.model_dump(mode='json') for a in world.access.values() if a.property_id in staff and a.user_id==identity],
        'room_types':[t.model_dump(mode='json') for t in world.room_types.values()],
        'rate_plans':[p.model_dump(mode='json') for p in world.rate_plans.values()],
        'rooms':[r.model_dump(mode='json') for r in world.rooms.values() if r.property_id in staff],
        'calendar':[c.model_dump(mode='json') for c in world.calendar.values() if c.property_id in staff],
        'guest_profiles':[guest_view(g) for g in world.guests.values() if g.id in own_guests or any(world.reservations[rid].guest_id==g.id for rid in bookings)],
        'public_guest_directory':[{'id':g.id,'name':g.name} for g in world.guests.values()] if staff else [],
        'reservations':[reservation_view(world,r) for r in world.reservations.values() if r.id in bookings],
        'assignments':[a.model_dump(mode='json') for a in world.assignments.values() if a.reservation_id in bookings],
        'payments':[p.model_dump(mode='json',exclude={'gateway_notice'}) for p in world.payments.values() if p.property_id in financial or p.reservation_id in own],
        'stays':[s.model_dump(mode='json') for s in world.stays.values() if s.reservation_id in bookings]}


def visible_ids(view):
    return {item['id'] for value in view.values() if isinstance(value,list) for item in value if isinstance(item,dict) and 'id' in item}


def query_data(world,actor,op):
    rid,name=op.resource_id,op.name
    if name=='get_property':return world.properties[rid].model_dump(mode='json',exclude={'owner_id'})
    if name=='get_availability':
        tid=op.params['room_type_id'];arrival=date.fromisoformat(op.params['arrival']);departure=date.fromisoformat(op.params['departure'])
        return {'room_type_id':tid,'nights':[{'day':day.isoformat(),'capacity':calendar_capacity(world,tid,day),
            'booked_or_held':occupied(world,tid,day),'available':calendar_capacity(world,tid,day)-occupied(world,tid,day)} for day in days(arrival,departure)]}
    if name=='get_room':
        room=world.rooms[rid]
        return room.model_dump(mode='json') if role(world,actor,op.scope_id) else room.model_dump(mode='json',exclude={'housekeeping','maintenance','current_stay_id'})
    if name=='get_reservation':
        r=world.reservations[rid]
        data=reservation_view(world,r);assignment=world.assignments.get('assign_'+rid)
        data['assignment']=assignment.model_dump(mode='json') if assignment else None
        check=world.identity_checks.get('identity_'+rid)
        data['guest_identity_status']=check.outcome if check else 'unchecked'
        return data
    if name=='get_guest':return guest_view(world.guests[rid])
    if name=='get_payment_ledger':
        r=world.reservations[rid]
        return {'payments':[p.model_dump(mode='json',exclude={'gateway_notice'}) for p in world.payments.values() if p.reservation_id==rid],
                'net_paid_cents':net_paid(world,rid),'refundable_cents':refund_limit(world,r),'deposit_cents':r.deposit_cents,'total_cents':r.total_cents}
    if name=='get_calendar_day':return world.calendar[rid].model_dump(mode='json')
    if name=='get_rate_plan':return world.rate_plans[rid].model_dump(mode='json')
    raise ValueError('unknown StayOps query')
