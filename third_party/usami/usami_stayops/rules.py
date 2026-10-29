"""Independent dated hotel specification. Never imports the actual applier."""
from datetime import date,datetime,time,timedelta,timezone
from usami_harness.models import Decision,digest,signed_notice,notice_valid
from .models import Reservation,Assignment,Payment,Cancellation,IdentityCheck,Stay

POLICY={
    'PROPERTY_PERMISSION_DENIED':'Staff roles apply only to their specific property. Guest booking ownership grants no staff privileges.',
    'ROOM_TYPE_UNAVAILABLE':'Every night of a half-open [arrival, departure) interval must have type inventory. Holds occupy inventory until explicitly released/expired.',
    'ROOM_INTERVAL_CONFLICT':'One physical room cannot be assigned to overlapping active reservations.',
    'ROOM_NOT_CLEAN':'Clean the unoccupied room before assignment or check-in.',
    'ROOM_IN_MAINTENANCE':'Maintenance rooms are unavailable. Unlocking maintenance restores physical room inventory.',
    'DEPOSIT_REQUIRED':'Successful gateway-attested payment must cover the quoted integer deposit before confirmation.',
    'PAYMENT_REQUIRED':'Full stay price and independently verified guest identity are required at check-in.',
    'HOLD_EXPIRED':'A hold is valid only before its explicit expiry on the frozen clock; expire/release it explicitly to free inventory.',
    'REFUND_EXCEEDS_ELIGIBLE':'Refunds cannot exceed paid balance after the frozen-time rate-plan cancellation fee, or genuine active-booking overpayment.',
    'REQUOTATION_REQUIRED':'Date/type changes revalidate the whole interval, release the old room assignment and return the reservation to a quoted hold for reconfirmation.',
}

ACTIVE=frozenset({'hold','confirmed','checked_in'})


def days(arrival,departure):
    return [arrival+timedelta(days=i) for i in range((departure-arrival).days)]


def role(world,actor,property_id):
    prop=world.properties.get(property_id)
    if prop and prop.owner_id==actor:return 'manager'
    return next((a.role for a in world.access.values() if a.user_id==actor and a.property_id==property_id),None)


def quote(world,room_type_id,plan_id,arrival,departure):
    room_type,plan=world.room_types[room_type_id],world.rate_plans[plan_id]
    total=sum(room_type.nightly_cents+(plan.weekend_surcharge_cents if day.weekday()>=5 else 0) for day in days(arrival,departure))
    return total,(total*plan.deposit_bps+9999)//10000


def paid(world,reservation_id,kind='charge'):
    return sum(p.amount_cents for p in world.payments.values() if p.reservation_id==reservation_id and p.kind==kind and p.outcome=='succeeded')


def net_paid(world,reservation_id):return paid(world,reservation_id)-paid(world,reservation_id,'refund')


def cancellation_fee(world,reservation,at):
    plan=world.rate_plans[reservation.rate_plan_id]
    deadline=datetime.combine(reservation.arrival,time.min,tzinfo=timezone.utc)-timedelta(hours=plan.free_cancel_hours)
    if at<=deadline:return 0
    first=world.room_types[reservation.room_type_id].nightly_cents+(plan.weekend_surcharge_cents if reservation.arrival.weekday()>=5 else 0)
    return reservation.total_cents if plan.late_fee=='full_stay' else first


def refund_limit(world,reservation):
    net=net_paid(world,reservation.id)
    if reservation.status in {'canceled','released','expired'}:
        cancellation=world.cancellations.get('cancel_'+reservation.id)
        return max(0,net-(cancellation.fee_cents if cancellation else 0))
    return max(0,net-reservation.total_cents)


def overlap(a,b,c,d):return a<d and c<b


def occupied(world,type_id,day,exclude=None):
    return sum(r.room_type_id==type_id and r.status in ACTIVE and r.id!=exclude and r.arrival<=day<r.departure for r in world.reservations.values())


def calendar_capacity(world,type_id,day):
    row=world.calendar.get(type_id+'_'+day.isoformat())
    if not row or row.closed:return 0
    physical=sum(room.room_type_id==type_id and not room.maintenance for room in world.rooms.values())
    return min(row.sale_limit,physical)


def available(world,type_id,arrival,departure,exclude=None):
    return all(calendar_capacity(world,type_id,day)>occupied(world,type_id,day,exclude) for day in days(arrival,departure))


def room_conflict(world,room_id,arrival,departure,exclude=None):
    return any(a.room_id==room_id and a.status=='active' and a.reservation_id!=exclude and overlap(arrival,departure,a.arrival,a.departure)
               for a in world.assignments.values())


def attested(world,notice,source,binding):
    try:return notice_valid(notice) and notice['domain']==world.domain and notice['source']==source+'-simulator-v0.1' and notice['binding']==binding and datetime.fromisoformat(notice['at'])<=world.clock
    except (KeyError,ValueError,TypeError):return False


def invariants(world):
    errors=[]
    def add(code,path):errors.append({'code':code,'path':path})
    collections=('users','properties','access','guests','room_types','rooms','calendar','rate_plans','reservations',
                 'assignments','payments','cancellations','identity_checks','stays')
    for collection in collections:
        for key,obj in getattr(world,collection).items():
            if key!=obj.id:add('ENTITY_KEY_MISMATCH',collection+'/'+key)
    for prop in world.properties.values():
        if prop.owner_id not in world.users:add('OWNER_MISSING','properties/'+prop.id)
    access=set()
    for grant in world.access.values():
        if grant.property_id not in world.properties or grant.user_id not in world.users:add('ACCESS_REFERENCE_INVALID','access/'+grant.id)
        key=(grant.property_id,grant.user_id)
        if key in access:add('DUPLICATE_SCOPED_ROLE','access/'+grant.id)
        access.add(key)
    for guest in world.guests.values():
        if guest.user_id not in world.users:add('GUEST_USER_MISSING','guests/'+guest.id)
    for room_type in world.room_types.values():
        if room_type.property_id not in world.properties:add('ROOM_TYPE_PROPERTY_INVALID','room_types/'+room_type.id)
    for room in world.rooms.values():
        room_type=world.room_types.get(room.room_type_id)
        if not room_type or room_type.property_id!=room.property_id:add('ROOM_BINDING_INVALID','rooms/'+room.id)
        if room.current_stay_id:
            stay=world.stays.get(room.current_stay_id)
            if not stay or stay.status!='checked_in' or stay.room_id!=room.id:add('ROOM_STAY_MISMATCH','rooms/'+room.id)
            if room.maintenance:add('MAINTENANCE_ROOM_OCCUPIED','rooms/'+room.id)
            if room.housekeeping!='clean':add('DIRTY_ROOM_OCCUPIED','rooms/'+room.id)
    for plan in world.rate_plans.values():
        if plan.property_id not in world.properties or len(set(plan.room_type_ids))!=len(plan.room_type_ids) or any(t not in world.room_types or world.room_types[t].property_id!=plan.property_id for t in plan.room_type_ids):
            add('RATE_PLAN_BINDING_INVALID','rate_plans/'+plan.id)
    calendar_keys=set()
    for row in world.calendar.values():
        room_type=world.room_types.get(row.room_type_id)
        key=(row.room_type_id,row.day)
        if key in calendar_keys or row.id!=row.room_type_id+'_'+row.day.isoformat() or not room_type or room_type.property_id!=row.property_id:
            add('CALENDAR_BINDING_INVALID','calendar/'+row.id)
        calendar_keys.add(key)
        physical=sum(r.room_type_id==row.room_type_id for r in world.rooms.values())
        if row.sale_limit>physical:add('CALENDAR_LIMIT_EXCEEDS_ROOMS','calendar/'+row.id)
        if occupied(world,row.room_type_id,row.day)>calendar_capacity(world,row.room_type_id,row.day):add('ROOM_TYPE_OVERBOOKED','calendar/'+row.id)
    for reservation in world.reservations.values():
        r=reservation;prop=world.properties.get(r.property_id);guest=world.guests.get(r.guest_id);room_type=world.room_types.get(r.room_type_id);plan=world.rate_plans.get(r.rate_plan_id)
        if not prop or not guest or not room_type or not plan or room_type.property_id!=r.property_id or plan.property_id!=r.property_id or r.room_type_id not in plan.room_type_ids:
            add('RESERVATION_BINDING_INVALID','reservations/'+r.id);continue
        if r.departure<=r.arrival or (r.departure-r.arrival).days>prop.max_stay_nights or r.occupants>room_type.max_occupants:
            add('RESERVATION_INTERVAL_INVALID','reservations/'+r.id)
        else:
            if (r.total_cents,r.deposit_cents)!=quote(world,r.room_type_id,r.rate_plan_id,r.arrival,r.departure):add('QUOTE_MISMATCH','reservations/'+r.id)
            if any(r.room_type_id+'_'+day.isoformat() not in world.calendar for day in days(r.arrival,r.departure)):add('CALENDAR_HORIZON_MISSING','reservations/'+r.id)
        if r.created_at.tzinfo is None or r.hold_expires_at.tzinfo is None or r.created_at>world.clock or r.hold_expires_at<=r.created_at:add('HOLD_TIME_INVALID','reservations/'+r.id)
        if net_paid(world,r.id)<0:add('REFUND_EXCEEDS_PAYMENT','reservations/'+r.id)
        if r.status in {'confirmed','checked_in','checked_out'} and net_paid(world,r.id)<r.deposit_cents:add('RESERVATION_DEPOSIT_REQUIRED','reservations/'+r.id)
        if r.status in {'checked_in','checked_out'} and net_paid(world,r.id)<r.total_cents:add('STAY_PAYMENT_REQUIRED','reservations/'+r.id)
        cancel=world.cancellations.get('cancel_'+r.id)
        if r.status in {'canceled','released','expired'}:
            if not cancel:add('CANCELLATION_RECORD_MISSING','reservations/'+r.id)
            elif cancel.canceled_at.tzinfo is None:add('CANCELLATION_TIME_INVALID','cancellations/'+cancel.id)
            elif cancel.fee_cents!=(cancellation_fee(world,r,cancel.canceled_at) if r.status=='canceled' else 0):add('CANCELLATION_FEE_INVALID','reservations/'+r.id)
            if cancel and net_paid(world,r.id)<min(paid(world,r.id),cancel.fee_cents):add('REFUND_EXCEEDS_POLICY','reservations/'+r.id)
        elif cancel:add('ACTIVE_BOOKING_CANCELED','reservations/'+r.id)
    active_assignments=[]
    for assignment in world.assignments.values():
        a=assignment;r=world.reservations.get(a.reservation_id);room=world.rooms.get(a.room_id)
        if not r or not room or a.id!='assign_'+a.reservation_id or r.property_id!=a.property_id or room.property_id!=a.property_id:
            add('ASSIGNMENT_BINDING_INVALID','assignments/'+a.id);continue
        if any(old not in world.rooms or world.rooms[old].property_id!=a.property_id for old in a.previous_rooms):add('ROOM_SWAP_HISTORY_INVALID','assignments/'+a.id)
        if a.status=='active':
            if r.status not in {'confirmed','checked_in'} or room.room_type_id!=r.room_type_id or (a.arrival,a.departure)!=(r.arrival,r.departure):add('ACTIVE_ASSIGNMENT_INVALID','assignments/'+a.id)
            if room.maintenance:add('MAINTENANCE_ROOM_ASSIGNED','assignments/'+a.id)
            if room.housekeeping!='clean':add('DIRTY_ROOM_ASSIGNED','assignments/'+a.id)
            for other in active_assignments:
                if other.room_id==a.room_id and overlap(a.arrival,a.departure,other.arrival,other.departure):add('ROOM_INTERVAL_CONFLICT','assignments/'+a.id)
            active_assignments.append(a)
    for cancellation in world.cancellations.values():
        r=world.reservations.get(cancellation.reservation_id)
        if not r or r.property_id!=cancellation.property_id or cancellation.id!='cancel_'+r.id or cancellation.canceled_at.tzinfo is None or cancellation.canceled_at>world.clock:add('CANCELLATION_BINDING_INVALID','cancellations/'+cancellation.id)
    keys=set()
    for payment in world.payments.values():
        p=payment;r=world.reservations.get(p.reservation_id);key=(p.reservation_id,p.kind,p.key)
        if not r or r.property_id!=p.property_id:add('PAYMENT_BINDING_INVALID','payments/'+p.id)
        if key in keys:add('DUPLICATE_PAYMENT_KEY','payments/'+p.id)
        keys.add(key)
        binding={'reservation_id':p.reservation_id,'kind':p.kind,'amount_cents':p.amount_cents,'key':p.key,'token':p.token,'outcome':p.outcome}
        if not attested(world,p.gateway_notice,'payment-gateway',binding) or (p.outcome=='succeeded')!=(p.token=='card_approved'):add('PAYMENT_ATTESTATION_INVALID','payments/'+p.id)
    for check in world.identity_checks.values():
        r=world.reservations.get(check.reservation_id);guest=world.guests.get(check.guest_id)
        binding={'reservation_id':check.reservation_id,'property_id':check.property_id,'guest_id':check.guest_id,'outcome':check.outcome}
        if not r or not guest or r.property_id!=check.property_id or r.guest_id!=check.guest_id or check.outcome!=guest.simulated_identity_result or not attested(world,check.notice,'guest-identity',binding):add('IDENTITY_ATTESTATION_INVALID','identity_checks/'+check.id)
    for stay in world.stays.values():
        s=stay;r=world.reservations.get(s.reservation_id);room=world.rooms.get(s.room_id);assignment=world.assignments.get('assign_'+s.reservation_id)
        if not r or not room or r.property_id!=s.property_id or room.property_id!=s.property_id or s.id!='stay_'+s.reservation_id:
            add('STAY_BINDING_INVALID','stays/'+s.id);continue
        check=world.identity_checks.get('identity_'+r.id)
        if not check or check.outcome!='verified':add('STAY_IDENTITY_REQUIRED','stays/'+s.id)
        if s.check_in_at.tzinfo is None or s.check_in_at>world.clock or not r.arrival<=s.check_in_at.date()<r.departure:add('CHECK_IN_TIME_INVALID','stays/'+s.id)
        if s.status=='checked_in' and (r.status!='checked_in' or room.current_stay_id!=s.id or not assignment or assignment.status!='active' or assignment.room_id!=s.room_id):add('STAY_ASSIGNMENT_MISMATCH','stays/'+s.id)
        if s.status=='checked_out' and (r.status!='checked_out' or not s.check_out_at or s.check_out_at.tzinfo is None or s.check_out_at<s.check_in_at or s.check_out_at>world.clock or room.current_stay_id==s.id or not assignment or assignment.status!='released'):add('CHECK_OUT_LINKAGE_INVALID','stays/'+s.id)
    return errors


def target(world,op):
    if op.name in {'create_hold','get_property','get_availability'}:return world.properties.get(op.resource_id)
    if op.name in {'mark_room_clean','set_room_maintenance','get_room'}:return world.rooms.get(op.resource_id)
    if op.name in {'change_calendar_limit','set_calendar_closed','get_calendar_day'}:return world.calendar.get(op.resource_id)
    if op.name=='get_guest':return world.guests.get(op.resource_id)
    if op.name=='get_rate_plan':return world.rate_plans.get(op.resource_id)
    return world.reservations.get(op.resource_id)


def decide(world,actor,op):
    deny=lambda code:Decision(allowed=False,error_code=code)
    if invariants(world):return deny('INVALID_WORLD')
    if actor not in world.users:return deny('UNKNOWN_ACTOR')
    prop=world.properties.get(op.scope_id)
    if not prop:return deny('PROPERTY_NOT_FOUND')
    obj=target(world,op)
    if obj is None:return deny('RESOURCE_NOT_FOUND')
    if (obj.id if op.name in {'create_hold','get_property','get_availability'} else getattr(obj,'property_id',op.scope_id))!=prop.id:return deny('PROPERTY_SCOPE_MISMATCH')
    membership=role(world,actor,prop.id)
    owner=isinstance(obj,Reservation) and world.guests[obj.guest_id].user_id==actor
    if op.is_query:
        public={'get_property','get_availability','get_rate_plan'}
        own_guest=op.name=='get_guest' and obj.user_id==actor
        own_room=op.name=='get_room' and any(a.room_id==obj.id and a.status=='active' and world.guests[world.reservations[a.reservation_id].guest_id].user_id==actor for a in world.assignments.values())
        if not membership and not owner and not own_guest and not own_room and op.name not in public:return deny('PROPERTY_PERMISSION_DENIED')
        if op.name=='get_payment_ledger' and membership not in {'manager','finance'} and not owner:return deny('FINANCIAL_PERMISSION_DENIED')
        if op.name=='get_availability':
            type_id=op.params['room_type_id']
            if type_id not in world.room_types or world.room_types[type_id].property_id!=prop.id:return deny('ROOM_TYPE_SCOPE_MISMATCH')
            if any(type_id+'_'+d.isoformat() not in world.calendar for d in days(date.fromisoformat(op.params['arrival']),date.fromisoformat(op.params['departure']))):return deny('DATE_OUT_OF_HORIZON')
        from .observations import query_data
        return Decision(allowed=True,expected_state=world.clone(),data=query_data(world,actor,op))
    guest_ops={'capture_payment','confirm_reservation','release_hold','expire_hold','change_dates','change_room_type','cancel_reservation','refund_reservation'}
    desk_ops={'create_hold','capture_payment','confirm_reservation','release_hold','expire_hold','change_dates','change_room_type','assign_room','unassign_room','cancel_reservation','verify_guest','check_in','swap_room','check_out'}
    if op.name=='create_hold':
        guest=world.guests.get(op.params['guest_id'])
        if not guest:return deny('GUEST_NOT_FOUND')
        if guest.user_id!=actor and membership not in {'manager','frontdesk'}:return deny('GUEST_IDENTITY_MISMATCH')
    elif membership=='manager' or owner and op.name in guest_ops or membership=='frontdesk' and op.name in desk_ops or membership=='housekeeping' and op.name=='mark_room_clean' or membership=='finance' and op.name in {'capture_payment','refund_reservation'}:pass
    else:return deny('PROPERTY_PERMISSION_DENIED')
    new=world.clone();p,rid,name=op.params,op.resource_id,op.name;data={}
    if name in {'create_hold','change_dates','change_room_type'}:
        old=obj if isinstance(obj,Reservation) else None
        if old and old.status not in {'hold','confirmed'}:return deny('RESERVATION_NOT_MODIFIABLE')
        if old and old.status=='hold' and world.clock>=old.hold_expires_at:return deny('HOLD_EXPIRED')
        tid=p.get('room_type_id',old.room_type_id if old else None);planid=p.get('rate_plan_id',old.rate_plan_id if old else None)
        arrival=date.fromisoformat(p['arrival']) if 'arrival' in p else old.arrival;departure=date.fromisoformat(p['departure']) if 'departure' in p else old.departure
        room_type=world.room_types.get(tid);plan=world.rate_plans.get(planid);occupants=old.occupants if old else p['occupants']
        if not room_type or room_type.property_id!=prop.id or not plan or plan.property_id!=prop.id or tid not in plan.room_type_ids:return deny('RATE_ROOM_TYPE_SCOPE_MISMATCH')
        if occupants>room_type.max_occupants:return deny('OCCUPANCY_LIMIT_EXCEEDED')
        if arrival<world.clock.date() or departure<=arrival or (departure-arrival).days>prop.max_stay_nights:return deny('STAY_INTERVAL_INVALID')
        if any(tid+'_'+d.isoformat() not in world.calendar for d in days(arrival,departure)):return deny('DATE_OUT_OF_HORIZON')
        if old and (tid,planid,arrival,departure)==(old.room_type_id,old.rate_plan_id,old.arrival,old.departure):return Decision(allowed=True,expected_state=new,data={'reservation_id':rid,'status':old.status,'total_cents':old.total_cents,'deposit_cents':old.deposit_cents})
        if not available(world,tid,arrival,departure,old.id if old else None):return deny('ROOM_TYPE_UNAVAILABLE')
        total,deposit=quote(world,tid,planid,arrival,departure)
        if old:
            assignment=world.assignments.get('assign_'+rid)
            if assignment and assignment.status=='active' and room_conflict(world,assignment.room_id,arrival,departure,rid):return deny('ROOM_INTERVAL_CONFLICT')
            booking=new.reservations[rid];booking.room_type_id=tid;booking.rate_plan_id=planid;booking.arrival=arrival;booking.departure=departure
            booking.total_cents=total;booking.deposit_cents=deposit;booking.status='hold';booking.hold_expires_at=world.clock+timedelta(minutes=30)
            if assignment:new.assignments[assignment.id].status='released'
        else:
            if p['reservation_id'] in world.reservations:return deny('RESERVATION_ID_CONFLICT')
            booking=Reservation(id=p['reservation_id'],property_id=prop.id,guest_id=p['guest_id'],room_type_id=tid,rate_plan_id=planid,
                arrival=arrival,departure=departure,occupants=occupants,total_cents=total,deposit_cents=deposit,
                created_at=world.clock,hold_expires_at=world.clock+timedelta(minutes=p['hold_minutes']))
            new.reservations[booking.id]=booking
        data={'reservation_id':booking.id,'status':booking.status,'total_cents':total,'deposit_cents':deposit}
    elif name in {'capture_payment','refund_reservation'}:
        kind='charge' if name=='capture_payment' else 'refund';token=p.get('token','card_approved')
        prior=next((pay for pay in world.payments.values() if pay.reservation_id==rid and pay.kind==kind and pay.key==p['key']),None)
        if prior:
            if prior.amount_cents!=p['amount_cents'] or prior.token!=token:return deny('IDEMPOTENCY_CONFLICT')
            return Decision(allowed=True,expected_state=new,data={'payment_id':prior.id,'outcome':prior.outcome})
        if kind=='charge':
            if obj.status in {'checked_out','released','expired'}:return deny('RESERVATION_PAYMENT_CLOSED')
            if obj.status=='hold' and world.clock>=obj.hold_expires_at:return deny('HOLD_EXPIRED')
            due=world.cancellations['cancel_'+rid].fee_cents if obj.status=='canceled' else obj.total_cents
            if p['amount_cents']>max(0,due-net_paid(world,rid)):return deny('PAYMENT_EXCEEDS_DUE')
        elif p['amount_cents']>refund_limit(world,obj):return deny('REFUND_EXCEEDS_ELIGIBLE')
        outcome='succeeded' if token=='card_approved' else 'failed'
        binding={'reservation_id':rid,'kind':kind,'amount_cents':p['amount_cents'],'key':p['key'],'token':token,'outcome':outcome}
        key='payment_'+digest([rid,kind,p['key']])[:16]
        new.payments[key]=Payment(id=key,property_id=prop.id,reservation_id=rid,kind=kind,amount_cents=p['amount_cents'],
            key=p['key'],token=token,outcome=outcome,gateway_notice=signed_notice(world.domain,'payment-gateway',binding,world.clock))
        data={'payment_id':key,'outcome':outcome}
    elif name=='confirm_reservation':
        if obj.status=='confirmed':return Decision(allowed=True,expected_state=new,data={'reservation_id':rid,'status':'confirmed'})
        if obj.status!='hold':return deny('RESERVATION_NOT_A_HOLD')
        if world.clock>=obj.hold_expires_at:return deny('HOLD_EXPIRED')
        if net_paid(world,rid)<obj.deposit_cents:return deny('DEPOSIT_REQUIRED')
        new.reservations[rid].status='confirmed';data={'reservation_id':rid,'status':'confirmed'}
    elif name in {'release_hold','expire_hold','cancel_reservation'}:
        end={'release_hold':'released','expire_hold':'expired','cancel_reservation':'canceled'}[name]
        if obj.status==end:return Decision(allowed=True,expected_state=new,data={'reservation_id':rid,'status':end,'fee_cents':world.cancellations['cancel_'+rid].fee_cents})
        if name in {'release_hold','expire_hold'} and obj.status!='hold':return deny('RESERVATION_NOT_A_HOLD')
        if name=='expire_hold' and world.clock<obj.hold_expires_at:return deny('HOLD_NOT_EXPIRED')
        if name=='cancel_reservation' and obj.status!='confirmed':return deny('RESERVATION_NOT_CANCELABLE')
        fee=cancellation_fee(world,obj,world.clock) if name=='cancel_reservation' else 0
        new.reservations[rid].status=end
        assignment=new.assignments.get('assign_'+rid)
        if assignment:assignment.status='released'
        key='cancel_'+rid;new.cancellations[key]=Cancellation(id=key,property_id=prop.id,reservation_id=rid,canceled_at=world.clock,fee_cents=fee)
        data={'reservation_id':rid,'status':end,'fee_cents':fee}
    elif name=='assign_room':
        room=world.rooms.get(p['room_id'])
        if not room or room.property_id!=prop.id or room.room_type_id!=obj.room_type_id:return deny('ROOM_TYPE_SCOPE_MISMATCH')
        if obj.status!='confirmed':return deny('RESERVATION_NOT_CONFIRMED')
        prior=world.assignments.get('assign_'+rid)
        if prior and prior.status=='active':
            if prior.room_id!=room.id:return deny('ASSIGNMENT_ALREADY_EXISTS')
            return Decision(allowed=True,expected_state=new,data={'assignment_id':prior.id,'room_id':room.id})
        if room.maintenance:return deny('ROOM_IN_MAINTENANCE')
        if room.housekeeping!='clean':return deny('ROOM_NOT_CLEAN')
        if room.current_stay_id or room_conflict(world,room.id,obj.arrival,obj.departure,rid):return deny('ROOM_INTERVAL_CONFLICT')
        key='assign_'+rid;new.assignments[key]=Assignment(id=key,property_id=prop.id,reservation_id=rid,room_id=room.id,arrival=obj.arrival,departure=obj.departure,
            previous_rooms=prior.previous_rooms if prior else [])
        data={'assignment_id':key,'room_id':room.id}
    elif name=='unassign_room':
        assignment=world.assignments.get('assign_'+rid)
        if obj.status!='confirmed' or not assignment or assignment.status!='active':return deny('ASSIGNMENT_NOT_RELEASABLE')
        new.assignments[assignment.id].status='released';data={'assignment_id':assignment.id,'status':'released'}
    elif name=='mark_room_clean':
        if obj.current_stay_id:return deny('ROOM_OCCUPIED')
        new.rooms[rid].housekeeping='clean';data={'room_id':rid,'housekeeping':'clean'}
    elif name=='set_room_maintenance':
        if p['enabled'] and (obj.current_stay_id or any(a.room_id==rid and a.status=='active' for a in world.assignments.values())):return deny('ROOM_ASSIGNED')
        physical=sum(r.room_type_id==obj.room_type_id and not r.maintenance and r.id!=rid for r in world.rooms.values())+(0 if p['enabled'] else 1)
        if p['enabled'] and any(occupied(world,obj.room_type_id,day)>min(row.sale_limit,physical) for row in world.calendar.values() if row.room_type_id==obj.room_type_id for day in [row.day]):return deny('MAINTENANCE_OVERBOOKS_TYPE')
        new.rooms[rid].maintenance=p['enabled'];data={'room_id':rid,'maintenance':p['enabled']}
    elif name in {'change_calendar_limit','set_calendar_closed'}:
        count=occupied(world,obj.room_type_id,obj.day)
        if name=='change_calendar_limit':
            physical=sum(r.room_type_id==obj.room_type_id for r in world.rooms.values())
            if p['sale_limit']>physical:return deny('CALENDAR_LIMIT_EXCEEDS_ROOMS')
            if p['sale_limit']<count:return deny('CALENDAR_LIMIT_BELOW_BOOKINGS')
            new.calendar[rid].sale_limit=p['sale_limit'];data={'calendar_id':rid,'sale_limit':p['sale_limit']}
        else:
            if p['enabled'] and count:return deny('CALENDAR_HAS_BOOKINGS')
            new.calendar[rid].closed=p['enabled'];data={'calendar_id':rid,'closed':p['enabled']}
    elif name=='verify_guest':
        guest=world.guests[obj.guest_id]
        binding={'reservation_id':rid,'property_id':prop.id,'guest_id':guest.id,'outcome':guest.simulated_identity_result}
        key='identity_'+rid;new.identity_checks[key]=IdentityCheck(id=key,property_id=prop.id,guest_id=guest.id,reservation_id=rid,
            outcome=guest.simulated_identity_result,notice=signed_notice(world.domain,'guest-identity',binding,world.clock))
        data={'check_id':key,'outcome':guest.simulated_identity_result}
    elif name=='check_in':
        if obj.status=='checked_in':return Decision(allowed=True,expected_state=new,data={'stay_id':'stay_'+rid,'status':'checked_in'})
        if obj.status!='confirmed':return deny('RESERVATION_NOT_CONFIRMED')
        if not obj.arrival<=world.clock.date()<obj.departure:return deny('OUTSIDE_CHECK_IN_WINDOW')
        assignment=world.assignments.get('assign_'+rid)
        if not assignment or assignment.status!='active':return deny('ROOM_ASSIGNMENT_REQUIRED')
        room=world.rooms[assignment.room_id]
        if room.maintenance:return deny('ROOM_IN_MAINTENANCE')
        if room.housekeeping!='clean':return deny('ROOM_NOT_CLEAN')
        if room.current_stay_id:return deny('ROOM_OCCUPIED')
        if net_paid(world,rid)<obj.total_cents:return deny('PAYMENT_REQUIRED')
        check=world.identity_checks.get('identity_'+rid)
        if not check or check.outcome!='verified':return deny('GUEST_IDENTITY_REQUIRED')
        key='stay_'+rid;new.stays[key]=Stay(id=key,property_id=prop.id,reservation_id=rid,room_id=room.id,check_in_at=world.clock)
        new.reservations[rid].status='checked_in';new.rooms[room.id].current_stay_id=key;data={'stay_id':key,'status':'checked_in'}
    elif name=='swap_room':
        if obj.status!='checked_in':return deny('RESERVATION_NOT_CHECKED_IN')
        room=world.rooms.get(p['room_id']);assignment=world.assignments['assign_'+rid];stay=world.stays['stay_'+rid]
        if not room or room.property_id!=prop.id or room.room_type_id!=obj.room_type_id:return deny('ROOM_TYPE_SCOPE_MISMATCH')
        if room.id==assignment.room_id:return Decision(allowed=True,expected_state=new,data={'stay_id':stay.id,'room_id':room.id})
        if room.maintenance:return deny('ROOM_IN_MAINTENANCE')
        if room.housekeeping!='clean':return deny('ROOM_NOT_CLEAN')
        if room.current_stay_id or room_conflict(world,room.id,obj.arrival,obj.departure,rid):return deny('ROOM_INTERVAL_CONFLICT')
        old=assignment.room_id
        new.rooms[old].current_stay_id=None;new.rooms[old].housekeeping='dirty';new.rooms[room.id].current_stay_id=stay.id
        new.assignments[assignment.id].previous_rooms.append(old);new.assignments[assignment.id].room_id=room.id;new.stays[stay.id].room_id=room.id
        data={'stay_id':stay.id,'room_id':room.id}
    elif name=='check_out':
        if obj.status=='checked_out':return Decision(allowed=True,expected_state=new,data={'stay_id':'stay_'+rid,'status':'checked_out'})
        if obj.status!='checked_in':return deny('RESERVATION_NOT_CHECKED_IN')
        stay=new.stays['stay_'+rid];assignment=new.assignments['assign_'+rid];room=new.rooms[stay.room_id]
        stay.status='checked_out';stay.check_out_at=world.clock;new.reservations[rid].status='checked_out';assignment.status='released'
        room.current_stay_id=None;room.housekeeping='dirty';data={'stay_id':stay.id,'status':'checked_out'}
    else:return deny('UNKNOWN_OPERATION')
    return Decision(allowed=True,expected_state=new,data=data)
