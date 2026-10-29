"""Actual hotel mutations, independent of Reference predicates and snapshots."""
from datetime import date,datetime,time,timedelta,timezone
from usami_harness.models import Result,digest,signed_notice
from .models import Reservation,Assignment,Payment,Cancellation,IdentityCheck,Stay


def _quote(world,tid,planid,arrival,departure):
    total=0;day=arrival
    while day<departure:
        total+=world.room_types[tid].nightly_cents
        if day.weekday() in (5,6):total+=world.rate_plans[planid].weekend_surcharge_cents
        day+=timedelta(days=1)
    bps=world.rate_plans[planid].deposit_bps
    return total,(total*bps+9999)//10000


def apply(world,actor,op,event_index):
    new=world.clone();name,rid,p=op.name,op.resource_id,op.params;scope=op.scope_id
    fail=lambda: (world.clone(),Result(success=False,error_code='MUTATION_NOT_REPRESENTABLE'))
    try:
        if scope not in new.properties or actor not in new.users:return fail()
        if op.is_query:
            from .observations import query_data
            return new,Result(success=True,data=query_data(world,actor,op))
        data={}
        if name=='create_hold':
            if rid not in new.properties or p['guest_id'] not in new.guests or p['room_type_id'] not in new.room_types or p['rate_plan_id'] not in new.rate_plans:return fail()
            if p['reservation_id'] in new.reservations:return fail()
            arrival,departure=date.fromisoformat(p['arrival']),date.fromisoformat(p['departure'])
            total,deposit=_quote(new,p['room_type_id'],p['rate_plan_id'],arrival,departure)
            reservation=Reservation(id=p['reservation_id'],property_id=scope,guest_id=p['guest_id'],room_type_id=p['room_type_id'],rate_plan_id=p['rate_plan_id'],
                arrival=arrival,departure=departure,occupants=p['occupants'],total_cents=total,deposit_cents=deposit,created_at=new.clock,
                hold_expires_at=new.clock+timedelta(minutes=p['hold_minutes']))
            new.reservations[reservation.id]=reservation
            data={'reservation_id':reservation.id,'status':reservation.status,'total_cents':total,'deposit_cents':deposit}
        elif name in {'capture_payment','refund_reservation'}:
            if rid not in new.reservations:return fail()
            kind='charge' if name=='capture_payment' else 'refund';token=p.get('token','card_approved')
            prior=next((pay for pay in new.payments.values() if pay.reservation_id==rid and pay.kind==kind and pay.key==p['key']),None)
            if prior and prior.amount_cents==p['amount_cents'] and prior.token==token:
                return new,Result(success=True,data={'payment_id':prior.id,'outcome':prior.outcome})
            key='payment_'+digest([rid,kind,p['key']])[:16]
            if key in new.payments:key+='_attempt_'+str(event_index)
            outcome='succeeded' if token=='card_approved' else 'failed'
            binding={'reservation_id':rid,'kind':kind,'amount_cents':p['amount_cents'],'key':p['key'],'token':token,'outcome':outcome}
            new.payments[key]=Payment(id=key,property_id=scope,reservation_id=rid,kind=kind,amount_cents=p['amount_cents'],key=p['key'],token=token,outcome=outcome,
                gateway_notice=signed_notice(new.domain,'payment-gateway',binding,new.clock))
            data={'payment_id':key,'outcome':outcome}
        elif name=='confirm_reservation':
            reservation=new.reservations[rid];reservation.status='confirmed';data={'reservation_id':rid,'status':'confirmed'}
        elif name in {'release_hold','expire_hold','cancel_reservation'}:
            reservation=new.reservations[rid]
            end={'release_hold':'released','expire_hold':'expired','cancel_reservation':'canceled'}[name]
            old_cancel=new.cancellations.get('cancel_'+rid)
            if reservation.status==end and old_cancel:return new,Result(success=True,data={'reservation_id':rid,'status':end,'fee_cents':old_cancel.fee_cents})
            fee=0
            if name=='cancel_reservation':
                plan=new.rate_plans[reservation.rate_plan_id]
                cutoff=datetime.combine(reservation.arrival,time.min,tzinfo=timezone.utc)-timedelta(hours=plan.free_cancel_hours)
                if new.clock>cutoff:
                    fee=reservation.total_cents if plan.late_fee=='full_stay' else new.room_types[reservation.room_type_id].nightly_cents+(plan.weekend_surcharge_cents if reservation.arrival.weekday()>=5 else 0)
            reservation.status=end
            assignment=new.assignments.get('assign_'+rid)
            if assignment:assignment.status='released'
            key='cancel_'+rid;new.cancellations[key]=Cancellation(id=key,property_id=scope,reservation_id=rid,canceled_at=new.clock,fee_cents=fee)
            data={'reservation_id':rid,'status':end,'fee_cents':fee}
        elif name in {'change_dates','change_room_type'}:
            reservation=new.reservations[rid]
            tid=p.get('room_type_id',reservation.room_type_id);plan=p.get('rate_plan_id',reservation.rate_plan_id)
            arrival=date.fromisoformat(p['arrival']) if 'arrival' in p else reservation.arrival
            departure=date.fromisoformat(p['departure']) if 'departure' in p else reservation.departure
            if (tid,plan,arrival,departure)==(reservation.room_type_id,reservation.rate_plan_id,reservation.arrival,reservation.departure):
                return new,Result(success=True,data={'reservation_id':rid,'status':reservation.status,'total_cents':reservation.total_cents,'deposit_cents':reservation.deposit_cents})
            total,deposit=_quote(new,tid,plan,arrival,departure)
            reservation.room_type_id=tid;reservation.rate_plan_id=plan;reservation.arrival=arrival;reservation.departure=departure
            reservation.total_cents=total;reservation.deposit_cents=deposit;reservation.status='hold';reservation.hold_expires_at=new.clock+timedelta(minutes=30)
            assignment=new.assignments.get('assign_'+rid)
            if assignment:assignment.status='released'
            data={'reservation_id':rid,'status':'hold','total_cents':total,'deposit_cents':deposit}
        elif name=='assign_room':
            reservation=new.reservations[rid];room=new.rooms[p['room_id']];key='assign_'+rid
            prior=new.assignments.get(key)
            if prior and prior.status=='active' and prior.room_id==room.id:
                return new,Result(success=True,data={'assignment_id':key,'room_id':room.id})
            new.assignments[key]=Assignment(id=key,property_id=scope,reservation_id=rid,room_id=room.id,arrival=reservation.arrival,departure=reservation.departure,
                previous_rooms=prior.previous_rooms if prior else [])
            data={'assignment_id':key,'room_id':room.id}
        elif name=='unassign_room':
            assignment=new.assignments['assign_'+rid];assignment.status='released';data={'assignment_id':assignment.id,'status':'released'}
        elif name=='mark_room_clean':
            new.rooms[rid].housekeeping='clean';data={'room_id':rid,'housekeeping':'clean'}
        elif name=='set_room_maintenance':
            new.rooms[rid].maintenance=p['enabled'];data={'room_id':rid,'maintenance':p['enabled']}
        elif name=='change_calendar_limit':
            new.calendar[rid].sale_limit=p['sale_limit'];data={'calendar_id':rid,'sale_limit':p['sale_limit']}
        elif name=='set_calendar_closed':
            new.calendar[rid].closed=p['enabled'];data={'calendar_id':rid,'closed':p['enabled']}
        elif name=='verify_guest':
            reservation=new.reservations[rid];guest=new.guests[reservation.guest_id];key='identity_'+rid
            binding={'reservation_id':rid,'property_id':scope,'guest_id':guest.id,'outcome':guest.simulated_identity_result}
            new.identity_checks[key]=IdentityCheck(id=key,property_id=scope,guest_id=guest.id,reservation_id=rid,outcome=guest.simulated_identity_result,
                notice=signed_notice(new.domain,'guest-identity',binding,new.clock))
            data={'check_id':key,'outcome':guest.simulated_identity_result}
        elif name=='check_in':
            reservation=new.reservations[rid];assignment=new.assignments['assign_'+rid];room=new.rooms[assignment.room_id];key='stay_'+rid
            if reservation.status=='checked_in' and key in new.stays:return new,Result(success=True,data={'stay_id':key,'status':'checked_in'})
            new.stays[key]=Stay(id=key,property_id=scope,reservation_id=rid,room_id=room.id,check_in_at=new.clock)
            reservation.status='checked_in';room.current_stay_id=key;data={'stay_id':key,'status':'checked_in'}
        elif name=='swap_room':
            assignment=new.assignments['assign_'+rid];stay=new.stays['stay_'+rid];room=new.rooms[p['room_id']]
            if room.id!=assignment.room_id:
                old=new.rooms[assignment.room_id];old.current_stay_id=None;old.housekeeping='dirty'
                room.current_stay_id=stay.id;assignment.previous_rooms.append(old.id);assignment.room_id=room.id;stay.room_id=room.id
            data={'stay_id':stay.id,'room_id':room.id}
        elif name=='check_out':
            reservation=new.reservations[rid];stay=new.stays['stay_'+rid];assignment=new.assignments['assign_'+rid];room=new.rooms[stay.room_id]
            if reservation.status=='checked_out':return new,Result(success=True,data={'stay_id':stay.id,'status':'checked_out'})
            stay.status='checked_out';stay.check_out_at=new.clock;reservation.status='checked_out';assignment.status='released';room.current_stay_id=None;room.housekeeping='dirty'
            data={'stay_id':stay.id,'status':'checked_out'}
        else:return fail()
        return new,Result(success=True,data=data)
    except (ValueError,KeyError,TypeError):return fail()
