"""Hotel-specific atomic compositions and explicit mutable-field declarations."""
from datetime import timedelta
from usami_harness.models import digest
from usami_harness.goals import AtomicGoal,JointGoal,Task,Witness,execute,freeze_task
from .models import Operation
from .generator import generate_world
from .rules import quote,net_paid,refund_limit

GOAL_FIELDS={'rooms':['housekeeping','maintenance','current_stay_id'],'calendar':['sale_limit','closed'],
    'reservations':['status','room_type_id','rate_plan_id','arrival','departure','total_cents','deposit_cents'],
    'assignments':['room_id','status','arrival','departure','previous_rooms'],'payments':['outcome','amount_cents','kind'],
    'cancellations':['fee_cents'],'identity_checks':['outcome'],'stays':['status','room_id']}
FRAME_FIELDS={'rooms':['housekeeping','maintenance','current_stay_id'],'calendar':['sale_limit','closed'],
    'reservations':['status','room_type_id','rate_plan_id','arrival','departure','total_cents','deposit_cents','hold_expires_at'],
    'assignments':['room_id','arrival','departure','status','previous_rooms'],'payments':[],
    'cancellations':[],'identity_checks':[],'stays':['status','room_id','check_out_at']}
DELETABLE=frozenset()
FAMILY={
    'occupied_room_link':'guest_check_in',
    'deposit_refusal':'deposit_integrity','refund_refusal':'refund_eligibility','overpay_refusal':'payment_integrity',
    'payment_key_refusal':'payment_integrity','expired_confirmation_refusal':'hold_expiration',
    'early_expiration_refusal':'hold_expiration','occupied_clean_refusal':'room_cleaning',
    'dirty_assignment_refusal':'room_assignment','maintenance_assignment_refusal':'maintenance_recovery',
    'overbooking_refusal':'date_inventory','overlap_refusal':'room_assignment',
    'scope_refusal':'property_authorization','staff_role_refusal':'property_authorization',
    'guest_staff_refusal':'property_authorization','occupancy_refusal':'guest_capacity',
    'early_checkin_refusal':'guest_check_in','identity_refusal':'guest_identity','balance_refusal':'guest_check_in',
    'date_conflict_refusal':'booking_date_change','calendar_reduction_refusal':'calendar_capacity',
    'calendar_closure_refusal':'calendar_reopening','maintenance_lock_refusal':'maintenance_scheduling',
    'cancel_stay_refusal':'booking_cancellation','swap_conflict_refusal':'room_swap',
    'cross_type_refusal':'booking_upgrade','unassign_stay_refusal':'room_unassignment',
}


class Builder:
    def __init__(self,seed,family):
        self.world,self.family,self.identity=generate_world(seed),family,'u0'
        self.ops,self.atoms,self.frame=[],[],{}

    def op(self,name,rid='b0',scope='p0',**params):return Operation(name=name,scope_id=scope,resource_id=rid,params=params)

    def hold(self,booking='b0',kind='standard',arrival='2026-01-15',departure='2026-01-17',plan='flex',guest='g0',scope='p0'):
        return self.op('create_hold',scope,scope,reservation_id=booking,guest_id=guest,room_type_id=scope+'_'+kind,
            rate_plan_id=scope+'_'+plan,arrival=arrival,departure=departure,occupants=2,hold_minutes=30)

    def payment(self,booking='b0',amount=2400,key='deposit',scope='p0'):
        return self.op('capture_payment',booking,scope,amount_cents=amount,key=key,token='card_approved')

    def prepare(self,*ops):
        from . import adapter
        self.world=adapter.World.model_validate(execute(adapter,self.world,'u0',list(ops)).final_state)

    def confirmed(self,booking='b0',**kwargs):
        hold=self.hold(booking,**kwargs);p=hold.params
        from datetime import date
        _,deposit=quote(self.world,p['room_type_id'],p['rate_plan_id'],date.fromisoformat(p['arrival']),date.fromisoformat(p['departure']))
        scope=kwargs.get('scope','p0')
        self.prepare(hold,self.payment(booking,deposit,scope=scope),self.op('confirm_reservation',booking,scope))

    def arrived(self,booking='b0',room='p0_standard_1'):
        self.confirmed(booking)
        self.prepare(self.op('assign_room',booking,room_id=room),self.payment(booking,9600,'balance'),
                     self.op('verify_guest',booking),self.op('check_in',booking))

    def add(self,*ops):self.ops.extend(ops)
    def terminal(self,collection,entity,field,value,comparison='eq'):
        self.atoms.append(AtomicGoal(collection=collection,entity=entity,field=field,value=value,comparison=comparison))
    def allow(self,collection,entity,*fields):
        key=collection+'/'+entity
        if entity not in getattr(self.world,collection):fields=('*',)
        if not fields:return
        if self.frame.get(key)==['*']:return
        self.frame[key]=sorted(set(self.frame.get(key,[])+list(fields)))

    def booking_flow(self,booking='b0',kind='standard',arrival='2026-01-15',departure='2026-01-17',plan='flex',include_hold=True,guest='g0'):
        if include_hold:self.add(self.hold(booking,kind,arrival,departure,plan,guest))
        from datetime import date
        _,deposit=quote(self.world,'p0_'+kind,'p0_'+plan,date.fromisoformat(arrival),date.fromisoformat(departure))
        self.add(self.payment(booking,deposit),self.op('confirm_reservation',booking))
        self.terminal('reservations',booking,'status','confirmed')

    def arrival_flow(self,booking='b0',room='p0_standard_0',create=True,clean=True,alternatives=False,checkout=False):
        if create:self.add(self.hold(booking),self.payment(booking),self.op('confirm_reservation',booking))
        if clean:self.add(self.op('mark_room_clean',room))
        self.add(self.op('assign_room',booking,room_id=room))
        if alternatives:self.add(self.op('assign_room',booking,room_id='p0_standard_1'))
        self.add(self.payment(booking,9600,'balance'),self.op('verify_guest',booking),self.op('check_in',booking))
        if checkout:self.add(self.op('check_out',booking))
        self.terminal('reservations',booking,'status','checked_out' if checkout else 'checked_in')

    def effects(self):
        rooms={r.id:set() for r in self.world.reservations.values()}
        for a in self.world.assignments.values():rooms.setdefault(a.reservation_id,set()).add(a.room_id)
        for op in self.ops:
            if op.name in {'assign_room','swap_room'}:rooms.setdefault(op.resource_id,set()).add(op.params['room_id'])
        for op in self.ops:
            name,rid,p=op.name,op.resource_id,op.params
            if name=='create_hold':self.allow('reservations',p['reservation_id'])
            elif name in {'capture_payment','refund_reservation'}:
                kind='charge' if name=='capture_payment' else 'refund';self.allow('payments','payment_'+digest([rid,kind,p['key']])[:16])
            elif name in {'confirm_reservation','release_hold','expire_hold','cancel_reservation','change_dates','change_room_type','check_in','check_out'}:
                fields=['status']
                if name in {'change_dates','change_room_type'}:fields+=['arrival','departure','room_type_id','rate_plan_id','total_cents','deposit_cents','hold_expires_at']
                self.allow('reservations',rid,*fields)
                if name in {'release_hold','expire_hold','cancel_reservation'}:self.allow('cancellations','cancel_'+rid)
                if name in {'release_hold','expire_hold','cancel_reservation','change_dates','change_room_type','check_out'} and ('assign_'+rid in self.world.assignments or rooms.get(rid)):
                    self.allow('assignments','assign_'+rid,'status')
            if name in {'assign_room','unassign_room','swap_room'}:
                self.allow('assignments','assign_'+rid,'room_id','arrival','departure','status','previous_rooms')
            if name in {'check_in','swap_room','check_out'}:
                self.allow('stays','stay_'+rid,'status','room_id','check_out_at')
                for room in rooms.get(rid,set()):self.allow('rooms',room,'current_stay_id','housekeeping')
            if name=='mark_room_clean':self.allow('rooms',rid,'housekeeping')
            if name=='set_room_maintenance':self.allow('rooms',rid,'maintenance')
            if name=='change_calendar_limit':self.allow('calendar',rid,'sale_limit')
            if name=='set_calendar_closed':self.allow('calendar',rid,'closed')
            if name=='verify_guest':self.allow('identity_checks','identity_'+rid)

    def freeze(self,message,refusal=None):
        from . import adapter
        if refusal:self.atoms=[AtomicGoal(collection='rooms',entity='p0_standard_0',field='housekeeping',value=self.world.rooms['p0_standard_0'].housekeeping)]
        else:self.effects()
        task=Task(domain=adapter.name,version=adapter.version,family=FAMILY.get(self.family,self.family),identity=self.identity,
            kind='policy_refusal' if refusal else 'completion',message=message,initial_state_hash=self.world.state_hash,
            goal=JointGoal(atoms=self.atoms,allowed_changes=self.frame),capabilities=[op.model_dump(mode='json') for op in self.ops],refusal_code=refusal)
        return self.world,freeze_task(adapter,self.world,task)


def make_task(seed,scenario):
    if '@' in scenario:
        base,route,phase=scenario.rsplit('@',2);world,task=make_task(seed,base)
        return progress_task(world,task,int(route),int(phase))
    family,_,variant=scenario.partition(':');b=Builder(seed,family);op=b.op
    if family=='booking_confirmation':
        kind='suite' if variant=='suite' else 'standard';plan='prepaid' if variant=='prepaid' else 'flex'
        if variant=='guest':b.identity='u3'
        b.booking_flow(kind=kind,plan=plan,arrival='2026-01-20',departure='2026-01-22')
        if variant=='assigned':
            b.add(op('assign_room',room_id='p0_standard_1'));b.terminal('assignments','assign_b0','status','active')
        message=f'Book b0 for g0, two occupants, p0_{kind}/p0_{plan}, January 20–22, 30-minute hold. Pay the required deposit with card_approved/deposit and confirm.'
        if variant=='assigned':message+=' Assign p0_standard_1 afterward.'
    elif family in {'arrival_preparation','guest_check_in'}:
        if family=='arrival_preparation':
            b.confirmed();b.add(op('mark_room_clean','p0_standard_0'),op('assign_room',room_id='p0_standard_0'));b.terminal('assignments','assign_b0','status','active')
            message='Prepare confirmed b0 arrival: clean and assign p0_standard_0 without changing price or payments.'
        elif variant=='frontdesk':
            b.confirmed();b.identity='u1';b.arrival_flow(room='p0_standard_1',create=False,clean=False)
            message='Check in confirmed b0 using clean p0_standard_1, verify guest identity and collect the remaining 9600 with card_approved/balance.'
        else:
            b.arrival_flow(alternatives=variant=='choice')
            message='Arrange b0 for g0, two occupants, p0_standard/p0_flex, Jan 15–17, hold 30 minutes. Pay 2400 deposit then 9600 balance (card_approved/deposit/balance), confirm, verify guest and check in using p0_standard_0 after cleaning.'+(' Either standard room is acceptable.' if variant=='choice' else '')
    elif family=='room_swap':
        b.arrived();b.add(op('mark_room_clean','p0_standard_0'),op('swap_room',room_id='p0_standard_0'))
        b.terminal('stays','stay_b0','room_id','p0_standard_0');b.terminal('rooms','p0_standard_1','current_stay_id',None)
        message='Move checked-in b0 from standard_1 to p0_standard_0 after cleaning it, atomically updating occupancy, assignment and stay.'
    elif family=='departure_turnover':
        b.arrived();b.add(op('check_out'),op('mark_room_clean','p0_standard_1'))
        b.terminal('stays','stay_b0','status','checked_out');b.terminal('rooms','p0_standard_1','housekeeping','clean')
        message='Check b0 out and clean its vacated p0_standard_1 for resale, retaining payment history.'
    elif family=='booking_date_change':
        if variant=='arrival':
            b.confirmed(arrival='2026-01-20',departure='2026-01-22');b.prepare(op('assign_room',room_id='p0_standard_1'))
            b.add(op('change_dates',arrival='2026-01-15',departure='2026-01-17'),op('confirm_reservation'),
                  op('assign_room',room_id='p0_standard_1'),b.payment(amount=9600,key='balance'),op('verify_guest'),op('check_in'))
            b.terminal('reservations','b0','status','checked_in');b.terminal('reservations','b0','arrival','2026-01-15')
            return b.freeze('Move future b0 to Jan15–17 after validating all nights, reconfirm and reassign p0_standard_1; collect9600/card_approved/balance, verify identity and check in. Preserve unrelated bookings/rooms.')
        b.confirmed();b.prepare(op('assign_room',room_id='p0_standard_1'))
        b.add(op('change_dates',arrival='2026-01-16',departure='2026-01-18'),b.payment(amount=200,key='difference'),op('confirm_reservation'),op('assign_room',room_id='p0_standard_1'))
        b.terminal('reservations','b0','status','confirmed');b.terminal('reservations','b0','arrival','2026-01-16');b.terminal('assignments','assign_b0','arrival','2026-01-16');b.terminal('assignments','assign_b0','status','active')
        message='Move b0 to Jan 16–18, revalidate both nights, pay the 200 deposit difference with card_approved/difference, reconfirm and reassign p0_standard_1.'
    elif family=='booking_upgrade':
        b.confirmed();b.prepare(op('assign_room',room_id='p0_standard_1'))
        b.add(op('change_room_type',room_type_id='p0_suite',rate_plan_id='p0_flex'),b.payment(amount=1600,key='difference'),op('confirm_reservation'),op('assign_room',room_id='p0_suite_1'))
        b.terminal('reservations','b0','room_type_id','p0_suite');b.terminal('reservations','b0','status','confirmed');b.terminal('assignments','assign_b0','room_id','p0_suite_1')
        message='Upgrade b0 to p0_suite under p0_flex; pay 1600 more with card_approved/difference, reconfirm and assign p0_suite_1, releasing its former standard room.'
    elif family=='booking_downgrade':
        b.confirmed(kind='suite');b.prepare(b.payment(amount=16000,key='balance'),op('assign_room',room_id='p0_suite_1'))
        b.add(op('change_room_type',room_type_id='p0_standard',rate_plan_id='p0_flex'),op('refund_reservation',amount_cents=8000,key='downgrade'),op('confirm_reservation'),op('assign_room',room_id='p0_standard_1'))
        b.terminal('reservations','b0','room_type_id','p0_standard');b.terminal('assignments','assign_b0','room_id','p0_standard_1');b.terminal('payments','payment_'+digest(['b0','refund','downgrade'])[:16],'outcome','succeeded')
        if variant=='arrival':
            b.add(op('verify_guest'),op('check_in'));b.terminal('reservations','b0','status','checked_in')
        message='Downgrade prepaid b0 to p0_standard/p0_flex, refund only the 8000 overpayment with downgrade, reconfirm and move assignment to p0_standard_1.'
        if variant=='arrival':message+=' Verify the guest and check in on the same dates.'
    elif family in {'booking_cancellation','cancellation_refund'}:
        future=variant!='late';plan='prepaid' if variant=='prepaid' else 'flex'
        b.confirmed(arrival='2026-01-20' if future else '2026-01-15',departure='2026-01-22' if future else '2026-01-17',plan=plan)
        if family=='cancellation_refund':
            due=b.world.reservations['b0'].total_cents-net_paid(b.world,'b0')
            if due:b.prepare(b.payment(amount=due,key='balance'))
        b.add(op('cancel_reservation'));b.terminal('reservations','b0','status','canceled')
        if family=='cancellation_refund':
            amount=6000 if not future else 12000
            b.add(op('refund_reservation',amount_cents=amount,key='cancelled'));b.terminal('payments','payment_'+digest(['b0','refund','cancelled'])[:16],'outcome','succeeded')
        message='Cancel confirmed b0 using its frozen-time plan policy, release its dates/room.'+(' Refund only the eligible '+str(amount)+' cents with cancelled.' if family=='cancellation_refund' else '')
    elif family=='deposit_completion':
        b.prepare(b.hold(arrival='2026-01-20',departure='2026-01-22'),b.payment(amount=1200,key='initial'))
        b.add(b.payment(amount=1200),op('confirm_reservation'));b.terminal('reservations','b0','status','confirmed')
        message='Complete the deposit of b0 with 1200 additional cents/card_approved/deposit and confirm, retaining its initial 1200 payment.'
    elif family in {'hold_release','hold_expiration'}:
        b.prepare(b.hold())
        if variant in {'refund','guest_refund'}:
            b.prepare(b.payment());b.add(op('release_hold'),op('refund_reservation',amount_cents=2400,key='released'))
            b.terminal('reservations','b0','status','released');b.terminal('payments','payment_'+digest(['b0','refund','released'])[:16],'outcome','succeeded')
            if variant=='guest_refund':b.identity='u3'
            return b.freeze('Release unconfirmed paid b0 hold, freeing all nights with no cancellation fee; refund its2400/card-approved balance using released. Preserve other entities.')
        if family=='hold_expiration':
            b.world.reservations['b0'].created_at-=timedelta(hours=2);b.world.reservations['b0'].hold_expires_at-=timedelta(hours=2)
        b.add(op('expire_hold') if family=='hold_expiration' else op('release_hold'));b.terminal('reservations','b0','status','expired' if family=='hold_expiration' else 'released')
        message='Explicitly '+('expire the elapsed' if family=='hold_expiration' else 'release the live')+' temporary hold b0 and free all its nights.'
    elif family=='maintenance_recovery':
        if variant=='arrival':
            b.world.rooms['p0_standard_0'].maintenance=True;b.prepare(b.hold('busy'))
            b.add(op('set_room_maintenance','p0_standard_0',enabled=False));b.arrival_flow(alternatives=False)
            return b.freeze('Restore p0_standard_0 from maintenance and clean it. Retain busy hold; arrange b0/g0/p0_standard/p0_flex/Jan15–17/two guests/hold30, pay2400 deposit and9600 balance with card_approved/deposit/balance, confirm, assign restored room, verify identity and check in.')
        b.prepare(b.hold('busy',kind='suite'))
        b.add(op('set_room_maintenance','p0_suite_0',enabled=False));b.booking_flow(kind='suite',arrival='2026-01-15',departure='2026-01-17')
        b.add(op('assign_room',room_id='p0_suite_0'));b.terminal('assignments','assign_b0','room_id','p0_suite_0')
        message='Restore p0_suite_0 from maintenance, then book/confirm b0 g0/p0_suite/p0_flex Jan 15–17, two occupants, hold 30, pay 4000 card_approved/deposit, assign restored suite_0. Keep existing busy hold.'
    elif family in {'calendar_reopening','calendar_capacity'}:
        ids=['p0_standard_2026-01-15','p0_standard_2026-01-16']
        for cid in ids:
            if family=='calendar_reopening':b.world.calendar[cid].closed=True;b.add(op('set_calendar_closed',cid,enabled=False))
            else:b.world.calendar[cid].sale_limit=0;b.add(op('change_calendar_limit',cid,sale_limit=1))
        if variant=='arrival':b.arrival_flow()
        else:b.booking_flow()
        message='Restore standard room sales on BOTH Jan 15/16 '+('by opening the dates' if family=='calendar_reopening' else 'to one saleable room per night')+', then hold b0 g0/p0_standard/p0_flex Jan 15–17, two occupants, 30 minutes; pay 2400 card_approved/deposit and confirm.'
        if variant=='arrival':message+=' Clean/assign standard_0, pay9600/card_approved/balance, verify guest and check in.'
    elif family=='room_reassignment':
        b.confirmed();b.prepare(op('assign_room',room_id='p0_standard_1'))
        b.add(op('unassign_room'),op('mark_room_clean','p0_standard_0'),op('assign_room',room_id='p0_standard_0'))
        b.terminal('assignments','assign_b0','room_id','p0_standard_0');b.terminal('assignments','assign_b0','status','active')
        message='Reassign unoccupied confirmed b0 from standard_1 to p0_standard_0 after cleaning it, retaining dates/type/payment.'
    elif family in {'booking_reallocation','hold_reallocation','expired_hold_reallocation'}:
        if variant in {'arrival','refund_arrival'}:
            b.confirmed('busy');b.prepare(op('assign_room','busy',room_id='p0_standard_1'),op('mark_room_clean','p0_standard_0'))
            if family=='booking_reallocation':
                b.confirmed();b.prepare(op('assign_room',room_id='p0_standard_0'))
                if variant=='refund_arrival':b.prepare(b.payment(amount=9600,key='balance'))
                b.add(op('cancel_reservation'));end='canceled'
            else:
                b.prepare(b.hold());b.world.reservations['b0'].created_at-=timedelta(hours=2);b.world.reservations['b0'].hold_expires_at-=timedelta(hours=2)
                b.add(op('expire_hold'));end='expired'
            if variant=='refund_arrival':
                b.add(op('refund_reservation',amount_cents=6000,key='cancelled'));b.terminal('payments','payment_'+digest(['b0','refund','cancelled'])[:16],'outcome','succeeded')
            b.arrival_flow('b1',room='p0_standard_0',clean=False);b.terminal('reservations','b0','status',end)
            return b.freeze('Free b0 standard inventory via '+('cancellation' if end=='canceled' else 'explicit expiration')+', retaining busy and its assigned room. '+('Refund6000 with cancelled after the frozen fee. ' if variant=='refund_arrival' else '')+'Arrange b1/g0/p0_standard/p0_flex/Jan15–17/two occupants/hold30, pay2400 deposit and9600 balance with card_approved/deposit/balance, confirm, assign standard_0, verify guest and check in.')
        b.prepare(b.hold('busy',kind='suite'))
        b.prepare(op('set_room_maintenance','p0_suite_0',enabled=False))
        if family=='booking_reallocation':b.confirmed(kind='suite')
        else:
            b.prepare(b.hold(kind='suite'))
            # Fill both restored suites with holds; cancellation/release must free one.
            if family=='expired_hold_reallocation':
                b.world.reservations['b0'].created_at-=timedelta(hours=2);b.world.reservations['b0'].hold_expires_at-=timedelta(hours=2)
        b.add(op('cancel_reservation') if family=='booking_reallocation' else op('expire_hold') if family=='expired_hold_reallocation' else op('release_hold'))
        b.booking_flow('b1',kind='suite')
        b.terminal('reservations','b0','status','canceled' if family=='booking_reallocation' else 'expired' if family=='expired_hold_reallocation' else 'released')
        message='Free b0 suite inventory via '+('cancellation' if family=='booking_reallocation' else 'explicit expiration' if family=='expired_hold_reallocation' else 'hold release')+', retain busy, then book b1/g0/p0_suite/p0_flex Jan 15–17, two occupants, 30-minute hold, pay 4000 card_approved/deposit and confirm.'
    elif family=='guest_identity':
        b.confirmed();b.add(op('verify_guest'));b.terminal('identity_checks','identity_b0','outcome','verified')
        message='Independently verify the guest of b0 without claiming an outcome or altering booking/payment/rooms.'
    elif family=='room_assignment':
        b.confirmed();b.add(op('assign_room',room_id='p0_standard_1'));b.terminal('assignments','assign_b0','status','active')
        message='Assign clean available p0_standard_1 to confirmed b0, preserving other state.'
    elif family=='room_unassignment':
        b.confirmed();b.prepare(op('assign_room',room_id='p0_standard_1'));b.add(op('unassign_room'));b.terminal('assignments','assign_b0','status','released')
        message='Release b0 room assignment before arrival, retaining its confirmed room-type booking and payment.'
    elif family=='room_cleaning':
        b.identity='u2' if variant=='housekeeping' else 'u0';b.add(op('mark_room_clean','p0_standard_0'));b.terminal('rooms','p0_standard_0','housekeeping','clean')
        message='Clean unoccupied p0_standard_0 for assignment, retaining its maintenance and occupancy state.'
    elif family=='maintenance_scheduling':
        b.add(op('set_room_maintenance','p0_standard_0',enabled=True));b.terminal('rooms','p0_standard_0','maintenance',True)
        message='Take unassigned p0_standard_0 out of service for maintenance without overbooking any night.'
    elif family=='sales_restriction':
        b.add(op('set_calendar_closed','p0_standard_2026-01-20',enabled=True));b.terminal('calendar','p0_standard_2026-01-20','closed',True)
        message='Close Jan 20 standard sales at p0 while preserving adjacent dates and all physical rooms.'
    elif family=='checkout':
        b.arrived();b.add(op('check_out'));b.terminal('stays','stay_b0','status','checked_out')
        b.terminal('reservations','b0','status','checked_out')
        message='Check b0 out, release its date/room links, and leave its room dirty for housekeeping.'
    elif family=='occupied_room_link':
        b.confirmed();b.prepare(op('assign_room',room_id='p0_standard_1'),b.payment(amount=9600,key='balance'),op('verify_guest'))
        b.add(op('check_in'));b.terminal('rooms','p0_standard_1','current_stay_id','stay_b0')
        message='Check in prepared b0 and establish the exact reciprocal room/stay occupancy link, retaining all payment and identity evidence.'
    elif family=='balance_payment':
        b.confirmed();b.add(b.payment(amount=9600,key='balance'));b.terminal('payments','payment_'+digest(['b0','charge','balance'])[:16],'outcome','succeeded')
        message='Collect the actual remaining 9600 of b0 using card_approved/balance while retaining its confirmed booking.'
    elif family=='complete_stay':
        b.arrival_flow(alternatives=variant=='choice',checkout=True)
        message='Arrange full b0 stay Jan 15–17 for g0/p0_standard/p0_flex/two guests/hold30, pay deposit 2400 and balance9600 with card_approved/deposit/balance, verify identity, assign clean standard_0, check in then check out.'+(' Either standard room may be used.' if variant=='choice' else '')
    elif family=='rate_date_requote':
        b.confirmed();b.prepare(op('assign_room',room_id='p0_standard_1'))
        if variant=='capacity':
            b.prepare(b.hold('busy',kind='suite',arrival='2026-01-17',departure='2026-01-19',guest='g1'))
            b.add(op('release_hold','busy'),op('set_room_maintenance','p0_suite_0',enabled=False))
        b.add(op('change_dates',arrival='2026-01-16',departure='2026-01-18'),
              op('change_room_type',room_type_id='p0_suite',rate_plan_id='p0_flex'),b.payment(amount=1800,key='difference'),
              op('confirm_reservation'),op('assign_room',room_id='p0_suite_1'))
        if variant=='capacity':b.add(op('assign_room',room_id='p0_suite_0'))
        b.terminal('reservations','b0','status','confirmed');b.terminal('reservations','b0','arrival','2026-01-16')
        b.terminal('reservations','b0','room_type_id','p0_suite');b.terminal('assignments','assign_b0','status','active')
        message='Requote b0 to Jan 16–18 and p0_suite/p0_flex, pay 1800 extra with card_approved/difference, reconfirm and assign a clean suite. '+('Restore needed capacity either by releasing busy or unlocking suite_0; either suite assignment is acceptable.' if variant=='capacity' else 'Use p0_suite_1.')
    elif family=='arrival_turnover_rebooking':
        b.confirmed('busy');b.prepare(op('mark_room_clean','p0_standard_0'),op('assign_room','busy',room_id='p0_standard_0'))
        b.add(b.hold(),b.payment(),op('confirm_reservation'),
              op('assign_room',room_id='p0_standard_1'),b.payment(amount=9600,key='balance'),op('verify_guest'),op('check_in'),op('check_out'),
              b.hold('b1'),b.payment('b1'),op('confirm_reservation','b1'),op('mark_room_clean','p0_standard_1'),
              op('assign_room','b1',room_id='p0_standard_1'),b.payment('b1',9600,'balance'),op('verify_guest','b1'),op('check_in','b1'))
        b.terminal('reservations','b0','status','checked_out');b.terminal('reservations','b1','status','checked_in')
        b.terminal('rooms','p0_standard_1','current_stay_id','stay_b1')
        message='Keep busy reservation/room unchanged. Arrange and check in b0 in standard_1, then check it out and clean/turn over the same room for b1. Both g0 bookings: Jan15–17, p0_standard/p0_flex/two occupants/hold30, payments 2400 deposit and9600 balance with card_approved/deposit/balance; independently verify each guest.'
    elif family.endswith('_refusal'):
        message,code=refusal_scenario(b,scenario);return b.freeze(message+' Refuse with the policy code and preserve all state.',code)
    else:raise ValueError('unknown StayOps scenario '+scenario)
    return b.freeze(message+' Preserve unrelated entities and immutable guest/property bindings.')


def progress_task(world,task,route,phase):
    task=task.model_copy(deep=True);witness=task.witnesses[route]
    if phase<1 or phase>=len(witness.operations):raise ValueError('require proper real workflow progress')
    from . import adapter
    world=adapter.World.model_validate(execute(adapter,world,task.identity,witness.operations[:phase]).final_state)
    frame={}
    for key,fields in task.goal.allowed_changes.items():
        collection,entity=key.split('/')
        if fields==['*'] and entity in getattr(world,collection):fields=FRAME_FIELDS[collection]
        if fields:frame[key]=fields
    task.goal.allowed_changes=frame;task.witnesses=[];task.initial_state_hash=world.state_hash;task.id='pending'
    task.search_nodes=0;task.search_exhausted=False
    task.message='Continue this in-progress request from actual observed hotel state; choose only remaining work. '+task.message
    return world,freeze_task(adapter,world,task)


def certificate_loader():
    bases={}
    def reconstruct(seed,recipe):
        base=recipe.rsplit('@',2)[0] if '@' in recipe else recipe;key=(seed,base)
        if key not in bases:bases[key]=make_task(seed,base)
        world,task=bases[key]
        if '@' in recipe:
            _,route,phase=recipe.rsplit('@',2);return progress_task(world,task,int(route),int(phase))
        return world.clone(),task.model_copy(deep=True)
    return reconstruct


def refusal_scenario(b,scenario):
    name,_,variant=scenario.partition(':');op=b.op
    if name=='deposit_refusal':
        b.prepare(b.hold(plan='prepaid' if variant=='prepaid' else 'flex'));b.add(op('confirm_reservation'));return 'Confirm b0 without its required successful deposit.','DEPOSIT_REQUIRED'
    if name=='refund_refusal':
        b.confirmed();b.add(op('refund_reservation',amount_cents=1,key='unearned'));return 'Refund one cent from active unchanged b0 with unearned.','REFUND_EXCEEDS_ELIGIBLE'
    if name=='overpay_refusal':
        b.prepare(b.hold());b.add(b.payment(amount=12001));return 'Pay b0 more than its actual price, 12001/card_approved/deposit.','PAYMENT_EXCEEDS_DUE'
    if name=='payment_key_refusal':
        b.prepare(b.hold(),b.payment(amount=1000,key='original'));b.add(b.payment(amount=1001,key='original'));return 'Reuse original payment key with a changed amount of1001/card_approved on b0.','IDEMPOTENCY_CONFLICT'
    if name in {'expired_confirmation_refusal','early_expiration_refusal'}:
        b.prepare(b.hold())
        if name=='expired_confirmation_refusal':
            b.world.reservations['b0'].created_at-=timedelta(hours=2);b.world.reservations['b0'].hold_expires_at-=timedelta(hours=2);b.add(op('confirm_reservation'))
            return 'Confirm elapsed hold b0 without recreating a valid hold.','HOLD_EXPIRED'
        b.add(op('expire_hold'));return 'Expire live b0 before its frozen expiry time.','HOLD_NOT_EXPIRED'
    if name=='occupied_clean_refusal':
        b.arrived();b.add(op('mark_room_clean','p0_standard_1'));return 'Mark occupied p0_standard_1 clean.','ROOM_OCCUPIED'
    if name in {'dirty_assignment_refusal','maintenance_assignment_refusal'}:
        if name=='dirty_assignment_refusal':
            b.confirmed();b.add(op('assign_room',room_id='p0_standard_0'));return 'Assign dirty p0_standard_0 to b0 without cleaning.','ROOM_NOT_CLEAN'
        b.confirmed(kind='suite');b.add(op('assign_room',room_id='p0_suite_0'));return 'Assign maintenance-locked p0_suite_0 to b0.','ROOM_IN_MAINTENANCE'
    if name=='overbooking_refusal':
        b.prepare(b.hold('busy',kind='suite'));b.add(b.hold(kind='suite'));return 'Add b0 suite booking Jan15–17 despite busy using the sole saleable suite.','ROOM_TYPE_UNAVAILABLE'
    if name=='overlap_refusal':
        b.confirmed();b.confirmed('b1');b.prepare(op('assign_room','b1',room_id='p0_standard_1'))
        b.add(op('assign_room',room_id='p0_standard_1'));return 'Assign b0 to p0_standard_1 overlapping the active b1 assignment.','ROOM_INTERVAL_CONFLICT'
    if name=='scope_refusal':
        b.add(op('set_room_maintenance','p1_standard_0','p0',enabled=False));return 'Unlock a partner p1 room using my p0 property scope.','PROPERTY_SCOPE_MISMATCH'
    if name=='staff_role_refusal':
        b.identity='u1' if variant!='housekeeping' else 'u2'
        b.add(op('set_room_maintenance','p2_standard_0','p2',enabled=False));return 'Change p2 room maintenance using my staff role from p0.','PROPERTY_PERMISSION_DENIED'
    if name=='guest_staff_refusal':
        b.confirmed();b.identity='u3';b.add(op('assign_room',room_id='p0_standard_1'));return 'As the owning guest, assign b0 to room standard_1 myself.','PROPERTY_PERMISSION_DENIED'
    if name=='occupancy_refusal':
        hold=b.hold();hold.params['occupants']=3;b.add(hold);return 'Book a two-person standard room for three occupants as b0.','OCCUPANCY_LIMIT_EXCEEDED'
    if name=='early_checkin_refusal':
        b.confirmed(arrival='2026-01-20',departure='2026-01-22');b.add(op('check_in'));return 'Check in b0 on January15 before its January20 arrival.','OUTSIDE_CHECK_IN_WINDOW'
    if name in {'identity_refusal','balance_refusal'}:
        if variant=='declined':b.world.guests['g0'].simulated_identity_result='declined'
        b.confirmed();b.prepare(op('assign_room',room_id='p0_standard_1'))
        if name=='identity_refusal':
            b.prepare(b.payment(amount=9600,key='balance'))
            if variant=='declined':b.prepare(op('verify_guest'))
            b.add(op('check_in'));return 'Check in b0 without an approved independent identity check.','GUEST_IDENTITY_REQUIRED'
        b.prepare(op('verify_guest'));b.add(op('check_in'));return 'Check in b0 with only its deposit and no balance payment.','PAYMENT_REQUIRED'
    if name=='date_conflict_refusal':
        b.confirmed();b.prepare(b.hold('busy1',arrival='2026-01-18',departure='2026-01-20'),b.hold('busy2',arrival='2026-01-19',departure='2026-01-21'))
        b.add(op('change_dates',arrival='2026-01-17',departure='2026-01-21'));return 'Extend b0 over January19 despite all standard inventory being held that night.','ROOM_TYPE_UNAVAILABLE'
    if name=='calendar_reduction_refusal':
        b.prepare(b.hold());b.add(op('change_calendar_limit','p0_standard_2026-01-15',sale_limit=0));return 'Reduce the Jan15 standard sale limit to zero while retaining b0.','CALENDAR_LIMIT_BELOW_BOOKINGS'
    if name=='calendar_closure_refusal':
        b.prepare(b.hold());b.add(op('set_calendar_closed','p0_standard_2026-01-15',enabled=True));return 'Close Jan15 standard sales while b0 still holds that night.','CALENDAR_HAS_BOOKINGS'
    if name=='maintenance_lock_refusal':
        b.confirmed();b.prepare(op('assign_room',room_id='p0_standard_1'));b.add(op('set_room_maintenance','p0_standard_1',enabled=True));return 'Lock b0 assigned room p0_standard_1 for maintenance.','ROOM_ASSIGNED'
    if name=='cancel_stay_refusal':
        b.arrived();b.add(op('cancel_reservation'));return 'Ordinarily cancel checked-in b0 instead of checking out.','RESERVATION_NOT_CANCELABLE'
    if name=='swap_conflict_refusal':
        b.arrived();b.prepare(op('mark_room_clean','p0_standard_0'));b.confirmed('b1');b.prepare(op('assign_room','b1',room_id='p0_standard_0'))
        b.add(op('swap_room',room_id='p0_standard_0'));return 'Move occupied b0 into standard_0 despite overlapping b1 assignment.','ROOM_INTERVAL_CONFLICT'
    if name=='cross_type_refusal':
        b.confirmed();b.add(op('change_room_type',room_type_id='p1_suite',rate_plan_id='p1_flex'));return 'Upgrade p0 booking b0 to another property p1 room type/plan.','RATE_ROOM_TYPE_SCOPE_MISMATCH'
    if name=='unassign_stay_refusal':
        b.arrived();b.add(op('unassign_room'));return 'Release checked-in b0 room assignment without checkout or swap.','ASSIGNMENT_NOT_RELEASABLE'
    raise ValueError('unknown StayOps refusal scenario '+scenario)
