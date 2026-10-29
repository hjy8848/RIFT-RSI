"""Native independent expected changes; actual mutations do not import this file."""
from datetime import datetime,timedelta
from usami_harness.models import Decision,digest,signed_notice,notice_valid
from usami_dialogue.process import consent_code
from .models import Quote,LedgerEntry,Verification

POLICY={
 'BOOKING_ACCESS_DENIED':'Only the authenticated passenger can change their own carrier-scoped booking.',
 'PASSENGER_VERIFICATION_REQUIRED':'Verify passenger identity with their actual document token before financial changes.',
 'REQUIRED_INFORMATION_MISSING':'Clarify booking, intended destination, travel times and fee limit before proposing a change.',
 'ITINERARY_UNAVAILABLE':'Every replacement flight must operate, have capacity and belong to this carrier.',
 'CONNECTION_INFEASIBLE':'A continuous itinerary must meet each airport minimum connection time.',
 'FARE_CHANGE_FORBIDDEN':'Voluntary changes must follow the ticket fare contract; airline cancellation waives the restriction.',
 'FARE_REFUND_FORBIDDEN':'Voluntary refund eligibility follows the fare contract, distinct from airline cancellation.',
 'CUSTOMER_CONSTRAINT_UNMET':'Do not exceed the disclosed fee limit or violate the selected destination/time/baggage needs.',
 'EXPLICIT_CONSENT_REQUIRED':'Commit only a real current quote with exact consent; revised preferences invalidate old terms.',
 'PAYMENT_LIMIT_EXCEEDED':'Additional charges require remaining simulated payment capacity.'}


def occupancy(world,flight,cabin,except_booking=None):return sum(b.status=='active' and b.cabin==cabin and flight in b.flights and b.id!=except_booking for b in world.bookings.values())
def quote_digest(q):return digest(q.model_dump(mode='json',exclude={'quote_hash','status'}))
def quote_key(booking,action,flights,cabin,baggage,revision,clock):return 'quote_'+digest([booking.id,booking.revision,action,flights,cabin,baggage,revision,clock.isoformat()])[:20]
def disrupted(world,booking):return any(world.flights[f].status=='cancelled' for f in booking.flights)
def net_paid(world,b):return b.initial_paid_cents+sum((1 if e.kind=='charge' else -1)*e.amount_cents for e in world.ledger.values() if e.booking_id==b.id)


def itinerary_code(world,booking,flights,cabin,check_capacity=True):
    if len(set(flights))!=len(flights):return 'CONNECTION_INFEASIBLE'
    rows=[world.flights.get(fid) for fid in flights]
    if not rows or any(f is None or f.carrier_id!=booking.carrier_id or cabin not in f.seats or cabin not in f.fares for f in rows):return 'ITINERARY_UNAVAILABLE'
    if any(f.status=='cancelled' or f.departure<=world.clock for f in rows):return 'ITINERARY_UNAVAILABLE'
    if check_capacity and any(occupancy(world,f.id,cabin,booking.id)>=f.seats[cabin] for f in rows):return 'ITINERARY_UNAVAILABLE'
    for first,second in zip(rows,rows[1:]):
        if first.destination!=second.origin or second.departure<first.arrival+timedelta(minutes=world.airports[first.destination].min_connection_minutes):return 'CONNECTION_INFEASIBLE'
    return None


def invariants(world):
    errors=[]
    def add(code,path):errors.append({'code':code,'path':path})
    for name in ('users','carriers','airports','flights','fare_rules','bookings','quotes','ledger','verifications','external_events'):
        for key,row in getattr(world,name).items():
            if key!=row.id:add('ENTITY_KEY_MISMATCH',name+'/'+key)
    for f in world.flights.values():
        if f.carrier_id not in world.carriers or f.origin not in world.airports or f.destination not in world.airports or f.origin==f.destination or f.departure.tzinfo is None or f.arrival.tzinfo is None or f.departure>=f.arrival:add('FLIGHT_REFERENCE_INVALID','flights/'+f.id)
        if set(f.seats)!={'economy','business'} or set(f.fares)!=set(f.seats) or any(isinstance(n,bool) or not isinstance(n,int) or n<0 for n in list(f.seats.values())+list(f.fares.values())):add('CABIN_INVENTORY_INVALID','flights/'+f.id)
        for cabin,count in f.seats.items():
            if occupancy(world,f.id,cabin)>count:add('SEAT_OVERBOOKED','flights/'+f.id)
    for r in world.fare_rules.values():
        if r.carrier_id not in world.carriers:add('FARE_CARRIER_INVALID','fare_rules/'+r.id)
    for b in world.bookings.values():
        rule=world.fare_rules.get(b.fare_rule_id)
        if b.passenger_id not in world.users or not rule or b.carrier_id!=rule.carrier_id or any(fid not in world.flights or world.flights[fid].carrier_id!=b.carrier_id for fid in b.flights):add('BOOKING_REFERENCE_INVALID','bookings/'+b.id);continue
        if len(set(b.flights))!=len(b.flights) or any(world.flights[a].destination!=world.flights[z].origin for a,z in zip(b.flights,b.flights[1:])):add('BOOKING_ROUTE_INVALID','bookings/'+b.id)
        if net_paid(world,b)!=b.fare_value_cents+b.retained_fees_cents+b.baggage_paid_cents:add('TICKET_ACCOUNTING_INVALID','bookings/'+b.id)
        if b.status=='cancelled' and (b.fare_value_cents or b.extra_baggage_kg or b.baggage_paid_cents):add('CANCELLED_TICKET_VALUE','bookings/'+b.id)
        if b.status=='active' and any(rule.baggage_included_kg+b.extra_baggage_kg>world.flights[f].max_baggage_kg for f in b.flights):add('BAGGAGE_LIMIT_EXCEEDED','bookings/'+b.id)
    for q in world.quotes.values():
        b=world.bookings.get(q.booking_id)
        expected_id=quote_key(b.model_copy(update={'revision':q.booking_revision}),q.action,q.flights,q.cabin,q.baggage_kg,q.revision,q.expires_at-timedelta(minutes=30)) if b else None
        if not b or q.expires_at.tzinfo is None or q.quote_hash!=quote_digest(q) or q.id!=expected_id or any(f not in world.flights for f in q.flights):add('QUOTE_INTEGRITY','quotes/'+q.id)
    for e in world.ledger.values():
        q=world.quotes.get(e.quote_id);b=world.bookings.get(e.booking_id)
        binding={'booking_id':e.booking_id,'quote_id':e.quote_id,'kind':e.kind,'amount_cents':e.amount_cents}
        expected=q.charge_cents if q and e.kind=='charge' else q.refund_cents if q else -1
        if not b or not q or q.status!='committed' or e.id!=e.kind+'_'+q.id or q.booking_id!=b.id or e.amount_cents!=expected or not notice_valid(e.notice) or e.notice.get('domain')!=world.domain or e.notice.get('source')!='airline-payment-simulator-v0.1' or e.notice.get('binding')!=binding:add('PAYMENT_ATTESTATION_INVALID','ledger/'+e.id)
    for v in world.verifications.values():
        b=world.bookings.get(v.booking_id);binding={'booking_id':v.booking_id,'passenger_id':b.passenger_id if b else None,'verified':v.verified}
        if not b or v.id!='verification_'+b.id or v.verified!=world.users[b.passenger_id].simulated_identity_valid or not notice_valid(v.notice) or v.notice.get('domain')!=world.domain or v.notice.get('source')!='passenger-identity-simulator-v0.1' or v.notice.get('binding')!=binding:add('IDENTITY_ATTESTATION_INVALID','verifications/'+v.id)
    for event in world.external_events.values():
        if event.flight_id not in world.flights or event.at.tzinfo is None:add('EXTERNAL_EVENT_INVALID','external_events/'+event.id)
    if len(set(world.applied_events))!=len(world.applied_events) or set(world.applied_events)-set(world.external_events):add('EXTERNAL_EVENT_INVALID','applied_events')
    for p in world.profiles.values():
        try:p.checked()
        except ValueError:add('PRIVATE_PROFILE_INTEGRITY','profiles')
        b=world.bookings.get(p.facts.get('booking_id'));user=world.users.get(p.identity)
        if not user or not b or b.passenger_id!=p.identity or p.protected_objects!=(b.id,) or p.facts.get('document_token')!=user.document_token or p.facts.get('destination') not in world.airports:add('PRIVATE_PROFILE_REFERENCE','profiles')
    return errors


def _quote(world,b,action,flights,cabin,charge,refund,fee,fare,baggage):
    q=Quote(id=quote_key(b,action,flights,cabin,baggage,world.interaction.revision,world.clock),booking_id=b.id,action=action,flights=flights,cabin=cabin,
        charge_cents=charge,refund_cents=refund,fee_cents=fee,fare_value_cents=fare,baggage_kg=baggage,revision=world.interaction.revision,booking_revision=b.revision,expires_at=world.clock+timedelta(minutes=30),quote_hash='')
    q.quote_hash=quote_digest(q);return q


def proposal(world,b,action,params):
    """Recompute terms both when quoting and committing: no stale inventory or fees."""
    ctx=world.interaction;rule=world.fare_rules[b.fare_rule_id]
    if b.status!='active':return None,'BOOKING_CANCELLED'
    if ctx.preferences.get('service')!=action:return None,'PREFERENCE_RESOLUTION_REQUIRED'
    if action=='rebook':
        if any(k not in ctx.known_facts for k in ('destination','earliest_departure','latest_arrival','max_fee_cents')):return None,'REQUIRED_INFORMATION_MISSING'
        flights=params['flights'];cabin=params['cabin'];code=itinerary_code(world,b,flights,cabin)
        if code:return None,code
        if flights==b.flights and cabin==b.cabin:return None,'NO_ITINERARY_CHANGE'
        if not disrupted(world,b) and not rule.changeable:return None,'FARE_CHANGE_FORBIDDEN'
        if not disrupted(world,b) and world.clock+timedelta(hours=rule.notice_hours)>min(world.flights[f].departure for f in b.flights):return None,'CHANGE_WINDOW_CLOSED'
        first=world.flights[flights[0]];last=world.flights[flights[-1]]
        if first.origin!=world.flights[b.flights[0]].origin or last.destination!=ctx.known_facts['destination'] or first.departure<datetime.fromisoformat(ctx.known_facts['earliest_departure']) or last.arrival>datetime.fromisoformat(ctx.known_facts['latest_arrival']):return None,'CUSTOMER_CONSTRAINT_UNMET'
        if ctx.preferences.get('itinerary')!=flights:return None,'PREFERENCE_RESOLUTION_REQUIRED'
        if any(rule.baggage_included_kg+b.extra_baggage_kg>world.flights[f].max_baggage_kg for f in flights):return None,'BAGGAGE_LIMIT_EXCEEDED'
        fare=sum(world.flights[f].fares[cabin] for f in flights);fee=0 if disrupted(world,b) else rule.change_fee_cents
        charge=max(0,fare-b.fare_value_cents)+fee;refund=max(0,b.fare_value_cents-fare)
        if charge>ctx.known_facts['max_fee_cents']:return None,'CUSTOMER_CONSTRAINT_UNMET'
        q=_quote(world,b,action,flights,cabin,charge,refund,fee,fare,b.extra_baggage_kg)
    elif action=='refund':
        if not disrupted(world,b) and not rule.refundable:return None,'FARE_REFUND_FORBIDDEN'
        if not disrupted(world,b) and world.clock>=min(world.flights[f].departure for f in b.flights):return None,'REFUND_WINDOW_CLOSED'
        fee=0 if disrupted(world,b) else min(rule.refund_fee_cents,b.fare_value_cents)
        q=_quote(world,b,action,list(b.flights),b.cabin,0,b.fare_value_cents-fee,fee,0,0)
    else:
        if ctx.known_facts.get('baggage_kg')!=params['extra_kg'] or 'max_fee_cents' not in ctx.known_facts:return None,'REQUIRED_INFORMATION_MISSING'
        kg=params['extra_kg']
        if any(rule.baggage_included_kg+kg>world.flights[f].max_baggage_kg or world.flights[f].status=='cancelled' or world.flights[f].departure<=world.clock for f in b.flights):return None,'BAGGAGE_LIMIT_EXCEEDED'
        if kg<=b.extra_baggage_kg:return None,'BAGGAGE_ALREADY_PURCHASED'
        charge=(kg-b.extra_baggage_kg)*rule.baggage_price_per_kg_cents
        if charge>ctx.known_facts['max_fee_cents']:return None,'CUSTOMER_CONSTRAINT_UNMET'
        q=_quote(world,b,action,list(b.flights),b.cabin,charge,0,0,b.fare_value_cents,kg)
    charged=sum(e.amount_cents for e in world.ledger.values() if e.kind=='charge' and world.bookings[e.booking_id].passenger_id==b.passenger_id)
    if charged+q.charge_cents>world.users[b.passenger_id].payment_limit_cents:return None,'PAYMENT_LIMIT_EXCEEDED'
    return q,None


def decide(world,actor,op):
    deny=lambda code:Decision(allowed=False,error_code=code)
    if invariants(world):return deny('INVALID_WORLD')
    if actor not in world.users:return deny('UNKNOWN_ACTOR')
    if op.scope_id not in world.carriers:return deny('CARRIER_NOT_FOUND')
    if op.name=='get_bookings':
        if op.resource_id!=op.scope_id:return deny('CARRIER_SCOPE_MISMATCH')
        from .observations import query_data
        return Decision(allowed=True,expected_state=world.clone(),data=query_data(world,actor,op))
    b=world.bookings.get(op.resource_id)
    if not b:return deny('BOOKING_NOT_FOUND')
    if b.carrier_id!=op.scope_id:return deny('CARRIER_SCOPE_MISMATCH')
    if b.passenger_id!=actor:return deny('BOOKING_ACCESS_DENIED')
    if op.is_query:
        if op.name=='get_quote' and (op.params['quote_id'] not in world.quotes or world.quotes[op.params['quote_id']].booking_id!=b.id):return deny('QUOTE_SCOPE_MISMATCH')
        from .observations import query_data
        return Decision(allowed=True,expected_state=world.clone(),data=query_data(world,actor,op))
    ctx=world.interaction
    if ctx.identity!=actor:return deny('CUSTOMER_CONTEXT_REQUIRED')
    if ctx.known_facts.get('booking_id')!=b.id:return deny('REQUIRED_INFORMATION_MISSING')
    new=world.clone();p=op.params
    if op.name=='verify_passenger':
        if ctx.known_facts.get('document_token')!=p['document_token']:return deny('REQUIRED_INFORMATION_MISSING')
        user=world.users[actor]
        if p['document_token']!=user.document_token or not user.simulated_identity_valid:return deny('PASSENGER_IDENTITY_INVALID')
        vid='verification_'+b.id;binding={'booking_id':b.id,'passenger_id':actor,'verified':True}
        new.verifications[vid]=Verification(id=vid,booking_id=b.id,verified=True,notice=signed_notice(world.domain,'passenger-identity',binding,world.clock))
        return Decision(allowed=True,expected_state=new,data={'booking_id':b.id,'verified':True})
    if not world.verifications.get('verification_'+b.id) or not world.verifications['verification_'+b.id].verified:return deny('PASSENGER_VERIFICATION_REQUIRED')
    if op.name=='commit_quote':
        q=world.quotes.get(p['quote_id'])
        if not q or q.booking_id!=b.id:return deny('QUOTE_SCOPE_MISMATCH')
        if q.status!='offered' or q.booking_revision!=b.revision:return deny('STALE_BOOKING_QUOTE')
        if world.clock>q.expires_at:return deny('QUOTE_EXPIRED')
        if q.revision!=ctx.revision:return deny('STALE_CUSTOMER_CHOICE')
        expected,code=proposal(world,b,q.action,{'flights':q.flights,'cabin':q.cabin,'extra_kg':q.baggage_kg})
        if code:return deny(code)
        if any(getattr(expected,k)!=getattr(q,k) for k in ('action','flights','cabin','charge_cents','refund_cents','fee_cents','fare_value_cents','baggage_kg')):return deny('QUOTE_TERMS_CHANGED')
        code=consent_code(ctx,op,q.quote_hash)
        if code:return deny(code)
        updated=new.bookings[b.id];updated.revision+=1;new.quotes[q.id].status='committed'
        if q.action=='rebook':updated.flights=list(q.flights);updated.cabin=q.cabin;updated.fare_value_cents=q.fare_value_cents;updated.retained_fees_cents+=q.fee_cents
        elif q.action=='refund':updated.status='cancelled';updated.fare_value_cents=0;updated.retained_fees_cents+=q.fee_cents+b.baggage_paid_cents;updated.baggage_paid_cents=0;updated.extra_baggage_kg=0
        else:updated.extra_baggage_kg=q.baggage_kg;updated.baggage_paid_cents+=q.charge_cents
        for kind,amount in [('charge',q.charge_cents),('refund',q.refund_cents)]:
            if amount:
                eid=kind+'_'+q.id;binding={'booking_id':b.id,'quote_id':q.id,'kind':kind,'amount_cents':amount}
                new.ledger[eid]=LedgerEntry(id=eid,booking_id=b.id,kind=kind,amount_cents=amount,quote_id=q.id,notice=signed_notice(world.domain,'airline-payment',binding,world.clock))
        return Decision(allowed=True,expected_state=new,data={'booking_id':b.id,'status':updated.status,'flights':updated.flights,'charge_cents':q.charge_cents,'refund_cents':q.refund_cents})
    action={'quote_change':'rebook','quote_refund':'refund','quote_baggage':'baggage'}[op.name]
    q,code=proposal(world,b,action,p)
    if code:return deny(code)
    if q.id in world.quotes:return Decision(allowed=True,expected_state=new,data={'quote':world.quotes[q.id].model_dump(mode='json')})
    new.quotes[q.id]=q
    return Decision(allowed=True,expected_state=new,data={'quote':q.model_dump(mode='json')})
