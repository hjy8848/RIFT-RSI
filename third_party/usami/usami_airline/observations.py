"""Scoped passenger observations; simulator identity outcomes and profiles stay private."""
from usami_dialogue.process import public_context


def observation(world,actor):
    return {'domain':world.domain,'identity':{'id':actor,'name':world.users[actor].name},'clock':world.clock.isoformat(),
        'bookings':[b.model_dump(mode='json') for b in sorted(world.bookings.values(),key=lambda row:row.id) if b.passenger_id==actor],
        'customer_provided':public_context(world.interaction)}


def visible_ids(view):return {r['id'] for rows in view.values() if isinstance(rows,list) for r in rows if isinstance(r,dict) and 'id' in r}


def _flight(world,f):
    raw=f.model_dump(mode='json')
    raw['available_seats']={c:n-sum(b.status=='active' and b.cabin==c and f.id in b.flights for b in sorted(world.bookings.values(),key=lambda row:row.id)) for c,n in f.seats.items()}
    return raw


def query_data(world,actor,op):
    if op.name=='get_bookings':return {'bookings':[b.model_dump(mode='json') for b in sorted(world.bookings.values(),key=lambda row:row.id) if b.passenger_id==actor and b.carrier_id==op.scope_id]}
    b=world.bookings[op.resource_id]
    if op.name=='get_booking':return {**b.model_dump(mode='json'),'flights':[_flight(world,world.flights[f]) for f in b.flights],
        'fare_rule':world.fare_rules[b.fare_rule_id].model_dump(mode='json'),
        'remaining_payment_capacity_cents':world.users[actor].payment_limit_cents-sum(e.amount_cents for e in sorted(world.ledger.values(),key=lambda row:row.id) if e.kind=='charge' and world.bookings[e.booking_id].passenger_id==actor),
        'verified':bool(world.verifications.get('verification_'+b.id) and world.verifications['verification_'+b.id].verified)}
    if op.name=='get_quote':return {'quote':world.quotes[op.params['quote_id']].model_dump(mode='json')}
    # Include connecting legs as well as direct flights, without disclosing a
    # Reference itinerary or its eligibility. The Service chooses the path.
    carrier=[f for f in sorted(world.flights.values(),key=lambda row:row.id) if f.carrier_id==b.carrier_id]
    origins={op.params['origin']}|{f.destination for f in carrier if f.origin==op.params['origin']}
    return {'flights':[_flight(world,f) for f in carrier if f.origin in origins],
        'airports':[a.model_dump(mode='json') for a in sorted(world.airports.values(),key=lambda row:row.id)]}
