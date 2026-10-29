"""Independent timetable, passenger contracts and fixed disruption distribution."""
from datetime import timedelta
from usami.randomness import StableRandom
from usami_harness.models import digest
from usami_benchmark.contracts import GeneratorConfig
from usami_dialogue.models import CustomerPrivateState
from .models import World,User,Carrier,Airport,Flight,FareRule,Booking,FlightEvent

GENERATOR_VERSION='airline-generator-v1'
RULES_VERSION='airline-0.1.0'


def routes(world,carrier,origin,destination):
    result=[]
    for first in world.flights.values():
        if first.carrier_id!=carrier or first.origin!=origin:continue
        if first.destination==destination:result.append([first.id])
        else:
            for last in world.flights.values():
                if last.carrier_id==carrier and last.origin==first.destination and last.destination==destination:result.append([first.id,last.id])
    return result


def generate_world(seed=42,config=None,*,diagnostics=None):
    cfg=GeneratorConfig.model_validate(config or {});rng=StableRandom(digest({'version':GENERATOR_VERSION,'seed':seed,'config':cfg.model_dump(mode='json')}))
    users={f'u{i}':User(id=f'u{i}',name=f'Passenger {i}',document_token=f'private_document_{i}',simulated_identity_valid=rng.random()<.9,payment_limit_cents=rng.choice([1000,15000,80000])) for i in range(4)}
    carriers={f'carrier{i}':Carrier(id=f'carrier{i}',name=f'Carrier {i}') for i in range(rng.randint(cfg.min_scopes,cfg.max_scopes))}
    airports={name:Airport(id=name,min_connection_minutes=rng.choice([30,60,90,120])) for name in ('A','B','C','D','E')}
    world=World(seed=seed,users=users,carriers=carriers,airports=airports,flights={},fare_rules={},bookings={},profiles={})
    for cid in carriers:
        flight_index=0
        def add(origin,destination,departure,duration):
            nonlocal flight_index
            key=cid+'_flight'+str(flight_index);flight_index+=1
            world.flights[key]=Flight(id=key,carrier_id=cid,origin=origin,destination=destination,departure=departure,arrival=departure+timedelta(minutes=duration),
                status=rng.choice(['scheduled']*8+['delayed','cancelled']),seats={'economy':0,'business':0},fares={'economy':rng.choice([5000,10000,18000]),'business':rng.choice([22000,35000,60000])},max_baggage_kg=rng.choice([23,32]))
        for day in range(1,rng.randint(3,cfg.max_resources+1)):
            add('A','B',world.clock+timedelta(days=day,hours=rng.choice([6,9,15])),rng.choice([90,120,180]))
        for hub in rng.sample(['C','D','E'],rng.randint(1,3)):
            for day in rng.sample([1,2,3,4],rng.randint(1,2)):
                departure=world.clock+timedelta(days=day,hours=6);duration=rng.choice([60,90,120]);gap=rng.choice([15,60,180])
                add('A',hub,departure,duration);add(hub,'B',departure+timedelta(minutes=duration+gap),rng.choice([60,90,150]))
        for index in range(rng.randint(2,4)):
            key=cid+'_fare'+str(index)
            world.fare_rules[key]=FareRule(id=key,carrier_id=cid,changeable=rng.random()<.7,refundable=rng.random()<.5,
                change_fee_cents=rng.choice([0,1500,5000]),refund_fee_cents=rng.choice([0,1000,3000]),notice_hours=rng.choice([0,6,24]),
                baggage_included_kg=rng.choice([0,15,23]),baggage_price_per_kg_cents=rng.choice([200,500,1000]))
    for uid in users:
        for index in range(rng.randint(2,4)):
            cid=rng.choice(tuple(carriers));options=routes(world,cid,'A','B')
            consistent=[r for r in options if all(world.flights[a].arrival+timedelta(minutes=airports[world.flights[a].destination].min_connection_minutes)<=world.flights[b].departure for a,b in zip(r,r[1:]))]
            chosen=rng.choice(consistent);cabin=rng.choice(['economy','economy','business']);rid=rng.choice([r.id for r in world.fare_rules.values() if r.carrier_id==cid]);bid=f'{uid}_booking{index}'
            amount=sum(world.flights[f].fares[cabin] for f in chosen)
            booking=Booking(id=bid,carrier_id=cid,passenger_id=uid,flights=chosen,cabin=cabin,fare_rule_id=rid,initial_paid_cents=amount,fare_value_cents=amount)
            world.bookings[bid]=booking
            facts={'booking_id':bid,'document_token':users[uid].document_token,'destination':'B',
                'earliest_departure':(world.clock+timedelta(hours=rng.choice([12,24,48]))).isoformat(),
                'latest_arrival':(world.clock+timedelta(days=rng.choice([2,3,5]))).isoformat(),
                'max_fee_cents':rng.choice([2000,10000,60000]),'baggage_kg':rng.choice([3,5,10])}
            world.profiles['profile_'+bid]=CustomerPrivateState(identity=uid,opening_request='I need help with my travel booking.',facts=facts,
                preference_space={'service':['rebook','refund','baggage'],'itinerary':options},initial_preferences={'service':rng.choice(['rebook','refund','baggage']),'itinerary':rng.choice(options)},
                confirmable_actions=('verify_passenger','quote_change','quote_refund','quote_baggage','commit_quote'),protected_objects=(bid,)).checked()
    for f in world.flights.values():
        for cabin in f.seats:
            used=sum(b.status=='active' and b.cabin==cabin and f.id in b.flights for b in world.bookings.values())
            f.seats[cabin]=used+rng.choice([0,1,2,5,10])
        if rng.random()<.35:
            eid='update_'+f.id
            world.external_events[eid]=FlightEvent(id=eid,flight_id=f.id,at=world.clock+timedelta(hours=1),status=rng.choice(['cancelled','delayed']),delay_minutes=rng.choice([30,120,300]))
    from .rules import invariants
    errors=invariants(world)
    if errors:raise ValueError(str(errors))
    return world
