"""Independent property/calendar distribution with real attested booking histories."""
from datetime import timedelta
from usami.randomness import StableRandom
from usami_benchmark.contracts import GeneratorConfig
from usami_benchmark.generation import NativeHistory
from usami_harness.models import digest
from usami_stayops.models import World,User,Property,Access,Guest,RoomType,Room,CalendarDay,RatePlan
from usami_stayops.rules import invariants

GENERATOR_VERSION='stayops-structural-v1'
RULES_VERSION='stayops-0.1.0'


def generate_world(seed=42,config=None,*,diagnostics=None):
    return _generate(seed,config,diagnostics=diagnostics)


def _generate(seed,config=None,*,diagnostics=None,world_type=World,native_adapter=None,distribution_version=GENERATOR_VERSION):
    cfg=GeneratorConfig.model_validate(config or {});rng=StableRandom(digest({'version':distribution_version,'seed':seed,'config':cfg.model_dump(mode='json')}))
    users={f'u{i}':User(id=f'u{i}',name=f'User {i}') for i in range(9)}
    props={f'p{i}':Property(id=f'p{i}',name=f'Property {i}',owner_id=f'u{i%3}',max_stay_nights=rng.choice([7,14,21])) for i in range(rng.randint(cfg.min_scopes,cfg.max_scopes))}
    guests={f'g{i}':Guest(id=f'g{i}',user_id=f'u{i+3}',name=f'Guest {i}',simulated_identity_result=rng.choice(['verified','verified','declined'])) for i in range(6)}
    world=world_type(seed=seed,users=users,properties=props,guests=guests,access={},room_types={},rooms={},calendar={},rate_plans={})
    for pid in props:
        for uid in rng.sample(list(users),rng.randint(2,5)):
            aid=pid+'_'+uid;world.access[aid]=Access(id=aid,property_id=pid,user_id=uid,role=rng.choice(['manager','frontdesk','housekeeping','finance','viewer']))
        tids=[]
        for k in range(rng.randint(2,3)):
            tid=f'{pid}_type{k}';tids.append(tid);count=rng.randint(cfg.min_resources,cfg.max_resources)
            world.room_types[tid]=RoomType(id=tid,property_id=pid,name=f'Type {k}',nightly_cents=rng.choice([4000,6000,10000,15000]),max_occupants=rng.choice([1,2,4]))
            for j in range(count):
                rid=f'{tid}_room{j}';world.rooms[rid]=Room(id=rid,property_id=pid,room_type_id=tid,number=str(k*100+j),housekeeping=rng.choice(['clean','clean','dirty']),maintenance=rng.random()<.2)
            for offset in range(30):
                day=world.clock.date()+timedelta(days=offset);cid=tid+'_'+day.isoformat()
                world.calendar[cid]=CalendarDay(id=cid,property_id=pid,room_type_id=tid,day=day,sale_limit=rng.randint(1,count),closed=rng.random()<.08)
        for j in range(rng.randint(2,3)):
            rid=f'{pid}_rate{j}';world.rate_plans[rid]=RatePlan(id=rid,property_id=pid,name=f'Rate {j}',room_type_ids=rng.sample(tids,rng.randint(1,len(tids))),deposit_bps=rng.choice([0,2000,5000,10000]),free_cancel_hours=rng.choice([0,24,48,168]),late_fee=rng.choice(['first_night','full_stay']),weekend_surcharge_cents=rng.choice([0,500,1500]))
    from usami_stayops import adapter as legacy_adapter
    adapter=native_adapter or legacy_adapter
    if adapter.invariants(world):raise ValueError(str(adapter.invariants(world)))
    history=NativeHistory(adapter,world,diagnostics)
    for pid,prop in props.items():
        plans=[p for p in world.rate_plans.values() if p.property_id==pid]
        for j in range(rng.randint(1,4)):
            stage=rng.randint(0,cfg.lifecycle_steps)
            if not stage:continue
            plan=rng.choice(plans);tid=rng.choice(plan.room_type_ids);guest=rng.choice(tuple(guests));rid=f'{pid}_booking{j}'
            arrival=world.clock.date()+timedelta(days=rng.choice([0,0,1,2,5,10]));departure=arrival+timedelta(days=rng.randint(1,4))
            if not history.step(prop.owner_id,'create_hold',pid,pid,{'reservation_id':rid,'guest_id':guest,'room_type_id':tid,'rate_plan_id':plan.id,'occupants':1,'arrival':arrival.isoformat(),'departure':departure.isoformat(),'hold_minutes':rng.choice([15,30,60])}):continue
            if stage<2:continue
            booking=history.state.reservations[rid]
            if booking.deposit_cents:
                history.step(prop.owner_id,'capture_payment',pid,rid,{'amount_cents':booking.deposit_cents if stage==2 else booking.total_cents,'key':'opening','token':rng.choice(['card_approved','card_approved','card_declined'])})
            elif stage>2:
                history.step(prop.owner_id,'capture_payment',pid,rid,{'amount_cents':booking.total_cents,'key':'opening','token':'card_approved'})
            if stage<3 or not history.step(prop.owner_id,'confirm_reservation',pid,rid):continue
            if stage==4:
                history.step(prop.owner_id,'cancel_reservation',pid,rid);continue
            rooms=[r for r in history.state.rooms.values() if r.room_type_id==tid and not r.maintenance and r.housekeeping=='clean' and not r.current_stay_id]
            rng.shuffle(rooms)
            if stage>=5 and rooms:history.step(prop.owner_id,'assign_room',pid,rid,{'room_id':rooms[0].id})
            if stage>=6:history.step(prop.owner_id,'verify_guest',pid,rid)
            if stage>=7:history.step(prop.owner_id,'check_in',pid,rid)
            if stage>=8:history.step(prop.owner_id,'check_out',pid,rid)
    return history.finish()
