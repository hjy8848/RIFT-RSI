"""Deterministic room/calendar fixture, without a fixed Discovery task."""
from datetime import date,timedelta
from usami.world.generator import StableRandom
from .models import World,User,Property,Access,Guest,RoomType,Room,CalendarDay,RatePlan


def generate_world(seed=42):
    rng=StableRandom(seed)
    users={f'u{i}':User(id=f'u{i}',name=name) for i,name in enumerate(['Property owner','Front desk','Housekeeping','Guest','Finance','Partner owner'])}
    properties={f'p{i}':Property(id=f'p{i}',name=name+' '+str(rng.randint(100,999)),owner_id='u5' if i==1 else 'u0')
                for i,name in enumerate(['City Hotel','Partner Hotel','Airport Hotel'])}
    guests={'g0':Guest(id='g0',user_id='u3',name='Guest'), 'g1':Guest(id='g1',user_id='u5',name='Partner guest')}
    access={}
    for p,u,role in [('p0','u1','frontdesk'),('p0','u2','housekeeping'),('p0','u4','finance'),('p2','u1','viewer')]:
        key=p+'_'+u;access[key]=Access(id=key,property_id=p,user_id=u,role=role)
    world=World(seed=seed,users=users,properties=properties,access=access,guests=guests,room_types={},rooms={},calendar={},rate_plans={})
    for pid in properties:
        types=[]
        for kind,price in [('standard',6000),('suite',10000)]:
            tid=pid+'_'+kind;types.append(tid)
            world.room_types[tid]=RoomType(id=tid,property_id=pid,name=kind,nightly_cents=price,max_occupants=2 if kind=='standard' else 4)
            for ordinal in range(2):
                rid=tid+'_'+str(ordinal)
                world.rooms[rid]=Room(id=rid,property_id=pid,room_type_id=tid,number=str(100+ordinal+(100 if kind=='suite' else 0)),
                    housekeeping='dirty' if kind=='standard' and ordinal==0 else 'clean',maintenance=kind=='suite' and ordinal==0)
            for offset in range(17):
                day=date(2026,1,15)+timedelta(days=offset);key=tid+'_'+day.isoformat()
                world.calendar[key]=CalendarDay(id=key,property_id=pid,room_type_id=tid,day=day,sale_limit=2)
        for suffix,bps,fee,hours in [('flex',2000,'first_night',48),('prepaid',10000,'full_stay',168)]:
            key=pid+'_'+suffix
            world.rate_plans[key]=RatePlan(id=key,property_id=pid,name=suffix,room_type_ids=types,
                deposit_bps=bps,late_fee=fee,free_cancel_hours=hours)
    return world
