"""Unconditioned residential topology, contractual liability and technician resources."""
from datetime import timedelta
from usami.randomness import StableRandom
from usami_harness.models import digest
from usami_benchmark.contracts import GeneratorConfig
from usami_dialogue.models import CustomerPrivateState
from .models import World,User,Property,Building,Unit,Contract,Asset,Technician,Slot,Busy,ExternalEvent
GENERATOR_VERSION='property-generator-v1'
RULES_VERSION='property-0.1.0'


def generate_world(seed=42,config=None,*,diagnostics=None):
    cfg=GeneratorConfig.model_validate(config or {});rng=StableRandom(digest({'version':GENERATOR_VERSION,'seed':seed,'config':cfg.model_dump(mode='json')}))
    users={f'u{i}':User(id=f'u{i}',name=f'Resident {i}',contact=f'private_contact_{i}@example.invalid') for i in range(8)}
    props={f'p{i}':Property(id=f'p{i}',name=f'Property {i}') for i in range(rng.randint(cfg.min_scopes,cfg.max_scopes))}
    w=World(seed=seed,users=users,properties=props,buildings={},units={},contracts={},assets={},technicians={},slots={},background_busy={},profiles={})
    categories=['plumbing','electrical','hvac','lock'];symptoms={'plumbing':'leak','electrical':'sparks','hvac':'not_cooling','lock':'stuck'}
    for pid in props:
        key='contract_'+pid
        w.contracts[key]=Contract(id=key,property_id=pid,covered_categories=rng.sample(categories,rng.randint(1,4)),coverage_cents=rng.choice([0,2000,5000]),resident_approval_limit_cents=rng.choice([0,3000,10000]),normal_start_hour=rng.choice([8,9]),normal_end_hour=rng.choice([17,18]),urgent_response_hours=rng.choice([8,24]),emergency_response_hours=rng.choice([2,4]))
        for index in range(rng.randint(2,4)):
            tid=pid+'_tech'+str(index);tech=Technician(id=tid,property_id=pid,name=f'Technician {index}',skills=list(categories) if index==0 else rng.sample(categories,rng.randint(1,3)),on_call=index==0 or rng.random()<.4);w.technicians[tid]=tech
            for j in range(rng.randint(cfg.min_resources,cfg.max_resources)):
                day=rng.choice([0,0,1,2]);hour=rng.choice([13,15,17,22]) if day==0 else rng.choice([8,10,14,18])
                start=(w.clock+timedelta(days=day)).replace(hour=hour,minute=rng.choice([0,30]));sid=tid+'_slot'+str(j)
                slot=Slot(id=sid,technician_id=tid,property_id=pid,start=start,end=start+timedelta(minutes=rng.choice([60,90,120])));w.slots[sid]=slot
            if rng.random()<.4:
                selected=rng.choice([s for s in w.slots.values() if s.technician_id==tid]);bid='busy_'+tid
                w.background_busy[bid]=Busy(id=bid,technician_id=tid,property_id=pid,start=selected.start,end=selected.end)
        for j in range(rng.randint(1,2)):
            bid=pid+'_building'+str(j);w.buildings[bid]=Building(id=bid,property_id=pid)
            for k in range(rng.randint(1,2)):
                uid=bid+'_unit'+str(k);owner=rng.choice(tuple(users));residents=rng.sample([u for u in users if u!=owner],rng.randint(1,2))
                w.units[uid]=Unit(id=uid,building_id=bid,property_id=pid,owner_id=owner,resident_ids=residents)
                for n,category in enumerate(rng.sample(categories,rng.randint(1,3))):
                    aid=uid+'_asset'+str(n);faulty=rng.random()<.8;hazard=faulty and rng.random()<.2
                    asset=Asset(id=aid,unit_id=uid,property_id=pid,category=category,condition='faulty' if faulty else 'working',simulated_symptom=symptoms[category] if faulty else 'routine_check',simulated_hazard=hazard,
                        responsibility=rng.choice(['owner','resident','property']),repair_cost_cents=rng.choice([1000,5000,15000]),duration_minutes=rng.choice([60,90]));w.assets[aid]=asset
                    actor=rng.choice([owner]+residents);slots=[s.id for s in w.slots.values() if s.property_id==pid]
                    facts={'unit_id':uid,'asset_id':aid,'symptom':asset.simulated_symptom,'observed_hazard':hazard,
                        'earliest_access':(w.clock+timedelta(hours=rng.choice([0,1,3]))).isoformat(),
                        'latest_access':(w.clock+timedelta(days=rng.choice([1,2,3]))).replace(hour=20).isoformat(),'max_budget_cents':rng.choice([0,2000,10000,20000])}
                    w.profiles['profile_'+aid]=CustomerPrivateState(identity=actor,opening_request='I need help with a problem in my home.',facts=facts,
                        preference_space={'slot_id':slots,'feedback':['satisfied','dissatisfied']},initial_preferences={'slot_id':rng.choice(slots),'feedback':'satisfied'},
                        confirmable_actions=('open_ticket','triage_ticket','request_owner_review','request_inspection','quote_visit','book_visit','cancel_visit','close_ticket','file_complaint','escalate_complaint'),protected_objects=(uid,)).checked()
                    for kind,outcome,hours in [('owner_review',rng.choice(['approved']*3+['denied']),1),('inspection',rng.choice(['hazard','safe','safe']),2),('technician_visit',rng.choice(['repaired']*3+['unresolved','no_show']),None),('complaint_review',rng.choice(['reviewed','dismissed']),72)]:
                        eid=kind+'_'+aid;w.external_events[eid]=ExternalEvent(id=eid,asset_id=aid,kind=kind,timing='appointment_end' if hours is None else 'fixed',at=w.clock+timedelta(hours=hours) if hours is not None else None,outcome=outcome)
    from .rules import invariants
    errors=invariants(w)
    if errors:raise ValueError(str(errors))
    return w
