"""Unconditioned multi-order disputes; no task, model or failure is an input."""
from datetime import timedelta
from usami.randomness import StableRandom
from usami_harness.models import digest
from usami_benchmark.contracts import GeneratorConfig
from usami_dialogue.models import CustomerPrivateState
from .models import World,User,Merchant,Product,Order,Line,Evidence,ExternalEvent

GENERATOR_VERSION='aftersales-generator-v1'
RULES_VERSION='aftersales-0.1.0'


def generate_world(seed=42,config=None,*,diagnostics=None):
    cfg=GeneratorConfig.model_validate(config or {});rng=StableRandom(digest({'version':GENERATOR_VERSION,'seed':seed,'config':cfg.model_dump(mode='json')}))
    users={f'u{i}':User(id=f'u{i}',name=f'Buyer {i}') for i in range(4)}
    merchants={f'm{i}':Merchant(id=f'm{i}',name=f'Merchant {i}',refund_approval_limit_cents=rng.choice([1500,5000,20000])) for i in range(rng.randint(cfg.min_scopes,cfg.max_scopes))}
    world=World(seed=seed,users=users,merchants=merchants,products={},orders={},lines={},evidence={},profiles={})
    for mid in merchants:
        for j in range(rng.randint(cfg.min_resources,cfg.max_resources)):
            key=f'{mid}_product{j}';stock=rng.randint(0,6)
            world.products[key]=Product(id=key,merchant_id=mid,name=f'Item {j}',stock=stock,opening_stock=stock,returnable=rng.random()<.85,return_days=rng.choice([7,14,30]),return_required=rng.random()<.6,replacement_days=rng.choice([1,3,7]))
    for uid in users:
        for j in range(rng.randint(2,4)):
            oid=f'{uid}_order{j}';mid=rng.choice(tuple(merchants));delivered=rng.random()<.85;age=rng.choice([1,3,7,15,31,45])
            order=Order(id=oid,merchant_id=mid,customer_id=uid,status='delivered' if delivered else rng.choice(['processing','shipped']),
                delivered_at=world.clock-timedelta(days=age) if delivered else None,payment_channel='original_'+uid,paid_cents=0)
            world.orders[oid]=order
            products=[p for p in world.products.values() if p.merchant_id==mid]
            for k,product in enumerate(rng.sample(products,rng.randint(1,min(3,len(products))))):
                lid=oid+'_line'+str(k);line=Line(id=lid,order_id=oid,product_id=product.id,quantity=rng.randint(1,2),unit_price_cents=rng.choice([800,2000,6000]));world.lines[lid]=line
                order.paid_cents+=line.quantity*line.unit_price_cents;issue=rng.choice(['damaged','wrong_item','missing_parts']);eid='photo_'+lid
                world.evidence[eid]=Evidence(id=eid,order_id=oid,line_id=lid,holder_id=uid,issue=issue,simulated_valid=rng.random()<.85)
                facts={'order_id':oid,'line_id':lid,'issue':issue,'evidence_id':eid,'deadline':(world.clock+timedelta(days=rng.choice([1,3,10]))).isoformat(),'pickup_slot':rng.choice(['morning','afternoon'])}
                actions=('open_case','verify_evidence','request_manager_review','quote_resolution','commit_resolution','schedule_return','release_refund')
                world.profiles['profile_'+lid]=CustomerPrivateState(identity=uid,opening_request='I need help with an item from one of my purchases.',facts=facts,
                    preference_space={'resolution':['refund','exchange','reship']},initial_preferences={'resolution':rng.choice(['refund','exchange','reship'])},
                    confirmable_actions=actions,protected_objects=(oid,)).checked()
                for kind,outcome in [('manager_review',rng.choice(['approved','approved','denied'])),('return_received',rng.choice(['received','received','missing']))]:
                    key=kind+'_'+lid;world.external_events[key]=ExternalEvent(id=key,kind=kind,line_id=lid,at=world.clock+timedelta(hours=1 if kind=='manager_review' else 48),outcome=outcome)
    from .rules import invariants
    if invariants(world):raise ValueError(str(invariants(world)))
    return world
