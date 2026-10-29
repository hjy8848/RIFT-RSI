"""Independent inventory topology and simulator-derived lifecycle distribution."""
from usami.randomness import StableRandom
from usami_benchmark.contracts import GeneratorConfig
from usami_benchmark.generation import NativeHistory
from usami_harness.models import signed_notice,digest
from usami_fulfillment.models import World,User,Organization,Warehouse,Access,SKU,Lot,ReceiptNotice,Receipt,Order,OrderLine
from usami_fulfillment.rules import invariants,physical_volume,price

GENERATOR_VERSION='fulfillment-structural-v1'
RULES_VERSION='fulfillment-0.1.0'


def generate_world(seed=42,config=None,*,diagnostics=None):
    cfg=GeneratorConfig.model_validate(config or {});rng=StableRandom(digest({'version':GENERATOR_VERSION,'seed':seed,'config':cfg.model_dump(mode='json')}))
    users={f'u{i}':User(id=f'u{i}',name=f'User {i}') for i in range(9)}
    orgs={f'o{i}':Organization(id=f'o{i}',name=f'Organization {i}',owner_id=f'u{i}') for i in range(rng.randint(2,3))}
    world=World(seed=seed,users=users,organizations=orgs,warehouses={},access={},skus={},lots={},receipt_notices={},orders={})
    for oid in orgs:
        for n in range(rng.randint(2,4)):
            sid=f'{oid}_sku{n}';world.skus[sid]=SKU(id=sid,org_id=oid,name=f'Product {n}',category=rng.choice(['general','fragile','restricted']),unit_price_cents=rng.choice([500,1200,2500]),unit_volume=rng.choice([1,2,3]),returnable=rng.random()<.85,return_window_days=rng.choice([7,14,30]),simulated_return_condition=rng.choice(['resalable','merchant_fault','customer_damage']))
    for n in range(rng.randint(cfg.min_scopes,cfg.max_scopes)):
        wid=f'w{n}';oid=rng.choice(tuple(orgs));world.warehouses[wid]=Warehouse(id=wid,org_id=oid,name=f'Warehouse {n}',receiving_enabled=rng.random()<.8,shipping_cents=rng.choice([0,300,700]))
        for uid in rng.sample(list(users),rng.randint(2,5)):
            aid=wid+'_'+uid;world.access[aid]=Access(id=aid,warehouse_id=wid,user_id=uid,role=rng.choice(['manager','picker','receiver','finance','viewer']))
        skus=[s for s in world.skus.values() if s.org_id==oid]
        for k in range(rng.randint(cfg.min_resources,cfg.max_resources)):
            sku=rng.choice(skus);lid=f'{wid}_lot{k}';nid=f'{wid}_opening{k}';quantity=rng.randint(2,15);quality=rng.choice(['available','available','quarantine'])
            binding={'notice_id':nid,'warehouse_id':wid,'sku_id':sku.id,'quantity':quantity,'quality':quality}
            world.receipt_notices[nid]=ReceiptNotice(id=nid,warehouse_id=wid,sku_id=sku.id,quantity=quantity,quality=quality,attestation=signed_notice(world.domain,'supplier',binding,world.clock))
            world.lots[lid]=Lot(id=lid,warehouse_id=wid,sku_id=sku.id,on_hand=quantity,quality=quality)
            rid=wid+'_receipt'+str(k);world.receipts[rid]=Receipt(id=rid,notice_id=nid,warehouse_id=wid,lot_id=lid,quantity=quantity,key='opening')
        for k in range(rng.randint(1,3)):
            sku=rng.choice(skus);nid=f'{wid}_inbound{k}';quantity=rng.randint(1,10);quality=rng.choice(['available','quarantine'])
            binding={'notice_id':nid,'warehouse_id':wid,'sku_id':sku.id,'quantity':quantity,'quality':quality}
            world.receipt_notices[nid]=ReceiptNotice(id=nid,warehouse_id=wid,sku_id=sku.id,quantity=quantity,quality=quality,attestation=signed_notice(world.domain,'supplier',binding,world.clock))
        world.warehouses[wid].capacity_volume=physical_volume(world,wid)+rng.choice([0,5,30,100])
        for k in range(rng.randint(1,3)):
            oid2=f'{wid}_order{k}';selected=rng.sample(skus,rng.randint(1,min(3,len(skus))))
            world.orders[oid2]=Order(id=oid2,warehouse_id=wid,customer_id=rng.choice(list(users)),shipping_cents=world.warehouses[wid].shipping_cents,
                lines=[OrderLine(id=f'line{j}',sku_id=s.id,quantity=rng.randint(1,8),unit_price_cents=s.unit_price_cents) for j,s in enumerate(selected)])
    if invariants(world):raise ValueError(str(invariants(world)))
    from usami_fulfillment import adapter
    history=NativeHistory(adapter,world,diagnostics)
    for order in world.orders.values():
        stage=rng.randint(0,cfg.lifecycle_steps);wid=order.warehouse_id;actor=orgs[world.warehouses[wid].org_id].owner_id
        if stage==0:continue
        history.step(actor,'capture_payment',wid,order.id,{'amount_cents':price(order),'key':'opening','token':rng.choice(['card_approved','card_approved','card_declined'])})
        if stage<2 or not history.step(actor,'confirm_order',wid,order.id):continue
        allocations=[]
        for line in order.lines:
            remaining=line.quantity
            for lot in world.lots.values():
                current=history.state.lots[lot.id]
                if lot.warehouse_id!=wid or lot.sku_id!=line.sku_id or current.quality!='available':continue
                quantity=min(remaining,current.on_hand-current.reserved)
                if quantity<1:continue
                rid=order.id+'_'+line.id+'_'+lot.id
                if history.step(actor,'reserve_inventory',wid,order.id,{'reservation_id':rid,'line_id':line.id,'lot_id':lot.id,'quantity':quantity}):
                    allocations.append(rid);remaining-=quantity
                if not remaining:break
        if stage<4:continue
        for rid in allocations:history.step(actor,'pick_reservation',wid,rid)
        if stage<5:continue
        for rid in allocations:history.step(actor,'pack_reservation',wid,rid)
        if stage<6 or not allocations:continue
        sid='shipment_'+order.id
        history.step(actor,'create_shipment',wid,order.id,{'shipment_id':sid,'reservation_ids':allocations,'key':'opening'})
        if stage>=7:history.step(actor,'dispatch_shipment',wid,sid)
        if stage>=8:history.step(actor,'confirm_delivery',wid,sid)
        if stage>=9:
            for rid in allocations:
                ret='return_'+rid
                if history.step(actor,'request_return',wid,sid,{'return_id':ret,'reservation_id':rid,'quantity':1,'reason':'customer_remorse'}):
                    history.step(actor,'receive_return',wid,ret)
                    history.step(actor,'inspect_return',wid,ret)
                    if stage>=10:history.step(actor,'restock_return',wid,ret)
    return history.finish()
