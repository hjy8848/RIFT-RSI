"""Frozen simulator inputs; seeds change labels, not invented task structures."""
from usami.randomness import StableRandom
from usami_harness.models import signed_notice
from .models import World, User, Organization, Warehouse, Access, SKU, Lot, ReceiptNotice, Receipt, Order, OrderLine


def generate_world(seed=42):
    rng = StableRandom(seed)
    users = {f'u{i}': User(id=f'u{i}', name=f'Employee {i}') for i in range(6)}
    organizations = {'o0': Organization(id='o0', name='North Trading', owner_id='u0'),
                     'o1': Organization(id='o1', name='South Trading', owner_id='u5')}
    warehouses = {f'w{i}': Warehouse(id=f'w{i}', org_id='o0' if i < 2 else 'o1',
                                  name=f'{("Primary", "Regional", "Partner")[i]} {rng.randint(100,999)}') for i in range(3)}
    access = {}
    for warehouse, user, role in [('w0','u1','picker'),('w1','u2','receiver'),('w0','u2','viewer'),
                                  ('w0','u4','finance'),('w1','u1','viewer')]:
        key = warehouse + '_' + user
        access[key] = Access(id=key, warehouse_id=warehouse, user_id=user, role=role)
    skus = {'sku0': SKU(id='sku0', org_id='o0', name='Industrial Widget', unit_price_cents=1200),
            'sku1': SKU(id='sku1', org_id='o0', name='Glass Sensor', category='fragile', unit_price_cents=2300, unit_volume=2, simulated_return_condition='merchant_fault'),
            'sku2': SKU(id='sku2', org_id='o1', name='Partner Module', unit_price_cents=800)}
    world = World(seed=seed, users=users, organizations=organizations, warehouses=warehouses,
                  access=access, skus=skus, lots={}, receipt_notices={}, orders={})
    for warehouse, sku, quantity in [('w0','sku0',8),('w0','sku0',8),('w0','sku1',6),('w1','sku0',4),('w2','sku2',8)]:
        n = len(world.lots); lid, nid = f'lot{n}', f'opening{n}'
        binding = {'notice_id':nid,'warehouse_id':warehouse,'sku_id':sku,'quantity':quantity,'quality':'available'}
        world.receipt_notices[nid] = ReceiptNotice(id=nid, warehouse_id=warehouse, sku_id=sku,
            quantity=quantity, attestation=signed_notice(world.domain,'supplier',binding,world.clock))
        world.lots[lid] = Lot(id=lid,warehouse_id=warehouse,sku_id=sku,on_hand=quantity)
        world.receipts['receipt'+str(n)] = Receipt(id='receipt'+str(n),notice_id=nid,warehouse_id=warehouse,
                                                 lot_id=lid,quantity=quantity,key='opening')
    for i,(warehouse,sku,quality) in enumerate([('w0','sku0','available'),('w1','sku0','available'),('w0','sku1','quarantine')]):
        nid=f'inbound{i}';binding={'notice_id':nid,'warehouse_id':warehouse,'sku_id':sku,'quantity':8,'quality':quality}
        world.receipt_notices[nid]=ReceiptNotice(id=nid,warehouse_id=warehouse,sku_id=sku,quantity=8,
            quality=quality,attestation=signed_notice(world.domain,'supplier',binding,world.clock))
    for i,(warehouse,sku) in enumerate([('w0','sku0'),('w1','sku0'),('w2','sku2')]):
        key=f'order{i}'
        world.orders[key]=Order(id=key,warehouse_id=warehouse,customer_id='u3' if i<2 else 'u5',
            lines=[OrderLine(id='line0',sku_id=sku,quantity=4,unit_price_cents=skus[sku].unit_price_cents)],
            shipping_cents=warehouses[warehouse].shipping_cents)
    return world
