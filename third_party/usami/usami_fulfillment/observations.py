"""Scoped operational views; simulator ground truth is never model input."""
from .rules import role, physical_volume, shipped, paid, refund_limit, price


def _sku(sku):
    return sku.model_dump(mode='json',exclude={'simulated_return_condition'})


def _order(world, order):
    data=order.model_dump(mode='json')
    data.update(total_due_cents=price(order),shipped_quantity=shipped(world,order.id),delivered_quantity=shipped(world,order.id,delivered=True))
    return data


def observation(world, identity):
    if identity not in world.users:
        raise ValueError('unknown customer identity')
    staff={w for w in world.warehouses if role(world,identity,w)}
    own={o.warehouse_id for o in world.orders.values() if o.customer_id==identity}
    visible=staff|own
    orders={o.id for o in world.orders.values() if o.warehouse_id in staff or o.customer_id==identity}
    financial={w for w in staff if role(world,identity,w) in {'manager','finance'}}
    return {'domain':world.domain,'version':world.version,'clock':world.clock.isoformat(),
            'identity':world.users[identity].model_dump(mode='json'),
            'warehouses':[w.model_dump(mode='json') if w.id in staff else {'id':w.id,'name':w.name,'org_id':w.org_id,'shipping_cents':w.shipping_cents}
                          for w in world.warehouses.values() if w.id in visible],
            'access':[a.model_dump(mode='json') for a in world.access.values() if a.warehouse_id in staff],
            'catalog':[_sku(s) for s in world.skus.values() if any(w.org_id==s.org_id and w.id in visible for w in world.warehouses.values())],
            'lots':[l.model_dump(mode='json') for l in world.lots.values() if l.warehouse_id in staff],
            'inbound_receipts':[n.model_dump(mode='json',exclude={'attestation'}) for n in world.receipt_notices.values() if n.warehouse_id in staff],
            'orders':[_order(world,o) for o in world.orders.values() if o.id in orders],
            'reservations':[r.model_dump(mode='json') for r in world.reservations.values() if r.order_id in orders],
            'shipments':[s.model_dump(mode='json',exclude={'carrier_notice'}) for s in world.shipments.values() if s.order_id in orders],
            'transfers':[t.model_dump(mode='json',exclude={'receiving_notice'}) for t in world.transfers.values() if t.source_warehouse_id in staff or t.target_warehouse_id in staff],
            'returns':[r.model_dump(mode='json',exclude={'receipt_notice','inspection_notice'}) for r in world.returns.values() if r.reservation_id in world.reservations and world.reservations[r.reservation_id].order_id in orders],
            'payments':[p.model_dump(mode='json',exclude={'gateway_notice'}) for p in world.payments.values() if p.warehouse_id in financial or world.orders[p.order_id].customer_id==identity],
            'public_directory':[{'id':u.id,'name':u.name} for u in world.users.values()]}


def visible_ids(view):
    return {obj['id'] for values in view.values() if isinstance(values,list) for obj in values if isinstance(obj,dict) and 'id' in obj}


def query_data(world, actor, op):
    name,rid=op.name,op.resource_id
    if name=='get_warehouse':
        w=world.warehouses[rid]
        if not role(world,actor,w.id):
            return {'id':w.id,'name':w.name,'org_id':w.org_id,'shipping_cents':w.shipping_cents}
        return {**w.model_dump(mode='json'),'used_volume':physical_volume(world,rid)}
    if name=='get_catalog':
        return {'products':[_sku(s) for s in world.skus.values() if s.org_id==world.warehouses[rid].org_id]}
    if name=='get_inventory':
        return {'lots':[{**l.model_dump(mode='json'),'available':l.on_hand-l.reserved if l.quality=='available' else 0,
                         'shipped':sum(world.reservations[r].quantity for s in world.shipments.values() if s.status in {'shipped','delivered'}
                                       for r in s.reservation_ids if world.reservations[r].lot_id==l.id)}
                        for l in world.lots.values() if l.warehouse_id==rid]}
    if name=='get_order':
        return _order(world,world.orders[rid])
    if name=='get_shipment':
        return world.shipments[rid].model_dump(mode='json',exclude={'carrier_notice'})
    if name=='get_returns':
        return {'returns':[r.model_dump(mode='json',exclude={'receipt_notice','inspection_notice'}) for r in world.returns.values()
                           if world.reservations[r.reservation_id].order_id==rid]}
    if name=='get_payment_ledger':
        return {'payments':[p.model_dump(mode='json',exclude={'gateway_notice'}) for p in world.payments.values() if p.order_id==rid],
                'charged_cents':paid(world,rid),'refunded_cents':paid(world,rid,'refund'),'refundable_cents':refund_limit(world,world.orders[rid])}
    raise ValueError('unknown read interface')
