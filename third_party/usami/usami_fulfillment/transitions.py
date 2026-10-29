"""Actual mutations. No Reference import; Sandbox bypasses business guards.

Only actor/resource existence, conflicting entity identity and representability
are guarded here. A failed mutation rolls back the whole cloned transaction.
"""
from usami_harness.models import Result, signed_notice, digest
from .models import Lot, Receipt, Order, OrderLine, Reservation, Handling, Shipment, Transfer, Return, Payment


def _target(world, name, rid):
    mapping = {
        'receive_inventory':'receipt_notices','inspect_lot':'lots','dispatch_transfer':'lots',
        'receive_transfer':'transfers','release_reservation':'reservations','pick_reservation':'reservations',
        'unpick_reservation':'reservations','pack_reservation':'reservations','unpack_reservation':'reservations',
        'cancel_shipment':'shipments','dispatch_shipment':'shipments','confirm_delivery':'shipments',
        'request_return':'shipments','get_shipment':'shipments','cancel_return':'returns','receive_return':'returns',
        'inspect_return':'returns','restock_return':'returns','create_order':'warehouses',
        'change_warehouse_capacity':'warehouses','set_warehouse_receiving':'warehouses',
        'get_warehouse':'warehouses','get_inventory':'warehouses','get_catalog':'warehouses'}
    return getattr(world,mapping.get(name,'orders')).get(rid)


def _line(order, line_id):
    return next((line for line in order.lines if line.id==line_id),None)


def _shipped(world, order, delivered=False):
    quantity = 0
    for shipment in world.shipments.values():
        if shipment.order_id!=order or shipment.status not in {'shipped','delivered'} or (delivered and shipment.status!='delivered'):
            continue
        for key in shipment.reservation_ids:
            quantity += world.reservations[key].quantity
    return quantity


def apply(world, actor, op, event_id):
    fail = lambda code: (world.clone(),Result(success=False,error_code=code))
    if actor not in world.users:
        return fail('UNKNOWN_ACTOR')
    if op.scope_id not in world.warehouses:
        return fail('WAREHOUSE_NOT_FOUND')
    new = world.clone()
    name,rid,p = op.name,op.resource_id,op.params
    obj = _target(new,name,rid)
    if obj is None:
        return fail('RESOURCE_NOT_FOUND')
    try:
        if op.is_query:
            from .observations import query_data
            data = query_data(world,actor,op)
        elif name == 'receive_inventory':
            prior = next((r for r in new.receipts.values() if r.notice_id==rid and r.key==p['key'] and r.lot_id==p['lot_id']),None)
            if prior:
                data={'receipt_id':prior.id,'lot_id':prior.lot_id}
            else:
                lot = new.lots.get(p['lot_id'])
                if lot and (lot.warehouse_id!=op.scope_id or lot.sku_id!=obj.sku_id or lot.quality!=obj.quality):
                    return fail('LOT_CONFLICT')
                if lot:
                    lot.on_hand += obj.quantity
                else:
                    new.lots[p['lot_id']]=Lot(id=p['lot_id'],warehouse_id=op.scope_id,sku_id=obj.sku_id,on_hand=obj.quantity,quality=obj.quality)
                key='receipt_'+digest([rid,p['key']])[:16]
                if key in new.receipts:
                    key += '_attempt_'+str(event_id)
                new.receipts[key]=Receipt(id=key,notice_id=rid,warehouse_id=op.scope_id,lot_id=p['lot_id'],quantity=obj.quantity,key=p['key'])
                data={'receipt_id':key,'lot_id':p['lot_id']}
        elif name == 'inspect_lot':
            if obj.quality != 'available':
                obj.quality='available'
                new.lot_inspections[rid]=signed_notice(world.domain,'stock-inspector',{'lot_id':rid,'warehouse_id':op.scope_id,'outcome':'passed'},world.clock)
            data={'lot_id':rid,'quality':'available'}
        elif name == 'dispatch_transfer':
            if p['target_warehouse_id'] not in new.warehouses:
                return fail('TRANSFER_DESTINATION_INVALID')
            existing = new.transfers.get(p['transfer_id'])
            if existing:
                if [existing.source_lot_id,existing.target_warehouse_id,existing.target_lot_id,existing.quantity,existing.key] != [rid,p['target_warehouse_id'],p['target_lot_id'],p['quantity'],p['key']]:
                    return fail('TRANSFER_ID_CONFLICT')
                data={'transfer_id':existing.id,'status':existing.status}
            else:
                obj.on_hand -= p['quantity']
                binding={'transfer_id':p['transfer_id'],'source_lot_id':rid,'target_warehouse_id':p['target_warehouse_id'],'quantity':p['quantity']}
                transfer=Transfer(id=p['transfer_id'],source_warehouse_id=op.scope_id,target_warehouse_id=p['target_warehouse_id'],source_lot_id=rid,
                    target_lot_id=p['target_lot_id'],quantity=p['quantity'],key=p['key'],receiving_notice=signed_notice(world.domain,'transfer-carrier',binding,world.clock))
                new.transfers[transfer.id]=transfer;data={'transfer_id':transfer.id,'status':transfer.status}
        elif name == 'receive_transfer':
            if obj.source_lot_id not in new.lots:
                return fail('RESOURCE_NOT_FOUND')
            if obj.status != 'received':
                sku=new.lots[obj.source_lot_id].sku_id;lot=new.lots.get(obj.target_lot_id)
                if lot and (lot.warehouse_id!=op.scope_id or lot.sku_id!=sku):
                    return fail('LOT_CONFLICT')
                if lot:
                    lot.on_hand += obj.quantity
                else:
                    new.lots[obj.target_lot_id]=Lot(id=obj.target_lot_id,warehouse_id=op.scope_id,sku_id=sku,on_hand=obj.quantity)
                obj.status='received'
            data={'transfer_id':rid,'status':'received','lot_id':obj.target_lot_id}
        elif name == 'create_order':
            if p['customer_id'] not in new.users or any(l['sku_id'] not in new.skus for l in p['lines']):
                return fail('RESOURCE_NOT_FOUND')
            order=Order(id=p['order_id'],warehouse_id=op.scope_id,customer_id=p['customer_id'],shipping_cents=obj.shipping_cents,
                        lines=[OrderLine(**l,unit_price_cents=new.skus[l['sku_id']].unit_price_cents) for l in p['lines']])
            if order.id in new.orders and new.orders[order.id]!=order:
                return fail('ORDER_ID_CONFLICT')
            new.orders[order.id]=order;data={'order_id':order.id,'status':order.status}
        elif name == 'modify_order_line':
            line=_line(obj,p['line_id'])
            if not line:
                return fail('ORDER_LINE_NOT_FOUND')
            line.quantity=p['quantity'];data={'order_id':rid,'line_id':line.id,'quantity':line.quantity}
        elif name in {'capture_payment','refund_order'}:
            kind='charge' if name=='capture_payment' else 'refund';token=p.get('token','card_approved')
            prior=next((pay for pay in new.payments.values() if pay.order_id==rid and pay.kind==kind and pay.key==p['key']
                        and pay.amount_cents==p['amount_cents'] and pay.token==token),None)
            if prior:
                data={'payment_id':prior.id,'outcome':prior.outcome}
            else:
                outcome='succeeded' if token=='card_approved' else 'failed'
                key='payment_'+digest([rid,kind,p['key']])[:16]
                if key in new.payments:
                    key += '_attempt_'+str(event_id)
                binding={'order_id':rid,'kind':kind,'amount_cents':p['amount_cents'],'key':p['key'],'token':token,'outcome':outcome}
                new.payments[key]=Payment(id=key,warehouse_id=op.scope_id,order_id=rid,kind=kind,amount_cents=p['amount_cents'],key=p['key'],token=token,
                    outcome=outcome,gateway_notice=signed_notice(world.domain,'payment-gateway',binding,world.clock))
                data={'payment_id':key,'outcome':outcome}
        elif name == 'confirm_order':
            if obj.status=='draft':
                obj.status='confirmed'
            data={'order_id':rid,'status':obj.status}
        elif name == 'reserve_inventory':
            lot=new.lots.get(p['lot_id']);line=_line(obj,p['line_id'])
            if not lot or not line:
                return fail('ALLOCATION_REFERENCE_MISSING')
            reservation=Reservation(id=p['reservation_id'],warehouse_id=op.scope_id,order_id=rid,line_id=line.id,lot_id=lot.id,quantity=p['quantity'])
            prior=new.reservations.get(reservation.id)
            if prior:
                if prior!=reservation:
                    return fail('RESERVATION_ID_CONFLICT')
            else:
                lot.reserved += p['quantity'];new.reservations[reservation.id]=reservation
            data={'reservation_id':reservation.id,'status':'reserved'}
        elif name in {'release_reservation','pick_reservation','unpick_reservation','pack_reservation','unpack_reservation'}:
            lot=new.lots.get(obj.lot_id)
            if not lot:
                return fail('RESOURCE_NOT_FOUND')
            end={'release_reservation':'released','pick_reservation':'picked','unpick_reservation':'reserved','pack_reservation':'packed','unpack_reservation':'picked'}[name]
            if obj.status!=end:
                if name=='release_reservation':
                    if obj.status=='reserved':lot.reserved -= obj.quantity
                    elif obj.status in {'picked','packed'}:lot.picked -= obj.quantity;lot.on_hand += obj.quantity
                    obj.released_quantity += obj.quantity
                elif name in {'pick_reservation','pack_reservation'}:
                    if obj.status not in {'picked','packed','shipped'}:
                        lot.on_hand -= obj.quantity;lot.picked += obj.quantity
                        if obj.status=='reserved':lot.reserved -= obj.quantity
                    kind='pick' if name=='pick_reservation' else 'pack';key=kind+'_'+rid
                    new.handling[key]=Handling(id=key,reservation_id=rid,warehouse_id=op.scope_id,kind=kind,quantity=obj.quantity)
                elif name=='unpick_reservation':
                    lot.picked -= obj.quantity;lot.on_hand += obj.quantity;lot.reserved += obj.quantity
                    new.handling.pop('pick_'+rid,None)
                else:
                    new.handling.pop('pack_'+rid,None)
                obj.status=end
            data={'reservation_id':rid,'status':end}
        elif name == 'create_shipment':
            if any(r not in new.reservations for r in p['reservation_ids']):
                return fail('SHIPMENT_ALLOCATION_INVALID')
            prior=new.shipments.get(p['shipment_id'])
            if prior:
                if prior.order_id!=rid or prior.key!=p['key'] or prior.reservation_ids!=p['reservation_ids']:
                    return fail('SHIPMENT_ID_CONFLICT')
                data={'shipment_id':prior.id,'status':prior.status}
            else:
                shipment=Shipment(id=p['shipment_id'],warehouse_id=op.scope_id,order_id=rid,reservation_ids=p['reservation_ids'],key=p['key'])
                new.shipments[shipment.id]=shipment;data={'shipment_id':shipment.id,'status':shipment.status}
        elif name == 'cancel_shipment':
            obj.status='canceled';data={'shipment_id':rid,'status':'canceled'}
        elif name == 'dispatch_shipment':
            if obj.status not in {'shipped','delivered'}:
                for allocation in obj.reservation_ids:
                    reservation=new.reservations.get(allocation)
                    if not reservation or reservation.lot_id not in new.lots:
                        return fail('RESOURCE_NOT_FOUND')
                    lot=new.lots[reservation.lot_id]
                    if reservation.status in {'picked','packed'}:lot.picked -= reservation.quantity
                    elif reservation.status!='shipped':
                        lot.on_hand -= reservation.quantity
                        if reservation.status=='reserved':lot.reserved -= reservation.quantity
                    reservation.status='shipped'
                obj.status='shipped'
                binding={'shipment_id':rid,'warehouse_id':op.scope_id,'order_id':obj.order_id,'reservation_ids':obj.reservation_ids,'outcome':'delivered'}
                obj.carrier_notice=signed_notice(world.domain,'carrier',binding,world.clock)
                order=new.orders[obj.order_id];remaining=sum(l.quantity-l.canceled for l in order.lines)
                order.status='shipped' if _shipped(new,order.id)==remaining else 'partially_shipped'
            data={'shipment_id':rid,'status':obj.status}
        elif name == 'confirm_delivery':
            if obj.status!='delivered':
                obj.status='delivered';obj.delivered_at=world.clock
                order=new.orders[obj.order_id]
                order.status='delivered' if _shipped(new,order.id,True)==sum(l.quantity-l.canceled for l in order.lines) else 'partially_delivered'
            data={'shipment_id':rid,'status':'delivered'}
        elif name in {'cancel_order','cancel_order_line'}:
            lines=obj.lines if name=='cancel_order' else [_line(obj,p['line_id'])]
            if not all(lines):
                return fail('ORDER_LINE_NOT_FOUND')
            if name=='cancel_order':
                for shipment in new.shipments.values():
                    if shipment.order_id==rid and shipment.status=='label_created':shipment.status='canceled'
            for line in lines:
                quantity=line.quantity-line.canceled if name=='cancel_order' else p['quantity']
                allocations=[r for r in new.reservations.values() if r.order_id==rid and r.line_id==line.id and r.status!='released']
                free=line.quantity-line.canceled-sum(r.quantity for r in allocations)
                needed=max(0,quantity-free)
                for reservation in sorted(allocations,key=lambda r:r.id):
                    if reservation.status!='reserved' or any(reservation.id in s.reservation_ids and s.status!='canceled' for s in new.shipments.values()):continue
                    take=min(needed,reservation.quantity)
                    if not take:break
                    new.lots[reservation.lot_id].reserved -= take;reservation.released_quantity += take
                    if take==reservation.quantity:reservation.status='released'
                    else:reservation.quantity -= take
                    needed -= take
                line.canceled += quantity
            if all(l.canceled==l.quantity for l in obj.lines):obj.status='canceled'
            elif _shipped(new,rid)==sum(l.quantity-l.canceled for l in obj.lines):
                obj.status='delivered' if _shipped(new,rid,True)==_shipped(new,rid) else 'shipped'
            data={'order_id':rid,'status':obj.status}
        elif name == 'request_return':
            reservation=new.reservations.get(p['reservation_id'])
            if not reservation:
                return fail('RETURN_ALLOCATION_INVALID')
            prior=new.returns.get(p['return_id'])
            if prior:
                if [prior.shipment_id,prior.reservation_id,prior.quantity,prior.reason] != [rid,p['reservation_id'],p['quantity'],p['reason']]:
                    return fail('RETURN_ID_CONFLICT')
                data={'return_id':prior.id,'status':prior.status}
            else:
                binding={'return_id':p['return_id'],'warehouse_id':op.scope_id,'reservation_id':reservation.id,'quantity':p['quantity']}
                ret=Return(id=p['return_id'],warehouse_id=op.scope_id,shipment_id=rid,reservation_id=reservation.id,quantity=p['quantity'],reason=p['reason'],
                           receipt_notice=signed_notice(world.domain,'return-carrier',binding,world.clock))
                new.returns[ret.id]=ret;data={'return_id':ret.id,'status':ret.status}
        elif name == 'cancel_return':
            obj.status='canceled';data={'return_id':rid,'status':'canceled'}
        elif name == 'receive_return':
            if obj.status not in {'received','inspected','restocked'}:obj.status='received'
            data={'return_id':rid,'status':obj.status}
        elif name == 'inspect_return':
            if obj.status not in {'inspected','restocked'}:
                reservation=new.reservations[obj.reservation_id];lot=new.lots[reservation.lot_id]
                condition=new.skus[lot.sku_id].simulated_return_condition
                obj.accepted=obj.quantity if condition=='resalable' else 0
                obj.scrapped=obj.quantity-obj.accepted;obj.refundable_units=0 if condition=='customer_damage' else obj.quantity
                obj.status='inspected'
                obj.inspection_notice=signed_notice(world.domain,'inspector',{'return_id':rid,'accepted':obj.accepted,'scrapped':obj.scrapped,'refundable_units':obj.refundable_units},world.clock)
            data={'return_id':rid,'accepted':obj.accepted,'scrapped':obj.scrapped,'refundable_units':obj.refundable_units}
        elif name == 'restock_return':
            if obj.status!='restocked':
                lot=new.lots[new.reservations[obj.reservation_id].lot_id]
                lot.on_hand += obj.accepted if obj.accepted else obj.quantity
                obj.status='restocked'
            data={'return_id':rid,'status':'restocked'}
        elif name == 'change_warehouse_capacity':
            obj.capacity_volume=p['capacity_volume'];data=p.copy()
        elif name == 'set_warehouse_receiving':
            obj.receiving_enabled=p['enabled'];data={'receiving_enabled':p['enabled']}
        else:
            return fail('UNKNOWN_OPERATION')
        return new,Result(success=True,data=data)
    except (ValueError,KeyError,TypeError):
        return fail('MUTATION_NOT_REPRESENTABLE')
