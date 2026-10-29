"""Independent inventory/fulfillment specification; no actual applier import."""
from datetime import datetime, timedelta
from usami_harness.models import Decision, signed_notice, notice_valid, digest
from .models import Lot, Receipt, Order, OrderLine, Reservation, Handling, Shipment, Transfer, Return, Payment

POLICY = {
    'STOCK_INSUFFICIENT': 'Only unreserved, quality-released physical stock may be allocated or transferred.',
    'ORDER_NOT_CONFIRMED': 'Capture the required price before confirmation, then allocate stock.',
    'SHIPMENT_NOT_PACKED': 'Every shipment allocation must be picked and packed before dispatch.',
    'ORDER_ALREADY_SHIPPED': 'Ordinary whole-order cancellation never reverses shipping. Cancel only unallocated remainder explicitly.',
    'RETURN_EXCEEDS_DELIVERY': 'Return requests cannot exceed delivered allocations, including previous open returns.',
    'REFUND_EXCEEDS_ELIGIBLE': 'Refund is bounded by paid balance and canceled/inspected refundable goods; shipped transport fees are nonrefundable.',
    'WAREHOUSE_SCOPE_MISMATCH': 'All linked goods/orders/records must belong to the declared warehouse.',
    'CAPACITY_EXCEEDED': 'Inbound goods and quarantined returns must fit physical warehouse volume.',
    'EXTERNAL_ATTESTATION_REQUIRED': 'Delivery, supplier receipt, payment and inspection consume host simulator evidence, never a Service claim.',
}


def role(world, actor, warehouse):
    w = world.warehouses.get(warehouse)
    if w and w.org_id in world.organizations and world.organizations[w.org_id].owner_id == actor:
        return 'manager'
    return next((a.role for a in world.access.values() if a.user_id == actor and a.warehouse_id == warehouse), None)


def get_line(order, line_id):
    return next((line for line in order.lines if line.id == line_id), None)


def shipments_for(world, order_id):
    return [s for s in world.shipments.values() if s.order_id == order_id and s.status in {'shipped','delivered'}]


def shipped(world, order_id, line_id=None, delivered=False):
    return sum(world.reservations[r].quantity for s in shipments_for(world,order_id)
               if not delivered or s.status == 'delivered' for r in s.reservation_ids
               if r in world.reservations and (line_id is None or world.reservations[r].line_id == line_id))


def allocated(world, order_id, line_id):
    return sum(r.quantity for r in world.reservations.values() if r.order_id == order_id and r.line_id == line_id and r.status != 'released')


def paid(world, order_id, kind='charge'):
    return sum(p.amount_cents for p in world.payments.values() if p.order_id == order_id and p.kind == kind and p.outcome == 'succeeded')


def price(order):
    goods = sum((line.quantity-line.canceled) * line.unit_price_cents for line in order.lines)
    return goods + order.shipping_cents if goods else 0


def refund_limit(world, order):
    eligible = sum(line.canceled * line.unit_price_cents for line in order.lines)
    if all(line.canceled == line.quantity for line in order.lines) and not shipments_for(world,order.id):
        eligible += order.shipping_cents
    for ret in world.returns.values():
        reservation = world.reservations.get(ret.reservation_id)
        if reservation and reservation.order_id == order.id and ret.status in {'inspected','restocked'}:
            line = get_line(order,reservation.line_id)
            if line:
                eligible += ret.refundable_units * line.unit_price_cents
    refunded = paid(world,order.id,'refund')
    return min(paid(world,order.id)-refunded, eligible-refunded)


def physical_volume(world, warehouse_id):
    lots = sum((lot.on_hand+lot.picked)*world.skus[lot.sku_id].unit_volume for lot in world.lots.values()
               if lot.warehouse_id == warehouse_id and lot.sku_id in world.skus)
    quarantine = sum((ret.quantity if ret.status=='received' else ret.accepted)*world.skus[world.lots[world.reservations[ret.reservation_id].lot_id].sku_id].unit_volume
                     for ret in world.returns.values() if ret.warehouse_id == warehouse_id and ret.status in {'received','inspected'}
                     and ret.reservation_id in world.reservations and world.reservations[ret.reservation_id].lot_id in world.lots
                     and world.lots[world.reservations[ret.reservation_id].lot_id].sku_id in world.skus)
    return lots + quarantine


def attested(world, notice, source, binding):
    try:
        return (notice_valid(notice) and notice['domain'] == world.domain and notice['source'] == source+'-simulator-v0.1'
                and notice['binding'] == binding and datetime.fromisoformat(notice['at']) <= world.clock)
    except (KeyError, ValueError, TypeError):
        return False


def target(world, op):
    name, rid = op.name, op.resource_id
    if name in {'receive_inventory'}:
        return world.receipt_notices.get(rid)
    if name in {'inspect_lot','dispatch_transfer'}:
        return world.lots.get(rid)
    if name == 'receive_transfer':
        return world.transfers.get(rid)
    if name in {'release_reservation','pick_reservation','unpick_reservation','pack_reservation','unpack_reservation'}:
        return world.reservations.get(rid)
    if name in {'cancel_shipment','dispatch_shipment','confirm_delivery','request_return','get_shipment'}:
        return world.shipments.get(rid)
    if name in {'cancel_return','receive_return','inspect_return','restock_return'}:
        return world.returns.get(rid)
    if name in {'create_order','change_warehouse_capacity','set_warehouse_receiving','get_warehouse','get_inventory','get_catalog'}:
        return world.warehouses.get(rid)
    return world.orders.get(rid)


def invariants(world):
    errors = []
    def add(code,path):
        errors.append({'code':code,'path':path})
    collections = ('users','organizations','warehouses','access','skus','lots','receipt_notices','receipts','orders',
                   'reservations','handling','shipments','transfers','returns','payments')
    for collection in collections:
        for key,obj in getattr(world,collection).items():
            if key != obj.id:
                add('ENTITY_KEY_MISMATCH',collection+'/'+key)
    for org in world.organizations.values():
        if org.owner_id not in world.users:
            add('OWNER_MISSING','organizations/'+org.id)
    seen = set()
    for access in world.access.values():
        if access.user_id not in world.users or access.warehouse_id not in world.warehouses:
            add('ACCESS_REFERENCE_INVALID','access/'+access.id)
        if (access.user_id,access.warehouse_id) in seen:
            add('DUPLICATE_SCOPED_ACCESS','access/'+access.id)
        seen.add((access.user_id,access.warehouse_id))
    for w in world.warehouses.values():
        if w.org_id not in world.organizations:
            add('WAREHOUSE_ORGANIZATION_INVALID','warehouses/'+w.id)
        if physical_volume(world,w.id) > w.capacity_volume:
            add('CAPACITY_EXCEEDED','warehouses/'+w.id)
    notice_receipts = set()
    for receipt in world.receipts.values():
        notice,lot = world.receipt_notices.get(receipt.notice_id),world.lots.get(receipt.lot_id)
        if not notice or not lot or notice.warehouse_id != receipt.warehouse_id or lot.warehouse_id != receipt.warehouse_id or notice.sku_id != lot.sku_id or receipt.quantity != notice.quantity:
            add('RECEIPT_BINDING_INVALID','receipts/'+receipt.id)
        if receipt.notice_id in notice_receipts:
            add('DUPLICATE_RECEIPT','receipts/'+receipt.id)
        notice_receipts.add(receipt.notice_id)
    for notice in world.receipt_notices.values():
        binding = {'notice_id':notice.id,'warehouse_id':notice.warehouse_id,'sku_id':notice.sku_id,'quantity':notice.quantity,'quality':notice.quality}
        if not attested(world,notice.attestation,'supplier',binding):
            add('SUPPLIER_ATTESTATION_INVALID','receipt_notices/'+notice.id)
        if notice.warehouse_id not in world.warehouses or notice.sku_id not in world.skus:
            add('SUPPLIER_REFERENCE_INVALID','receipt_notices/'+notice.id)
    for lot_id,notice in world.lot_inspections.items():
        lot=world.lots.get(lot_id)
        if not lot or not attested(world,notice,'stock-inspector',{'lot_id':lot_id,'warehouse_id':lot.warehouse_id,'outcome':'passed'}):
            add('STOCK_INSPECTION_ATTESTATION_INVALID','lot_inspections/'+lot_id)
    for lot in world.lots.values():
        if lot.warehouse_id not in world.warehouses or lot.sku_id not in world.skus:
            add('LOT_REFERENCE_INVALID','lots/'+lot.id)
            continue
        if world.warehouses[lot.warehouse_id].org_id != world.skus[lot.sku_id].org_id:
            add('CROSS_ORGANIZATION_STOCK','lots/'+lot.id)
        if lot.reserved > lot.on_hand:
            add('STOCK_OVERALLOCATED','lots/'+lot.id)
        rs = [r for r in world.reservations.values() if r.lot_id == lot.id]
        if lot.reserved != sum(r.quantity for r in rs if r.status == 'reserved') or lot.picked != sum(r.quantity for r in rs if r.status in {'picked','packed'}):
            add('STOCK_BUCKET_MISMATCH','lots/'+lot.id)
        if lot.quality == 'quarantine' and any(r.status in {'reserved','picked','packed'} for r in rs):
            add('QUARANTINED_STOCK_ALLOCATED','lots/'+lot.id)
        quarantined_source=any(r.lot_id==lot.id and r.notice_id in world.receipt_notices and world.receipt_notices[r.notice_id].quality=='quarantine' for r in world.receipts.values())
        if lot.quality=='available' and quarantined_source and lot.id not in world.lot_inspections:
            add('QUALITY_RELEASE_ATTESTATION_MISSING','lots/'+lot.id)
        supply = sum(r.quantity for r in world.receipts.values() if r.lot_id == lot.id)
        supply += sum(t.quantity for t in world.transfers.values() if t.target_lot_id == lot.id and t.status == 'received')
        supply += sum(ret.accepted for ret in world.returns.values() if ret.status == 'restocked'
                      and ret.reservation_id in world.reservations and world.reservations[ret.reservation_id].lot_id == lot.id)
        consumption = lot.on_hand + lot.picked + sum(t.quantity for t in world.transfers.values() if t.source_lot_id == lot.id)
        consumption += sum(world.reservations[r].quantity for s in world.shipments.values() if s.status in {'shipped','delivered'}
                           for r in s.reservation_ids if r in world.reservations and world.reservations[r].lot_id == lot.id)
        if supply != consumption:
            add('INVENTORY_NOT_CONSERVED','lots/'+lot.id)
    for order in world.orders.values():
        if order.warehouse_id not in world.warehouses or order.customer_id not in world.users or len({l.id for l in order.lines}) != len(order.lines):
            add('ORDER_REFERENCE_INVALID','orders/'+order.id)
        for line in order.lines:
            sku = world.skus.get(line.sku_id)
            if not sku or sku.unit_price_cents != line.unit_price_cents or (order.warehouse_id in world.warehouses and sku.org_id != world.warehouses[order.warehouse_id].org_id):
                add('ORDER_PRODUCT_INVALID','orders/'+order.id)
            if line.canceled + allocated(world,order.id,line.id) > line.quantity:
                add('ORDER_OVERALLOCATED','orders/'+order.id)
        if paid(world,order.id,'refund') > paid(world,order.id):
            add('REFUND_EXCEEDS_PAYMENT','orders/'+order.id)
        if refund_limit(world,order) < 0:
            add('REFUND_EXCEEDS_ELIGIBLE','orders/'+order.id)
        if paid(world,order.id) > sum(l.quantity*l.unit_price_cents for l in order.lines)+order.shipping_cents:
            add('PAYMENT_EXCEEDS_ORDER_PRICE','orders/'+order.id)
        credit = sum(ret.refundable_units*get_line(order,world.reservations[ret.reservation_id].line_id).unit_price_cents
                     for ret in world.returns.values() if ret.status in {'inspected','restocked'} and ret.reservation_id in world.reservations
                     and world.reservations[ret.reservation_id].order_id==order.id and get_line(order,world.reservations[ret.reservation_id].line_id))
        if order.status not in {'draft','canceled'} and paid(world,order.id)-paid(world,order.id,'refund') < max(0,price(order)-credit):
            add('ORDER_PAYMENT_REQUIRED','orders/'+order.id)
        if order.status=='draft' and any(r.order_id==order.id and r.status!='released' for r in world.reservations.values()):
            add('DRAFT_ORDER_ALLOCATED','orders/'+order.id)
        if order.status == 'canceled' and shipped(world,order.id):
            add('SHIPPED_ORDER_CANCELED','orders/'+order.id)
    for r in world.reservations.values():
        order,lot = world.orders.get(r.order_id),world.lots.get(r.lot_id)
        line = get_line(order,r.line_id) if order else None
        if not order or not lot or not line or r.warehouse_id != order.warehouse_id or lot.warehouse_id != r.warehouse_id or line.sku_id != lot.sku_id:
            add('RESERVATION_BINDING_INVALID','reservations/'+r.id)
        if r.status in {'picked','packed','shipped'}:
            proof = world.handling.get('pick_'+r.id)
            if not proof or proof.quantity != r.quantity or proof.reservation_id != r.id or proof.kind != 'pick' or proof.warehouse_id != r.warehouse_id:
                add('PICK_EVIDENCE_MISSING','reservations/'+r.id)
        if r.status in {'packed','shipped'}:
            proof = world.handling.get('pack_'+r.id)
            if not proof or proof.quantity != r.quantity or proof.reservation_id != r.id or proof.kind != 'pack' or proof.warehouse_id != r.warehouse_id:
                add('PACK_EVIDENCE_MISSING','reservations/'+r.id)
    allocation_shipments = set()
    for proof in world.handling.values():
        reservation=world.reservations.get(proof.reservation_id)
        if not reservation or reservation.warehouse_id!=proof.warehouse_id or proof.id!=proof.kind+'_'+proof.reservation_id or reservation.quantity!=proof.quantity:
            add('HANDLING_BINDING_INVALID','handling/'+proof.id)
    for s in world.shipments.values():
        order = world.orders.get(s.order_id)
        if not order or s.warehouse_id != order.warehouse_id or len(set(s.reservation_ids)) != len(s.reservation_ids):
            add('SHIPMENT_BINDING_INVALID','shipments/'+s.id)
        for rid in s.reservation_ids:
            r = world.reservations.get(rid)
            if not r or r.order_id != s.order_id or r.warehouse_id != s.warehouse_id:
                add('SHIPMENT_ALLOCATION_INVALID','shipments/'+s.id)
            if s.status != 'canceled':
                if rid in allocation_shipments:
                    add('DUPLICATE_SHIPMENT_ALLOCATION','shipments/'+s.id)
                allocation_shipments.add(rid)
            if s.status in {'shipped','delivered'} and r and r.status != 'shipped':
                add('SHIPPING_STAGE_INVALID','shipments/'+s.id)
        if s.status in {'shipped','delivered'} and not attested(world,s.carrier_notice,'carrier',{'shipment_id':s.id,'warehouse_id':s.warehouse_id,'order_id':s.order_id,'reservation_ids':s.reservation_ids,'outcome':'delivered'}):
            add('CARRIER_ATTESTATION_INVALID','shipments/'+s.id)
        if s.status == 'delivered' and s.delivered_at is None:
            add('DELIVERY_TIMESTAMP_MISSING','shipments/'+s.id)
        elif s.status == 'delivered':
            try:
                if s.delivered_at!=datetime.fromisoformat(s.carrier_notice['at']):
                    add('DELIVERY_TIMESTAMP_INVALID','shipments/'+s.id)
            except (KeyError,ValueError,TypeError):
                add('DELIVERY_TIMESTAMP_INVALID','shipments/'+s.id)
    for t in world.transfers.values():
        source = world.lots.get(t.source_lot_id)
        dest = world.warehouses.get(t.target_warehouse_id)
        if not source or not dest or source.warehouse_id != t.source_warehouse_id or source.warehouse_id not in world.warehouses or (source and dest and source.warehouse_id in world.warehouses and world.warehouses[source.warehouse_id].org_id != dest.org_id):
            add('TRANSFER_BINDING_INVALID','transfers/'+t.id)
        if not attested(world,t.receiving_notice,'transfer-carrier',{'transfer_id':t.id,'source_lot_id':t.source_lot_id,'target_warehouse_id':t.target_warehouse_id,'quantity':t.quantity}):
            add('TRANSFER_ATTESTATION_INVALID','transfers/'+t.id)
        if t.status == 'received' and (t.target_lot_id not in world.lots or world.lots[t.target_lot_id].warehouse_id != t.target_warehouse_id or not source or world.lots[t.target_lot_id].sku_id != source.sku_id):
            add('TRANSFER_RECEIPT_INVALID','transfers/'+t.id)
    returns_per_allocation = {}
    for ret in world.returns.values():
        s,r = world.shipments.get(ret.shipment_id),world.reservations.get(ret.reservation_id)
        if not s or not r or r.id not in s.reservation_ids or ret.warehouse_id != s.warehouse_id:
            add('RETURN_BINDING_INVALID','returns/'+ret.id)
        if ret.status != 'canceled':
            returns_per_allocation[ret.reservation_id] = returns_per_allocation.get(ret.reservation_id,0)+ret.quantity
            if not s or s.status != 'delivered':
                add('RETURN_BEFORE_DELIVERY','returns/'+ret.id)
        if r and returns_per_allocation.get(r.id,0) > r.quantity:
            add('RETURN_EXCEEDS_DELIVERY','returns/'+ret.id)
        if ret.status in {'inspected','restocked'} and ret.accepted+ret.scrapped != ret.quantity:
            add('RETURN_INSPECTION_NOT_CONSERVED','returns/'+ret.id)
        if ret.status == 'restocked' and ret.accepted <= 0:
            add('DAMAGED_GOODS_RESTOCKED','returns/'+ret.id)
        binding = {'return_id':ret.id,'warehouse_id':ret.warehouse_id,'reservation_id':ret.reservation_id,'quantity':ret.quantity}
        if not attested(world,ret.receipt_notice,'return-carrier',binding):
            add('RETURN_ATTESTATION_INVALID','returns/'+ret.id)
        if ret.status in {'inspected','restocked'} and r and r.lot_id in world.lots and world.lots[r.lot_id].sku_id in world.skus:
            condition = world.skus[world.lots[r.lot_id].sku_id].simulated_return_condition
            expected={'return_id':ret.id,'accepted':ret.quantity if condition=='resalable' else 0,
                      'scrapped':0 if condition=='resalable' else ret.quantity,
                      'refundable_units':0 if condition=='customer_damage' else ret.quantity}
            if not attested(world,ret.inspection_notice,'inspector',expected) or (ret.accepted,ret.scrapped,ret.refundable_units)!=(expected['accepted'],expected['scrapped'],expected['refundable_units']):
                add('INSPECTION_ATTESTATION_INVALID','returns/'+ret.id)
    keys = set()
    for payment in world.payments.values():
        key = (payment.order_id,payment.kind,payment.key)
        if key in keys:
            add('DUPLICATE_PAYMENT_KEY','payments/'+payment.id)
        keys.add(key)
        order = world.orders.get(payment.order_id)
        if not order or order.warehouse_id != payment.warehouse_id:
            add('PAYMENT_BINDING_INVALID','payments/'+payment.id)
        binding = {'order_id':payment.order_id,'kind':payment.kind,'amount_cents':payment.amount_cents,'key':payment.key,'token':payment.token,'outcome':payment.outcome}
        if not attested(world,payment.gateway_notice,'payment-gateway',binding) or (payment.outcome == 'succeeded') != (payment.token == 'card_approved'):
            add('PAYMENT_ATTESTATION_INVALID','payments/'+payment.id)
    return errors


def decide(world, actor, op):
    deny = lambda code: Decision(allowed=False,error_code=code)
    if invariants(world):
        return deny('INVALID_WORLD')
    if actor not in world.users:
        return deny('UNKNOWN_ACTOR')
    warehouse = world.warehouses.get(op.scope_id)
    if not warehouse:
        return deny('WAREHOUSE_NOT_FOUND')
    obj = target(world,op)
    if obj is None:
        return deny('RESOURCE_NOT_FOUND')
    scope = obj.id if isinstance(obj,type(warehouse)) else obj.target_warehouse_id if op.name=='receive_transfer' else obj.warehouse_id
    if scope != warehouse.id:
        return deny('WAREHOUSE_SCOPE_MISMATCH')
    membership = role(world,actor,warehouse.id)
    own_order = (obj if isinstance(obj,Order) else world.orders.get(obj.order_id) if hasattr(obj,'order_id') else
                 world.orders.get(world.reservations[obj.reservation_id].order_id) if isinstance(obj,Return) else None)
    buyer = bool(own_order and own_order.customer_id == actor)
    if op.is_query:
        if not membership and not buyer and op.name not in {'get_catalog'}:
            return deny('WAREHOUSE_PERMISSION_DENIED')
        if op.name == 'get_inventory' and not membership:
            return deny('WAREHOUSE_PERMISSION_DENIED')
        if op.name == 'get_payment_ledger' and membership not in {'manager','finance'} and not buyer:
            return deny('FINANCIAL_PERMISSION_DENIED')
        from .observations import query_data
        return Decision(allowed=True,expected_state=world.clone(),data=query_data(world,actor,op))
    buyer_ops = {'capture_payment','confirm_order','modify_order_line','cancel_order','cancel_order_line','request_return','cancel_return'}
    receiver_ops = {'receive_inventory','receive_transfer','receive_return','inspect_lot','inspect_return','restock_return'}
    picker_ops = {'reserve_inventory','release_reservation','pick_reservation','unpick_reservation','pack_reservation','unpack_reservation','create_shipment','cancel_shipment','dispatch_shipment','confirm_delivery'}
    if op.name == 'create_order':
        if op.params['customer_id'] != actor and membership != 'manager':
            return deny('CUSTOMER_IDENTITY_MISMATCH')
    elif membership == 'manager':
        pass
    elif op.name in buyer_ops and buyer:
        pass
    elif op.name in receiver_ops and membership == 'receiver':
        pass
    elif op.name in picker_ops and membership == 'picker':
        pass
    elif op.name in {'capture_payment','refund_order'} and membership == 'finance':
        pass
    else:
        return deny('WAREHOUSE_PERMISSION_DENIED')
    new = world.clone()
    name,rid,p = op.name,op.resource_id,op.params
    data = {}
    if name == 'receive_inventory':
        prior = next((r for r in world.receipts.values() if r.notice_id == rid),None)
        if prior:
            if prior.key != p['key'] or prior.lot_id != p['lot_id']:
                return deny('IDEMPOTENCY_CONFLICT')
            return Decision(allowed=True,expected_state=new,data={'receipt_id':prior.id,'lot_id':prior.lot_id})
        if not warehouse.receiving_enabled:
            return deny('RECEIVING_DISABLED')
        sku = world.skus[obj.sku_id]
        if sku.category not in warehouse.allowed_categories:
            return deny('PRODUCT_NOT_ACCEPTED')
        if physical_volume(world,warehouse.id)+obj.quantity*sku.unit_volume > warehouse.capacity_volume:
            return deny('CAPACITY_EXCEEDED')
        old = world.lots.get(p['lot_id'])
        if old and (old.warehouse_id != warehouse.id or old.sku_id != obj.sku_id or old.quality != obj.quality):
            return deny('LOT_CONFLICT')
        if old:
            new.lots[old.id].on_hand += obj.quantity
        else:
            new.lots[p['lot_id']] = Lot(id=p['lot_id'],warehouse_id=warehouse.id,sku_id=obj.sku_id,on_hand=obj.quantity,quality=obj.quality)
        key = 'receipt_'+digest([rid,p['key']])[:16]
        new.receipts[key] = Receipt(id=key,notice_id=rid,warehouse_id=warehouse.id,lot_id=p['lot_id'],quantity=obj.quantity,key=p['key'])
        data = {'receipt_id':key,'lot_id':p['lot_id']}
    elif name == 'inspect_lot':
        if obj.quality == 'available':
            return Decision(allowed=True,expected_state=new,data={'lot_id':rid,'quality':'available'})
        new.lots[rid].quality = 'available'
        new.lot_inspections[rid] = signed_notice(world.domain,'stock-inspector',{'lot_id':rid,'warehouse_id':warehouse.id,'outcome':'passed'},world.clock)
        data = {'lot_id':rid,'quality':'available'}
    elif name == 'dispatch_transfer':
        old = next((t for t in world.transfers.values() if t.source_warehouse_id == warehouse.id and t.key == p['key']),None)
        if old:
            if [old.id,old.source_lot_id,old.target_warehouse_id,old.target_lot_id,old.quantity] != [p['transfer_id'],rid,p['target_warehouse_id'],p['target_lot_id'],p['quantity']]:
                return deny('IDEMPOTENCY_CONFLICT')
            return Decision(allowed=True,expected_state=new,data={'transfer_id':old.id,'status':old.status})
        dest = world.warehouses.get(p['target_warehouse_id'])
        if not dest or dest.id == warehouse.id or dest.org_id != warehouse.org_id:
            return deny('TRANSFER_DESTINATION_INVALID')
        if role(world,actor,dest.id) not in {'manager','receiver'}:
            return deny('DESTINATION_PERMISSION_DENIED')
        if obj.quality != 'available' or obj.on_hand-obj.reserved < p['quantity']:
            return deny('STOCK_INSUFFICIENT')
        if p['transfer_id'] in world.transfers:
            return deny('TRANSFER_ID_CONFLICT')
        new.lots[rid].on_hand -= p['quantity']
        binding = {'transfer_id':p['transfer_id'],'source_lot_id':rid,'target_warehouse_id':dest.id,'quantity':p['quantity']}
        t = Transfer(id=p['transfer_id'],source_warehouse_id=warehouse.id,target_warehouse_id=dest.id,source_lot_id=rid,
                     target_lot_id=p['target_lot_id'],quantity=p['quantity'],key=p['key'],receiving_notice=signed_notice(world.domain,'transfer-carrier',binding,world.clock))
        new.transfers[t.id] = t
        data = {'transfer_id':t.id,'status':t.status}
    elif name == 'receive_transfer':
        if obj.status == 'received':
            return Decision(allowed=True,expected_state=new,data={'transfer_id':rid,'status':'received','lot_id':obj.target_lot_id})
        sku = world.skus[world.lots[obj.source_lot_id].sku_id]
        if not warehouse.receiving_enabled:
            return deny('RECEIVING_DISABLED')
        if sku.category not in warehouse.allowed_categories:
            return deny('PRODUCT_NOT_ACCEPTED')
        if physical_volume(world,warehouse.id)+obj.quantity*sku.unit_volume > warehouse.capacity_volume:
            return deny('CAPACITY_EXCEEDED')
        lot = world.lots.get(obj.target_lot_id)
        if lot and (lot.warehouse_id != warehouse.id or lot.sku_id != sku.id or lot.quality != 'available'):
            return deny('LOT_CONFLICT')
        if lot:
            new.lots[lot.id].on_hand += obj.quantity
        else:
            new.lots[obj.target_lot_id] = Lot(id=obj.target_lot_id,warehouse_id=warehouse.id,sku_id=sku.id,on_hand=obj.quantity)
        new.transfers[rid].status = 'received'
        data = {'transfer_id':rid,'status':'received','lot_id':obj.target_lot_id}
    elif name == 'create_order':
        if p['customer_id'] not in world.users:
            return deny('UNKNOWN_CUSTOMER')
        if any(l['sku_id'] not in world.skus or world.skus[l['sku_id']].org_id != warehouse.org_id for l in p['lines']):
            return deny('PRODUCT_SCOPE_MISMATCH')
        order = Order(id=p['order_id'],warehouse_id=warehouse.id,customer_id=p['customer_id'],shipping_cents=warehouse.shipping_cents,
                      lines=[OrderLine(**l,unit_price_cents=world.skus[l['sku_id']].unit_price_cents) for l in p['lines']])
        if order.id in world.orders and world.orders[order.id] != order:
            return deny('ORDER_ID_CONFLICT')
        new.orders[order.id] = order
        data = {'order_id':order.id,'status':order.status}
    elif name == 'modify_order_line':
        line = get_line(obj,p['line_id'])
        if not line:
            return deny('ORDER_LINE_NOT_FOUND')
        if obj.status != 'draft' or any(pay.order_id == rid for pay in world.payments.values()):
            return deny('ORDER_ALREADY_COMMITTED')
        get_line(new.orders[rid],line.id).quantity = p['quantity']
        data = {'order_id':rid,'line_id':line.id,'quantity':p['quantity']}
    elif name in {'capture_payment','refund_order'}:
        kind = 'charge' if name=='capture_payment' else 'refund'
        token = p.get('token','card_approved')
        prior = next((pay for pay in world.payments.values() if pay.order_id==rid and pay.kind==kind and pay.key==p['key']),None)
        if prior:
            if prior.amount_cents != p['amount_cents'] or prior.token != token:
                return deny('IDEMPOTENCY_CONFLICT')
            return Decision(allowed=True,expected_state=new,data={'payment_id':prior.id,'outcome':prior.outcome})
        if kind=='charge' and obj.status!='draft':
            return deny('ORDER_PAYMENT_CLOSED')
        if kind=='charge' and p['amount_cents'] > price(obj)-paid(world,rid):
            return deny('PAYMENT_EXCEEDS_DUE')
        if kind=='refund' and p['amount_cents'] > refund_limit(world,obj):
            return deny('REFUND_EXCEEDS_ELIGIBLE')
        outcome = 'succeeded' if token=='card_approved' else 'failed'
        binding = {'order_id':rid,'kind':kind,'amount_cents':p['amount_cents'],'key':p['key'],'token':token,'outcome':outcome}
        key = 'payment_'+digest([rid,kind,p['key']])[:16]
        new.payments[key] = Payment(id=key,warehouse_id=warehouse.id,order_id=rid,kind=kind,amount_cents=p['amount_cents'],
            key=p['key'],token=token,outcome=outcome,gateway_notice=signed_notice(world.domain,'payment-gateway',binding,world.clock))
        data = {'payment_id':key,'outcome':outcome}
    elif name == 'confirm_order':
        if obj.status == 'canceled':
            return deny('ORDER_CANCELED')
        if obj.status == 'draft' and paid(world,rid)-paid(world,rid,'refund') < price(obj):
            return deny('PAYMENT_REQUIRED')
        if obj.status == 'draft':
            new.orders[rid].status = 'confirmed'
        data = {'order_id':rid,'status':new.orders[rid].status}
    elif name == 'reserve_inventory':
        lot = world.lots.get(p['lot_id']); line = get_line(obj,p['line_id'])
        if not lot or not line:
            return deny('ALLOCATION_REFERENCE_MISSING')
        reservation = Reservation(id=p['reservation_id'],warehouse_id=warehouse.id,order_id=rid,line_id=line.id,lot_id=lot.id,quantity=p['quantity'])
        old = world.reservations.get(reservation.id)
        if old:
            if old != reservation:
                return deny('RESERVATION_ID_CONFLICT')
            return Decision(allowed=True,expected_state=new,data={'reservation_id':old.id,'status':old.status})
        if obj.status not in {'confirmed','partially_shipped','partially_delivered'}:
            return deny('ORDER_NOT_CONFIRMED')
        if lot.warehouse_id != warehouse.id or lot.sku_id != line.sku_id:
            return deny('ALLOCATION_SCOPE_MISMATCH')
        if allocated(world,rid,line.id)+line.canceled+p['quantity'] > line.quantity:
            return deny('ORDER_OVERALLOCATED')
        if lot.quality != 'available':
            return deny('QUALITY_HOLD')
        if lot.on_hand-lot.reserved < p['quantity']:
            return deny('STOCK_INSUFFICIENT')
        new.lots[lot.id].reserved += p['quantity']
        new.reservations[reservation.id] = reservation
        data = {'reservation_id':reservation.id,'status':'reserved'}
    elif name in {'release_reservation','pick_reservation','unpick_reservation','pack_reservation','unpack_reservation'}:
        linked = any(rid in s.reservation_ids and s.status!='canceled' for s in world.shipments.values())
        required = {'release_reservation':'reserved','pick_reservation':'reserved','unpick_reservation':'picked','pack_reservation':'picked','unpack_reservation':'packed'}[name]
        end = {'release_reservation':'released','pick_reservation':'picked','unpick_reservation':'reserved','pack_reservation':'packed','unpack_reservation':'picked'}[name]
        if obj.status == end:
            return Decision(allowed=True,expected_state=new,data={'reservation_id':rid,'status':end})
        if obj.status != required:
            return deny('RESERVATION_STAGE_INVALID')
        if name in {'release_reservation','unpick_reservation','unpack_reservation'} and linked:
            return deny('SHIPMENT_STILL_ASSIGNED')
        lot = new.lots[obj.lot_id]
        if name=='release_reservation':
            lot.reserved -= obj.quantity
            new.reservations[rid].released_quantity += obj.quantity
        elif name=='pick_reservation':
            lot.reserved -= obj.quantity; lot.on_hand -= obj.quantity; lot.picked += obj.quantity
            new.handling['pick_'+rid] = Handling(id='pick_'+rid,reservation_id=rid,warehouse_id=warehouse.id,kind='pick',quantity=obj.quantity)
        elif name=='unpick_reservation':
            lot.picked -= obj.quantity; lot.on_hand += obj.quantity; lot.reserved += obj.quantity
            new.handling.pop('pick_'+rid,None)
        elif name=='pack_reservation':
            new.handling['pack_'+rid] = Handling(id='pack_'+rid,reservation_id=rid,warehouse_id=warehouse.id,kind='pack',quantity=obj.quantity)
        else:
            new.handling.pop('pack_'+rid,None)
        new.reservations[rid].status = end
        data = {'reservation_id':rid,'status':end}
    elif name == 'create_shipment':
        old = next((s for s in world.shipments.values() if s.order_id==rid and s.key==p['key']),None)
        if old:
            if old.id != p['shipment_id'] or old.reservation_ids != p['reservation_ids']:
                return deny('IDEMPOTENCY_CONFLICT')
            return Decision(allowed=True,expected_state=new,data={'shipment_id':old.id,'status':old.status})
        if p['shipment_id'] in world.shipments:
            return deny('SHIPMENT_ID_CONFLICT')
        for allocation in p['reservation_ids']:
            r = world.reservations.get(allocation)
            if not r or r.order_id != rid or r.warehouse_id != warehouse.id:
                return deny('SHIPMENT_ALLOCATION_INVALID')
            if r.status not in {'reserved','picked','packed'} or any(allocation in s.reservation_ids and s.status!='canceled' for s in world.shipments.values()):
                return deny('SHIPMENT_ALLOCATION_UNAVAILABLE')
        new.shipments[p['shipment_id']] = Shipment(id=p['shipment_id'],warehouse_id=warehouse.id,order_id=rid,reservation_ids=p['reservation_ids'],key=p['key'])
        data = {'shipment_id':p['shipment_id'],'status':'label_created'}
    elif name == 'cancel_shipment':
        if obj.status not in {'label_created','canceled'}:
            return deny('SHIPMENT_ALREADY_DISPATCHED')
        new.shipments[rid].status = 'canceled'
        data = {'shipment_id':rid,'status':'canceled'}
    elif name == 'dispatch_shipment':
        if obj.status in {'shipped','delivered'}:
            return Decision(allowed=True,expected_state=new,data={'shipment_id':rid,'status':obj.status})
        if obj.status != 'label_created':
            return deny('SHIPMENT_CANCELED')
        if any(world.reservations[r].status != 'packed' for r in obj.reservation_ids):
            return deny('SHIPMENT_NOT_PACKED')
        for allocation in obj.reservation_ids:
            r = new.reservations[allocation]
            new.lots[r.lot_id].picked -= r.quantity; r.status = 'shipped'
        new.shipments[rid].status = 'shipped'
        binding = {'shipment_id':rid,'warehouse_id':warehouse.id,'order_id':obj.order_id,'reservation_ids':obj.reservation_ids,'outcome':'delivered'}
        new.shipments[rid].carrier_notice = signed_notice(world.domain,'carrier',binding,world.clock)
        order = new.orders[obj.order_id]
        total = sum(l.quantity-l.canceled for l in order.lines)
        order.status = 'shipped' if shipped(new,order.id)==total else 'partially_shipped'
        data = {'shipment_id':rid,'status':'shipped'}
    elif name == 'confirm_delivery':
        if obj.status == 'delivered':
            return Decision(allowed=True,expected_state=new,data={'shipment_id':rid,'status':'delivered'})
        binding = {'shipment_id':rid,'warehouse_id':warehouse.id,'order_id':obj.order_id,'reservation_ids':obj.reservation_ids,'outcome':'delivered'}
        if obj.status != 'shipped' or not attested(world,obj.carrier_notice,'carrier',binding):
            return deny('EXTERNAL_ATTESTATION_REQUIRED')
        new.shipments[rid].status = 'delivered'; new.shipments[rid].delivered_at = world.clock
        order = new.orders[obj.order_id]
        order.status = 'delivered' if shipped(new,order.id,delivered=True)==sum(l.quantity-l.canceled for l in order.lines) else 'partially_delivered'
        data = {'shipment_id':rid,'status':'delivered'}
    elif name in {'cancel_order','cancel_order_line'}:
        if name=='cancel_order' and shipped(world,rid):
            return deny('ORDER_ALREADY_SHIPPED')
        lines = obj.lines if name=='cancel_order' else [get_line(obj,p['line_id'])]
        if not all(lines):
            return deny('ORDER_LINE_NOT_FOUND')
        if name=='cancel_order' and any(r.order_id==rid and r.status in {'picked','packed'} for r in world.reservations.values()):
            return deny('PICKED_CANCEL_REQUIRES_REVERSAL')
        if name=='cancel_order':
            for shipment in new.shipments.values():
                if shipment.order_id==rid and shipment.status=='label_created':shipment.status='canceled'
        for line in lines:
            quantity = line.quantity-line.canceled if name=='cancel_order' else p['quantity']
            if quantity+shipped(world,rid,line.id)+line.canceled > line.quantity:
                return deny('ORDER_REMAINDER_EXCEEDED')
            unallocated = line.quantity-line.canceled-allocated(new,rid,line.id)
            release_needed = max(0,quantity-unallocated)
            candidates = sorted((r for r in new.reservations.values() if r.order_id==rid and r.line_id==line.id and r.status=='reserved'
                                 and not any(r.id in s.reservation_ids and s.status!='canceled' for s in new.shipments.values())),key=lambda r:r.id)
            if sum(r.quantity for r in candidates) < release_needed:
                return deny('ORDER_REMAINDER_COMMITTED')
            for reservation in candidates:
                released = min(release_needed,reservation.quantity)
                if not released:break
                new.lots[reservation.lot_id].reserved -= released
                reservation.released_quantity += released
                if released==reservation.quantity:reservation.status='released'
                else:reservation.quantity -= released
                release_needed -= released
            get_line(new.orders[rid],line.id).canceled += quantity
        order = new.orders[rid]
        if all(l.canceled==l.quantity for l in order.lines):
            order.status = 'canceled'
        elif shipped(new,rid)==sum(l.quantity-l.canceled for l in order.lines):
            order.status = 'delivered' if shipped(new,rid,delivered=True)==shipped(new,rid) else 'shipped'
        data = {'order_id':rid,'status':order.status}
    elif name == 'request_return':
        r = world.reservations.get(p['reservation_id'])
        if not r or r.id not in obj.reservation_ids:
            return deny('RETURN_ALLOCATION_INVALID')
        ret = Return(id=p['return_id'],warehouse_id=warehouse.id,shipment_id=rid,reservation_id=r.id,quantity=p['quantity'],reason=p['reason'],
            receipt_notice=signed_notice(world.domain,'return-carrier',{'return_id':p['return_id'],'warehouse_id':warehouse.id,'reservation_id':r.id,'quantity':p['quantity']},world.clock))
        old = world.returns.get(ret.id)
        if old:
            if old.model_dump(exclude={'status','accepted','scrapped','refundable_units','inspection_notice'}) != ret.model_dump(exclude={'status','accepted','scrapped','refundable_units','inspection_notice'}):
                return deny('RETURN_ID_CONFLICT')
            return Decision(allowed=True,expected_state=new,data={'return_id':old.id,'status':old.status})
        sku = world.skus[world.lots[r.lot_id].sku_id]
        if obj.status != 'delivered' or not obj.delivered_at:
            return deny('DELIVERY_REQUIRED')
        if not sku.returnable:
            return deny('PRODUCT_NOT_RETURNABLE')
        if world.clock > obj.delivered_at+timedelta(days=sku.return_window_days):
            return deny('RETURN_WINDOW_EXPIRED')
        if sum(t.quantity for t in world.returns.values() if t.reservation_id==r.id and t.status!='canceled')+p['quantity'] > r.quantity:
            return deny('RETURN_EXCEEDS_DELIVERY')
        new.returns[ret.id] = ret
        data = {'return_id':ret.id,'status':'requested'}
    elif name == 'cancel_return':
        if obj.status not in {'requested','canceled'}:
            return deny('RETURN_ALREADY_RECEIVED')
        new.returns[rid].status = 'canceled'; data={'return_id':rid,'status':'canceled'}
    elif name == 'receive_return':
        if obj.status in {'received','inspected','restocked'}:
            return Decision(allowed=True,expected_state=new,data={'return_id':rid,'status':obj.status})
        if obj.status != 'requested':
            return deny('RETURN_STAGE_INVALID')
        if not warehouse.receiving_enabled:
            return deny('RECEIVING_DISABLED')
        sku = world.skus[world.lots[world.reservations[obj.reservation_id].lot_id].sku_id]
        if physical_volume(world,warehouse.id)+obj.quantity*sku.unit_volume > warehouse.capacity_volume:
            return deny('CAPACITY_EXCEEDED')
        new.returns[rid].status='received'; data={'return_id':rid,'status':'received'}
    elif name == 'inspect_return':
        if obj.status in {'inspected','restocked'}:
            return Decision(allowed=True,expected_state=new,data={'return_id':rid,'accepted':obj.accepted,'scrapped':obj.scrapped,'refundable_units':obj.refundable_units})
        if obj.status != 'received':
            return deny('RETURN_RECEIPT_REQUIRED')
        ret = new.returns[rid];ret.status='inspected'
        condition = world.skus[world.lots[world.reservations[obj.reservation_id].lot_id].sku_id].simulated_return_condition
        ret.accepted = obj.quantity if condition=='resalable' else 0
        ret.scrapped = obj.quantity-ret.accepted
        ret.refundable_units = 0 if condition=='customer_damage' else obj.quantity
        ret.inspection_notice = signed_notice(world.domain,'inspector',{'return_id':rid,'accepted':ret.accepted,'scrapped':ret.scrapped,'refundable_units':ret.refundable_units},world.clock)
        data={'return_id':rid,'accepted':ret.accepted,'scrapped':ret.scrapped,'refundable_units':ret.refundable_units}
    elif name == 'restock_return':
        if obj.status == 'restocked':
            return Decision(allowed=True,expected_state=new,data={'return_id':rid,'status':'restocked'})
        if obj.status != 'inspected':
            return deny('INSPECTION_REQUIRED')
        if obj.accepted <= 0:
            return deny('NO_RESTOCKABLE_GOODS')
        lot = new.lots[world.reservations[obj.reservation_id].lot_id]
        if physical_volume(world,warehouse.id) > warehouse.capacity_volume:
            return deny('CAPACITY_EXCEEDED')
        lot.on_hand += obj.accepted;new.returns[rid].status='restocked'
        data={'return_id':rid,'status':'restocked'}
    elif name == 'change_warehouse_capacity':
        if p['capacity_volume'] < physical_volume(world,warehouse.id):
            return deny('CAPACITY_BELOW_USAGE')
        new.warehouses[rid].capacity_volume=p['capacity_volume'];data=p.copy()
    elif name == 'set_warehouse_receiving':
        new.warehouses[rid].receiving_enabled=p['enabled'];data={'receiving_enabled':p['enabled']}
    return Decision(allowed=True,expected_state=new,data=data)
