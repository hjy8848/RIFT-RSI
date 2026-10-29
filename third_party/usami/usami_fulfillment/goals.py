"""Inventory-specific atomic compositions and explicit field effects."""
from usami_harness.goals import AtomicGoal, JointGoal, Task, Witness, freeze_task, execute
from usami_harness.models import digest
from .models import Operation, Order, OrderLine, Access
from .generator import generate_world
from .rules import price, physical_volume

GOAL_FIELDS = {
    'warehouses': ['capacity_volume','receiving_enabled'],
    'lots': ['on_hand','reserved','picked','quality'],
    'orders': ['status','lines'], 'reservations': ['status','quantity','released_quantity','lot_id'],
    'shipments': ['status','reservation_ids'], 'transfers': ['status','quantity'],
    'returns': ['status','accepted','scrapped','refundable_units'],
    'payments': ['outcome','amount_cents','kind'], 'handling': ['kind','quantity'],
    'receipts': ['quantity'], 'lot_inspections': ['binding'],
}
FRAME_FIELDS = {
    'warehouses': ['capacity_volume','receiving_enabled'],
    'lots': ['on_hand','reserved','picked','quality'],
    'orders': ['status','lines'], 'reservations': ['status','quantity','released_quantity'],
    'shipments': ['status','carrier_notice','delivered_at'], 'transfers': ['status'],
    'returns': ['status','accepted','scrapped','refundable_units','inspection_notice'],
    'payments': [], 'handling': [], 'receipts': [], 'lot_inspections': [],
}
DELETABLE = frozenset({'handling'})

# Coarse business workflow categories. Different rejection codes of one workflow
# are not separate families, and progress phases never rename their family.
FAMILY = {
    'warehouse_scope_refusal':'warehouse_authorization','staff_role_refusal':'warehouse_authorization',
    'stock_refusal':'inventory_allocation','packing_refusal':'shipment_dispatch',
    'shipped_cancel_refusal':'order_cancellation','return_quantity_refusal':'return_eligibility',
    'return_window_refusal':'return_eligibility','nonreturnable_refusal':'return_eligibility',
    'undelivered_return_refusal':'return_eligibility','damaged_refund_refusal':'return_refund',
    'unearned_refund_refusal':'refund_eligibility','payment_key_refusal':'payment_integrity',
    'closed_charge_refusal':'payment_integrity','committed_edit_refusal':'draft_amendment',
    'quality_refusal':'quality_release','transfer_scope_refusal':'regional_transfer',
    'transfer_access_refusal':'regional_transfer','receiving_refusal':'inbound_admission',
    'capacity_refusal':'warehouse_capacity','product_admission_refusal':'inbound_admission',
    'capacity_reduction_refusal':'warehouse_capacity','picked_release_refusal':'allocation_release',
    'receiving_recovery':'inbound_admission','capacity_recovery':'warehouse_capacity',
    'inbound_recovery':'inbound_admission',
}


class Builder:
    def __init__(self, seed, family, identity='u0'):
        self.world, self.family, self.identity = generate_world(seed), family, identity
        self.ops, self.atoms, self.frame = [], [], {}

    def op(self, name, resource='order0', scope='w0', **params):
        return Operation(name=name, scope_id=scope, resource_id=resource, params=params)

    def prepare(self, operations):
        from . import adapter
        self.world = adapter.World.model_validate(execute(adapter, self.world, 'u0', operations).final_state)

    def allow(self, collection, entity, *fields):
        key = collection + '/' + entity
        if entity not in getattr(self.world, collection):
            fields = ('*',)
        if self.frame.get(key) == ['*']:
            return
        self.frame[key] = sorted(set(self.frame.get(key, []) + list(fields)))

    def terminal(self, collection, entity, field, value, comparison='eq'):
        self.atoms.append(AtomicGoal(collection=collection, entity=entity, field=field, value=value, comparison=comparison))

    def add(self, *operations):
        self.ops.extend(operations)

    def paid(self, order='order0', scope='w0', key='pay', amount=None):
        return self.op('capture_payment', order, scope, amount_cents=amount or price(self.world.orders[order]), key=key, token='card_approved')

    def confirmed(self, order='order0', scope='w0'):
        self.prepare([self.paid(order,scope), self.op('confirm_order',order,scope)])

    def allocated(self, order='order0', scope='w0', reservation='r0', lot='lot0', quantity=4):
        self.confirmed(order,scope)
        self.prepare([self.op('reserve_inventory',order,scope,reservation_id=reservation,line_id='line0',lot_id=lot,quantity=quantity)])

    def shipped(self, delivered=False, sku='sku0'):
        self.world.orders['order0'].lines[0].sku_id = sku
        self.world.orders['order0'].lines[0].unit_price_cents = self.world.skus[sku].unit_price_cents
        self.allocated(lot='lot2' if sku=='sku1' else 'lot0')
        ops=[self.op('pick_reservation','r0'),self.op('pack_reservation','r0'),
             self.op('create_shipment',shipment_id='s0',reservation_ids=['r0'],key='ship'),self.op('dispatch_shipment','s0')]
        if delivered:
            ops.append(self.op('confirm_delivery','s0'))
        self.prepare(ops)

    def flow(self, order='order0', scope='w0', reservation='r0', lot='lot0', quantity=4,
             shipment='s0', payment=True, delivery=False, alternatives=False, line='line0'):
        if payment:
            self.add(self.paid(order,scope),self.op('confirm_order',order,scope))
        self.add(self.op('reserve_inventory',order,scope,reservation_id=reservation,line_id=line,lot_id=lot,quantity=quantity))
        if alternatives:
            self.add(self.op('reserve_inventory',order,scope,reservation_id=reservation,line_id=line,lot_id='lot1',quantity=quantity))
        self.add(self.op('pick_reservation',reservation,scope),self.op('pack_reservation',reservation,scope),
                 self.op('create_shipment',order,scope,shipment_id=shipment,reservation_ids=[reservation],key=shipment),
                 self.op('dispatch_shipment',shipment,scope))
        if delivery:
            self.add(self.op('confirm_delivery',shipment,scope))
        self.terminal('shipments',shipment,'status','delivered' if delivery else 'shipped')

    def effects(self):
        # A static, operation-specific effect declaration. No reference terminal
        # snapshot is used to infer permissible changes.
        reservations={r.id:r.model_dump() for r in self.world.reservations.values()}
        orders={o.id:o for o in self.world.orders.values()}
        shipments={s.id:s.model_dump() for s in self.world.shipments.values()}
        transfers={t.id:t.model_dump() for t in self.world.transfers.values()}
        returns={r.id:r.model_dump() for r in self.world.returns.values()}
        reservation_lots={r.id:{r.lot_id} for r in self.world.reservations.values()}
        for op in self.ops:
            p,rid,name=op.params,op.resource_id,op.name
            if name=='reserve_inventory':
                reservations[p['reservation_id']]={'order_id':rid,'lot_id':p['lot_id']}
                reservation_lots.setdefault(p['reservation_id'],set()).add(p['lot_id'])
            elif name=='create_shipment':
                shipments[p['shipment_id']]={'order_id':rid,'reservation_ids':p['reservation_ids']}
            elif name=='dispatch_transfer':
                transfers[p['transfer_id']]={'source_lot_id':rid,'target_lot_id':p['target_lot_id']}
            elif name=='request_return':
                returns[p['return_id']]={'reservation_id':p['reservation_id']}
        for op in self.ops:
            p,rid,name=op.params,op.resource_id,op.name
            if name in {'change_warehouse_capacity','set_warehouse_receiving'}:
                self.allow('warehouses',rid,'capacity_volume' if name=='change_warehouse_capacity' else 'receiving_enabled')
            elif name=='receive_inventory':
                self.allow('lots',p['lot_id'],'on_hand')
                self.allow('receipts','receipt_'+digest([rid,p['key']])[:16])
            elif name=='inspect_lot':
                self.allow('lots',rid,'quality');self.allow('lot_inspections',rid)
            elif name in {'dispatch_transfer','receive_transfer'}:
                t=transfers[p['transfer_id'] if name=='dispatch_transfer' else rid]
                self.allow('transfers',p['transfer_id'] if name=='dispatch_transfer' else rid,'status')
                self.allow('lots',t['source_lot_id'] if name=='dispatch_transfer' else t['target_lot_id'],'on_hand')
            elif name=='create_order':
                self.allow('orders',p['order_id'])
            elif name in {'capture_payment','refund_order'}:
                kind='charge' if name=='capture_payment' else 'refund'
                self.allow('payments','payment_'+digest([rid,kind,p['key']])[:16])
            elif name in {'confirm_order','modify_order_line'}:
                self.allow('orders',rid,'status' if name=='confirm_order' else 'lines')
            elif name in {'reserve_inventory','release_reservation','pick_reservation','unpick_reservation','pack_reservation','unpack_reservation'}:
                rr=p['reservation_id'] if name=='reserve_inventory' else rid
                r=reservations[rr]
                self.allow('reservations',rr,'status','released_quantity')
                for lot_id in reservation_lots[rr]:
                    self.allow('lots',lot_id,'on_hand','reserved','picked')
                # Every admissible alternative lot is declared as a possible physical effect.
                if name=='reserve_inventory':
                    self.allow('lots',p['lot_id'],'reserved')
                if 'pick' in name or 'pack' in name:
                    kind='pack' if 'pack' in name else 'pick'
                    self.allow('handling',kind+'_'+rr,'*')
            elif name in {'create_shipment','dispatch_shipment','confirm_delivery','cancel_shipment'}:
                sid=p['shipment_id'] if name=='create_shipment' else rid
                s=shipments[sid]
                self.allow('shipments',sid,'status','carrier_notice','delivered_at')
                if name in {'dispatch_shipment','confirm_delivery'}:
                    self.allow('orders',s['order_id'],'status')
                if name=='dispatch_shipment':
                    for rr in s['reservation_ids']:
                        self.allow('reservations',rr,'status')
                        for lot_id in reservation_lots[rr]:
                            self.allow('lots',lot_id,'picked','on_hand','reserved')
            elif name in {'cancel_order','cancel_order_line'}:
                self.allow('orders',rid,'status','lines')
                for rr,r in reservations.items():
                    if r['order_id']==rid:
                        self.allow('reservations',rr,'status','quantity','released_quantity');self.allow('lots',r['lot_id'],'reserved')
                for sid,s in shipments.items():
                    if s['order_id']==rid:
                        self.allow('shipments',sid,'status')
            elif name in {'request_return','cancel_return','receive_return','inspect_return','restock_return'}:
                retid=p['return_id'] if name=='request_return' else rid
                self.allow('returns',retid,'status','accepted','scrapped','refundable_units','inspection_notice')
                if name=='restock_return':
                    self.allow('lots',reservations[returns[retid]['reservation_id']]['lot_id'],'on_hand')

    def freeze(self, message, refusal=None):
        from . import adapter
        if refusal:
            self.atoms=[AtomicGoal(collection='orders',entity='order0',field='status',value=self.world.orders['order0'].status)]
        else:
            self.effects()
        task=Task(domain=adapter.name,version=adapter.version,family=FAMILY.get(self.family,self.family),identity=self.identity,
            kind='policy_refusal' if refusal else 'completion',message=message,initial_state_hash=self.world.state_hash,
            goal=JointGoal(atoms=self.atoms,allowed_changes=self.frame),
            capabilities=[op.model_dump(mode='json') for op in self.ops],refusal_code=refusal)
        return self.world,freeze_task(adapter,self.world,task)


def make_task(seed, scenario):
    """Independent business compositions; IDs and amounts are not structure labels."""
    if '@' in scenario:
        base,route,phase=scenario.rsplit('@',2)
        world,task=make_task(seed,base)
        return progress_task(world,task,int(route),int(phase))
    b=Builder(seed,scenario.split(':')[0]);variant=scenario.split(':')[1] if ':' in scenario else ''
    op=b.op
    if b.family=='paid_fulfillment':
        if variant=='picker':
            b.confirmed();b.identity='u1';b.flow(payment=False,alternatives=True)
        elif variant=='delivered':
            b.flow(alternatives=True,delivery=True)
        elif variant=='created':
            b.add(op('create_order','w0',order_id='new',customer_id='u3',lines=[{'id':'line0','sku_id':'sku0','quantity':4}]))
            # Explicit price for the not-yet-existing order.
            b.add(op('capture_payment','new',amount_cents=5300,key='pay',token='card_approved'),op('confirm_order','new'))
            b.flow('new',payment=False,alternatives=True)
        else:
            b.flow(alternatives=True)
        message='Fulfill '+('new (create for u3, four sku0 units)' if variant=='created' else 'order0')+'. Use r0 and s0; pay when needed, use either lot0 or lot1, and '+('confirm delivery' if variant=='delivered' else 'dispatch the shipment')+'. Shipping/payment reference is s0/pay.'
    elif b.family=='supply_replenishment':
        b.world.orders['order0'].lines[0].quantity=20
        b.add(op('receive_inventory','inbound0',lot_id='new_stock',key='receipt'))
        b.add(b.paid(),op('confirm_order'))
        for rid,lot,q in [('r0','lot0',8),('r1','lot1',8),('r2','new_stock',4)]:
            b.add(op('reserve_inventory',reservation_id=rid,line_id='line0',lot_id=lot,quantity=q),op('pick_reservation',rid),op('pack_reservation',rid))
        b.add(op('create_shipment',shipment_id='s0',reservation_ids=['r0','r1','r2'],key='s0'),op('dispatch_shipment','s0'))
        b.terminal('shipments','s0','status','shipped')
        message='Replenish using inbound0 into new_stock with receipt key receipt, then fulfill all 20 units of order0. Use allocations r0/r1/r2 (8 lot0, 8 lot1, 4 new_stock) and shipment s0, payment key pay.'
    elif b.family=='quality_release':
        b.world.orders['order0'].lines[0].sku_id='sku1';b.world.orders['order0'].lines[0].unit_price_cents=2300
        b.add(op('receive_inventory','inbound2',lot_id='new_stock',key='receipt'),op('inspect_lot','new_stock'))
        b.flow(lot='new_stock')
        message='Receive quarantined inbound2 into new_stock using receipt, obtain independent inspection, then fulfill order0 using r0 and s0/payment pay. Do not use existing fragile stock.'
    elif b.family in {'reservation_reallocation','cancel_reallocation','label_reallocation'}:
        b.allocated(quantity=4)
        b.world.orders['order1']=Order(id='order1',warehouse_id='w0',customer_id='u3',lines=[OrderLine(id='line0',sku_id='sku0',quantity=8,unit_price_cents=1200)],shipping_cents=500)
        b.confirmed('order1')
        if b.family=='label_reallocation':
            b.prepare([op('create_shipment',shipment_id='s0',reservation_ids=['r0'],key='ship')])
            b.add(op('cancel_shipment','s0'))
        b.add(op('cancel_order') if b.family=='cancel_reallocation' else op('release_reservation','r0'))
        b.flow('order1',reservation='r1',lot='lot0',quantity=8,shipment='s1',payment=False)
        b.terminal('orders','order0','status','canceled') if b.family=='cancel_reallocation' else b.terminal('reservations','r0','status','released')
        message='Free order0 allocation r0 (void its s0 label if present), '+('cancel order0' if b.family=='cancel_reallocation' else 'release r0')+', then fulfill order1 from lot0, eight units, using r1 and s1/key s1. Do not draw from lot1.'
    elif b.family=='regional_transfer':
        b.world.orders['order1'].lines[0].quantity=8
        if variant=='closed':
            b.world.warehouses['w1'].receiving_enabled=False;b.add(op('set_warehouse_receiving','w1','w1',enabled=True))
        if variant=='capacity':
            b.world.warehouses['w1'].capacity_volume=4;b.add(op('change_warehouse_capacity','w1','w1',capacity_volume=12))
        b.add(op('dispatch_transfer','lot0',transfer_id='t0',target_warehouse_id='w1',target_lot_id='arrival',quantity=8,key='move'),op('receive_transfer','t0','w1'))
        b.flow('order1','w1',lot='arrival',quantity=8)
        b.terminal('transfers','t0','status','received')
        message='Transfer eight sku0 units from lot0 to w1 lot arrival (t0/move); restore receiving/capacity to 12 if required. Receive them and fulfill order1 using arrival, r0 and s0/key s0; payment pay.'
    elif b.family=='packed_cancellation':
        b.allocated();b.prepare([op('pick_reservation','r0'),op('pack_reservation','r0'),op('create_shipment',shipment_id='s0',reservation_ids=['r0'],key='ship')])
        b.add(op('cancel_shipment','s0'),op('unpack_reservation','r0'),op('unpick_reservation','r0'),op('cancel_order'))
        if variant=='refund':
            b.add(op('refund_order',amount_cents=5300,key='cancel-refund'))
            b.terminal('payments','payment_'+digest(['order0','refund','cancel-refund'])[:16],'outcome','succeeded')
        b.terminal('orders','order0','status','canceled')
        message='Cancel packed unshipped order0 safely, voiding s0, reversing r0 handling and freeing stock.'+(' Refund 5300 with key cancel-refund.' if variant=='refund' else '')
    elif b.family in {'return_restock','return_refund','return_replacement','return_withdrawal'}:
        b.shipped(delivered=True,sku='sku1' if variant=='merchant_fault' else 'sku0')
        b.add(op('request_return','s0',return_id='ret0',reservation_id='r0',quantity=2,reason='customer_remorse'))
        if variant=='receiving':
            b.world.warehouses['w0'].receiving_enabled=False
            b.add(op('set_warehouse_receiving','w0',enabled=True))
        if variant=='capacity':
            b.world.warehouses['w0'].capacity_volume=physical_volume(b.world,'w0')
            b.add(op('change_warehouse_capacity','w0',capacity_volume=physical_volume(b.world,'w0')+2))
        if b.family=='return_withdrawal':
            b.add(op('cancel_return','ret0'));b.terminal('returns','ret0','status','canceled')
            message='Create a return ret0 for two delivered r0 units from s0, customer_remorse, then withdraw the request before goods are received.'
        else:
            b.add(op('receive_return','ret0'),op('inspect_return','ret0'))
            if b.family!='return_refund':
                b.add(op('restock_return','ret0'));b.terminal('returns','ret0','status','restocked')
            if b.family in {'return_refund','return_replacement'} or variant=='refund':
                amount=4600 if variant=='merchant_fault' else 2400
                b.add(op('refund_order',amount_cents=amount,key='returned'))
                b.terminal('payments','payment_'+digest(['order0','refund','returned'])[:16],'outcome','succeeded')
            if b.family=='return_replacement':
                # Consume ALL remaining lot0 stock through an independently legitimate transfer.
                b.prepare([op('dispatch_transfer','lot0',transfer_id='pre_move',target_warehouse_id='w1',target_lot_id='pre_arrival',quantity=4,key='pre_move')])
                b.world.orders['order1']=Order(id='order1',warehouse_id='w0',customer_id='u3',lines=[OrderLine(id='line0',sku_id='sku0',quantity=2,unit_price_cents=1200)],shipping_cents=500)
                b.confirmed('order1')
                b.flow('order1',reservation='r1',lot='lot0',quantity=2,shipment='s1',payment=False)
            message='Return two delivered r0 units from s0 as ret0 (customer_remorse), receive and independently inspect them. '+('Restore receiving and physical capacity to 26 if required. ' if variant in {'receiving','capacity'} else '')+('Restock accepted goods. ' if b.family!='return_refund' else '')+('Refund eligible goods with key returned. ' if b.family in {'return_refund','return_replacement'} or variant=='refund' else '')+('Use only this restocked lot0 to fulfill replacement order1 with r1 and s1/key s1.' if b.family=='return_replacement' else '')
    elif b.family=='draft_amendment':
        b.add(op('modify_order_line',line_id='line0',quantity=6))
        b.add(op('capture_payment',amount_cents=7700,key='pay',token='card_approved'),op('confirm_order'))
        if variant=='fulfill':
            b.flow(quantity=6,payment=False,alternatives=True)
        else:
            b.terminal('orders','order0','status','confirmed');b.terminal('orders','order0','lines.0.quantity',6)
        message='Change draft order0 line0 to six units, pay 7700 using card_approved/pay and confirm.'+(' Fulfill with r0 and s0/key s0 from either lot0 or lot1.' if variant=='fulfill' else '')
    elif b.family=='partial_cancel_fulfillment':
        b.allocated();b.add(op('cancel_order_line',line_id='line0',quantity=2))
        b.add(op('pick_reservation','r0'),op('pack_reservation','r0'),op('create_shipment',shipment_id='s0',reservation_ids=['r0'],key='s0'),op('dispatch_shipment','s0'))
        if variant=='refund':
            b.add(op('refund_order',amount_cents=2400,key='partial'));b.terminal('payments','payment_'+digest(['order0','refund','partial'])[:16],'outcome','succeeded')
        b.terminal('shipments','s0','status','shipped');b.terminal('orders','order0','lines.0.canceled',2)
        message='Cancel two of order0 line0 units from r0, then ship the remaining two using s0/key s0.'+(' Refund the canceled 2400 cents with partial.' if variant=='refund' else '')
    elif b.family in {'capacity_recovery','receiving_recovery','inbound_recovery'}:
        if b.family!='receiving_recovery':
            b.world.warehouses['w0'].capacity_volume=28;b.add(op('change_warehouse_capacity','w0',capacity_volume=36))
        if b.family!='capacity_recovery':
            b.world.warehouses['w0'].receiving_enabled=False;b.add(op('set_warehouse_receiving','w0',enabled=True))
        b.add(op('receive_inventory','inbound0',lot_id='arrival',key='receipt'));b.terminal('lots','arrival','on_hand',8)
        message='Restore w0 inbound admission, enabling receiving and/or increasing capacity to 36 where needed, then receive inbound0 into arrival using receipt. Preserve all orders and existing goods.'
    elif b.family=='capacity_turnover':
        b.world.warehouses['w0'].capacity_volume=32
        b.allocated();b.flow(payment=False,lot='lot0')
        # Existing r0 means reserve is an idempotent no-op and search discards it.
        b.add(op('receive_inventory','inbound0',lot_id='arrival',key='receipt'));b.terminal('lots','arrival','on_hand',8)
        message='Dispatch paid allocated order0 via r0 and s0/key s0 to free physical capacity, then receive inbound0 into arrival/receipt. Keep w0 capacity at 32.'
    elif b.family=='delivery_confirmation':
        b.shipped();b.add(op('confirm_delivery','s0'));b.terminal('shipments','s0','status','delivered')
        message='Confirm shipment s0 delivery using its independent carrier proof; preserve stock and payment history.'
    elif b.family=='partial_fulfillment':
        b.flow(quantity=2,alternatives=True);b.terminal('orders','order0','status','partially_shipped')
        message='Pay/confirm order0 if needed and ship only two line0 units, leaving the remainder open. Use r0 and s0/key s0, either lot0 or lot1, payment pay.'
    elif b.family=='fulfillment_return':
        b.flow(alternatives=True,delivery=True)
        b.add(op('request_return','s0',return_id='ret0',reservation_id='r0',quantity=2,reason='customer_remorse'),
              op('receive_return','ret0'),op('inspect_return','ret0'),op('restock_return','ret0'),op('refund_order',amount_cents=2400,key='returned'))
        b.terminal('returns','ret0','status','restocked');b.terminal('payments','payment_'+digest(['order0','refund','returned'])[:16],'outcome','succeeded')
        message='Fulfill order0 with r0 and s0/key s0, payment pay, using either lot0 or lot1. After independent delivery, process a two-unit remorse return ret0, inspect/restock it and refund 2400 with returned.'
    elif b.family=='multi_line_fulfillment':
        b.world.orders['order0'].lines.append(OrderLine(id='line1',sku_id='sku1',quantity=2,unit_price_cents=2300))
        b.add(b.paid(),op('confirm_order'))
        for rid,line,lot,q in [('r0','line0','lot0',4),('r1','line1','lot2',2)]:
            b.add(op('reserve_inventory',reservation_id=rid,line_id=line,lot_id=lot,quantity=q),op('pick_reservation',rid),op('pack_reservation',rid))
        b.add(op('create_shipment',shipment_id='s0',reservation_ids=['r0','r1'],key='s0'),op('dispatch_shipment','s0'))
        b.terminal('shipments','s0','status','shipped');b.terminal('orders','order0','status','shipped')
        message='Pay/confirm and fulfill BOTH order0 product lines in one s0 shipment, four sku0 from lot0/r0 and two sku1 from lot2/r1. Use pay and s0 keys; preserve SKU distinctions.'
    elif b.family=='split_fulfillment':
        b.add(b.paid(),op('confirm_order'))
        for rid,lot,sid in [('r0','lot0','s0'),('r1','lot1','s1')]:
            b.add(op('reserve_inventory',reservation_id=rid,line_id='line0',lot_id=lot,quantity=2),op('pick_reservation',rid),op('pack_reservation',rid),
                  op('create_shipment',shipment_id=sid,reservation_ids=[rid],key=sid),op('dispatch_shipment',sid))
            b.terminal('shipments',sid,'status','shipped')
        b.terminal('orders','order0','status','shipped')
        message='Pay/confirm order0 (pay), fulfill all four units in two separate shipments: two from lot0/r0 in s0/key s0 and two from lot1/r1 in s1/key s1.'
    elif b.family=='order_cancellation':
        if variant=='allocated':
            b.allocated()
        elif variant=='paid':
            b.confirmed()
        b.add(op('cancel_order'));b.terminal('orders','order0','status','canceled')
        if variant=='refund':
            b.confirmed();b.add(op('refund_order',amount_cents=5300,key='cancel-refund'));b.terminal('payments','payment_'+digest(['order0','refund','cancel-refund'])[:16],'outcome','succeeded')
        message='Cancel unshipped order0 and free any shelf allocations.'+(' Refund its 5300 payment with cancel-refund.' if variant=='refund' else '')
    elif b.family=='allocation_release':
        b.allocated();b.add(op('release_reservation','r0'));b.terminal('reservations','r0','status','released')
        message='Release unpicked order0 allocation r0 while retaining the paid confirmed order.'
    elif b.family=='payment_completion':
        b.prepare([b.paid(amount=1200,key='initial')]);b.add(b.paid(amount=4100),op('confirm_order'));b.terminal('orders','order0','status','confirmed')
        message='Complete order0 payment with exactly 4100 cents using card_approved/pay, retaining the existing 1200 payment, then confirm.'
    elif b.family.endswith('_refusal'):
        message,code=refusal_scenario(b,scenario)
        return b.freeze(message+' Refuse with the applicable policy code and preserve all state.',code)
    else:
        raise ValueError('unknown fulfillment business scenario: '+scenario)
    return b.freeze(message+' Keep unrelated entities and configuration unchanged.')


def progress_task(world,task,route,phase):
    task=task.model_copy(deep=True)
    witness=task.witnesses[route]
    if phase<1 or phase>=len(witness.operations):
        raise ValueError('phase must be a proper business-state prefix')
    from . import adapter
    world=adapter.World.model_validate(execute(adapter,world,task.identity,witness.operations[:phase]).final_state)
    # Initial progress is a real, independently verified workflow state. Existing
    # entities lose whole-entity exemptions on their immutable identity/bindings.
    frame={}
    for key,fields in task.goal.allowed_changes.items():
        collection,entity=key.split('/')
        if fields==['*'] and entity in getattr(world,collection) and collection not in DELETABLE:
            fields=FRAME_FIELDS[collection]
        if fields:frame[key]=fields
    task.goal.allowed_changes=frame
    task.witnesses=[];task.initial_state_hash=world.state_hash;task.id='pending'
    task.search_nodes=0;task.search_exhausted=False
    task.message='Continue this in-progress business request from actual current state; query current progress and do only remaining work. '+task.message
    return world,freeze_task(adapter,world,task)


def certificate_loader():
    """Per-validation cache of pure base certificates, with detached results.

    This cache belongs to a single validation call. No session, domain or trial
    receives a mutable cached World. The phase recipe still executes actual
    events and independently re-certifies the remaining business problem.
    """
    bases={}
    def reconstruct(seed,recipe):
        base=recipe.rsplit('@',2)[0] if '@' in recipe else recipe
        key=(seed,base)
        if key not in bases:bases[key]=make_task(seed,base)
        world,task=bases[key]
        if '@' in recipe:
            _,route,phase=recipe.rsplit('@',2)
            return progress_task(world,task,int(route),int(phase))
        return world.clone(),task.model_copy(deep=True)
    return reconstruct


def refusal_scenario(b,scenario):
    from datetime import timedelta
    op=b.op;name=scenario.split(':')[0];variant=scenario.split(':')[1] if ':' in scenario else ''
    if name=='warehouse_scope_refusal':
        b.add(op('confirm_order','order0','w1'));return 'Confirm w0 order0 using w1 scope.','WAREHOUSE_SCOPE_MISMATCH'
    if name=='staff_role_refusal':
        b.identity='u1' if variant!='buyer' else 'u3'
        b.add(op('receive_inventory','inbound0',lot_id='arrival',key='receipt'));return 'Receive inbound0 into arrival with receipt.','WAREHOUSE_PERMISSION_DENIED'
    if name=='stock_refusal':
        b.world.orders['order0'].lines[0].quantity=20;b.confirmed()
        b.add(op('reserve_inventory',reservation_id='r0',line_id='line0',lot_id='lot0',quantity=20));return 'Allocate all twenty order0 units from lot0 as r0.','STOCK_INSUFFICIENT'
    if name=='packing_refusal':
        b.allocated()
        if variant=='picked':b.prepare([op('pick_reservation','r0')])
        b.prepare([op('create_shipment',shipment_id='s0',reservation_ids=['r0'],key='ship')])
        b.add(op('dispatch_shipment','s0'));return 'Dispatch s0 immediately without further handling.','SHIPMENT_NOT_PACKED'
    if name=='shipped_cancel_refusal':
        b.shipped(delivered=variant=='delivered');b.add(op('cancel_order'));return 'Cancel already dispatched order0 through ordinary cancellation.','ORDER_ALREADY_SHIPPED'
    if name=='return_quantity_refusal':
        b.shipped(delivered=True);b.add(op('request_return','s0',return_id='ret0',reservation_id='r0',quantity=5,reason='customer_remorse'));return 'Return five delivered r0 units from s0 as ret0/customer_remorse.','RETURN_EXCEEDS_DELIVERY'
    if name=='unearned_refund_refusal':
        if variant=='paid':b.confirmed()
        b.add(op('refund_order',amount_cents=1,key='unearned'));return 'Refund one cent on uncanceled order0 using unearned.','REFUND_EXCEEDS_ELIGIBLE'
    if name=='payment_key_refusal':
        b.prepare([b.paid(amount=1200,key='original')]);b.add(b.paid(amount=1300,key='original'));return 'Charge order0 1300 with card_approved reusing the original key from its earlier 1200 charge.','IDEMPOTENCY_CONFLICT'
    if name=='committed_edit_refusal':
        b.confirmed();b.add(op('modify_order_line',line_id='line0',quantity=6));return 'Edit already paid confirmed order0 line0 to six units.','ORDER_ALREADY_COMMITTED'
    if name=='quality_refusal':
        b.world.orders['order0'].lines[0].sku_id='sku1';b.world.orders['order0'].lines[0].unit_price_cents=2300
        b.prepare([op('receive_inventory','inbound2',lot_id='arrival',key='receipt')]);b.confirmed()
        b.add(op('reserve_inventory',reservation_id='r0',line_id='line0',lot_id='arrival',quantity=4));return 'Reserve four quarantined arrival units for order0 as r0 without inspection.','QUALITY_HOLD'
    if name=='transfer_scope_refusal':
        b.add(op('dispatch_transfer','lot0',transfer_id='t0',target_warehouse_id='w2',target_lot_id='arrival',quantity=4,key='move'));return 'Transfer four lot0 units into another organization warehouse w2 as t0/move/arrival.','TRANSFER_DESTINATION_INVALID'
    if name=='transfer_access_refusal':
        b.world.access['w0_u1'].role='manager';b.identity='u1'
        b.add(op('dispatch_transfer','lot0',transfer_id='t0',target_warehouse_id='w1',target_lot_id='arrival',quantity=4,key='move'));return 'As w0 manager but w1 viewer, transfer four lot0 units to w1 via t0/move/arrival.','DESTINATION_PERMISSION_DENIED'
    if name=='receiving_refusal':
        b.world.warehouses['w0'].receiving_enabled=False;b.add(op('receive_inventory','inbound0',lot_id='arrival',key='receipt'));return 'Receive inbound0 into arrival/receipt while receiving remains disabled.','RECEIVING_DISABLED'
    if name=='capacity_refusal':
        b.world.warehouses['w0'].capacity_volume=28;b.add(op('receive_inventory','inbound0',lot_id='arrival',key='receipt'));return 'Receive inbound0 into arrival/receipt without freeing or expanding full w0 capacity.','CAPACITY_EXCEEDED'
    if name=='product_admission_refusal':
        b.world.warehouses['w0'].allowed_categories=['general'];b.add(op('receive_inventory','inbound2',lot_id='arrival',key='receipt'));return 'Receive fragile inbound2 into general-only w0 as arrival/receipt.','PRODUCT_NOT_ACCEPTED'
    if name=='return_window_refusal':
        b.shipped(delivered=True);b.world.clock+=timedelta(days=31)
        b.add(op('request_return','s0',return_id='ret0',reservation_id='r0',quantity=1,reason='customer_remorse'));return 'Request ret0 for one r0 unit from s0 after the product return window.','RETURN_WINDOW_EXPIRED'
    if name=='nonreturnable_refusal':
        b.world.skus['sku0'].returnable=False;b.shipped(delivered=True)
        b.add(op('request_return','s0',return_id='ret0',reservation_id='r0',quantity=1,reason='customer_remorse'));return 'Request ret0 for one nonreturnable r0 unit from delivered s0.','PRODUCT_NOT_RETURNABLE'
    if name=='damaged_refund_refusal':
        b.world.skus['sku0'].simulated_return_condition='customer_damage';b.shipped(delivered=True)
        b.prepare([op('request_return','s0',return_id='ret0',reservation_id='r0',quantity=1,reason='merchant_damage'),op('receive_return','ret0'),op('inspect_return','ret0')])
        b.add(op('refund_order',amount_cents=1200,key='damage'));return 'Refund 1200 for ret0 with damage even though independent inspection classified customer damage.','REFUND_EXCEEDS_ELIGIBLE'
    if name=='undelivered_return_refusal':
        b.shipped();b.add(op('request_return','s0',return_id='ret0',reservation_id='r0',quantity=1,reason='customer_remorse'));return 'Request return ret0 for one r0 unit from s0 before delivery confirmation.','DELIVERY_REQUIRED'
    if name=='capacity_reduction_refusal':
        b.add(op('change_warehouse_capacity','w0',capacity_volume=20));return 'Reduce w0 physical capacity to 20 while retaining all goods.','CAPACITY_BELOW_USAGE'
    if name=='picked_release_refusal':
        b.allocated();b.prepare([op('pick_reservation','r0')]);b.add(op('release_reservation','r0'));return 'Release picked r0 directly without physically reversing its pick.','RESERVATION_STAGE_INVALID'
    if name=='closed_charge_refusal':
        b.confirmed();b.add(b.paid(amount=1,key='new'));return 'Add a new one-cent charge to confirmed order0 with card_approved/new.','ORDER_PAYMENT_CLOSED'
    raise ValueError('unknown refusal scenario: '+scenario)
