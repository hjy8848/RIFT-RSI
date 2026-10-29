"""Conservative, ID-neutral relational colour refinement with policy-derived bins.

These are structural grouping signatures, not a complete graph-isomorphism proof.
Raw numeric jitter, display labels, timestamps and simulator signatures do not add diversity.
"""
from usami_harness.models import digest

REFS={'org_id':'organizations','owner_id':'users','user_id':'users','customer_id':'users','project_id':'projects','invoice_id':'invoices',
 'warehouse_id':'warehouses','source_warehouse_id':'warehouses','target_warehouse_id':'warehouses','sku_id':'skus','lot_id':'lots',
 'source_lot_id':'lots','target_lot_id':'lots','notice_id':'receipt_notices','order_id':'orders','reservation_id':'reservations','reservation_ids':'reservations',
 'shipment_id':'shipments','guest_id':'guests','property_id':'properties','room_type_id':'room_types','room_type_ids':'room_types',
 'rate_plan_id':'rate_plans','room_id':'rooms','current_stay_id':'stays','subscription_id':'subscriptions','dependencies':'instances'}
CATEGORIES={'status','role','quality','category','returnable','receiving_enabled','housekeeping','maintenance','kind','outcome',
 'simulated_return_condition','simulated_identity_result','late_fee','auto_renew','plan_id','closed','payment_method','method','token'}


def relations(domain,world,permissions_only=False):
    raw=world.model_dump(mode='json');nodes={};edges={}
    refs=dict(REFS);categories=set(CATEGORIES)
    allowed={'users','organizations','projects','warehouses','properties','access','memberships'}
    if domain=='aftersales':
        refs.update(merchant_id='merchants',product_id='products',line_id='lines',holder_id='users',case_id='cases',agreement_quote='quotes')
        categories.update({'issue','return_required','simulated_valid','option','review_requested'})
        allowed={'users','merchants','orders'}
    if domain=='airline':
        refs.update(carrier_id='carriers',passenger_id='users',fare_rule_id='fare_rules',booking_id='bookings',quote_id='quotes',flight_id='flights',flights='flights',origin='airports',destination='airports')
        categories.update({'cabin','changeable','refundable','verified','action'})
        allowed={'users','carriers','bookings'}
    if domain=='property':
        refs.update(building_id='buildings',unit_id='units',asset_id='assets',technician_id='technicians',slot_id='slots',booked_slot_id='slots',resident_ids='users',reporter_id='users',requester_id='users',ticket_id='tickets',plan_id='plans',appointment_id='appointments')
        categories.update({'condition','responsibility','on_call','skills','severity','simulated_symptom','simulated_hazard','timing'})
        allowed={'users','properties','buildings','units'}
    if domain=='education':
        refs.update(school_id='schools',student_id='users',requester_id='users',course_id='courses',original_course_id='courses',course_ids='courses',package_id='packages',teacher_id='teachers',class_id='classes',enrollment_id='enrollments',case_id='cases',quote_id='quotes',slot_id='slots',appointment_id='appointments',target_class_id='classes')
        categories.update({'subject','subjects','level','max_level','transferable','extension_reasons','no_show_consumes','origin','reason','action'})
        allowed={'users','schools','enrollments'}
    for collection,value in raw.items():
        if not isinstance(value,dict) or collection in {'rules','config'} or (permissions_only and collection not in allowed):continue
        for key,obj in value.items():
            if not isinstance(obj,dict) or 'id' not in obj:continue
            token=(collection,key);nodes[token]={'type':collection};edges[token]=[]
    def scan(token,value,prefix=''):
        if isinstance(value,dict):
            for field,item in value.items():
                if field in refs:
                    values=item if isinstance(item,list) else [item]
                    for position,target in enumerate(values):
                        right=(refs[field],target) if isinstance(target,str) else None
                        if right in nodes:
                            label=prefix+field+(':'+str(position) if domain=='airline' and field=='flights' else '')
                            edges[token].append((label,right));edges[right].append(('reverse:'+label,token))
                elif field in categories and (not permissions_only or field=='role' or (field=='status' and token[0] in {'access','memberships'})):
                    nodes[token][prefix+field]=sorted(item,key=digest) if (domain=='property' and field=='skills') or (domain=='education' and field in {'subjects','extension_reasons'}) else item
                elif isinstance(item,(list,dict)) and field not in {'attestation','gateway_notice','carrier_notice','receipt_notice','inspection_notice','receiving_notice','notice','payments','refunds'}:
                    scan(token,item,prefix+field+'.')
        elif isinstance(value,list):
            items=[]
            for item in value:
                if isinstance(item,dict):
                    # List order is not a new business structure.
                    items.append({k:v for k,v in item.items() if k in categories})
                    scan(token,item,prefix+'item.')
            if items:nodes[token][prefix+'categories']=sorted(items,key=digest)
    for token in nodes:
        scan(token,raw[token[0]][token[1]])
        if not permissions_only:nodes[token].update(policy_features(domain,world,*token))
    colours={k:digest(v) for k,v in nodes.items()}
    base=dict(colours)
    for _ in range(6):colours={k:digest([base[k],sorted((label,colours[other]) for label,other in edges[k])]) for k in nodes}
    return digest({'version':'business-relations-v1','nodes':sorted(colours.values())})


def relation(a,b):return 'below' if a<b else 'equal' if a==b else 'above'


def policy_features(domain,w,collection,key):
    obj=getattr(w,collection)[key];features={}
    if domain=='cloudops':
        from usami_cloudops.rules import usage
        if collection=='projects':features['quota_usage']={k:relation(v,getattr(obj.quota,k)) for k,v in usage(w,key).items()}
        if collection=='instances':
            quota=w.projects[obj.project_id].quota
            features['fits_alone']=obj.cpu<=quota.cpu and obj.memory<=quota.memory and quota.instances>0
    elif domain=='fulfillment':
        from usami_fulfillment.rules import physical_volume,paid,price
        if collection=='warehouses':features['capacity']=relation(physical_volume(w,key),obj.capacity_volume)
        if collection=='lots':features['free_stock']='none' if obj.on_hand==obj.reserved else 'available';features['picked']=bool(obj.picked)
        if collection=='orders':
            features['paid']=relation(paid(w,key)-paid(w,key,'refund'),price(obj))
            features['line_stock']=sorted((relation(l.quantity-l.canceled,sum(lot.on_hand-lot.reserved for lot in w.lots.values() if lot.warehouse_id==obj.warehouse_id and lot.sku_id==l.sku_id and lot.quality=='available')),bool(l.canceled)) for l in obj.lines)
    elif domain=='stayops':
        from usami_stayops.rules import occupied,calendar_capacity,net_paid,cancellation_fee
        if collection=='calendar':features['occupancy']=relation(occupied(w,obj.room_type_id,obj.day),calendar_capacity(w,obj.room_type_id,obj.day));features['weekend']=obj.day.weekday()>=5
        if collection=='reservations':features.update(payment=relation(net_paid(w,key),obj.total_cents),deposit=relation(net_paid(w,key),obj.deposit_cents),arrival=relation(obj.arrival,w.clock.date()),cancel_fee='free' if not cancellation_fee(w,obj,w.clock) else 'penalty',hold_expired=w.clock>=obj.hold_expires_at)
        if collection=='rate_plans':features['deposit']='none' if not obj.deposit_bps else 'full' if obj.deposit_bps==10000 else 'partial';features['free_window']='none' if not obj.free_cancel_hours else 'positive'
    elif domain=='aftersales':
        from datetime import timedelta
        if collection=='lines':
            order=w.orders[obj.order_id];product=w.products[obj.product_id]
            features.update(return_window='not_delivered' if order.delivered_at is None else 'open' if w.clock<=order.delivered_at+timedelta(days=product.return_days) else 'expired',
                replacement_stock=relation(product.stock,obj.quantity),approval_required=obj.quantity*obj.unit_price_cents>w.merchants[order.merchant_id].refund_approval_limit_cents,
                paid_balance=relation(order.paid_cents-sum(r.amount_cents for r in w.refunds.values() if r.order_id==order.id),obj.quantity*obj.unit_price_cents))
            profiles=[p for p in w.profiles.values() if p.facts.get('line_id')==key]
            features['deadline_fits']=sorted(w.clock+timedelta(days=product.replacement_days)<=__import__('datetime').datetime.fromisoformat(p.facts['deadline']) for p in profiles)
    elif domain=='airline':
        from datetime import timedelta
        from usami_airline.rules import occupancy,disrupted
        if collection=='flights':
            features['seat_usage']={c:relation(occupancy(w,key,c),n) for c,n in obj.seats.items()}
            features['connection_boundaries']=sorted(relation(f.departure,obj.arrival+timedelta(minutes=w.airports[obj.destination].min_connection_minutes)) for f in w.flights.values() if f.carrier_id==obj.carrier_id and f.origin==obj.destination)
        if collection=='bookings':
            rule=w.fare_rules[obj.fare_rule_id]
            features.update(disrupted=disrupted(w,obj),change_window=relation(min(w.flights[f].departure for f in obj.flights),w.clock+timedelta(hours=rule.notice_hours)),baggage_fit=all(rule.baggage_included_kg+obj.extra_baggage_kg<=w.flights[f].max_baggage_kg for f in obj.flights))
            profiles=[p for p in w.profiles.values() if p.facts.get('booking_id')==key]
            features['refund_fee']=relation(obj.fare_value_cents,rule.refund_fee_cents)
            features['change_fee_within_customer_budget']=sorted(rule.change_fee_cents<=p.facts['max_fee_cents'] for p in profiles)
    elif domain=='property':
        from datetime import timedelta
        from usami_property.rules import contract,cost_terms,overlaps
        if collection=='slots':
            c=next(c for c in w.contracts.values() if c.property_id==obj.property_id)
            features.update(past=obj.start<=w.clock,working_hours=obj.start.hour>=c.normal_start_hour and obj.end.date()==obj.start.date() and (obj.end.hour<c.normal_end_hour or obj.end.hour==c.normal_end_hour and not obj.end.minute),
                busy=any(b.technician_id==obj.technician_id and overlaps(obj,b) for b in w.background_busy.values()),
                response_windows={k:obj.start<=w.clock+timedelta(hours=v) for k,v in [('urgent',c.urgent_response_hours),('emergency',c.emergency_response_hours)]})
        if collection=='assets':
            unit=w.units[obj.unit_id];c=contract(w,unit);profiles=[p for p in w.profiles.values() if p.facts.get('asset_id')==key]
            features['customer_contract_boundaries']=sorted((p.identity==unit.owner_id,relation(cost_terms(w,obj,p.identity)[0],c.resident_approval_limit_cents),cost_terms(w,obj,p.identity)[2]<=p.facts['max_budget_cents']) for p in profiles)
            features['slot_duration_fit']=sorted((s.end-s.start).total_seconds()>=obj.duration_minutes*60 for s in w.slots.values() if s.property_id==obj.property_id and obj.category in w.technicians[s.technician_id].skills)
    elif domain=='education':
        from datetime import timedelta
        from usami_education.rules import occupancy,used_units,reservation_value,teacher_busy,student_busy
        if collection=='classes':
            features['capacity']=relation(occupancy(w,key),obj.capacity)
        if collection=='enrollments':
            c=w.contracts[obj.contract_id];course=w.courses[obj.course_id]
            features.update(expiry=relation(obj.expires_at,w.clock),refund_window=relation(w.clock,obj.enrolled_at+timedelta(days=c.refund_days)),
                consumed_limit=relation(used_units(w,key)*10000,c.total_units*c.refund_consumed_limit_bps),
                credit_for_one_unit=relation(obj.remaining_credit_cents-reservation_value(w,key),course.unit_price_cents),
                extension_remaining=relation(obj.extended_days,c.max_extension_days),
                class_fits_validity=all(m.end<=obj.expires_at for m in w.classes[obj.class_id].meetings))
        if collection=='slots':
            features.update(past=obj.start<=w.clock,teacher_busy=teacher_busy(w,obj.teacher_id,obj),
                student_conflicts=sorted(student_busy(w,u,obj) for u in w.users),
                duration_fit=sorted((obj.end-obj.start).total_seconds()>=c.unit_minutes*60 for c in w.courses.values() if c.school_id==obj.school_id))
    elif domain=='saas':
        if collection=='subscriptions':
            members=[m for m in w.memberships.values() if m.org_id==obj.org_id and m.status=='active'] if hasattr(obj,'org_id') else []
            features['seat_usage']=relation(len(members),obj.seats);features['expiry']=relation(obj.period_end,w.clock)
        if collection=='invoices':features['refundable']=bool(obj.refundable_amount_cents);features['refund_window']=bool(obj.paid_at and (w.clock-obj.paid_at).days<=w.rules.refund_window_days)
    return features


def signatures(domain,world):return {'structure':relations(domain,world),'permissions':relations(domain,world,True)}
