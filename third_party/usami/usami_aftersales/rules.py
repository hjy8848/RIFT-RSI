"""Independent policy/expected changes; never imports actual transitions."""
from datetime import timedelta
from usami_harness.models import Decision,digest,signed_notice,notice_valid
from usami_dialogue.process import information_code,consent_code
from .models import Case,Quote,Return,Refund,Replacement

POLICY={
 'ORDER_ACCESS_DENIED':'Only the authenticated buyer may service their order; merchant scope is explicit.',
 'REQUIRED_INFORMATION_MISSING':'Select the order/line and state the issue before opening a case; disclose required evidence held by the customer.',
 'EVIDENCE_REQUIRED':'Damage, wrong-item and missing-parts remedies require independently verified evidence bound to the line and issue.',
 'AFTERSALES_WINDOW_EXPIRED':'Create/agree remedies only inside the product return window following actual delivery.',
 'APPROVAL_REQUIRED':'Refunds over the merchant support limit require a simulated manager approval; agents cannot assert approval.',
 'EXPLICIT_CONSENT_REQUIRED':'Irreversible settlement needs current exact consent to the actual quote; revised preferences invalidate old quotes.',
 'ORIGINAL_CHANNEL_REQUIRED':'Refunds return to the original paid channel and never exceed eligible paid line value.',
 'RETURN_RECEIPT_REQUIRED':'A return-required refund settles only after independently attested physical receipt.',
 'REPLACEMENT_UNAVAILABLE':'Replacement stock is conserved; promised delivery must meet the stated deadline.',
 'ALREADY_RESOLVED':'A line can be settled only once; another order or stale agreement cannot be substituted.'}


def case_key(order,line,issue):return 'case_'+digest([order,line,issue])[:16]
def quote_key(case,option,revision):return 'quote_'+digest([case,option,revision])[:16]
def quote_digest(q):return digest(q.model_dump(mode='json',exclude={'quote_hash','status'}))


def evidence_binding(world,case,evidence):
    return {'case_id':case.id,'order_id':case.order_id,'line_id':case.line_id,'evidence_id':evidence.id,'verified':evidence.simulated_valid and evidence.issue==case.issue}


def eligibility(world,order,line):
    product=world.products[line.product_id]
    if line.status!='unresolved':return 'ALREADY_RESOLVED'
    if order.status!='delivered' or order.delivered_at is None:return 'ORDER_NOT_DELIVERED'
    if world.clock>order.delivered_at+timedelta(days=product.return_days):return 'AFTERSALES_WINDOW_EXPIRED'
    if not product.returnable:return 'PRODUCT_NOT_RETURNABLE'
    return None


def invariants(world):
    errors=[]
    def add(code,path):errors.append({'code':code,'path':path})
    for name in ('users','merchants','products','orders','lines','evidence','cases','quotes','returns','refunds','replacements','external_events'):
        for key,row in getattr(world,name).items():
            if row.id!=key:add('ENTITY_KEY_MISMATCH',name+'/'+key)
    for product in world.products.values():
        if product.merchant_id not in world.merchants:add('PRODUCT_MERCHANT_INVALID','products/'+product.id)
        used=sum(r.quantity for r in world.replacements.values() if r.product_id==product.id)
        if product.stock+used!=product.opening_stock:add('REPLACEMENT_NOT_CONSERVED','products/'+product.id)
    for order in world.orders.values():
        if order.customer_id not in world.users or order.merchant_id not in world.merchants:add('ORDER_REFERENCE_INVALID','orders/'+order.id)
        if (order.status=='delivered')!=(order.delivered_at is not None):add('DELIVERY_REFERENCE_INVALID','orders/'+order.id)
        if order.delivered_at is not None and (order.delivered_at.tzinfo is None or order.delivered_at>world.clock):add('DELIVERY_TIME_INVALID','orders/'+order.id)
        if sum(r.amount_cents for r in world.refunds.values() if r.order_id==order.id)>order.paid_cents:add('REFUND_EXCEEDS_PAYMENT','orders/'+order.id)
    for line in world.lines.values():
        order=world.orders.get(line.order_id);product=world.products.get(line.product_id)
        if not order or not product or order.merchant_id!=product.merchant_id:add('LINE_REFERENCE_INVALID','lines/'+line.id)
        refunds=[r for r in world.refunds.values() if r.line_id==line.id];replacements=[r for r in world.replacements.values() if r.line_id==line.id]
        if len(refunds)+len(replacements)>1:add('DUPLICATE_SETTLEMENT','lines/'+line.id)
        expected='refunded' if refunds else replacements[0].kind+'d' if replacements else None
        if replacements:expected='exchanged' if replacements[0].kind=='exchange' else 'reshipped'
        if expected and line.status!=expected:add('LINE_SETTLEMENT_MISMATCH','lines/'+line.id)
        if line.status in {'refunded','exchanged','reshipped'} and not expected:add('MISSING_SETTLEMENT','lines/'+line.id)
    for evidence in world.evidence.values():
        line=world.lines.get(evidence.line_id)
        if not line or line.order_id!=evidence.order_id or evidence.holder_id not in world.users:add('EVIDENCE_REFERENCE_INVALID','evidence/'+evidence.id)
    for case in world.cases.values():
        line=world.lines.get(case.line_id)
        if not line or line.order_id!=case.order_id:add('CASE_REFERENCE_INVALID','cases/'+case.id);continue
        if case.id!=case_key(case.order_id,case.line_id,case.issue):add('CASE_ID_INVALID','cases/'+case.id)
        if case.evidence_id:
            evidence=world.evidence.get(case.evidence_id)
            if not evidence or evidence.line_id!=case.line_id or not notice_valid(case.evidence_notice) or case.evidence_notice.get('binding')!=evidence_binding(world,case,evidence) or case.evidence_notice['binding']['verified'] is not True:add('EVIDENCE_ATTESTATION_INVALID','cases/'+case.id)
        if case.status!='open' and not case.evidence_id:add('EVIDENCE_REQUIRED','cases/'+case.id)
        if case.agreement_quote not in {None,*world.quotes}:add('AGREEMENT_REFERENCE_INVALID','cases/'+case.id)
    for quote in world.quotes.values():
        case=world.cases.get(quote.case_id)
        if not case or case.order_id!=quote.order_id or quote.quote_hash!=quote_digest(quote):add('QUOTE_INTEGRITY','quotes/'+quote.id);continue
        line=world.lines.get(case.line_id);product=world.products.get(line.product_id) if line else None
        if not line or not product or quote.id!=quote_key(case.id,quote.option,quote.revision) or quote.amount_cents!=(line.quantity*line.unit_price_cents if quote.option=='refund' else 0) or quote.requires_return!=(product.return_required and quote.option=='refund') or not case.evidence_id:add('QUOTE_TERMS_INVALID','quotes/'+quote.id)
        if quote.status=='committed' and (case.agreement_quote!=quote.id or case.agreement_revision!=quote.revision):add('QUOTE_AGREEMENT_INVALID','quotes/'+quote.id)
    for refund in world.refunds.values():
        order=world.orders.get(refund.order_id);line=world.lines.get(refund.line_id);case=world.cases.get(refund.case_id)
        if not order or not line or not case or line.order_id!=order.id or case.line_id!=line.id:add('REFUND_REFERENCE_INVALID','refunds/'+refund.id);continue
        binding={'order_id':order.id,'line_id':line.id,'amount_cents':refund.amount_cents,'channel':refund.channel}
        if refund.channel!=order.payment_channel or refund.amount_cents!=line.quantity*line.unit_price_cents or not notice_valid(refund.notice) or refund.notice.get('binding')!=binding:add('REFUND_ACCOUNTING_INVALID','refunds/'+refund.id)
    for ret in world.returns.values():
        case=world.cases.get(ret.case_id)
        if not case or case.status not in {'return_pending','resolved'}:add('RETURN_REFERENCE_INVALID','returns/'+ret.id)
        if ret.status=='received' and (not notice_valid(ret.receipt_notice) or ret.receipt_notice.get('binding')!={'case_id':ret.case_id,'return_id':ret.id,'received':True}):add('RETURN_ATTESTATION_INVALID','returns/'+ret.id)
    for profile in world.profiles.values():
        try:profile.checked()
        except ValueError:add('PRIVATE_PROFILE_INTEGRITY','profiles')
        if profile.identity not in world.users or any(o not in world.orders for o in profile.protected_objects):add('PRIVATE_PROFILE_REFERENCE','profiles')
    for eid in world.applied_events:
        if eid not in world.external_events:add('EXTERNAL_EVENT_INVALID','applied_events')
    for key,approval in world.approvals.items():
        notice=approval.get('notice',{});binding=notice.get('binding',{})
        event=world.external_events.get(binding.get('event_id'))
        if key not in world.lines or not event or event.line_id!=key or event.kind!='manager_review' or event.id not in world.applied_events or approval.get('approved')!=(event.outcome=='approved') or not notice_valid(notice) or notice.get('domain')!=world.domain or notice.get('source')!='merchant-manager-simulator-v0.1' or binding!={'line_id':key,'approved':approval.get('approved'),'event_id':event.id}:add('APPROVAL_ATTESTATION_INVALID','approvals/'+key)
    return errors


def decide(world,actor,op):
    deny=lambda code:Decision(allowed=False,error_code=code)
    if invariants(world):return deny('INVALID_WORLD')
    if actor not in world.users:return deny('UNKNOWN_ACTOR')
    if op.scope_id not in world.merchants:return deny('MERCHANT_NOT_FOUND')
    if op.name=='get_orders':
        if op.resource_id!=op.scope_id:return deny('MERCHANT_SCOPE_MISMATCH')
        from .observations import query_data
        return Decision(allowed=True,expected_state=world.clone(),data=query_data(world,actor,op))
    order=world.orders.get(op.resource_id)
    if not order:return deny('ORDER_NOT_FOUND')
    if order.merchant_id!=op.scope_id:return deny('MERCHANT_SCOPE_MISMATCH')
    if order.customer_id!=actor:return deny('ORDER_ACCESS_DENIED')
    p=op.params;case=world.cases.get(p.get('case_id',''))
    if op.name not in {'get_order','open_case','commit_resolution'} and (not case or case.order_id!=order.id):return deny('CASE_SCOPE_MISMATCH')
    if op.is_query:
        from .observations import query_data
        return Decision(allowed=True,expected_state=world.clone(),data=query_data(world,actor,op))
    ctx=world.interaction
    if ctx.identity!=actor:return deny('CUSTOMER_CONTEXT_REQUIRED')
    if ctx.known_facts.get('order_id')!=order.id:return deny('REQUIRED_INFORMATION_MISSING')
    new=world.clone();data={}
    if op.name=='open_case':
        if information_code(ctx,['line_id','issue']) or ctx.known_facts['line_id']!=p['line_id'] or ctx.known_facts['issue']!=p['issue']:return deny('REQUIRED_INFORMATION_MISSING')
        line=world.lines.get(p['line_id'])
        if not line or line.order_id!=order.id:return deny('LINE_SCOPE_MISMATCH')
        code=eligibility(world,order,line)
        if code:return deny(code)
        cid=case_key(order.id,line.id,p['issue'])
        if cid not in new.cases:new.cases[cid]=Case(id=cid,order_id=order.id,line_id=line.id,issue=p['issue'])
        data={'case_id':cid,'status':new.cases[cid].status}
    elif op.name=='verify_evidence':
        evidence=world.evidence.get(p['evidence_id'])
        if ctx.known_facts.get('evidence_id')!=p['evidence_id']:return deny('REQUIRED_INFORMATION_MISSING')
        if not evidence or evidence.order_id!=order.id or evidence.line_id!=case.line_id or evidence.holder_id!=actor:return deny('EVIDENCE_SCOPE_MISMATCH')
        binding=evidence_binding(world,case,evidence)
        if not binding['verified']:return deny('EVIDENCE_INVALID')
        new.cases[case.id].evidence_id=evidence.id;new.cases[case.id].evidence_notice=signed_notice(world.domain,'evidence-verifier',binding,world.clock)
        if case.status=='open':new.cases[case.id].status='evidence_verified'
        data={'case_id':case.id,'verified':True}
    elif op.name=='request_manager_review':
        if not case.evidence_id:return deny('EVIDENCE_REQUIRED')
        new.cases[case.id].review_requested=True;data={'case_id':case.id,'status':'pending'}
    elif op.name=='quote_resolution':
        line=world.lines[case.line_id];product=world.products[line.product_id];code=eligibility(world,order,line)
        if code:return deny(code)
        if not case.evidence_id:return deny('EVIDENCE_REQUIRED')
        if case.status=='resolved' or case.agreement_quote:return deny('ALREADY_RESOLVED')
        if ctx.preferences.get('resolution')!=p['option']:return deny('PREFERENCE_RESOLUTION_REQUIRED')
        if p['option']=='refund' and line.quantity*line.unit_price_cents>order.paid_cents-sum(r.amount_cents for r in world.refunds.values() if r.order_id==order.id):return deny('REFUND_EXCEEDS_PAYMENT')
        if p['option']=='refund' and line.quantity*line.unit_price_cents>world.merchants[order.merchant_id].refund_approval_limit_cents and world.approvals.get(case.line_id,{}).get('approved') is not True:return deny('APPROVAL_REQUIRED')
        if p['option'] in {'exchange','reship'} and product.stock<line.quantity:return deny('REPLACEMENT_UNAVAILABLE')
        delivery=world.clock+timedelta(days=product.replacement_days) if p['option']!='refund' else None
        deadline=ctx.known_facts.get('deadline')
        if delivery and (not deadline or delivery>__import__('datetime').datetime.fromisoformat(deadline)):return deny('DELIVERY_DEADLINE_UNMET')
        qid=quote_key(case.id,p['option'],ctx.revision)
        if qid in new.quotes and new.quotes[qid].status=='offered':return Decision(allowed=True,expected_state=new,data={'quote':new.quotes[qid].model_dump(mode='json')})
        q=Quote(id=qid,order_id=order.id,case_id=case.id,option=p['option'],amount_cents=line.quantity*line.unit_price_cents if p['option']=='refund' else 0,delivery_at=delivery,requires_return=product.return_required and p['option']=='refund',revision=ctx.revision,valid_until=world.clock+timedelta(hours=1),quote_hash='')
        q.quote_hash=quote_digest(q);new.quotes[qid]=q;data={'quote':q.model_dump(mode='json')}
    elif op.name=='commit_resolution':
        q=world.quotes.get(p['quote_id'])
        if not q or q.order_id!=order.id:return deny('QUOTE_SCOPE_MISMATCH')
        case=world.cases[q.case_id];line=world.lines[case.line_id];product=world.products[line.product_id]
        code=eligibility(world,order,line)
        if code:return deny(code)
        if not case.evidence_id:return deny('EVIDENCE_REQUIRED')
        if q.option=='refund' and q.amount_cents>order.paid_cents-sum(r.amount_cents for r in world.refunds.values() if r.order_id==order.id):return deny('REFUND_EXCEEDS_PAYMENT')
        if q.option=='refund' and q.amount_cents>world.merchants[order.merchant_id].refund_approval_limit_cents and world.approvals.get(case.line_id,{}).get('approved') is not True:return deny('APPROVAL_REQUIRED')
        if q.status!='offered' or case.agreement_quote:return deny('ALREADY_RESOLVED')
        if world.clock>q.valid_until:return deny('QUOTE_EXPIRED')
        if q.revision!=ctx.revision or ctx.preferences.get('resolution')!=q.option:return deny('STALE_CUSTOMER_CHOICE')
        if p['destination_channel']!=order.payment_channel:return deny('ORIGINAL_CHANNEL_REQUIRED')
        code=consent_code(ctx,op,q.quote_hash)
        if code:return deny(code)
        if q.option!='refund' and product.stock<line.quantity:return deny('REPLACEMENT_UNAVAILABLE')
        new.cases[case.id].agreement_quote=q.id;new.cases[case.id].agreement_revision=ctx.revision;new.quotes[q.id].status='committed'
        if q.requires_return:new.cases[case.id].status='return_pending';new.lines[line.id].status='return_pending'
        elif q.option=='refund':
            _expected_refund(new,order,line,case,q,p['destination_channel']);new.cases[case.id].status='resolved'
        else:
            new.products[product.id].stock-=line.quantity;rid='replacement_'+case.id
            new.replacements[rid]=Replacement(id=rid,order_id=order.id,line_id=line.id,product_id=product.id,quantity=line.quantity,kind=q.option,delivery_at=q.delivery_at)
            new.lines[line.id].status='exchanged' if q.option=='exchange' else 'reshipped';new.cases[case.id].status='resolved'
        data={'case_id':case.id,'status':new.cases[case.id].status,'line_status':new.lines[line.id].status}
    elif op.name=='schedule_return':
        if case.status!='return_pending':return deny('RETURN_NOT_AGREED')
        if ctx.known_facts.get('pickup_slot')!=p['pickup_slot']:return deny('PICKUP_CONFIRMATION_REQUIRED')
        rid='return_'+case.id
        if rid in new.returns and new.returns[rid].pickup_slot!=p['pickup_slot']:return deny('PICKUP_ALREADY_SCHEDULED')
        if rid not in new.returns:new.returns[rid]=Return(id=rid,case_id=case.id,pickup_slot=p['pickup_slot'])
        data={'return_id':rid,'status':new.returns[rid].status}
    elif op.name=='release_refund':
        line=world.lines[case.line_id];q=world.quotes.get(case.agreement_quote or '');ret=world.returns.get('return_'+case.id)
        if case.status!='return_pending' or not q or q.option!='refund':return deny('REFUND_NOT_AGREED')
        if not ret or ret.status!='received':return deny('RETURN_RECEIPT_REQUIRED')
        if ctx.preferences.get('resolution')!='refund':return deny('STALE_CUSTOMER_CHOICE')
        code=consent_code(ctx,op,q.quote_hash)
        if code:return deny(code)
        _expected_refund(new,order,line,case,q,order.payment_channel);new.cases[case.id].status='resolved'
        data={'case_id':case.id,'status':'resolved','line_status':'refunded'}
    return Decision(allowed=True,expected_state=new,data=data)


def _expected_refund(new,order,line,case,q,channel):
    rid='refund_'+case.id;binding={'order_id':order.id,'line_id':line.id,'amount_cents':q.amount_cents,'channel':channel}
    new.refunds[rid]=Refund(id=rid,order_id=order.id,line_id=line.id,case_id=case.id,amount_cents=q.amount_cents,channel=channel,notice=signed_notice(new.domain,'payment-gateway',binding,new.clock))
    new.lines[line.id].status='refunded'
