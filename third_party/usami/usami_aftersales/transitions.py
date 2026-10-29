"""Actual mutations. No policy/Reference import; Sandbox can expose wrong decisions."""
from datetime import timedelta
from usami_harness.models import Result,digest,signed_notice
from .models import Case,Quote,Return,Refund,Replacement


def apply(world,actor,op,event_index):
    new=world.clone();p=op.params;order=new.orders.get(op.resource_id)
    try:
        if op.is_query:
            from .observations import query_data
            return new,Result(success=True,data=query_data(world,actor,op))
        if not order:return world.clone(),Result(success=False,error_code='MUTATION_NOT_REPRESENTABLE')
        case=new.cases.get(p.get('case_id',''));data={}
        if op.name=='open_case':
            line=new.lines[p['line_id']];cid='case_'+digest([order.id,line.id,p['issue']])[:16]
            if cid not in new.cases:new.cases[cid]=Case(id=cid,order_id=order.id,line_id=line.id,issue=p['issue'])
            data={'case_id':cid,'status':new.cases[cid].status}
        elif op.name=='verify_evidence':
            evidence=new.evidence[p['evidence_id']]
            verified=evidence.simulated_valid and evidence.issue==case.issue
            if not verified:return world.clone(),Result(success=False,error_code='EVIDENCE_INVALID')
            binding={'case_id':case.id,'order_id':case.order_id,'line_id':case.line_id,'evidence_id':evidence.id,'verified':verified}
            case.evidence_id=evidence.id;case.evidence_notice=signed_notice(new.domain,'evidence-verifier',binding,new.clock)
            if case.status=='open':case.status='evidence_verified'
            data={'case_id':case.id,'verified':True}
        elif op.name=='request_manager_review':
            case.review_requested=True;data={'case_id':case.id,'status':'pending'}
        elif op.name=='quote_resolution':
            line=new.lines[case.line_id];product=new.products[line.product_id];revision=new.interaction.revision
            qid='quote_'+digest([case.id,p['option'],revision])[:16]
            if qid in new.quotes and new.quotes[qid].status=='offered':return new,Result(success=True,data={'quote':new.quotes[qid].model_dump(mode='json')})
            delivery=new.clock+timedelta(days=product.replacement_days) if p['option'] in {'exchange','reship'} else None
            q=Quote(id=qid,order_id=order.id,case_id=case.id,option=p['option'],amount_cents=line.unit_price_cents*line.quantity if p['option']=='refund' else 0,
                delivery_at=delivery,requires_return=product.return_required and p['option']=='refund',revision=revision,valid_until=new.clock+timedelta(hours=1),quote_hash='')
            q.quote_hash=digest(q.model_dump(mode='json',exclude={'quote_hash','status'}));new.quotes[qid]=q;data={'quote':q.model_dump(mode='json')}
        elif op.name=='commit_resolution':
            q=new.quotes[p['quote_id']];case=new.cases[q.case_id];line=new.lines[case.line_id];product=new.products[line.product_id]
            q.status='committed';case.agreement_quote=q.id;case.agreement_revision=new.interaction.revision
            if q.requires_return:case.status='return_pending';line.status='return_pending'
            elif q.option=='refund':_refund(new,order,line,case,q,p['destination_channel']);case.status='resolved'
            else:
                product.stock-=line.quantity;rid='replacement_'+case.id
                new.replacements[rid]=Replacement(id=rid,order_id=order.id,line_id=line.id,product_id=product.id,quantity=line.quantity,kind=q.option,delivery_at=q.delivery_at)
                line.status='exchanged' if q.option=='exchange' else 'reshipped';case.status='resolved'
            data={'case_id':case.id,'status':case.status,'line_status':line.status}
        elif op.name=='schedule_return':
            rid='return_'+case.id
            if rid not in new.returns:new.returns[rid]=Return(id=rid,case_id=case.id,pickup_slot=p['pickup_slot'])
            data={'return_id':rid,'status':new.returns[rid].status}
        elif op.name=='release_refund':
            q=new.quotes[case.agreement_quote];line=new.lines[case.line_id]
            _refund(new,order,line,case,q,order.payment_channel);case.status='resolved'
            data={'case_id':case.id,'status':'resolved','line_status':'refunded'}
        return new,Result(success=True,data=data)
    except (KeyError,ValueError,TypeError,AttributeError):return world.clone(),Result(success=False,error_code='MUTATION_NOT_REPRESENTABLE')


def _refund(state,order,line,case,quote,channel):
    key='refund_'+case.id;value=quote.amount_cents
    binding={'order_id':order.id,'line_id':line.id,'amount_cents':value,'channel':channel}
    state.refunds[key]=Refund(id=key,order_id=order.id,line_id=line.id,case_id=case.id,amount_cents=value,channel=channel,
        notice=signed_notice(state.domain,'payment-gateway',binding,state.clock));line.status='refunded'
