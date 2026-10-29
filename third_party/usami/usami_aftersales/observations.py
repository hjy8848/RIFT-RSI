"""No evidence simulator outcome or private preference is a Service observation."""
from usami_dialogue.process import public_context


def observation(world,actor):
    return {'domain':world.domain,'identity':world.users[actor].model_dump(mode='json'),'clock':world.clock.isoformat(),
        'orders':[o.model_dump(mode='json') for o in sorted(world.orders.values(),key=lambda row:row.id) if o.customer_id==actor],
        'customer_provided':public_context(world.interaction)}


def visible_ids(view):
    return {r['id'] for rows in view.values() if isinstance(rows,list) for r in rows if isinstance(r,dict) and 'id' in r}


def query_data(world,actor,op):
    if op.name=='get_orders':return {'orders':[o.model_dump(mode='json') for o in sorted(world.orders.values(),key=lambda row:row.id) if o.customer_id==actor and o.merchant_id==op.scope_id]}
    order=world.orders[op.resource_id]
    if op.name=='get_order':return {**order.model_dump(mode='json'),'lines':[l.model_dump(mode='json') for l in sorted(world.lines.values(),key=lambda row:row.id) if l.order_id==order.id]}
    case=world.cases[op.params['case_id']]
    if op.name=='get_case':return {**case.model_dump(mode='json',exclude={'evidence_notice'}),'quotes':[q.model_dump(mode='json') for q in sorted(world.quotes.values(),key=lambda row:row.id) if q.case_id==case.id],'returns':[r.model_dump(mode='json',exclude={'receipt_notice'}) for r in sorted(world.returns.values(),key=lambda row:row.id) if r.case_id==case.id]}
    product=world.products[world.lines[case.line_id].product_id]
    return {'return_days':product.return_days,'returnable':product.returnable,'return_required':product.return_required,
        'replacement_stock':product.stock,'replacement_days':product.replacement_days,
        'refund_approval_limit_cents':world.merchants[order.merchant_id].refund_approval_limit_cents}
