"""Frozen profile binding and customer-side quote choices, independent of policy approval."""
from usami_harness.models import digest
from .models import Operation


def private_state(world,profile_id):return world.profiles[profile_id].checked()


def validate_private(world,private):
    private=private.checked()
    if not any(p==private for p in world.profiles.values()):raise ValueError('private state is not an independently frozen world profile')
    return private


def offer_for(world,operation):
    if operation is None:return None
    op=Operation.model_validate(operation)
    if op.name=='commit_resolution':q=world.quotes.get(op.params['quote_id'])
    elif op.name=='release_refund':
        case=world.cases.get(op.params['case_id']);q=world.quotes.get(case.agreement_quote or '') if case else None
    else:return None
    if not q or q.order_id!=op.resource_id:return None
    return {'operation':op.model_dump(mode='json'),'quote_hash':q.quote_hash,'world_hash':world.state_hash,
        'revision':world.interaction.revision,'commitments':{'option':q.option,'amount_cents':q.amount_cents,'order_id':q.order_id},
        'option':q.option,'delivery_at':q.delivery_at.isoformat() if q.delivery_at else None}


def consent_allowed(private,context,op,offer):
    # Customer fit, not a hidden Reference safety gate. An unlawful Service offer
    # may still be agreed to by an ordinary buyer and exposed by the business Oracle.
    if op['resource_id'] not in private.protected_objects or offer['option']!=context.preferences.get('resolution'):return False
    deadline=private.facts.get('deadline')
    if offer['delivery_at'] and deadline and offer['delivery_at']>deadline:return False
    return True


def intent_allowed(world,private,operation):
    op=Operation.model_validate(operation)
    if op.is_query or op.resource_id not in private.protected_objects or op.name not in private.confirmable_actions:return False
    order=world.orders.get(op.resource_id)
    if not order or order.merchant_id!=op.scope_id:return False
    if op.name=='open_case':return op.params['line_id']==private.facts['line_id'] and op.params['issue']==private.facts['issue']
    quote_id=op.params.get('quote_id')
    if quote_id:
        quote=world.quotes.get(quote_id)
        if not quote or quote.order_id!=order.id:return False
        case=world.cases.get(quote.case_id)
        if not case or case.line_id!=private.facts['line_id']:return False
    case_id=op.params.get('case_id')
    if case_id:
        case=world.cases.get(case_id)
        if not case or case.line_id!=private.facts['line_id']:return False
    return True


def normalize_intent(private,intent):
    """An exact Customer-declared operation is itself an information source."""
    from usami_dialogue.models import CustomerIntent
    data=intent.model_dump(mode='json');facts=dict(data['disclosures'])
    for raw in data['requested_operations']+([data['confirm_operation']] if data['confirm_operation'] else []):
        if raw.get('resource_id')==private.facts['order_id']:facts.setdefault('order_id',private.facts['order_id'])
        for key in ('line_id','issue','evidence_id','pickup_slot'):
            if key in raw.get('params',{}) and raw['params'][key]==private.facts.get(key):facts.setdefault(key,private.facts[key])
    data['disclosures']=facts
    return CustomerIntent.model_validate(data)


def candidate_intents(world,private):
    order=world.orders[private.facts['order_id']];base={'scope_id':order.merchant_id,'resource_id':order.id}
    result=[Operation(name='open_case',**base,params={'line_id':private.facts['line_id'],'issue':private.facts['issue']}).model_dump(mode='json')]
    for case in world.cases.values():
        if case.line_id!=private.facts['line_id']:continue
        for name,params in [('verify_evidence',{'case_id':case.id,'evidence_id':private.facts['evidence_id']}),('request_manager_review',{'case_id':case.id}),('schedule_return',{'case_id':case.id,'pickup_slot':private.facts['pickup_slot']}),('release_refund',{'case_id':case.id})]+[('quote_resolution',{'case_id':case.id,'option':o}) for o in private.preference_space['resolution']]:
            result.append(Operation(name=name,**base,params=params).model_dump(mode='json'))
    for quote in world.quotes.values():
        if quote.order_id==order.id and world.cases[quote.case_id].line_id==private.facts['line_id']:
            result.append(Operation(name='commit_resolution',**base,params={'quote_id':quote.id,'destination_channel':order.payment_channel}).model_dump(mode='json'))
    return sorted(result,key=digest)


def external_ready(world,event_id):
    event=world.external_events[event_id]
    if event_id in world.applied_events:return False
    cases=[c for c in world.cases.values() if c.line_id==event.line_id]
    if len(cases)!=1:return False
    case=cases[0]
    return case.review_requested if event.kind=='manager_review' else 'return_'+case.id in world.returns and world.returns['return_'+case.id].status=='scheduled'
