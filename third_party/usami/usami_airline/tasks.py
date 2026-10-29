"""Native ticketing goals; fees, seats and quote consent remain executable rules."""
from usami_dialogue.tasks import DialogueTask,Goal,FrameRule,RefusalGoal,checked_task
from .customer import private_state
from .models import Operation


def frames(world,private,*,refusal_only=False):
    bid=private.facts['booking_id']
    business=[FrameRule(collection='bookings',selector_value=bid,mutable_fields=['flights','cabin','status','fare_value_cents','retained_fees_cents','baggage_paid_cents','extra_baggage_kg','revision']),
        FrameRule(collection='verifications',selector_field='booking_id',selector_value=bid,allow_create=True),
        FrameRule(collection='quotes',selector_field='booking_id',selector_value=bid,mutable_fields=['status'],allow_create=True),
        FrameRule(collection='ledger',selector_field='booking_id',selector_value=bid,allow_create=True)]
    if refusal_only:business=[FrameRule(collection='verifications',selector_field='booking_id',selector_value=bid,allow_create=True)]
    return business+[
        FrameRule(collection='flights',selector_value=e.flight_id,mutable_fields=['status','departure','arrival']) for e in world.external_events.values() if e.flight_id in world.bookings[bid].flights]


def compose(world,profile_id,*,family='rebook',mixed=False,external_events=False,refusal_operation=None,refusal_code=None):
    from . import adapter
    p=private_state(world,profile_id);b=world.bookings[p.facts['booking_id']]
    pure_refusal=family.startswith('refuse_')
    if pure_refusal:
        if mixed or refusal_operation is None or refusal_code is None:raise ValueError('policy refusal needs an exact real operation and native code')
        goals=[]
    elif family=='rebook':goals=[Goal(collection='bookings',entity=b.id,field='flights',accepted_values=[f for f in p.preference_space['itinerary'] if f!=b.flights])]
    elif family=='refund':goals=[Goal(collection='bookings',entity=b.id,field='status',accepted_values=['cancelled'])]
    elif family=='baggage':goals=[Goal(collection='bookings',entity=b.id,field='extra_baggage_kg',accepted_values=[p.facts['baggage_kg']])]
    else:raise ValueError('unregistered airline task family')
    if mixed and family=='rebook':raise ValueError('rebooking has no general automatic duplicate-refusal premise')
    refusal=[RefusalGoal(operation=refusal_operation,code=refusal_code)] if pure_refusal else []
    if mixed:
        op=Operation(name='quote_refund' if family=='refund' else 'quote_baggage',scope_id=b.carrier_id,resource_id=b.id,params={} if family=='refund' else {'extra_kg':p.facts['baggage_kg']})
        refusal=[RefusalGoal(operation=op.model_dump(mode='json'),code='BOOKING_CANCELLED' if family=='refund' else 'BAGGAGE_ALREADY_PURCHASED')]
    task=DialogueTask(domain=world.domain,rules_version=world.version,world_hash=world.state_hash,profile_id=profile_id,private_hash=p.private_hash,
        family=family,goals=goals,refusals=refusal,frames=frames(world,p,refusal_only=pure_refusal),external_schedule=[e.id for e in world.external_events.values() if external_events and e.flight_id in b.flights])
    return checked_task(adapter,world,task)


def validate_task(world,private,task):
    b=world.bookings[private.facts['booking_id']]
    values={'rebook':('flights',[f for f in private.preference_space['itinerary'] if f!=b.flights]),'refund':('status',['cancelled']),'baggage':('extra_baggage_kg',[private.facts['baggage_kg']])}
    pure_refusal=task.family in {'refuse_rebook','refuse_refund','refuse_baggage'}
    if pure_refusal:
        if task.goals or len(task.refusals)!=1:raise ValueError('policy refusal requires one exact native probe')
        r=task.refusals[0];operation=Operation.model_validate(r.operation)
        expected_name={'refuse_rebook':'quote_change','refuse_refund':'quote_refund','refuse_baggage':'quote_baggage'}[task.family]
        from .customer import intent_allowed
        allowed_codes={'FARE_REFUND_FORBIDDEN','FARE_CHANGE_FORBIDDEN','REFUND_WINDOW_CLOSED','CHANGE_WINDOW_CLOSED','CUSTOMER_CONSTRAINT_UNMET','ITINERARY_UNAVAILABLE','CONNECTION_INFEASIBLE','BAGGAGE_LIMIT_EXCEEDED','PAYMENT_LIMIT_EXCEEDED','NO_ITINERARY_CHANGE'}
        if operation.name!=expected_name or not intent_allowed(world,private,r.operation) or r.code not in allowed_codes:raise ValueError('refusal operation/code outside native passenger scope')
    else:
        if task.family not in values or len(task.goals)!=1:raise ValueError('unregistered airline family/goals')
        field,accepted=values[task.family];g=task.goals[0]
        if g.collection!='bookings' or g.entity!=b.id or g.field!=field or g.accepted_values!=accepted:raise ValueError('goal exceeds the frozen passenger task scope')
    if task.frames!=frames(world,private,refusal_only=pure_refusal):raise ValueError('frame differs from native airline contract')
    if len(task.refusals)>1 or (task.refusals and task.family=='rebook'):raise ValueError('unsupported airline refusal goal')
    if task.refusals and not pure_refusal:
        expected=Operation(name='quote_refund' if task.family=='refund' else 'quote_baggage',scope_id=b.carrier_id,resource_id=b.id,params={} if task.family=='refund' else {'extra_kg':private.facts['baggage_kg']})
        if task.refusals[0].operation!=expected.model_dump(mode='json') or task.refusals[0].code!=('BOOKING_CANCELLED' if task.family=='refund' else 'BAGGAGE_ALREADY_PURCHASED'):raise ValueError('refusal goal is not a real native duplicate boundary')
    if any(world.external_events[e].flight_id not in b.flights for e in task.external_schedule if e in world.external_events):raise ValueError('external schedule outside this booking')


def search(world,task,*,max_nodes=1024,max_depth=12,max_paths=2):
    from . import adapter
    from usami_dialogue.planner import search as native_search
    p=private_state(world,task.profile_id)
    service=task.family.removeprefix('refuse_')
    if task.family=='refuse_rebook':itineraries=[task.refusals[0].operation['params']['flights']]
    else:itineraries=p.preference_space['itinerary'] if service=='rebook' else [world.bookings[p.facts['booking_id']].flights]
    options=[{'service':service,'itinerary':route} for route in itineraries]
    return native_search(adapter,world,task,options,max_nodes=max_nodes,max_depth=max_depth,max_paths=max_paths)


def certified_tasks(world,profile_id,*,family='rebook',mixed=False,external_events=False,max_nodes=1024,max_paths=2,refusal_operation=None,refusal_code=None):
    from . import adapter
    from usami_dialogue.planner import certified_paths
    task=compose(world,profile_id,family=family,mixed=mixed,external_events=external_events,refusal_operation=refusal_operation,refusal_code=refusal_code)
    result=search(world,task,max_nodes=max_nodes,max_paths=max_paths)
    return task,result,certified_paths(adapter,world,task,result)


def customer_task_brief(world,private,task):
    import json
    # No expected outcome, policy code, discovery label or witness is exposed.
    prefix='Your booking is '+private.facts['booking_id']+'. '
    if task.family.startswith('refuse_'):
        request=task.refusals[0].operation
        return prefix+'You want this specific service request: '+json.dumps({'operation':request['name'],'parameters':request['params']},sort_keys=True)+'. Discuss this request and respond to real explanations. Keep this request as the goal rather than switching to a different service.'
    if task.family=='rebook':brief='You want a different itinerary to the stated destination, within your frozen time and fee constraints. Choose from acceptable alternatives and compare actual offers.'
    elif task.family=='refund':brief='You want a refund for this booking. Discuss the actual fare terms and confirm only a real offer that fits your needs.'
    else:brief='You want the additional baggage quantity in your frozen facts, within your actual fee limit. Obtain the actual terms before confirming.'
    if task.refusals:brief+=' You also want to request this same service again after it has been fulfilled; respond to the real explanation.'
    return prefix+brief


def task_intent_allowed(private,task,intent):
    service=task.family.removeprefix('refuse_')
    if 'service' in intent.preferences and intent.preferences['service']!=service:return False
    allowed_itineraries=task.goals[0].accepted_values if task.family=='rebook' else [task.refusals[0].operation['params']['flights']] if task.family=='refuse_rebook' else private.preference_space['itinerary']
    if 'itinerary' in intent.preferences and intent.preferences['itinerary'] not in allowed_itineraries:return False
    mapping={'quote_change':'rebook','quote_refund':'refund','quote_baggage':'baggage'}
    return all(op.get('name') not in mapping or mapping[op['name']]==service for op in intent.requested_operations)
