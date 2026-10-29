"""Ordinary passenger scope and preference/offer fit, without hidden rule answers."""
from datetime import datetime
from usami_harness.models import digest
from .models import Operation


def private_state(world,profile_id):return world.profiles[profile_id].checked()
def validate_private(world,private):
    private=private.checked()
    if not any(p==private for p in world.profiles.values()):raise ValueError('Customer profile differs from the frozen airline world')
    return private


def intent_allowed(world,private,operation):
    op=Operation.model_validate(operation);b=world.bookings.get(op.resource_id)
    if op.is_query or not b or b.id not in private.protected_objects or b.passenger_id!=private.identity or b.carrier_id!=op.scope_id or op.name not in private.confirmable_actions:return False
    if op.name=='commit_quote':return op.params['quote_id'] in world.quotes and world.quotes[op.params['quote_id']].booking_id==b.id
    if op.name=='verify_passenger':return op.params['document_token']==private.facts['document_token']
    if op.name=='quote_change':return op.params['flights'] in private.preference_space['itinerary']
    if op.name=='quote_baggage':return op.params['extra_kg']==private.facts['baggage_kg']
    return True


def normalize_intent(private,intent):
    from usami_dialogue.models import CustomerIntent
    raw=intent.model_dump(mode='json');facts=raw['disclosures']
    for op in raw['requested_operations']+([raw['confirm_operation']] if raw['confirm_operation'] else []):
        if op.get('resource_id')==private.facts['booking_id']:facts.setdefault('booking_id',private.facts['booking_id'])
        if op.get('params',{}).get('document_token')==private.facts['document_token']:facts.setdefault('document_token',private.facts['document_token'])
        if op.get('params',{}).get('extra_kg')==private.facts['baggage_kg']:facts.setdefault('baggage_kg',private.facts['baggage_kg'])
    return CustomerIntent.model_validate(raw)


def offer_for(world,operation):
    if operation is None:return None
    op=Operation.model_validate(operation)
    if op.name!='commit_quote':return None
    q=world.quotes.get(op.params['quote_id'])
    if not q or q.booking_id!=op.resource_id:return None
    flights=[world.flights[f] for f in q.flights]
    return {'operation':op.model_dump(mode='json'),'quote_hash':q.quote_hash,'world_hash':world.state_hash,'revision':world.interaction.revision,
        'action':q.action,'flights':q.flights,'charge_cents':q.charge_cents,'refund_cents':q.refund_cents,'baggage_kg':q.baggage_kg,
        'departure':flights[0].departure.isoformat(),'arrival':flights[-1].arrival.isoformat(),'destination':flights[-1].destination,
        'commitments':{'action':q.action,'flights':q.flights,'charge_cents':q.charge_cents,'refund_cents':q.refund_cents}}


def consent_allowed(private,context,operation,offer):
    if operation['resource_id'] not in private.protected_objects or offer['action']!=context.preferences.get('service'):return False
    if offer['charge_cents']>private.facts['max_fee_cents']:return False
    if offer['action']=='rebook':return (offer['flights']==context.preferences.get('itinerary') and offer['destination']==private.facts['destination'] and datetime.fromisoformat(offer['departure'])>=datetime.fromisoformat(private.facts['earliest_departure']) and datetime.fromisoformat(offer['arrival'])<=datetime.fromisoformat(private.facts['latest_arrival']))
    return offer['action']!='baggage' or offer['baggage_kg']==private.facts['baggage_kg']


def candidate_intents(world,private):
    b=world.bookings[private.facts['booking_id']];base={'scope_id':b.carrier_id,'resource_id':b.id}
    inputs=[('verify_passenger',{'document_token':private.facts['document_token']}),('quote_refund',{}),('quote_baggage',{'extra_kg':private.facts['baggage_kg']})]
    inputs += [('quote_change',{'flights':flights,'cabin':cabin}) for flights in private.preference_space['itinerary'] for cabin in ('economy','business')]
    inputs += [('commit_quote',{'quote_id':q.id}) for q in world.quotes.values() if q.booking_id==b.id]
    return sorted([Operation(name=n,**base,params=p).model_dump(mode='json') for n,p in inputs],key=digest)


def external_ready(world,event_id):return event_id not in world.applied_events
