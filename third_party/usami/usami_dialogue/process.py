"""Trusted Customer event reducer. No Service tool may create disclosures or consent."""
from usami_harness.models import digest
from .models import InteractionState,ConsentRecord,CustomerIntent


def initialize(private):
    private=private.checked()
    return InteractionState(identity=private.identity)


def customer_event(state,private,intent,*,turn_id,world_hash,offer=None,consent_check=None):
    private=private.checked();intent=CustomerIntent.model_validate(intent.model_dump(mode='json'))
    if state.identity!=private.identity:raise ValueError('customer identity cannot change')
    new=state.model_copy(deep=True)
    for key,value in intent.disclosures.items():
        if key not in private.facts or value!=private.facts[key]:raise ValueError('customer fabricated or changed an objective fact')
        new.known_facts[key]=value
    changed=False
    for key,value in intent.preferences.items():
        if key not in private.preference_space or value not in private.preference_space[key]:raise ValueError('preference outside frozen profile')
        if new.preferences.get(key)!=value:changed=True
        new.preferences[key]=value
    if changed or intent.reject_offer:
        new.revision+=1;new.consents=[]
        # A superseded offer cannot commit the customer to an old choice.
        new.commitments={}
    if intent.confirm_operation is not None:
        op=intent.confirm_operation
        if changed or intent.reject_offer:raise ValueError('a changed choice needs a fresh offer before consent')
        if (not offer or op!=offer.get('operation') or offer.get('revision')!=new.revision or
                not isinstance(offer.get('quote_hash'),str) or not offer['quote_hash'] or offer.get('world_hash')!=world_hash):
            raise ValueError('confirmation must bind the actual current offer and state')
        if op.get('name') not in private.confirmable_actions or op.get('resource_id') not in private.protected_objects:
            raise ValueError('confirmation exceeds frozen customer task scope')
        if consent_check is None or not consent_check(private,new,op,offer):raise ValueError('offer violates customer constraints or business binding')
        new.consents.append(ConsentRecord(operation=op,revision=new.revision,quote_hash=offer['quote_hash'],world_hash=world_hash,turn_id=turn_id,statement=intent.message))
        new.commitments=dict(offer.get('commitments',{}))
    return new


def information_code(context,required):
    return 'REQUIRED_INFORMATION_MISSING' if any(k not in context.known_facts for k in required) else None


def consent_code(context,operation,quote_hash):
    canonical=operation.model_dump(mode='json') if hasattr(operation,'model_dump') else operation
    return None if any(c.operation==canonical and c.quote_hash==quote_hash and c.revision==context.revision for c in context.consents) else 'EXPLICIT_CONSENT_REQUIRED'


def public_context(context):
    return {'identity':context.identity,'customer_provided_facts':dict(context.known_facts),
        'preferences':dict(context.preferences),'revision':context.revision,'commitments':dict(context.commitments)}
