"""Protocol facts, preferences and consent have objective, mutable-set consequences."""
import pytest
from usami_dialogue.models import CustomerPrivateState,CustomerIntent
from usami_dialogue.process import initialize,customer_event,consent_code,information_code,public_context


def private():
    return CustomerPrivateState(identity='buyer',opening_request='My purchase arrived damaged.',facts={'order_id':'orderA','issue':'damaged','photo_id':'photoA'},
        preference_space={'resolution':['refund','exchange']},initial_preferences={'resolution':'refund'},confirmable_actions=('commit_resolution',),protected_objects=('orderA',)).checked()


def offer(state):
    return {'operation':{'name':'commit_resolution','resource_id':'orderA','params':{'quote_id':'quoteA'}},'revision':state.revision,
        'world_hash':'worldA','quote_hash':'quote-hash','commitments':{'amount_cents':1000}}


def test_missing_information_and_private_separation():
    p=private();s=initialize(p)
    assert 'photoA' not in str(public_context(s))
    assert public_context(s)['preferences']=={}
    assert information_code(s,['order_id','issue'])=='REQUIRED_INFORMATION_MISSING'
    s=customer_event(s,p,CustomerIntent(message='This is my order.',disclosures={'order_id':'orderA','issue':'damaged'}),turn_id='t1',world_hash='worldA')
    assert information_code(s,['order_id','issue']) is None
    assert 'photoA' not in str(public_context(s))


@pytest.mark.parametrize('disclosure',[{'order_id':'made-up'},{'photo_id':'not-held'},{'oracle_answer':True}])
def test_fabricated_disclosure_is_rejected(disclosure):
    p=private()
    with pytest.raises(ValueError,match='fabricated'):customer_event(initialize(p),p,CustomerIntent(message='x',disclosures=disclosure),turn_id='t1',world_hash='worldA')


def test_consent_exact_binding_and_preference_revision():
    p=private();s=initialize(p);o=offer(s)
    assert consent_code(s,o['operation'],o['quote_hash'])=='EXPLICIT_CONSENT_REQUIRED'
    s=customer_event(s,p,CustomerIntent(message='I confirm this quoted refund.',confirm_operation=o['operation']),turn_id='t1',world_hash='worldA',offer=o,consent_check=lambda *a:True)
    assert consent_code(s,o['operation'],o['quote_hash']) is None
    changed=customer_event(s,p,CustomerIntent(message='I now prefer an exchange.',preferences={'resolution':'exchange'}),turn_id='t2',world_hash='worldA')
    assert changed.revision==1 and not changed.consents and not changed.commitments
    assert consent_code(changed,o['operation'],o['quote_hash'])=='EXPLICIT_CONSENT_REQUIRED'
    with pytest.raises(ValueError,match='current offer'):customer_event(changed,p,CustomerIntent(message='yes',confirm_operation=o['operation']),turn_id='t3',world_hash='worldA',offer=o,consent_check=lambda *a:True)


@pytest.mark.parametrize('mutation',['wrong_state','wrong_scope','no_check'])
def test_confirmation_cannot_be_manufactured_by_service(mutation):
    p=private();s=initialize(p);o=offer(s);check=lambda *a:True
    if mutation=='wrong_state':o['world_hash']='old-world'
    if mutation=='wrong_scope':o['operation']['resource_id']='other-order'
    if mutation=='no_check':check=None
    with pytest.raises(ValueError):customer_event(s,p,CustomerIntent(message='yes',confirm_operation=o['operation']),turn_id='t1',world_hash='worldA',offer=o,consent_check=check)


def test_private_fact_hash_detects_deep_mutation():
    p=private();p.facts['order_id']='other-order'
    with pytest.raises(ValueError,match='mutated'):initialize(p)


def test_dependency_proof_cannot_be_only_a_changed_error_label():
    from usami_dialogue.models import DependencyProof
    with pytest.raises(ValueError,match='actual Reference decision'):
        DependencyProof(kind='missing_information',operation={'name':'commit'},control_hash='a',changed_hash='b',
            expected_control_code='MISSING_INFO',expected_changed_code='CONSENT_REQUIRED',control_allowed=False,changed_allowed=False,reason='Neither variant permits the action').checked()
