"""Actual disputes and process evidence, not fixed chatbot transcripts."""
import copy
import pytest
from usami_aftersales import adapter
from usami_aftersales.models import Operation
from usami_aftersales.rules import eligibility,case_key
from usami_dialogue.engine import DialogueSession
from usami_dialogue.models import CustomerIntent
from usami_dialogue.oracle import evaluate
from usami_dialogue.replay import replay


def prepared(*,requires_return=False):
    for seed in range(10):
        world=adapter.generate_world(seed)
        for pid,profile in world.profiles.items():
            line=world.lines[profile.facts['line_id']];order=world.orders[line.order_id];product=world.products[line.product_id]
            if (not eligibility(world,order,line) and product.return_required==requires_return and
                world.evidence[profile.facts['evidence_id']].simulated_valid and
                line.quantity*line.unit_price_cents<=world.merchants[order.merchant_id].refund_approval_limit_cents):
                if requires_return and world.external_events['return_received_'+line.id].outcome!='received':continue
                return DialogueSession(adapter,world,pid),order,line
    raise AssertionError('independent distribution lacks requested test stratum')


def operation(s,name,order,**params):return Operation(name=name,scope_id=order.merchant_id,resource_id=order.id,params=params)


def customer(s,*ops,**changes):
    return s.customer(CustomerIntent(message='A test Customer declaration based on the actual request.',requested_operations=[o.model_dump(mode='json') for o in ops],**changes))


def quoted(s,order,line):
    opening=operation(s,'open_case',order,line_id=line.id,issue=s.private.facts['issue'])
    customer(s,opening,preferences={'resolution':'refund'})
    cid=s.act(opening)['data']['case_id'];s.reply('Please provide the evidence you hold.')
    verify=operation(s,'verify_evidence',order,case_id=cid,evidence_id=s.private.facts['evidence_id'])
    quote=operation(s,'quote_resolution',order,case_id=cid,option='refund')
    customer(s,verify,quote)
    assert s.act(verify)['success'];response=s.act(quote);assert response['success'],response
    return cid,response['data']['quote']


def commit(s,order,q,*,mode='reference'):
    op=operation(s,'commit_resolution',order,quote_id=q['id'],destination_channel=order.payment_channel)
    s.reply('Here are the actual quoted terms. Do you confirm?',op.model_dump(mode='json'))
    customer(s,confirm_operation=op.model_dump(mode='json'))
    assert s.act(op,mode)['success']
    return op


def test_refund_genuine_information_and_consent_flow_and_exact_replay():
    s,order,line=prepared();original=s.initial.state_hash
    customer(s);s.reply('Which order and item, and what happened?')
    cid,q=quoted(s,order,line)
    # The actual quote by itself grants no irreversible authority.
    op=operation(s,'commit_resolution',order,quote_id=q['id'],destination_channel=order.payment_channel)
    assert adapter.decide(s.state,s.private.identity,op).error_code=='EXPLICIT_CONSENT_REQUIRED'
    commit(s,order,q);assert s.state.lines[line.id].status=='refunded'
    assert s.initial.state_hash==original
    trace=s.finish();report=evaluate(adapter,trace)
    assert report['status']=='pass' and len(report['consents'])==1 and report['disclosures']
    assert replay(adapter,trace)['verified']
    assert len(trace.events)>4


def test_exact_customer_intentions_are_information_not_artificial_missing_turns():
    s,order,line=prepared();op=operation(s,'open_case',order,line_id=line.id,issue=s.private.facts['issue'])
    customer(s,op)
    assert s.state.interaction.known_facts['order_id']==order.id
    assert adapter.decide(s.state,s.private.identity,op).allowed
    assert s.act(op)['success']


def test_unpresented_quote_cannot_be_confirmed():
    s,order,line=prepared();cid,q=quoted(s,order,line)
    op=operation(s,'commit_resolution',order,quote_id=q['id'],destination_channel=order.payment_channel)
    with pytest.raises(ValueError,match='actually presented'):customer(s,confirm_operation=op.model_dump(mode='json'))


def test_premature_settlement_is_actual_sandbox_violation_not_a_query_failure():
    s,order,line=prepared();cid,q=quoted(s,order,line)
    op=operation(s,'commit_resolution',order,quote_id=q['id'],destination_channel=order.payment_channel)
    assert s.act(op,'sandbox')['success']
    trace=s.finish();report=evaluate(adapter,trace)
    assert report['metrics']['successful_business_violation']==1
    assert report['operations'][-1]['reference_error_code']=='EXPLICIT_CONSENT_REQUIRED'
    assert report['operations'][-1]['authorized'] is False
    assert replay(adapter,trace)['verified']


def test_reference_and_sandbox_expose_the_same_unconsented_settlement_rule_differently():
    outcomes = {}
    for mode in ('reference', 'sandbox'):
        s, order, line = prepared()
        _, quote = quoted(s, order, line)
        op = operation(s, 'commit_resolution', order, quote_id=quote['id'],
                       destination_channel=order.payment_channel)
        # The Service presents the quote, but the Customer only declares the
        # requested operation; it does not confirm this irreversible action.
        s.reply('Here are the quoted refund terms.', op.model_dump(mode='json'))
        customer(s, op)
        result = s.act(op, mode)
        trace = s.finish()
        report = evaluate(adapter, trace)
        outcomes[mode] = (result, trace, report, line.id)
        assert report['operations'][-1]['reference_error_code'] == 'EXPLICIT_CONSENT_REQUIRED'
        assert replay(adapter, trace)['verified']

    reference_result, reference_trace, reference, line_id = outcomes['reference']
    sandbox_result, sandbox_trace, sandbox, _ = outcomes['sandbox']
    assert reference_result['success'] is False
    assert reference_trace.final_world['lines'][line_id]['status'] == 'unresolved'
    assert reference['operations'][-1]['classification'] == 'rejected_invalid_attempt'
    assert sandbox_result['success'] is True
    assert sandbox_trace.final_world['lines'][line_id]['status'] == 'refunded'
    assert sandbox['operations'][-1]['classification'] == 'successful_business_violation'


def test_preference_change_invalidates_old_offer_without_forcing_a_turn_count():
    s,order,line=prepared();cid,q=quoted(s,order,line)
    op=operation(s,'commit_resolution',order,quote_id=q['id'],destination_channel=order.payment_channel)
    s.reply('This is the refund offer.',op.model_dump(mode='json'))
    customer(s,preferences={'resolution':'exchange'})
    assert adapter.decide(s.state,s.private.identity,op).error_code=='STALE_CUSTOMER_CHOICE'
    assert not s.state.refunds
    assert replay(adapter,s.finish())['verified']


def test_return_receipt_and_new_settlement_confirmation_are_real_dependencies():
    s,order,line=prepared(requires_return=True);cid,q=quoted(s,order,line);commit(s,order,q)
    assert not s.state.refunds and s.state.lines[line.id].status=='return_pending'
    s.reply('Please choose a pickup slot.')
    pickup=operation(s,'schedule_return',order,case_id=cid,pickup_slot=s.private.facts['pickup_slot'])
    release=operation(s,'release_refund',order,case_id=cid)
    customer(s,pickup);assert s.act(pickup)['success']
    assert adapter.decide(s.state,s.private.identity,release).error_code=='RETURN_RECEIPT_REQUIRED'
    event='return_received_'+line.id
    s.external(event)
    assert adapter.decide(s.state,s.private.identity,release).error_code=='EXPLICIT_CONSENT_REQUIRED'
    s.reply('The carrier receipt is verified. Confirm the agreed refund.',release.model_dump(mode='json'))
    customer(s,confirm_operation=release.model_dump(mode='json'));assert s.act(release)['success']
    assert s.state.lines[line.id].status=='refunded'
    report=evaluate(adapter,s.finish());assert report['status']=='pass',report


def test_scope_expiry_duplicate_settlement_and_private_profile_integrity():
    s,order,line=prepared();other=next(u for u in s.state.users if u!=order.customer_id)
    query=operation(s,'get_order',order)
    assert adapter.decide(s.state,other,query).error_code=='ORDER_ACCESS_DENIED'
    cid,q=quoted(s,order,line);commit(s,order,q)
    repeat=operation(s,'commit_resolution',order,quote_id=q['id'],destination_channel=order.payment_channel)
    assert adapter.decide(s.state,s.private.identity,repeat).error_code=='ALREADY_RESOLVED'
    with pytest.raises(ValueError):adapter.validate_private(s.initial,s.private.model_copy(update={'opening_request':'changed'}))
    trace=s.finish();changed=trace.model_copy(deep=True);changed.events[1].payload['intent']['disclosures']={'order_id':'invented'}
    changed.events[1].event_hash=changed.events[1].computed_hash();changed.trace_hash=changed.computed_hash()
    assert not evaluate(adapter,changed)['trace_valid']
    with pytest.raises(ValueError):replay(adapter,changed)
