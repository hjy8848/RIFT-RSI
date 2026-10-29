"""A native blocked attempt is not evidence of a Service decision to refuse."""
import pytest
from usami_dialogue.oracle import evaluate
from usami_dialogue.replay import replay
from test_vnext_aftersales import prepared,operation,customer,quoted


def test_service_false_refusal_is_scored_without_inventing_business_event():
    s,order,line=prepared()
    op=operation(s,'open_case',order,line_id=line.id,issue=s.private.facts['issue'])
    customer(s,op)
    assert s.adapter.decide(s.state,s.private.identity,op).allowed
    s.reply('I cannot help.',refused_operation=op.model_dump(mode='json'),refusal_code='ORDER_ACCESS_DENIED')
    trace=s.finish();report=evaluate(s.adapter,trace)
    assert report['trace_valid'] and report['status']=='fail'
    assert report['metrics']['false_rejection']==1
    assert report['operations']==[] and not s.state.cases
    assert replay(s.adapter,trace)['verified']


@pytest.mark.parametrize('wrong_reason',[False,True])
def test_exact_customer_probe_correct_refusal_and_wrong_reason(wrong_reason):
    s,order,line=prepared();cid,q=quoted(s,order,line)
    op=operation(s,'commit_resolution',order,quote_id=q['id'],destination_channel=order.payment_channel)
    # A request for settlement is not consent to the actual presented quote.
    customer(s,op)
    assert s.adapter.decide(s.state,s.private.identity,op).error_code=='EXPLICIT_CONSENT_REQUIRED'
    s.reply('Please confirm first.',refused_operation=op.model_dump(mode='json'),
        refusal_code='ORDER_ACCESS_DENIED' if wrong_reason else 'EXPLICIT_CONSENT_REQUIRED')
    trace=s.finish();report=evaluate(s.adapter,trace)
    assert report['trace_valid'] and report['status']==('fail' if wrong_reason else 'pass')
    assert report['service_decisions'][-1]['classification']==('incorrect_refusal_reason' if wrong_reason else 'correct_refusal')
    assert report['metrics']['correct_refusal']==(0 if wrong_reason else 1)
    assert not s.state.refunds
    assert replay(s.adapter,trace)['verified']


def test_attempt_then_claim_and_unrequested_refusal_do_not_earn_correct_refusal():
    s,order,line=prepared();cid,q=quoted(s,order,line)
    op=operation(s,'commit_resolution',order,quote_id=q['id'],destination_channel=order.payment_channel)
    customer(s,op);assert not s.act(op)['success']
    s.reply('I refused.',refused_operation=op.model_dump(mode='json'),refusal_code='EXPLICIT_CONSENT_REQUIRED')
    report=evaluate(s.adapter,s.finish())
    assert report['metrics']['rejected_invalid_attempt']==1
    assert report['metrics']['correct_refusal']==0
    assert report['service_decisions'][-1]['classification']=='unattributed_refusal'
