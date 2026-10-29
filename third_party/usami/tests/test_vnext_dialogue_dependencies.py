"""Counterfactuals must change the native allowed-action set and bind real traces."""
import pytest
from usami_dialogue.dependencies import certify_dependencies,verify_dependency
from test_vnext_aftersales import prepared,quoted,commit,operation,customer


def test_refund_proves_missing_information_preference_and_consent_without_turn_reward():
    s,order,line=prepared();cid,q=quoted(s,order,line);commit(s,order,q)
    trace=s.finish();proofs=certify_dependencies(s.adapter,trace)
    kinds={p.proof.kind for p in proofs}
    assert {'missing_information','preference_resolution','explicit_consent'}<=kinds
    assert all(p.proof.control_allowed!=p.proof.changed_allowed for p in proofs)
    assert all(verify_dependency(s.adapter,trace,p) for p in proofs)
    consent=next(p for p in proofs if p.proof.kind=='explicit_consent')
    assert consent.proof.expected_changed_code=='EXPLICIT_CONSENT_REQUIRED'
    # Re-sealing hashes does not authorize an evaluator to change business facts.
    tampered=consent.model_copy(deep=True)
    tampered.changed_world['orders'][order.id]['paid_cents']+=1
    tampered.certificate_hash=tampered.computed_hash()
    with pytest.raises(ValueError,match='exceed'):verify_dependency(s.adapter,trace,tampered)


def test_redundant_fact_is_not_evidence_of_dialogue_necessity():
    s,order,line=prepared();cid,q=quoted(s,order,line);commit(s,order,q)
    proofs=certify_dependencies(s.adapter,s.finish())
    # Deadline disclosure cannot be claimed necessary for a cash refund.
    assert not any(p.proof.kind=='missing_information' and p.intervention=={'fact':'deadline'} for p in proofs)


def test_external_manager_event_changes_actual_reference_choices():
    from usami_aftersales import adapter
    from usami_aftersales.rules import eligibility
    from usami_dialogue.engine import DialogueSession
    selected=None
    for seed in range(10):
        world=adapter.generate_world(seed)
        for pid,p in world.profiles.items():
            line=world.lines[p.facts['line_id']];order=world.orders[line.order_id]
            event=world.external_events['manager_review_'+line.id]
            if not eligibility(world,order,line) and world.evidence[p.facts['evidence_id']].simulated_valid and event.outcome=='approved' and line.quantity*line.unit_price_cents>world.merchants[order.merchant_id].refund_approval_limit_cents:
                selected=(world,pid,order,line);break
        if selected:break
    assert selected is not None
    world,pid,order,line=selected;s=DialogueSession(adapter,world,pid)
    opening=operation(s,'open_case',order,line_id=line.id,issue=s.private.facts['issue'])
    customer(s,opening,preferences={'resolution':'refund'})
    cid=s.act(opening)['data']['case_id']
    verify=operation(s,'verify_evidence',order,case_id=cid,evidence_id=s.private.facts['evidence_id'])
    review=operation(s,'request_manager_review',order,case_id=cid)
    customer(s,verify,review);assert s.act(verify)['success'];assert s.act(review)['success']
    s.external('manager_review_'+line.id)
    trace=s.finish();proofs=certify_dependencies(adapter,trace)
    state=next(p for p in proofs if p.proof.kind=='state_update' and p.proof.operation['name']=='quote_resolution')
    assert not state.proof.control_allowed and state.proof.changed_allowed
    assert state.proof.expected_control_code=='APPROVAL_REQUIRED'
    assert verify_dependency(adapter,trace,state)


def test_actual_later_choice_cancels_a_prior_cross_turn_commitment():
    from test_vnext_education import quoted,operation as edu_operation,customer as edu_customer,SLOT
    s,cid,q=quoted();e=s.state.enrollments[s.private.facts['enrollment_id']]
    op=edu_operation(s,'commit_quote',quote_id=q['id'],payment_channel=e.payment_channel)
    s.reply('Do you confirm this actual time and study-credit reservation?',op.model_dump(mode='json'))
    edu_customer(s,confirm_operation=op.model_dump(mode='json'))
    assert s.adapter.decide(s.state,s.private.identity,op).allowed
    edu_customer(s,preferences={'slot_id':next(x for x in s.private.preference_space['slot_id'] if x!=SLOT)})
    assert s.adapter.decide(s.state,s.private.identity,op).error_code=='STALE_CUSTOMER_CHOICE'
    trace=s.finish();proofs=certify_dependencies(s.adapter,trace)
    c=next(p for p in proofs if p.proof.kind=='cross_turn_commitment')
    assert c.proof.control_allowed and not c.proof.changed_allowed and verify_dependency(s.adapter,trace,c)
    assert c.proof.expected_changed_code=='STALE_CUSTOMER_CHOICE'
    tampered=c.model_copy(deep=True);tampered.intervention['prior_consent_turn_id']='invented';tampered.certificate_hash=tampered.computed_hash()
    with pytest.raises(ValueError,match='prior consent'):verify_dependency(s.adapter,trace,tampered)
    assert not trace.final_world['appointments']


def test_mixed_goal_has_actual_state_transition_and_correct_refusal_dependency():
    from usami_education import adapter
    from usami_education.tasks import certified_tasks
    w=adapter.generate_world(0);task,search,rows=certified_tasks(w,'profile_u1_enrollment1',family='refund',mixed=True,max_nodes=128,max_paths=1)
    assert rows
    trace=rows[0]['trace'];proofs=rows[0]['certificate'].dependencies
    c=next(p for p in proofs if p.proof.kind=='partial_completion')
    assert c.proof.operation['name']=='quote_refund' and c.proof.expected_changed_code=='ENROLLMENT_REFUNDED'
    assert c.proof.control_allowed and not c.proof.changed_allowed and verify_dependency(adapter,trace,c)
    tampered=c.model_copy(deep=True);tampered.intervention['refusal_event_id']='dialogue_0';tampered.certificate_hash=tampered.computed_hash()
    with pytest.raises(ValueError,match='later correct refusal'):verify_dependency(adapter,trace,tampered)


def test_a_completion_without_an_actual_refusal_does_not_get_partial_credit():
    s,order,line=prepared();cid,q=quoted(s,order,line);commit(s,order,q)
    assert not any(p.proof.kind=='partial_completion' for p in certify_dependencies(s.adapter,s.finish()))
