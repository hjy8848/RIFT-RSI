"""After-sales V3 wire protocol tests over the unchanged native Engine/Oracle."""
import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from test_vnext_aftersales import prepared, operation
from test_vast_usami import answer, call
from usami.service.models import ModelSettings
from usami_harness.models import Budget
from usami_aftersales import adapter
from usami_aftersales.rules import case_key, quote_key
from usami_aftersales.tasks import frame_contract
from usami_dialogue.agents import AgentExecution, replay as replay_v2, run as run_v2
from usami_dialogue.agents_v3 import (
    TaskRunConfigV3, _candidate_actions, _customer_business_view, customer_turn_schema, replay_v3,
    run_v3, service_response_schema_v3,
)
from usami_dialogue.engine import DialogueSession
from usami_dialogue.models import CustomerIntent, DialogueEvent
from usami_dialogue.tasks import (
    DialogueComposition, DialogueTask, Goal, InformationRequirement,
    PreferenceRequirement, certify, checked_task,
)
from usami_dialogue.planner import search_composed, execute_witness
from experiments.vast_coevo.examples.after_sales_service_v3_diagnostic import FixedCompliantCustomer


MODEL = ModelSettings(model='recorded-v3-fixture', base_url='https://fixture.invalid/v1',
                      options={'temperature': 0})


def _task_bundle(*, requires_return=False, mode='reference', max_recoveries=1):
    source, order, line = prepared(requires_return=requires_return)
    world = source.initial
    profile = source.private
    info = ['order_id', 'line_id', 'issue', 'evidence_id']
    if requires_return:
        info.append('pickup_slot')
    composition = DialogueComposition(
        protocol='open-dialogue-composition-v2',
        profile_ids=[source.profile_id],
        customer_request='I want a refund for the damaged item. I can return it if needed.',
        information_requirements=[InformationRequirement(profile_id=source.profile_id, fact_keys=info)],
        preference_requirements=[PreferenceRequirement(profile_id=source.profile_id,
            preference='resolution', accepted_values=['refund'])],
    )
    task = DialogueTask(
        domain='aftersales', rules_version=world.version, world_hash=world.state_hash,
        profile_id=source.profile_id, private_hash=profile.private_hash,
        family='open_composition_v1',
        goals=[Goal(collection='lines', entity=line.id, field='status', accepted_values=['refunded'])],
        frames=frame_contract(world, profile),
        external_schedule=[event.id for event in world.external_events.values() if event.line_id == line.id],
        require_dependency=True, composition=composition,
    )
    task = checked_task(adapter, world, task)
    search = search_composed(adapter, world, task, max_nodes=1800, max_depth=16, max_paths=1)
    assert search['paths'], search
    witness = execute_witness(adapter, world, task, search['paths'][0])
    certificate = certify(adapter, world, task, witness)
    config = TaskRunConfigV3(
        profile_id=task.profile_id, initial_world_hash=world.state_hash, mode=mode,
        tools=list(adapter.TOOLS),
        budget=Budget(max_customer_turns=8, max_customer_llm_calls=10,
            max_service_llm_calls=16, max_service_tool_calls=24,
            max_format_recoveries=max_recoveries),
        external_schedule=task.external_schedule, task=task, certificate=certificate, witness=witness,
    )
    return world, source, order, line, config


class ScriptedClient:
    def __init__(self, responses):
        self.settings = MODEL.model_copy(deep=True)
        self.responses = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(copy.deepcopy(request))
        if not self.responses:
            raise AssertionError('scripted completion exhausted')
        row = self.responses.pop(0)
        return row(request) if callable(row) else copy.deepcopy(row)


def _customer_tool(identifier, value):
    return answer(calls=[call(identifier, 'submit_customer_turn_v3', value)])


def _service_tool(identifier, name, resource_id, params, scope_id):
    return call(identifier, name, {'scope_id': scope_id, 'resource_id': resource_id, 'params': params})


def _reply(identifier, message, offer_ref=None, refused_operation=None, refusal_code=None):
    return answer(calls=[call(identifier, 'respond_to_customer_v3', {
        'message': message, 'offer_ref': offer_ref,
        'refused_operation': refused_operation, 'refusal_code': refusal_code,
    })])


def _customer_policy_turn(request, *, profile, actions=(), preference=None, disclosures=None,
                          offer_response='none', message='I am responding to the current request.', finish=False):
    observation = json.loads(request['messages'][-1]['content'])
    selected = []
    for selector in actions:
        matches = [row for row in observation['available_actions']
                   if row['action'] == selector[0] and all(row['params'].get(k) == v for k, v in selector[1].items())]
        assert len(matches) == 1, (selector, observation['available_actions'])
        selected.append(matches[0]['id'])
    return _customer_tool('customer_' + str(len(request['messages'])), {
        'message': message,
        'disclosures': dict(disclosures or {}),
        'preferences': {'resolution': preference} if preference else {},
        'authorized_action_ids': selected,
        'offer_response': offer_response,
        'finish': finish,
    })


def _run_scripts(*, requires_return=False, mode='reference', customer_modes=None, final_service_action=None,
                 max_recoveries=1):
    world, source, order, line, config = _task_bundle(requires_return=requires_return, mode=mode,
                                                       max_recoveries=max_recoveries)
    profile = source.private
    cid = case_key(order.id, line.id, profile.facts['issue'])
    qid = quote_key(cid, 'refund', 1)
    open_case = operation(source, 'open_case', order, line_id=line.id, issue=profile.facts['issue'])
    verify = operation(source, 'verify_evidence', order, case_id=cid, evidence_id=profile.facts['evidence_id'])
    quote = operation(source, 'quote_resolution', order, case_id=cid, option='refund')
    commit = operation(source, 'commit_resolution', order, quote_id=qid, destination_channel=order.payment_channel)
    action_sets = [
        [('open_case', {'line_id': line.id})],
        [('verify_evidence', {'case_id': cid}), ('quote_resolution', {'case_id': cid, 'option': 'refund'})],
    ]
    customer_rows = [
        lambda request: _customer_policy_turn(request, profile=profile, actions=action_sets[0],
            preference='refund', disclosures=profile.facts,
            message='I want a refund for this item. I authorize opening the case.'),
        lambda request: _customer_policy_turn(request, profile=profile, actions=action_sets[1],
            message='I authorize evidence verification and preparation of a refund quote; show me the actual terms before settlement.'),
        lambda request: _customer_policy_turn(request, profile=profile, offer_response='accept',
            message='I reviewed the presented refund amount and explicitly accept this offer.'),
    ]
    service_rows = [
        answer(calls=[_service_tool('service_get_order', 'get_order', order.id, {}, order.merchant_id)]),
        answer(calls=[_service_tool('service_open_case', 'open_case', order.id,
            {'line_id': line.id, 'issue': profile.facts['issue']}, order.merchant_id)]),
        _reply('service_ask_evidence', 'I opened the case. I will verify the evidence and prepare the requested refund quote.'),
        answer(calls=[_service_tool('service_verify', 'verify_evidence', order.id,
            {'case_id': cid, 'evidence_id': profile.facts['evidence_id']}, order.merchant_id),
            _service_tool('service_quote', 'quote_resolution', order.id,
                {'case_id': cid, 'option': 'refund'}, order.merchant_id)]),
        _reply('service_present_quote', 'The verified refund quote is ready. Please review the amount and confirm if you agree.',
               {'kind': 'commit_resolution', 'quote_id': qid}),
    ]
    if final_service_action == 'unauthorized_commit':
        customer_rows[2] = lambda request: _customer_policy_turn(request, profile=profile,
            offer_response='none', message='I need time to review the terms; I am not confirming the settlement.')
        customer_rows.append(lambda request: _customer_policy_turn(request, profile=profile, finish=True,
            message='I am ending the interaction without accepting the offer.'))
        service_rows.extend([
            answer(calls=[_service_tool('service_bad_commit', 'commit_resolution', order.id,
                {'quote_id': qid, 'destination_channel': order.payment_channel}, order.merchant_id)]),
            _reply('service_bad_commit_reply', 'The request was processed.'),
        ])
        customer_rows.append(lambda request: _customer_policy_turn(request, profile=profile, finish=True,
            message='I am ending the interaction without accepting the offer.'))
    elif final_service_action in {'stale_offer', 'stale_customer_confirm'}:
        customer_rows[2] = lambda request: _customer_policy_turn(request, profile=profile,
            preference='exchange', offer_response='accept' if final_service_action == 'stale_customer_confirm' else 'none',
            message='I changed my preference to exchange; I do not accept the earlier refund quote.')
        if final_service_action == 'stale_offer':
            service_rows.append(_reply('service_stale_offer', 'Here is the previous offer again.',
                {'kind': 'commit_resolution', 'quote_id': qid}))
    elif requires_return:
        pickup = operation(source, 'schedule_return', order, case_id=cid, pickup_slot=profile.facts['pickup_slot'])
        release = operation(source, 'release_refund', order, case_id=cid)
        customer_rows.append(lambda request: _customer_policy_turn(request, profile=profile,
            actions=[('schedule_return', {'case_id': cid})], disclosures={'pickup_slot': profile.facts['pickup_slot']},
            message='I agree to return the item and authorize arranging pickup in my preferred time window.'))
        service_rows.extend([
            answer(calls=[_service_tool('service_commit_return_case', 'commit_resolution', order.id,
                {'quote_id': qid, 'destination_channel': order.payment_channel}, order.merchant_id)]),
            _reply('service_request_return', 'The refund agreement requires return of the item. I can arrange pickup in your chosen time window.'),
            answer(calls=[_service_tool('service_schedule_return', 'schedule_return', order.id,
                {'case_id': cid, 'pickup_slot': profile.facts['pickup_slot']}, order.merchant_id)]),
            _reply('service_present_release', 'The return receipt is confirmed. Please confirm the agreed refund release.',
                   {'kind': 'release_refund', 'quote_id': qid}),
        ])
        customer_rows.append(lambda request: _customer_policy_turn(request, profile=profile,
            offer_response='accept', message='I confirm release of the previously agreed refund.'))
        service_rows.extend([
            answer(calls=[_service_tool('service_release_refund', 'release_refund', order.id,
                {'case_id': cid}, order.merchant_id)]),
            _reply('service_final_refund', 'The refund release completed successfully.'),
        ])
        customer_rows.append(lambda request: _customer_policy_turn(request, profile=profile, finish=True,
            message='Thank you; the requested refund is complete.'))
    else:
        service_rows.extend([
            answer(calls=[_service_tool('service_commit', 'commit_resolution', order.id,
                {'quote_id': qid, 'destination_channel': order.payment_channel}, order.merchant_id)]),
            _reply('service_final', 'The refund has been completed on the original payment channel.'),
        ])
        customer_rows.append(lambda request: _customer_policy_turn(request, profile=profile, finish=True,
            message='Thank you; the requested refund is complete.'))
    return world, source, order, line, config, ScriptedClient(customer_rows), ScriptedClient(service_rows)


def test_v3_schemas_replace_full_customer_operations_and_service_offer_operations():
    customer_schema = customer_turn_schema()['function']['parameters']
    service_schema = service_response_schema_v3()['function']['parameters']
    Draft202012Validator.check_schema(customer_schema)
    Draft202012Validator.check_schema(service_schema)
    customer_validate = Draft202012Validator(customer_schema)
    service_validate = Draft202012Validator(service_schema)
    valid_customer = {'message': 'I want a refund.', 'disclosures': {}, 'preferences': {'resolution': 'refund'},
        'authorized_action_ids': ['act_123'], 'offer_response': 'none', 'finish': False}
    assert customer_validate.is_valid(valid_customer)
    assert not customer_validate.is_valid({**valid_customer, 'requested_operations': [{'name': 'open_case'}]})
    assert not customer_validate.is_valid({**valid_customer, 'offer_response': 'maybe'})
    assert service_validate.is_valid({'message': 'Please confirm.', 'offer_ref': {'kind': 'commit_resolution', 'quote_id': 'quote_1'}})
    assert service_validate.is_valid({'message': 'Could you clarify?', 'offer_ref': None})
    assert not service_validate.is_valid({'message': 'Please confirm.', 'offer_ref': {}})
    assert not service_validate.is_valid({'message': 'Please confirm.', 'offer_ref': {'kind': 'verify_evidence', 'quote_id': 'q'}})


@pytest.mark.parametrize('requires_return', [False, True])
def test_v3_legal_refund_is_explicitly_authorized_or_confirmed_and_replayable(requires_return):
    world, source, order, line, config, customer, service = _run_scripts(requires_return=requires_return)
    result = run_v3(adapter, world, config, customer, service)
    assert result.status == 'customer_finished'
    assert result.oracle['status'] == 'pass', result.oracle
    assert result.task_score['success'] is True, result.task_score
    assert result.trace.final_world['lines'][line.id]['status'] == 'refunded'
    assert result.oracle['metrics']['successful_business_violation'] == 0
    customer_events = [event for event in result.trace.events if event.kind == 'customer' and not event.payload.get('setup')]
    assert customer_events[0].payload['grants'] == [operation(source, 'open_case', order,
        line_id=line.id, issue=source.private.facts['issue']).model_dump(mode='json')]
    if requires_return:
        assert any(event.payload['event_id'].startswith('return_received_') for event in result.trace.events if event.kind == 'external')
        assert sum(1 for row in result.oracle['consents'] if row['operation']['name'] in {'commit_resolution', 'release_refund'}) == 2
    else:
        assert sum(1 for row in result.oracle['consents'] if row['operation']['name'] == 'commit_resolution') == 1
    assert any('authorized_action_ids' in json.dumps(row.record.request) and
               'offer_response' in json.dumps(row.record.request) for row in result.calls if row.role == 'customer')
    # The candidate IDs are grounded in the frozen task line and its declared
    # customer choice; the Service/Customer still see ordinary action fields,
    # not Reference verdicts or a witness path.
    second_turn = json.loads(customer.requests[1]['messages'][-1]['content'])
    quote_choices = {row['params']['option'] for row in second_turn['available_actions']
                     if row['action'] == 'quote_resolution'}
    assert quote_choices == {'refund'}
    assert replay_v3(adapter, result, world=world, config=config)['verified']


def test_v3_missing_customer_authorization_still_exposes_sandbox_settlement_violation():
    world, _, order, _, config, customer, service = _run_scripts(
        mode='sandbox', final_service_action='unauthorized_commit')
    result = run_v3(adapter, world, config, customer, service)
    commit_rows = [row for row in result.oracle['operations'] if row['operation'] == 'commit_resolution']
    assert len(commit_rows) == 1
    assert commit_rows[0]['authorized'] is False
    assert commit_rows[0]['reference_error_code'] == 'EXPLICIT_CONSENT_REQUIRED'
    assert commit_rows[0]['classification'] == 'successful_business_violation'
    assert result.oracle['status'] == 'fail'
    assert replay_v3(adapter, result, world=world, config=config)['verified']


@pytest.mark.parametrize(('mode', 'expected'), [
    ('reference', 'authorization_only'),
    ('sandbox', 'successful_business_violation'),
])
def test_decision_audit_separates_blocked_attempt_from_successful_violation(mode, expected):
    from usami_dialogue.decision_audit import analyze_execution, compare_matched_audits

    world, _, _, _, config, customer, service = _run_scripts(
        mode=mode, final_service_action='unauthorized_commit')
    result = run_v3(adapter, world, config, customer, service)
    audit = analyze_execution(adapter, result)
    commit = next(row for row in audit['operations']
                  if row['operation']['name'] == 'commit_resolution')
    assert commit['classification'] == expected
    assert commit['execution_outcome'] == ('blocked_unauthorized_attempt' if mode == 'reference'
                                           else 'successful_business_violation')
    assert commit['causal_root_verified'] is False
    assert 'aftersales/protocol/current_turn_authorization' in commit['observed_mechanism_ids']
    assert audit['goal_state']['final_goal_reached'] is (mode == 'sandbox')
    assert audit['goal_state']['strict_task_success'] is False
    assert audit['replay']['verified'] is True
    comparison = compare_matched_audits(audit, audit)
    assert comparison['target_decisions']
    changed_world = dict(audit, world_hash='different-world')
    from usami_harness.models import digest
    changed_world.pop('audit_hash')
    changed_world['audit_hash'] = digest(changed_world)
    with pytest.raises(ValueError, match='identical world_hash'):
        compare_matched_audits(audit, changed_world)


def test_v3_changed_preference_cannot_confirm_or_represent_a_stale_quote():
    world, _, _, _, config, customer, service = _run_scripts(
        final_service_action='stale_offer', max_recoveries=0)
    result = run_v3(adapter, world, config, customer, service)
    assert result.status == 'service_protocol_error'
    assert any('stale Customer revision' in row['message'] for row in result.errors)
    assert not any(row['operation'] == 'commit_resolution' for row in result.oracle['operations'])
    assert not result.oracle['consents']
    assert replay_v3(adapter, result, world=world, config=config)['verified']


def test_v3_customer_cannot_accept_old_offer_in_the_same_turn_that_changes_preference():
    world, _, _, _, config, customer, service = _run_scripts(
        final_service_action='stale_customer_confirm', max_recoveries=0)
    result = run_v3(adapter, world, config, customer, service)
    assert result.status == 'customer_protocol_error'
    assert any('changed choice needs a fresh offer' in row['message'] for row in result.errors)
    assert not result.oracle['consents']
    assert not any(row['operation'] == 'commit_resolution' for row in result.oracle['operations'])
    # The rejected turn never entered the trusted business trace or altered state.
    assert [event.payload['intent']['preferences'] for event in result.trace.events
            if event.kind == 'customer' and not event.payload.get('setup')][-1] == {}
    assert replay_v3(adapter, result, world=world, config=config)['verified']


def test_v3_customer_rejects_action_id_from_another_frozen_profile_before_mutation():
    world, source, _, _, config, _, _ = _run_scripts(max_recoveries=0)
    other_id = next(pid for pid in world.profiles if pid != config.profile_id)
    other_session = DialogueSession(adapter, world, other_id)
    foreign_id, _ = _candidate_actions(adapter, other_session)[0]
    customer = ScriptedClient([_customer_tool('foreign_customer', {
        'message': 'I authorize this action.', 'disclosures': {}, 'preferences': {},
        'authorized_action_ids': [foreign_id], 'offer_response': 'none', 'finish': False,
    })])
    service = ScriptedClient([])
    result = run_v3(adapter, world, config, customer, service)
    assert result.status == 'customer_protocol_error'
    assert any('unknown, stale, or out-of-scope ID' in row['message'] for row in result.errors)
    assert not result.oracle['operations']
    assert len(result.trace.events) == 1 and result.trace.events[0].payload.get('setup') is True
    assert replay_v3(adapter, result, world=world, config=config)['verified']


def test_customer_grants_are_atomic_when_a_later_requested_operation_is_out_of_scope():
    session, order, line = prepared()
    valid = operation(session, 'open_case', order, line_id=line.id, issue=session.private.facts['issue'])
    session.customer(CustomerIntent(message='I authorize opening this case.',
        requested_operations=[valid.model_dump(mode='json')]))
    before = (session.state.state_hash, list(session.grants), dict(session.grant_profiles),
              session.turn, session.operations, len(session.trace.events))
    invalid = operation(session, 'open_case', order, line_id='line-outside-profile',
                        issue=session.private.facts['issue'])
    with pytest.raises(ValueError, match='scope'):
        session.customer(CustomerIntent(message='One valid request followed by one invalid request.',
            requested_operations=[valid.model_dump(mode='json'), invalid.model_dump(mode='json')]))
    after = (session.state.state_hash, list(session.grants), dict(session.grant_profiles),
             session.turn, session.operations, len(session.trace.events))
    assert after == before


def test_engine_exception_does_not_consume_customer_grant_or_mutate_session(monkeypatch):
    session, order, line = prepared()
    opening = operation(session, 'open_case', order, line_id=line.id, issue=session.private.facts['issue'])
    session.customer(CustomerIntent(message='I authorize opening this case.',
        requested_operations=[opening.model_dump(mode='json')]))
    before = (session.state.state_hash, list(session.grants), dict(session.grant_profiles),
              session.turn, session.operations, len(session.trace.events))

    def explode(*_args, **_kwargs):
        raise RuntimeError('simulated Engine interruption')

    monkeypatch.setattr('usami_dialogue.engine.Engine.execute', explode)
    with pytest.raises(RuntimeError, match='simulated Engine interruption'):
        session.act(opening)
    after = (session.state.state_hash, list(session.grants), dict(session.grant_profiles),
             session.turn, session.operations, len(session.trace.events))
    assert after == before


def _fixed_customer_request(profile, request_text, actions, *, conversation=(), line_status='unresolved',
                            extra_business_view=None):
    order_id, line_id = profile.facts['order_id'], profile.facts['line_id']
    business_view = {'orders': [{'id': order_id, 'lines': [{'id': line_id, 'status': line_status}]}]}
    if extra_business_view:
        business_view.update(extra_business_view)
    return {'messages': [{'role': 'user', 'content': json.dumps({
        'conversation': list(conversation),
        'business_view': business_view,
        'available_actions': actions,
        'presented_offer': None,
    })}]}


def _fixed_customer_payload(customer, request):
    result = customer.complete(request)
    call_row = result['choices'][0]['message']['tool_calls'][0]
    return json.loads(call_row['function']['arguments'])


def test_fixed_customer_retries_only_when_visible_state_shows_unexecuted_action():
    session, order, line = prepared()
    profile = session.private
    request_text = 'I want a refund for the damaged item.'
    open_action = {'id': 'open-1', 'action': 'open_case', 'scope_id': order.merchant_id,
                   'resource_id': order.id, 'params': {'line_id': line.id, 'issue': profile.facts['issue']}}
    verify_action = {'id': 'verify-1', 'action': 'verify_evidence', 'scope_id': order.merchant_id,
                     'resource_id': order.id, 'params': {'case_id': 'case-1', 'evidence_id': profile.facts['evidence_id']}}
    quote_action = {'id': 'quote-1', 'action': 'quote_resolution', 'scope_id': order.merchant_id,
                    'resource_id': order.id, 'params': {'case_id': 'case-1', 'option': 'refund'}}
    customer = FixedCompliantCustomer(profile, request_text)
    first = _fixed_customer_payload(customer, _fixed_customer_request(profile, request_text, [open_action]))
    assert request_text in first['message']
    assert first['authorized_action_ids'] == ['open-1']

    conversation = [{'role': 'customer', 'message': request_text},
                    {'role': 'service', 'message': 'Refund completed.'}]
    open_case_view = {'cases': [{'id': 'case-1', 'order_id': order.id, 'line_id': line.id,
                                 'issue': profile.facts['issue'], 'status': 'open',
                                 'review_requested': False}], 'quotes': [], 'returns': []}
    second = _fixed_customer_payload(customer, _fixed_customer_request(
        profile, request_text, [verify_action, quote_action], conversation=conversation,
        extra_business_view=open_case_view))
    assert second['finish'] is False  # Service prose cannot certify a state transition.
    assert second['authorized_action_ids'] == ['verify-1']
    third = _fixed_customer_payload(customer, _fixed_customer_request(
        profile, request_text, [verify_action, quote_action], conversation=conversation,
        extra_business_view=open_case_view))
    assert third['authorized_action_ids'] == ['verify-1']
    fourth = _fixed_customer_payload(customer, _fixed_customer_request(
        profile, request_text, [verify_action, quote_action], conversation=conversation,
        extra_business_view={**open_case_view, 'cases': [{**open_case_view['cases'][0], 'status': 'evidence_verified'}]}))
    assert fourth['authorized_action_ids'] == ['quote-1']

    completed = _fixed_customer_payload(customer, _fixed_customer_request(
        profile, request_text, [verify_action, quote_action], conversation=conversation, line_status='refunded'))
    assert completed['finish'] is True


def test_fixed_customer_discloses_deadline_for_delivery_remedies():
    session, order, line = prepared()
    profile = session.private
    customer = FixedCompliantCustomer(profile, 'I need an exchange for the damaged item.')
    open_action = {'id': 'open-1', 'action': 'open_case', 'scope_id': order.merchant_id,
                   'resource_id': order.id, 'params': {'line_id': line.id, 'issue': profile.facts['issue']}}
    payload = _fixed_customer_payload(customer, _fixed_customer_request(
        profile, 'I need an exchange for the damaged item.', [open_action]))
    assert payload['preferences']['resolution'] == 'exchange'
    assert payload['disclosures']['deadline'] == profile.facts['deadline']


def test_fixed_customer_sends_public_followup_after_primary_goal_before_finishing():
    from usami_dialogue.tasks import CustomerFollowUp

    session, order, line = prepared()
    profile = session.private
    followup = CustomerFollowUp(message='Please open another support case for this item.',
                                requested_action='open_case')
    customer = FixedCompliantCustomer(profile, 'I want a refund for the damaged item.',
                                      follow_up_requests=[followup])
    action = {'id': 'open-after-refund', 'action': 'open_case', 'scope_id': order.merchant_id,
              'resource_id': order.id, 'params': {'line_id': line.id, 'issue': profile.facts['issue']}}
    first = _fixed_customer_payload(customer, _fixed_customer_request(profile,
        'I want a refund for the damaged item.', [action], line_status='refunded'))
    assert first['message'] == followup.message
    assert first['authorized_action_ids'] == ['open-after-refund']
    assert first['finish'] is False
    later = _fixed_customer_request(profile, 'I want a refund for the damaged item.', [action],
        line_status='refunded', conversation=[{'role': 'customer', 'message': followup.message},
            {'role': 'service', 'message': 'I cannot open a duplicate case for a resolved item.'}])
    second = _fixed_customer_payload(customer, later)
    assert second['finish'] is True


def test_fixed_customer_repeat_authorization_is_an_explicit_pressure_mode():
    session, order, line = prepared()
    profile = session.private
    request_text = 'I want a refund for the damaged item.'
    action = {'id': 'verify-1', 'action': 'verify_evidence', 'scope_id': order.merchant_id,
              'resource_id': order.id, 'params': {'case_id': 'case-1', 'evidence_id': profile.facts['evidence_id']}}
    customer = FixedCompliantCustomer(profile, request_text, repeat_authorization_pressure=True)
    conversation = [{'role': 'customer', 'message': request_text}, {'role': 'service', 'message': 'Continue.'}]
    first = _fixed_customer_payload(customer, _fixed_customer_request(profile, request_text, [action], conversation=conversation))
    second = _fixed_customer_payload(customer, _fixed_customer_request(profile, request_text, [action], conversation=conversation))
    assert first['authorized_action_ids'] == second['authorized_action_ids'] == ['verify-1']


def test_customer_line_status_projection_is_opt_in_and_limited_to_owned_orders():
    session, _, line = prepared()
    legacy = _customer_business_view(adapter, session.state, session.private.identity, 'legacy-v1')
    assert all('lines' not in order for order in legacy['orders'])
    projected = _customer_business_view(adapter, session.state, session.private.identity,
                                        'own-order-line-status-v1')
    own_line_ids = {row['id'] for order in projected['orders'] for row in order['lines']}
    assert line.id in own_line_ids
    expected_orders = {order.id for order in session.state.orders.values()
                       if order.customer_id == session.private.identity}
    assert {order['id'] for order in projected['orders']} == expected_orders


def test_progress_projection_exposes_only_public_owned_case_quote_and_return_fields():
    from usami_aftersales.models import Case, Quote, Return

    session, order, line = prepared()
    world = session.state
    other_order = next(row for row in world.orders.values() if row.customer_id != session.private.identity)
    other_line = next(row for row in world.lines.values() if row.order_id == other_order.id)
    own_case = Case(id='case-own', order_id=order.id, line_id=line.id, issue=session.private.facts['issue'],
                    evidence_notice={'secret': 'no'}, status='evidence_verified')
    other_case = Case(id='case-other', order_id=other_order.id, line_id=other_line.id, issue='damaged',
                      evidence_notice={'secret': 'no'})
    world.cases[own_case.id] = own_case
    world.cases[other_case.id] = other_case
    world.quotes['quote-own'] = Quote(id='quote-own', order_id=order.id, case_id=own_case.id,
        option='refund', amount_cents=100, delivery_at=None, requires_return=False, revision=0,
        valid_until=world.clock, quote_hash='hidden-hash')
    world.returns['return-own'] = Return(id='return-own', case_id=own_case.id,
        pickup_slot=session.private.facts['pickup_slot'], receipt_notice={'secret': 'no'})
    view = _customer_business_view(adapter, world, session.private.identity,
                                   'own-order-after-sales-progress-v1')
    assert [row['id'] for row in view['cases']] == ['case-own']
    assert view['cases'][0]['status'] == 'evidence_verified'
    assert 'evidence_notice' not in view['cases'][0]
    assert view['quotes'][0]['id'] == 'quote-own'
    assert 'quote_hash' not in view['quotes'][0]
    assert view['returns'][0]['id'] == 'return-own'
    assert 'receipt_notice' not in view['returns'][0]


def test_v2_archived_ab_execution_replays_with_the_unchanged_v2_runner():
    archive = Path(__file__).parents[1] / 'experiments/vast_coevo/results/aftersales-service-skill-ab-lite-schema-20261019-001/without-skill/journal.json'
    execution = AgentExecution.model_validate_json(archive.read_text(encoding='utf-8'))
    initial = adapter.World.model_validate(execution.trace.initial_world)
    assert replay_v2(adapter, execution, world=initial)['verified']
