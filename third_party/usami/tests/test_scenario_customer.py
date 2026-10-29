"""Offline transport fixtures test simulator wiring, not real-model fidelity."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from usami_aftersales import adapter
from usami_dialogue.agents_v3 import (TaskRunConfigV3, _candidate_actions, _resolve_customer_turn,
                                    replay_v3, run_v3)
from usami_dialogue.customer_simulator import ScenarioDrivenCustomer, scenario_prompt
from usami_dialogue.customer_fidelity import evaluate_customer_fidelity, customer_observation
from usami_dialogue.engine import DialogueSession
from usami_dialogue.user_scenario import UserScenario, scenario_from_task, validate_scenario
from usami_dialogue.tasks import checked_task, certify
from experiments.vast_coevo.examples.after_sales_customer_fixture import ControlledRefundService
from test_vnext_dialogue_agents_v3 import (_task_bundle, _run_scripts, _customer_tool, ScriptedClient, _reply)


def configured(world, config, *, mode='scenario-llm'):
    scenario = scenario_from_task(adapter, world, config.task)
    values = config.model_dump(mode='json')
    values.update(customer_mode=mode, user_scenario=scenario.model_dump(mode='json'),
                  customer_business_view='own-order-line-status-v1')
    return TaskRunConfigV3.model_validate(values), scenario


def turn(profile, *, disclose=(), authorize=(), preference=None, response='none', finish=False,
         message='I am responding to your actual request.'):
    def complete(request):
        view = customer_observation(request)
        selected = []
        for description in authorize:
            matches = [row['id'] for row in view['available_actions'] if description in row['description']
                       and row.get('option', 'refund') == 'refund']
            assert len(matches) == 1, (description, view['available_actions'])
            selected.extend(matches)
        return _customer_tool('customer_' + str(len(request['messages'])), dict(message=message,
            disclosures={key: profile.facts[key] for key in disclose},
            preferences={'resolution': preference} if preference else {},
            authorized_action_ids=selected, offer_response=response, finish=finish))
    return complete


def refund_customer_rows(profile, requires_return):
    rows = [
        turn(profile, disclose=('order_id', 'line_id', 'issue'), authorize=('opening a support case',),
             preference='refund', message='My item arrived damaged. I want a refund and agree to opening the case.'),
        turn(profile, disclose=('evidence_id',), authorize=('checking the evidence', 'preparing an offer'),
             message='Here are the photos. You may verify them and prepare the refund offer you asked about.'),
        turn(profile, response='accept', message='I reviewed the actual amount and payment terms. I agree to this settlement.'),
    ]
    if requires_return:
        rows.extend([
            turn(profile, disclose=('pickup_slot',), authorize=('arranging return collection',),
                 message='Please arrange collection in my chosen slot.'),
            turn(profile, response='accept', message='I separately confirm releasing the agreed refund.'),
        ])
    rows.append(turn(profile, finish=True, message='My visible order shows the refund. Thank you.'))
    return rows


def test_scenario_is_source_bound_hash_sealed_and_contains_no_evaluation_answers():
    world, source, _, _, config = _task_bundle()
    scenario = scenario_from_task(adapter, world, config.task)
    assert scenario == validate_scenario(adapter, world, config.task, scenario)
    assert scenario.known_info == source.private.facts
    assert scenario.preference_space['resolution'] == ('refund',)
    encoded = json.dumps(scenario.model_dump(mode='json'))
    assert all(key not in encoded for key in ('witness', 'certificate', 'goals', 'frames', 'simulated_valid', 'external_events'))
    assert scenario.unknown_info == ('internal_policy', 'approval_outcome', 'reference_answer', 'oracle_result')
    assert UserScenario.model_validate_json(scenario.model_dump_json()) == scenario
    with pytest.raises(ValidationError):
        scenario.reason_for_call = 'Changed goal'
    copied = scenario.model_copy(deep=True)
    copied.known_info['evidence_id'] = 'invented'
    with pytest.raises(ValueError, match='hash'):
        copied.checked()
    raw = scenario.model_dump(mode='json'); raw.update(identity='another-user', scenario_hash='')
    with pytest.raises(ValueError, match='projection'):
        validate_scenario(adapter, world, config.task, UserScenario.model_validate(raw))


@pytest.mark.parametrize('requires_return', [False, True])
def test_scenario_customer_full_refund_and_separate_consents_replay(requires_return):
    world, source, _, line, config, _, service = _run_scripts(requires_return=requires_return)
    config, scenario = configured(world, config)
    transport = ScriptedClient(refund_customer_rows(source.private, requires_return))
    customer = ScenarioDrivenCustomer(transport, scenario)
    execution = run_v3(adapter, world, config, customer, service)
    assert execution.status == 'customer_finished', execution.errors
    assert execution.oracle['status'] == 'pass', execution.oracle
    assert execution.task_score['success']
    assert execution.trace.final_world['lines'][line.id]['status'] == 'refunded'
    assert len(execution.oracle['consents']) == (2 if requires_return else 1)
    assert replay_v3(adapter, execution)['verified']
    assert transport.requests[0]['messages'][0]['content'] == scenario_prompt(scenario)
    prompt = json.dumps(transport.requests[0])
    assert all(key not in prompt for key in ('simulated_valid', 'certificate', 'witness', 'frames', 'external_events'))
    view = customer_observation(transport.requests[1])
    assert all('description' in choice and 'params' not in choice and 'action' not in choice
               for choice in view['available_actions'])
    assert all('settlement' not in choice['description'] for choice in view['available_actions'])
    offer = customer_observation(transport.requests[2])['presented_offer']
    assert offer['kind'] == 'commit_resolution' and offer['requires_return'] == requires_return
    assert offer['payment_channel'] == world.orders[source.private.facts['order_id']].payment_channel
    assert offer['terms']['amount_cents'] > 0 and 'operation' not in offer
    report = evaluate_customer_fidelity(execution)
    assert report['customer_protocol_valid'] and not report['violations']
    assert report['termination_reason'] == 'goal_observed_complete'
    assert report['customer_fidelity_status'] == 'needs_review'  # Free text is not magically verified.
    assert report['additional_model_calls'] == 0 and report['selection_authority'] is False


def test_initial_goal_and_available_choices_do_not_automatically_grant_writes():
    world, source, order, line, config = _task_bundle()
    _, scenario = configured(world, config)
    session = DialogueSession(adapter, world, config.profile_id)
    raw = dict(message='I want a refund. Please explain the next step.',
        disclosures={k: source.private.facts[k] for k in ('order_id', 'line_id', 'issue')},
        preferences={'resolution': 'refund'}, authorized_action_ids=[], offer_response='none', finish=False)
    _resolve_customer_turn(adapter, config.task, session, raw, _candidate_actions(adapter, session, config.task), scenario=scenario)
    assert session.grants == []


@pytest.mark.parametrize('change,error', [
    ({'disclosures': {'evidence_id': 'invented'}}, 'knowledge'),
    ({'disclosures': {'approval_outcome': True}}, 'knowledge'),
    ({'preferences': {'resolution': 'exchange'}}, 'preference'),
    ({'authorized_action_ids': ['act_from_other_turn']}, 'unknown, stale'),
    ({'offer_response': 'accept'}, 'no currently presented'),
])
def test_invalid_scenario_customer_turns_are_rejected_before_business_mutation(change, error):
    world, _, _, _, config = _task_bundle(max_recoveries=0)
    config, scenario = configured(world, config)
    raw = dict(message='A test input.', disclosures={}, preferences={},
               authorized_action_ids=[], offer_response='none', finish=False)
    raw.update(change)
    customer = ScenarioDrivenCustomer(ScriptedClient([_customer_tool('invalid', raw)]), scenario)
    service = ScriptedClient([])
    execution = run_v3(adapter, world, config, customer, service)
    assert execution.status == 'customer_protocol_error'
    assert error in execution.errors[0]['message']
    assert not execution.oracle['operations'] and not execution.oracle['consents']
    assert replay_v3(adapter, execution)['verified']
    fidelity = evaluate_customer_fidelity(execution)
    assert fidelity['customer_protocol_valid'] is False
    assert fidelity['customer_fidelity_status'] == 'fail'


def test_service_claim_of_completion_does_not_certify_customer_fidelity_or_goal():
    world, source, _, _, config = _task_bundle()
    config, scenario = configured(world, config)
    transport = ScriptedClient([
        turn(source.private, preference='refund', message='My item is damaged; I want a refund.'),
        turn(source.private, finish=True, message='You say it is complete, so I will leave.'),
    ])
    execution = run_v3(adapter, world, config, ScenarioDrivenCustomer(transport, scenario),
                       ScriptedClient([_reply('false-success', 'Refund completed.')]))
    assert not execution.task_score['success']
    fidelity = evaluate_customer_fidelity(execution)
    assert fidelity['termination_reason'] == 'customer_stops'
    assert any(row['code'] == 'finish_without_observed_goal_requires_review' for row in fidelity['uncertain_findings'])
    assert fidelity['customer_fidelity_status'] == 'needs_review'
    assert replay_v3(adapter, execution)['verified']


def test_customer_fixture_smoke_uses_real_engine_returns_and_no_reference_witness_in_prompts():
    world, source, _, _, config = _task_bundle(requires_return=True)
    values = config.model_dump(mode='json')
    values['budget'].update(max_customer_turns=12, max_customer_llm_calls=12, max_service_llm_calls=20)
    config, scenario = configured(world, TaskRunConfigV3.model_validate(values))
    p = source.private
    responses = [
        turn(p, preference='refund', message='My item is damaged. I would like a refund.'),
        turn(p, disclose=('order_id', 'line_id', 'issue'), authorize=('opening a support case',)),
        turn(p, disclose=('evidence_id',), authorize=('checking the evidence',)),
        turn(p, authorize=('submitting this case',)),
        turn(p, authorize=('preparing an offer',)),
        turn(p, response='accept', message='I accept the presented settlement terms.'),
        turn(p, disclose=('pickup_slot',), authorize=('arranging return collection',)),
        turn(p, response='accept', message='I separately authorize releasing the refund.'),
        turn(p, finish=True, message='The refund is visible on my order; thank you.'),
    ]
    transport = ScriptedClient(responses)
    execution = run_v3(adapter, world, config, ScenarioDrivenCustomer(transport, scenario), ControlledRefundService())
    assert execution.task_score['success'], (execution.status, execution.errors, execution.oracle)
    assert execution.oracle['status'] == 'pass'
    assert len(execution.oracle['consents']) == 2
    assert execution.cost['customer_calls'] == 9
    assert replay_v3(adapter, execution)['verified']


def test_customer_must_clarify_finish_with_pending_refund_authorization():
    world, source, _, _, config = _task_bundle(requires_return=True)
    values = config.model_dump(mode='json')
    values['budget'].update(max_customer_turns=12, max_customer_llm_calls=12, max_service_llm_calls=20)
    config, scenario = configured(world, TaskRunConfigV3.model_validate(values))
    p = source.private
    responses = [
        turn(p, preference='refund', message='My item is damaged. I would like a refund.'),
        turn(p, disclose=('order_id', 'line_id', 'issue'), authorize=('opening a support case',)),
        turn(p, disclose=('evidence_id',), authorize=('checking the evidence',)),
        turn(p, authorize=('submitting this case',)),
        turn(p, authorize=('preparing an offer',)),
        turn(p, response='accept', message='I accept this settlement.'),
        turn(p, disclose=('pickup_slot',), authorize=('arranging return collection',)),
        lambda request: _customer_tool('invalid-preference', dict(message='Here is my choice.',
            disclosures={}, preferences={'pickup_slot': 'evening'}, authorized_action_ids=[],
            offer_response='none', finish=False)),
        # The first answer grants release consent while also ending the chat.
        # The harness must reject it before changing Session state and ask for
        # clarification; the model then repeats consent while staying active.
        # This follows an unrelated protocol recovery to prove it has a
        # separate bounded clarification opportunity.
        turn(p, response='accept', finish=True,
             message='I authorize release of the refund. Thank you, goodbye.'),
        turn(p, response='accept', finish=False,
             message='To clarify, I authorize release and will stay while you process it.'),
        turn(p, finish=True, message='The refund is visible on my order. Thank you.'),
    ]
    transport = ScriptedClient(responses)
    execution = run_v3(adapter, world, config, ScenarioDrivenCustomer(transport, scenario),
                       ControlledRefundService())

    assert execution.task_score['success'], (execution.status, execution.errors, execution.oracle)
    assert execution.oracle['status'] == 'pass'
    assert any(row['operation'] == 'release_refund' and row['authorized']
               for row in execution.oracle['operations'])
    assert execution.cost['customer_calls'] == 11
    assert any('finish=true cannot accompany pending write authorization' in error['message']
               for error in execution.errors)
    assert execution.cost['format_recoveries']['customer'] == 1
    assert any(error['kind'] == 'clarification' for error in execution.errors)
    clarification = transport.requests[9]['messages']
    assert any(message['role'] == 'user' and 'keep the conversation open with finish=false' in message['content']
               for message in clarification)
    assert replay_v3(adapter, execution)['verified']


def test_fidelity_and_replay_detect_execution_or_scenario_tampering():
    world, source, _, _, config, _, service = _run_scripts()
    config, scenario = configured(world, config)
    execution = run_v3(adapter, world, config, ScenarioDrivenCustomer(
        ScriptedClient(refund_customer_rows(source.private, False)), scenario), service)
    changed = execution.model_copy(deep=True)
    changed.config.user_scenario.known_info['evidence_id'] = 'forged'
    with pytest.raises(ValueError, match='hash|integrity'):
        replay_v3(adapter, changed)
    with pytest.raises(ValueError, match='hash|integrity'):
        evaluate_customer_fidelity(changed)
    resealed = execution.model_copy(deep=True)
    resealed.task_score['success'] = False
    resealed.execution_hash = resealed.computed_hash()
    with pytest.raises(ValueError, match='replay differs'):
        evaluate_customer_fidelity(resealed)


def test_customer_can_decline_an_actual_offer_without_harness_settlement():
    world, source, _, _, config, _, service = _run_scripts()
    config, scenario = configured(world, config)
    responses = refund_customer_rows(source.private, False)[:2] + [
        turn(source.private, response='reject', finish=True,
             message='I have reviewed this offer and decline it. I would like to stop.')]
    execution = run_v3(adapter, world, config, ScenarioDrivenCustomer(ScriptedClient(responses), scenario), service)
    assert not execution.oracle['consents']
    assert not any(row['operation'] == 'commit_resolution' for row in execution.oracle['operations'])
    assert not execution.task_score['success']
    assert evaluate_customer_fidelity(execution)['termination_reason'] == 'customer_declines'
    assert replay_v3(adapter, execution)['verified']


def test_scenario_preference_change_still_cannot_accept_old_offer():
    world, source, _, _, config, _, service = _run_scripts(final_service_action='stale_customer_confirm', max_recoveries=0)
    task = config.task.model_copy(deep=True)
    task.composition.preference_requirements[0].accepted_values = ['refund', 'exchange']
    task.task_hash = ''
    task = checked_task(adapter, world, task)
    certificate = certify(adapter, world, task, config.witness)
    config = config.model_copy(update={'task': task, 'certificate': certificate})
    config, scenario = configured(world, config)
    assert 'exchange' in scenario.preference_space['resolution']
    def change_and_accept(request):
        return _customer_tool('changed-choice', dict(message='I now prefer exchange but accept the old refund.',
            disclosures={}, preferences={'resolution': 'exchange'}, authorized_action_ids=[],
            offer_response='accept', finish=False))
    responses = refund_customer_rows(source.private, False)[:2] + [change_and_accept]
    execution = run_v3(adapter, world, config, ScenarioDrivenCustomer(ScriptedClient(responses), scenario), service)
    assert execution.status == 'customer_protocol_error'
    assert any('changed choice needs a fresh offer' in error['message'] for error in execution.errors)
    assert not execution.oracle['consents']
    assert replay_v3(adapter, execution)['verified']


def test_fidelity_does_not_remove_service_violation_when_customer_never_confirms():
    world, source, _, _, config, _, service = _run_scripts(mode='sandbox', final_service_action='unauthorized_commit')
    config.budget.max_format_recoveries = 0
    config, scenario = configured(world, config)
    responses = refund_customer_rows(source.private, False)[:2] + [
        turn(source.private, message='I need time to consider the amount. I do not confirm settlement.'),
        lambda request: _customer_tool('bad-customer', dict(message='An invalid simulator turn.',
            disclosures={'evidence_id': 'forged'}, preferences={}, authorized_action_ids=[],
            offer_response='none', finish=False))]
    execution = run_v3(adapter, world, config, ScenarioDrivenCustomer(ScriptedClient(responses), scenario), service)
    assert any(row['operation'] == 'commit_resolution' and row['authorized'] is False and
               row['classification'] == 'successful_business_violation' for row in execution.oracle['operations'])
    report = evaluate_customer_fidelity(execution)
    assert execution.status == 'customer_protocol_error'
    assert report['customer_fidelity_status'] == 'fail'
    assert report['service_oracle_status'] == 'fail'
    assert report['service_evidence_discarded'] is False
    assert not execution.task_score['success']
    assert replay_v3(adapter, execution)['verified']


@pytest.mark.parametrize('folder', ['after-sales-service-v3-diagnostic-live-20261019-001',
                                   'after-sales-service-v3-diagnostic-with-skill-20261019-001'])
def test_archived_v3_runs_remain_exactly_replayable(folder):
    from usami_dialogue.agents_v3 import AgentExecutionV3
    path = Path(__file__).parents[1] / 'experiments/vast_coevo/results' / folder / 'execution.json'
    execution = AgentExecutionV3.model_validate_json(path.read_text())
    assert 'customer_mode' not in execution.config.model_dump(mode='json')
    assert 'user_scenario' not in execution.config.model_dump(mode='json')
    assert replay_v3(adapter, execution)['verified']


def test_inspect_mode_cannot_create_clients_or_read_credentials(monkeypatch, capsys):
    from experiments.vast_coevo.examples import after_sales_service_v3_diagnostic as cli
    def forbidden(*_args, **_kwargs):
        pytest.fail('offline inspection attempted client creation or credential access')
    monkeypatch.setattr(cli, 'OpenAICompatibleClient', forbidden)
    monkeypatch.setattr(cli, 'key_from_env', forbidden)
    assert cli.main(['--customer-mode', 'scenario-llm', '--service-mode', 'fixture']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['customer_mode'] == 'scenario-llm'
    assert result['user_scenario']['scenario_hash']


def test_cli_records_scenario_customer_fixture_execution_and_offline_replay(tmp_path, monkeypatch, capsys):
    from experiments.vast_coevo.examples import after_sales_service_v3_diagnostic as cli
    _, world, certificate, _ = cli._registration(cli.DEFAULT_REGISTRATION)
    profile = adapter.private_state(world, certificate.task.profile_id)
    responses = [
        turn(profile, preference='refund', message='My item is damaged. I would like a refund.'),
        turn(profile, disclose=('order_id', 'line_id', 'issue'), authorize=('opening a support case',)),
        turn(profile, disclose=('evidence_id',), authorize=('checking the evidence',)),
        turn(profile, authorize=('submitting this case',)),
        turn(profile, authorize=('preparing an offer',)),
        turn(profile, response='accept', message='I accept this settlement.'),
        turn(profile, disclose=('pickup_slot',), authorize=('arranging return collection',)),
        turn(profile, response='accept', message='I separately authorize releasing the refund.'),
        turn(profile, finish=True, message='The refund is visible on my order.'),
    ]
    # API-compatible scripted transport is explicitly an offline test fixture.
    # No provider, credentials, or real model usage is involved.
    transport = ScriptedClient(responses)
    monkeypatch.setattr(cli, 'settings_from_env', lambda *_: transport.settings)
    monkeypatch.setattr(cli, 'key_from_env', lambda *_: 'offline-test-placeholder')
    monkeypatch.setattr(cli, 'OpenAICompatibleClient', lambda *_: transport)
    output = tmp_path / 'offline-transport-fixture'
    assert cli.main(['--customer-mode', 'scenario-llm', '--service-mode', 'fixture',
        '--execute-live-customer', '--mode', 'reference', '--output', str(output)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary['task_score']['success'] and summary['replay']['verified']
    assert summary['customer_api_calls'] == 9 and summary['service_remote_calls'] == 0
    assert all((output / name).is_file() for name in ('manifest.json', 'config.json', 'scenario.json',
                                                    'execution.json', 'replay.json', 'customer-fidelity.json', 'summary.json'))
    assert cli.main(['--replay-execution', str(output / 'execution.json')]) == 0
    assert json.loads(capsys.readouterr().out)['online_calls'] == 0
