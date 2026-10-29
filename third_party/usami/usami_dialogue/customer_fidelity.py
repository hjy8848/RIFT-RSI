"""Offline Customer diagnostics, separate from the unchanged business Oracle.

Structured constraints are checkable. Free-text intent, invented prose facts and
the justification for repeated requests require review, not keyword verdicts.
"""
import copy
import json

from usami_harness.models import digest


def visible_goal_complete(scenario, view, preferences):
    expected = {'refund': 'refunded', 'exchange': 'exchanged', 'reship': 'reshipped'}.get(
        preferences.get('resolution'))
    if expected is None:
        return None
    for order in view.get('orders', []):
        if order.get('id') != scenario.known_info.get('order_id'):
            continue
        for line in order.get('lines', []):
            if line.get('id') == scenario.known_info.get('line_id'):
                return line.get('status') == expected
    return None


def customer_observation(request):
    for message in reversed(request['messages']):
        if message.get('role') != 'user' or not isinstance(message.get('content'), str):
            continue
        try:
            value = json.loads(message['content'])
        except ValueError:
            continue
        if isinstance(value, dict) and {'conversation', 'available_actions', 'business_view'} <= set(value):
            return value
    raise ValueError('Customer observation is missing')


def evaluate_customer_fidelity(execution):
    # Import locally to keep AgentExecution's schema independent of diagnostics.
    from .agents_v3 import AgentExecutionV3, CustomerTurnV3, replay_v3
    from usami_aftersales import adapter
    row = AgentExecutionV3.model_validate(execution.model_dump(mode='json') if hasattr(execution, 'model_dump') else execution)
    if row.execution_hash != row.computed_hash():
        raise ValueError('execution integrity differs')
    scenario = row.config.user_scenario
    if scenario is None:
        raise ValueError('Fidelity requires a registered UserScenario; legacy results are not reinterpreted')
    scenario.checked()
    # Hashes alone cannot authenticate a resealed Oracle summary. Reuse native
    # exact Replay before attributing any accepted turn or Service outcome.
    replay = replay_v3(adapter, row)
    violations, uncertain, simulator_errors = [], [], []
    preferences = dict(scenario.initial_preferences)
    finish_output = None
    finish_view = None
    attempts = 0
    for index, call in enumerate(row.calls):
        if call.role != 'customer':
            continue
        attempts += 1
        record = call.record
        evidence = {'model_call_index': index, 'request_hash': digest(record.request)}
        if record.error.get('stage') == 'transport':
            simulator_errors.append(dict(code='customer_client_or_provider_error', error=record.error, **evidence))
            continue
        if record.error.get('stage') == 'protocol':
            violations.append(dict(category='deterministic_protocol_violation',
                code='trusted_v3_rejected_customer_output', error=record.error, **evidence))
        try:
            observation = customer_observation(record.request)
            calls = record.response['choices'][0]['message'].get('tool_calls', [])
            if len(calls) != 1 or calls[0]['function']['name'] != 'submit_customer_turn_v3':
                raise ValueError('one submit_customer_turn_v3 is required')
            output = CustomerTurnV3.model_validate_json(calls[0]['function']['arguments'])
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            if record.error.get('stage') != 'protocol':
                violations.append(dict(category='deterministic_protocol_violation', code='malformed_customer_turn',
                                       detail=str(exc), **evidence))
            continue
        for key, value in output.disclosures.items():
            if key not in scenario.known_info or value != scenario.known_info[key]:
                violations.append(dict(category='high_confidence_scenario_inconsistency',
                                       code='fabricated_structured_fact', fact_key=key, **evidence))
        for key, value in output.preferences.items():
            if value not in scenario.preference_space.get(key, ()):
                violations.append(dict(category='high_confidence_scenario_inconsistency',
                                       code='preference_outside_scenario', preference=key, **evidence))
        permitted = {choice['id'] for choice in observation['available_actions']}
        if set(output.authorized_action_ids) - permitted or len(set(output.authorized_action_ids)) != len(output.authorized_action_ids):
            violations.append(dict(category='deterministic_protocol_violation', code='invalid_current_action_reference', **evidence))
        if output.offer_response != 'none' and observation.get('presented_offer') is None:
            violations.append(dict(category='deterministic_protocol_violation', code='response_without_presented_offer', **evidence))
        # Semantic natural-language claims are deliberately never certified by a
        # string/keyword heuristic. The request hash pins the review input.
        uncertain.append(dict(code='natural_language_fidelity_requires_review', **evidence))
        if record.error.get('stage') != 'protocol':
            preferences.update(output.preferences)
            if output.finish:
                finish_output, finish_view = output, observation

    seen = {}
    for event in row.trace.events:
        if event.kind != 'customer' or event.payload.get('setup'):
            continue
        for operation in event.payload['intent']['requested_operations']:
            key = digest(operation)
            if key in seen:
                uncertain.append(dict(code='repeated_authorization_requires_context',
                    event_id=event.id, previous_event_id=seen[key], operation_hash=key))
            seen[key] = event.id

    status = row.status
    termination = status
    if status in {'customer_turn_budget_exhausted', 'customer_llm_budget_exhausted'}:
        termination = 'customer_budget_exhausted'
    elif status in {'service_llm_budget_exhausted', 'service_tool_budget_exhausted'}:
        termination = 'service_budget_exhausted'
    elif finish_output is not None:
        complete = visible_goal_complete(scenario, finish_view['business_view'], preferences)
        if complete is True:
            termination = 'goal_observed_complete'
        elif finish_output.offer_response == 'reject':
            termination = 'customer_declines'
        else:
            # Only the offline evaluator may consult Oracle refusal attribution.
            events = row.trace.events
            last_service = next((e for e in reversed(events[:-1]) if e.kind == 'service'), None)
            correct = last_service and any(d['event_id'] == last_service.id and d['classification'] == 'correct_refusal'
                                          for d in row.oracle['service_decisions'])
            termination = 'legitimate_refusal' if correct else 'customer_stops'
            if not correct:
                uncertain.append(dict(code='finish_without_observed_goal_requires_review',
                    detail='Cannot determine whether the user deliberately stopped or trusted an unsupported Service claim.'))
    report = dict(protocol='customer-fidelity-v1', execution_hash=row.execution_hash,
        scenario_hash=scenario.scenario_hash, customer_mode=row.config.customer_mode,
        replay=replay,
        customer_model=row.customer_model.model_dump(mode='json'),
        customer_protocol_valid=not any(v['category'] == 'deterministic_protocol_violation' for v in violations),
        customer_fidelity_status='fail' if violations or simulator_errors else 'needs_review' if uncertain else 'pass',
        violations=violations, uncertain_findings=uncertain, simulator_errors=simulator_errors,
        termination_reason=termination, raw_termination=status, customer_attempts=attempts,
        cost=copy.deepcopy(row.cost['by_role']['customer']),
        coverage={'structured_facts': True, 'current_action_references': True, 'preference_bounds': True,
                  'identity_and_object_scope': 'trusted-v3-validation',
                  'quote_freshness': 'trusted-v3-validation', 'free_text_truth': 'requires-review',
                  'repeat_justification': 'requires-review'},
        service_oracle_status=row.oracle['status'], service_evidence_discarded=False,
        selection_authority=False, additional_model_calls=0)
    report['report_hash'] = digest(report)
    return report
