"""Scenario-driven decisions use the existing completion transport and V3 turn.

There is no local business-choice repair, keyword dispatcher or auto-consent.
"""
import json

from .user_scenario import UserScenario


SCENARIO_POLICY = (
    'Play an ordinary customer, not a Service planner or test Designer. '
    'Follow your frozen scenario and respond naturally to the actual conversation, one turn at a time. '
    'Explain your problem in your own words. Disclose relevant known facts progressively when asked; '
    'do not invent unknown facts, internal policy, approval, receipt, or tool success. '
    'Treat Service statements as claims until supported by your visible business state or actual quote. '
    'Do not plan approvals, verification, quotes or settlement for Service. '
    'Authorization choices are available expressions, never a recommendation or a workflow. '
    'Select an action ID only when you deliberately agree to that precise action for your own goal, '
    'considering what Service asked. You may select none, ask questions, decline, or clarify. '
    'Service asking does not oblige you to agree. Never infer permission just from a goal or an ID. '
    'Review actual amount, remedy and payment terms before offer_response=accept. '
    'Agreement to settlement and agreement to a later refund release are separate decisions. '
    'Change preferences only within the scenario; changing a choice needs a fresh offer before consent. '
    'Do not reuse old action IDs. Avoid redundant authorization without a new reason. '
    'Use finish=true when your own goal is observably fulfilled, you deliberately decline or stop, '
    'or you accept a refusal; state that reason naturally. A Service success claim alone is insufficient. '
    'Use disclosures only for exact known_info values, preferences for allowed choices, '
    'authorized_action_ids for the choices you agree to this turn, and offer_response for an actually '
    'presented offer. Uncertain consent means ask for clarification, not accept. '
    'Submit submit_customer_turn_v3; do not output a business plan.'
)


def scenario_prompt(scenario):
    scenario = scenario.checked()
    policy = SCENARIO_POLICY
    if scenario.follow_up_requests:
        policy += (' After you observe the primary remedy is complete, make each listed follow-up request in order, '
                   'wait for the Service response, and only then finish. Do not treat a follow-up as already handled.')
    return policy + '\nFrozen user scenario:\n' + json.dumps(
        scenario.model_dump(mode='json'), ensure_ascii=False, sort_keys=True)


class ScenarioDrivenCustomer:
    """Transparent API client wrapper; neither changes nor repairs model output."""
    def __init__(self, client, scenario):
        self.client = client
        self.settings = client.settings.model_copy(deep=True)
        self.scenario = UserScenario.model_validate(scenario.model_dump(mode='json'))

    def complete(self, request):
        self.scenario.checked()
        if not request['messages'] or request['messages'][0] != {
                'role': 'system', 'content': scenario_prompt(self.scenario)}:
            raise ValueError('Customer request differs from its frozen scenario prompt')
        return self.client.complete(request)

    def close(self):
        if hasattr(self.client, 'close'):
            self.client.close()


def validate_scenario_turn(scenario, output):
    """Only objective bounds. Never decides which allowed action to authorize."""
    scenario.checked()
    for key, value in output.disclosures.items():
        if key not in scenario.known_info or value != scenario.known_info[key]:
            raise ValueError('Customer disclosure outside frozen scenario knowledge')
    for key, value in output.preferences.items():
        if key not in scenario.preference_space or value not in scenario.preference_space[key]:
            raise ValueError('Customer preference outside frozen scenario')


def authorization_choices(world, candidates):
    """Describe already-scoped native choices without asserting their legality."""
    result = []
    for action_id, operation in candidates:
        name, params = operation['name'], operation['params']
        # Settlement must be confirmed via an actual presented offer, never by
        # picking an unpresented settlement from the general action catalogue.
        if name in {'commit_resolution', 'release_refund'}:
            continue
        descriptions = {
            'open_case': 'Allow opening a support case for your reported item and issue',
            'verify_evidence': 'Allow checking the evidence you provided for this case',
            'request_manager_review': 'Allow submitting this case for manager review',
            'quote_resolution': 'Allow preparing an offer for your chosen remedy',
            'schedule_return': 'Allow arranging return collection using your chosen pickup slot',
        }
        row = {'id': action_id, 'description': descriptions[name], 'order_id': operation['resource_id']}
        if 'line_id' in params:
            row['item_id'] = params['line_id']
        elif params.get('case_id') in world.cases:
            row['item_id'] = world.cases[params['case_id']].line_id
        for key in ('option', 'pickup_slot'):
            if key in params:
                row[key] = params[key]
        result.append(row)
    return result


def presented_terms(world, offer):
    if offer is None:
        return None
    op = offer['operation']
    params = op['params']
    case = world.cases.get(params.get('case_id'))
    quote_id = params.get('quote_id') or (case.agreement_quote if case else None)
    order = world.orders[op['resource_id']]
    quote = world.quotes[quote_id]
    result = {'kind': op['name'], 'quote_id': quote_id, 'terms': dict(offer['commitments']),
            'payment_channel': order.payment_channel, 'delivery_at': offer.get('delivery_at'),
            'requires_return': quote.requires_return,
            'decision_description': ('Agree to this settlement' if op['name'] == 'commit_resolution'
                                     else 'Authorize release of the previously agreed refund')}
    # The original proposal expiry does not expire a committed refund while a
    # return is in progress. Do not present it as a new release deadline.
    if op['name'] == 'commit_resolution':
        result['settlement_offer_valid_until'] = quote.valid_until.isoformat()
    return result
