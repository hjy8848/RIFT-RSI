"""Opt-in After-sales dialogue protocol v3.

V3 replaces Customer-authored business operations with turn-scoped action IDs,
and Service-authored full quote operations with a quote reference. The existing
DialogueSession, business Engine, Reference, Oracle and trace format remain the
single source of execution semantics. The v2 runner is intentionally untouched.
"""
from __future__ import annotations

import copy
import json
from typing import Literal, Optional

from pydantic import Field

from usami_harness.models import Budget, Model, digest
from usami_harness.tools import parse, schemas
from usami.service.client import RecordedCompletionClient
from usami.service.models import LLMRecord, ModelSettings, ServiceError, strict_json
from usami.service.runner import completion_message
from .agents import SkillPromptContext
from .engine import DialogueSession, ExecutionTrace
from .oracle import evaluate
from .tasks import DialogueTask, TaskCertificate, checked_task, score, verify_certificate
from .user_scenario import UserScenario, validate_scenario
from .customer_simulator import (authorization_choices, presented_terms, scenario_prompt,
                                 validate_scenario_turn)

PROTOCOL_V3_FINISH_CLARIFICATION = 'business-dialogue-task-agents-v3-finish-clarification-v1'


class CustomerTurnV3(Model):
    """Small semantic intent; exact Operations are resolved by trusted code."""
    message: str = Field(min_length=1, max_length=8000)
    disclosures: dict
    preferences: dict
    authorized_action_ids: list[str] = Field(max_length=32)
    offer_response: Literal['none', 'accept', 'reject']
    finish: bool


class CustomerTurnClarificationRequired(ValueError):
    """The turn is otherwise valid, but its requested control flow is ambiguous."""


class OfferReference(Model):
    model_config = {'extra': 'forbid'}
    kind: Literal['commit_resolution', 'release_refund']
    quote_id: str = Field(min_length=1, max_length=160)


class TaskRunConfigV3(Model):
    # Keep legacy v3 runs exactly replayable while new runs use the clarified
    # finish/authorization control-flow contract.
    protocol: Literal['business-dialogue-task-agents-v3',
                      'business-dialogue-task-agents-v3-finish-clarification-v1'] = (
        PROTOCOL_V3_FINISH_CLARIFICATION
    )
    profile_id: str
    initial_world_hash: str
    mode: Literal['reference', 'sandbox'] = 'reference'
    tools: list[str]
    budget: Budget = Field(default_factory=Budget)
    external_schedule: list[str] = Field(default_factory=list)
    skill_contexts: dict[str, SkillPromptContext] = Field(default_factory=dict,
                                                           exclude_if=lambda value: not value)
    # Opt-in public status projection for deterministic Customer drivers. The
    # legacy prompt remains byte-for-byte stable for archived v3 replays.
    customer_business_view: Literal['legacy-v1', 'own-order-line-status-v1',
                                    'own-order-after-sales-progress-v1'] = Field(
        default='legacy-v1', exclude_if=lambda value: value == 'legacy-v1')
    customer_mode: Literal['legacy-v3', 'deterministic', 'scenario-llm'] = Field(
        default='legacy-v3', exclude_if=lambda value: value == 'legacy-v3')
    user_scenario: Optional[UserScenario] = Field(default=None, exclude_if=lambda value: value is None)
    task: DialogueTask
    certificate: TaskCertificate
    witness: ExecutionTrace


class AgentCallV3(Model):
    role: Literal['customer', 'service']
    record: LLMRecord


class AgentExecutionV3(Model):
    protocol: Literal['business-dialogue-task-agents-v3',
                      'business-dialogue-task-agents-v3-finish-clarification-v1'] = (
        PROTOCOL_V3_FINISH_CLARIFICATION
    )
    config: TaskRunConfigV3
    customer_model: ModelSettings
    service_model: ModelSettings
    calls: list[AgentCallV3]
    tools: list[dict]
    trace: ExecutionTrace
    oracle: dict
    task_score: dict
    status: str
    errors: list[dict]
    cost: dict
    execution_hash: str = ''

    def computed_hash(self):
        return digest(self.model_dump(mode='json', exclude={'execution_hash'}))


CUSTOMER_POLICY_V3 = (
    'You are the ordinary customer in one frozen after-sales task, not a test Designer. '
    'You may share only facts in your private profile and may choose only its listed preferences. '
    'When you authorize a write, select only the exact action IDs you intend to authorize this turn; '
    'candidate actions are choices, not instructions or claims that policy permits execution. '
    'Never invent an ID or reuse one from another turn. Use offer_response=accept only after comparing '
    'the formally presented quote terms with your request. Use reject to decline a current offer, and '
    'none when no offer is being accepted or declined. Do not infer hidden Reference or Oracle answers.'
)

SERVICE_POLICY_V3 = (
    'You are the Service assistant in a frozen after-sales business world. Choose your own queries, '
    'business tools, and order. A tool result is the only evidence that an operation occurred. '
    'The authenticated customer identity is fixed. Only the exact Customer-declared writes for the '
    'current turn are authorized; neither your message nor tool arguments create authority. '
    'Obtain required facts, verify evidence, request any required review, and independently decide '
    'whether a quote or settlement is appropriate. Do not invent quotes, approval, receipt, consent, '
    'or success. Before presenting a quote, use respond_to_customer_v3 with a real quote ID returned '
    'by the business system; the host will show its actual terms. A quote reference does not grant '
    'settlement authority. Never use a Reference witness or hidden evaluator state.'
)


def customer_turn_schema():
    return {
        'type': 'function',
        'function': {
            'name': 'submit_customer_turn_v3',
            'description': 'Submit one customer response with turn-scoped action grants.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'message': {'type': 'string', 'minLength': 1, 'maxLength': 8000},
                    'disclosures': {'type': 'object'},
                    'preferences': {'type': 'object'},
                    'authorized_action_ids': {'type': 'array', 'items': {'type': 'string'},
                                              'maxItems': 32, 'uniqueItems': True},
                    'offer_response': {'type': 'string', 'enum': ['none', 'accept', 'reject']},
                    'finish': {'type': 'boolean'},
                },
                'required': ['message', 'disclosures', 'preferences', 'authorized_action_ids',
                             'offer_response', 'finish'],
                'additionalProperties': False,
            },
        },
    }


def service_response_schema_v3():
    return {
        'type': 'function',
        'function': {
            'name': 'respond_to_customer_v3',
            'description': 'Reply or present a real current after-sales quote by reference.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'message': {'type': 'string', 'minLength': 1},
                    'offer_ref': {
                        'anyOf': [
                            {'type': 'object', 'properties': {
                                'kind': {'type': 'string', 'enum': ['commit_resolution', 'release_refund']},
                                'quote_id': {'type': 'string', 'minLength': 1, 'maxLength': 160},
                            }, 'required': ['kind', 'quote_id'], 'additionalProperties': False},
                            {'type': 'null'},
                        ]
                    },
                    'refused_operation': {'anyOf': [{'type': 'object'}, {'type': 'null'}]},
                    'refusal_code': {'anyOf': [{'type': 'string'}, {'type': 'null'}]},
                },
                'required': ['message', 'offer_ref'],
                'additionalProperties': False,
            },
        },
    }


def _task_resolutions(task, private):
    values = []
    composition = task.composition
    for row in composition.preference_requirements:
        if row.profile_id == task.profile_id and row.preference == 'resolution':
            values.extend(row.accepted_values)
    if not values:
        terminal_to_option = {'refunded': 'refund', 'exchanged': 'exchange', 'reshipped': 'reship'}
        for goal in task.goals:
            if goal.collection == 'lines' and goal.entity == private.facts.get('line_id') and goal.field == 'status':
                values.extend(terminal_to_option[value] for value in goal.accepted_values
                              if value in terminal_to_option)
    return set(values) & set(private.preference_space.get('resolution', []))


def _operation_in_task_scope(adapter, task, session, raw):
    operation = adapter.Operation.model_validate(raw)
    private = session.private
    world = session.state
    if operation.resource_id not in private.protected_objects:
        return False
    # A composed task has one frozen line frame. Bind all case/quote actions to
    # that line in addition to the ordinary customer profile boundary.
    if operation.name == 'open_case':
        line_id = operation.params.get('line_id')
        if line_id != private.facts.get('line_id'):
            return False
    elif operation.name in {'verify_evidence', 'request_manager_review', 'quote_resolution',
                            'schedule_return', 'release_refund'}:
        case = world.cases.get(operation.params.get('case_id', ''))
        if case is None or case.line_id != private.facts.get('line_id'):
            return False
    elif operation.name == 'commit_resolution':
        quote = world.quotes.get(operation.params.get('quote_id', ''))
        case = world.cases.get(quote.case_id) if quote else None
        if quote is None or case is None or case.line_id != private.facts.get('line_id'):
            return False
    elif operation.name not in {'get_orders', 'get_order', 'get_case', 'get_options'}:
        return False
    # A task's declared positive outcomes determine which remedy choices may be
    # authorized. Read-only queries remain available under native identity
    # rules, but off-task writes are not offered as Customer grant candidates.
    allowed = _task_resolutions(task, private)
    if operation.name == 'quote_resolution' and operation.params.get('option') not in allowed:
        return False
    if operation.name == 'commit_resolution':
        quote = world.quotes.get(operation.params.get('quote_id', ''))
        if quote is None or quote.option not in allowed:
            return False
    if operation.name == 'release_refund':
        case = world.cases.get(operation.params.get('case_id', ''))
        quote = world.quotes.get(case.agreement_quote or '') if case else None
        if quote is None or quote.option not in allowed:
            return False
    if operation.name == 'schedule_return' and 'refund' not in allowed:
        return False
    return bool(allowed) or operation.model_dump(mode='json') in [row.operation for row in task.refusals]


def _candidate_actions(adapter, session, task=None):
    """Return only native Customer choices inside the active frozen profile."""
    world = session.state
    rows = []
    for raw in adapter.candidate_intents(world, session.private):
        operation = adapter.Operation.model_validate(raw)
        canonical = operation.model_dump(mode='json')
        if not adapter.intent_allowed(world, session.private, canonical):
            continue
        if task is not None and not _operation_in_task_scope(adapter, task, session, canonical):
            continue
        action_id = 'act_' + digest({
            'protocol': 'business-dialogue-task-agents-v3',
            'world_hash': world.state_hash,
            'profile_hash': session.private.private_hash,
            'task_hash': task.task_hash if task is not None else None,
            'turn': session.turn + 1,
            'operation': canonical,
        })[:20]
        rows.append((action_id, canonical))
    rows.sort(key=lambda item: item[0])
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError('candidate action ID collision')
    return rows


def _action_view(rows):
    return [{'id': action_id, 'action': operation['name'], 'scope_id': operation['scope_id'],
             'resource_id': operation['resource_id'], 'params': operation['params']}
            for action_id, operation in rows]


def _customer_business_view(adapter, world, identity, version):
    view = adapter.observation(world, identity)
    if version in {'own-order-line-status-v1', 'own-order-after-sales-progress-v1'} and adapter.name == 'aftersales':
        for order in view.get('orders', []):
            order['lines'] = [line.model_dump(mode='json') for line in sorted(world.lines.values(), key=lambda row: row.id)
                              if line.order_id == order['id']]
    if version == 'own-order-after-sales-progress-v1' and adapter.name == 'aftersales':
        own_order_ids = {row['id'] for row in view.get('orders', [])}
        cases = [row for row in sorted(world.cases.values(), key=lambda item: item.id)
                 if row.order_id in own_order_ids]
        case_ids = {row.id for row in cases}
        view['cases'] = [{key: getattr(row, key) for key in
                         ('id', 'order_id', 'line_id', 'issue', 'status', 'review_requested',
                          'agreement_quote', 'agreement_revision')} for row in cases]
        view['quotes'] = [{key: getattr(row, key).isoformat() if key in {'delivery_at', 'valid_until'}
                           and getattr(row, key) is not None else getattr(row, key)
                           for key in ('id', 'case_id', 'option', 'amount_cents', 'delivery_at',
                                       'requires_return', 'revision', 'valid_until', 'status')}
                          for row in sorted(world.quotes.values(), key=lambda item: item.id)
                          if row.case_id in case_ids]
        view['returns'] = [{key: getattr(row, key) for key in
                           ('id', 'case_id', 'pickup_slot', 'status')}
                           for row in sorted(world.returns.values(), key=lambda item: item.id)
                           if row.case_id in case_ids]
    return view


def _current_offer_operation(adapter, session, task):
    """Check current presentation freshness without evaluating business legality."""
    presented = session.presented_offer
    if not isinstance(presented, dict) or not isinstance(presented.get('operation'), dict):
        raise ValueError('there is no currently presented quote to accept')
    world = session.state
    operation = adapter.Operation.model_validate(presented['operation'])
    if not _operation_in_task_scope(adapter, task, session, operation.model_dump(mode='json')):
        raise ValueError('presented quote is outside the frozen task scope')
    current = adapter.offer_for(world, operation.model_dump(mode='json'))
    if current != presented:
        raise ValueError('presented quote is stale or no longer matches current business state')
    if presented.get('world_hash') != world.state_hash or presented.get('revision') != world.interaction.revision:
        raise ValueError('presented quote version is stale')
    quote = _quote_for_operation(world, operation)
    if quote is None or (operation.name == 'commit_resolution' and world.clock > quote.valid_until):
        raise ValueError('presented quote is missing or expired')
    return operation.model_dump(mode='json')


def _quote_for_operation(world, operation):
    if operation.name == 'commit_resolution':
        return world.quotes.get(operation.params.get('quote_id'))
    if operation.name == 'release_refund':
        case = world.cases.get(operation.params.get('case_id'))
        return world.quotes.get(case.agreement_quote or '') if case else None
    return None


def _resolve_offer_ref(adapter, session, task, raw):
    if raw is None:
        return None
    ref = OfferReference.model_validate(raw)
    world = session.state
    quote = world.quotes.get(ref.quote_id)
    if quote is None:
        raise ValueError('offer_ref must identify an existing quote')
    if quote.order_id not in session.private.protected_objects:
        raise ValueError('offer_ref is outside the frozen Customer task scope')
    case = world.cases.get(quote.case_id)
    order = world.orders.get(quote.order_id)
    if (case is None or order is None or case.order_id != order.id or
            case.line_id != session.private.facts.get('line_id')):
        raise ValueError('offer_ref does not belong to the frozen Customer profile')
    if quote.revision != world.interaction.revision:
        raise ValueError('offer_ref identifies a quote from a stale Customer revision')
    if world.interaction.preferences.get('resolution') != quote.option:
        raise ValueError('offer_ref does not match the current Customer preference')
    if ref.kind == 'commit_resolution':
        if world.clock > quote.valid_until:
            raise ValueError('offer_ref identifies an expired quote')
        if quote.status != 'offered' or case.agreement_quote is not None:
            raise ValueError('commit_resolution offer_ref is not an open quote')
        operation = adapter.Operation(name='commit_resolution', scope_id=order.merchant_id,
            resource_id=order.id, params={'quote_id': quote.id, 'destination_channel': order.payment_channel})
    else:
        if quote.option != 'refund' or quote.status != 'committed' or case.agreement_quote != quote.id:
            raise ValueError('release_refund offer_ref is not bound to a committed refund quote')
        if case.agreement_revision != quote.revision:
            raise ValueError('release_refund offer_ref agreement revision differs')
        operation = adapter.Operation(name='release_refund', scope_id=order.merchant_id,
            resource_id=order.id, params={'case_id': case.id})
    canonical = operation.model_dump(mode='json')
    if not _operation_in_task_scope(adapter, task, session, canonical):
        raise ValueError('offer_ref is outside the frozen task scope')
    offer = adapter.offer_for(world, canonical)
    if offer is None or offer.get('quote_hash') != quote.quote_hash:
        raise ValueError('offer_ref does not resolve to the quote terms in the current world')
    return canonical


def _resolve_customer_turn(adapter, task, session, raw, candidates, *, scenario=None,
                           protocol=PROTOCOL_V3_FINISH_CLARIFICATION):
    output = CustomerTurnV3.model_validate(raw)
    if scenario is not None:
        validate_scenario_turn(scenario, output)
    ids = output.authorized_action_ids
    if len(ids) != len(set(ids)):
        raise ValueError('authorized_action_ids contains duplicates')
    available = dict(candidates)
    unknown = [action_id for action_id in ids if action_id not in available]
    if unknown:
        raise ValueError('authorized_action_ids contains an unknown, stale, or out-of-scope ID')
    offer_response = output.offer_response
    confirm = None
    reject = False
    if offer_response == 'accept':
        confirm = _current_offer_operation(adapter, session, task)
    elif offer_response == 'reject':
        if session.presented_offer is None:
            raise ValueError('offer_response=reject requires a currently presented quote')
        _current_offer_operation(adapter, session, task)
        reject = True
    requested = [available[action_id] for action_id in ids]
    if confirm is not None and any(op == confirm for op in requested):
        raise ValueError('confirming an offer already grants its exact operation; do not duplicate its action ID')
    # Do not accept authorization and termination in the same turn. run_v3
    # stops immediately on finish, before it can forward this turn's one-use
    # grants to Service. Reject before mutating DialogueSession so the
    # dedicated clarification path can ask Customer without losing consent.
    if (protocol == PROTOCOL_V3_FINISH_CLARIFICATION and output.finish and
            (confirm is not None or requested)):
        raise CustomerTurnClarificationRequired(
            'finish=true cannot accompany pending write authorization or offer acceptance; '
            'clarify by continuing with finish=false so Service can act, or withdraw the authorization'
        )
    intent = {
        'message': output.message,
        'disclosures': output.disclosures,
        'preferences': output.preferences,
        'requested_operations': requested,
        'confirm_operation': confirm,
        'reject_offer': reject,
        'finish': output.finish,
    }
    # DialogueSession remains authoritative for private facts, action bounds,
    # consent, exact one-use grants and current quote matching.
    session.customer(intent)
    return output


def _validate_registration(adapter, world, config):
    if adapter.name != 'aftersales':
        raise ValueError('business-dialogue-task-agents-v3 currently supports After-sales only')
    task = checked_task(adapter, world, config.task)
    if task != config.task:
        raise ValueError('v3 task must carry its canonical frozen content hash')
    if task.composition is None or task.composition.profile_ids != [config.profile_id]:
        raise ValueError('v3 requires one frozen Customer profile and identity scope')
    if task.profile_id != config.profile_id or task.world_hash != config.initial_world_hash:
        raise ValueError('task and run profile/world registration differ')
    if config.certificate.task != task:
        raise ValueError('task differs from its private native certificate')
    verify_certificate(adapter, world, config.certificate, config.witness)
    if config.initial_world_hash != world.state_hash or set(config.tools) - set(adapter.TOOLS):
        raise ValueError('frozen world/tool configuration differs')
    if len(config.tools) != len(set(config.tools)) or len(config.external_schedule) != len(set(config.external_schedule)):
        raise ValueError('duplicate tool/external registration')
    if config.external_schedule != task.external_schedule:
        raise ValueError('task external schedule differs from run registration')
    if set(config.external_schedule) - set(world.external_events):
        raise ValueError('unregistered external event')
    for context in config.skill_contexts.values():
        SkillPromptContext.checked(context)
    if config.customer_mode == 'legacy-v3':
        if config.user_scenario is not None:
            raise ValueError('a UserScenario requires an explicit Customer mode')
    else:
        if config.user_scenario is None or config.customer_business_view not in {
                'own-order-line-status-v1', 'own-order-after-sales-progress-v1'}:
            raise ValueError('explicit Customer modes require a frozen scenario and an explicit public status view')
        validate_scenario(adapter, world, task, config.user_scenario)
        from .preflight import customer_interaction_preflight
        preflight = customer_interaction_preflight(adapter, world, task, config.user_scenario)
        if preflight['status'] == 'fail':
            raise ValueError('Customer interaction preflight failed: ' + json.dumps(preflight, sort_keys=True))
        if 'customer' in config.skill_contexts:
            raise ValueError('scenario Customer is fixed; Customer Skill overrides are not supported')


def run_v3(adapter, world, config, customer_client, service_client):
    """Run one After-sales task using V3 references and canonical native execution."""
    raw_config = config.model_dump(mode='json') if isinstance(config, TaskRunConfigV3) else config
    config = TaskRunConfigV3.model_validate(raw_config)
    _validate_registration(adapter, world, config)
    session = DialogueSession(adapter, world, config.profile_id,
                              max_operations=config.budget.max_service_tool_calls,
                              profile_ids=config.task.composition.profile_ids)
    customer_model = customer_client.settings.model_copy(deep=True)
    service_model = service_client.settings.model_copy(deep=True)
    customer_tools = [customer_turn_schema()]
    service_tools = schemas(adapter, config.tools) + [service_response_schema_v3()]
    if config.user_scenario is not None:
        customer_system = scenario_prompt(config.user_scenario)
        if hasattr(customer_client, 'scenario') and customer_client.scenario != config.user_scenario:
            raise ValueError('Customer client and registered scenario differ')
    else:
        customer_system = CUSTOMER_POLICY_V3 + ' Frozen customer profile: ' + json.dumps(
            session.private.model_dump(mode='json'), sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        customer_system += ' Frozen task request: ' + config.task.composition.customer_request
    if 'customer' in config.skill_contexts:
        context = SkillPromptContext.checked(config.skill_contexts['customer'])
        customer_system += '\n\nApplied declarative Customer Skills from snapshot ' + context.snapshot_hash + ':\n' + context.rendered
    customer_messages = [{'role': 'system', 'content': customer_system}]
    service_system = (SERVICE_POLICY_V3 + ' Authenticated identity: ' + session.private.identity +
        '. Public policy: ' + json.dumps(adapter.POLICY, sort_keys=True, ensure_ascii=False) +
        '. Budget: ' + config.budget.model_dump_json())
    if 'service' in config.skill_contexts:
        context = SkillPromptContext.checked(config.skill_contexts['service'])
        service_system += '\n\nApplied declarative Service Skills from snapshot ' + context.snapshot_hash + ':\n' + context.rendered
    service_messages = [{'role': 'system', 'content': service_system}]
    calls = []
    tool_records = []
    errors = []
    seen_call_ids = set()
    seen_customer_call_ids = set()
    conversation = []
    pending = list(config.external_schedule)
    counts = {'customer': 0, 'service': 0}
    recoveries = {'customer': 0, 'service': 0}
    status = 'customer_turn_budget_exhausted'
    stopped = False

    def complete(role, messages, tools):
        client = customer_client if role == 'customer' else service_client
        model = customer_model if role == 'customer' else service_model
        request = copy.deepcopy(model.request(messages, tools))
        if role == 'customer':
            request['tool_choice'] = {'type': 'function', 'function': {'name': 'submit_customer_turn_v3'}}
        row = AgentCallV3(role=role, record=LLMRecord(request=request))
        counts[role] += 1
        calls.append(row)
        try:
            row.record.response = client.complete(copy.deepcopy(request))
        except Exception as exc:
            row.record.response = getattr(exc, 'response', None)
            row.record.error = {'stage': 'transport', 'code': getattr(exc, 'code', 'PROVIDER_ERROR'),
                                'message': str(exc)}
            errors.append({'role': role, 'kind': 'transport', 'message': str(exc)})
            return None
        return row.record

    def protocol_error(role, record, exc, messages):
        record.error = {'stage': 'protocol', 'code': 'OUTPUT_INVALID', 'message': str(exc)}
        errors.append({'role': role, 'kind': 'protocol', 'message': str(exc)})
        recoveries[role] += 1
        if recoveries[role] > config.budget.max_format_recoveries:
            return False
        messages.append({'role': 'user', 'content': 'The previous output was invalid and caused no business action: ' +
                         str(exc) + '. Correct it within the remaining frozen budget.'})
        return True

    for _turn_index in range(config.budget.max_customer_turns):
        if counts['customer'] >= config.budget.max_customer_llm_calls:
            status = 'customer_llm_budget_exhausted'
            break
        candidates = _candidate_actions(adapter, session, config.task)
        if config.customer_mode == 'scenario-llm':
            candidates = [(action_id, operation) for action_id, operation in candidates
                          if operation['name'] not in {'commit_resolution', 'release_refund'}]
        actions_view = (authorization_choices(session.state, candidates) if config.customer_mode == 'scenario-llm'
                        else _action_view(candidates))
        customer_observation = {
            'conversation': conversation,
            'business_view': _customer_business_view(adapter, session.state, session.private.identity,
                                                       config.customer_business_view),
            'available_actions': actions_view,
            'presented_offer': (presented_terms(session.state, session.presented_offer)
                                if config.customer_mode == 'scenario-llm' else session.presented_offer),
        }
        customer_messages.append({'role': 'user', 'content': json.dumps(
            customer_observation, ensure_ascii=False, sort_keys=True)})
        customer_output = None
        finish_clarification_used = False
        while counts['customer'] < config.budget.max_customer_llm_calls:
            record = complete('customer', customer_messages, customer_tools)
            if record is None:
                status = 'customer_provider_error'
                stopped = True
                break
            try:
                message, proposals = completion_message(record.response, seen_customer_call_ids)
                if (len(proposals) != 1 or
                        proposals[0]['function']['name'] != 'submit_customer_turn_v3'):
                    raise ValueError('one submit_customer_turn_v3 call is required')
                parsed = strict_json(proposals[0]['function']['arguments'])
                customer_output = _resolve_customer_turn(adapter, config.task, session, parsed, candidates,
                                                          scenario=config.user_scenario, protocol=config.protocol)
                customer_messages.append(message)
                customer_messages.append({'role': 'tool', 'tool_call_id': proposals[0]['id'],
                    'content': json.dumps({'accepted': True, 'turn': session.turn,
                                           'granted_action_count': len(session.grants)}, sort_keys=True)})
                break
            except CustomerTurnClarificationRequired as exc:
                record.error = {'stage': 'protocol', 'code': 'OUTPUT_CLARIFICATION_REQUIRED',
                                'message': str(exc)}
                errors.append({'role': 'customer', 'kind': 'clarification', 'message': str(exc)})
                if finish_clarification_used:
                    status = 'customer_protocol_error'
                    stopped = True
                    break
                finish_clarification_used = True
                customer_messages.append({'role': 'user', 'content':
                    'Your last turn both authorized a pending operation and asked to end the conversation. '
                    'The Service has not received that authorization. Please clarify: keep the conversation '
                    'open with finish=false so Service can act, or withdraw the authorization before finishing.'})
                continue
            except (ValueError, TypeError, KeyError, ServiceError) as exc:
                if not protocol_error('customer', record, exc, customer_messages):
                    status = 'customer_protocol_error'
                    stopped = True
                    break
        if stopped:
            break
        if customer_output is None:
            status = 'customer_llm_budget_exhausted'
            break
        conversation.append({'role': 'customer', 'message': customer_output.message})
        if customer_output.finish:
            status = 'customer_finished'
            break
        grant_view = json.dumps(session.grants, sort_keys=True, ensure_ascii=False)
        service_messages.append({'role': 'user', 'content': customer_output.message +
            '\nExact Customer-declared single-use writes for this turn: ' + grant_view})
        replied = False
        while counts['service'] < config.budget.max_service_llm_calls:
            record = complete('service', service_messages, service_tools)
            if record is None:
                status = 'service_provider_error'
                stopped = True
                break
            try:
                message, actions = completion_message(record.response, seen_call_ids)
            except (ValueError, TypeError, KeyError, ServiceError) as exc:
                if not protocol_error('service', record, exc, service_messages):
                    status = 'service_protocol_error'
                    stopped = True
                    break
                continue
            response_calls = [action for action in actions
                              if action['function']['name'] == 'respond_to_customer_v3']
            if response_calls:
                try:
                    if len(actions) != 1:
                        raise ValueError('reply/offer cannot be mixed with business tools')
                    args = strict_json(response_calls[0]['function']['arguments'])
                    if (not isinstance(args, dict) or not {'message', 'offer_ref'} <= set(args) or
                            set(args) - {'message', 'offer_ref', 'refused_operation', 'refusal_code'}):
                        raise ValueError('reply requires message and offer_ref')
                    operation = _resolve_offer_ref(adapter, session, config.task, args['offer_ref'])
                    session.reply(args['message'], operation,
                        refused_operation=args.get('refused_operation'), refusal_code=args.get('refusal_code'))
                except (ValueError, TypeError, KeyError, ServiceError) as exc:
                    if not protocol_error('service', record, exc, service_messages):
                        status = 'service_protocol_error'
                        stopped = True
                        break
                    continue
                service_messages.append(message)
                service_messages.append({'role': 'tool', 'tool_call_id': response_calls[0]['id'],
                    'content': json.dumps({'delivered': True, 'presented_offer': session.presented_offer},
                                          sort_keys=True, ensure_ascii=False)})
                conversation.append({'role': 'service', 'message': args['message'],
                    'presented_offer': (presented_terms(session.state, session.presented_offer)
                                        if config.customer_mode == 'scenario-llm' else session.presented_offer)})
                replied = True
                break
            if not actions:
                text = message.get('content')
                if not isinstance(text, str) or not text.strip():
                    if not protocol_error('service', record, ValueError('empty Service response'), service_messages):
                        status = 'service_protocol_error'
                        stopped = True
                        break
                    continue
                session.reply(text)
                service_messages.append(message)
                conversation.append({'role': 'service', 'message': text, 'presented_offer': None})
                replied = True
                break
            service_messages.append(message)
            for action in actions:
                if len(tool_records) >= config.budget.max_service_tool_calls:
                    status = 'service_tool_budget_exhausted'
                    stopped = True
                    break
                before_hash = session.state.state_hash
                try:
                    operation = parse(adapter, action['function']['name'],
                                      action['function']['arguments'], config.tools)
                    result = ({'success': True, 'error_code': None,
                               'data': adapter.observation(session.state, session.private.identity)}
                              if operation is None else session.act(operation, config.mode))
                    result = {'success': True, 'error_code': None, 'data': result} if isinstance(result, dict) and 'success' not in result else result
                except (ValueError, TypeError, KeyError, ServiceError) as exc:
                    result = {'success': False, 'error_code': 'TOOL_ARGUMENT_INVALID', 'data': {},
                              'message': str(exc)}
                tool_records.append({'call_id': action['id'], 'name': action['function']['name'],
                    'arguments': action['function']['arguments'], 'result': result,
                    'before_hash': before_hash, 'after_hash': session.state.state_hash})
                service_messages.append({'role': 'tool', 'tool_call_id': action['id'],
                    'content': json.dumps(result, ensure_ascii=False, sort_keys=True)})
                if adapter.invariants(session.state):
                    status = 'business_world_invalid'
                    stopped = True
                    break
                for event_id in list(pending):
                    if adapter.external_ready(session.state, event_id):
                        session.external(event_id)
                        pending.remove(event_id)
                        service_messages.append({'role': 'user', 'content':
                            'A pre-registered external simulator update has arrived. Query current business state before deciding.'})
            if stopped:
                break
        if stopped:
            break
        if not replied:
            status = 'service_llm_budget_exhausted'
            break

    trace = session.finish()
    oracle = evaluate(adapter, trace, initial_world=world, private=session.private)
    task_score = score(adapter, world, config.task, trace)
    token_fields = ('input_tokens', 'output_tokens', 'total_tokens')
    cost = {key: 0 for key in token_fields}
    usage_complete = True
    by_role = {role: {**{key: 0 for key in token_fields}, 'calls': counts[role], 'usage_complete': True}
               for role in ('customer', 'service')}
    for row in calls:
        usage = row.record.response.get('usage', {}) if isinstance(row.record.response, dict) else {}
        if not isinstance(usage, dict):
            usage = {}
        required = ('prompt_tokens', 'completion_tokens', 'total_tokens')
        if not all(isinstance(usage.get(key), int) and not isinstance(usage[key], bool) and usage[key] >= 0
                   for key in required):
            usage_complete = False
            by_role[row.role]['usage_complete'] = False
            continue
        for target, source in (('input_tokens', 'prompt_tokens'), ('output_tokens', 'completion_tokens'),
                               ('total_tokens', 'total_tokens')):
            cost[target] += usage[source]
            by_role[row.role][target] += usage[source]
    cost.update({'usage_complete': usage_complete, 'by_role': by_role,
                 'customer_calls': counts['customer'], 'service_calls': counts['service'],
                 'tool_calls': len(tool_records), 'format_recoveries': recoveries})
    result = AgentExecutionV3(protocol=config.protocol, config=config,
        customer_model=customer_model, service_model=service_model,
        calls=calls, tools=tool_records, trace=trace, oracle=oracle, task_score=task_score,
        status=status, errors=errors, cost=cost)
    result.execution_hash = result.computed_hash()
    return result


def replay_v3(adapter, execution, *, world=None, config=None):
    raw = execution.model_dump(mode='json') if isinstance(execution, AgentExecutionV3) else execution
    row = AgentExecutionV3.model_validate(raw)
    if row.protocol != row.config.protocol or row.execution_hash != row.computed_hash():
        raise ValueError('v3 execution/config integrity differs')
    initial = adapter.World.model_validate(row.trace.initial_world)
    if world is not None and world != initial:
        raise ValueError('registered world differs')
    if config is not None and config != row.config:
        raise ValueError('registered config differs')
    customer = RecordedCompletionClient(row.customer_model, [call.record for call in row.calls if call.role == 'customer'])
    service = RecordedCompletionClient(row.service_model, [call.record for call in row.calls if call.role == 'service'])
    rebuilt = run_v3(adapter, initial, row.config, customer, service)
    if rebuilt != row or customer.index != len(customer.records) or service.index != len(service.records):
        raise ValueError('v3 model request/response, action resolution, tool call or replay differs')
    return {'verified': True, 'protocol': row.protocol, 'model_calls': len(row.calls),
            'business_events': len(row.oracle['operations']), 'task_success': row.task_score['success']}
