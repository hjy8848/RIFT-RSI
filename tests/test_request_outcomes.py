import pytest

from rift_usami_fixtures import distribution_and_slot, scenario_draft
from usami_v2.scenarios.builder import build_world
from usami_v2.scenarios.designer import RequestDraft, TaskDraft, compile_task
from usami_v2.scenarios.tasks import check_task_scope, parse_task_spec


def compile_refund(limit, seed=3, style=7):
    distribution, _ = distribution_and_slot(1)
    data = scenario_draft().model_dump(mode='json', exclude={'record_hash'})
    data['refund_approval_limit_cents'] = limit
    from usami_v2.scenarios.designer import ScenarioDraft
    spec = ScenarioDraft.model_validate(data).compile(distribution)
    build = build_world(spec, seed)
    draft = TaskDraft(task_id='derive-refund', customer_alias='customer', style_seed=style,
        requests=(RequestDraft(goal_id='refund', line_alias='damaged', action='refund', quantity=1),))
    return build.world, compile_task(draft, spec, build, distribution)


def test_refund_outcome_crosses_approval_threshold_without_wording_leak():
    world_ok, ok = compile_refund(20000)
    world_no, no = compile_refund(19999)
    assert len(ok.remedy_goals) == 1 and not ok.refusal_goals
    assert no.refusal_goals[0].reason_code == 'APPROVAL_REQUIRED'
    assert ok.customer_request == no.customer_request
    assert check_task_scope(world_ok, parse_task_spec(ok.model_dump(mode='json'))) == ok
    assert check_task_scope(world_no, parse_task_spec(no.model_dump(mode='json'))) == no


def test_style_is_deterministic_and_cannot_change_goal_binding():
    world, task = compile_refund(20000)
    _, same = compile_refund(20000)
    assert task.customer_request == same.customer_request
    changed = task.model_dump(mode='json', exclude={'record_hash'})
    changed['request_bundle'].pop('record_hash')
    changed['request_bundle']['requests'][0].pop('record_hash')
    changed['request_bundle']['requests'][0]['quantity'] = 2
    with pytest.raises(ValueError):
        check_task_scope(world, parse_task_spec(changed))


def test_renderer_varies_sentence_style_across_seeds():
    from collections import Counter
    from usami_v2.scenarios.tasks import render_customer_request_v2
    world, task = compile_refund(20000)
    intent = task.request_bundle.requests
    counts = Counter(render_customer_request_v2(world, task.customer_id, intent, seed) for seed in range(1000))
    assert max(counts.values()) < 100
    assert len(counts) >= 12


def test_exchange_and_reship_wording_never_overlaps():
    from usami_v2.scenarios.tasks import CustomerRequestIntent, render_customer_request_v2
    world, task = compile_refund(20000)
    line_id = task.request_bundle.requests[0].line_id

    def texts(action):
        intent = CustomerRequestIntent(goal_id='g', line_id=line_id, action=action, quantity=1)
        return {render_customer_request_v2(world, task.customer_id, (intent,), seed).split('\n')[1]
                for seed in range(200)}
    exchange, reship = texts('exchange'), texts('reship')
    assert all('different variant' in text and 'same product' not in text for text in exchange)
    assert all('same product' in text and 'different variant' not in text for text in reship)
    assert not any('replace' in text.casefold() for text in exchange | reship)


def test_new_requests_are_exclusive_with_legacy_goals():
    from usami_v2.scenarios.designer import RemedyDraft
    with pytest.raises(ValueError, match='cannot be combined'):
        TaskDraft(task_id='x', customer_alias='customer',
                  requests=(RequestDraft(goal_id='a', line_alias='damaged', action='refund', quantity=1),),
                  remedy_goals=(RemedyDraft(goal_id='b', line_alias='damaged', quantity=1,
                                            allowed_remedies=('refund',)),))


def test_symbolic_request_is_grounded_before_native_derivation():
    from usami_v2.scenarios.grounding import AbstractChallenge, LineSelector, compile_grounded, ground_challenge
    distribution, _ = distribution_and_slot(1)
    spec = scenario_draft().compile(distribution)
    build = build_world(spec, 1)
    challenge = AbstractChallenge(task_id='symbolic', customer_alias='customer', hypothesis='refund',
        line_selectors=(LineSelector(alias='item', family='cable'),),
        requests=(RequestDraft(goal_id='request', line_alias='item', action='refund', quantity=1),))
    grounded = ground_challenge(challenge, (('world', spec, build),))
    assert grounded.status == 'GROUNDED'
    task = compile_grounded(grounded.candidates[0], spec, build, distribution)
    assert task.remedy_goals[0].line_id == dict(build.private_aliases)['accessory']


@pytest.mark.parametrize('family,knobs', [
    *[(f'F{i}', {}) for i in range(1, 12)],
    ('F3', {'amount_ratio': 1.1, 'refusal': True}),
    ('F4', {'age_days': 31, 'refusal': True}),
    ('F5', {'fulfillment_status': 'shipped', 'refusal': True}),
    ('F6', {'extra_charge_cents': 3000, 'customer_limit_cents': 2000, 'refusal': True}),
    ('F8', {'stock': 0, 'refusal': True}),
    ('F8', {'evidence_valid': False, 'refusal': True}),
])
def test_family_outcomes_are_derived_from_request(family, knobs):
    from rift.benchmark.families import FAMILIES, synthetic_distribution
    from usami_v2.scenarios.designer import task_design_capabilities
    instance = FAMILIES[family].instance(knobs, 1)
    distribution = synthetic_distribution()
    spec = instance.draft.compile(distribution)
    build = build_world(spec, 1)
    challenge = instance.challenges[0]
    requests = []
    for goal in challenge.remedy_goals:
        requests.append(RequestDraft(goal_id=goal.goal_id,
            line_alias=goal.line_alias.replace('item_', 'line_'), action=goal.allowed_remedies[0],
            quantity=goal.quantity, required_attributes=goal.required_attributes,
            maximum_extra_charge_cents=goal.maximum_extra_charge_cents,
            latest_arrival_within_days=goal.latest_arrival_within_days))
    for goal in challenge.cancellation_goals:
        requests.append(RequestDraft(goal_id=goal.goal_id, line_alias=goal.line_alias.replace('item_', 'line_'),
                                     action='cancel', quantity=1))
    for goal in challenge.refusal_goals:
        requests.append(RequestDraft(goal_id=goal.goal_id, line_alias=goal.line_alias.replace('item_', 'line_'),
            action=goal.requested_action, quantity=1, maximum_extra_charge_cents=goal.maximum_extra_charge_cents))
    capabilities = task_design_capabilities(spec, build, return_receipt_event_configured=family == 'F9')
    task = compile_task(TaskDraft(task_id='derived', customer_alias='customer', requests=tuple(requests)),
                        spec, build, distribution, capabilities=capabilities)
    assert len(task.remedy_goals) == len(challenge.remedy_goals)
    assert len(task.cancellation_goals) == len(challenge.cancellation_goals)
    assert [row.reason_code for row in task.refusal_goals] == [row.reason_code for row in challenge.refusal_goals]


def test_decision_signature_ignores_names_but_detects_policy_outcome():
    from rift.benchmark.families import FAMILIES, synthetic_distribution
    from rift.probes.fingerprint import decision_fingerprint, same_decision
    distribution = synthetic_distribution()
    signatures = []
    for seed, amount in ((1, .8), (2, .8), (1, 1.1)):
        instance = FAMILIES['F3'].instance({'amount_ratio': amount, 'refusal': amount > 1}, seed)
        spec = instance.draft.compile(distribution)
        build = build_world(spec, seed)
        request = RequestDraft(goal_id='goal_0', line_alias='line_0', action='refund', quantity=1)
        task = compile_task(TaskDraft(task_id='signature', customer_alias='customer', requests=(request,)),
                            spec, build, distribution)
        signatures.append(decision_fingerprint(spec, build, task))
    assert same_decision(signatures[0], signatures[1])
    assert not same_decision(signatures[0], signatures[2])
    assert signatures[0].goals[0].outcome == 'success'
    assert signatures[2].goals[0].outcome == 'APPROVAL_REQUIRED'


def test_decision_signature_ignores_product_names():
    from rift.benchmark.generator import _unseal
    from rift.probes.fingerprint import decision_fingerprint, same_decision
    from usami_v2.scenarios.designer import ScenarioDraft
    distribution, _ = distribution_and_slot(1)
    data = _unseal(scenario_draft().model_dump(mode='json'))
    signatures = []
    for family in ('device', 'renamed_product'):
        renamed = _unseal(data)
        for sku in renamed['skus']:
            if sku['family'] == 'device':
                sku['family'] = family
        spec = ScenarioDraft.model_validate(renamed).compile(distribution)
        build = build_world(spec, 1)
        task = compile_task(TaskDraft(task_id='name-test', customer_alias='customer', requests=(
            RequestDraft(goal_id='refund', line_alias='damaged', action='refund', quantity=1),)),
            spec, build, distribution)
        signatures.append(decision_fingerprint(spec, build, task))
    assert same_decision(*signatures)


def test_public_reference_can_disambiguate_same_sku_by_delivery_or_order_date():
    from copy import deepcopy
    from rift.benchmark.generator import _unseal
    from usami_v2.scenarios.designer import ScenarioDraft
    distribution, _ = distribution_and_slot(1)
    data = _unseal(scenario_draft().model_dump(mode='json'))
    first = data['orders'][0]
    first['ordered_days_ago'] = 10
    second = deepcopy(first)
    second.update(alias='other_order', delivered_days_ago={'low': 4, 'high': 4}, ordered_days_ago=8)
    second['lines'] = [deepcopy(first['lines'][0])]
    second['lines'][0]['alias'] = 'other_line'
    data['orders'].append(second)
    spec = ScenarioDraft.model_validate(data).compile(distribution)
    build = build_world(spec, 1)
    def compile_with(reference):
        request = RequestDraft(goal_id='refund', line_alias='damaged', action='refund', quantity=1,
                               reference=reference)
        return compile_task(TaskDraft(task_id='reference', customer_alias='customer', requests=(request,)),
                            spec, build, distribution)
    with pytest.raises(ValueError, match='AMBIGUOUS_TARGET'):
        compile_with('attributes')
    age = compile_with('delivery_age')
    dated = compile_with('order_date')
    assert 'delivered about 1 day ago' in age.customer_request
    assert 'order placed on 2026-01-05' in dated.customer_request
    assert verify_build_for_reference(spec, build)


def verify_build_for_reference(spec, build):
    from usami_v2.scenarios.builder import verify_build
    return verify_build(spec, build)['verified']


def test_same_line_disjoint_remedies_use_v4_and_score_separately():
    from rift.benchmark.generator import _unseal
    from usami_v2.scenarios.designer import ScenarioDraft
    from usami_v2.scenarios.task_certification import CustomerScenario, certify_task
    from usami_v2.scenarios.tasks import delegated_line_ids, evaluate_state_goals
    from usami_v2.commerce.models import World
    distribution, _ = distribution_and_slot(1)
    data = _unseal(scenario_draft().model_dump(mode='json'))
    data['orders'][0]['lines'][0]['quantity'] = {'low': 3, 'high': 3}
    data['orders'][0]['lines'][0]['paid_cents'] = {'low': 60000, 'high': 60000}
    spec = ScenarioDraft.model_validate(data).compile(distribution)
    build = build_world(spec, 1)
    requests = (RequestDraft(goal_id='refund', line_alias='damaged', action='refund', quantity=1),
                RequestDraft(goal_id='exchange', line_alias='damaged', action='exchange', quantity=1))
    task = compile_task(TaskDraft(task_id='multi', customer_alias='customer', requests=requests),
                        spec, build, distribution)
    assert task.protocol == 'commerce-task-v4'
    assert len(delegated_line_ids(task)) == 1
    assert parse_task_spec(task.model_dump(mode='json')) == task
    scenario = CustomerScenario(customer_id=task.customer_id, delegated_line_ids=delegated_line_ids(task))
    proof = certify_task(build.world, task, scenario)
    assert proof.status == 'CERTIFIED'
    attempt = next(row for row in proof.attempts if row.status == 'WITNESS')
    operations = [step for step in attempt.steps if step.kind == 'operation']
    final = World.model_validate(operations[-1].evidence['trace']['final_state'])
    assert [row.status for row in evaluate_state_goals(build.world, final, task)] == ['PASS', 'PASS']
    refund = next(step for step in operations
                  if step.evidence['request']['operation']['name'] == 'return_delivered_order_items')
    partial = World.model_validate(refund.evidence['trace']['final_state'])
    assert [row.status for row in evaluate_state_goals(build.world, partial, task)] == ['PASS', 'FAIL']
    exchange_only = RequestDraft(goal_id='both', line_alias='damaged', action='exchange', quantity=2)
    other_task = compile_task(TaskDraft(task_id='two-exchanges', customer_alias='customer', requests=(exchange_only,)),
                              spec, build, distribution)
    other_proof = certify_task(build.world, other_task,
        CustomerScenario(customer_id=other_task.customer_id, delegated_line_ids=delegated_line_ids(other_task)))
    other_witness = next(row for row in other_proof.attempts if row.status == 'WITNESS')
    other_operations = [step for step in other_witness.steps if step.kind == 'operation']
    twice_exchanged = World.model_validate(other_operations[-1].evidence['trace']['final_state'])
    assert [row.status for row in evaluate_state_goals(build.world, twice_exchanged, task)] == ['FAIL', 'FAIL']
    overlapping = (requests[0], RequestDraft(goal_id='second', line_alias='damaged', action='refund', quantity=1))
    with pytest.raises(ValueError, match='disjoint remedies'):
        compile_task(TaskDraft(task_id='invalid', customer_alias='customer', requests=overlapping),
                     spec, build, distribution)
    from usami_v2.scenarios.grounding import AbstractChallenge, LineSelector, compile_grounded, ground_challenge
    symbolic = AbstractChallenge(task_id='symbolic-multi', customer_alias='customer', hypothesis='two remedies',
        line_selectors=(LineSelector(alias='item', family='device', minimum_quantity=3),),
        requests=(RequestDraft(goal_id='refund', line_alias='item', action='refund', quantity=1),
                  RequestDraft(goal_id='exchange', line_alias='item', action='exchange', quantity=1)))
    candidate = ground_challenge(symbolic, (('world', spec, build),)).candidates[0]
    assert compile_grounded(candidate, spec, build, distribution).protocol == 'commerce-task-v4'


def test_engine_history_partial_refund_is_replayed_and_visible():
    from rift.benchmark.generator import _unseal
    from usami_v2.scenarios.builder import verify_build
    from usami_v2.scenarios.designer import ScenarioDraft
    from usami_v2.commerce.observations import _public_line_rows
    distribution, _ = distribution_and_slot(1)
    data = _unseal(scenario_draft().model_dump(mode='json'))
    data['orders'][0]['lines'][0]['quantity'] = {'low': 3, 'high': 3}
    data['orders'][0]['lines'][0]['paid_cents'] = {'low': 24000, 'high': 24000}
    data['history'] = [{'operation': 'return_delivered_order_items', 'line_alias': 'damaged', 'quantity': 1}]
    spec = ScenarioDraft.model_validate(data).compile(distribution)
    build = build_world(spec, 1)
    assert build.status == 'SAT', build.reason
    assert verify_build(spec, build)['verified']
    line_id = dict(build.private_aliases)['damaged']
    line = build.world.lines[line_id]
    public = next(row for row in _public_line_rows(build.world, line.order_id) if row['id'] == line_id)
    assert public['settled_quantity'] == 1
    assert not build.world.delegations and not build.world.consents
    def compile_quantity(quantity):
        request = RequestDraft(goal_id='again', line_alias='damaged', action='refund', quantity=quantity)
        return compile_task(TaskDraft(task_id='after-history', customer_alias='customer', requests=(request,)),
                            spec, build, distribution)
    assert compile_quantity(2).remedy_goals
    assert compile_quantity(3).refusal_goals[0].reason_code == 'PURCHASED_QUANTITY_UNAVAILABLE'


def test_failed_history_has_explicit_reason_and_default_hash_fields_are_omitted():
    from rift.benchmark.generator import _unseal
    from usami_v2.scenarios.designer import ScenarioDraft
    distribution, _ = distribution_and_slot(1)
    data = _unseal(scenario_draft().model_dump(mode='json'))
    spec = ScenarioDraft.model_validate(data).compile(distribution)
    baseline = build_world(spec, 1)
    assert 'history' not in spec.model_dump(mode='json')
    assert 'observation_protocol' not in baseline.world.model_dump(mode='json')
    data['history'] = [{'operation': 'return_delivered_order_items', 'line_alias': 'damaged', 'quantity': 2}]
    bad = build_world(ScenarioDraft.model_validate(data).compile(distribution), 1)
    assert bad.status == 'UNSAT'
    assert 'HISTORY_STEP_FAILED:0:PURCHASED_QUANTITY_UNAVAILABLE' in bad.reason


@pytest.mark.parametrize('received', [False, True])
def test_history_return_pending_or_received_can_be_certified(received):
    from rift.benchmark.generator import _unseal
    from usami_v2.scenarios.designer import ScenarioDraft, task_design_capabilities
    from usami_v2.scenarios.builder import verify_build
    from usami_v2.commerce.observations import _public_line_rows
    distribution, _ = distribution_and_slot(1)
    data = _unseal(scenario_draft().model_dump(mode='json'))
    next(sku for sku in data['skus'] if sku['alias'] == 'original')['return_required'] = True
    data['history'] = [{'operation': 'request_item_return', 'line_alias': 'damaged', 'quantity': 1}]
    if received:
        data['history'].append({'operation': 'return_received', 'line_alias': 'damaged', 'quantity': 1})
    spec = ScenarioDraft.model_validate(data).compile(distribution)
    build = build_world(spec, 1)
    assert build.status == 'SAT', build.reason
    assert verify_build(spec, build)['verified']
    line_id = dict(build.private_aliases)['damaged']
    public = next(row for row in _public_line_rows(build.world, build.world.lines[line_id].order_id)
                  if row['id'] == line_id)
    assert public['return_status'] == (['received'] if received else ['scheduled'])
    capabilities = task_design_capabilities(spec, build, return_receipt_event_configured=not received)
    request = RequestDraft(goal_id='refund', line_alias='damaged', action='refund', quantity=1)
    task = compile_task(TaskDraft(task_id='pending-return', customer_alias='customer', requests=(request,)),
                        spec, build, distribution, capabilities=capabilities)
    assert task.remedy_goals


def test_history_exchange_replays_native_inventory_and_settlement():
    from rift.benchmark.generator import _unseal
    from usami_v2.scenarios.builder import verify_build
    from usami_v2.scenarios.designer import ScenarioDraft
    distribution, _ = distribution_and_slot(1)
    data = _unseal(scenario_draft().model_dump(mode='json'))
    data['history'] = [{'operation': 'exchange_delivered_order_items', 'line_alias': 'damaged',
                        'target_sku_alias': 'correct_choice_D', 'quantity': 1}]
    spec = ScenarioDraft.model_validate(data).compile(distribution)
    build = build_world(spec, 1)
    assert build.status == 'SAT', build.reason
    assert len(build.world.settlements) == 1 and len(build.world.fulfillments) == 1
    assert verify_build(spec, build)['verified']


def test_ordered_requests_certify_shared_inventory_conflict():
    from copy import deepcopy
    from rift.benchmark.families import FAMILIES, synthetic_distribution
    from rift.benchmark.generator import _unseal
    from usami_v2.scenarios.designer import ScenarioDraft, TaskCapabilityUnsupported
    from usami_v2.scenarios.schema import Attribute
    instance = FAMILIES['F6'].instance({}, 4)
    data = _unseal(instance.draft.model_dump(mode='json'))
    original = next(row for row in data['skus'] if row['alias'] == 'target_0')
    other = deepcopy(original)
    other['alias'] = 'target_other'
    other['attributes'][0]['value'] = 'other_model'
    data['skus'].append(other)
    order = deepcopy(data['orders'][0])
    order['alias'] = 'order_other'
    order['lines'][0].update(alias='line_other', original_sku='target_other')
    data['orders'].append(order)
    for row in data['stock']:
        if row['sku_alias'] == 'replacement':
            row['quantity'] = {'low': 1 if row['warehouse_alias'] == 'fast' else 0,
                               'high': 1 if row['warehouse_alias'] == 'fast' else 0}
    distribution = synthetic_distribution()
    spec = ScenarioDraft.model_validate(data).compile(distribution)
    build = build_world(spec, 4)
    model = next(row for row in data['skus'] if row['alias'] == 'replacement')['attributes'][0]['value']
    attributes = (Attribute(name='model', value=model),)
    requests = (RequestDraft(goal_id='first', line_alias='line_0', action='exchange', quantity=1,
                             required_attributes=attributes),
                RequestDraft(goal_id='second', line_alias='line_other', action='exchange', quantity=1,
                             required_attributes=attributes))
    with pytest.raises(TaskCapabilityUnsupported, match='jointly certified'):
        compile_task(TaskDraft(task_id='unordered', customer_alias='customer', requests=requests),
                     spec, build, distribution)
    task = compile_task(TaskDraft(task_id='ordered', customer_alias='customer', requests=requests,
                                  sequential=True), spec, build, distribution)
    assert len(task.remedy_goals) == 1
    assert task.refusal_goals[0].reason_code == 'INSUFFICIENT_STOCK'
    assert 'in the order listed' in task.customer_request


def test_ordered_refusal_before_cancellation_is_checked_at_the_right_state():
    from rift.benchmark.families import FAMILIES, synthetic_distribution
    from rift.benchmark.generator import _unseal
    from usami_v2.scenarios.designer import ScenarioDraft
    from usami_v2.scenarios.schema import Attribute
    instance = FAMILIES['F6'].instance({}, 6)
    data = _unseal(instance.draft.model_dump(mode='json'))
    for row in data['stock']:
        if row['sku_alias'] == 'replacement':
            row['quantity'] = {'low': 0, 'high': 0}
    data['orders'].append({'alias': 'restock_order', 'customer_alias': 'customer',
        'fulfillment_status': 'unshipped', 'delivered_days_ago': {'low': 1, 'high': 1},
        'lines': [{'alias': 'restock_line', 'original_sku': 'replacement',
                   'quantity': {'low': 1, 'high': 1}, 'paid_cents': {'low': 9000, 'high': 9000}}]})
    distribution = synthetic_distribution()
    spec = ScenarioDraft.model_validate(data).compile(distribution)
    build = build_world(spec, 6)
    model = next(row for row in data['skus'] if row['alias'] == 'replacement')['attributes'][0]['value']
    requests = (RequestDraft(goal_id='first', line_alias='line_0', action='exchange', quantity=1,
                             required_attributes=(Attribute(name='model', value=model),)),
                RequestDraft(goal_id='second', line_alias='restock_line', action='cancel', quantity=1))
    task = compile_task(TaskDraft(task_id='refusal-first', customer_alias='customer', requests=requests,
                                  sequential=True), spec, build, distribution)
    assert task.refusal_goals[0].reason_code == 'INSUFFICIENT_STOCK'
    assert task.cancellation_goals[0].goal_id == 'second'
    from usami_v2.commerce.models import Operation
    from usami_v2.runtime.runner import _execution
    from usami_v2.scenarios.task_certification import CustomerScenario, _customer_delegation
    from usami_v2.scenarios.tasks import _ordered_refusal_observed, delegated_line_ids
    scenario = CustomerScenario(customer_id=task.customer_id, delegated_line_ids=delegated_line_ids(task))
    state, _ = _customer_delegation(build.world, task, scenario)
    target = build.world.lines[task.refusal_goals[0].line_ids[0]]
    scope = build.world.orders[target.order_id].merchant_id
    read_op = Operation(name='get_order_details', scope_id=scope, resource_id=target.order_id, params={})
    cancel_op = Operation(name='cancel_order', scope_id=scope,
                          resource_id=task.cancellation_goals[0].order_id, params={})
    after_read, read = _execution(state, task.customer_id, read_op, 'read-first', 'reference')
    _, cancel = _execution(after_read, task.customer_id, cancel_op, 'cancel-second', 'reference')
    after_cancel, premature = _execution(state, task.customer_id, cancel_op, 'cancel-first', 'reference')
    _, late_read = _execution(after_cancel, task.customer_id, read_op, 'read-second', 'reference')
    assert _ordered_refusal_observed(task, task.refusal_goals[0], (read, cancel))
    assert not _ordered_refusal_observed(task, task.refusal_goals[0], (premature, late_read))


def test_customer_decline_is_scoped_when_other_requests_can_succeed():
    from rift.benchmark.families import FAMILIES, synthetic_distribution
    from rift.benchmark.generator import _unseal
    from rift.benchmark.usami import default_customer_profile, default_customer_scenario
    from usami_v2.runtime.customer import CustomerDecision, CustomerRuntime
    from usami_v2.scenarios.designer import ScenarioDraft
    from usami_v2.scenarios.refusal_certification import certify_refusal
    from usami_v2.scenarios.schema import Attribute
    base = FAMILIES['F6'].instance({}, 8)
    data = _unseal(base.draft.model_dump(mode='json'))
    data['skus'].append({'alias': 'other', 'family': 'other', 'price_cents': {'low': 1000, 'high': 1000}})
    data['orders'].append({'alias': 'other_order', 'customer_alias': 'customer',
        'delivered_days_ago': {'low': 1, 'high': 1},
        'lines': [{'alias': 'other_line', 'original_sku': 'other', 'quantity': {'low': 1, 'high': 1},
                   'paid_cents': {'low': 1000, 'high': 1000}}]})
    distribution = synthetic_distribution()
    spec = ScenarioDraft.model_validate(data).compile(distribution)
    build = build_world(spec, 8)
    model = next(sku for sku in data['skus'] if sku['alias'] == 'replacement')['attributes'][0]['value']
    requests = (RequestDraft(goal_id='decline', line_alias='line_0', action='exchange', quantity=1,
        required_attributes=(Attribute(name='model', value=model),), maximum_extra_charge_cents=0),
        RequestDraft(goal_id='refund', line_alias='other_line', action='refund', quantity=1))
    task = compile_task(TaskDraft(task_id='mixed-decline', customer_alias='customer',
        requests=requests, sequential=True), spec, build, distribution)
    assert task.refusal_goals[0].reason_code == 'CUSTOMER_DECLINED'
    scenario = default_customer_scenario(task, spec, build)
    profile = default_customer_profile(task, spec, build)
    assert scenario.accepts_proposals and scenario.decline_line_ids == task.refusal_goals[0].line_ids
    assert profile.decline_line_ids == scenario.decline_line_ids
    assert certify_refusal(build.world, task, scenario).full_task_reachability_certified
    runtime = CustomerRuntime(profile, task=task)
    initial = runtime.initialize(build.world, task.customer_request).world
    runtime.apply(initial, CustomerDecision(message='No', declines_proposal=True,
                  declined_line_id=profile.decline_line_ids[0]), event_index=1)
    with pytest.raises(ValueError, match='configured request line'):
        runtime.apply(initial, CustomerDecision(message='No', declines_proposal=True,
                      declined_line_id=dict(build.private_aliases)['other_line']), event_index=1)
    from rift.runtime.ascend_910b3 import _UsamiCustomer
    from usami_v2.runtime.agents import CustomerPrompt
    line = profile.decline_line_ids[0]
    decision = _UsamiCustomer(1).decide(CustomerPrompt(public_request=task.customer_request,
        last_service_message='Please confirm the ' + profile.decline_references[line]), profile)
    assert decision.declines_proposal and decision.declined_line_id == line and not decision.finish
    other = dict(build.private_aliases)['other_line']
    accept = _UsamiCustomer(2).decide(CustomerPrompt(public_request=task.customer_request,
        last_service_message='Please confirm the ' + profile.request_references[other]), profile)
    assert not accept.declines_proposal and 'Yes' in accept.message
    clarify = _UsamiCustomer(3).decide(CustomerPrompt(public_request=task.customer_request,
        last_service_message='Can I proceed with this item?'), profile)
    assert 'Which item' in clarify.message
    from types import SimpleNamespace
    from usami_v2.scenarios.tasks import _ordered_decline_observed
    decline_event = SimpleNamespace(kind='customer_transition', payload={'transition_kind': 'decision',
        'decision': {'declines_proposal': True, 'declined_line_id': line}})
    refund_event = SimpleNamespace(kind='service_tool', payload={'operation': {
        'name': 'return_delivered_order_items', 'resource_id': task.remedy_goals[0].line_id},
        'result': {'success': True}})
    assert _ordered_decline_observed(task, task.refusal_goals[0], (decline_event, refund_event))
    assert not _ordered_decline_observed(task, task.refusal_goals[0], (refund_event, decline_event))
