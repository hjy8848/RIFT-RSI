"""Seeded, boundary-oriented synthetic commerce tasks.

The generator chooses public requests and world facts. Usami derives and
independently certifies each answer; this module never supplies an answer key.
"""
from dataclasses import dataclass
import random

from .families import FAMILIES, synthetic_distribution


BOUNDARY_SAMPLING = {
    'approval_cents': {'boundary_fraction': .4, 'minimum': 1, 'maximum': 50000},
    'return_days': {'boundary_fraction': .4, 'minimum': 0, 'maximum': 90},
    'stock': {'boundary_fraction': .4, 'minimum': 0, 'maximum': 10, 'wide_floor': 5, 'wide_multiplier': 2},
    'extra_charge_cents': {'boundary_fraction': .4, 'minimum': 0, 'maximum': 20000,
                           'wide_floor': 5000, 'wide_multiplier': 2},
    'arrival_days': {'boundary_fraction': .4, 'minimum': 0, 'maximum': 30,
                     'wide_floor': 5, 'wide_multiplier': 2},
}

PRODUCT_FAMILIES = ('wireless_headphones', 'smart_watch', 'tablet', 'camera',
                    'keyboard', 'monitor', 'speaker', 'coffee_maker')


def sample_boundary(rng, threshold, *, minimum, maximum, boundary_fraction=.4):
    """Deterministically sample around a threshold or uniformly over a range."""
    if rng.random() < boundary_fraction:
        return max(minimum, min(maximum, threshold + rng.choice((-1, 0, 1))))
    return rng.randint(minimum, maximum)


def _sample(rng, name, threshold):
    rule = BOUNDARY_SAMPLING[name]
    return sample_boundary(rng, threshold, minimum=rule['minimum'], maximum=rule['maximum'],
                           boundary_fraction=rule['boundary_fraction'])


def _fixed(value):
    return {'low': value, 'high': value}


def _unseal(value):
    if isinstance(value, dict):
        return {key: _unseal(item) for key, item in value.items() if key != 'record_hash'}
    if isinstance(value, (list, tuple)):
        return [_unseal(item) for item in value]
    return value


@dataclass(frozen=True)
class GeneratedTask:
    family_id: str
    seed: int
    spec: object
    build: object
    task: object
    distribution: object


def generate_instance(family_id, seed, *, split='train', near_distractors=0, grid_mode=None):
    """Generate and certify one reproducible F-family task without model calls."""
    from usami_v2.scenarios.builder import build_world
    from usami_v2.scenarios.designer import (RequestDraft, ScenarioDraft, TaskDraft,
                                             compile_task, task_design_capabilities)
    if (family_id not in FAMILIES or type(seed) is not int or seed < 0
            or type(near_distractors) is not int or not 0 <= near_distractors <= 3
            or grid_mode not in (None, 'one', 'none', 'multiple')
            or (grid_mode is not None and family_id not in ('F6', 'F7'))):
        raise ValueError('known family and nonnegative integer seed required')
    rng = random.Random(seed)
    base = FAMILIES[family_id].instance({}, seed)
    data = _unseal(base.draft.model_dump(mode='json'))
    data['scenario_id'] = f'generated-{family_id}-{seed}'
    for sku in data['skus']:
        family = sku['family']
        if family.startswith('product_') and family[8:].isdigit():
            sku['family'] = PRODUCT_FAMILIES[(seed + int(family[8:])) % len(PRODUCT_FAMILIES)]
    limit = rng.randint(4000, 30000)
    data['refund_approval_limit_cents'] = limit
    warehouse_count = rng.randint(2, 4)
    for index in range(2, warehouse_count):
        data['warehouses'].append({'alias': f'extra_{index}', 'delivery_days': _fixed(1),
                                   'delivery_fee_cents': _fixed(0)})
    for warehouse in data['warehouses']:
        warehouse['delivery_days'] = _fixed(rng.randint(1, 12))
        warehouse['delivery_fee_cents'] = _fixed(rng.randint(0, 500))
    return_days = rng.randint(7, 60)
    target_skus = {line['original_sku'] for order in data['orders'] for line in order['lines']
                   if line['alias'].startswith('line_')}
    for sku in data['skus']:
        if sku['alias'] in target_skus:
            sku['return_days'] = return_days
    requests = []
    challenge = base.challenges[0]
    shared_credit = _sample(rng, 'approval_cents', limit) if family_id == 'F11' else None
    for order in data['orders']:
        order['delivered_days_ago'] = _fixed(
            _sample(rng, 'return_days', return_days) if family_id == 'F4' else rng.randint(0, return_days))
        order['ordered_days_ago'] = order['delivered_days_ago']['low'] + rng.randint(1, 7)
        if family_id == 'F5':
            order['fulfillment_status'] = rng.choice(('unshipped', 'shipped', 'delivered'))
        for line in order['lines']:
            if not line['alias'].startswith('line_'):
                continue
            quantity = rng.randint(1, 5)
            requested = rng.randint(1, quantity)
            if family_id == 'F5':
                requested = quantity
            # Refund amount is unit credit times requested quantity.
            credit = (shared_credit if shared_credit is not None else
                      _sample(rng, 'approval_cents', limit) if family_id in ('F1', 'F2', 'F3', 'F9', 'F10')
                      else rng.randint(1000, max(1000, limit // 2)))
            credit = max(requested, credit)
            unit, remainder = divmod(credit, requested)
            line['quantity'] = _fixed(quantity)
            # Usami assigns the remainder to the first purchased units. This
            # preserves the sampled boundary exactly even for partial returns.
            line['paid_cents'] = _fixed(unit * quantity + remainder)
            for sku in data['skus']:
                if sku['alias'] == line['original_sku']:
                    sku['price_cents'] = _fixed(unit)
            goal_id = 'goal_' + line['alias'].split('_')[-1]
            reference = rng.choice(('attributes', 'price', 'order_date') if family_id == 'F5'
                                   else ('attributes', 'delivery_age', 'price', 'order_date'))
            if family_id == 'F5':
                requests.append(RequestDraft(goal_id=goal_id, line_alias=line['alias'], action='cancel',
                                             quantity=quantity, reference=reference))
            else:
                source = next(goal for goal in challenge.remedy_goals if goal.goal_id == goal_id)
                action = source.allowed_remedies[0]
                requests.append(RequestDraft(goal_id=goal_id, line_alias=line['alias'], action=action,
                    quantity=requested, required_attributes=source.required_attributes,
                    maximum_extra_charge_cents=source.maximum_extra_charge_cents,
                    latest_arrival_within_days=source.latest_arrival_within_days, reference=reference))
    if family_id == 'F8':
        quantity = requests[0].quantity
        rule = BOUNDARY_SAMPLING['stock']
        amount = sample_boundary(rng, quantity, minimum=0,
            maximum=max(rule['wide_floor'], quantity * rule['wide_multiplier'] - 1),
            boundary_fraction=rule['boundary_fraction'])
        sku_alias = requests[0].line_alias.replace('line_', 'target_')
        for warehouse in data['warehouses']:
            row = next((row for row in data['stock'] if row['sku_alias'] == sku_alias
                        and row['warehouse_alias'] == warehouse['alias']), None)
            if row is None:
                row = {'sku_alias': sku_alias, 'warehouse_alias': warehouse['alias']}
                data['stock'].append(row)
            row['quantity'] = _fixed(amount)
    # For replacements the candidate price, route fee, quantity and Customer
    # bounds jointly decide feasibility; keep a stock-bearing fastest route.
    if family_id in ('F6', 'F7'):
        request = requests[0]
        target = next(row for row in data['skus'] if row['alias'] == 'replacement')
        target_line = next(line for order in data['orders'] for line in order['lines'] if line['alias'] == request.line_alias)
        paid, purchased = target_line['paid_cents']['low'], target_line['quantity']['low']
        credit = paid // purchased * request.quantity + min(request.quantity, paid % purchased)
        price = max(1, (credit + request.quantity - 1) // request.quantity + rng.randint(500, 5000))
        target['price_cents'] = _fixed(price)
        for warehouse in data['warehouses']:
            row = next((row for row in data['stock'] if row['sku_alias'] == 'replacement'
                        and row['warehouse_alias'] == warehouse['alias']), None)
            if row is None:
                row = {'sku_alias': 'replacement', 'warehouse_alias': warehouse['alias']}
                data['stock'].append(row)
            row['quantity'] = _fixed(request.quantity + rng.randint(0, 4))
        updates = request.model_dump(mode='json', exclude={'record_hash'})
        # A bounded catalog grid gives the customer multiple alternatives to
        # compare. Stock is sampled independently for each variant/warehouse.
        target['attributes'] = [{'name': 'color', 'value': 'blue'},
                                {'name': 'storage', 'value': 'standard'},
                                {'name': 'model', 'value': f'variant_0_{seed}'}]
        updates['required_attributes'] = [{'name': 'color', 'value': 'blue'}]
        for index in range(1, rng.randint(3, 8)):
            alias = f'replacement_grid_{index}'
            color = rng.choice(('blue', 'red', 'green'))
            storage = rng.choice(('standard', 'large', 'compact'))
            premium = {'blue': 0, 'red': 150, 'green': 300}[color] + {
                'standard': 0, 'large': 1500, 'compact': 400}[storage]
            data['skus'].append({'alias': alias, 'family': target['family'],
                'attributes': [{'name': 'color', 'value': color}, {'name': 'storage', 'value': storage},
                               {'name': 'model', 'value': f'variant_{index}_{seed}'}],
                'price_cents': _fixed(max(1, price + premium + rng.randint(-200, 200)))})
            for warehouse in data['warehouses']:
                data['stock'].append({'sku_alias': alias, 'warehouse_alias': warehouse['alias'],
                                      'quantity': _fixed(rng.randint(0, request.quantity + 2))})
        # Place the threshold against the best actually feasible SKU, not a
        # preselected variant. Otherwise another cheaper/faster option can
        # silently erase the intended boundary.
        option_terms = []
        for sku in data['skus']:
            if not (sku['alias'] == 'replacement' or sku['alias'].startswith('replacement_grid_')):
                continue
            if not any(row['name'] == 'color' and row['value'] == 'blue' for row in sku['attributes']):
                continue
            routes = []
            for warehouse in data['warehouses']:
                stock = next(row['quantity']['low'] for row in data['stock']
                             if row['sku_alias'] == sku['alias'] and row['warehouse_alias'] == warehouse['alias'])
                if stock >= request.quantity:
                    routes.append((warehouse['delivery_days']['low'], warehouse['delivery_fee_cents']['low'], warehouse['alias']))
            if routes:
                days, fee, _ = min(routes)
                option_terms.append((max(0, sku['price_cents']['low'] * request.quantity + fee - credit), days))
        if family_id == 'F6':
            threshold = min(row[0] for row in option_terms)
            rule = BOUNDARY_SAMPLING['extra_charge_cents']
            updates['maximum_extra_charge_cents'] = sample_boundary(
                rng, threshold, minimum=0, maximum=max(rule['wide_floor'], threshold * rule['wide_multiplier']),
                boundary_fraction=rule['boundary_fraction'])
        else:
            threshold = min(row[1] for row in option_terms)
            rule = BOUNDARY_SAMPLING['arrival_days']
            updates['latest_arrival_within_days'] = sample_boundary(
                rng, threshold, minimum=0, maximum=max(rule['wide_floor'], threshold * rule['wide_multiplier']),
                boundary_fraction=rule['boundary_fraction'])
        if grid_mode is not None:
            # Controlled grids isolate one violated customer constraint per
            # distractor, so finding the unique choice requires comparisons.
            variants = [row for row in data['skus'] if row['alias'] == 'replacement'
                        or row['alias'].startswith('replacement_grid_')]
            if grid_mode == 'one' and family_id == 'F7':
                for warehouse in data['warehouses']:
                    warehouse['delivery_days'] = _fixed(1 if warehouse['alias'] == 'fast' else 7)
            for index, sku in enumerate(variants):
                color = next(row for row in sku['attributes'] if row['name'] == 'color')
                if grid_mode == 'one':
                    violation = None if index == 0 else ('color', 'terms', 'stock')[(index - 1) % 3]
                    color['value'] = 'red' if violation == 'color' else 'blue'
                    if family_id == 'F6':
                        baseline_price = max(1, credit // request.quantity)
                        sku['price_cents'] = _fixed(
                            (credit + 5000) // request.quantity + 1000 if violation == 'terms' else baseline_price)
                else:
                    violation = None
                    color['value'] = 'blue' if index == 0 or (grid_mode == 'multiple' and index == 1) else 'red'
                for row in data['stock']:
                    if row['sku_alias'] == sku['alias']:
                        empty = (grid_mode == 'none' and index == 0) or violation == 'stock'
                        if grid_mode == 'one' and family_id == 'F7' and violation == 'terms':
                            empty = row['warehouse_alias'] == 'fast'
                        row['quantity'] = _fixed(0 if empty else request.quantity + 1)
            if family_id == 'F6':
                updates['maximum_extra_charge_cents'] = 5000 if grid_mode == 'one' else 200000
            else:
                updates['latest_arrival_within_days'] = 3 if grid_mode == 'one' else 30
        requests[0] = RequestDraft.model_validate(updates)
    if near_distractors:
        from copy import deepcopy
        request = requests[0]
        target_order = next(order for order in data['orders']
                            if any(line['alias'] == request.line_alias for line in order['lines']))
        target_line = next(line for line in target_order['lines'] if line['alias'] == request.line_alias)
        target_sku = next(sku for sku in data['skus'] if sku['alias'] == target_line['original_sku'])
        can_use_age = target_order.get('fulfillment_status', 'delivered') == 'delivered'
        if can_use_age:
            updated = request.model_dump(mode='json', exclude={'record_hash'})
            updated['reference'] = 'delivery_age'
            requests[0] = RequestDraft.model_validate(updated)
        for index in range(near_distractors):
            other_line = deepcopy(target_line)
            other_line['alias'] = f'near_line_{index}'
            age = (target_sku.get('return_days', 30) + index + 1
                   if target_order['delivered_days_ago']['low'] <= target_sku.get('return_days', 30)
                   else index)
            if index == 1 or not can_use_age:
                other_sku = deepcopy(target_sku)
                other_sku['alias'] = f'near_sku_{index}'
                attributes = other_sku.setdefault('attributes', [])
                if attributes:
                    attributes[0]['value'] += f'_near_{index}'
                else:
                    attributes.append({'name': 'variant', 'value': f'near_{index}'})
                data['skus'].append(other_sku)
                other_line['original_sku'] = other_sku['alias']
            other_order = {'alias': f'near_order_{index}', 'customer_alias': 'customer',
                           'fulfillment_status': 'delivered', 'delivered_days_ago': _fixed(age),
                           'lines': [other_line]}
            data['orders'].append(other_order)
            if index == 2 and can_use_age and not target_sku.get('return_required', False):
                other_line['quantity'] = _fixed(1)
                other_line['paid_cents'] = _fixed(min(limit, target_line['paid_cents']['low']))
                # Choose an in-window historical line so the Engine itself
                # creates the already-settled distractor.
                other_order['delivered_days_ago'] = _fixed(0 if target_order['delivered_days_ago']['low'] else 1)
                data.setdefault('history', []).append({'operation': 'return_delivered_order_items',
                                                       'line_alias': other_line['alias'], 'quantity': 1})
    distribution = synthetic_distribution(split)
    spec = ScenarioDraft.model_validate(data).compile(distribution)
    build = build_world(spec, seed)
    if build.status != 'SAT':
        raise ValueError('generated world was not constructed: ' + build.reason)
    capabilities = task_design_capabilities(spec, build, return_receipt_event_configured=family_id == 'F9')
    task = compile_task(TaskDraft(task_id=f'generated-{family_id}-{seed}', customer_alias='customer',
        requests=tuple(requests), style_seed=seed), spec, build, distribution, capabilities=capabilities)
    return GeneratedTask(family_id, seed, spec, build, task, distribution)


def generate_composite(seed, *, count=None, coupling='none', with_history=False, split='train'):
    """Generate 1–4 atomic requests, optionally coupling stock or purchased units."""
    from usami_v2.scenarios.builder import build_world
    from usami_v2.scenarios.designer import RequestDraft, ScenarioDraft, TaskDraft, compile_task
    from usami_v2.scenarios.schema import Attribute
    if type(seed) is not int or seed < 0 or coupling not in ('none', 'shared_inventory', 'same_line'):
        raise ValueError('invalid composite generation arguments')
    rng = random.Random(seed)
    count = rng.randint(1, 4) if count is None else count
    if type(count) is not int or not 1 <= count <= 4 or (coupling != 'none' and count < 2):
        raise ValueError('composite request count must be 1–4 and coupled tasks need at least two')
    data = {'scenario_id': f'composite-{seed}', 'customer_aliases': ['customer'],
            'skus': [], 'orders': [], 'warehouses': [
                {'alias': 'fast', 'delivery_days': _fixed(2), 'delivery_fee_cents': _fixed(0)},
                {'alias': 'slow', 'delivery_days': _fixed(5), 'delivery_fee_cents': _fixed(0)}],
            'stock': [], 'default_stock': _fixed(0), 'refund_approval_limit_cents': 10000}
    requests = []
    line_count = count - 1 if coupling == 'same_line' else count
    actions = []
    for index in range(line_count):
        if coupling == 'shared_inventory' and index in (0, line_count - 1):
            action = 'exchange'
        elif coupling == 'same_line' and index == 0:
            action = 'refund'
        else:
            action = rng.choice(('refund', 'reship', 'exchange', 'cancel'))
        actions.append(action)
    if with_history:
        actions[0] = 'refund' if coupling != 'shared_inventory' else 'exchange'
    for index, action in enumerate(actions):
        shared = coupling == 'shared_inventory' and index in (0, line_count - 1)
        family = PRODUCT_FAMILIES[seed % len(PRODUCT_FAMILIES)] if shared else PRODUCT_FAMILIES[(seed + index + 1) % len(PRODUCT_FAMILIES)]
        sku_alias, line_alias = f'original_{index}', f'line_{index}'
        quantity = 2 if (index == 0 and (coupling == 'same_line' or with_history)) else 1
        price = rng.randint(2000, 8000)
        data['skus'].append({'alias': sku_alias, 'family': family,
            'attributes': [{'name': 'model', 'value': f'original_{index}'}], 'price_cents': _fixed(price)})
        age = index + 1
        data['orders'].append({'alias': f'order_{index}', 'customer_alias': 'customer',
            'delivered_days_ago': _fixed(age), 'ordered_days_ago': age + 5,
            'fulfillment_status': 'unshipped' if action == 'cancel' else 'delivered',
            'lines': [{'alias': line_alias, 'original_sku': sku_alias, 'quantity': _fixed(quantity),
                       'paid_cents': _fixed(price * quantity)}]})
        reference = rng.choice(('attributes', 'price', 'order_date') if action == 'cancel'
                               else ('attributes', 'delivery_age', 'price', 'order_date'))
        attrs = ()
        if action == 'exchange' or (coupling == 'same_line' and index == 0):
            replacement = 'shared_replacement' if shared else f'replacement_{index}'
            if not any(row['alias'] == replacement for row in data['skus']):
                data['skus'].append({'alias': replacement, 'family': family,
                    'attributes': [{'name': 'model', 'value': replacement}], 'price_cents': _fixed(price + 500)})
                for warehouse in ('fast', 'slow'):
                    data['stock'].append({'sku_alias': replacement, 'warehouse_alias': warehouse,
                        'quantity': _fixed(1 if shared and warehouse == 'fast' else (0 if shared else count + 2))})
            if action == 'exchange':
                attrs = (Attribute(name='model', value=replacement),)
        if action == 'reship':
            data['stock'].append({'sku_alias': sku_alias, 'warehouse_alias': 'fast', 'quantity': _fixed(3)})
        requests.append(RequestDraft(goal_id=f'goal_{index}', line_alias=line_alias, action=action,
                                     quantity=quantity if action == 'cancel' else 1,
                                     required_attributes=attrs, reference=reference))
        if coupling == 'same_line' and index == 0:
            requests.append(RequestDraft(goal_id='same_line_exchange', line_alias=line_alias,
                action='exchange', quantity=1, required_attributes=(Attribute(name='model', value='replacement_0'),),
                reference=reference))
    if with_history:
        data['history'] = [{'operation': 'return_delivered_order_items', 'line_alias': 'line_0', 'quantity': 1}]
        # The historical settlement must leave enough units for every goal.
        first_line = data['orders'][0]['lines'][0]
        if coupling == 'same_line':
            first_line['quantity'] = _fixed(3)
            first_line['paid_cents'] = _fixed(data['skus'][0]['price_cents']['low'] * 3)
    distribution = synthetic_distribution(split)
    spec = ScenarioDraft.model_validate(data).compile(distribution)
    build = build_world(spec, seed)
    if build.status != 'SAT':
        raise ValueError('generated composite world was not constructed: ' + build.reason)
    task = compile_task(TaskDraft(task_id=f'composite-{seed}', customer_alias='customer',
        requests=tuple(requests), style_seed=seed, sequential=coupling == 'shared_inventory'),
        spec, build, distribution)
    return GeneratedTask('composite', seed, spec, build, task, distribution)
