"""Reproducible synthetic commerce families for curriculum experiments.

These are controlled synthetic assumptions, not a sample from customer traffic.
The caller supplies a DistributionSpec with a synthetic_assumption source.
"""
from dataclasses import dataclass
from hashlib import sha256
import json

from rift.schemas import DataPurpose, Identity, Split
from rift.tasks import Proposal


def _fixed(value):
    return {'low': value, 'high': value}


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def synthetic_distribution(split='train'):
    from usami_v2.scenarios.distribution import BusinessSource, BusinessStratum, DistributionSpec
    source = BusinessSource(source_id='rift-synthetic-families', kind='synthetic_assumption',
        reference='RIFT structured commerce family generators', version='1',
        supported_claims=('controlled synthetic task distribution',))
    return DistributionSpec(distribution_id='rift-synthetic-families-' + split, version=1,
        population_scope='synthetic controlled commerce instances', sources=(source,),
        strata=(BusinessStratum(stratum_id='families', weight=1, weight_basis_source_id=source.source_id,
                                contexts=('synthetic-commerce',)),), split=split)


@dataclass(frozen=True)
class FamilyInstance:
    family_id: str
    knobs: dict
    world_seed: int
    draft: object
    challenges: tuple

    def proposal(self, *, n_tasks=1, ability_id='NEW'):
        if type(n_tasks) is not int or n_tasks != len(self.challenges):
            raise ValueError('the World must contain one distinct challenge instance per requested task')
        rows = []
        for index, source in enumerate(self.challenges):
            value = source.model_dump(mode='json', exclude={'record_hash'})
            value['task_id'] = f'{self.family_id}-task-{self.world_seed}-{index}'
            rows.append(value)
        return Proposal(f'{self.family_id}-{self.world_seed}-{_digest(self.knobs)[:12]}',
            f'controlled synthetic family {self.family_id}', ability_id=ability_id,
            world_draft=self.draft, challenges=tuple(rows))

    def identity(self, task, spec, build, *, split=Split.TRAIN, purpose=DataPurpose.TRAIN):
        if (spec.record_hash != build.scenario_hash or task.world_hash != build.world_hash
                or build.seed != self.world_seed or spec.scenario_id != self.draft.scenario_id):
            raise ValueError('family identity must bind its compiled spec, world and task')
        metadata = {'family_id': self.family_id, 'knobs': self.knobs, 'spec_hash': spec.record_hash,
                    'challenge_hashes': [row.record_hash for row in self.challenges],
                    'world_seed': self.world_seed}
        # Family is shared by siblings with the same arm; lineage identifies an
        # individual generated instance under instance-disjoint split policy.
        family = f'{self.family_id}:{_digest(self.knobs)}'
        return Identity(build.world_hash, task.record_hash, family,
                        json.dumps(metadata, sort_keys=True, separators=(',', ':')),
                        split, purpose)


class Family:
    def __init__(self, family_id):
        if family_id not in FAMILY_IDS:
            raise ValueError('unknown synthetic family')
        self.family_id = family_id

    def build(self, knobs, seed):
        instance = self.instance(knobs, seed)
        return instance.draft, instance.challenges

    def instance(self, knobs, seed, *, n_tasks=1):
        from usami_v2.scenarios.designer import ScenarioDraft
        from usami_v2.scenarios.grounding import AbstractChallenge
        if (not isinstance(knobs, dict) or type(seed) is not int or seed < 0
                or type(n_tasks) is not int or not 1 <= n_tasks <= 32):
            raise ValueError('family knobs and a non-negative seed are required')
        knobs = json.loads(json.dumps(knobs, sort_keys=True, allow_nan=False))
        data, challenges = _construct(self.family_id, knobs, seed)
        if n_tasks > 1:
            data, challenges = _expand_instances(data, challenges[0], n_tasks)
        draft = ScenarioDraft.model_validate(data).checked()
        return FamilyInstance(self.family_id, knobs, seed, draft,
                              tuple(AbstractChallenge.model_validate(row).checked() for row in challenges))

    def sibling(self, instance, new_seed):
        if not isinstance(instance, FamilyInstance) or instance.family_id != self.family_id:
            raise ValueError('instance belongs to another family')
        if new_seed == instance.world_seed:
            raise ValueError('a sibling must use a different World seed')
        return self.instance(instance.knobs, new_seed, n_tasks=len(instance.challenges))


FAMILY_IDS = ('F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11')
FAMILIES = {name: Family(name) for name in FAMILY_IDS}


def _expand_instances(world, challenge, n_tasks):
    """Create separate named products/orders for each same-capability task."""
    from copy import deepcopy
    result = deepcopy(world)
    result['skus'], result['orders'], result['stock'] = [], [], []
    challenges = []
    for index in range(n_tasks):
        suffix = f'_instance_{index}'
        sku_aliases = {row['alias']: row['alias'] + suffix for row in world['skus']}
        for source in world['skus']:
            sku = deepcopy(source)
            sku['alias'] = sku_aliases[source['alias']]
            sku.setdefault('attributes', []).append({'name': 'instance', 'value': f'variant_{index}'})
            result['skus'].append(sku)
        for source in world['orders']:
            order = deepcopy(source)
            order['alias'] += suffix
            for line in order['lines']:
                line['alias'] += suffix
                line['original_sku'] = sku_aliases[line['original_sku']]
            result['orders'].append(order)
        for source in world['stock']:
            stock = deepcopy(source)
            stock['sku_alias'] = sku_aliases[source['sku_alias']]
            result['stock'].append(stock)
        new_challenge = deepcopy(challenge)
        new_challenge['task_id'] += suffix
        for selector in new_challenge['line_selectors']:
            selector.setdefault('attributes', []).append({'name': 'instance', 'value': f'variant_{index}'})
        for goal in new_challenge['remedy_goals']:
            if 'exchange' in goal['allowed_remedies']:
                goal.setdefault('required_attributes', []).append({'name': 'instance', 'value': f'variant_{index}'})
        challenges.append(new_challenge)
    if len(result['skus']) > 64 or len(result['orders']) > 64:
        raise ValueError('requested number of task instances exceeds native World bounds')
    return result, tuple(challenges)


def _construct(family, knobs, seed):
    defaults = {
        'F1': {'distractor_orders': 0},
        'F2': {'purchased_quantity': 3, 'refund_quantity': 1},
        'F3': {'amount_ratio': 0.8, 'refusal': False},
        'F4': {'age_days': 20, 'refusal': False},
        'F5': {'fulfillment_status': 'unshipped', 'refusal': False},
        'F6': {'extra_charge_cents': 1000, 'customer_limit_cents': 2000, 'refusal': False},
        'F7': {'fast_stock': 0, 'deadline_days': 7},
        'F8': {'stock': 3, 'evidence_valid': True, 'refusal': False},
        'F9': {'receipt_delay_days': 1},
        'F10': {'distractor_lines': 2},
        'F11': {'target_count': 2, 'order_count': 2},
    }
    unknown = set(knobs) - set(defaults[family])
    if unknown:
        raise ValueError('unsupported knobs for {}: {}'.format(family, ', '.join(sorted(unknown))))
    values = {**defaults[family], **knobs}
    def integer(name, lower, upper):
        value = values[name]
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f'{name} must be an integer in [{lower}, {upper}]')
        return value
    refusal = values.get('refusal', False)
    if type(refusal) is not bool:
        raise ValueError('refusal must be a boolean')
    count = integer('target_count', 1, 8) if family == 'F11' else 1
    orders_count = integer('order_count', 1, count) if family == 'F11' else 1
    quantity = integer('purchased_quantity', 1, 20) if family == 'F2' else 1
    requested = integer('refund_quantity', 1, quantity) if family == 'F2' else 1
    amount = 8000
    limit = 10000
    age = integer('age_days', 0, 90) if family == 'F4' else 1
    status = values['fulfillment_status'] if family == 'F5' else 'delivered'
    if status not in ('unshipped', 'shipped', 'delivered'):
        raise ValueError('invalid fulfillment status')
    if family == 'F3':
        ratio = values['amount_ratio']
        if type(ratio) not in (int, float) or not 0 < ratio <= 10:
            raise ValueError('amount_ratio must be in (0, 10]')
        amount = round(limit * ratio)
        if amount < 1 or (amount > limit) != refusal:
            raise ValueError('F3 refusal must agree with whether the amount exceeds the limit')
    if family == 'F4' and (age > 30) != refusal:
        raise ValueError('F4 refusal must agree with the return-window boundary')
    if family == 'F5' and (status != 'unshipped') != refusal:
        raise ValueError('F5 refusal must agree with the order status')
    extra = integer('extra_charge_cents', 0, 100000) if family == 'F6' else 0
    cap = integer('customer_limit_cents', 0, 100000) if family == 'F6' else 0
    if family == 'F6' and (extra > cap) != refusal:
        raise ValueError('F6 refusal must agree with the customer charge limit')
    stock = integer('stock', 0, 100) if family == 'F8' else 4
    evidence_valid = values.get('evidence_valid', True)
    if type(evidence_valid) is not bool:
        raise ValueError('evidence_valid must be a boolean')
    if family == 'F8' and (stock == 0 or not evidence_valid) != refusal:
        raise ValueError('F8 refusal must agree with missing stock or evidence')
    if family == 'F9':
        integer('receipt_delay_days', 0, 28)  # supplied to the Host's receipt schedule
    sku_rows, order_rows, selectors, goals = [], [], [], []
    for index in range(count):
        target = f'target_{index}'
        sku_rows.append({'alias': target, 'family': f'product_{index}',
            'attributes': [{'name': 'model', 'value': target}], 'price_cents': _fixed(amount),
            'return_required': family == 'F9'})
        line = {'alias': f'line_{index}', 'original_sku': target, 'quantity': _fixed(quantity),
                'paid_cents': _fixed(amount), 'evidence_valid': evidence_valid}
        order_index = index % orders_count
        while len(order_rows) <= order_index:
            order_rows.append({'alias': f'order_{len(order_rows)}', 'customer_alias': 'customer',
                'delivered_days_ago': _fixed(age), 'fulfillment_status': status, 'lines': []})
        order_rows[order_index]['lines'].append(line)
        selectors.append({'alias': f'item_{index}', 'family': f'product_{index}'})
        goals.append({'goal_id': f'goal_{index}', 'line_alias': f'item_{index}',
                      'quantity': requested, 'allowed_remedies': ['refund']})
    stock_rows = []
    if family in ('F6', 'F7'):
        sku_rows.append({'alias': 'replacement', 'family': 'product_0',
            'attributes': [{'name': 'model', 'value': 'replacement'}],
            'price_cents': _fixed(amount + extra)})
        fast = integer('fast_stock', 0, 20) if family == 'F7' else 4
        stock_rows.extend([{'sku_alias': 'replacement', 'warehouse_alias': 'fast', 'quantity': _fixed(fast)},
                           {'sku_alias': 'replacement', 'warehouse_alias': 'slow', 'quantity': _fixed(4)}])
        goals[0]['allowed_remedies'] = ['exchange']
        goals[0]['required_attributes'] = [{'name': 'model', 'value': 'replacement'}]
        if family == 'F6': goals[0]['maximum_extra_charge_cents'] = cap
        if family == 'F7':
            deadline = integer('deadline_days', 1, 30)
            if deadline < (2 if fast else 5):
                raise ValueError('F7 deadline cannot be met by an in-stock warehouse')
            goals[0]['latest_arrival_within_days'] = deadline
    if family == 'F8':
        stock_rows.extend({'sku_alias': 'target_0', 'warehouse_alias': warehouse,
                           'quantity': _fixed(stock)} for warehouse in ('fast', 'slow'))
        goals[0]['allowed_remedies'] = ['reship']
    if family == 'F1':
        extra_orders = integer('distractor_orders', 0, 8)
    else:
        extra_orders = 0
    extra_lines = integer('distractor_lines', 0, 8) if family == 'F10' else 0
    for index in range(extra_orders + extra_lines):
        sku = f'distractor_{index}'
        sku_rows.append({'alias': sku, 'family': sku, 'price_cents': _fixed(1000 + index)})
        line = {'alias': f'distractor_line_{index}', 'original_sku': sku,
                'quantity': _fixed(1), 'paid_cents': _fixed(1000 + index)}
        if index < extra_orders:
            order_rows.append({'alias': f'distractor_order_{index}', 'customer_alias': 'customer',
                               'delivered_days_ago': _fixed(1), 'lines': [line]})
        else:
            order_rows[0]['lines'].append(line)
    cancellation_goals = []
    if family == 'F5' and not refusal:
        cancellation_goals, goals = [{'goal_id': 'goal_0', 'line_alias': 'item_0'}], []
    refusal_goals = []
    if refusal:
        codes = {'F3': 'APPROVAL_REQUIRED', 'F4': 'RETURN_WINDOW_EXPIRED',
                 'F5': 'ORDER_NOT_CANCELLABLE', 'F6': 'CUSTOMER_DECLINED',
                 'F8': 'RESHIP_NOT_ELIGIBLE' if not evidence_valid else 'INSUFFICIENT_STOCK'}
        actions = {'F3': 'refund', 'F4': 'refund', 'F5': 'cancel', 'F6': 'exchange', 'F8': 'reship'}
        refusal_goals = [{'goal_id': 'goal_0', 'line_alias': 'item_0',
                          'reason_code': codes[family], 'requested_action': actions[family]}]
        if family == 'F6':
            # This is evidence that the Customer declined the offer, not a proof
            # that a commerce rule forbids the exchange.
            refusal_goals[0]['maximum_extra_charge_cents'] = cap
        goals = []
    data = {'scenario_id': f'{family}-world-{seed}', 'customer_aliases': ['customer'],
        'skus': sku_rows, 'warehouses': [
            {'alias': 'fast', 'delivery_days': _fixed(2), 'delivery_fee_cents': _fixed(0)},
            {'alias': 'slow', 'delivery_days': _fixed(5), 'delivery_fee_cents': _fixed(0)}],
        'orders': order_rows, 'stock': stock_rows, 'default_stock': _fixed(4),
        'refund_approval_limit_cents': limit}
    challenge = {'task_id': f'{family}-challenge-{seed}', 'customer_alias': 'customer',
        'hypothesis': f'controlled synthetic capability {family}', 'line_selectors': selectors,
        'remedy_goals': goals, 'cancellation_goals': cancellation_goals, 'refusal_goals': refusal_goals}
    # A sibling changes the publicly visible product instance while preserving
    # all knob-controlled policy boundaries and skills.
    for sku in sku_rows:
        for attribute in sku.get('attributes', ()):
            if attribute['name'] == 'model':
                attribute['value'] += f'_variant_{seed}'
    for goal in goals:
        for attribute in goal.get('required_attributes', ()):
            if attribute['name'] == 'model':
                attribute['value'] += f'_variant_{seed}'
    return data, (challenge,)
