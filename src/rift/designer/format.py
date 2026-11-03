"""The compact plan a Designer writes, and its exact translation into Usami drafts.

The Designer states one World with fixed values and the Customers' requests in it.
It never states whether a request should succeed: Usami derives that answer and
certifies it. Everything a model can get wrong is reported as a readable reason.
"""
from hashlib import sha256
import json
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

ACTIONS = ('refund', 'exchange', 'reship', 'cancel')
REFERENCES = ('attributes', 'delivery_age', 'price', 'order_date')


class _Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


class PlanSku(_Strict):
    alias: str = Field(min_length=1)
    family: str = Field(min_length=1, description='Replacement-compatibility group; exchanges stay inside one family.')
    attributes: Dict[str, str] = Field(default_factory=dict)
    price_cents: int = Field(ge=1)
    return_days: int = Field(default=30, ge=0)
    return_required: bool = Field(default=False, description='Refund only after the item is physically returned.')


class PlanWarehouse(_Strict):
    alias: str = Field(min_length=1)
    delivery_days: int = Field(ge=0)
    delivery_fee_cents: int = Field(default=0, ge=0)


class PlanStock(_Strict):
    sku: str
    warehouse: str
    quantity: int = Field(ge=0)


class PlanLine(_Strict):
    alias: str = Field(min_length=1)
    sku: str
    quantity: int = Field(ge=1, le=20)
    paid_cents: int = Field(ge=0)
    issue: Literal['damaged', 'wrong_item', 'missing_parts'] = 'damaged'
    evidence_valid: bool = True


class PlanOrder(_Strict):
    alias: str = Field(min_length=1)
    customer: str = 'customer'
    status: Literal['unshipped', 'shipped', 'delivered'] = 'delivered'
    delivered_days_ago: int = Field(ge=0, description='For a delivered order: days since delivery; '
                                                      'otherwise days until expected delivery.')
    ordered_days_ago: Optional[int] = Field(default=None, ge=0)
    lines: List[PlanLine] = Field(min_length=1, max_length=8)


class PlanHistory(_Strict):
    operation: Literal['return_delivered_order_items', 'cancel_order', 'exchange_delivered_order_items',
                       'reship_order_items', 'request_item_return', 'return_received']
    line: Optional[str] = None
    order: Optional[str] = None
    quantity: int = Field(default=1, ge=1)
    target_sku: Optional[str] = None


class PlanWorld(_Strict):
    customers: List[str] = Field(default_factory=lambda: ['customer'], min_length=1, max_length=4)
    refund_approval_limit_cents: int = Field(ge=0, description='A refund above this needs approval and is refused.')
    skus: List[PlanSku] = Field(min_length=1, max_length=16)
    warehouses: List[PlanWarehouse] = Field(min_length=1, max_length=4)
    stock: List[PlanStock] = Field(default_factory=list)
    default_stock: int = Field(default=0, ge=0)
    orders: List[PlanOrder] = Field(min_length=1, max_length=8)
    history: List[PlanHistory] = Field(default_factory=list, description='Earlier actions, executed before the task.')


class PlanRequest(_Strict):
    line: str
    action: Literal['refund', 'exchange', 'reship', 'cancel']
    quantity: int = Field(default=1, ge=1)
    attributes: Dict[str, str] = Field(default_factory=dict,
                                       description='Exchange/reship only: attributes the new item must have.')
    max_extra_charge_cents: Optional[int] = Field(default=None, ge=0)
    arrive_within_days: Optional[int] = Field(default=None, ge=0)
    refer_by: Literal['attributes', 'delivery_age', 'price', 'order_date'] = 'attributes'


class PlanTask(_Strict):
    customer: str = 'customer'
    requests: List[PlanRequest] = Field(min_length=1, max_length=4)
    in_order: bool = Field(default=False, description='Handle the requests in the listed order.')


class DesignerPlan(_Strict):
    hypothesis: str = Field(min_length=1, description='The weakness these tasks train.')
    world: PlanWorld
    tasks: List[PlanTask] = Field(min_length=1, max_length=4)


class PlanError(ValueError):
    """A Designer output that cannot become a native draft; ``code`` groups reasons."""

    def __init__(self, code, message):
        super().__init__(f'{code}: {message}')
        self.code = code


def plan_schema():
    """JSON Schema for constrained decoding of one DesignerPlan."""
    return DesignerPlan.model_json_schema()


def extract_json(text):
    """The outermost JSON object in a model response (tolerates fences and stray prose)."""
    text = text.split('</think>')[-1]
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        raise PlanError('NO_JSON', 'the response contains no JSON object')
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise PlanError('BAD_JSON', str(exc)) from exc


def parse_plan(value):
    if isinstance(value, str):
        value = extract_json(value)
    try:
        return DesignerPlan.model_validate(value)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = '.'.join(str(part) for part in first['loc'])
        raise PlanError('SCHEMA', f'{where}: {first["msg"]} ({len(exc.errors())} error(s))') from exc


def _fixed(value):
    return {'low': value, 'high': value}


def plan_drafts(plan, plan_id):
    """``(ScenarioDraft data, [TaskDraft data])`` for a parsed plan."""
    world = plan.world
    lines = {line.alias: line for order in world.orders for line in order.lines}
    data = {
        'scenario_id': plan_id, 'customer_aliases': list(world.customers),
        'refund_approval_limit_cents': world.refund_approval_limit_cents,
        'default_stock': _fixed(world.default_stock),
        'skus': [{'alias': row.alias, 'family': row.family,
                  'attributes': [{'name': name, 'value': value} for name, value in row.attributes.items()],
                  'price_cents': _fixed(row.price_cents), 'return_days': row.return_days,
                  'return_required': row.return_required} for row in world.skus],
        'warehouses': [{'alias': row.alias, 'delivery_days': _fixed(row.delivery_days),
                        'delivery_fee_cents': _fixed(row.delivery_fee_cents)} for row in world.warehouses],
        'stock': [{'sku_alias': row.sku, 'warehouse_alias': row.warehouse, 'quantity': _fixed(row.quantity)}
                  for row in world.stock],
        'orders': [{'alias': row.alias, 'customer_alias': row.customer, 'fulfillment_status': row.status,
                    'delivered_days_ago': _fixed(row.delivered_days_ago),
                    **({} if row.ordered_days_ago is None else {'ordered_days_ago': row.ordered_days_ago}),
                    'lines': [{'alias': line.alias, 'original_sku': line.sku, 'quantity': _fixed(line.quantity),
                               'paid_cents': _fixed(line.paid_cents), 'issue': line.issue,
                               'evidence_valid': line.evidence_valid} for line in row.lines]}
                   for row in world.orders],
    }
    if world.history:
        data['history'] = [{key: value for key, value in {
            'operation': row.operation, 'line_alias': row.line, 'order_alias': row.order,
            'quantity': row.quantity, 'target_sku_alias': row.target_sku}.items() if value is not None}
            for row in world.history]
    style = int(sha256(plan_id.encode()).hexdigest()[:8], 16)
    tasks = []
    for index, task in enumerate(plan.tasks):
        requests = []
        for number, request in enumerate(task.requests):
            line = lines.get(request.line)
            if line is None:
                raise PlanError('UNKNOWN_LINE', f'task {index} request {number} names unknown line {request.line!r}')
            if request.action in ('refund', 'cancel') and (request.attributes or request.arrive_within_days is not None
                                                           or request.max_extra_charge_cents is not None):
                raise PlanError('REQUEST_INVALID', f'task {index} request {number}: a {request.action} request '
                                'cannot carry attributes, a charge limit or a deadline')
            requests.append({
                'goal_id': f'g{number}', 'line_alias': request.line, 'action': request.action,
                # A cancellation always covers the whole line; the Designer need not restate it.
                'quantity': line.quantity if request.action == 'cancel' else request.quantity,
                'required_attributes': [{'name': name, 'value': value} for name, value in request.attributes.items()],
                'maximum_extra_charge_cents': request.max_extra_charge_cents,
                'latest_arrival_within_days': request.arrive_within_days, 'reference': request.refer_by})
        tasks.append({'task_id': f'{plan_id}-t{index}', 'customer_alias': task.customer,
                      'requests': requests, 'style_seed': style + index,
                      'sequential': task.in_order and len(requests) > 1})
    return data, tasks


def plan_proposal(value, plan_id):
    """Parse, translate and natively validate a Designer output into a training-plan Proposal."""
    from rift.benchmark.families import synthetic_distribution
    from rift.tasks import Proposal
    from usami_v2.scenarios.designer import ScenarioDraft
    plan = parse_plan(value)
    data, tasks = plan_drafts(plan, plan_id)
    try:
        # Compiling checks cross-references (unique aliases, known SKUs) before any build.
        ScenarioDraft.model_validate(data).compile(synthetic_distribution())
        return Proposal(plan_id, plan.hypothesis, world_draft=data, challenges=tuple(tasks))
    except ValidationError as exc:
        first = exc.errors()[0]
        raise PlanError('WORLD_INVALID', f'{first["msg"]} ({".".join(str(p) for p in first["loc"])})') from exc
    except ValueError as exc:
        raise PlanError('WORLD_INVALID', str(exc)) from exc


def plan_from_drafts(data, task, hypothesis):
    """Inverse of ``plan_drafts`` for a generator instance: a worked example in plan form."""
    low = lambda value: value['low'] if isinstance(value, dict) else value
    world = {
        'customers': list(data['customer_aliases']),
        'refund_approval_limit_cents': data.get('refund_approval_limit_cents', 50000),
        'default_stock': low(data.get('default_stock', {'low': 0})),
        'skus': [{'alias': row['alias'], 'family': row['family'],
                  'attributes': {item['name']: item['value'] for item in row.get('attributes', ())},
                  'price_cents': low(row['price_cents']), 'return_days': row.get('return_days', 30),
                  'return_required': row.get('return_required', False)} for row in data['skus']],
        'warehouses': [{'alias': row['alias'], 'delivery_days': low(row['delivery_days']),
                        'delivery_fee_cents': low(row['delivery_fee_cents'])} for row in data['warehouses']],
        'stock': [{'sku': row['sku_alias'], 'warehouse': row['warehouse_alias'], 'quantity': low(row['quantity'])}
                  for row in data.get('stock', ())],
        'orders': [{'alias': row['alias'], 'customer': row['customer_alias'],
                    'status': row.get('fulfillment_status', 'delivered'),
                    'delivered_days_ago': low(row['delivered_days_ago']),
                    **({'ordered_days_ago': row['ordered_days_ago']} if row.get('ordered_days_ago') is not None else {}),
                    'lines': [{'alias': line['alias'], 'sku': line['original_sku'], 'quantity': low(line['quantity']),
                               'paid_cents': low(line['paid_cents']), 'issue': line.get('issue', 'damaged'),
                               'evidence_valid': line.get('evidence_valid', True)} for line in row['lines']]}
                   for row in data['orders']],
        'history': [{key: value for key, value in {
            'operation': row['operation'], 'line': row.get('line_alias'), 'order': row.get('order_alias'),
            'quantity': row.get('quantity', 1), 'target_sku': row.get('target_sku_alias')}.items()
            if value is not None} for row in data.get('history', ())],
    }
    requests = [{'line': row['line_alias'], 'action': row['action'], 'quantity': row['quantity'],
                 'attributes': {item['name']: item['value'] for item in row.get('required_attributes', ())},
                 'max_extra_charge_cents': row.get('maximum_extra_charge_cents'),
                 'arrive_within_days': row.get('latest_arrival_within_days'),
                 'refer_by': row.get('reference', 'attributes')} for row in task['requests']]
    return DesignerPlan.model_validate({'hypothesis': hypothesis, 'world': world, 'tasks': [
        {'customer': task['customer_alias'], 'requests': requests, 'in_order': task.get('sequential', False)}]})
