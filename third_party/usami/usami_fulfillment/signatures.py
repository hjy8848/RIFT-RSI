"""Anonymous goal/operation/causal shapes, separate from parameter instances."""
from usami_harness.models import digest
from .rules import role, decide, price, paid, physical_volume


def task_signatures(world, task):
    entities = {identifier: collection for collection in world.__class__.model_fields
                if isinstance(getattr(world, collection), dict)
                for identifier in getattr(world, collection)}
    for atom in task.goal.atoms:
        entities.setdefault(atom.entity,atom.collection)
    for operation in task.capabilities:
        name, params = operation['name'], operation['params']
        for field, collection in [('order_id','orders'), ('reservation_id','reservations'),
                                  ('shipment_id','shipments'), ('transfer_id','transfers'),
                                  ('return_id','returns'), ('lot_id','lots'), ('target_lot_id','lots')]:
            if field in params:
                entities.setdefault(params[field], collection)
    aliases, counters = {}, {}
    def symbolic(value, key=''):
        if isinstance(value, dict):
            return {k: symbolic(v, k) for k, v in sorted(value.items())}
        if isinstance(value, list):
            return [symbolic(v, key) for v in value]
        if isinstance(value, str) and value in entities:
            if value not in aliases:
                category = entities[value]
                counters[category] = counters.get(category, 0) + 1
                aliases[value] = category + ':' + str(counters[category])
            return aliases[value]
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, (int, float)):
            return '<integer>'
        if key in {'key','name','id','line_id'}:
            return '<' + key + '>'
        return value
    goals = [symbolic(atom.model_dump(mode='json')) for atom in task.goal.atoms]
    ops = [symbolic(operation) for operation in task.capabilities]
    op_shapes = {digest(raw): shape for raw, shape in zip(task.capabilities, ops)}
    edge_set = set()
    for witness in task.witnesses:
        for edge in witness.direct_edges:
            a, b = witness.operations[edge['from']], witness.operations[edge['to']]
            edge_set.add((digest(op_shapes[digest(a)]), digest(op_shapes[digest(b)]), edge['error_code']))
    used_shapes=({digest(op_shapes[digest(op)]):op_shapes[digest(op)] for witness in task.witnesses for op in witness.operations}
                 if task.kind=='completion' else {digest(op):op for op in ops})
    structure = {'kind': task.kind, 'goals': sorted(goals, key=digest) if task.kind=='completion' else ['total_frame_preservation'],
                 'operations': sorted(used_shapes.values(), key=digest), 'causal_edges': sorted(edge_set)}
    # Record business decision branches without names, IDs, seed or raw amounts.
    features = []
    for raw in task.capabilities:
        op = world_operation(raw)
        decision = decide(world, task.identity, op)
        warehouse = world.warehouses.get(op.scope_id)
        order = world.orders.get(op.resource_id)
        lot = world.lots.get(raw['params'].get('lot_id', op.resource_id))
        sku = world.skus.get(lot.sku_id) if lot else None
        reservation=world.reservations.get(raw['params'].get('reservation_id',''))
        if not sku and reservation and reservation.lot_id in world.lots:
            sku=world.skus[world.lots[reservation.lot_id].sku_id]
        features.append({'operation': op.name, 'role': role(world, task.identity, op.scope_id),
            'initial_decision': decision.error_code or 'ALLOWED',
            'scope_relation': 'same' if op.scope_id == getattr(order or lot, 'warehouse_id', op.scope_id) else 'different',
            'buyer': bool(order and order.customer_id == task.identity),
            'order_status': order.status if order else None,
            'payment': ('none' if not paid(world, order.id) else 'full' if paid(world, order.id) >= price(order) else 'partial') if order else None,
            'quality': lot.quality if lot else None,
            'category': sku.category if sku else None,
            'product_returnable': sku.returnable if sku else None,
            'simulator_inspection_branch': sku.simulated_return_condition if sku and op.name in {'request_return','inspect_return','refund_order'} else None,
            'receiving': warehouse.receiving_enabled if warehouse else None,
            'capacity': 'full' if warehouse and physical_volume(world, warehouse.id) == warehouse.capacity_volume else 'space'})
    return digest(structure), digest({'structure': structure, 'branches': sorted(features, key=digest),
                                      'refusal': task.refusal_code})


def world_operation(raw):
    from .models import Operation
    return Operation.model_validate(raw)
