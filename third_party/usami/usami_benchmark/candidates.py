"""Finite native Benchmark candidates from already frozen worlds.

This deterministic evaluator-side enumerator is not a Designer or a Service
policy. Its task choices never enter generation and it cannot edit world bytes.
The candidate list is a bounded lower bound on task space, not free discovery.
"""
import json
from usami_harness.models import digest
from usami_harness.goals import JointGoal
from .tasks import TaskSelection
from .tool_tasks import load_world, binding, visible_ids
from .splits import verified_world


def selections(entry):
    entry = verified_world(entry)
    domain = entry.spec.domain
    if domain in {'aftersales', 'airline', 'property', 'education'}:
        families = {
            'aftersales': [('dispute_resolution', {}), ('settlement_with_duplicate_refusal', {'mixed': True})],
            'airline': [(f, {}) for f in ('rebook', 'refund', 'baggage', 'refuse_rebook', 'refuse_refund', 'refuse_baggage')]
                       + [('refund', {'mixed': True}), ('baggage', {'mixed': True})],
            'property': [(f, {}) for f in ('repair', 'schedule', 'complaint', 'refuse_open', 'refuse_visit', 'refuse_close')]
                        + [('repair', {'mixed': True}), ('repair', {'inspection': True})],
            'education': [(f, {}) for f in ('refund', 'transfer', 'extend', 'tutoring', 'attend_lesson',
                                           'refuse_refund', 'refuse_transfer', 'refuse_extension', 'refuse_lesson')]
                         + [('refund', {'mixed': True})],
        }[domain]
        for pid in sorted(entry.world['profiles']):
            for family, parameters in families:
                yield TaskSelection(kind='dialogue', family=family, profile_id=pid, parameters=parameters)
        return
    world = load_world(entry); native, _ = binding(domain, world)
    for identity in sorted(world.users):
        permitted = visible_ids(domain, world, identity)

        def op(name, scope, resource, **params):
            field = 'org_id' if domain == 'saas' else 'project_id' if domain == 'cloudops' else 'scope_id'
            return native.Operation.model_validate({'name': name, field: scope, 'resource_id': resource, 'params': params})

        operations = []
        if domain == 'saas':
            for organization in world.organizations.values():
                sid = organization.subscription_id
                operations += [op('cancel_renewal', organization.id, sid), op('expire_subscription', organization.id, sid)]
                operations += [op('change_seats', organization.id, sid, seats=world.subscriptions[sid].seats + 1)]
            for member in world.memberships.values():
                operations += [op(name, member.org_id, member.id) for name in ('activate_member', 'deactivate_member', 'remove_member')]
        elif domain == 'cloudops':
            for instance in world.instances.values():
                operations += [op(name, instance.project_id, instance.id) for name in ('start_instance', 'stop_instance', 'delete_instance')]
                operations += [op('detach_dependency', instance.project_id, instance.id, dependency_id=dep) for dep in instance.dependencies]
            for project in world.projects.values():
                operations += [op('resume_project', project.id, project.id, invoice_id=project.invoice_id),
                               op('pay_invoice', project.id, project.invoice_id, payment_method='card_approved', idempotency_key='benchmark-payment')]
                instances = [i for i in world.instances.values() if i.project_id == project.id and i.id in permitted]
                if project.id in permitted and instances:
                    writes = [op('start_instance', project.id, i.id).model_dump(mode='json') for i in instances]
                    goal = JointGoal(atoms=[{'collection': 'instances', 'entity': i.id, 'field': 'status', 'value': 'running'} for i in instances],
                                     allowed_changes={'instances/' + i.id: ['status'] for i in instances})
                    yield TaskSelection(kind='tool', family='existing_stack_start', identity=identity, goal=goal,
                        capabilities=writes, message='Bring all these existing instances online, preserving unrelated state. Authorized optional choices: ' + json.dumps(writes, sort_keys=True))
        elif domain == 'fulfillment':
            for warehouse in world.warehouses.values():
                operations.append(op('set_warehouse_receiving', warehouse.id, warehouse.id, enabled=not warehouse.receiving_enabled))
            for order in world.orders.values():
                operations += [op(name, order.warehouse_id, order.id) for name in ('confirm_order', 'cancel_order')]
            for lot in world.lots.values():
                operations.append(op('inspect_lot', lot.warehouse_id, lot.id))
            for reservation in world.reservations.values():
                operations += [op(name, reservation.warehouse_id, reservation.id) for name in ('pick_reservation', 'pack_reservation', 'release_reservation')]
            for shipment in world.shipments.values():
                operations += [op(name, shipment.warehouse_id, shipment.id) for name in ('dispatch_shipment', 'confirm_delivery', 'cancel_shipment')]
        else:
            for room in world.rooms.values():
                operations += [op('mark_room_clean', room.property_id, room.id),
                               op('set_room_maintenance', room.property_id, room.id, enabled=not room.maintenance)]
            for reservation in world.reservations.values():
                operations += [op(name, reservation.property_id, reservation.id) for name in ('confirm_reservation', 'cancel_reservation', 'release_hold', 'expire_hold', 'check_in', 'check_out')]
        for operation in sorted(operations, key=digest):
            raw = operation.model_dump(mode='json')
            # Object visibility is independently enforced again by the compiler.
            if operation.resource_id not in permitted:
                continue
            decision = native.decide(world, identity, operation)
            yield TaskSelection(kind='tool', family=operation.name, identity=identity, capabilities=[raw],
                refusal_operations=[] if decision.allowed else [raw],
                message='Please perform this requested business action, preserving unrelated state: ' + json.dumps(raw, sort_keys=True))


def scheduled_selections(entry, limit):
    if type(limit) is not int or limit <= 0:
        raise ValueError('positive finite candidate limit required')
    rows = {digest(row): row for row in selections(entry)}
    return [rows[key] for key in sorted(rows)[:limit]]
