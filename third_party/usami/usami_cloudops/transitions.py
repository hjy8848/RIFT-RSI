"""Actual mutations: existence/schema only, independent of Reference decisions."""
from .models import Access, Instance, Payment, Quota, Result, digest


def apply(world, actor, op, event_id):
    new = world.clone(); name, rid, p = op.name, op.resource_id, op.params
    fail = lambda code: (world.clone(), Result(success=False, error_code=code))
    if actor not in world.users: return fail('UNKNOWN_ACTOR')
    if op.project_id not in world.projects: return fail('PROJECT_NOT_FOUND')
    kind = 'instances' if name in {'start_instance','stop_instance','delete_instance','attach_dependency','detach_dependency','get_instance'} else 'invoices' if name in {'pay_invoice','get_invoice'} else 'projects'
    obj = getattr(new, kind).get(rid)
    if not obj: return fail('RESOURCE_NOT_FOUND')
    data = {}
    if name == 'create_instance':
        instance = Instance(id=p['instance_id'], name=p['name'], project_id=op.project_id, cpu=p['cpu'], memory=p['memory'])
        if instance.id in new.instances and new.instances[instance.id] != instance: return fail('INSTANCE_ID_CONFLICT')
        new.instances[instance.id] = instance; data = {'instance_id': instance.id}
    elif name in {'start_instance','stop_instance'}:
        obj.status = 'running' if name == 'start_instance' else 'stopped'; data = {'instance_id': rid, 'status': obj.status}
    elif name == 'delete_instance': del new.instances[rid]; data = {'deleted_id': rid}
    elif name in {'attach_dependency','detach_dependency'}:
        target = p['dependency_id']
        if target not in new.instances: return fail('DEPENDENCY_NOT_FOUND')
        if name == 'attach_dependency' and target not in obj.dependencies: obj.dependencies.append(target)
        if name == 'detach_dependency' and target in obj.dependencies: obj.dependencies.remove(target)
        data = {'instance_id': rid, 'dependencies': list(obj.dependencies)}
    elif name == 'change_project_quota': obj.quota = Quota(**p); data = p.copy()
    elif name in {'grant_project_access','revoke_project_access'}:
        uid = p['user_id']
        if uid not in new.users: return fail('UNKNOWN_USER')
        existing = next((a for a in new.access.values() if a.project_id == op.project_id and a.user_id == uid), None)
        key = existing.id if existing else op.project_id + '_' + uid
        if not existing and key in new.access: return fail('ACCESS_ID_CONFLICT')
        if name == 'grant_project_access': new.access[key] = Access(id=key, project_id=op.project_id, user_id=uid, role=p['role'])
        else:
            if key not in new.access: return fail('ACCESS_NOT_FOUND')
            del new.access[key]
        data = {'access_id': key}
    elif name == 'pay_invoice':
        prior = next((a for a in obj.payments if a.key == p['idempotency_key'] and a.method == p['payment_method']), None)
        if prior: data = {'payment_id': prior.id, 'outcome': prior.outcome}
        else:
            outcome = 'succeeded' if p['payment_method'] == 'card_approved' else 'failed'
            payment = Payment(id='charge_' + digest([rid, p['idempotency_key']])[:16], key=p['idempotency_key'], method=p['payment_method'], amount_cents=obj.amount_cents, outcome=outcome, at=world.clock)
            obj.payments.append(payment); obj.status = 'paid' if outcome == 'succeeded' else 'failed'
            data = {'payment_id': payment.id, 'outcome': outcome}
    elif name == 'resume_project':
        if p['invoice_id'] not in new.invoices: return fail('INVOICE_NOT_FOUND')
        obj.status = 'active'; data = {'project_id': rid, 'status': 'active'}
    elif name == 'get_project':
        data = obj.model_dump(mode='json'); running = [i for i in world.instances.values() if i.project_id == op.project_id and i.status == 'running']
        data['usage'] = {'cpu': sum(i.cpu for i in running), 'memory': sum(i.memory for i in running), 'instances': len(running)}
    elif name == 'list_instances': data = {'instances': [i.model_dump(mode='json') for i in world.instances.values() if i.project_id == op.project_id]}
    elif name == 'get_project_members': data = {'access': [a.model_dump(mode='json') for a in world.access.values() if a.project_id == op.project_id]}
    else: data = obj.model_dump(mode='json')
    return new, Result(success=True, data=data)
