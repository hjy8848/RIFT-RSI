"""Independent policy and expected transitions. Never imports the Engine/applier."""
from .models import Access, Decision, Instance, Payment, QUERIES, Quota

POLICY = {
 'PROJECT_ACCESS_REQUIRED': 'Compute operations require an explicit project Admin/Operator role, including for organization owners. Owners can administer projects and grant scoped access to themselves.',
 'PERMISSION_DENIED': 'Owner/Admin manages quota, billing and access. Operator starts/stops; Viewer only reads nonfinancial project data. Admin cannot change an Admin grant.',
 'CROSS_PROJECT_RESOURCE': 'Every resource/dependency/funding invoice belongs to the explicitly requested project.',
 'QUOTA_EXCEEDED': 'Running CPU, memory and instance count must all fit quota.',
 'QUOTA_BELOW_USAGE': 'Quota reductions cannot drop below current running usage.',
 'DEPENDENCY_NOT_RUNNING': 'All direct dependencies must be running before a resource starts.',
 'DEPENDENTS_RUNNING': 'Stop running dependents before stopping their dependency.',
 'DEPENDENCY_STILL_ATTACHED': 'Detach all incoming references before deleting a stopped instance.',
 'DEPENDENT_MUST_BE_STOPPED': 'Change dependency edges only while the dependent is stopped.',
 'DEPENDENCY_CYCLE': 'Dependency edges must form a project-local directed acyclic graph.',
 'PROJECT_SUSPENDED': 'Suspended projects cannot start resources.',
 'INVOICE_NOT_PAID': 'Resume only after the project current invoice is successfully paid. Payment does not automatically resume.',
 'IDEMPOTENCY_CONFLICT': 'Payment key is scoped to its invoice and immutable parameters/outcome. Same-key retries return the original result.',
 'ALREADY_PAID': 'A paid invoice cannot be charged again using a different payment key.',
 'OWNER_PROTECTED': 'Organization owners are fixed identities; access tools cannot mutate organization ownership.',
}


def role(world, actor, project):
    return next((a.role for a in world.access.values() if a.project_id == project and a.user_id == actor), None)


def owner(world, actor, project):
    return world.organizations[world.projects[project].org_id].owner_id == actor


def usage(world, project):
    running = [i for i in world.instances.values() if i.project_id == project and i.status == 'running']
    return {'cpu': sum(i.cpu for i in running), 'memory': sum(i.memory for i in running), 'instances': len(running)}


def cycles(instances):
    visited, active = set(), set()
    def visit(key):
        if key in active: return True
        if key in visited or key not in instances: return False
        active.add(key)
        if any(visit(d) for d in instances[key].dependencies): return True
        active.remove(key); visited.add(key)
        return False
    return any(visit(key) for key in instances)


def invariants(world):
    errors = []
    def add(code, path): errors.append({'code': code, 'path': path})
    for collection in ('users','organizations','projects','access','instances','invoices'):
        for key, obj in getattr(world, collection).items():
            if key != obj.id: add('ENTITY_KEY_MISMATCH', collection + '/' + key)
    for org in world.organizations.values():
        if org.owner_id not in world.users: add('OWNER_MISSING', 'organizations/' + org.id)
    seen = set()
    for grant in world.access.values():
        if grant.project_id not in world.projects or grant.user_id not in world.users: add('ACCESS_REFERENCE_INVALID', 'access/' + grant.id)
        if (grant.project_id, grant.user_id) in seen: add('DUPLICATE_ACCESS', 'access/' + grant.id)
        seen.add((grant.project_id, grant.user_id))
    for project in world.projects.values():
        invoice = world.invoices.get(project.invoice_id)
        if project.org_id not in world.organizations or not invoice or invoice.project_id != project.id: add('PROJECT_BILLING_REFERENCE', 'projects/' + project.id)
        elif project.status == 'active' and invoice.status != 'paid': add('ACTIVE_PROJECT_UNPAID', 'projects/' + project.id)
        if any(usage(world, project.id)[k] > getattr(project.quota, k) for k in ('cpu','memory','instances')): add('QUOTA_EXCEEDED', 'projects/' + project.id + '/quota')
    for instance in world.instances.values():
        if instance.project_id not in world.projects: add('INSTANCE_PROJECT_MISSING', 'instances/' + instance.id)
        if len(instance.dependencies) != len(set(instance.dependencies)): add('DUPLICATE_DEPENDENCY', 'instances/' + instance.id)
        for dep in instance.dependencies:
            target = world.instances.get(dep)
            if not target: add('DANGLING_DEPENDENCY', 'instances/' + instance.id)
            elif target.project_id != instance.project_id: add('CROSS_PROJECT_DEPENDENCY', 'instances/' + instance.id)
            elif instance.status == 'running' and target.status != 'running': add('RUNNING_DEPENDENCY_STOPPED', 'instances/' + instance.id)
        if instance.status == 'running' and instance.project_id in world.projects and world.projects[instance.project_id].status != 'active': add('SUSPENDED_RESOURCE_RUNNING', 'instances/' + instance.id)
    if cycles(world.instances): add('DEPENDENCY_CYCLE', 'instances')
    payment_ids = set()
    for invoice in world.invoices.values():
        if invoice.project_id not in world.projects: add('INVOICE_PROJECT_MISSING', 'invoices/' + invoice.id)
        keys = set(); successes = 0
        for payment in invoice.payments:
            if payment.key in keys or payment.id in payment_ids: add('DUPLICATE_CHARGE_KEY', 'invoices/' + invoice.id)
            keys.add(payment.key); payment_ids.add(payment.id)
            successes += payment.outcome == 'succeeded'
            if payment.amount_cents != invoice.amount_cents or payment.at != world.clock: add('PAYMENT_ACCOUNTING_INVALID', 'invoices/' + invoice.id)
            if (payment.method == 'card_approved') != (payment.outcome == 'succeeded'): add('PAYMENT_OUTCOME_INVALID', 'invoices/' + invoice.id)
        if successes > 1: add('DUPLICATE_CHARGE', 'invoices/' + invoice.id)
        expected = 'paid' if successes else 'failed' if invoice.payments else 'unpaid'
        if invoice.status != expected: add('PAYMENT_STATUS_INVALID', 'invoices/' + invoice.id)
    return errors


def decide(world, actor, op):
    deny = lambda code: Decision(allowed=False, error_code=code)
    if invariants(world): return deny('INVALID_WORLD')
    if actor not in world.users: return deny('UNKNOWN_ACTOR')
    project = world.projects.get(op.project_id)
    if not project: return deny('PROJECT_NOT_FOUND')
    privileged = owner(world, actor, project.id)
    membership = role(world, actor, project.id)
    if not privileged and not membership: return deny('CROSS_PROJECT_ACCESS')
    name, rid, params = op.name, op.resource_id, op.params
    kind = 'instance' if name in {'start_instance','stop_instance','delete_instance','attach_dependency','detach_dependency','get_instance'} else 'invoice' if name in {'pay_invoice','get_invoice'} else 'project'
    obj = (world.instances if kind == 'instance' else world.invoices if kind == 'invoice' else world.projects).get(rid)
    if not obj: return deny('RESOURCE_NOT_FOUND')
    if (obj.id if kind == 'project' else obj.project_id) != project.id: return deny('CROSS_PROJECT_RESOURCE')
    admin = privileged or membership == 'admin'
    if name == 'get_invoice' and not admin: return deny('PERMISSION_DENIED')
    if not op.is_query:
        if name in {'start_instance','stop_instance','create_instance','delete_instance','attach_dependency','detach_dependency'}:
            if membership not in {'admin','operator'}: return deny('PROJECT_ACCESS_REQUIRED')
            if name not in {'start_instance','stop_instance'} and membership != 'admin': return deny('PERMISSION_DENIED')
        elif not admin: return deny('PERMISSION_DENIED')
    new = world.clone(); data = {}
    if name == 'create_instance':
        instance = Instance(id=params['instance_id'], name=params['name'], project_id=project.id, cpu=params['cpu'], memory=params['memory'])
        old = world.instances.get(instance.id)
        if old and old != instance: return deny('INSTANCE_ID_CONFLICT')
        new.instances[instance.id] = instance; data = {'instance_id': instance.id}
    elif name == 'start_instance':
        if obj.status == 'running': return Decision(allowed=True, expected_state=new, data={'instance_id': rid, 'status': 'running'})
        if project.status != 'active': return deny('PROJECT_SUSPENDED')
        if any(world.instances[d].status != 'running' for d in obj.dependencies): return deny('DEPENDENCY_NOT_RUNNING')
        used = usage(world, project.id)
        if used['cpu'] + obj.cpu > project.quota.cpu or used['memory'] + obj.memory > project.quota.memory or used['instances'] + 1 > project.quota.instances: return deny('QUOTA_EXCEEDED')
        new.instances[rid].status = 'running'; data = {'instance_id': rid, 'status': 'running'}
    elif name == 'stop_instance':
        if any(i.status == 'running' and rid in i.dependencies for i in world.instances.values()): return deny('DEPENDENTS_RUNNING')
        new.instances[rid].status = 'stopped'; data = {'instance_id': rid, 'status': 'stopped'}
    elif name == 'delete_instance':
        if obj.status != 'stopped': return deny('INSTANCE_RUNNING')
        if any(rid in i.dependencies for i in world.instances.values()): return deny('DEPENDENCY_STILL_ATTACHED')
        del new.instances[rid]; data = {'deleted_id': rid}
    elif name in {'attach_dependency','detach_dependency'}:
        dep = world.instances.get(params['dependency_id'])
        if not dep: return deny('DEPENDENCY_NOT_FOUND')
        if dep.project_id != project.id: return deny('CROSS_PROJECT_RESOURCE')
        if obj.status != 'stopped': return deny('DEPENDENT_MUST_BE_STOPPED')
        deps = new.instances[rid].dependencies
        if name == 'attach_dependency' and dep.id not in deps: deps.append(dep.id)
        if name == 'detach_dependency' and dep.id in deps: deps.remove(dep.id)
        if cycles(new.instances): return deny('DEPENDENCY_CYCLE')
        data = {'instance_id': rid, 'dependencies': list(deps)}
    elif name == 'change_project_quota':
        if any(params[k] < usage(world, project.id)[k] for k in params): return deny('QUOTA_BELOW_USAGE')
        new.projects[rid].quota = Quota(**params); data = params.copy()
    elif name in {'grant_project_access','revoke_project_access'}:
        uid = params['user_id']
        if uid not in world.users: return deny('UNKNOWN_USER')
        old = next((a for a in world.access.values() if a.project_id == project.id and a.user_id == uid), None)
        key = old.id if old else project.id + '_' + uid
        if not old and key in world.access: return deny('ACCESS_ID_CONFLICT')
        if not privileged and (params.get('role') == 'admin' or (old and old.role == 'admin')): return deny('PERMISSION_DENIED')
        if name == 'grant_project_access': new.access[key] = Access(id=key, project_id=project.id, user_id=uid, role=params['role'])
        else:
            if not old: return deny('ACCESS_NOT_FOUND')
            del new.access[key]
        data = {'access_id': key}
    elif name == 'pay_invoice':
        key, method = params['idempotency_key'], params['payment_method']
        prior = next((p for p in obj.payments if p.key == key), None)
        if prior:
            if prior.method != method: return deny('IDEMPOTENCY_CONFLICT')
            data = {'payment_id': prior.id, 'outcome': prior.outcome}
        else:
            if obj.status == 'paid': return deny('ALREADY_PAID')
            from .models import digest
            outcome = 'succeeded' if method == 'card_approved' else 'failed'
            payment = Payment(id='charge_' + digest([rid, key])[:16], key=key, method=method, amount_cents=obj.amount_cents, outcome=outcome, at=world.clock)
            new.invoices[rid].payments.append(payment); new.invoices[rid].status = 'paid' if outcome == 'succeeded' else 'failed'
            data = {'payment_id': payment.id, 'outcome': outcome}
    elif name == 'resume_project':
        invoice = world.invoices.get(params['invoice_id'])
        if not invoice: return deny('INVOICE_NOT_FOUND')
        if invoice.project_id != project.id or invoice.id != project.invoice_id: return deny('CROSS_PROJECT_RESOURCE')
        if invoice.status != 'paid': return deny('INVOICE_NOT_PAID')
        new.projects[rid].status = 'active'; data = {'project_id': rid, 'status': 'active'}
    elif name == 'get_project':
        data = project.model_dump(mode='json'); data['usage'] = usage(world, project.id)
    elif name == 'list_instances': data = {'instances': [i.model_dump(mode='json') for i in world.instances.values() if i.project_id == project.id]}
    elif name == 'get_project_members': data = {'access': [a.model_dump(mode='json') for a in world.access.values() if a.project_id == project.id]}
    else: data = obj.model_dump(mode='json')
    return Decision(allowed=True, expected_state=new, data=data)
