"""Independent read intentions and observable information-value projections."""
from .customer_protocol_v2 import ReadProbe
from usami_cloudops.observations import observation
from usami_cloudops.rules import decide
from vast_coevo.schema import digest


def fact(probe, data):
    if probe.purpose == 'quota': return {k: data.get(k) for k in ('quota', 'usage')}
    if probe.purpose == 'permissions': return data.get('access', [])
    if probe.purpose == 'billing': return {k: data.get(k) for k in ('status', 'amount_cents', 'payments')}
    if probe.purpose == 'catalog':
        return {key: sorted(row['id'] for row in data.get(key, []))
                for key in ('projects', 'instances', 'public_directory')}
    if 'instances' in data:
        return [{key: row.get(key) for key in (('id', 'dependencies') if probe.purpose == 'dependencies' else ('id', 'status'))}
                for row in data['instances']]
    return data.get('dependencies', []) if probe.purpose == 'dependencies' else data.get('status')


def initial_fact(view, probe):
    if probe.name == 'get_customer_context': return fact(probe, view)
    project = next((row for row in view['projects'] if row['id'] == probe.project_id), None)
    if not project: return None
    if probe.name == 'get_project_members':
        return fact(probe, {'access': [row for row in view['project_access'] if row['project_id'] == probe.project_id]})
    instances = [row for row in view['instances'] if row['project_id'] == probe.project_id]
    if probe.name == 'list_instances': return fact(probe, {'instances': instances})
    if probe.name == 'get_project':
        data = dict(project)
        running = [row for row in instances if row['status'] == 'running']
        data['usage'] = {'cpu': sum(row['cpu'] for row in running), 'memory': sum(row['memory'] for row in running), 'instances': len(running)}
        return fact(probe, data)
    collection = view['invoices'] if probe.name == 'get_invoice' else instances
    row = next((row for row in collection if row['id'] == probe.resource_id), None)
    return fact(probe, row) if row else None


def assess_read(raw, world, identity, permitted, tools, initial_view):
    probe = ReadProbe.model_validate(raw)
    operation = probe.operation()
    reason = None
    if probe.name not in tools: reason = 'read_tool_not_in_frozen_interface'
    elif operation is not None:
        collection = world.instances if probe.name == 'get_instance' else world.invoices if probe.name == 'get_invoice' else world.projects
        if probe.project_id not in permitted or probe.resource_id not in permitted:
            reason = 'read_object_not_in_permitted_observations'
        elif probe.project_id not in world.projects or probe.resource_id not in collection:
            reason = 'read_object_not_resolvable_as_declared_type'
    decision = decide(world, identity, operation) if operation else None
    allowed = decision.allowed if decision else True
    code = decision.error_code if decision else None
    data = decision.data if decision and allowed else observation(world, identity) if operation is None else None
    value = fact(probe, data) if data is not None else None
    key = digest({'read_intent': raw, 'allowed': allowed, 'code': code, 'fact': value})
    information = 'denial_boundary' if not allowed else 'tests_known_fact' if value == initial_fact(initial_view, probe) else 'potential_new_fact'
    return probe, reason, dict(kind='read', read_intent=raw, prestate_hash=world.state_hash,
        allowed=allowed, error_code=code, expected_fact_hash=digest(value)), dict(
        purpose=probe.purpose, value=information, fact_hash=digest(value),
        learned_by_customer='not_inferred_from_service_tools'), key
