from .models import PARAMETERS, TOOLS, Operation
DESCRIPTIONS = {
 'create_instance': 'Create a stopped project-local instance using a customer-chosen unused instance_id, name, CPU and memory.',
 'start_instance': 'Start after billing recovery, running dependencies, scoped access and quota checks.',
 'stop_instance': 'Stop only after all running direct dependents are stopped.',
 'delete_instance': 'Delete a stopped instance only after all incoming references are detached.',
 'attach_dependency': 'Add a project-local DAG edge while the dependent instance is stopped.',
 'detach_dependency': 'Remove a dependency while the dependent is stopped.',
 'change_project_quota': 'Change running CPU/memory/count quota, maximum 64 CPU, 256 memory units and 16 instances.',
 'grant_project_access': 'Owner/Admin grants a project role. Owner may grant itself an operational role. Project Admin cannot change Admin grants.',
 'revoke_project_access': 'Remove an existing scoped grant; Owner identity is immutable.',
 'pay_invoice': 'Simulated full charge. card_approved succeeds; card_declined records a failure. Scoped immutable idempotency key.',
 'resume_project': 'Restore only with the current paid project invoice; payment alone does not resume.',
 'get_project': 'Query project status, quota, billing reference and current running usage.',
 'list_instances': 'Query scoped resources and dependencies.', 'get_instance': 'Query one scoped instance.',
 'get_invoice': 'Owner/Admin query one scoped invoice and payment history.',
 'get_project_members': 'Query scoped project access.', 'get_customer_context': 'Query permitted project catalog for the fixed authenticated identity.'}


def schemas(names):
    if len(set(names)) != len(names) or set(names) - set(TOOLS): raise ValueError('invalid CloudOps tool set')
    result = []
    for name in names:
        if name == 'get_customer_context': shape = {'type':'object','properties':{},'additionalProperties':False}
        else: shape = {'type':'object','properties':{'project_id': {'type':'string'}, 'resource_id':{'type':'string'},
            'params': PARAMETERS[name].model_json_schema()}, 'required':['project_id','resource_id','params'], 'additionalProperties':False}
        result.append({'type':'function','function':{'name':name,'description':DESCRIPTIONS[name],'parameters':shape}})
    return result


def parse(name, arguments, names):
    from usami.service.models import strict_json
    if name not in names: raise ValueError('tool is not in frozen CloudOps interface')
    args = strict_json(arguments)
    if name == 'get_customer_context':
        if args != {}: raise ValueError('context tool takes no arguments')
        return None
    return Operation(name=name, **args)
