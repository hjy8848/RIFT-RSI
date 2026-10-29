"""Per-write decision audit from actual model-visible messages. Never enforcement."""
import json
from usami_cloudops.models import Operation
from usami_cloudops.rules import decide, role, owner, usage, invariants
from vast_coevo.schema import digest


def tool_facts(record, actor):
    """Successful native query/write result facts, never model prose or Oracle."""
    if not record.response.get('success'): return {}
    try: args = json.loads(record.arguments)
    except (ValueError, TypeError): return {}
    name, data, values = record.name, record.response.get('data', {}), {}
    pid = args.get('project_id')
    def project(row):
        key = row.get('id', pid)
        for field in ('status', 'quota', 'usage'):
            if field in row: values[f'project:{key}:{field}'] = row[field]
    def members(project_id, rows):
        rows = [row for row in rows if row['project_id'] == project_id]
        values[f'project:{project_id}:actor_role'] = next((row['role'] for row in rows
            if row['user_id'] == actor and row['project_id'] == project_id), None)
        values[f'project:{project_id}:members'] = rows
    def instance(row):
        for field in ('status', 'dependencies', 'cpu', 'memory'):
            if field in row: values[f'instance:{row["id"]}:{field}'] = row[field]
    if name == 'get_customer_context':
        orgs = {row['id']: row for row in data.get('organizations', [])}
        for row in data.get('projects', []):
            project(row); members(row['id'], data.get('project_access', []))
            values[f'project:{row["id"]}:owner'] = orgs.get(row['org_id'], {}).get('owner_id') == actor
            running = [i for i in data.get('instances', []) if i['project_id'] == row['id'] and i['status'] == 'running']
            values[f'project:{row["id"]}:usage'] = {'cpu': sum(i['cpu'] for i in running),
                'memory': sum(i['memory'] for i in running), 'instances': len(running)}
        for row in data.get('instances', []): instance(row)
        for row in data.get('invoices', []): values[f'invoice:{row["id"]}:status'] = row['status']
    elif name == 'get_project': project(data)
    elif name == 'get_project_members': members(pid, data.get('access', []))
    elif name == 'list_instances':
        for row in data.get('instances', []): instance(row)
    elif name == 'get_instance' and 'id' in data: instance(data)
    elif name == 'get_invoice' and 'id' in data: values[f'invoice:{data["id"]}:status'] = data['status']
    elif name in {'start_instance', 'stop_instance'} and 'instance_id' in data and 'status' in data:
        values[f'instance:{data["instance_id"]}:status'] = data['status']
    elif name == 'change_project_quota' and all(key in data for key in ('cpu', 'memory', 'instances')):
        values[f'project:{pid}:quota'] = {key: data[key] for key in ('cpu', 'memory', 'instances')}
    elif name == 'resume_project' and 'status' in data: values[f'project:{pid}:status'] = data['status']
    return {key: dict(value=value, source_call_id=record.call_id, source_tool=name) for key, value in values.items()}


def update_observations(observed, record, actor):
    fresh = tool_facts(record, actor)
    if record.name in {'start_instance', 'stop_instance'} and record.response.get('success'):
        args = json.loads(record.arguments)
        rid, pid = args.get('resource_id'), args.get('project_id')
        status = f'instance:{rid}:status'; used = f'project:{pid}:usage'
        required = (status, used, f'instance:{rid}:cpu', f'instance:{rid}:memory')
        if status in fresh and all(key in observed for key in required):
            delta = int(fresh[status]['value'] == 'running') - int(observed[status]['value'] == 'running')
            prior = observed[used]['value']
            fresh[used] = dict(value={'cpu': prior['cpu'] + delta*observed[required[2]]['value'],
                'memory': prior['memory'] + delta*observed[required[3]]['value'], 'instances': prior['instances'] + delta},
                source_call_id=record.call_id, source_tool=record.name,
                derivation='prior observed usage/status/sizing plus actual successful write result')
    observed.update(fresh)


def required_facts(world, actor, operation):
    """Public rule predicates for supported writes, with privileged actual values."""
    op, values = operation, {}
    if invariants(world): return {}, 'censored_invalid_prestate'
    project = world.projects.get(op.project_id)
    if not project: return {}, 'unresolvable_project'
    prefix = f'project:{project.id}:'
    compute = {'start_instance','stop_instance','create_instance','delete_instance','attach_dependency','detach_dependency'}
    values[prefix+'actor_role'] = role(world, actor, project.id)
    if op.name not in compute: values[prefix+'owner'] = owner(world, actor, project.id)
    instance = world.instances.get(op.resource_id)
    if instance and op.name in compute:
        values[f'instance:{instance.id}:status'] = instance.status
        values[f'instance:{instance.id}:dependencies'] = instance.dependencies
    if op.name == 'start_instance':
        values[prefix+'status'] = project.status
        values[prefix+'quota'] = project.quota.model_dump(mode='json')
        values[prefix+'usage'] = usage(world, project.id)
        if instance:
            values[f'instance:{instance.id}:cpu'] = instance.cpu
            values[f'instance:{instance.id}:memory'] = instance.memory
            for dependency in instance.dependencies: values[f'instance:{dependency}:status'] = world.instances[dependency].status
    if op.name == 'change_project_quota': values[prefix+'usage'] = usage(world, project.id)
    if op.name == 'resume_project':
        invoice = world.invoices.get(op.params['invoice_id'])
        if invoice: values[f'invoice:{invoice.id}:status'] = invoice.status
    if op.name in {'grant_project_access','revoke_project_access'}:
        values[prefix+'members'] = [row.model_dump(mode='json') for row in world.access.values() if row.project_id == project.id]
    supported = {'start_instance', 'change_project_quota', 'resume_project', 'grant_project_access', 'revoke_project_access'}
    return values, 'public_predicate_projection' if op.name in supported else 'partial_predicate_projection'


def scope_audit(snapshot):
    """Conservative domain-specific-token audit; not a semantic equivalence proof."""
    from usami_cloudops.models import WRITES
    risks = []
    for skill in snapshot.skills:
        if skill.type != 'decision': continue
        text = json.dumps(skill.model_dump(mode='json', exclude={'scope','trigger','content_hash'}), ensure_ascii=False)
        mentions = sorted(name for name in WRITES if name in text)
        domains = set(skill.scope.domains)
        if mentions and domains - {'cloudops'}:
            risks.append(dict(skill_id=skill.id, content_hash=skill.content_hash, declared_domains=skill.scope.domains,
                domain_specific_operations=mentions, unvalidated_semantic_equivalence=sorted(domains-{'cloudops'}),
                recommendation='keep historical snapshot; split or author a separately scoped candidate before future selection'))
    return risks


def audit_writes(evidence, execution, service_snapshot):
    tools = {record.call_id: record for record in execution.tool_records}
    calls = {}
    for call in evidence.model_calls:
        if call.role != 'service' or not isinstance(call.response, dict): continue
        try: entries = call.response['choices'][0]['message'].get('tool_calls') or []
        except (TypeError, KeyError, IndexError): continue
        for tool in entries: calls[tool['id']] = call
    oracle = {row['event_id']: row for row in execution.oracle['operations']}
    by_event = {record.event_id: record for record in execution.tool_records if record.event_id}
    rows = []
    for event in execution.world_trace.events:
        if event.kind != 'operation': continue
        operation = Operation.model_validate(event.payload['operation'])
        if operation.is_query: continue
        record, row = by_event[event.id], oracle[event.id]
        call = calls.get(record.call_id)
        observed, source_refs = {}, []
        if call is not None:
            for index, message in enumerate(call.base_request.get('messages', [])):
                if message.get('role') != 'tool': continue
                prior = tools.get(message.get('tool_call_id'))
                if prior is None: continue
                # Verify the body actually presented to this decision call.
                try: same = json.loads(message['content']) == prior.response
                except (ValueError, TypeError): same = False
                if same:
                    update_observations(observed, prior, evidence.config.identity)
                    source_refs.append(f'{call.id}/base_request/messages/{index}')
        required, scope = required_facts(event.before, evidence.config.identity, operation)
        facts = []
        for key, actual in required.items():
            available = observed.get(key)
            facts.append(dict(fact=key, actual=actual, observed=available,
                status='not_query_verified' if available is None else 'matches_prestate' if available['value'] == actual else 'stale_or_incorrect'))
        injections = [entry for entry in evidence.injections if call and entry.call_id == call.id and entry.role == 'service']
        injected = [entry for entry in injections if entry.rendered and
                    any(message.get('content') == entry.rendered for message in call.request.get('messages', []))]
        decision = decide(event.before, evidence.config.identity, operation)
        prefix = f'project:{operation.project_id}:'
        blocking = {'PROJECT_ACCESS_REQUIRED': [prefix+'actor_role'],
            'PERMISSION_DENIED': [prefix+'actor_role', prefix+'owner'],
            'PROJECT_SUSPENDED': [prefix+'status']}.get(decision.error_code, [])
        if decision.error_code == 'DEPENDENCY_NOT_RUNNING':
            instance = event.before.instances[operation.resource_id]
            blocking = [f'instance:{instance.id}:dependencies'] + [f'instance:{key}:status'
                for key in instance.dependencies if event.before.instances[key].status != 'running']
        blocking_verified = bool(blocking) and all(any(fact['fact'] == key and fact['status'] == 'matches_prestate'
            for fact in facts) for key in blocking)
        labels = []
        if not injected: labels.append('no_skill_in_this_decision_request')
        if not row['authorized']: labels.append('unrequested_or_consumed_write_grant')
        if any(fact['status'] == 'stale_or_incorrect' for fact in facts): labels.append('stale_or_incorrect_query_facts')
        elif any(fact['status'] == 'not_query_verified' for fact in facts): labels.append('insufficient_query_evidence_not_proof_of_model_ignorance')
        if row['classification'] == 'successful_business_violation' and (blocking_verified or (facts and
                all(fact['status'] == 'matches_prestate' for fact in facts))):
            labels.append('decision_inconsistent_with_verified_query_facts')
        rows.append(dict(event_id=event.id, model_call_id=call.id if call else None, tool_call_id=record.call_id,
            customer_message=next((message.get('content') for message in reversed(call.base_request.get('messages', []))
                if message.get('role') == 'user'), None) if call else None,
            operation=operation.model_dump(mode='json'), prestate_hash=event.before.state_hash,
            actual_skill_injection=[dict(snapshot_hash=entry.snapshot_hash, hits=[hit.model_dump(mode='json') for hit in entry.hits],
                rendered=entry.rendered, request_hash=entry.request_hash) for entry in injected],
            exact_customer_authorized=row['authorized'], reference_allowed=row['reference_allowed'],
            reference_error_code=decision.error_code, outcome=row, facts=facts, fact_projection_scope=scope,
            diagnosis=labels, model_visible_tool_source_refs=source_refs,
            reference_blocking_fact_keys=blocking, reference_blocking_facts_verified=blocking_verified,
            customer_assertions='unverified prose; not silently promoted to queried facts',
            interpretation='observable decision evidence only; no inference about private reasoning'))
    return dict(schema_version=1, kind='cloudops-service-write-audit', episode_id=evidence.episode_id,
        config_id=evidence.config.config_id, track=evidence.track, partition=evidence.config.partition,
        termination=evidence.termination,
        evidence_hash=evidence.evidence_hash, service_snapshot_hash=service_snapshot.snapshot_hash,
        writes=rows, scope_risks=scope_audit(service_snapshot), sandbox_unchanged=True)
