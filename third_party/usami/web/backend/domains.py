"""Desktop research explorer for independent domains, without changing frozen runtimes.

Manual commands use an explicit human-selected actor and server-owned authority.
They never become Service/Customer benchmark grants or scores. Published model
runs are read-only and replay through their original domain benchmark modules.
"""
from dataclasses import dataclass, field
from functools import lru_cache
import importlib
import json
from pathlib import Path
import secrets
import threading
import time
from typing import Any, Dict, Literal

from fastapi import Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import Field
from usami_harness.models import Model
from usami_harness.tools import inline_schema

ROOT = Path(__file__).resolve().parents[2]
COOKIE = 'usami_domain_session'
TTL = 6 * 60 * 60
SPLITS = {'train': 'service-train.json', 'validation': 'service-validation.json',
          'test-demo': 'service-test-demo.json', 'discovery': 'discovery-demo.json'}
DOMAIN_INFO = {
    'saas': {'name': 'SaaS', 'release': 'v0.2.0', 'version': '0.2.0', 'scope': 'organizations',
             'description': '组织、成员、订阅与账单生命周期'},
    'cloudops': {'name': 'CloudOps', 'release': 'cloudops-v0.1', 'version': '0.1.0', 'scope': 'projects',
                 'description': '资源依赖、项目配额与计费恢复'},
    'fulfillment': {'name': 'FulfillmentOps', 'release': 'fulfillment-v0.1', 'version': '0.1.0', 'scope': 'warehouses',
                    'description': '库存守恒、仓储履约、运输与退货'},
    'stayops': {'name': 'StayOps', 'release': 'stayops-v0.1', 'version': '0.1.0', 'scope': 'properties',
                'description': '日期库存、房间分配与住宿生命周期'},
}
DOCUMENTS = {'README.md', 'BUSINESS_MODEL.md', 'REPORT.md', 'QUALITY_AUDIT.md', 'VERIFICATION.md'}


class Generate(Model):
    seed: int = Field(default=42, strict=True)


class ManualOperation(Model):
    actor_id: str = Field(min_length=1)
    operation: Dict[str, Any]
    mode: Literal['reference', 'sandbox'] = 'reference'
    expected_state_hash: str


class Load(Model):
    suite: str
    instance_id: str
    expected_state_hash: str


class Witness(Load):
    route: int = Field(default=0, strict=True, ge=0, le=100)


@lru_cache(maxsize=3)
def runtime(domain):
    if domain not in DOMAIN_INFO or domain == 'saas':
        raise HTTPException(404, 'Unknown independent domain')
    package = 'usami_' + domain
    modules = {name: importlib.import_module(package + '.' + name)
               for name in ('models', 'engine', 'generator', 'rules', 'oracle', 'tools', 'benchmark')}
    modules['scope_field'] = 'project_id' if domain == 'cloudops' else 'scope_id'
    if domain != 'cloudops':
        modules['request'] = importlib.import_module('usami_harness.models').Request
    else:
        modules['request'] = modules['models'].Request
    return modules


def artifact(domain, name):
    if domain not in DOMAIN_INFO:
        raise HTTPException(404, 'Unknown domain')
    path = ROOT / 'benchmarks' / DOMAIN_INFO[domain]['release'] / name
    if not path.is_file():
        raise HTTPException(503, 'Published release artifact unavailable')
    return path


@lru_cache(maxsize=32)
def _read(path, modified):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read(domain, name):
    path = artifact(domain, name)
    return _read(str(path), path.stat().st_mtime_ns)


def suite_data(domain, split):
    runtime(domain)
    if split not in SPLITS:
        raise HTTPException(404, 'Unknown public split')
    return read(domain, SPLITS[split])


def instance_data(domain, split, identifier):
    row = next((i for i in suite_data(domain, split)['instances'] if i['id'] == identifier), None)
    if row is None:
        raise HTTPException(404, 'Instance is not in this domain/split')
    return row


def pilot_data(domain, run_hash):
    runtime(domain)
    pilot = 'pilot' if domain == 'cloudops' else 'pilot-v1'
    root = artifact(domain, 'pilot-report.json').parent / pilot
    # The browser never supplies a filename, filesystem path or uploaded profile.
    for track in ('service', 'discovery'):
        for path in sorted((root / track).glob('*.run.json')):
            run = _read(str(path), path.stat().st_mtime_ns)
            if run['run_hash'] == run_hash:
                return run, root / (track + '-profile.json')
    raise HTTPException(404, 'Unknown published pilot run in this domain')


def metrics(domain, coverage):
    if 'metrics' in coverage:
        return coverage['metrics']
    if domain == 'saas':
        return {'frozen_service_tasks': coverage['accepted_instances'],
                'business_families': len(coverage['business_families']),
                'independent_structures': coverage['structural_diversity']['unique'],
                'semantic_decision_signatures': coverage['semantic_diversity']['unique'],
                'multi_path_tasks': coverage['tasks_with_alternative_routes']}
    families = coverage.get('business_families', 0)
    if isinstance(families, (dict, list)):
        families = len(families)
    return {'frozen_service_tasks': coverage.get('service_tasks', coverage.get('accepted_instances', 0)),
            'business_families': families,
            'independent_structures': coverage.get('independent_structures', 0),
            'semantic_decision_signatures': coverage.get('semantic_decision_signatures', 0),
            'multi_path_tasks': coverage.get('multi_path_tasks', 0)}


@dataclass
class Session:
    domain: str
    initial: Any
    engine: Any
    source: Any = None
    touched: float = field(default_factory=time.monotonic)
    lock: Any = field(default_factory=threading.RLock)


def finalized(session):
    trace = session.engine.trace.model_copy(deep=True)
    if not trace.events or trace.events[-1].kind != 'final':
        trace.append('final', {'state_hash': session.engine.state.state_hash})
    trace.trace_hash = trace.computed_hash()
    return trace


def audit(session):
    return runtime(session.domain)['oracle'].evaluate(finalized(session), session.engine.requests)


def snapshot(session):
    world = session.engine.state
    raw = world.model_dump(mode='json')
    # Some audit ledgers use their actual mapping key as the record identifier
    # (e.g. lot_inspections). Keep those records visible without changing World.
    collections = {key: [dict(record, id=record.get('id', identifier)) for identifier, record in value.items()]
                   for key, value in raw.items()
                   if isinstance(value, dict) and all(isinstance(v, dict) for v in value.values())}
    return {'domain': session.domain, 'world': raw, 'collections': collections,
            'state_hash': world.state_hash, 'initial_state_hash': session.initial.state_hash,
            'exploration_source': session.source, 'kind': 'manual_exploration',
            'summary': {key: len(rows) for key, rows in collections.items()},
            'events': [{'id': e.id, 'kind': e.kind, 'payload': e.payload,
                        'before_hash': None if e.before is None else
                        e.before.state_hash if session.domain == 'cloudops' else runtime(session.domain)['models'].World.model_validate(e.before).state_hash,
                        'after_hash': None if e.after is None else
                        e.after.state_hash if session.domain == 'cloudops' else runtime(session.domain)['models'].World.model_validate(e.after).state_hash}
                       for e in session.engine.trace.events],
            'evaluation': audit(session),
            'invariants': runtime(session.domain)['rules'].invariants(world),
            'operations_used': sum(e.kind == 'operation' for e in session.engine.trace.events),
            'operation_budget': session.engine.trace.operation_budget}


def assert_current(session, expected):
    if expected != session.engine.state.state_hash:
        raise HTTPException(409, '世界已改变，请刷新后重新执行。')


def operate(session, actor, raw, mode):
    from web.backend.app import changes
    rt = runtime(session.domain)
    operation = rt['models'].Operation.model_validate(raw)
    if actor not in session.engine.state.users:
        raise HTTPException(422, 'Unknown manual actor')
    request = rt['request'](id='manual_' + secrets.token_hex(12), identity=actor,
                            operation=operation if session.domain == 'cloudops' else operation.model_dump(mode='json'))
    session.engine.register(request)
    event = session.engine.execute(actor, operation, request.id, mode)
    report = audit(session)
    before = event.before.model_dump(mode='json') if session.domain == 'cloudops' else event.before
    after = event.after.model_dump(mode='json') if session.domain == 'cloudops' else event.after
    return {'event': event.model_dump(mode='json'), 'changes': changes(before, after),
            'evaluation': next(row for row in report['operations'] if row['event_id'] == event.id),
            'kind': 'manual_exploration'}


REFERENCES = {'org_id': 'organizations', 'owner_id': 'users', 'user_id': 'users', 'project_id': 'projects',
              'customer_id': 'users',
              'warehouse_id': 'warehouses', 'source_warehouse_id': 'warehouses', 'destination_warehouse_id': 'warehouses',
              'target_warehouse_id': 'warehouses', 'source_lot_id': 'lots', 'target_lot_id': 'lots',
              'source_id': 'warehouses', 'destination_id': 'warehouses', 'sku_id': 'skus', 'lot_id': 'lots',
              'order_id': 'orders', 'shipment_id': 'shipments', 'return_id': 'returns', 'receipt_id': 'receipts',
              'property_id': 'properties', 'guest_id': 'guests', 'room_type_id': 'room_types', 'rate_plan_id': 'rate_plans',
              'reservation_id': 'reservations', 'room_id': 'rooms', 'current_stay_id': 'stays', 'invoice_id': 'invoices'}


def graph_data(session, scope='', calendar=False):
    data = snapshot(session)
    collections = data['collections']
    scope_kind = DOMAIN_INFO[session.domain]['scope']
    scopes = {v['id'] for v in collections[scope_kind]}
    if scope and scope not in scopes:
        raise HTTPException(404, 'Unknown domain scope')
    nodes, edges = [], []
    for kind, rows in collections.items():
        if kind == 'calendar' and not calendar:
            continue
        for row in rows:
            related = row.get('project_id', row.get('warehouse_id', row.get('property_id')))
            transfer_scopes = {row.get('source_warehouse_id'), row.get('target_warehouse_id')} - {None}
            if not related and row.get('lot_id'):
                lot = next((v for v in collections.get('lots', []) if v['id'] == row['lot_id']), None)
                related = lot.get('warehouse_id') if lot else None
            if scope and ((kind == scope_kind and row['id'] != scope) or (related and related != scope)):
                continue
            if scope and transfer_scopes and scope not in transfer_scopes:
                continue
            label = row.get('name', row.get('number', row.get('status', row['id'])))
            nodes.append({'id': kind + '/' + row['id'], 'kind': kind, 'label': str(label), 'entity': row})
    keys = {n['id'] for n in nodes}
    for node in nodes:
        row = node['entity']
        for field_name, collection in REFERENCES.items():
            value = row.get(field_name)
            if isinstance(value, str) and collection + '/' + value in keys:
                edges.append({'id': node['id'] + ':' + field_name, 'source': collection + '/' + value,
                              'target': node['id'], 'label': row.get('role', field_name.removesuffix('_id')),
                              'kind': field_name, 'entity': row})
        for dep in row.get('dependencies', []):
            if 'instances/' + dep in keys:
                edges.append({'id': node['id'] + ':dependency:' + dep, 'source': 'instances/' + dep,
                              'target': node['id'], 'label': 'required before start', 'kind': 'dependency', 'entity': row})
        for field_name, collection in [('reservation_ids', 'reservations'), ('room_type_ids', 'room_types'), ('previous_rooms', 'rooms')]:
            for index, target in enumerate(row.get(field_name, [])):
                if collection + '/' + target in keys:
                    edges.append({'id': node['id'] + ':' + field_name + ':' + str(index), 'source': collection + '/' + target,
                                  'target': node['id'], 'label': field_name, 'kind': field_name, 'entity': row})
        for index, line in enumerate(row.get('lines', [])):
            if isinstance(line, dict) and 'skus/' + str(line.get('sku_id')) in keys:
                edges.append({'id': node['id'] + ':line:' + str(index), 'source': 'skus/' + line['sku_id'],
                              'target': node['id'], 'label': 'order line · ' + str(line.get('quantity', '')), 'kind': 'order_line', 'entity': line})
    return {'nodes': nodes, 'edges': edges, 'state_hash': data['state_hash']}


def install_domain_routes(app):
    app.state.domain_sessions = {}
    store_lock = threading.RLock()

    def current(domain: str, request: Request):
        runtime(domain)
        key = request.cookies.get(COOKIE)
        with store_lock:
            session = app.state.domain_sessions.get(key, {}).get(domain)
            if session is None or time.monotonic() - session.touched > TTL:
                raise HTTPException(401, '请创建该领域的世界会话。')
            session.touched = time.monotonic()
            return session

    @app.get('/api/domains')
    def catalog():
        return {'domains': [{**info, 'id': domain, 'metrics': metrics(domain, read(domain, 'coverage.json'))}
                            for domain, info in DOMAIN_INFO.items()]}

    @app.post('/api/domains/{domain}/sessions')
    def create(domain: str, command: Generate, request: Request, response: Response):
        rt = runtime(domain)
        world = rt['generator'].generate_world(command.seed)
        with store_lock:
            now = time.monotonic()
            for key in list(app.state.domain_sessions):
                group = app.state.domain_sessions[key]
                if all(now - s.touched > TTL for s in group.values()):
                    del app.state.domain_sessions[key]
            key = request.cookies.get(COOKIE)
            if key not in app.state.domain_sessions:
                if len(app.state.domain_sessions) >= 64:
                    raise HTTPException(503, 'Local session capacity reached')
                key = secrets.token_urlsafe(32)
                app.state.domain_sessions[key] = {}
            session = Session(domain, world.clone(), rt['engine'].Engine(world))
            app.state.domain_sessions[key][domain] = session
        response.set_cookie(COOKIE, key, httponly=True, samesite='strict', max_age=TTL)
        return snapshot(session)

    @app.get('/api/domains/{domain}/world')
    def world(session=Depends(current)):
        with session.lock:
            return snapshot(session)

    @app.post('/api/domains/{domain}/world/generate')
    def generate(command: Generate, session=Depends(current)):
        with session.lock:
            initial = runtime(session.domain)['generator'].generate_world(command.seed)
            session.initial, session.engine = initial.clone(), runtime(session.domain)['engine'].Engine(initial)
            session.source = None
            return snapshot(session)

    @app.post('/api/domains/{domain}/world/reset')
    def reset(session=Depends(current)):
        with session.lock:
            session.engine = runtime(session.domain)['engine'].Engine(session.initial)
            return snapshot(session)

    @app.get('/api/domains/{domain}/tools')
    def tools(domain: str):
        rt = runtime(domain)
        return {'scope_field': rt['scope_field'], 'tools': [
            {'name': name, 'description': rt['tools'].DESCRIPTIONS[name],
             'query': name in (rt['models'].QUERIES if domain == 'cloudops' else rt['models'].Operation.queries),
             'params': inline_schema(model)} for name, model in rt['models'].PARAMETERS.items()]}

    @app.get('/api/domains/{domain}/observation')
    def observation(identity: str, session=Depends(current)):
        with session.lock:
            if identity not in session.engine.state.users:
                raise HTTPException(404, 'Unknown customer identity')
            module = importlib.import_module('usami_' + session.domain + '.observations')
            return module.observation(session.engine.state, identity)

    @app.post('/api/domains/{domain}/operations')
    def execute(command: ManualOperation, session=Depends(current)):
        with session.lock:
            assert_current(session, command.expected_state_hash)
            if sum(e.kind == 'operation' for e in session.engine.trace.events) >= session.engine.trace.operation_budget:
                raise HTTPException(409, '手动操作预算已用完，请重置世界。')
            result = operate(session, command.actor_id, command.operation, command.mode)
            return {'snapshot': snapshot(session), 'operation': result}

    @app.get('/api/domains/{domain}/world/graph')
    def graph(scope: str = '', calendar: bool = False, session=Depends(current)):
        with session.lock:
            return graph_data(session, scope, calendar)

    @app.get('/api/domains/{domain}/events/{event_id}')
    def event(event_id: str, session=Depends(current)):
        with session.lock:
            item = next((e for e in session.engine.trace.events if e.id == event_id), None)
            if item is None:
                raise HTTPException(404, 'Unknown current-world event')
            return item.model_dump(mode='json')

    @app.get('/api/domains/{domain}/export/{kind}')
    def export(kind: str, session=Depends(current)):
        if kind not in {'world', 'trace'}:
            raise HTTPException(404, 'Unknown export')
        with session.lock:
            data = session.engine.state if kind == 'world' else finalized(session)
            return JSONResponse(data.model_dump(mode='json'), headers={
                'Content-Disposition': f'attachment; filename="usami-{session.domain}-{kind}.json"'})

    @app.post('/api/domains/{domain}/replay')
    def replay(session=Depends(current)):
        with session.lock:
            rt = runtime(session.domain)
            engine = rt['engine'].replay_trace(finalized(session), session.engine.requests)
            return {'replayed': True, 'matches_current_state': engine.state == session.engine.state,
                    'final_state_hash': engine.state.state_hash, 'evaluation': audit(session)}

    @app.get('/api/domains/{domain}/benchmark')
    def benchmark(domain: str):
        runtime(domain)
        suites = []
        for split in SPLITS:
            data = suite_data(domain, split)
            suites.append({key: data[key] for key in ('suite_id', 'track', 'split', 'mode', 'tools', 'budget', 'trials', 'manifest_hash', 'dataset_hash')} |
                          {'key': split, 'instance_count': len(data['instances'])})
        pilot = read(domain, 'pilot-report.json')
        coverage = read(domain, 'coverage.json')
        return {'domain': domain, **DOMAIN_INFO[domain], 'suites': suites, 'coverage': coverage,
                'metrics': metrics(domain, coverage), 'service_model': pilot['service_model']['model'],
                'customer_model': pilot['customer_model']['model'],
                'documents': [name for name in ('BUSINESS_MODEL.md', 'REPORT.md', 'QUALITY_AUDIT.md', 'VERIFICATION.md', 'README.md')
                              if (ROOT / 'benchmarks' / DOMAIN_INFO[domain]['release'] / name).is_file()],
                'pilot': {key: pilot.get(key) for key in ('service_report', 'discovery_report', 'interpretation', 'official_full_release_score', 'preflight_attempts')}}

    @app.get('/api/domains/{domain}/benchmark/suites/{split}')
    def instances(domain: str, split: str):
        return {'instances': [{'id': i['id'], 'identity': i['identity'], 'seed': i['world']['seed'],
                               'initial_hash': i['initial_hash'], 'recipe': i.get('generation_recipe'),
                               'task': {k: i['task'][k] for k in ('family', 'kind', 'message', 'structure_signature', 'semantic_signature')} if i['task'] else None}
                              for i in suite_data(domain, split)['instances']]}

    @app.get('/api/domains/{domain}/benchmark/suites/{split}/instances/{identifier}')
    def instance(domain: str, split: str, identifier: str):
        return instance_data(domain, split, identifier)

    @app.post('/api/domains/{domain}/benchmark/load')
    def load(command: Load, session=Depends(current)):
        rt = runtime(session.domain)
        path = artifact(session.domain, SPLITS.get(command.suite, '')) if command.suite in SPLITS else None
        if path is None:
            raise HTTPException(404, 'Unknown split')
        manifest = rt['benchmark'].load(path) if session.domain == 'cloudops' else rt['benchmark'].load(path, verify_generation=False)
        item = next((i for i in manifest.instances if i.id == command.instance_id), None)
        if item is None:
            raise HTTPException(404, 'Instance is not in this domain/split')
        initial = item.world.clone() if session.domain == 'cloudops' else rt['models'].World.model_validate(item.world)
        if initial.state_hash != item.initial_hash:
            raise HTTPException(409, 'Frozen initial world hash differs')
        with session.lock:
            assert_current(session, command.expected_state_hash)
            session.initial, session.engine = initial.clone(), rt['engine'].Engine(initial)
            session.source = {'suite': command.suite, 'instance_id': item.id, 'identity': item.identity, 'kind': 'manual_exploration'}
            return snapshot(session)

    @app.post('/api/domains/{domain}/benchmark/witness')
    def witness(command: Witness, session=Depends(current)):
        with session.lock:
            assert_current(session, command.expected_state_hash)
            if not session.source or session.source['suite'] != command.suite or session.source['instance_id'] != command.instance_id or session.engine.state != session.initial:
                raise HTTPException(409, '先载入该冻结实例或重置；参考路径从冻结初态执行。')
            item = instance_data(session.domain, command.suite, command.instance_id)
            paths = item['task']['witnesses'] if item['task'] else []
            if command.route >= len(paths):
                raise HTTPException(404, 'No legal reference witness at this index')
            results = [operate(session, item['identity'], op, 'reference') for op in paths[command.route]['operations']]
            return {'snapshot': snapshot(session), 'steps': results, 'kind': 'manual_reference_witness', 'model_calls': 0}

    @app.get('/api/domains/{domain}/benchmark/pilots/{run_hash}')
    def pilot(domain: str, run_hash: str):
        return pilot_data(domain, run_hash)[0]

    @app.post('/api/domains/{domain}/benchmark/pilots/{run_hash}/replay')
    def replay_pilot(domain: str, run_hash: str):
        rt = runtime(domain)
        raw, profile = pilot_data(domain, run_hash)
        manifest = rt['benchmark'].load(profile)
        run = rt['benchmark'].Run.model_validate(raw)
        return rt['benchmark'].score(manifest, run)

    @app.get('/api/domains/{domain}/reports/{document}')
    def document(domain: str, document: str):
        runtime(domain)
        if document not in DOCUMENTS:
            raise HTTPException(404, 'Unknown public report')
        return PlainTextResponse(artifact(domain, document).read_text(encoding='utf-8'))
