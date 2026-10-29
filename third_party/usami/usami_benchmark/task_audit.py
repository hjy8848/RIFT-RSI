"""Evidence-derived task coverage and conservative task leakage grouping.

This is an offline evaluator. It never supplies private proofs to an Agent or
changes a task, world, rule, score or selection. A grouping signature is not a
proof that every pair in different groups is a different independent problem.
"""
import importlib
import re
from collections import Counter
from itertools import combinations

from usami_harness.models import digest
from .contracts import DOMAINS
from .evaluation import CAPABILITIES
from .signatures import CATEGORIES, policy_features
from .tasks import CertifiedTask, verify_task
from .tool_tasks import binding, load_world


PROTOCOL = 'certified-task-coverage-v1'
DEPENDENCY_CAPABILITIES = {
    'missing_information': 'information_resolution',
    'preference_resolution': 'preference_negotiation',
    'explicit_consent': 'explicit_consent',
    'cross_turn_commitment': 'cross_turn_commitments',
    'partial_completion': 'mixed_completion_refusal',
}


def _objects(world):
    return {key: (collection, row) for collection, rows in world.items()
            if isinstance(rows, dict) for key, row in rows.items()
            if isinstance(row, dict) and (row.get('id') == key or collection == 'profiles')}


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _abstract(value, field=''):
    """Numeric jitter and free-form prose cannot create extra semantic groups."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return {'number_sign': 'positive' if value > 0 else 'negative' if value < 0 else 'zero'}
    if isinstance(value, str):
        if field in {'name', 'collection', 'field', 'fields', 'mutable_fields', 'selector_field',
                     'comparison', 'action', 'resolution', 'role', 'status', 'option',
                     'service', 'payment_method', 'value', 'accepted_values'}:
            return re.sub(r'(?<=\.)\d+(?=\.|$)', '#', value) if field == 'field' else value
        if field in CATEGORIES:
            return value
        return {'text': True}
    if isinstance(value, list):
        return sorted((_abstract(v, field) for v in value), key=digest)
    if isinstance(value, dict):
        return {k: _abstract(v, k) for k, v in sorted(value.items())}
    raise ValueError('task signature requires JSON data')


def anonymous_task_signature(record):
    """Rooted task relation grouping, independent of IDs, prose and path order.

    Includes only selected business objects, up to two outgoing relation hops,
    and membership records joining the selected identity/context. The complete
    world grouping is reported separately. Six-round colour refinement can
    merge non-isomorphic graphs; it must not be called an equivalence proof.
    Numeric values use sign/policy bins, so collisions are intentionally possible.
    Callers must verify the native task separately before interpreting this hash.
    """
    raw = record.frozen.world
    objects = _objects(raw)
    task = record.task
    if record.track == 'A':
        identity = task.identity
        terms = [('capability', row) for row in task.capabilities]
        terms += [('refusal', row) for row in task.refusal_operations]
        if task.goal:
            terms += [('goal', row.model_dump(mode='json')) for row in task.goal.atoms]
            for key, fields in task.goal.allowed_changes.items():
                collection, entity = key.split('/', 1)
                terms.append(('frame', {'collection': collection, 'entity': entity, 'fields': fields}))
    else:
        identity = raw['profiles'][task.profile_id]['identity']
        terms = [('goal', row.model_dump(mode='json')) for row in task.goals]
        terms += [('refusal', row.operation) for row in task.refusals]
        terms += [('frame', row.model_dump(mode='json')) for row in task.frames]
        terms += [('profile', {'profile_id': task.profile_id})]
        terms += [('external', {'event_id': eid}) for eid in task.external_schedule]
    selected = {s for _, row in terms for s in _strings(row) if s in objects} | {identity}
    for _ in range(2):
        selected |= {s for key in list(selected) if key in objects
                     for s in _strings(objects[key][1]) if s in objects}
    for key, (collection, row) in objects.items():
        refs = set(_strings(row)) - {key}
        if collection in {'access', 'memberships'} and identity in refs and len(refs & selected) > 1:
            selected.add(key)
    labels = {}; edges = {}

    def node(key, label):
        labels[key] = label
        edges[key] = []

    def link(left, relation, right):
        edges[left].append((relation, right))
        edges[right].append(('reverse:' + relation, left))

    world = (load_world(record.frozen) if record.track == 'A' else
             importlib.import_module('usami_' + record.frozen.spec.domain + '.adapter').World.model_validate(raw))
    for key in sorted(selected & objects.keys()):
        collection, row = objects[key]
        categories = {k: _abstract(v, k) for k, v in row.items()
                      if k in CATEGORIES and k not in {'plan_id', 'token'}}
        node(('object', key), {'type': collection, 'categories': categories,
                              'policy': policy_features(record.frozen.spec.domain, world, collection, key),
                              'customer_identity': key == identity})

    def encode(source, value, field='', path=''):
        if isinstance(value, str) and ('object', value) in labels:
            link(source, path, ('object', value))
            return {'reference': objects[value][0]}
        if isinstance(value, dict):
            return {k: encode(source, v, k, path + '.' + k) for k, v in sorted(value.items())}
        if isinstance(value, list):
            ordered = field in {'flights', 'itinerary'}
            rows = [encode(source, v, field, path + (':' + str(i) if ordered else ':item'))
                    for i, v in enumerate(value)]
            return rows if ordered else sorted(rows, key=digest)
        return _abstract(value, field)

    for key in sorted(selected & objects.keys()):
        # Edges preserve business relations, excluding names and signed receipts.
        row = objects[key][1]
        encode(('object', key), {k: v for k, v in row.items()
                               if k not in {'id', 'name', 'payments', 'refunds', 'notice', 'attestation'}})
    for i, (kind, row) in enumerate(terms):
        key = ('term', i)
        node(key, {'type': kind})
        labels[key]['value'] = encode(key, row)
    base = {key: digest(value) for key, value in labels.items()}
    colours = dict(base)
    for _ in range(6):
        colours = {key: digest([base[key], sorted((name, colours[other]) for name, other in edges[key])])
                   for key in labels}
    return digest({'protocol': 'rooted-business-task-group-v1', 'domain': record.frozen.spec.domain,
                   'rules': record.frozen.spec.rules_version, 'nodes': sorted(colours.values())})


def _path_mechanism(operations):
    # Different independent operation order is retained in certificates, but does
    # not by itself count as a distinct business mechanism in this audit.
    return digest(sorted((_abstract(row) for row in operations), key=digest))


def describe_task(record):
    """Rebuild native proof first; derive claims from actual certificate evidence."""
    record = CertifiedTask.model_validate(record.model_dump(mode='json') if isinstance(record, CertifiedTask) else record)
    verify_task(record)
    task = record.task
    evidence = {key: [] for key in CAPABILITIES}
    paths = []; mechanisms = set(); boundaries = set(); dependencies = Counter()
    has_completion = task.goal is not None if record.track == 'A' else bool(task.goals)
    has_refusal = bool(task.refusal_operations) if record.track == 'A' else bool(task.refusals)
    outcome = 'mixed' if has_completion and has_refusal else 'completion' if has_completion else 'refusal'
    difficulty = 'unclassified'
    if record.track == 'A':
        certificate = record.tool_certificate
        world = load_world(record.frozen); native, _ = binding(record.frozen.spec.domain, world)
        for index, witness in enumerate(certificate.witnesses):
            ref = {'certificate_hash': certificate.certificate_hash, 'witness_index': index}
            state = world.clone()
            for operation in witness['operations']:
                decision = native.decide(state, task.identity, native.Operation.model_validate(operation))
                if not decision.allowed:
                    raise ValueError('native witness contains a disallowed operation')
                boundaries.add((operation['name'], 'allowed', 'OK'))
                state = decision.expected_state
            for refusal in witness['refusals']:
                boundaries.add((refusal['operation']['name'], 'refused', refusal['reference_code']))
            omissions = witness['omissions']
            necessary = all(not row['objective_after_remaining_legal_steps'] for row in omissions)
            paths.append({'business_writes': len(witness['operations']), 'causal_depth': witness['causal_depth'],
                          'all_writes_omission_necessary': necessary, 'direct_edges': witness['direct_edges']})
            mechanisms.add(_path_mechanism(witness['operations']))
            evidence['identity_authorization'].append({**ref, 'basis': 'Native execution/refusal uses the frozen identity and exact request.'})
            evidence['side_effect_consistency'].append({**ref, 'basis': 'Certified objective frame and independent native execution/Oracle.'})
            if witness['direct_edges']:
                evidence['causal_dependencies'].append({**ref, 'omission_edges': witness['direct_edges']})
                evidence['state_preconditions'].append({**ref, 'omission_edges': witness['direct_edges']})
            if has_refusal:
                evidence['refusal_decisions'].append({**ref, 'native_codes': [r['reference_code'] for r in witness['refusals']]})
            if outcome == 'mixed':
                evidence['mixed_completion_refusal'].append(ref)
        if has_completion and all(p['all_writes_omission_necessary'] for p in paths):
            shortest = min(p['business_writes'] for p in paths)
            depth = min(p['causal_depth'] for p in paths)
            difficulty = 'basic' if shortest == 1 else 'complex' if depth >= 2 else 'medium'
    else:
        for index, witness in enumerate(record.dialogue_witnesses):
            certificate = witness.certificate
            ref = {'certificate_hash': certificate.certificate_hash, 'witness_index': index,
                   'trace_hash': witness.trace.trace_hash}
            operations = []
            for event in witness.trace.events:
                if event.kind == 'business':
                    operations += [e['payload']['operation'] for e in event.payload['trace']['events']
                                   if e['kind'] == 'operation']
            paths.append({'business_writes': len(operations), 'causal_depth': None,
                          'all_writes_omission_necessary': None, 'direct_edges': []})
            mechanisms.add(_path_mechanism(operations))
            for row in certificate.oracle['operations']:
                # Process and business errors retain their actual independent code.
                operation = next(e['payload']['operation'] for event in witness.trace.events
                                 if event.id == row['dialogue_event_id']
                                 for e in event.payload['trace']['events'] if e['kind'] == 'operation')
                boundaries.add((operation['name'], 'allowed' if row['reference_allowed'] else 'refused', row['reference_error_code'] or 'OK'))
            for goal, reasons in zip(task.refusals, certificate.evaluation['observed_refusal_reasons']):
                for reason in reasons:
                    boundaries.add((goal.operation['name'], 'refused', reason))
            evidence['identity_authorization'].append({**ref, 'basis': 'Verified scoped Customer intent and actual authorization before native writes.'})
            evidence['side_effect_consistency'].append({**ref, 'basis': 'Objective frames and full business/process Oracle and Replay.'})
            if has_refusal:
                evidence['refusal_decisions'].append({**ref, 'native_codes': certificate.evaluation['observed_refusal_reasons']})
            if outcome == 'mixed':
                evidence['mixed_completion_refusal'].append(ref)
            for dependency in certificate.dependencies:
                dependencies[dependency.proof.kind] += 1
                capability = DEPENDENCY_CAPABILITIES.get(dependency.proof.kind)
                if capability:
                    evidence[capability].append({'certificate_hash': dependency.certificate_hash,
                                                 'trace_hash': dependency.trace_hash, 'event_id': dependency.event_id})
    if len(mechanisms) > 1:
        evidence['alternative_paths'] = [{'task_record_hash': record.record_hash, 'distinct_mechanisms': len(mechanisms)}]
    grouping = anonymous_task_signature(record)
    decision_shape = {'domain': record.frozen.spec.domain, 'rules': record.frozen.spec.rules_version,
                      'outcome': outcome, 'boundaries': [list(row) for row in sorted(boundaries)],
                      'dependencies': sorted(dependencies),
                      'causal_depths': sorted({p['causal_depth'] for p in paths}, key=str)}
    return {'protocol': PROTOCOL, 'task_hash': record.task_hash, 'record_hash': record.record_hash,
            'domain': record.frozen.spec.domain, 'track': record.track, 'family': task.family,
            'partition': record.frozen.spec.partition, 'world_hash': record.frozen.world_hash,
            'registration_hash': record.frozen.registration_hash, 'world_structure': record.frozen.structure_signature,
            'task_structure_signature': grouping, 'decision_signature': digest(decision_shape),
            'decision_shape': decision_shape, 'outcome': outcome, 'difficulty': difficulty,
            'difficulty_protocol': 'omission-certified-tool-depth-v1; dialogue/refusal unclassified',
            'paths': paths, 'mechanisms': len(mechanisms), 'dependency_categories': dict(sorted(dependencies.items())),
            'capabilities': {key: {'status': 'CERTIFIED_WITNESS_EVIDENCE' if refs else 'NOT_AUDITED', 'evidence': refs}
                             for key, refs in evidence.items()},
            'scope': 'Certificate coverage, not model performance or exhaustive rule coverage. No N/A is inferred from missing evidence.'}


def audit_tasks(records, *, per_domain_target=None):
    if per_domain_target is not None and (type(per_domain_target) is not int or per_domain_target <= 0):
        raise ValueError('positive per-domain target required')
    rows = [describe_task(record) for record in records]
    if not rows or len({r['task_hash'] for r in rows}) != len(rows):
        raise ValueError('audit requires distinct, nonempty certified tasks')
    domains = {}
    for domain in DOMAINS:
        selected = [r for r in rows if r['domain'] == domain]
        counts = Counter((capability, r['family'], r['difficulty'], r['outcome'])
                         for r in selected for capability, item in r['capabilities'].items() if item['evidence'])
        groups = Counter(r['task_structure_signature'] for r in selected)
        domains[domain] = {
            'tasks': len(selected), 'target': per_domain_target,
            'shortfall': max(0, per_domain_target - len(selected)) if per_domain_target else None,
            'worlds': len({r['world_hash'] for r in selected}), 'task_structure_groups': len(groups),
            'structural_group_shortfall': max(0, per_domain_target - len(groups)) if per_domain_target else None,
            'duplicate_group_fraction': 1 - len(groups) / len(selected) if selected else None,
            'decision_groups': len({r['decision_signature'] for r in selected}),
            'families': dict(sorted(Counter(r['family'] for r in selected).items())),
            'difficulty': dict(sorted(Counter(r['difficulty'] for r in selected).items())),
            'outcomes': dict(sorted(Counter(r['outcome'] for r in selected).items())),
            'capabilities': {key: sum(bool(r['capabilities'][key]['evidence']) for r in selected) for key in CAPABILITIES},
            'matrix': [{'capability': key[0], 'family': key[1], 'difficulty': key[2], 'outcome': key[3], 'tasks': n}
                       for key, n in sorted(counts.items())],
            'causal_depth': dict(sorted(Counter(str(p['causal_depth']) for r in selected for p in r['paths']).items())),
            'causal_depth_scope': 'Per certified witness, not per-task weighting; unknown dialogue depth is not zero.',
        }
    partitions = sorted({r['partition'] for r in rows}); overlaps = {}
    for left, right in combinations(partitions, 2):
        def shared(field):
            return [list(item) for item in sorted({(r['domain'], r[field]) for r in rows if r['partition'] == left}
                          & {(r['domain'], r[field]) for r in rows if r['partition'] == right})]
        overlaps[left + ' / ' + right] = {field: shared(field) for field in
            ('world_hash', 'world_structure', 'task_structure_signature', 'decision_signature', 'family')}
    report = {'protocol': PROTOCOL, 'task_count': len(rows), 'domains': domains, 'rows': rows,
            'domain_weight': {d: 1 / 8 for d in DOMAINS}, 'track_weight': {'A': .5, 'B': .5},
            'split_overlaps': overlaps,
            'split_status': 'AUDITED' if len(partitions) > 1 else 'NOT_TESTED_SINGLE_PARTITION',
            'world_and_task_structure_isolation': (not any(v[field] for v in overlaps.values()
                for field in ('world_hash', 'world_structure', 'task_structure_signature'))) if len(partitions) > 1 else None,
            'interpretation': 'All overlap groups are disclosed. Shared coarse decisions/families need a preregistered generalization policy; they are not silently treated as private test independence.',
            'grouping_limit': 'Rooted colour refinement is conservative grouping, not graph isomorphism or a count of independently new business problems. Text/names/IDs and raw numeric jitter do not establish novelty.',
            'model_experiments': 'NOT RUN', 'coverage_hash': digest(rows)}
    report['report_hash'] = digest(report)
    return report


def verify_audit(records, report):
    """Recompute quantities and all evidence; a self-resealed label is not proof."""
    if report.get('report_hash') != digest({k: v for k, v in report.items() if k != 'report_hash'}):
        raise ValueError('task audit integrity differs')
    targets = {row['target'] for row in report['domains'].values()}
    if len(targets) != 1:
        raise ValueError('task audit must use one equal per-domain target')
    rebuilt = audit_tasks(records, per_domain_target=next(iter(targets)))
    if rebuilt != report:
        raise ValueError('task coverage/partition/evidence audit does not reproduce')
    return True


def audit_archive(archive, output, *, per_domain_target=None):
    """Create a separate immutable audit; never replace the source archive."""
    import gzip
    import json
    from pathlib import Path
    from vast_coevo.schema import atomic_json
    from .artifacts import verify_archive
    from .audit import snapshot_sources

    archive = Path(archive)
    manifest = json.loads((archive / 'manifest.json').read_text())
    records = []
    for entry in manifest['cases']:
        name = Path(entry['path'])
        if name.name != str(name) or name.suffix != '.gz':
            raise ValueError('archive case path escapes directory')
        records.append(CertifiedTask.model_validate(json.loads(gzip.decompress((archive / name).read_bytes()))['task']))
    output = Path(output)
    if any(r.frozen.spec.partition == 'heldout' for r in records):
        try:
            output.resolve().relative_to(Path(__file__).resolve().parents[1])
        except ValueError:
            pass
        else:
            raise ValueError('private Held-out task audit must remain outside the public repository')
    output.mkdir(mode=0o700 if any(r.frozen.spec.partition == 'heldout' for r in records) else 0o755,
                 parents=True, exist_ok=False)
    snapshot_sources(output)
    atomic_json(output / 'registration.json', {
        'protocol': PROTOCOL, 'source_manifest_hash': manifest['manifest_hash'],
        'per_domain_target': per_domain_target, 'task_hashes': [r.task_hash for r in records],
        'model_calls': 0,
    }, immutable=True)
    try:
        verified = verify_archive(archive)
        report = audit_tasks(records, per_domain_target=per_domain_target)
        # Roundtrip before announcing success, including tuple/list stability.
        loaded = json.loads(json.dumps(report, sort_keys=True))
        verify_audit(records, loaded)
        atomic_json(output / 'coverage.json', loaded, immutable=True)
        result = {'verified': True, 'tasks': len(records), 'source_archive': verified,
                  'report_hash': report['report_hash'], 'model_calls': 0}
        atomic_json(output / 'verification.json', result, immutable=True)
        return result
    except Exception as exc:
        atomic_json(output / 'failure.json', {'type': type(exc).__name__, 'message': str(exc),
                                            'retry': 'not attempted'}, immutable=True)
        raise
