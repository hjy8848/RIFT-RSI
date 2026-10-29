"""Preregistered native candidate certification and honest balanced release selection.

No task conditions generation; no model performance influences sampling. Failed
construction and unfilled strata remain visible. Never substitute duplicated
questions or unclassified dialogue tasks to meet a requested release size.
"""
import gzip
import json
from collections import Counter
from pathlib import Path
from pydantic import Field, StrictInt, model_validator
from usami_harness.models import Model, digest
from vast_coevo.schema import atomic_json
from .contracts import DOMAINS, WorldSpec
from .tasks import ConstructionBudget, CertifiedTask, compile_task
from .worlds import freeze, WorldPool
from .candidates import scheduled_selections
from .audit import snapshot_sources
from .task_audit import describe_task, audit_tasks


class DatasetPlan(Model):
    protocol: str = 'native-balanced-dataset-v1'
    worlds: list[WorldSpec] = Field(min_length=1, max_length=2000)
    candidates_per_world: int = Field(default=32, strict=True, ge=1, le=512)
    construction_budget: ConstructionBudget = Field(default_factory=ConstructionBudget)
    quotas: dict[str, StrictInt] = Field(default_factory=lambda: {
        'basic': 8, 'medium': 12, 'complex': 8, 'refusal': 12})
    known_world_hashes: list[str] = Field(default_factory=list)
    known_structure_signatures: list[str] = Field(default_factory=list)
    @model_validator(mode='after')
    def checked(self):
        if self.protocol != 'native-balanced-dataset-v1':
            raise ValueError('unknown dataset protocol')
        if len({s.spec_hash for s in self.worlds}) != len(self.worlds):
            raise ValueError('duplicate independent world specification')
        if set(self.quotas) != {'basic', 'medium', 'complex', 'refusal'} or any(type(v) is not int or v < 0 for v in self.quotas.values()) or not sum(self.quotas.values()):
            raise ValueError('nonnegative four-stratum quotas required')
        for values in (self.known_world_hashes,self.known_structure_signatures):
            if len(values)!=len(set(values)) or any(len(v)!=64 or set(v)-set('0123456789abcdef') for v in values):
                raise ValueError('known training registry requires distinct exact hashes')
        return self


def select_balanced(records, quotas):
    """Input is private certified tasks; selection never reads model results."""
    if set(quotas) != {'basic', 'medium', 'complex', 'refusal'} or any(type(v) is not int or v < 0 for v in quotas.values()):
        raise ValueError('invalid frozen stratum quotas')
    rows = [describe_task(record) for record in records]
    if len({r['task_hash'] for r in rows}) != len(rows):
        raise ValueError('duplicate task records')
    selected = []; excluded = []; counts = {d: Counter() for d in DOMAINS}; seen = set()
    for row in sorted(rows, key=lambda r: r['task_hash']):
        key = (row['domain'], row['task_structure_signature'])
        stratum = 'refusal' if row['outcome'] == 'refusal' else row['difficulty']
        if key in seen:
            excluded.append({'task_hash': row['task_hash'], 'reason': 'duplicate_rooted_task_structure'})
        elif stratum not in quotas:
            excluded.append({'task_hash': row['task_hash'], 'reason': 'no_certified_difficulty_stratum'})
        elif counts[row['domain']][stratum] >= quotas[stratum]:
            excluded.append({'task_hash': row['task_hash'], 'reason': 'stratum_quota_full'})
        else:
            selected.append(row['task_hash']); seen.add(key); counts[row['domain']][stratum] += 1
    shortfalls = {d: {s: max(0, n - counts[d][s]) for s, n in quotas.items()} for d in DOMAINS}
    return {'protocol': 'native-balanced-selection-v1', 'quotas_per_domain': quotas,
            'selected': selected, 'excluded': excluded,
            'counts': {d: dict(sorted(counts[d].items())) for d in DOMAINS},
            'shortfalls': shortfalls, 'complete': not any(n for row in shortfalls.values() for n in row.values()),
            'mixed_tasks': sum(r['task_hash'] in selected and r['outcome'] == 'mixed' for r in rows),
            'scope': 'Rooted grouping is conservative, not a proof of independent task novelty. Missing evidence cannot fill a quota.'}


def _write_record(path, record):
    data = json.dumps(record.model_dump(mode='json'), sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    with path.open('xb') as stream:
        stream.write(gzip.compress(data, mtime=0))


def build(plan, root):
    plan = DatasetPlan.model_validate(plan.model_dump(mode='json') if isinstance(plan, DatasetPlan) else plan)
    root = Path(root)
    if any(s.partition == 'heldout' for s in plan.worlds):
        try:
            root.resolve().relative_to(Path(__file__).resolve().parents[1])
        except ValueError:
            pass
        else:
            raise ValueError('private Held-out dataset must remain outside public repository')
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    snapshot_sources(root)
    atomic_json(root / 'plan.json', {'plan': plan.model_dump(mode='json'), 'plan_hash': digest(plan)}, immutable=True)
    records = []; attempts = []; current_stage = 'world_generation'
    try:
        # Freeze the entire independent pool before enumerating ANY tasks.
        worlds = [freeze(spec) for spec in plan.worlds]
        if any(w.spec.partition != 'train' and (w.world_hash in plan.known_world_hashes or w.structure_signature in plan.known_structure_signatures) for w in worlds):
            raise ValueError('previously known world cannot be relabelled as independent evaluation')
        WorldPool(worlds).save(root / 'world-pool.json')
        current_stage = 'candidate_registration'; schedule = []
        for world in worlds:
            for selection in scheduled_selections(world, plan.candidates_per_world):
                schedule.append({'world_registration': world.registration_hash, 'selection': selection.model_dump(mode='json')})
        atomic_json(root / 'candidate-schedule.json', {'schedule': schedule, 'schedule_hash': digest(schedule),
            'method': 'Finite evaluator-side native enumeration, not autonomous Designer evidence.'}, immutable=True)
        by_world = {w.registration_hash: w for w in worlds}
        (root / 'records').mkdir(); (root / 'attempts').mkdir(); current_stage = 'native_certification'
        for index, row in enumerate(schedule):
            item = {'index': index, 'candidate_hash': digest(row), 'world_registration': row['world_registration']}
            try:
                record = compile_task(by_world[row['world_registration']], row['selection'], plan.construction_budget)
                if any(r.task_hash == record.task_hash for r in records):
                    raise ValueError('duplicate certified task hash')
                _write_record(root / 'records' / (record.task_hash + '.json.gz'), record)
                records.append(record)
                item.update(status='certified', task_hash=record.task_hash, record_hash=record.record_hash)
            except ValueError as exc:
                item.update(status='construction_rejected', reason=str(exc))
            attempts.append(item)
            atomic_json(root / 'attempts' / (str(index) + '.json'), item, immutable=True)
        current_stage = 'balanced_selection'
        selection = select_balanced(records, plan.quotas)
        atomic_json(root / 'selection.json', selection, immutable=True)
        coverage = audit_tasks(records, per_domain_target=sum(plan.quotas.values())) if records else None
        if coverage:
            atomic_json(root / 'candidate-coverage.json', coverage, immutable=True)
        manifest = {'protocol': 'native-balanced-dataset-manifest-v1', 'plan_hash': digest(plan),
            'schedule_hash': digest(schedule), 'attempts': len(attempts), 'certified': len(records),
            'selected': selection['selected'], 'quota_complete': selection['complete'], 'shortfalls': selection['shortfalls'],
            'release_status': 'CANDIDATE_BANK',
            'model_experiments': 'NOT RUN', 'model_calls': 0,
            'record_hashes': {r.task_hash: r.record_hash for r in records},
            'coverage_hash': coverage['report_hash'] if coverage else None,
            'scope': 'Finite independently generated candidate bank and stratified task selection. Incomplete strata are not a completed Pilot/Core release.'}
        manifest['manifest_hash'] = digest(manifest)
        atomic_json(root / 'manifest.json', manifest, immutable=True)
        return manifest
    except Exception as exc:
        atomic_json(root / 'failure.json', {'stage': current_stage, 'type': type(exc).__name__, 'reason': str(exc),
            'saved_certified_tasks': len(records), 'saved_attempts': len(attempts), 'retry': 'not attempted'}, immutable=True)
        raise


def verify_dataset(root):
    from .artifacts import verify_sources
    root = Path(root); verify_sources(root)
    payload = json.loads((root / 'plan.json').read_text()); plan = DatasetPlan.model_validate(payload['plan'])
    if payload['plan_hash'] != digest(plan):
        raise ValueError('dataset plan hash differs')
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('protocol') != 'native-balanced-dataset-manifest-v1' or manifest['manifest_hash'] != digest({k: v for k, v in manifest.items() if k != 'manifest_hash'}):
        raise ValueError('dataset manifest hash differs')
    worlds = [freeze(spec) for spec in plan.worlds]
    if any(w.spec.partition != 'train' and (w.world_hash in plan.known_world_hashes or w.structure_signature in plan.known_structure_signatures) for w in worlds):
        raise ValueError('previously known world cannot be relabelled as independent evaluation')
    saved = json.loads((root / 'world-pool.json').read_text())
    pool = WorldPool(worlds)
    if saved['pool_hash'] != pool.pool_hash or saved['worlds'] != [pool.load(k).model_dump(mode='json') for k in sorted(pool.entries)]:
        raise ValueError('independent world pool differs')
    schedule = [{'world_registration': w.registration_hash, 'selection': s.model_dump(mode='json')}
                for w in worlds for s in scheduled_selections(w, plan.candidates_per_world)]
    registered = json.loads((root / 'candidate-schedule.json').read_text())
    if registered['schedule'] != schedule or registered['schedule_hash'] != digest(schedule) or manifest['schedule_hash'] != digest(schedule):
        raise ValueError('finite candidate schedule differs')
    by_world = {w.registration_hash: w for w in worlds}; records = []
    if {p.name for p in (root/'attempts').glob('*.json')} != {str(i)+'.json' for i in range(len(schedule))}:
        raise ValueError('candidate attempt file set differs')
    for i, row in enumerate(schedule):
        attempt = json.loads((root / 'attempts' / (str(i) + '.json')).read_text())
        expected = {'index': i, 'candidate_hash': digest(row), 'world_registration': row['world_registration']}
        try:
            record = compile_task(by_world[row['world_registration']], row['selection'], plan.construction_budget)
        except ValueError as exc:
            expected.update(status='construction_rejected', reason=str(exc))
        else:
            raw = json.loads(gzip.decompress((root / 'records' / (record.task_hash + '.json.gz')).read_bytes()))
            if CertifiedTask.model_validate(raw) != record:
                raise ValueError('saved native certificate differs')
            records.append(record); expected.update(status='certified', task_hash=record.task_hash, record_hash=record.record_hash)
        if attempt != expected:
            raise ValueError('candidate outcome/failure evidence differs')
    actual = select_balanced(records, plan.quotas)
    if {p.name for p in (root/'records').glob('*.json.gz')} != {r.task_hash+'.json.gz' for r in records}:
        raise ValueError('native record file set differs')
    if actual != json.loads((root / 'selection.json').read_text()) or manifest['selected'] != actual['selected'] or manifest['shortfalls'] != actual['shortfalls'] or manifest['quota_complete'] != actual['complete']:
        raise ValueError('balanced selection or shortfalls differ')
    coverage = audit_tasks(records, per_domain_target=sum(plan.quotas.values())) if records else None
    if coverage is not None and json.loads((root / 'candidate-coverage.json').read_text()) != coverage:
        raise ValueError('candidate coverage differs')
    if manifest['release_status'] != 'CANDIDATE_BANK' or manifest['plan_hash'] != digest(plan) or manifest['attempts'] != len(schedule) or manifest['certified'] != len(records) or manifest['record_hashes'] != {r.task_hash: r.record_hash for r in records} or manifest['coverage_hash'] != (coverage['report_hash'] if coverage else None) or manifest['model_experiments'] != 'NOT RUN' or manifest['model_calls'] != 0:
        raise ValueError('native dataset provenance or counts differ')
    return {'verified': True, 'attempts': len(schedule), 'certified': len(records), 'selected': len(actual['selected']),
            'quota_complete': actual['complete'], 'release_status': 'CANDIDATE_BANK', 'model_calls': 0}
