"""Immutable eight-domain task/episode archives with complete runtime source pins."""
import gzip,hashlib,json
from collections import Counter
from pathlib import Path
from usami_harness.models import digest
from vast_coevo.schema import atomic_json
from .audit import snapshot_sources
from .tasks import CertifiedTask,verify_task
from .episodes import EpisodeExecution,replay


def verify_sources(root):
    root=Path(root);pins=json.loads((root/'source-pins.json').read_text());repo=Path(__file__).resolve().parents[1]
    folders=[p for p in repo.glob('usami*') if p.is_dir()]+[repo/'integrations',repo/'vast_coevo']
    actual={str(p.relative_to(repo)) for folder in folders for p in folder.rglob('*.py')}
    if not pins or set(pins)!=actual:raise ValueError('runtime source set differs; use the archived source snapshot')
    for relative,expected in pins.items():
        p=Path(relative)
        if p.is_absolute() or '..' in p.parts or p.suffix!='.py':raise ValueError('invalid source pin path')
        if hashlib.sha256((repo/p).read_bytes()).hexdigest()!=expected:raise ValueError('runtime source differs; use the archived source snapshot: '+relative)
    return pins


def save_archive(root,tasks,episodes=()):
    tasks=[CertifiedTask.model_validate(t.model_dump(mode='json') if isinstance(t,CertifiedTask) else t) for t in tasks]
    episodes=[EpisodeExecution.model_validate(e.model_dump(mode='json') if isinstance(e,EpisodeExecution) else e) for e in episodes]
    if not tasks or len({t.task_hash for t in tasks})!=len(tasks):raise ValueError('archive requires distinct certified task hashes')
    root=Path(root)
    if any(t.frozen.spec.partition=='heldout' for t in tasks):
        try:root.resolve().relative_to(Path(__file__).resolve().parents[1])
        except ValueError:pass
        else:raise ValueError('private Held-out artifacts must remain outside public repository')
    grouped={t.task_hash:[] for t in tasks}
    if len({e.execution_hash for e in episodes})!=len(episodes):raise ValueError('duplicate episode journal')
    for e in episodes:
        if e.config.task.task_hash not in grouped:raise ValueError('episode task is not registered in archive')
        grouped[e.config.task.task_hash].append(e)
    root.mkdir(parents=True,exist_ok=False);snapshot_sources(root);entries=[];kinds=Counter()
    # Write each actual validated record immediately. A failure retains earlier
    # files/source pins; it never replaces an older archive with a successful retry.
    for i,t in enumerate(tasks):
        verify_task(t);actual=[]
        for e in grouped[t.task_hash]:
            if e.config.task!=t:raise ValueError('episode differs from exact private task proof')
            replay(e);actual.append(e.model_dump(mode='json'));kinds[e.config.evidence_kind]+=1
        raw={'task':t.model_dump(mode='json'),'episodes':actual};data=json.dumps(raw,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
        compressed=gzip.compress(data,mtime=0);name='case-'+str(i)+'.json.gz';(root/name).write_bytes(compressed)
        entries.append({'path':name,'content_hash':digest(raw),'compressed_sha256':hashlib.sha256(compressed).hexdigest(),
            'task_hash':t.task_hash,'record_hash':t.record_hash,'domain':t.frozen.spec.domain,'world_hash':t.frozen.world_hash,'episodes':len(actual)})
    pins=json.loads((root/'source-pins.json').read_text())
    manifest={'protocol':'eight-domain-task-archive-v1','cases':entries,'tasks':len(tasks),'episodes':len(episodes),'evidence_kinds':dict(kinds),
        'source_pins_hash':digest(pins),'model_experiments':'NOT RUN' if not kinds['real_model'] else 'DECLARED_REAL_MODEL_EPISODES',
        'scope':'Native proof and exact episode journals. Distinct hashes are not a semantic-diversity or release-quota audit.'}
    manifest['manifest_hash']=digest(manifest);atomic_json(root/'manifest.json',manifest,immutable=True)
    return manifest


def verify_archive(root):
    root=Path(root);pins=verify_sources(root);m=json.loads((root/'manifest.json').read_text())
    if m.get('protocol')!='eight-domain-task-archive-v1' or m.get('manifest_hash')!=digest({k:v for k,v in m.items() if k!='manifest_hash'}) or m.get('source_pins_hash')!=digest(pins):raise ValueError('task archive/source manifest integrity differs')
    tasks=set();episodes=set();kinds=Counter();domains=Counter()
    for entry in m['cases']:
        p=Path(entry['path'])
        if p.name!=str(p) or p.suffix!='.gz':raise ValueError('archive case path escapes directory')
        content=(root/p).read_bytes()
        if hashlib.sha256(content).hexdigest()!=entry['compressed_sha256']:raise ValueError('compressed task case integrity differs')
        raw=json.loads(gzip.decompress(content))
        if digest(raw)!=entry['content_hash']:raise ValueError('task case content differs')
        task=CertifiedTask.model_validate(raw['task']);verify_task(task)
        if task.task_hash!=entry['task_hash'] or task.record_hash!=entry['record_hash'] or task.frozen.world_hash!=entry['world_hash'] or task.frozen.spec.domain!=entry['domain'] or task.task_hash in tasks:raise ValueError('task archive index differs')
        tasks.add(task.task_hash);domains[task.frozen.spec.domain]+=1
        if len(raw['episodes'])!=entry['episodes']:raise ValueError('episode count differs')
        for value in raw['episodes']:
            e=EpisodeExecution.model_validate(value)
            if e.config.task!=task or e.execution_hash in episodes:raise ValueError('episode scope or uniqueness differs')
            replay(e);episodes.add(e.execution_hash);kinds[e.config.evidence_kind]+=1
    if len(tasks)!=m['tasks'] or len(episodes)!=m['episodes'] or dict(kinds)!=m['evidence_kinds']:raise ValueError('archive quantities differ')
    expected='NOT RUN' if not kinds['real_model'] else 'DECLARED_REAL_MODEL_EPISODES'
    if m['model_experiments']!=expected:raise ValueError('native/scripted artifacts cannot claim real-model experiments')
    return {'verified':True,'tasks':len(tasks),'episodes':len(episodes),'domains':dict(domains),'evidence_kinds':dict(kinds),'model_experiments':expected}
