"""Source-pinned offline native examples; these are not model experiments."""
import gzip,hashlib,importlib,json
from pathlib import Path
from usami_harness.models import digest
from usami_benchmark.audit import snapshot_sources
from usami_benchmark.contracts import WorldSpec
from usami_benchmark.worlds import FrozenWorld,freeze
from vast_coevo.schema import atomic_json
from .engine import ExecutionTrace
from .tasks import TaskCertificate,verify_certificate

NATIVE_DOMAINS=('aftersales','airline','property','education')


def adapter_for(domain):
    if domain not in NATIVE_DOMAINS:raise ValueError('native dialogue artifact domain is not executable: '+domain)
    return importlib.import_module('usami_'+domain+'.adapter')


def save_examples(root,domain,rows):
    """rows contain independently frozen worlds and actual certified traces."""
    root=Path(root);root.mkdir(parents=True,exist_ok=False);adapter=adapter_for(domain)
    snapshot_sources(root);pins=json.loads((root/'source-pins.json').read_text());entries=[]
    for index,row in enumerate(rows):
        frozen=FrozenWorld.model_validate(row['frozen'].model_dump(mode='json'));trace=row['trace'];certificate=row['certificate']
        world=adapter.World.model_validate(frozen.world)
        if freeze(frozen.spec)!=frozen:raise ValueError('example world differs from independent registered generation')
        verify_certificate(adapter,world,certificate,trace)
        case={'frozen':frozen.model_dump(mode='json'),'trace':trace.model_dump(mode='json'),'certificate':certificate.model_dump(mode='json')}
        payload=json.dumps(case,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode();name='case-'+str(index)+'.json.gz';compressed=gzip.compress(payload,mtime=0)
        (root/name).write_bytes(compressed)
        entries.append({'path':name,'content_hash':digest(case),'compressed_sha256':hashlib.sha256(compressed).hexdigest(),
            'task_hash':certificate.task.task_hash,'trace_hash':trace.trace_hash,'family':certificate.task.family,'world_hash':frozen.world_hash})
    if not entries:raise ValueError('cannot publish an empty native evidence example')
    manifest={'protocol':'native-dialogue-example-v1','domain':domain,'evidence_kind':'native_reference_witness',
        'model_experiments':'NOT RUN','model_calls':0,'source_pins_hash':digest(pins),'cases':entries}
    manifest['manifest_hash']=digest(manifest);atomic_json(root/'manifest.json',manifest,immutable=True)
    return manifest


def verify_examples(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text());pins=json.loads((root/'source-pins.json').read_text())
    if manifest.get('protocol')!='native-dialogue-example-v1' or manifest['manifest_hash']!=digest({k:v for k,v in manifest.items() if k!='manifest_hash'}):raise ValueError('native manifest integrity differs')
    if manifest.get('evidence_kind')!='native_reference_witness' or manifest.get('model_experiments')!='NOT RUN' or manifest.get('model_calls')!=0:raise ValueError('native proofs cannot claim real model execution')
    if manifest['source_pins_hash']!=digest(pins) or not pins:raise ValueError('native source manifest differs')
    repo=Path(__file__).resolve().parents[1]
    packages=[p for p in repo.glob('usami*') if p.is_dir()]+[repo/'integrations',repo/'vast_coevo']
    expected_paths={str(p.relative_to(repo)) for folder in packages for p in folder.rglob('*.py')}
    if set(pins)!=expected_paths:raise ValueError('runtime source file set differs from archived example')
    for relative,expected in pins.items():
        name=Path(relative)
        if name.is_absolute() or '..' in name.parts or name.suffix!='.py':raise ValueError('source pin path is not repository relative Python')
        path=repo/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('current source differs from example: '+relative+'; use its source snapshot')
    adapter=adapter_for(manifest['domain']);verified=[]
    if not manifest['cases']:raise ValueError('native example has no cases')
    for entry in manifest['cases']:
        relative=Path(entry['path'])
        if relative.name!=str(relative) or relative.suffix!='.gz':raise ValueError('case path escapes its registered directory')
        compressed=(root/relative).read_bytes()
        if hashlib.sha256(compressed).hexdigest()!=entry['compressed_sha256']:raise ValueError('compressed case hash differs')
        raw=json.loads(gzip.decompress(compressed));frozen=FrozenWorld.model_validate(raw['frozen']).checked()
        if digest(raw)!=entry['content_hash'] or frozen.world_hash!=entry['world_hash'] or freeze(frozen.spec)!=frozen:raise ValueError('registered native world or case content differs')
        trace=ExecutionTrace.model_validate(raw['trace']);certificate=TaskCertificate.model_validate(raw['certificate'])
        if trace.trace_hash!=entry['trace_hash'] or certificate.task.task_hash!=entry['task_hash'] or certificate.task.family!=entry['family']:raise ValueError('case index provenance differs')
        verify_certificate(adapter,adapter.World.model_validate(frozen.world),certificate,trace)
        verified.append({'family':certificate.task.family,'trace_hash':trace.trace_hash,'task_hash':certificate.task.task_hash})
    return {'verified':True,'domain':manifest['domain'],'cases':verified,'model_experiments':'NOT RUN'}
