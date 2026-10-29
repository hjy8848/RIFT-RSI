"""Independent world audit; unimplemented task measurements remain explicitly NOT RUN."""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from usami_harness.models import digest
from vast_coevo.schema import atomic_json
from .registry import default_spec,generate,invariants
from .worlds import freeze,partition_for


def state_coverage(world):
    """Current public business states, latent facts and future events are separate.

    This is a descriptive field census, not evidence of tested Reference boundaries.
    IDs, text, raw timestamps and monetary jitter do not add state categories.
    """
    current=Counter();latent=Counter();future=Counter();past=Counter();applied=set(world.get('applied_events',[]))
    fields={'status','role','quality','housekeeping','maintenance','receiving_enabled',
        'condition','responsibility','on_call','skills','severity','kind','issue','return_required',
        'review_requested','cabin','changeable','refundable','verified','action','subject','subjects',
        'level','max_level','transferable','extension_reasons','no_show_consumes','origin','reason',
        'last_lesson_status','late_fee','auto_renew','closed','outcome'}
    for collection,values in world.items():
        if not isinstance(values,dict):continue
        for key,obj in values.items():
            if not isinstance(obj,dict):continue
            for field,value in obj.items():
                if collection=='external_events':
                    if field in {'kind','outcome','timing'}:(past if key in applied else future)[collection+'.'+field+'='+str(value)]+=1
                elif field.startswith('simulated_'):
                    if isinstance(value,(str,bool,int)) or value is None:latent[collection+'.'+field+'='+str(value)]+=1
                elif field in fields:
                    choices=sorted(set(value),key=str) if isinstance(value,list) else [value]
                    for choice in choices:
                        if isinstance(choice,(str,bool,int)) or choice is None:current[collection+'.'+field+'='+str(choice)]+=1
    return {'state_coverage':dict(sorted(current.items())),
        'latent_fact_coverage':dict(sorted(latent.items())),
        'frozen_external_event_coverage':dict(sorted(future.items())),
        'historical_external_event_coverage':dict(sorted(past.items()))}


def audit_domain(domain,seeds,root=None,*,generator_version=None,rules_version=None):
    if (generator_version is None)!=(rules_version is None):raise ValueError('generator/rules overrides must be paired')
    rows=[];worlds=[];states=Counter();latent=Counter();future=Counter();past=Counter();partitions={};invalid=[];exclusions=[];histories=[]
    for seed in seeds:
        try:
            spec=default_spec(domain,seed)
            if generator_version is not None:spec.generator_version=generator_version;spec.rules_version=rules_version
            row=freeze(spec);row.spec.partition=partition_for(row.structure_signature)
            for diagnostic in row.generation_diagnostics:
                if diagnostic.get('kind')=='native_dialogue_lifecycle_v2':
                    histories.extend(diagnostic['histories'])
                    exclusions.extend({'seed':seed,**d} for d in diagnostic['excluded_native_conflicts'])
                else:exclusions.append({'seed':seed,**diagnostic})
            world=row.world
            census=state_coverage(world)
            states.update(census['state_coverage']);latent.update(census['latent_fact_coverage']);future.update(census['frozen_external_event_coverage']);past.update(census['historical_external_event_coverage'])
            partitions.setdefault(row.spec.partition,set()).add(row.structure_signature)
            rows.append({'seed':seed,'spec_hash':row.spec.spec_hash,'registration_hash':row.registration_hash,'world_hash':row.world_hash,'structure':row.structure_signature,'permissions':row.permission_signature,'partition':row.spec.partition})
            worlds.append(row.model_dump(mode='json'))
        except (ValueError,TypeError) as exc:invalid.append({'seed':seed,'error':str(exc)})
    structures={r['structure'] for r in rows}
    overlaps={a+' / '+b:len(partitions[a]&partitions[b]) for a in partitions for b in partitions if a<b}
    report={'protocol':'eight-domain-world-audit-v3','domain':domain,'attempted':len(seeds),'valid_worlds':len(rows),'invalid':invalid,
        'generation_exclusions':exclusions,
        'generation_history':{'sessions':len(histories),'nonempty_sessions':sum(h['actual_business_operations']+h['external_events']>0 for h in histories),
            'actual_business_operations':sum(h['actual_business_operations'] for h in histories),'actual_external_events':sum(h['external_events'] for h in histories),
            'operation_distribution':dict(sorted(Counter(step['operation']['name'] for h in histories for step in h['steps'] if step['kind']=='business').items())),
            'scope':'Actual generator simulator histories verified by native Oracle/Replay, not model calls or task success.'},'unique_world_hashes':len({r['world_hash'] for r in rows}),'anonymous_structures':len(structures),'permission_topologies':len({r['permissions'] for r in rows}),
        'duplicate_structure_ratio':1-len(structures)/len(rows) if rows else None,'state_coverage':dict(sorted(states.items())),
        'latent_fact_coverage':dict(sorted(latent.items())),'frozen_external_event_coverage':dict(sorted(future.items())),'historical_external_event_coverage':dict(sorted(past.items())),
        'state_coverage_scope':'Descriptive category census; not executed Reference decision or task coverage. Future events and hidden simulator facts are separate.',
        'split_structure_overlaps':overlaps,'minimum_structural_goal_met':len(rows)>=100 and len(structures)>=30,
        'reference_decision_coverage':{'status':'NOT RUN','reason':'World generation audit is separate from the forthcoming certified task audit.'},
        'task_structure_coverage':{'status':'NOT RUN'},'causal_depth_distribution':{'status':'NOT RUN'},'rows':rows,
        'signature_caveat':'Conservative relational colour refinement; not an isomorphism proof. Numeric jitter and labels are excluded.',
        'model_experiments':'NOT RUN'}
    if root:
        root=Path(root);root.mkdir(parents=True,exist_ok=True)
        atomic_json(root/(domain+'-world-audit.json'),report,immutable=True)
        payload=json.dumps({'protocol':'eight-domain-world-pool-v1','worlds':worlds},sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
        path=root/(domain+'-worlds.json.gz')
        if path.exists():raise FileExistsError(path)
        path.write_bytes(gzip.compress(payload,mtime=0))
        atomic_json(root/(domain+'-worlds-integrity.json'),{'canonical_content_hash':digest(json.loads(payload)),'world_count':len(worlds)},immutable=True)
    return report


def snapshot_sources(root):
    import zipfile,hashlib
    repo=Path(__file__).resolve().parents[1]
    packages=[p for p in repo.glob('usami*') if p.is_dir()]+[repo/'integrations',repo/'vast_coevo']
    paths=sorted({p for folder in packages for p in folder.rglob('*.py')})
    pins={}
    with zipfile.ZipFile(root/'source-snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            relative=str(path.relative_to(repo));data=path.read_bytes();pins[relative]=hashlib.sha256(data).hexdigest()
            entry=zipfile.ZipInfo(relative,date_time=(2026,10,17,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(entry,data)
    atomic_json(root/'source-pins.json',pins,immutable=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--domains',nargs='+',default=['saas','cloudops','fulfillment','stayops'])
    parser.add_argument('--generator-version');parser.add_argument('--rules-version')
    parser.add_argument('--count',type=int,default=100);parser.add_argument('--start-seed',type=int,default=10000);parser.add_argument('--root',required=True)
    args=parser.parse_args(argv);root=Path(args.root)
    if (args.generator_version or args.rules_version) and len(args.domains)!=1:raise ValueError('explicit version overrides require exactly one domain')
    if root.exists():raise ValueError('audit output must be new')
    root.mkdir(parents=True)
    snapshot_sources(root)
    reports=[audit_domain(d,list(range(args.start_seed,args.start_seed+args.count)),root,generator_version=args.generator_version,rules_version=args.rules_version) for d in args.domains]
    result={r['domain']:{k:r[k] for k in ('attempted','valid_worlds','anonymous_structures','permission_topologies','minimum_structural_goal_met','invalid')} for r in reports}
    atomic_json(root/'summary.json',result,immutable=True);print(json.dumps(result,indent=2))
    return int(any(r['invalid'] for r in reports))


if __name__=='__main__':raise SystemExit(main())
