"""Explicit independent CloudOps CLI. SaaS retains python -m usami unchanged."""
import argparse
import json
from pathlib import Path
import sys
from usami.service.client import OpenAICompatibleClient, settings_from_env
from usami.interaction.__main__ import provider_environment
from .models import World
from .generator import generate_world
from .engine import Trace,replay_trace
from .oracle import evaluate
from .benchmark import Run,load,validate,write_json,run_instance,score
from .release import build_release
from .suite import run_suite
from .baselines import ProofClient,OneStepClient,StopCustomer


def main(argv=None):
    p=argparse.ArgumentParser(description='Usami CloudOps v0.1 independent domain; SaaS: python -m usami')
    sub=p.add_subparsers(dest='command',required=True)
    g=sub.add_parser('generate');g.add_argument('--seed',type=int,default=42);g.add_argument('--out',type=Path,required=True)
    i=sub.add_parser('inspect');i.add_argument('world',type=Path)
    f=sub.add_parser('freeze');f.add_argument('--out',type=Path,required=True)
    v=sub.add_parser('validate');v.add_argument('--manifest',type=Path,required=True)
    profile=sub.add_parser('profile');profile.add_argument('--manifest',type=Path,required=True);profile.add_argument('--out',type=Path,required=True);profile.add_argument('--strategy',default='')
    for cmd in ('evaluate','replay'):
        r=sub.add_parser(cmd);r.add_argument('evidence',type=Path);r.add_argument('--manifest',type=Path);r.add_argument('--out',type=Path)
    r=sub.add_parser('run');r.add_argument('--manifest',type=Path,required=True);r.add_argument('--instance');r.add_argument('--trial',default='r0');r.add_argument('--out',type=Path,required=True)
    s=sub.add_parser('run-suite');s.add_argument('--manifest',type=Path,required=True);s.add_argument('--out',type=Path,required=True);s.add_argument('--max-runs',type=int);s.add_argument('--baseline',choices=['reference-proof','one-step'])
    args=p.parse_args(argv)
    try:
        if args.command=='generate':
            world=generate_world(args.seed);write_json(args.out,world);result={'domain':'cloudops','state_hash':world.state_hash,'out':str(args.out)}
        elif args.command=='inspect':
            world=World.model_validate_json(args.world.read_text(encoding='utf-8'));from .rules import invariants
            result={'world':world.model_dump(mode='json'),'invariants':invariants(world),'state_hash':world.state_hash}
        elif args.command=='freeze':result=build_release(args.out)
        elif args.command=='validate':
            m=load(args.manifest);result={'valid':True,'instances':len(m.instances),'manifest_hash':m.manifest_hash}
        elif args.command=='profile':
            if args.out.exists():raise ValueError('profile already exists; choose a new path')
            m=load(args.manifest);m.service_model=settings_from_env(provider_environment('service'))
            if m.track=='customer_discovery':m.customer_model=settings_from_env(provider_environment('customer'));m.strategy=args.strategy
            elif args.strategy:raise ValueError('strategy is for open Discovery only')
            m.seal();validate(m);write_json(args.out,m);result={'manifest_hash':m.manifest_hash,'dataset_hash':m.dataset_hash}
        elif args.command in {'evaluate','replay'}:
            data=json.loads(args.evidence.read_text(encoding='utf-8'))
            if 'manifest_hash' in data:
                if not args.manifest:raise ValueError('external frozen Manifest required for run evidence')
                result=score(load(args.manifest),Run.model_validate(data))
            else:
                trace=Trace.model_validate(data)
                if args.command=='replay':replay_trace(trace)
                result=evaluate(trace)
            if args.out:write_json(args.out,result)
        else:
            m=load(args.manifest)
            def clients(manifest,instance):
                if getattr(args,'baseline',None):
                    if manifest.track!='service_task':raise ValueError('proof baselines are Service software checks only')
                    cls=ProofClient if args.baseline=='reference-proof' else OneStepClient
                    return (cls(manifest.service_model,instance.task) if cls is ProofClient else cls(manifest.service_model,instance.world,instance.task)),None
                service=OpenAICompatibleClient.from_env(provider_environment('service'))
                try:return service,OpenAICompatibleClient.from_env(provider_environment('customer')) if manifest.track=='customer_discovery' else None
                except Exception:service.close();raise
            if args.command=='run-suite':result=run_suite(m,args.out,clients,args.max_runs)
            else:
                if args.out.exists():raise ValueError('run evidence already exists; no overwrite')
                instance=next((i for i in m.instances if i.id==args.instance),m.instances[0] if args.instance is None else None)
                if instance is None:raise ValueError('unknown frozen instance')
                service,customer=clients(m,instance)
                try:run=run_instance(m,instance,service,customer,args.trial)
                finally:
                    service.close()
                    if customer:customer.close()
                write_json(args.out,run);result=score(m,run)
        print(json.dumps(result,ensure_ascii=False,indent=2));return 0
    except (ValueError,OSError) as exc:print('usami-cloudops: '+str(exc),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
