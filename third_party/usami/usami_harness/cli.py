"""Small explicit-domain CLI; replay never initializes a live provider."""
import argparse
import json
import sys
from pathlib import Path
from .benchmark import load, write_json, replay_run, Run
from .suite import run_suite
from .provider import client,model_settings


def main(adapter,build_release,argv=None):
    parser=argparse.ArgumentParser(description=adapter.name+' executable business domain')
    sub=parser.add_subparsers(dest='command',required=True)
    generate=sub.add_parser('world');generate.add_argument('--seed',type=int,default=42);generate.add_argument('--out',type=Path)
    build=sub.add_parser('build');build.add_argument('--out',type=Path,required=True)
    verify=sub.add_parser('verify');verify.add_argument('manifest',type=Path)
    run=sub.add_parser('run');run.add_argument('manifest',type=Path);run.add_argument('--out',type=Path,required=True);run.add_argument('--max-runs',type=int)
    freeze=sub.add_parser('profile');freeze.add_argument('manifest',type=Path);freeze.add_argument('--out',type=Path,required=True)
    replay=sub.add_parser('replay');replay.add_argument('manifest',type=Path);replay.add_argument('run',type=Path)
    args=parser.parse_args(argv)
    try:
        if args.command=='world':
            world=adapter.generate_world(args.seed)
            if args.out:write_json(args.out,world)
            print(json.dumps({'domain':adapter.name,'seed':args.seed,'state_hash':world.state_hash,
                              'invariants':adapter.invariants(world),'out':str(args.out) if args.out else None},indent=2))
        elif args.command=='build':
            print(json.dumps(build_release(args.out),indent=2))
        elif args.command=='verify':
            manifest=load(adapter,args.manifest)
            print(json.dumps({'verified':True,'manifest_hash':manifest.manifest_hash,'instances':len(manifest.instances)}))
        elif args.command=='profile':
            if args.out.exists():raise ValueError('profile output exists; choose a new preregistration path')
            manifest=load(adapter,args.manifest)
            manifest.service_model=model_settings('service')
            if manifest.track=='customer_discovery':manifest.customer_model=model_settings('customer')
            manifest.suite_id+='-live';manifest.seal();write_json(args.out,manifest)
            print(json.dumps({'preregistered':True,'path':str(args.out),'manifest_hash':manifest.manifest_hash}))
        elif args.command=='run':
            manifest=load(adapter,args.manifest)
            def clients(manifest,instance):
                service=client('service')
                try:return service,client('customer') if manifest.track=='customer_discovery' else None
                except BaseException:service.close();raise
            print(json.dumps(run_suite(adapter,manifest,args.out,clients,args.max_runs),indent=2))
        else:
            manifest=load(adapter,args.manifest)
            result=replay_run(adapter,manifest,Run.model_validate_json(args.run.read_text()))
            print(json.dumps({'replayed':True,'run_hash':result.run_hash}))
        return 0
    except (ValueError,OSError) as exc:
        print(f'{adapter.name}: {exc}',file=sys.stderr)
        return 2
