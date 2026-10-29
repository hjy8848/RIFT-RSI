"""Offline native certification and an explicit opt-in real-model episode.

Examples:
  python -m usami_aftersales inspect --seed 42
  python -m usami_aftersales certify --seed 42 --profile profile_u0_order0_line0 --output NEW_DIRECTORY
  python -m usami_aftersales verify --bundle NEW_DIRECTORY/bundle.json
  python -m usami_aftersales run --bundle NEW_DIRECTORY/bundle.json --online --output ANOTHER_NEW_DIRECTORY
"""
import argparse,hashlib,json,os
from pathlib import Path
from vast_coevo.schema import atomic_json
from usami_harness.models import Budget,digest
from usami_benchmark.audit import snapshot_sources
from usami_dialogue.tasks import DialogueTask,score,verify_certificate
from usami_dialogue.engine import ExecutionTrace
from . import adapter
from .tasks import certified_tasks


def _new_root(path):
    root=Path(path)
    root.mkdir(parents=True,exist_ok=False)
    snapshot_sources(root)
    return root


def _load(path):
    path=Path(path);raw=json.loads(path.read_text())
    if raw['bundle_hash']!=digest({k:v for k,v in raw.items() if k!='bundle_hash'}):raise ValueError('bundle content hash differs')
    repo=Path(__file__).resolve().parents[1]
    for name,expected in raw['source_pins'].items():
        target=repo/name
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=expected:raise ValueError('source differs from archived bundle: '+name+'; use source-snapshot.zip')
    world=adapter.World.model_validate(raw['world']);task=DialogueTask.model_validate(raw['task'])
    from usami_dialogue.tasks import checked_task
    checked_task(adapter,world,task)
    return raw,world,task


def _env(role):
    env=dict(os.environ)
    prefix='USAMI_'+role.upper()+'_LLM_'
    for key,value in os.environ.items():
        if key.startswith(prefix):env['USAMI_LLM_'+key[len(prefix):]]=value
    return env


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    inspect=sub.add_parser('inspect');inspect.add_argument('--seed',type=int,default=42)
    certificate=sub.add_parser('certify');certificate.add_argument('--seed',type=int,default=42);certificate.add_argument('--profile',required=True)
    certificate.add_argument('--output',required=True);certificate.add_argument('--mixed',action='store_true');certificate.add_argument('--resolution',choices=['refund','exchange','reship'])
    certificate.add_argument('--max-search-nodes',type=int,default=1024);certificate.add_argument('--max-paths',type=int,default=2)
    verify=sub.add_parser('verify');verify.add_argument('--bundle',required=True)
    online=sub.add_parser('run');online.add_argument('--bundle',required=True);online.add_argument('--output',required=True);online.add_argument('--online',action='store_true')
    online.add_argument('--mode',choices=['reference','sandbox'],default='sandbox')
    online.add_argument('--customer-turns',type=int,default=8);online.add_argument('--customer-calls',type=int,default=12)
    online.add_argument('--service-calls',type=int,default=32);online.add_argument('--tool-calls',type=int,default=64)
    args=parser.parse_args(argv)
    if args.command=='inspect':
        world=adapter.generate_world(args.seed)
        print(json.dumps({'seed':args.seed,'world_hash':world.state_hash,'profiles':{k:{'identity':p.identity,'order_id':p.facts['order_id'],'line_id':p.facts['line_id']} for k,p in world.profiles.items()}},indent=2));return 0
    if args.command=='certify':
        if args.max_search_nodes<1 or args.max_paths<1:raise ValueError('search bounds must be positive')
        world=adapter.generate_world(args.seed);root=_new_root(args.output)
        task,result,rows=certified_tasks(world,args.profile,resolutions=[args.resolution] if args.resolution else None,mixed=args.mixed,max_nodes=args.max_search_nodes,max_paths=args.max_paths)
        payload={'protocol':'aftersales-certified-bundle-v1','world':world.model_dump(mode='json'),'task':task.model_dump(mode='json'),
            'source_pins':json.loads((root/'source-pins.json').read_text()),'search':result,
            'witnesses':[{'trace':r['trace'].model_dump(mode='json'),'certificate':r['certificate'].model_dump(mode='json')} for r in rows],
            'model_experiment':'NOT RUN','status':'CERTIFIED' if rows else 'NO_WITNESS_WITHIN_BOUNDS'}
        payload['bundle_hash']=digest(payload);atomic_json(root/'bundle.json',payload,immutable=True)
        print(json.dumps({'status':payload['status'],'world_hash':world.state_hash,'certificates':len(rows),'search_nodes':result['nodes'],'output':str(root)}));return 0 if rows else 2
    raw,world,task=_load(args.bundle)
    if args.command=='verify':
        if not raw['witnesses']:raise ValueError('bundle has no certificate')
        for row in raw['witnesses']:verify_certificate(adapter,world,row['certificate'],ExecutionTrace.model_validate(row['trace']))
        print(json.dumps({'verified':True,'certificates':len(raw['witnesses']),'world_hash':world.state_hash}));return 0
    if not args.online:raise ValueError('real model calls require explicit --online; offline certification never calls a model')
    from usami.service.client import OpenAICompatibleClient
    from usami_dialogue.agents import run,replay,TaskRunConfig
    budget=Budget(max_customer_turns=args.customer_turns,max_customer_llm_calls=args.customer_calls,max_service_llm_calls=args.service_calls,max_service_tool_calls=args.tool_calls)
    if not raw['witnesses']:raise ValueError('online task execution requires a verified private witness')
    config=TaskRunConfig(task=task,certificate=raw['witnesses'][0]['certificate'],witness=raw['witnesses'][0]['trace'],profile_id=task.profile_id,initial_world_hash=world.state_hash,mode=args.mode,tools=list(adapter.TOOLS),budget=budget,external_schedule=task.external_schedule)
    root=_new_root(args.output)
    atomic_json(root/'registration.json',{'task':task.model_dump(mode='json'),'config':config.model_dump(mode='json'),'world':world.model_dump(mode='json'),'source_bundle_hash':raw['bundle_hash']},immutable=True)
    c=OpenAICompatibleClient.from_env(_env('customer'));s=None
    try:
        s=OpenAICompatibleClient.from_env(_env('service'))
        execution=run(adapter,world,config,c,s)
    finally:
        c.close()
        if s is not None:s.close()
    atomic_json(root/'execution.json',execution.model_dump(mode='json'),immutable=True)
    result={'score':score(adapter,world,task,execution.trace),'replay':replay(adapter,execution,world=world,config=config),'cost':execution.cost,'status':execution.status}
    atomic_json(root/'result.json',result,immutable=True);print(json.dumps(result,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
