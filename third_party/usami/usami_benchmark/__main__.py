"""Native eight-domain proof, offline Replay and explicitly opted-in model CLI."""
import argparse,json,sys
from pathlib import Path
from usami_harness.models import Budget
from vast_coevo.schema import atomic_json
from .contracts import WorldSpec
from .worlds import freeze
from .tasks import TaskSelection,ConstructionBudget,CertifiedTask,compile_task,verify_task


def _read(path):
    from usami.service.models import strict_json
    return strict_json(Path(path).read_text())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    prove=sub.add_parser('prove',help='Compile a selection against independently generated frozen bytes; no model calls.')
    prove.add_argument('request');prove.add_argument('--output',required=True)
    check=sub.add_parser('verify-task');check.add_argument('path')
    check=sub.add_parser('replay');check.add_argument('path')
    check=sub.add_parser('verify-archive');check.add_argument('path')
    audit=sub.add_parser('audit-tasks',help='Rebuild native certificates and audit coverage/split grouping without model calls.')
    audit.add_argument('path',help='Source-pinned eight-domain task archive')
    audit.add_argument('--output',required=True,help='New immutable audit directory')
    audit.add_argument('--per-domain-target',type=int)
    dataset=sub.add_parser('build-dataset',help='Freeze a finite world/candidate plan and certify native tasks; no model calls.')
    dataset.add_argument('path');dataset.add_argument('--output',required=True)
    dataset=sub.add_parser('verify-dataset');dataset.add_argument('path')
    execute=sub.add_parser('run',help='Real model execution only with explicit --online and total call ceiling.')
    execute.add_argument('path');execute.add_argument('--output',required=True);execute.add_argument('--online',action='store_true')
    execute.add_argument('--max-llm-calls',type=int,required=True);execute.add_argument('--customer-calls',type=int,default=8);execute.add_argument('--service-calls',type=int,default=16)
    execute.add_argument('--tool-calls',type=int,default=32);execute.add_argument('--turns',type=int,default=8);execute.add_argument('--mode',choices=('sandbox','reference'),default='sandbox')
    args=parser.parse_args(argv)
    if args.command=='prove':
        raw=_read(args.request)
        if not isinstance(raw,dict) or set(raw)-{'world_spec','selection','construction_budget'}:raise ValueError('request accepts only world_spec, selection and construction_budget')
        frozen=freeze(WorldSpec.model_validate(raw['world_spec']));selection=TaskSelection.model_validate(raw['selection'])
        task=compile_task(frozen,selection,ConstructionBudget.model_validate(raw.get('construction_budget',{})))
        target=Path(args.output)
        if frozen.spec.partition=='heldout':
            try:target.resolve().relative_to(Path(__file__).resolve().parents[1])
            except ValueError:pass
            else:raise ValueError('private Held-out task must be outside public repository')
        atomic_json(target,task.model_dump(mode='json'),immutable=True)
        print(json.dumps({'certified':True,'domain':frozen.spec.domain,'task_hash':task.task_hash,'model_experiments':'NOT RUN'}));return 0
    if args.command=='verify-task':print(json.dumps({'verified':verify_task(_read(args.path)),'model_calls':0}));return 0
    if args.command=='verify-archive':
        from .artifacts import verify_archive
        print(json.dumps(verify_archive(args.path),sort_keys=True));return 0
    if args.command=='audit-tasks':
        from .task_audit import audit_archive
        print(json.dumps(audit_archive(args.path,args.output,per_domain_target=args.per_domain_target),sort_keys=True));return 0
    if args.command in {'build-dataset','verify-dataset'}:
        from .datasets import build,verify_dataset
        result=build(_read(args.path),args.output) if args.command=='build-dataset' else verify_dataset(args.path)
        print(json.dumps(result,sort_keys=True));return 0
    if args.command=='replay':
        from .episodes import replay
        print(json.dumps(replay(_read(args.path)),sort_keys=True));return 0
    if not args.online:parser.error('run requires --online; no model client or key is read without this explicit opt-in')
    record=CertifiedTask.model_validate(_read(args.path));verify_task(record)
    customer_calls=args.customer_calls if record.track=='B' else 0
    if args.max_llm_calls<=0 or customer_calls+args.service_calls>args.max_llm_calls:parser.error('role call maxima exceed the explicitly supplied total call ceiling')
    budget=Budget(max_customer_turns=args.turns,max_customer_llm_calls=customer_calls,max_service_llm_calls=args.service_calls,max_service_tool_calls=args.tool_calls)
    root=Path(args.output)
    if record.frozen.spec.partition=='heldout':
        try:root.resolve().relative_to(Path(__file__).resolve().parents[1])
        except ValueError:pass
        else:raise ValueError('private Held-out journal must be outside public repository')
    from usami_benchmark.audit import snapshot_sources
    root.mkdir(parents=True,exist_ok=False);snapshot_sources(root)
    atomic_json(root/'registered-task.json',record.model_dump(mode='json'),immutable=True)
    atomic_json(root/'budget.json',{'total_call_ceiling':args.max_llm_calls,'native_budget':budget.model_dump(mode='json')},immutable=True)
    try:
        from usami.service.client import OpenAICompatibleClient,settings_from_env,key_from_env
        from .episodes import EpisodeConfig,run,replay
        settings=settings_from_env();key=key_from_env()
        customer=OpenAICompatibleClient(settings,key);service=OpenAICompatibleClient(settings,key)
        config=EpisodeConfig(task=record,customer_model=settings,service_model=settings,evidence_kind='real_model',mode=args.mode,budget=budget)
        atomic_json(root/'config.json',config.model_dump(mode='json'),immutable=True)
        result=run(config,customer,service)
        atomic_json(root/'journal.json',result.model_dump(mode='json'),immutable=True)
        verified=replay(result);atomic_json(root/'replay.json',verified,immutable=True)
        print(json.dumps({'output':str(root),'metrics':result.metrics,'cost':result.cost,'replay':verified},sort_keys=True))
        return 0
    except Exception as exc:
        # Native runtimes retain provider/model errors in their actual journals.
        # Startup/integrity failures are saved separately; never silently retried.
        atomic_json(root/'failure.json',{'stage':'startup_or_verification','type':type(exc).__name__,'message':str(exc),'retry':'not attempted'},immutable=True)
        raise


if __name__=='__main__':raise SystemExit(main())
