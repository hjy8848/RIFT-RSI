"""Explicit new-protocol run, offline replay and frozen-snapshot evaluation."""
import argparse
import json
from pathlib import Path
from vast_coevo.designer.storage import atomic_json, digest
from vast_coevo.skills.library import empty_snapshot, LibrarySnapshot, LibraryStore
from vast_coevo.models import EpisodeEvidence, ModelSpec
from integrations.usami.adapter import UsamiAdapter, NativeArtifact, source_hashes
from integrations.usami.domains import UsamiProfile
from .controller import DesignerPlan, DesignerServiceController, Ledger
from .backend import SearchBackend, Trial


def native_factory(model, role, phase):
    from usami.service.client import OpenAICompatibleClient, key_from_env
    from integrations.usami.domains import model_settings
    return OpenAICompatibleClient(model_settings(model),key_from_env())


def replay_run(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    frozen=manifest['plan']['protocol']=='frozen-task-discovery-v1'
    if frozen:
        from integrations.usami.frozen.controller import FrozenPlan,preparation,archived_plan
        from integrations.usami.frozen.pool import WorldPool
        from integrations.usami.frozen.backend import FrozenSearchBackend
        from integrations.usami.frozen.composer import FrozenTask
    plan=archived_plan(manifest['plan']) if frozen else DesignerPlan.model_validate(manifest['plan'])
    if manifest['source_pins']!=source_hashes():raise ValueError('original source snapshot required')
    artifacts=[NativeArtifact.model_validate_json(p.read_text()) for p in (root/'native'/'episodes').glob('*/native.json')]
    profiles={p.config_id:p for p in plan.profiles}
    profiles.update({a.profile.config_id:a.profile for a in artifacts})
    adapter=UsamiAdapter(root/'native',tuple(profiles.values()),plan.tasks)
    store=LibraryStore(root/'libraries');initial=manifest['initial']
    customer=LibrarySnapshot.model_validate(initial['customer'])
    ledger=Ledger(root/'offline',plan.max_llm_calls,plan.max_tool_calls,lambda *_: (_ for _ in ()).throw(ValueError('Replay must not call a model')))
    backend=(FrozenSearchBackend if frozen else SearchBackend)(root,adapter,customer,plan.customer_model,plan.service_model,plan.designer_model,plan.budget,ledger.client,ledger)
    if frozen:
        pool=WorldPool(plan.world_specs);data=preparation(plan.world_specs,plan.budget);pool.verify_saved(root)
        if json.loads((root/'world-pool-audit.json').read_text())!=data['audit']:raise ValueError('world pool coverage audit differs')
        backend.configure(pool,[FrozenTask.model_validate(s).checked() for s in data['catalog']],plan.max_prompt_worlds)
    checked=[]
    for p in (root/'native'/'episodes').glob('*/evidence.json'):
        evidence=EpisodeEvidence.model_validate_json(p.read_text());result=adapter.replay(evidence,store.load(evidence.customer_snapshot_hash),store.load(evidence.service_snapshot_hash))
        if not result.verified:raise ValueError('native replay failed: '+result.reason)
        checked.append(evidence.evidence_hash)
    trials=[]
    for p in (root/'trials').glob('*/trial.json'):
        trial=Trial.model_validate_json(p.read_text()).checked()
        backend.replay(trial,store.load(trial.designer_hash),store.load(trial.service_hash),root/'offline'/'replay'/trial.trial_hash)
        frozen=p.with_name('frozen.json')
        if trial.scene is not None and (not frozen.exists() or json.loads(frozen.read_text())!=trial.scene):raise ValueError('pre-execution scene freeze differs')
        trials.append(trial.trial_hash)
    report=json.loads((root/'report.json').read_text())
    if sorted(report['trials'])!=sorted(trials):raise ValueError('report omitted or substituted trials')
    from .replay import verify_selections
    verify_selections(root,manifest,report,adapter,store)
    result={'verified':True,'native_episodes':len(checked),'designer_trials':len(trials),'source_pins':True,'selection_replay':True}
    atomic_json(root/'replay.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','replay','evaluate'])
    parser.add_argument('--root',required=True);parser.add_argument('--plan')
    parser.add_argument('--designer-library');parser.add_argument('--customer-library');parser.add_argument('--service-library')
    parser.add_argument('--models-from-env',action='store_true',help='resolve model settings once, before freezing the manifest')
    parser.add_argument('--strategy',choices=['random_legal','error_only','correct_only','bidirectional','bidirectional_fixed_designer'])
    parser.add_argument('--source-run',help='completed run whose immutable libraries are evaluated')
    parser.add_argument('--designer-hash');parser.add_argument('--service-hash')
    args=parser.parse_args(argv)
    if args.command=='replay':result=replay_run(args.root)
    else:
        if not args.plan:parser.error('--plan required')
        raw=json.loads(Path(args.plan).read_text())
        frozen=raw.get('protocol')=='frozen-task-discovery-v1'
        if frozen:
            from integrations.usami.frozen.controller import FrozenPlan,FrozenController
        plan_type=FrozenPlan if frozen else DesignerPlan
        plan=plan_type.model_validate(raw)
        if args.models_from_env:
            from integrations.usami.cli import model_from_env
            data=plan.model_dump(mode='json')
            for role in ('designer','customer','service'):
                data[role+'_model']=model_from_env(role).model_dump(mode='json')
            data['designer_reflector_model']=model_from_env('designer_reflector',fallback=ModelSpec.model_validate(data['designer_model'])).model_dump(mode='json')
            data['service_reflector_model']=model_from_env('service_reflector',fallback=ModelSpec.model_validate(data['service_model'])).model_dump(mode='json')
            plan=plan_type.model_validate(data)
        if args.strategy:plan=plan_type.model_validate({**plan.model_dump(mode='json'),'strategy':args.strategy})
        load=lambda path,role:LibrarySnapshot.model_validate_json(Path(path).read_text()) if path else empty_snapshot(role)
        if args.command=='run':
            result=(FrozenController if frozen else DesignerServiceController)(args.root,plan,load(args.designer_library,'designer'),load(args.customer_library,'customer'),load(args.service_library,'service'),native_factory).run()
        else:
            from .evaluation import evaluate
            if not args.source_run or not args.designer_hash or not args.service_hash:parser.error('evaluation requires source run and exact D/S hashes')
            result=evaluate(args.root,plan,args.source_run,args.designer_hash,args.service_hash,native_factory)
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result.get('status','completed')=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
