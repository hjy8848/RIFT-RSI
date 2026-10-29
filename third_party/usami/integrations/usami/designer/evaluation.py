"""Post-training snapshot evaluation. No Reflection or publication methods used."""
import json
from pathlib import Path
from vast_coevo.skills.library import LibraryStore
from vast_coevo.designer.evaluation import summarize
from vast_coevo.designer.storage import atomic_json
from .controller import DesignerServiceController
from .cli import replay_run


def evaluate(root, plan, source_run, designer_hash, service_hash, factory):
    source=Path(source_run).resolve();root=Path(root).resolve()
    if root==source or source in root.parents:raise ValueError('evaluation requires a separate output directory')
    replay_run(source)
    report=json.loads((source/'report.json').read_text())
    if report['status']!='completed':raise ValueError('independent evaluation requires a completed source run')
    original=json.loads((source/'manifest.json').read_text());old_plan=type(plan).model_validate(original['plan'])
    if any(getattr(plan,k)!=getattr(old_plan,k) for k in ('designer_model','customer_model','service_model','budget')):
        raise ValueError('evaluation must freeze the source actor models and episode budget')
    old_seeds={p.seed for p in old_plan.profiles}
    if old_seeds & {p.seed for p in plan.profiles}:raise ValueError('evaluation must use independently frozen worlds')
    store=LibraryStore(source/'libraries');d=store.load(designer_hash);s=store.load(service_hash);c=store.load(report['fixed_customer'])
    # A deliberately separate output and no invocation of controller.run/update.
    controller_type=DesignerServiceController
    if plan.protocol=='frozen-task-discovery-v1':
        from integrations.usami.frozen.controller import FrozenController
        controller_type=FrozenController
    experiment=controller_type(root,plan,d,c,s,factory)
    root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/'evaluation-manifest.json',{'purpose':'development_diagnostic','plan':plan,'source_run':str(source),
        'designer_hash':designer_hash,'service_hash':service_hash,'customer_hash':c.snapshot_hash,'reflection':False},immutable=True)
    for snapshot in (d,c,s):experiment.store.save(snapshot)
    experiment.prepare_run()
    for ref in plan.initial_tasks:
        case=experiment.task_case(ref)
        evidence=experiment.adapter.run_service_task(case.config,ref,c,s,'evaluation.seed.'+ref)
        experiment.backend.register_source(evidence);experiment.bank.add_anchor(experiment.adapter,evidence)
    pool=tuple(experiment.bank.anchors.values());rows=[]
    for index in range(plan.trials_per_generation):
        rows.append(experiment.backend.trial(d,s,pool,experiment.bank,'evaluation.'+str(index),plan.strategy,plan.random_seed+index))
    result={'status':'completed','purpose':'development_diagnostic','reflection_calls':0,'selection_changes':0,
            'designer_hash':d.snapshot_hash,'service_hash':s.snapshot_hash,'customer_hash':c.snapshot_hash,
            'metrics':summarize([t.discovery for t in rows],experiment.backend.costs(rows)),'trials':[t.trial_hash for t in rows]}
    from integrations.usami.costs import _sum
    from types import SimpleNamespace
    result['cost']=_sum([SimpleNamespace(response=r['response']) for r in experiment.ledger.logs],experiment.ledger.tools)
    for t in rows:experiment.backend.replay(t,d,s,root/'offline'/t.trial_hash)
    result['replay_verified']=True
    atomic_json(root/'evaluation-report.json',result,immutable=True);return result
