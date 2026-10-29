"""Recompute D selection and the unchanged S Evolver entirely from recorded calls."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from usami.service.models import LLMRecord
from vast_coevo.models import ModelCall, EpisodeEvidence
from vast_coevo.skills.library import LibraryStore, apply_patch
from vast_coevo.skills.models import SkillPatch
from vast_coevo.skills.validation import StaticSkillGate
from vast_coevo.reflection.reflector import ServiceReflector
from vast_coevo.evolution.evolvers import ServiceEvolver
from vast_coevo.evidence.archive import EvidenceArchive
from vast_coevo.designer.evaluation import compare
from vast_coevo.designer.reflection import reflect,OBJECTIVE
from integrations.usami.adapter import _recorded_wire_client
from integrations.usami.domains import model_settings
from .backend import Trial
from .compiler import Anchor, source_group
from .attribution import static_context, known_contracts
from .discovery import anchor_from
from .controller import DesignerPlan, DesignerServiceController, feedback_for


def recorded_client(call, model):
    if call is None:return SimpleNamespace(complete=lambda _: (_ for _ in ()).throw(ValueError('unexpected offline model call')))
    call=ModelCall.model_validate(call)
    record=LLMRecord(request=call.request,response=call.response,error={'stage':'transport',**call.error} if call.error else {})
    return _recorded_wire_client(model_settings(model),[record],[call])


def verify_selections(root, manifest, report, adapter, store):
    root=Path(root);frozen=manifest['plan']['protocol']=='frozen-task-discovery-v1'
    if frozen:
        from integrations.usami.frozen.controller import FrozenPlan,archived_plan,frozen_static_inputs,OBJECTIVE as FROZEN_OBJECTIVE
        from integrations.usami.frozen.pool import WorldPool
        from integrations.usami.frozen.discovery import frozen_anchor_from
    plan=archived_plan(manifest['plan']) if frozen else DesignerPlan.model_validate(manifest['plan'])
    pool=WorldPool(plan.world_specs) if frozen else None
    trials={t.trial_hash:t for t in (Trial.model_validate_json(p.read_text()).checked() for p in (root/'trials').glob('*/trial.json'))}
    evidence={e.episode_id:e for e in (EpisodeEvidence.model_validate_json(p.read_text()) for p in (root/'native'/'episodes').glob('*/evidence.json'))}
    initial=manifest['initial'];customer=store.load(initial['customer']['snapshot_hash'])
    designer=store.load(initial['designer']['snapshot_hash']);service=store.load(initial['service']['snapshot_hash'])
    backend=SimpleNamespace(costs=lambda rows: __import__('integrations.usami.designer.backend',fromlist=['SearchBackend']).SearchBackend.costs(None,rows))
    class OfflineAdapter:
        def __getattr__(self,name):return getattr(adapter,name)
        def run_episode(self,config,c,s,episode_id):return evidence[episode_id]
        def run_service_task(self,config,ref,c,s,episode_id):return evidence[episode_id]
        def run_failure_case(self,source,c,s,episode_id):return evidence[episode_id]
    offline=OfflineAdapter()
    # Reuse only config/case assembly from the controller, never its run/client methods.
    assembly=SimpleNamespace(plan=plan,adapter=adapter,task_case=lambda ref,purpose=None:
        DesignerServiceController.task_case(SimpleNamespace(plan=plan,adapter=adapter),ref,purpose))
    discovery,tasks=DesignerServiceController.validation(assembly)
    learned={};learned_discoveries=[]
    def learn(e):
        try:a=(frozen_anchor_from if frozen else anchor_from)(adapter,e)
        except ValueError:return
        learned[a.anchor_hash]=a
    for ref in plan.initial_tasks:learn(evidence['seed.'+ref])
    for row in report['generations']:
        g=row['generation'];saved=json.loads((root/f'g{g}.json').read_text())
        if saved!=row:raise ValueError('generation report differs from immutable checkpoint')
        if (row['parent_designer'],row['parent_service'],row['customer_hash'])!=(designer.snapshot_hash,service.snapshot_hash,customer.snapshot_hash):
            raise ValueError('cross-generation role/parent mismatch')
        training=[trials[h] for h in row['training']]
        for t in training:
            if t.evidence:learn(t.evidence);learned_discoveries.append(t.discovery)
        if any((t.designer_hash,t.customer_hash,t.service_hash)!=(designer.snapshot_hash,customer.snapshot_hash,service.snapshot_hash) for t in training):
            raise ValueError('generation executed an unselected actor')
        record=row['designer'];reflection=record['reflection']
        if reflection is not None:
            model=plan.designer_reflector_model or plan.designer_model
            replay=reflect(recorded_client(reflection['call'],model),model,designer,feedback_for(backend,training),f'g{g}',objective=FROZEN_OBJECTIVE if frozen else OBJECTIVE)
            if replay!=reflection:raise ValueError('Designer Reflection replay differs')
            if reflection['patch'] is not None:
                patch=SkillPatch.model_validate(reflection['patch'])
                context=frozen_static_inputs(learned.values(),pool) if frozen else static_context(learned.values())
                static_frozen=json.loads((root/f'g{g}.designer-static.json').read_text())
                if context!=record.get('static_context') or static_frozen!=context:
                    raise ValueError('Designer static validation inputs differ from complete verified memory')
                static=StaticSkillGate(allowed_tools=context['allowed_tools'],allowed_domains=context['allowed_domains'],forbidden_identifiers=context['forbidden_identifiers']).validate(designer,patch)
                if static.model_dump(mode='json')!=record.get('static'):raise ValueError('Designer static gate differs')
                if static.passed and patch.action!='NOOP':
                    candidate=apply_patch(designer,patch,f'g{g}.designer-reflection')
                    if candidate.snapshot_hash!=record['candidate_hash']:raise ValueError('Designer candidate hash differs')
                    if not plan.enabled('designer_dynamic_gate'):
                        if record['outcome']!='accepted' or 'dynamic' in record or 'paired_trials' in record:
                            raise ValueError('static-only Designer selection differs')
                        designer=store.load(candidate.snapshot_hash)
                    else:
                        pairs=record['paired_trials'];old=[trials[h] for h in pairs['parent']];new=[trials[h] for h in pairs['candidate']]
                        for a,b in zip(old,new):
                            if (a.anchors,a.strategy,a.random_seed,a.customer_hash,a.service_hash)!=(b.anchors,b.strategy,b.random_seed,b.customer_hash,b.service_hash):
                                raise ValueError('Designer pairing changed sources, strategy, seeds or counterpart')
                            if (a.designer_hash,b.designer_hash)!=(designer.snapshot_hash,candidate.snapshot_hash):raise ValueError('Designer pairing snapshot mismatch')
                        if frozen:
                            by_hash={e.evidence_hash:e for e in evidence.values()}
                            world_groups={e.world_hash:e.structure_hash for e in pool.entries.values()}
                            expected_groups=[world_groups[by_hash[t.anchors[0]['evidence_hash']].config.initial_world_hash] for t in old]
                        else:expected_groups=[source_group(Anchor.model_validate(t.anchors[0])) for t in old]
                        if record['paired_groups']!=expected_groups:raise ValueError('source cluster definition differs')
                        if record['known_roots']!=known_contracts(learned.values(),learned_discoveries):raise ValueError('pre-validation known contracts differ')
                        result=compare(designer.snapshot_hash,candidate.snapshot_hash,[t.discovery for t in old],[t.discovery for t in new],
                            backend.costs(old),backend.costs(new),plan.search_policy,record['known_roots'],record['paired_groups'])
                        if result.model_dump(mode='json')!=record['dynamic']:raise ValueError('Designer dynamic selection differs')
                        if result.decision=='ACCEPT':designer=store.load(candidate.snapshot_hash)
        if row['selected_designer']!=designer.snapshot_hash or record['selected_hash']!=designer.snapshot_hash:
            raise ValueError('Designer rollback/selection mismatch')
        if plan.training_selection_protocol is not None and not plan.enabled('designer_dynamic_gate'):
            if reflection is None:
                outcome='fixed'
            elif reflection['error'] or reflection['patch'] is None or not record.get('static',{}).get('passed'):
                outcome='rejected'
            else:outcome='noop' if reflection['patch']['action']=='NOOP' else 'accepted'
            if record['outcome']!=outcome or 'dynamic' in record or 'paired_trials' in record:
                raise ValueError('static-only Designer outcome differs')
        if plan.training_selection_protocol is not None:
            expected=DesignerServiceController.selection_metadata(SimpleNamespace(plan=plan),'designer',row['parent_designer'],designer.snapshot_hash)
            if any(record.get(k)!=v for k,v in expected.items()):raise ValueError('Designer selection semantics differ')
        current=tuple(t.evidence for t in training if t.evidence)
        s_record=row['service']
        if current:
            model=plan.service_reflector_model or plan.service_model
            reflector=ServiceReflector(recorded_client(s_record['reflection']['call'],model),model)
            with tempfile.TemporaryDirectory(prefix='designer-service-replay-') as folder:
                temporary=LibraryStore(Path(folder)/'libraries')
                for h in store.hashes():temporary.save(store.load(h),publish=False)
                evolved=ServiceEvolver(offline,reflector,temporary,EvidenceArchive(Path(folder)/'archive'),plan.service_policy).update(
                    g,customer,service,current,discovery,tasks,f'g{g}.service',dynamic_gate=plan.enabled('service_dynamic_gate'))
                expected=evolved.model_dump(mode='json')
                if plan.training_selection_protocol is not None:
                    expected.update(DesignerServiceController.selection_metadata(SimpleNamespace(plan=plan),'service',service.snapshot_hash,evolved.selected_snapshot_hash))
                if expected!=s_record:raise ValueError('original Service Evolution/Gate replay differs')
                service=store.load(evolved.selected_snapshot_hash)
        if plan.training_selection_protocol is not None:
            expected=DesignerServiceController.selection_metadata(SimpleNamespace(plan=plan),'service',row['parent_service'],service.snapshot_hash)
            if any(s_record.get(k)!=v for k,v in expected.items()):raise ValueError('Service selection semantics differ')
        if row['selected_service']!=service.snapshot_hash:raise ValueError('Service selected snapshot mismatch')
    if plan.training_selection_protocol is not None:
        flags={'version':plan.training_selection_protocol,**{k:plan.enabled(k) for k in
            ('designer_quality_scoring','designer_dynamic_gate','service_dynamic_gate')}}
        if report.get('selection_configuration')!=flags or report.get('performance_improved')!='unknown':
            raise ValueError('selection configuration/claim differs')
        if not plan.enabled('designer_quality_scoring') and 'metrics' in report:
            raise ValueError('disabled quality scoring appears in training report')
    if (report['selected_designer'],report['selected_service'],report['fixed_customer'])!=(designer.snapshot_hash,service.snapshot_hash,customer.snapshot_hash):
        raise ValueError('final selected snapshots differ')
