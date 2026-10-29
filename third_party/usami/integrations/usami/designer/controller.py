"""Opt-in recursive Designer–Service loop. Old CoEvolutionController is untouched."""
import copy
import json
import random
from pathlib import Path
from typing import Literal, Optional, Tuple
from pydantic import Field, model_validator
from vast_coevo.schema import FrozenModel
from vast_coevo.designer.storage import atomic_json, digest
from vast_coevo.models import Budget, ModelSpec
from vast_coevo.skills.library import LibrarySnapshot, LibraryStore, apply_patch
from vast_coevo.skills.models import SkillPatch
from vast_coevo.skills.validation import StaticSkillGate
from vast_coevo.designer.models import Ablation, SearchPolicy
from vast_coevo.designer.evaluation import compare, summarize
from vast_coevo.designer.reflection import reflect, OBJECTIVE
from vast_coevo.evolution.models import ValidationCase, GatePolicy
from vast_coevo.evolution.evolvers import ServiceEvolver
from vast_coevo.reflection.reflector import ServiceReflector
from vast_coevo.evidence.archive import EvidenceArchive
from integrations.usami.adapter import UsamiAdapter, source_hashes
from integrations.usami.domains import UsamiProfile, ServiceTaskProfile, model_settings
from .backend import SearchBackend
from .discovery import DiscoveryBank
from .compiler import source_group
from .attribution import static_context, known_contracts


class DesignerPlan(FrozenModel):
    protocol: Literal['designer-service-v1'] = 'designer-service-v1'
    training_selection_protocol: Optional[Literal['explicit-selection-v2','legacy-dynamic-v1']] = Field(default=None, exclude_if=lambda v:v is None)
    designer_quality_scoring: Optional[bool] = Field(default=None, exclude_if=lambda v:v is None)
    designer_dynamic_gate: Optional[bool] = Field(default=None, exclude_if=lambda v:v is None)
    service_dynamic_gate: Optional[bool] = Field(default=None, exclude_if=lambda v:v is None)
    strategy: Ablation = 'bidirectional'

    def enabled(self, name):
        value=getattr(self,name)
        return True if value is None else value
    random_seed: int = 731
    generations: int = Field(default=2, ge=1, le=100)
    max_prompt_sources: int = Field(default=8, ge=2, le=32)
    trials_per_generation: int = Field(default=2, ge=1, le=100)
    designer_validation_pairs: int = Field(default=8, ge=1, le=100)
    designer_model: ModelSpec
    customer_model: ModelSpec
    service_model: ModelSpec
    designer_reflector_model: Optional[ModelSpec] = None
    service_reflector_model: Optional[ModelSpec] = None
    budget: Budget = Field(default_factory=lambda:Budget(max_customer_turns=3,max_customer_llm_calls=3,max_service_llm_calls=8,max_service_tool_calls=12))
    max_llm_calls: int = Field(default=300, ge=1)
    max_tool_calls: int = Field(default=500, ge=1)
    search_policy: SearchPolicy = Field(default_factory=SearchPolicy)
    service_policy: GatePolicy = Field(default_factory=GatePolicy)
    profiles: Tuple[UsamiProfile, ...]
    tasks: Tuple[ServiceTaskProfile, ...]
    initial_tasks: Tuple[str, ...] = Field(min_length=1)
    validation_profiles: Tuple[str, ...] = Field(min_length=1)
    legal_tasks: Tuple[str, ...] = Field(min_length=1)
    refusal_tasks: Tuple[str, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def isolated(self):
        if self.enabled('designer_dynamic_gate') and not self.enabled('designer_quality_scoring'):
            raise ValueError('Designer dynamic gate requires explicit quality scoring')
        # Preserve byte-exact native query observations across JSON persistence.
        object.__setattr__(self,'profiles',tuple(UsamiProfile.model_validate(json.loads(json.dumps(p.model_dump(mode='json'),sort_keys=True))) for p in self.profiles))
        profiles={p.config_id:p for p in self.profiles}; tasks={t.task_ref:t for t in self.tasks}
        if len(profiles)!=len(self.profiles) or len(tasks)!=len(self.tasks):raise ValueError('duplicate registered source')
        if any(p.domain!='cloudops' or p.split=='heldout' or p.examiner_protocol or p.read_probe_protocol for p in self.profiles):
            raise ValueError('Designer training requires independent CloudOps development profiles')
        if set(self.initial_tasks+self.legal_tasks+self.refusal_tasks)-set(tasks) or set(self.validation_profiles)-set(profiles):raise ValueError('unregistered task/profile')
        if any(t.config_id not in profiles for t in self.tasks):raise ValueError('task profile missing')
        if any(profiles[p].fixed_customer_brief is None for p in self.validation_profiles):
            raise ValueError('validation Discovery must also use the fixed Customer policy')
        if self.budget.max_service_tool_calls<1:raise ValueError('CloudOps requires at least one reserved tool call')
        if self.budget.max_customer_llm_calls<2 or self.budget.max_customer_turns<2:
            raise ValueError('new protocol requires a genuine multi-turn Customer budget')
        return self


class Ledger:
    def __init__(self, root, max_calls, max_tools, factory):
        self.root=Path(root);self.max_calls=max_calls;self.max_tools=max_tools;self.factory=factory
        self.calls=0;self.tools=0;self.logs=[]

    def reserve_tools(self, amount):
        if self.tools+amount>self.max_tools:raise ValueError('global business tool budget insufficient for bounded episode')

    def client(self, model, role, phase):
        inner=self.factory(model,role,phase)
        if inner.settings!=model_settings(model):raise ValueError('client differs from frozen model configuration')
        ledger=self
        class Client:
            settings=inner.settings
            def complete(self, request):
                if ledger.calls>=ledger.max_calls:raise ValueError('global model-call budget exhausted')
                ledger.calls+=1;number=ledger.calls
                pending={'number':number,'phase':phase,'role':role,'model':model,'request':copy.deepcopy(request)}
                path=ledger.root/'calls'/str(number)
                atomic_json(path.with_suffix('.started.json'),pending,immutable=True)
                try: response=inner.complete(copy.deepcopy(request))
                except (Exception,KeyboardInterrupt) as exc:
                    completed={**pending,'error':{'type':type(exc).__name__,'code':getattr(exc,'code','client_error'),'message':str(exc)},'response':getattr(exc,'response',None)}
                    atomic_json(path.with_suffix('.completed.json'),completed,immutable=True);ledger.logs.append(completed);raise
                completed={**pending,'error':None,'response':copy.deepcopy(response)}
                atomic_json(path.with_suffix('.completed.json'),completed,immutable=True);ledger.logs.append(completed)
                return response
            def close(self):
                if hasattr(inner,'close'):inner.close()
        return Client()


class MeteredAdapter:
    def __init__(self, adapter, ledger, archive):self.base=adapter;self.ledger=ledger;self.archive=archive
    def __getattr__(self,name):return getattr(self.base,name)
    def _run(self,method,config,*args):
        self.ledger.reserve_tools(config.budget.max_service_tool_calls)
        row=method(config,*args);self.ledger.tools+=row.consumption.tool_calls;self.archive.save(row);return row
    def run_episode(self,config,*args):return self._run(self.base.run_episode,config,*args)
    def run_service_task(self,config,*args):return self._run(self.base.run_service_task,config,*args)
    def run_failure_case(self,source,*args):
        self.ledger.reserve_tools(source.config.budget.max_service_tool_calls)
        row=self.base.run_failure_case(source,*args);self.ledger.tools+=row.consumption.tool_calls;self.archive.save(row);return row


def feedback_for(backend, training):
    return [{'id':t.id,'proposal':t.proposal,'diff':t.scene['diff'] if t.scene else None,
        'reference_before':t.scene['reference_before'] if t.scene else None,'reference_after':t.scene['reference_after'] if t.scene else None,
        'discovery':t.discovery.model_dump(mode='json'),'error':t.error,'cost':backend.costs([t])} for t in training]


class DesignerServiceController:
    backend_type=SearchBackend
    bank_type=DiscoveryBank
    reflector_objective=OBJECTIVE
    def prepare_run(self):pass
    def group_for(self,anchor):return source_group(anchor)
    def static_inputs(self):return static_context(self.bank.anchors.values())
    def __init__(self, root, plan, designer, customer, service, factory):
        self.root=Path(root);self.plan=type(plan).model_validate(plan.model_dump(mode='json'))
        self.designer=LibrarySnapshot.model_validate(designer.model_dump(mode='json'))
        self.customer=LibrarySnapshot.model_validate(customer.model_dump(mode='json'))
        self.service=LibrarySnapshot.model_validate(service.model_dump(mode='json'))
        if (designer.role,customer.role,service.role)!=('designer','customer','service'):raise ValueError('three separate role libraries required')
        self.fixed_customer_hash=customer.snapshot_hash
        self.store=LibraryStore(self.root/'libraries');self.archive=EvidenceArchive(self.root/'evidence')
        self.ledger=Ledger(self.root,plan.max_llm_calls,plan.max_tool_calls,factory)
        raw=UsamiAdapter(self.root/'native',plan.profiles,plan.tasks,
            lambda spec,role,config,episode_id:self.ledger.client(spec,role,episode_id))
        self.adapter=MeteredAdapter(raw,self.ledger,self.archive)
        self.backend=self.backend_type(self.root,self.adapter,self.customer,plan.customer_model,plan.service_model,plan.designer_model,
            plan.budget,self.ledger.client,self.ledger)
        self.bank=self.bank_type(self.root/'memory');self.history=[]

    def task_case(self, ref, purpose=None):
        task=self.adapter._tasks[ref]
        config=self.adapter.config_for(task.config_id,self.plan.customer_model,self.plan.service_model,self.plan.budget,task_ref=ref)
        return ValidationCase(id=ref,config=config,task_ref=ref,purpose=purpose)

    def validation(self):
        discovery=tuple(ValidationCase(id=p,config=self.adapter.config_for(p,self.plan.customer_model,self.plan.service_model,self.plan.budget)) for p in self.plan.validation_profiles)
        tasks=tuple(self.task_case(t,'legal_completion') for t in self.plan.legal_tasks)+tuple(self.task_case(t,'correct_refusal') for t in self.plan.refusal_tasks)
        return discovery,tasks

    def selection_metadata(self, role, parent, selected):
        if self.plan.training_selection_protocol is None:return {}
        if self.plan.enabled(role+'_dynamic_gate'):return {}
        return {'selection_mode':'evidence_only_static',
                'performance_validated':False,
                'skill_updated':parent!=selected,'performance_improved':'unknown'}

    def designer_update(self, generation, training):
        parent=self.designer;prefix='g'+str(generation)
        if self.plan.strategy in {'random_legal','bidirectional_fixed_designer'}:
            return {'outcome':'fixed','parent_hash':parent.snapshot_hash,'selected_hash':parent.snapshot_hash,'reflection':None}
        feedback=feedback_for(self.backend,training)
        model=self.plan.designer_reflector_model or self.plan.designer_model
        reflection=reflect(self.ledger.client(model,'designer_reflector',prefix+'.designer-reflection'),model,parent,feedback,prefix,objective=self.reflector_objective)
        record={'parent_hash':parent.snapshot_hash,'selected_hash':parent.snapshot_hash,'reflection':reflection,'outcome':'rejected'}
        if reflection['error'] or reflection['patch'] is None:return record
        patch=SkillPatch.model_validate(reflection['patch'])
        context=self.static_inputs()
        record['static_context']=context
        atomic_json(self.root/f'{prefix}.designer-static.json',context,immutable=True)
        static=StaticSkillGate(allowed_tools=context['allowed_tools'],allowed_domains=context['allowed_domains'],forbidden_identifiers=context['forbidden_identifiers']).validate(parent,patch)
        record['static']=static.model_dump(mode='json')
        if not static.passed:return record
        if patch.action=='NOOP':record['outcome']='noop';return record
        candidate=apply_patch(parent,patch,prefix+'.designer-reflection');self.store.save(candidate,publish=False)
        record['candidate_hash']=candidate.snapshot_hash
        if not self.plan.enabled('designer_dynamic_gate'):
            self.store.save(candidate);self.designer=candidate
            record.update(outcome='accepted',selected_hash=candidate.snapshot_hash,reason='static_gate_passed_snapshot_updated')
            return record
        anchors=tuple(sorted((a for a in self.bank.anchors.values() if self.plan.strategy!='error_only' or a.branch=='error_amplification'),key=lambda a:a.anchor_hash))
        if self.plan.strategy=='correct_only':anchors=tuple(a for a in anchors if a.branch=='correct_to_failure')
        # Both arms start from identical verified memory; no opposite-arm feedback.
        old_bank=self.bank.fork(self.root/'validation'/prefix/'parent-memory')
        new_bank=self.bank.fork(self.root/'validation'/prefix/'candidate-memory')
        old=[];new=[];groups=[]
        for index in range(self.plan.designer_validation_pairs if anchors else 0):
            anchor=anchors[index%len(anchors)];groups.append(self.group_for(anchor))
            for label,snapshot,bank,rows in [('parent',parent,old_bank,old),('candidate',candidate,new_bank,new)]:
                rows.append(self.backend.trial(snapshot,self.service,[anchor],bank,f'{prefix}.designer-gate.{label}.{index}',
                                                self.plan.strategy,self.plan.random_seed+index))
        known=known_contracts(self.bank.anchors.values(),self.bank.records)
        decision=compare(parent.snapshot_hash,candidate.snapshot_hash,[t.discovery for t in old],[t.discovery for t in new],
            self.backend.costs(old),self.backend.costs(new),self.plan.search_policy,known,groups)
        record['known_roots']=sorted(known);record['paired_groups']=groups
        record['dynamic']=decision.model_dump(mode='json');record['paired_trials']={'parent':[t.trial_hash for t in old],'candidate':[t.trial_hash for t in new]}
        record['outcome']={'ACCEPT':'accepted','REJECT':'rejected','INCONCLUSIVE':'inconclusive'}[decision.decision]
        if decision.decision=='ACCEPT':self.store.save(candidate);self.designer=candidate;record['selected_hash']=candidate.snapshot_hash
        return record

    def source_pool(self, generation):
        pool=list(self.bank.anchors.values());random.Random(self.plan.random_seed+generation).shuffle(pool)
        selected=[];seen=set()
        # Retain the full bank; rotate a finite, branch/structure-stratified view.
        for anchor in pool:
            key=(anchor.branch,source_group(anchor))
            if key not in seen:selected.append(anchor);seen.add(key)
            if len(selected)>=self.plan.max_prompt_sources:break
        return tuple(selected)

    def run(self):
        self.root.mkdir(parents=True,exist_ok=False)
        initial=dict(designer=self.designer,customer=self.customer,service=self.service)
        atomic_json(self.root/'manifest.json',{'plan':self.plan,'initial':initial,'source_pins':source_hashes()},immutable=True)
        for snapshot in initial.values():self.store.save(snapshot)
        status='completed';error=None
        try:
            self.prepare_run()
            for ref in self.plan.initial_tasks:
                case=self.task_case(ref)
                evidence=self.adapter.run_service_task(case.config,ref,self.customer,self.service,'seed.'+ref)
                self.backend.register_source(evidence);self.bank.add_anchor(self.adapter,evidence)
            discovery,tasks=self.validation()
            for generation in range(self.plan.generations):
                parent_d,parent_s=self.designer,self.service
                pool=self.source_pool(generation) # Frozen before either actor learns this generation.
                atomic_json(self.root/f'g{generation}.sources.json',[a.anchor_hash for a in pool],immutable=True)
                training=[];current=[]
                for index in range(self.plan.trials_per_generation):
                    trial=self.backend.trial(self.designer,self.service,pool,self.bank,f'g{generation}.training.{index}',self.plan.strategy,self.plan.random_seed+generation*1000+index)
                    training.append(trial)
                    if trial.evidence:
                        current.append(trial.evidence)
                        try:self.bank.add_anchor(self.adapter,trial.evidence)
                        except ValueError:pass # Untriggered/uncertain episodes remain complete evidence, never become valid search sources.
                d_record=self.designer_update(generation,training)
                d_record.update(self.selection_metadata('designer',parent_d.snapshot_hash,self.designer.snapshot_hash))
                atomic_json(self.root/f'g{generation}.designer.json',d_record,immutable=True)
                if not current:
                    s_record={'outcome':'no_training_execution','selected_snapshot_hash':self.service.snapshot_hash}
                else:
                    model=self.plan.service_reflector_model or self.plan.service_model
                    reflector=ServiceReflector(self.ledger.client(model,'service_reflector',f'g{generation}.service-reflection'),model)
                    result=ServiceEvolver(self.adapter,reflector,self.store,self.archive,self.plan.service_policy).update(
                        generation,self.customer,self.service,tuple(current),discovery,tasks,f'g{generation}.service',
                        dynamic_gate=self.plan.enabled('service_dynamic_gate'))
                    s_record=result.model_dump(mode='json');self.service=self.store.load(result.selected_snapshot_hash)
                s_record.update(self.selection_metadata('service',parent_s.snapshot_hash,self.service.snapshot_hash))
                if self.customer.snapshot_hash!=self.fixed_customer_hash:raise ValueError('fixed Customer mutated')
                row={'generation':generation,'customer_hash':self.fixed_customer_hash,'parent_designer':parent_d.snapshot_hash,
                    'parent_service':parent_s.snapshot_hash,'selected_designer':self.designer.snapshot_hash,'selected_service':self.service.snapshot_hash,
                    'training':[t.trial_hash for t in training],'designer':d_record,'service':s_record}
                atomic_json(self.root/f'g{generation}.json',row,immutable=True);self.history.append(row)
        except (Exception,KeyboardInterrupt) as exc:status='blocked';error=type(exc).__name__+': '+str(exc)
        from integrations.usami.costs import _sum
        from types import SimpleNamespace
        costs=_sum([SimpleNamespace(response=r['response']) for r in self.ledger.logs],self.ledger.tools)
        unfinished=[str(p.parent) for p in (self.root/'trials').glob('*/registered.json') if not p.with_name('trial.json').exists()]
        report={'unfinalized_trials':unfinished,'cost_complete':not unfinished,'protocol':self.plan.protocol,'status':status,'error':error,'generations':self.history,
            'selected_designer':self.designer.snapshot_hash,'selected_service':self.service.snapshot_hash,'fixed_customer':self.fixed_customer_hash,
            **({'metrics':summarize(self.bank.records,costs)} if self.plan.enabled('designer_quality_scoring') else {}),'cost':costs,
            'cost_by_role':{role:_sum([SimpleNamespace(response=r['response']) for r in self.ledger.logs if r['role']==role]) for role in sorted({r['role'] for r in self.ledger.logs})},
            'trials':[t.trial_hash for t in self.backend.trials],'research_claim':'implementation only; no generalization claim'}
        if self.plan.training_selection_protocol is not None:
            report['selection_configuration']={'version':self.plan.training_selection_protocol,**{k:self.plan.enabled(k) for k in
                ('designer_quality_scoring','designer_dynamic_gate','service_dynamic_gate')}}
            report['performance_improved']='unknown'
        atomic_json(self.root/'report.json',report,immutable=True)
        return report
