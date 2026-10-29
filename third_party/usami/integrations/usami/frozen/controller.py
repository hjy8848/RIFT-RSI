"""New authoring protocol, reusing the original recursive lifecycle and Service Evolver."""
import json
from functools import lru_cache
from typing import Literal,Tuple
from pydantic import Field,model_validator
from vast_coevo.models import Budget
from vast_coevo.designer.storage import atomic_json,digest
from integrations.usami.domains import UsamiProfile,ServiceTaskProfile
from integrations.usami.designer.controller import DesignerPlan,DesignerServiceController
from integrations.usami.designer.attribution import static_context
from .pool import WorldSpec,WorldPool
from .catalog import audit
from .composer import FrozenTask
from .backend import FrozenSearchBackend
from .discovery import TaskDiscoveryBank

OBJECTIVE=(
 'You are the Designer Skill Reflector for frozen-world task discovery. Learn to choose the NEXT valuable task, not to edit worlds. '
 'Use actual verified errors and correct tasks to select existing worlds, identities, entities, relationships, operation combinations and goals. '
 'Your transformations field describes task-space selection/composition, never changes to initial worlds. '
 'Separate valid task construction, actual errors, reproducible decision boundaries and independently attributed rule roots. '
 'Do not reward complexity or generic task failure; keep repeated errors, cost and missing evidence explicit. '
 'No editing permissions/billing/dependencies before execution, no code, fixed IDs or hidden answers. '
 'Return one submit_designer_patch (or NOOP), role designer, skill type design. ADD/UPDATE must include the full skill: '
 'id, version, trigger, selection, transformations, interpretation, stop. Omit managed domains/hash. '
 'ADD starts at1; UPDATE increments once. Cite only provided trial IDs. Customer remains fixed; no retries.'
)


@lru_cache(maxsize=4)
def prepared(specs_json,budget_json):
    specs=tuple(WorldSpec.model_validate(x) for x in json.loads(specs_json));budget=Budget.model_validate_json(budget_json)
    pool=WorldPool(specs);report,catalog=audit(pool,budget)
    profiles=[];tasks=[];initial=[];legal=[];refusal=[];validation=[]
    for entry in pool.entries.values():
        scenes=[s for s in catalog if s.proposal.world_id==entry.spec.id]
        simple=next(s for s in scenes if s.certificate['kind']=='completion' and len(s.witness)==1 and s.proposal.target.name=='start_instance')
        negative=next(s for s in scenes if s.certificate['kind']=='policy_refusal' and s.certificate['reference_code']=='PROJECT_ACCESS_REQUIRED')
        multi=next(s for s in scenes if s.certificate['kind']=='completion' and len(s.witness)>1)
        for label,scene in (('legal',simple),('refusal',negative),('multi',multi)):
            cid=entry.spec.id+'-'+label;ref='boot-'+cid
            p=UsamiProfile(config_id=cid,domain='cloudops',seed=entry.spec.seed,identity=scene.proposal.identity,mode='sandbox',frozen_world=entry.world)
            profiles.append(p);tasks.append(ServiceTaskProfile(task_ref=ref,config_id=cid,scenario='frozen-existing-entity',frozen_task=scene.native_task))
            if label=='legal':initial.append(ref);legal.append(ref)
            elif label=='refusal':initial.append(ref);refusal.append(ref)
            else:legal.append(ref)
        cid=entry.spec.id+'-dialogue'
        profiles.append(UsamiProfile(config_id=cid,domain='cloudops',seed=entry.spec.seed,identity=simple.proposal.identity,mode='sandbox',frozen_world=entry.world,
            fixed_customer_brief=simple.brief,fixed_task_spec={'kind':'completion','goal':simple.proposal.goal.model_dump(mode='json'),'reference_code':None,'task_protocol':'frozen-task-discovery-v1'}))
        validation.append(cid)
    fields={'profiles':[p.model_dump(mode='json') for p in profiles],'tasks':[t.model_dump(mode='json') for t in tasks],
        'initial_tasks':initial,'validation_profiles':validation,'legal_tasks':legal,'refusal_tasks':refusal}
    return json.dumps({'fields':fields,'audit':report,'catalog':[s.model_dump(mode='json') for s in catalog]},sort_keys=True)


def preparation(specs,budget):
    return json.loads(prepared(json.dumps([s.model_dump(mode='json') for s in specs],sort_keys=True),budget.model_dump_json()))


class FrozenPlan(DesignerPlan):
    protocol: Literal['frozen-task-discovery-v1']='frozen-task-discovery-v1'
    training_selection_protocol: Literal['explicit-selection-v2','legacy-dynamic-v1']='explicit-selection-v2'
    designer_quality_scoring: bool=False
    designer_dynamic_gate: bool=False
    service_dynamic_gate: bool=False
    world_specs: Tuple[WorldSpec,...]=Field(min_length=1,max_length=128)
    max_prompt_worlds: int=Field(default=4,ge=1,le=8)
    @model_validator(mode='before')
    @classmethod
    def derive_registration(cls,raw):
        raw=dict(raw)
        specs=tuple(WorldSpec.model_validate(x) for x in raw.get('world_specs',[]))
        if not specs:raise ValueError('frozen-world plan needs preregistered independent world specs')
        budget=Budget.model_validate(raw.get('budget',{})) if 'budget' in raw else Budget(max_customer_turns=3,max_customer_llm_calls=3,max_service_llm_calls=8,max_service_tool_calls=12)
        data=preparation(specs,budget)['fields']
        for key,value in data.items():
            if key in raw and raw[key]!=value:raise ValueError('profiles/tasks must be derived from frozen world registration: '+key)
            raw[key]=value
        return raw


class FrozenController(DesignerServiceController):
    backend_type=FrozenSearchBackend
    bank_type=TaskDiscoveryBank
    reflector_objective=OBJECTIVE
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.pool=WorldPool(self.plan.world_specs);data=preparation(self.plan.world_specs,self.plan.budget)
        self.coverage_audit=data['audit'];self.catalog=[FrozenTask.model_validate(s).checked() for s in data['catalog']]
        self.backend.configure(self.pool,self.catalog,self.plan.max_prompt_worlds)
    def prepare_run(self):
        self.pool.save(self.root)
        atomic_json(self.root/'world-pool-audit.json',self.coverage_audit,immutable=True)
    def group_for(self,anchor):
        from vast_coevo.models import EpisodeEvidence
        row=EpisodeEvidence.model_validate_json((self.root/'sources'/(anchor.evidence_hash+'.json')).read_text())
        entry=next((e for e in self.pool.entries.values() if e.world_hash==row.config.initial_world_hash),None)
        if entry is None:raise ValueError('learning source outside frozen pool')
        return entry.structure_hash # Name/seed changes cannot manufacture independent worlds.
    def static_inputs(self):
        return frozen_static_inputs(self.bank.anchors.values(),self.pool)


def frozen_static_inputs(anchors,pool):
    value=static_context(anchors)
    pool_ids={k for e in pool.entries.values() for c in ('projects','instances','access','users','invoices') for k in e.world[c]}
    pool_ids.update(pool.entries);pool_ids.update(e.world_hash for e in pool.entries.values())
    value['forbidden_identifiers']=sorted(set(value['forbidden_identifiers'])|pool_ids);value['world_pool_hash']=pool.pool_hash
    return value


def archived_plan(raw):
    """Missing selector in an archived manifest denotes the original dynamic protocol."""
    raw=dict(raw)
    if 'training_selection_protocol' not in raw:
        raw.update(training_selection_protocol='legacy-dynamic-v1',designer_quality_scoring=True,
                   designer_dynamic_gate=True,service_dynamic_gate=True)
    return FrozenPlan.model_validate(raw)
