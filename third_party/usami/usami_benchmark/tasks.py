"""One frozen task-only compiler and exact proof record for eight native domains.

Designer-facing selections contain no editable world, witness, certificate,
expected verdict or refusal answer. The private compiler discovers and proves
native outcomes. A bounded failure is not a claim of impossibility.
"""
import importlib
from typing import Literal,Optional
from pydantic import Field
from usami_harness.models import Model,digest
from usami_harness.goals import JointGoal
from usami_dialogue.tasks import (TaskCertificate,DialogueComposition,InformationRequirement,PreferenceRequirement,
    CustomerFollowUp,Goal,CreatedObjectGoal,FrameRule,RefusalGoal,checked_task as checked_dialogue,certify as certify_dialogue)
from usami_dialogue.engine import ExecutionTrace
from usami_dialogue.planner import search as dialogue_search,certified_paths
from .contracts import TRACKS
from .worlds import FrozenWorld
from .splits import verified_world
from .tool_tasks import ToolTask,ToolCertificate,certify as certify_tool,load_world as tool_world,binding,expected_goal
from .public_policy import catalogue

class ConstructionBudget(Model):
    max_nodes:int=Field(default=512,strict=True,ge=1,le=20000)
    max_depth:int=Field(default=12,strict=True,ge=1,le=16)
    max_paths:int=Field(default=1,strict=True,ge=1,le=8)

class OpenDialogueComposition(Model):
    """Designer-authored objectives bound to selected profiles in a frozen world."""
    # v2 aligns the open-composition compiler with native refusal-only tasks:
    # a verified refusal is a complete objective and need not invent a
    # separate information/consent dependency. Serialized v1 remains replayable.
    protocol:Literal['open-dialogue-composition-v1','open-dialogue-composition-v2','open-dialogue-composition-v3','open-dialogue-composition-v4']='open-dialogue-composition-v2'
    profile_ids:list[str]=Field(min_length=1,max_length=64)
    customer_request:str=Field(min_length=1,max_length=6000)
    goals:list[Goal]=Field(default_factory=list,max_length=32)
    created_goals:list[CreatedObjectGoal]=Field(default_factory=list,max_length=32,exclude_if=lambda value:not value)
    refusal_operations:list[dict]=Field(default_factory=list,max_length=16)
    frames:list[FrameRule]=Field(min_length=1,max_length=128)
    external_schedule:list[str]=Field(default_factory=list,max_length=64)
    information_requirements:list[InformationRequirement]=Field(default_factory=list,max_length=64)
    preference_requirements:list[PreferenceRequirement]=Field(default_factory=list,max_length=64)
    confirmation_requirements:list[dict]=Field(default_factory=list,max_length=16)
    follow_up_requests:list[CustomerFollowUp]=Field(default_factory=list,max_length=16,
        exclude_if=lambda value:not value)

class TaskSelection(Model):
    kind:Literal['tool','dialogue']
    family:str=Field(min_length=1,max_length=160)
    identity:Optional[str]=None
    profile_id:Optional[str]=None
    message:Optional[str]=Field(default=None,min_length=1,max_length=6000)
    goal:Optional[JointGoal]=None
    capabilities:list[dict]=Field(default_factory=list,max_length=16)
    refusal_operations:list[dict]=Field(default_factory=list,max_length=4)
    parameters:dict=Field(default_factory=dict)
    composition:Optional[OpenDialogueComposition]=Field(default=None,exclude_if=lambda value:value is None)

class DialogueWitness(Model):
    certificate:TaskCertificate
    trace:ExecutionTrace

class CertifiedTask(Model):
    protocol:Literal['eight-domain-certified-task-v1']='eight-domain-certified-task-v1'
    frozen:FrozenWorld
    selection:TaskSelection
    construction_budget:ConstructionBudget
    tool_certificate:Optional[ToolCertificate]=None
    dialogue_witnesses:list[DialogueWitness]=Field(default_factory=list)
    search_summary:dict
    record_hash:str=''
    @property
    def task(self):return self.tool_certificate.task if self.tool_certificate is not None else self.dialogue_witnesses[0].certificate.task
    @property
    def task_hash(self):return self.task.task_hash
    @property
    def track(self):return TRACKS[self.frozen.spec.domain]
    def computed_hash(self):return digest(self.model_dump(mode='json',exclude={'record_hash'}))

PARAMETERS={
 'aftersales':{'resolutions','mixed'},
 'airline':{'mixed','external_events','probe_parameters'},
 'property':{'slot_ids','mixed','inspection','probe_parameters'},
 'education':{'class_ids','slot_ids','extension_days','mixed','academic_review','probe_parameters'},
}


def _probe(domain,w,p,family,params):
    adapter=importlib.import_module('usami_'+domain+'.adapter')
    if domain=='airline':
        e=w.bookings[p.facts['booking_id']];scope=e.carrier_id
        names={'refuse_refund':'quote_refund','refuse_rebook':'quote_change','refuse_baggage':'quote_baggage'}
        defaults={'refuse_refund':{},'refuse_rebook':{'flights':p.initial_preferences['itinerary']},'refuse_baggage':{'extra_kg':p.facts['baggage_kg']}}
    elif domain=='property':
        from usami_property.rules import ticket_key
        e=w.units[p.facts['unit_id']];scope=e.property_id;tid=ticket_key(p.facts['asset_id']);slot=w.slots[p.initial_preferences['slot_id']]
        names={'refuse_open':'open_ticket','refuse_visit':'quote_visit','refuse_close':'close_ticket'}
        defaults={'refuse_open':{'asset_id':p.facts['asset_id'],'symptom':p.facts['symptom']},'refuse_visit':{'ticket_id':tid,'slot_id':slot.id,'technician_id':slot.technician_id},'refuse_close':{'ticket_id':tid}}
    elif domain=='education':
        from usami_education.rules import case_key
        e=w.enrollments[p.facts['enrollment_id']];scope=e.school_id;cid=case_key(e.id,p.facts['request_id'])
        names={'refuse_refund':'quote_refund','refuse_transfer':'quote_transfer','refuse_extension':'quote_extension','refuse_lesson':'quote_lesson'}
        defaults={'refuse_refund':{'case_id':cid},'refuse_transfer':{'case_id':cid,'target_class_id':p.initial_preferences['class_id'],'topup_units':0},'refuse_extension':{'case_id':cid,'days':p.initial_preferences['extension_days']},'refuse_lesson':{'case_id':cid,'slot_id':p.initial_preferences['slot_id']}}
    else:raise ValueError('native domain has no standalone refusal family')
    if family not in names:raise ValueError('unregistered native refusal family')
    if set(params)-set(defaults[family]):raise ValueError('probe parameters exceed the native family')
    # Model cannot replace source case/ticket/fault bindings; only real alternatives.
    if any(k in params and params[k]!=defaults[family][k] for k in ('case_id','ticket_id','asset_id','symptom')):raise ValueError('probe changes a frozen request binding')
    return adapter.Operation(name=names[family],scope_id=scope,resource_id=e.id,params={**defaults[family],**params}).model_dump(mode='json')


def _preferences(domain,w,p,t):
    if domain=='education':return importlib.import_module('usami_education.tasks').preference_options(w,p,t)
    if domain=='airline':
        service=t.family.removeprefix('refuse_');route=t.refusals[0].operation['params'].get('flights',w.bookings[p.facts['booking_id']].flights)
        return [{'service':service,'itinerary':route}]
    return [{'slot_id':t.refusals[0].operation['params'].get('slot_id',p.initial_preferences['slot_id']),'feedback':'satisfied'}]


def _dialogue(frozen,selection,budget):
    domain=frozen.spec.domain;adapter=importlib.import_module('usami_'+domain+'.adapter');module=importlib.import_module('usami_'+domain+'.tasks')
    w=adapter.World.model_validate(frozen.world)
    if selection.profile_id not in w.profiles:raise ValueError('profile is not in the frozen world')
    p=adapter.private_state(w,selection.profile_id)
    if selection.composition is not None:
        spec=OpenDialogueComposition.model_validate(selection.composition.model_dump(mode='json'))
        if selection.family!='open_composition_v1' or spec.profile_ids[0]!=selection.profile_id:
            raise ValueError('open composition needs family open_composition_v1 and its primary frozen profile')
        if selection.identity not in {None,p.identity} or selection.parameters or selection.message or selection.goal or selection.capabilities or selection.refusal_operations:
            raise ValueError('open composition cannot override identity, world, task, or operation scope')
        if len({digest(row) for row in spec.refusal_operations})!=len(spec.refusal_operations):raise ValueError('duplicate composed refusal request')
        from usami_dialogue.tasks import DialogueTask
        from usami_dialogue.planner import search_composed,execute_witness
        codes=list(catalogue(domain,frozen.spec.rules_version)['errors'])
        if spec.refusal_operations and not codes:raise ValueError('native Reference has no registered refusal codes')
        task=DialogueTask(domain=domain,rules_version=w.version,world_hash=w.state_hash,profile_id=selection.profile_id,
            private_hash=p.private_hash,family='open_composition_v1',
            protocol='certified-dialogue-task-v2' if spec.created_goals else 'certified-dialogue-task-v1',
            goals=spec.goals,created_goals=spec.created_goals,
            refusals=[RefusalGoal(operation=raw,code=codes[0]) for raw in spec.refusal_operations],
            frames=spec.frames,external_schedule=spec.external_schedule,
            require_dependency=(spec.protocol=='open-dialogue-composition-v1' or bool(spec.goals or spec.created_goals)),
            composition=DialogueComposition(protocol=spec.protocol,profile_ids=spec.profile_ids,customer_request=spec.customer_request,
                information_requirements=spec.information_requirements,preference_requirements=spec.preference_requirements,
                confirmation_requirements=spec.confirmation_requirements,
                follow_up_requests=spec.follow_up_requests))
        task=checked_dialogue(adapter,w,task)
        result=search_composed(adapter,w,task,max_nodes=budget.max_nodes,max_depth=budget.max_depth,
            max_paths=budget.max_paths,refusal_code_options=[set(codes) for _ in task.refusals])
        rows=[]
        for path,matched in zip(result['paths'],result.get('matched_refusal_codes',[[] for _ in result['paths']])):
            resolved=task.model_copy(deep=True)
            for refusal,code in zip(resolved.refusals,matched):refusal.code=code
            resolved.task_hash='';resolved=checked_dialogue(adapter,w,resolved)
            trace=execute_witness(adapter,w,resolved,path)
            rows.append(DialogueWitness(trace=trace,certificate=certify_dialogue(adapter,w,resolved,trace)))
        if not rows:raise ValueError('bounded Reference search did not prove the open composition')
        return rows,{k:v for k,v in result.items() if k not in {'paths','matched_refusal_codes'}}
    if selection.family=='open_composition_v1':raise ValueError('open composition payload required')
    if selection.identity is not None and selection.identity!=p.identity:raise ValueError('execution identity differs from frozen Customer profile')
    if selection.message is not None or selection.goal is not None or selection.capabilities or selection.refusal_operations:raise ValueError('dialogue goals must use native Customer choices, not replace facts, frames or requests')
    if set(selection.parameters)-PARAMETERS[domain]:raise ValueError('unregistered native task parameters')
    kwargs=dict(selection.parameters);probe=kwargs.pop('probe_parameters',{})
    if not isinstance(probe,dict):raise ValueError('probe parameters must be an object')
    if domain=='aftersales':
        expected='settlement_with_duplicate_refusal' if kwargs.get('mixed',False) else 'dispute_resolution'
        if selection.family!=expected:raise ValueError('unregistered aftersales family')
    else:kwargs['family']=selection.family
    if selection.family.startswith('refuse_'):
        raw=_probe(domain,w,p,selection.family,probe);templates=[]
        # Discover admissible static native reason vocabulary, then ONE bounded
        # Reference search determines the actual reason. Designer never sets it.
        for code in catalogue(domain,frozen.spec.rules_version)['errors']:
            try:templates.append(module.compose(w,selection.profile_id,refusal_operation=raw,refusal_code=code,**kwargs))
            except ValueError:continue
        if not templates:raise ValueError('no native refusal task for this scope/operation')
        t=templates[0];options={x.refusals[0].code for x in templates}
        result=dialogue_search(adapter,w,t,_preferences(domain,w,p,t),max_nodes=budget.max_nodes,max_depth=budget.max_depth,max_paths=budget.max_paths,refusal_code_options=options)
        if not result['paths']:raise ValueError('bounded native search did not prove a policy refusal')
        code=result['matched_refusal_codes'][0][0]
        t=module.compose(w,selection.profile_id,refusal_operation=raw,refusal_code=code,**kwargs)
        result['paths']=[path for path,codes in zip(result['paths'],result['matched_refusal_codes']) if codes==[code]]
    else:
        if probe:raise ValueError('probe parameters only apply to a refusal family')
        t=module.compose(w,selection.profile_id,**kwargs)
        result=module.search(w,t,max_nodes=budget.max_nodes,max_depth=budget.max_depth,max_paths=budget.max_paths)
    if not result['paths']:raise ValueError('bounded native dialogue search did not prove reachability')
    rows=certified_paths(adapter,w,t,result)
    return [DialogueWitness(**r) for r in rows],{k:v for k,v in result.items() if k not in {'paths','matched_refusal_codes'}}


def compile_task(frozen,selection,budget=None):
    frozen=verified_world(frozen);selection=TaskSelection.model_validate(selection.model_dump(mode='json') if isinstance(selection,TaskSelection) else selection)
    budget=ConstructionBudget.model_validate(budget.model_dump(mode='json') if isinstance(budget,ConstructionBudget) else budget or {})
    if (selection.kind=='tool')!=(TRACKS[frozen.spec.domain]=='A'):raise ValueError('selection kind differs from native domain track')
    if selection.kind=='tool':
        if selection.composition is not None or selection.parameters or selection.profile_id is not None or not selection.identity or not selection.message:raise ValueError('tool task needs exact identity, ordinary message and native capabilities')
        goal=selection.goal
        if goal is None and not selection.refusal_operations and len(selection.capabilities)==1:
            world=tool_world(frozen);b,_=binding(frozen.spec.domain,world);d=b.decide(world,selection.identity,b.Operation.model_validate(selection.capabilities[0]))
            if not d.allowed:raise ValueError('a completion needs an actual reachable goal or explicit refusal request')
            goal=expected_goal(frozen.spec.domain,world,d.expected_state)
        t=ToolTask(domain=frozen.spec.domain,registration_hash=frozen.registration_hash,world_hash=frozen.world_hash,identity=selection.identity,family=selection.family,message=selection.message,goal=goal,capabilities=selection.capabilities,refusal_operations=selection.refusal_operations)
        c=certify_tool(frozen,t,max_nodes=budget.max_nodes,max_depth=budget.max_depth,max_paths=budget.max_paths)
        record=CertifiedTask(frozen=frozen,selection=selection,construction_budget=budget,tool_certificate=c,search_summary={'nodes':c.search_nodes,'exhausted':c.search_exhausted,'max_nodes':c.max_nodes,'max_depth':c.max_depth})
    else:
        rows,summary=_dialogue(frozen,selection,budget)
        record=CertifiedTask(frozen=frozen,selection=selection,construction_budget=budget,dialogue_witnesses=rows,search_summary=summary)
    return record.model_copy(update={'record_hash':record.computed_hash()})


def verify_task(record):
    record=CertifiedTask.model_validate(record.model_dump(mode='json') if isinstance(record,CertifiedTask) else record)
    if record.record_hash!=record.computed_hash():raise ValueError('certified task record integrity differs')
    rebuilt=compile_task(record.frozen,record.selection,record.construction_budget)
    if rebuilt!=record:raise ValueError('native task construction/certificate Replay differs')
    return True
