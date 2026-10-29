"""LLM-facing task Designer for frozen, native benchmark worlds.

Dialogue Designers query a compact, frozen catalogue and submit business goals.
The explicit legacy protocol retains full-context single-shot behavior. Neither
may edit the world or supply Reference outcomes; native certification is shared.
"""

import importlib
import json
from typing import Any, Literal

from pydantic import Field
from usami_harness.models import Model, digest
from usami.service.models import ModelSettings, strict_json
from usami.service.runner import completion_message

from .public_policy import catalogue
from .tasks import (CertifiedTask, ConstructionBudget, OpenDialogueComposition, TaskSelection,
                    compile_task, verify_task)
from .tool_tasks import binding, visible_ids
from .splits import verified_world
from .contracts import TRACKS


DESIGNER_POLICY=(
    'You are the Benchmark Task Designer. You inspect a complete frozen business world and choose a useful, '
    'multi-step Service challenge from entities and rules that actually exist. You may select several related '
    'Customer profiles only when they share the same authenticated identity. You may combine multiple completion '
    'goals, exact business operations that should be refused, declared state frames, information needs, permitted '
    'Customer choices and confirmations. The world is read-only. Never invent or edit entities, operations, rules, '
    'permissions, evidence, external events or expected outcomes. Do not provide Reference answers or refusal codes. '
    'For dialogue domains, submit only kind, family, profile_id, and composition; put the Customer request inside '
    'composition.customer_request, never in a top-level message or identity field. Use family open_composition_v1. '
    'For tool domains, '
    'provide exact customer-visible operations and a JointGoal. Prefer distinct but coherent business objectives; '
    'do not repeat the same operation merely to inflate task size. The native compiler will reject unsupported proposals.'
)


class DesignerExecution(Model):
    protocol:Literal['frozen-world-task-designer-v1']='frozen-world-task-designer-v1'
    frozen_hash:str
    model:ModelSettings
    request:dict
    response:Any
    selection:TaskSelection
    task:CertifiedTask
    prompt_context:dict=Field(default_factory=dict,exclude_if=lambda value:not value)
    cost:dict=Field(default_factory=dict)
    execution_hash:str=''
    def computed_hash(self):return digest(self.model_dump(mode='json',exclude={'execution_hash'}))


class LegacyDialogueDesignerProposal(Model):
    """Dialogue-track proposal shape excludes tool-only and compiler-owned fields."""
    model_config = {"title": "DialogueDesignerProposal"}
    kind:Literal['dialogue']
    family:Literal['open_composition_v1']
    profile_id:str=Field(min_length=1)
    composition:OpenDialogueComposition


def _dialogue_contract(frozen,world):
    domain=frozen.spec.domain
    adapter=importlib.import_module('usami_'+domain+'.adapter')
    tasks=importlib.import_module('usami_'+domain+'.tasks')
    profiles=[]
    for profile_id in sorted(world.profiles):
        private=adapter.private_state(world,profile_id)
        if domain=='aftersales':frames=tasks.frame_contract(world,private)
        elif domain=='airline':frames=tasks.frames(world,private)
        elif domain=='property':frames=tasks.frames(world,private,inspection=True)
        else:frames=tasks.frames(world,private)
        profiles.append({'profile_id':profile_id,'identity':private.identity,
            'private_facts':private.facts,'preference_space':private.preference_space,
            'initial_preferences':private.initial_preferences,'protected_objects':list(private.protected_objects),
            'confirmable_actions':list(private.confirmable_actions),
            'native_candidate_intents':adapter.candidate_intents(world,private),
            'native_frame_contract':[row.model_dump(mode='json') for row in frames],
            'external_events':[event.model_dump(mode='json') for event in world.external_events.values()
                               if any(value in private.protected_objects for value in event.model_dump(mode='json').values())]})
    return profiles


def designer_view(frozen, *, protocol=None):
    """Compact by default for dialogue worlds; explicit v1 stays replayable."""
    if protocol is None:
        protocol = "frozen-world-designer-exploration-v4" if TRACKS[frozen.spec.domain] == "B" else "frozen-world-task-designer-v1"
    if protocol in {"frozen-world-designer-exploration-v2", "frozen-world-designer-exploration-v3",
                    "frozen-world-designer-exploration-v4"}:
        from .designer_exploration import exploration_view
        return exploration_view(frozen, protocol=protocol)
    if protocol != "frozen-world-task-designer-v1":
        raise ValueError("unknown Designer protocol")
    frozen=verified_world(frozen)
    domain=frozen.spec.domain;world_module=importlib.import_module('usami_benchmark.tool_tasks')
    body={'protocol':'frozen-world-designer-view-v1','frozen_registration_hash':frozen.registration_hash,
        'world_hash':frozen.world_hash,'domain':domain,'rules_version':frozen.spec.rules_version,
        'track':'A' if domain in {'saas','cloudops','fulfillment','stayops'} else 'B',
        'world':frozen.world,'public_business_policy':catalogue(domain,frozen.spec.rules_version)}
    if body['track']=='B':
        adapter=importlib.import_module('usami_'+domain+'.adapter')
        world=adapter.World.model_validate(frozen.world)
        body['customer_profiles_and_native_task_contracts']=_dialogue_contract(frozen,world)
    else:
        world=world_module.load_world(frozen);native,_=binding(domain,world)
        identities=[]
        for identity in sorted(world.users):
            identities.append({'identity':identity,'visible_entity_ids':sorted(visible_ids(domain,world,identity))})
        body['customer_identities_and_visible_entities']=identities
        body['operation_schema']=native.Operation.model_json_schema()
        # ``goal_adapter`` is a light namespace for some legacy domains, while
        # native adapters expose the same contract as a module. Serialize only
        # that public field allowlist instead of assuming a Pydantic model.
        body['native_goal_fields']=binding(domain,world)[1].GOAL_FIELDS
    return body


def _designer_tools(frozen):
    if TRACKS[frozen.spec.domain]=='B':
        schema=LegacyDialogueDesignerProposal.model_json_schema()
        world=importlib.import_module('usami_'+frozen.spec.domain+'.adapter').World.model_validate(frozen.world)
        schema['properties']['profile_id']['enum']=sorted(world.profiles)
    else:
        schema=TaskSelection.model_json_schema()
    return [{'type':'function','function':{'name':'submit_frozen_world_task',
        'description':'Submit one task selection grounded in the supplied frozen world.',
        'parameters':schema}}]


def _designer_messages(frozen,prompt_context=None):
    context=prompt_context or {}
    policy=DESIGNER_POLICY
    rendered=context.get('skill_rendered')
    if rendered:
        policy+='\n\nApplied declarative Designer Skill from snapshot '+str(context.get('skill_snapshot_hash','unknown'))+':\n'+rendered
    messages=[{'role':'system','content':policy},
        {'role':'user','content':json.dumps(designer_view(frozen,protocol='frozen-world-task-designer-v1'),ensure_ascii=False,sort_keys=True,separators=(',',':'))}]
    if context:
        visible={key:value for key,value in context.items() if key!='skill_rendered'}
        messages.append({'role':'user','content':'Verified interaction evidence and current discovery hypothesis:\n'+json.dumps(visible,ensure_ascii=False,sort_keys=True,separators=(',',':'))})
    return messages


def _selection_from_response(response,frozen=None):
    message,actions=completion_message(response,set())
    if len(actions)!=1 or actions[0]['function']['name']!='submit_frozen_world_task':
        raise ValueError('Designer must return exactly one submit_frozen_world_task call')
    raw=strict_json(actions[0]['function']['arguments'])
    if frozen is not None and TRACKS[frozen.spec.domain]=='B':
        proposal=LegacyDialogueDesignerProposal.model_validate(raw)
        return TaskSelection.model_validate(proposal.model_dump(mode='json'))
    return TaskSelection.model_validate(raw)


def design_and_compile(frozen,client,construction_budget=None,prompt_context=None, *,
                       protocol=None, designer_budget=None, journal_path=None):
    """Explore and correct dialogue proposals within explicit budgets.

    Tool domains and explicit v1 calls retain their single-shot protocol.
    No protocol retries transport failures or accepts model-authored proofs.
    """
    if protocol is None:
        protocol = "frozen-world-designer-exploration-v4" if TRACKS[frozen.spec.domain] == "B" else "frozen-world-task-designer-v1"
    if protocol in {"frozen-world-designer-exploration-v2", "frozen-world-designer-exploration-v3",
                    "frozen-world-designer-exploration-v4"}:
        from .designer_exploration import design
        return design(frozen, client, construction_budget, prompt_context,
                      budget=designer_budget, journal_path=journal_path, protocol=protocol)
    if protocol != "frozen-world-task-designer-v1":
        raise ValueError("unknown Designer protocol")
    frozen=verified_world(frozen)
    settings=client.settings.model_copy(deep=True)
    prompt_context=dict(prompt_context or {})
    request=settings.request(_designer_messages(frozen,prompt_context),_designer_tools(frozen))
    response=client.complete(request)
    selection=_selection_from_response(response,frozen)
    task=compile_task(frozen,selection,construction_budget or ConstructionBudget())
    usage=response.get('usage',{}) if isinstance(response,dict) else {}
    cost={'calls':1,'input_tokens':usage.get('prompt_tokens'),'output_tokens':usage.get('completion_tokens'),
          'total_tokens':usage.get('total_tokens'),'usage_complete':all(isinstance(usage.get(k),int) for k in ('prompt_tokens','completion_tokens','total_tokens'))}
    row=DesignerExecution(frozen_hash=frozen.registration_hash,model=settings,request=request,response=response,
        selection=selection,task=task,prompt_context=prompt_context,cost=cost)
    return row.model_copy(update={'execution_hash':row.computed_hash()})


def verify_designer_execution(record):
    protocol = record.get("protocol") if isinstance(record,dict) else record.protocol
    if protocol in {"frozen-world-designer-exploration-v2", "frozen-world-designer-exploration-v3",
                    "frozen-world-designer-exploration-v4"}:
        from .designer_exploration import verify_execution
        return verify_execution(record)
    row=DesignerExecution.model_validate(record.model_dump(mode='json') if isinstance(record,DesignerExecution) else record)
    if row.execution_hash!=row.computed_hash():raise ValueError('Designer execution integrity differs')
    if row.frozen_hash!=row.task.frozen.registration_hash or row.selection!=row.task.selection:
        raise ValueError('Designer proposal differs from the compiled frozen task')
    expected_request=row.model.request(_designer_messages(row.task.frozen,row.prompt_context),
                                       _designer_tools(row.task.frozen))
    if row.request!=expected_request:raise ValueError('recorded Designer request differs from the frozen-world task prompt')
    if _selection_from_response(row.response,row.task.frozen)!=row.selection:
        raise ValueError('compiled task selection differs from the raw Designer model response')
    if verify_task(row.task) is not True:raise ValueError('native task proof failed to replay')
    return {'verified':True,'model_calls':1,'task_hash':row.task.task_hash,'task_replay':True}


# The current high-level proposal excludes compiler-owned frames. Historical
# v1 parsing above explicitly retains its original wire model and prompt.
from .designer_exploration import DialogueDesignerProposal
