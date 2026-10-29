"""Declarative completion/refusal goals with explicit frames, independent of witnesses.

A witness establishes reachability. Evaluation never compares an Agent's path to
that witness: any legal, authorized, frame-preserving solution can succeed.
"""
from typing import Any,Literal,Optional
from pydantic import Field
from usami_harness.models import Model,digest
from usami_harness.goals import field_value
from .dependencies import DependencyCertificate,certify_dependencies,verify_dependency
from .oracle import evaluate
from .replay import replay


class Goal(Model):
    collection:str
    entity:str
    field:str
    accepted_values:list[Any]=Field(min_length=1)
    def satisfied(self,world):
        obj=getattr(world,self.collection).get(self.entity)
        if obj is None:return False
        try:return field_value(obj,self.field) in self.accepted_values
        except ValueError:return False


class CreatedObjectGoal(Model):
    """A goal for one newly-created object, identified without guessing its ID.

    Selectors are grounded in a selected frozen Customer profile. The initial
    world must contain no matching object and the final world must contain
    exactly one whose public target field has an accepted value.
    """
    profile_id:str
    collection:str
    selectors:dict[str,Any]=Field(min_length=1,max_length=16)
    field:str
    accepted_values:list[Any]=Field(min_length=1,max_length=32)

    def matches(self,world):
        rows=getattr(world,self.collection,None)
        if not isinstance(rows,dict):return []
        matches=[]
        for row in rows.values():
            try:
                if all(type(field_value(row,key)) is type(value) and field_value(row,key)==value
                       for key,value in self.selectors.items()):
                    matches.append(row)
            except ValueError:
                continue
        return matches

    def satisfied(self,initial,final):
        if self.matches(initial):return False
        matches=self.matches(final)
        if len(matches)!=1:return False
        try:return field_value(matches[0],self.field) in self.accepted_values
        except ValueError:return False


class FrameRule(Model):
    collection:str
    selector_field:str='id'
    selector_value:Any
    mutable_fields:list[str]=Field(default_factory=list)
    allow_create:bool=False


class RefusalGoal(Model):
    operation:dict
    code:str


class InformationRequirement(Model):
    """Customer facts a composed challenge requires the Service to obtain."""
    profile_id:str
    fact_keys:list[str]=Field(min_length=1,max_length=32)

class PreferenceRequirement(Model):
    """Permitted task-specific choices drawn from one frozen profile."""
    profile_id:str
    preference:str=Field(min_length=1,max_length=96)
    accepted_values:list[Any]=Field(min_length=1,max_length=32)


class CustomerFollowUp(Model):
    """A public customer request that must occur after the primary outcome.

    ``requested_action`` is harness metadata for deterministic Customer
    contract fixtures only. It is deliberately not copied into UserScenario,
    and therefore is never sent to a live Customer model.
    """
    message:str=Field(min_length=1,max_length=6000)
    requested_action:Optional[str]=Field(default=None,min_length=1,max_length=96)


class DialogueComposition(Model):
    """Versioned task scope for evaluator-composed, multi-profile challenges."""
    # v1 preserves the original rule that every composed task claims a process
    # dependency. v2 permits an explicit refusal-only task with no completion
    # goal to be certified on its native policy boundary alone.
    protocol:Literal['open-dialogue-composition-v1','open-dialogue-composition-v2','open-dialogue-composition-v3','open-dialogue-composition-v4']='open-dialogue-composition-v1'
    profile_ids:list[str]=Field(min_length=1,max_length=64)
    customer_request:str=Field(min_length=1,max_length=6000)
    information_requirements:list[InformationRequirement]=Field(default_factory=list,max_length=64)
    preference_requirements:list[PreferenceRequirement]=Field(default_factory=list,max_length=64)
    confirmation_requirements:list[dict]=Field(default_factory=list,max_length=16)
    follow_up_requests:list[CustomerFollowUp]=Field(default_factory=list,max_length=16,
        exclude_if=lambda value:not value)


class DialogueTask(Model):
    protocol:Literal['certified-dialogue-task-v1','certified-dialogue-task-v2']='certified-dialogue-task-v1'
    domain:str
    rules_version:str
    world_hash:str
    profile_id:str
    private_hash:str
    family:str
    goals:list[Goal]=Field(default_factory=list)
    created_goals:list[CreatedObjectGoal]=Field(default_factory=list,max_length=32,exclude_if=lambda value:not value)
    refusals:list[RefusalGoal]=Field(default_factory=list)
    frames:list[FrameRule]=Field(default_factory=list)
    external_schedule:list[str]=Field(default_factory=list)
    require_dependency:bool=True
    composition:Optional[DialogueComposition]=Field(default=None,exclude_if=lambda value:value is None)
    task_hash:str=''
    def computed_hash(self):return digest(self.model_dump(mode='json',exclude={'task_hash'}))


class TaskCertificate(Model):
    protocol:Literal['dialogue-task-certificate-v1']='dialogue-task-certificate-v1'
    task:DialogueTask
    witness_trace_hash:str
    oracle:dict
    evaluation:dict
    dependencies:list[DependencyCertificate]
    certificate_hash:str=''
    def computed_hash(self):return digest(self.model_dump(mode='json',exclude={'certificate_hash'}))


def checked_task(adapter,world,task):
    task=DialogueTask.model_validate(task.model_dump(mode='json') if isinstance(task,DialogueTask) else task)
    private=adapter.private_state(world,task.profile_id)
    if (task.domain!=adapter.name or task.rules_version!=world.version or task.world_hash!=world.state_hash or task.private_hash!=private.private_hash):raise ValueError('task source registration differs')
    if not task.goals and not task.created_goals and not task.refusals:raise ValueError('task needs an objective completion or refusal goal')
    if task.created_goals and (task.protocol!='certified-dialogue-task-v2' or task.composition is None):
        raise ValueError('created-object goals require certified-dialogue-task-v2 and a versioned open composition')
    if task.protocol=='certified-dialogue-task-v2' and not task.created_goals:
        raise ValueError('certified-dialogue-task-v2 requires a created-object goal')
    if task.composition is None:
        adapter.validate_task(world,private,task)
    else:
        from .composition import validate_composed_task
        validate_composed_task(adapter,world,task)
    if len(set(task.external_schedule))!=len(task.external_schedule) or set(task.external_schedule)-set(world.external_events):raise ValueError('unregistered external schedule')
    value=task.computed_hash()
    if task.task_hash and task.task_hash!=value:raise ValueError('task content hash differs')
    return task.model_copy(update={'task_hash':value},deep=True)


def frame_preserved(initial,final,frames):
    left=initial.model_dump(mode='json');right=final.model_dump(mode='json')
    # These are controlled only by recorded Customer/external events. Their
    # correctness is separately verified by the full process/native Oracle.
    for raw in (left,right):
        raw.pop('interaction',None);raw.pop('clock',None);raw.pop('applied_events',None)
    for rule in frames:
        before=left.get(rule.collection);after=right.get(rule.collection)
        if not isinstance(before,dict) or not isinstance(after,dict):return False
        for key in list(set(before)|set(after)):
            a=before.get(key);b=after.get(key)
            def matches(obj):
                if obj is None:return False
                if rule.selector_field=='$key':return key==rule.selector_value
                try:return field_value(obj,rule.selector_field)==rule.selector_value
                except ValueError:return False
            if not matches(a) and not matches(b):continue
            if a is None:
                if rule.allow_create and matches(b):after.pop(key,None)
            elif b is not None:
                for field in rule.mutable_fields:
                    a.pop(field,None);b.pop(field,None)
    return left==right


def score(adapter,world,task,trace):
    task=checked_task(adapter,world,task)
    scope_ok=(trace.scoped_profile_ids==task.composition.profile_ids) if task.composition is not None else not trace.scoped_profile_ids
    source_ok=(trace.initial_world==world.model_dump(mode='json') and trace.profile_id==task.profile_id and trace.private.private_hash==task.private_hash and scope_ok)
    report=evaluate(adapter,trace,initial_world=world,private=adapter.private_state(world,task.profile_id))
    replay_ok=False
    if report['trace_valid']:
        try:replay_ok=replay(adapter,trace)['verified']
        except ValueError:pass
    final=adapter.World.model_validate(trace.final_world)
    completed=[g.satisfied(final) for g in task.goals]
    created_completed=[g.satisfied(world,final) for g in task.created_goals]
    refused=[any(r['classification']=='correct_refusal' and r['operation']==g.operation for r in report['service_decisions']) for g in task.refusals]
    frame=frame_preserved(world,final,task.frames)
    scheduled=all(e.payload['event_id'] in task.external_schedule for e in trace.events if e.kind=='external')
    information=[];confirmations=[];preferences=[]
    if task.composition is not None:
        for requirement in task.composition.information_requirements:
            profile=adapter.private_state(world,requirement.profile_id)
            disclosed={}
            for event in trace.events:
                if event.kind!='customer' or event.payload.get('setup') or event.payload.get('profile_id')!=requirement.profile_id:continue
                disclosed.update(event.payload['intent'].get('disclosures',{}))
            information.append(all(key in profile.facts and disclosed.get(key)==profile.facts[key] for key in requirement.fact_keys))
        actual_consents={digest(row['operation']) for row in report['consents']}
        confirmations=[digest(op) in actual_consents for op in task.composition.confirmation_requirements]
        for requirement in task.composition.preference_requirements:
            values=[event.payload['intent'].get('preferences',{}).get(requirement.preference)
                    for event in trace.events if event.kind=='customer' and not event.payload.get('setup')
                    and event.payload.get('profile_id')==requirement.profile_id]
            preferences.append(any(value in requirement.accepted_values for value in values))
    success=source_ok and report['status']=='pass' and replay_ok and all(completed) and all(created_completed) and all(refused) and frame and scheduled
    if task.composition is not None:
        success=success and all(information) and all(confirmations) and all(preferences)
    result={'protocol':'dialogue-task-score-v3' if task.created_goals else 'dialogue-task-score-v2','success':bool(success),'source_verified':source_ok,'completion_subgoals':completed,'refusal_subgoals':refused,
        'intended_refusal_boundaries':[g.code for g in task.refusals],
        'observed_refusal_reasons':[[r['reference_code'] for r in report['service_decisions'] if r['classification']=='correct_refusal' and r['operation']==g.operation] for g in task.refusals],
        'frame_preserved':frame,'external_schedule_compliant':scheduled,'oracle_status':report['status'],'replay_verified':replay_ok,
        'actual_business_operations':len(report['operations']),'actual_customer_turns':sum(e.kind=='customer' and not e.payload.get('setup') for e in trace.events),
        'metrics':report['metrics']}
    # Do not change the serialized certificate of a legacy task just because
    # the composition protocol adds new diagnostic fields.
    if task.composition is not None:
        result.update(information_requirements=information,preference_requirements=preferences,
            confirmation_requirements=confirmations)
    if task.created_goals:
        result['created_completion_subgoals']=created_completed
    return result


def certify(adapter,world,task,trace):
    task=checked_task(adapter,world,task);result=score(adapter,world,task,trace)
    if not result['success']:raise ValueError('witness fails objective task, frame, Oracle or Replay: '+str(result))
    dependencies=certify_dependencies(adapter,trace)
    if task.require_dependency and not dependencies:raise ValueError('claimed dialogue task has no actual information/consent dependency')
    row=TaskCertificate(task=task,witness_trace_hash=trace.trace_hash,oracle=evaluate(adapter,trace),evaluation=result,dependencies=dependencies)
    row.certificate_hash=row.computed_hash();return row


def verify_certificate(adapter,world,certificate,trace):
    c=TaskCertificate.model_validate(certificate.model_dump(mode='json') if isinstance(certificate,TaskCertificate) else certificate)
    if c.certificate_hash!=c.computed_hash() or c.witness_trace_hash!=trace.trace_hash:raise ValueError('task certificate integrity/source differs')
    # Rebuild the entire canonical proof set from replayed source below. Equality
    # checks every intervention/state/decision, without replaying the same full
    # trace once again for each individual certificate.
    if c!=certify(adapter,world,c.task,trace):raise ValueError('task certificate does not reproduce')
    return True
