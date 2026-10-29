"""Static guards for open, frozen-world dialogue task compositions.

The Designer may choose several related Customer profiles and combine goals.
This module only checks that the declaration is grounded in those profiles and
native task contracts. Native Reference search, Engine, Oracle and Replay still
prove that a selected composition is executable and valid.
"""

import importlib
import re
from usami_harness.goals import field_value
from usami_harness.models import digest
from .scope import load_profiles
from .tasks import DialogueComposition


_PRIVATE_FIELDS={'notice','gateway_notice','signature','attestation','private_hash','quote_hash',
                'simulated_symptom','simulated_hazard','inspection_notice','repair_notice',
                'closure_notice','receipt_notice','evidence_notice','review_notice','attendance_notice'}


def _contract(adapter,world,private):
    tasks=importlib.import_module('usami_'+adapter.name+'.tasks')
    if adapter.name=='aftersales':return tasks.frame_contract(world,private)
    if adapter.name=='airline':return tasks.frames(world,private)
    if adapter.name=='property':return tasks.frames(world,private,inspection=True)
    if adapter.name=='education':return tasks.frames(world,private)
    raise ValueError('open composition requires a registered dialogue domain')


def _ids(value):
    if isinstance(value,str):yield value
    elif isinstance(value,dict):
        for item in value.values():yield from _ids(item)
    elif isinstance(value,(list,tuple)):
        for item in value:yield from _ids(item)


def _scope_ids(adapter,world,private,contract):
    collections={k:v for k,v in world.model_dump(mode='json').items() if isinstance(v,dict)}
    known={key for rows in collections.values() for key,obj in rows.items() if isinstance(obj,dict) and obj.get('id')==key}
    result=set(private.protected_objects)|{v for v in _ids(private.facts) if v in known}
    result.add(private.identity)
    for rule in contract:
        if isinstance(rule.selector_value,str) and rule.selector_value in known:result.add(rule.selector_value)
        elif rule.selector_field=='$key':result.add(str(rule.selector_value))
    for raw in adapter.candidate_intents(world,private):
        result.update(v for v in _ids(raw) if v in known)
    return result


def _matches(rule,goal,world):
    if rule.collection!=goal.collection:return False
    if rule.selector_field=='$key':return rule.selector_value==goal.entity
    row=getattr(world,goal.collection).get(goal.entity)
    if row is None:return False
    try:return field_value(row,rule.selector_field)==rule.selector_value
    except ValueError:return False


def _rule_key(rule):
    return (rule.collection,rule.selector_field,rule.selector_value,rule.allow_create)


def _collection_fields(adapter,collection):
    schema=adapter.World.model_json_schema()
    raw=schema.get('properties',{}).get(collection,{})
    entity=(raw.get('additionalProperties') or {}).get('$ref','').rsplit('/',1)[-1]
    definition=schema.get('$defs',{}).get(entity,{})
    return definition.get('properties',{})


def _scalar(value):
    return value is None or type(value) in {str,int,float,bool}


def _add_grounded_values(values,raw):
    if isinstance(raw,dict):
        for key,value in raw.items():
            if _scalar(value):values.setdefault(key,[]).append(value)
            _add_grounded_values(values,value)
    elif isinstance(raw,(list,tuple)):
        for value in raw:_add_grounded_values(values,value)


def _created_goal_matches(goal,row):
    try:return all(_scalar(value) and field_value(row,key)==value and type(field_value(row,key)) is type(value)
                   for key,value in goal.selectors.items())
    except ValueError:return False


def validate_composed_task(adapter,world,task):
    composition=DialogueComposition.model_validate(task.composition.model_dump(mode='json'))
    if task.family!='open_composition_v1':raise ValueError('composition protocol requires its registered family')
    expected_dependency=(composition.protocol=='open-dialogue-composition-v1' or bool(task.goals or task.created_goals))
    if task.require_dependency!=expected_dependency:
        raise ValueError('composed task dependency policy differs from its registered protocol')
    profile_rows=load_profiles(adapter,world,task.profile_id,composition.profile_ids)
    if not task.goals and not task.created_goals and not task.refusals:raise ValueError('composed task needs at least one completion or refusal objective')
    if len(task.goals)>32 or len(task.created_goals)>32 or len(task.refusals)>16:raise ValueError('composed task exceeds objective limits')
    if task.created_goals and composition.protocol!='open-dialogue-composition-v3':
        raise ValueError('created-object goals require open-dialogue-composition-v3')
    if composition.protocol=='open-dialogue-composition-v3' and not task.created_goals:
        raise ValueError('open-dialogue-composition-v3 requires at least one created-object goal')

    contracts={pid:_contract(adapter,world,private) for pid,private in profile_rows}
    scope_by_profile={pid:_scope_ids(adapter,world,private,contracts[pid]) for pid,private in profile_rows}
    scope=set().union(*scope_by_profile.values())
    if task.external_schedule and (len(set(task.external_schedule))!=len(task.external_schedule) or set(task.external_schedule)-set(world.external_events)):
        raise ValueError('composed task references an unregistered external event')

    native_rules={}
    for pid,private in profile_rows:
        for rule in contracts[pid]:native_rules.setdefault(_rule_key(rule),[]).append(rule)
    if not task.frames:raise ValueError('composed task must declare its allowed state-change frame')
    for rule in task.frames:
        candidates=native_rules.get(_rule_key(rule),[])
        if not candidates or not any(set(rule.mutable_fields)<=set(base.mutable_fields) and
                                     (not rule.allow_create or base.allow_create) for base in candidates):
            raise ValueError('composed frame exceeds the native Customer object contract')

    for goal in task.goals:
        collection=getattr(world,goal.collection,None)
        if not isinstance(collection,dict) or goal.entity not in scope:
            raise ValueError('completion target is outside selected frozen Customer objects')
        row=collection.get(goal.entity)
        if row is None:raise ValueError('completion targets must name an existing frozen business object')
        parts=goal.field.split('.')
        if not goal.field or set(parts)&_PRIVATE_FIELDS:raise ValueError('completion cannot inspect private or evaluator-only fields')
        try:field_value(row,goal.field)
        except ValueError as exc:raise ValueError('completion field is not part of the frozen business state') from exc
        if not any(_matches(rule,goal,world) and set(parts)<=set(rule.mutable_fields) for rule in task.frames):
            raise ValueError('completion target is not covered by its declared native frame')

    seen_created=set()
    for goal in task.created_goals:
        if goal.profile_id not in scope_by_profile:
            raise ValueError('created-object goal must be anchored to a selected Customer profile')
        if not goal.selectors or any(not _scalar(value) for value in goal.selectors.values()):
            raise ValueError('created-object selectors must be nonempty grounded scalar values')
        if any(not key or key=='id' or key.startswith('_') or key in _PRIVATE_FIELDS for key in goal.selectors):
            raise ValueError('created-object selectors cannot use generated IDs or private fields')
        if not goal.field or goal.field=='id' or set(goal.field.split('.'))&_PRIVATE_FIELDS:
            raise ValueError('created-object completion cannot inspect private or generated-ID fields')
        field_schema=_collection_fields(adapter,goal.collection)
        if goal.field not in field_schema:
            raise ValueError('created-object target field is not part of the native business object schema')
        if goal.profile_id not in contracts:
            raise ValueError('created-object goal profile is not selected')
        private=next(p for pid,p in profile_rows if pid==goal.profile_id)
        grounded={}
        _add_grounded_values(grounded,private.facts)
        _add_grounded_values(grounded,private.preference_space)
        _add_grounded_values(grounded,private.initial_preferences)
        _add_grounded_values(grounded,adapter.candidate_intents(world,private))
        for collection,rows in world.model_dump(mode='json').items():
            if not isinstance(rows,dict):continue
            for key,row in rows.items():
                if key in scope_by_profile[goal.profile_id] or (isinstance(row,dict) and row.get('id') in scope_by_profile[goal.profile_id]):
                    _add_grounded_values(grounded,row)
        for key,value in goal.selectors.items():
            if key not in field_schema:
                raise ValueError(f'created-object selector {goal.collection}.{key} is not a native field')
            matches=[candidate for candidate in grounded.get(key,[]) if type(candidate) is type(value) and candidate==value]
            if not matches:
                raise ValueError(f'created-object selector {goal.collection}.{key}={value!r} is not grounded in the selected Customer profile')

        candidates=[rule for rule in contracts[goal.profile_id]
                    if rule.collection==goal.collection and rule.allow_create
                    and goal.field in rule.mutable_fields
                    and any(key==rule.selector_field and value==rule.selector_value
                            for key,value in goal.selectors.items())]
        candidates=[rule for rule in candidates if all(key not in rule.mutable_fields for key in goal.selectors)]
        if not candidates:
            raise ValueError('created-object goal is not covered by a native create-enabled Customer frame')
        if not any(any(_rule_key(rule)==_rule_key(base) and set(rule.mutable_fields)<=set(base.mutable_fields)
                       and rule.allow_create for rule in task.frames) for base in candidates):
            raise ValueError('created-object goal exceeds the task native create-enabled frame')
        collection=getattr(world,goal.collection,None)
        if not isinstance(collection,dict):
            raise ValueError('created-object target must use a registered native collection')
        if any(_created_goal_matches(goal,row) for row in collection.values()):
            raise ValueError('created-object goal must not match an object already present in the frozen world')
        signature=digest(goal.model_dump(mode='json'))
        if signature in seen_created:raise ValueError('duplicate created-object goal')
        seen_created.add(signature)
        enum=field_schema[goal.field].get('enum')
        if enum is not None and any(value not in enum for value in goal.accepted_values):
            raise ValueError(f'created-object values for {goal.collection}.{goal.field} are outside the native vocabulary')

    for refusal in task.refusals:
        op=adapter.Operation.model_validate(refusal.operation)
        if op.is_query or op.model_dump(mode='json')!=refusal.operation:
            raise ValueError('refusal objective must be one exact canonical business write')
        if not any(op.resource_id in scope_by_profile[pid] and adapter.intent_allowed(world,private,refusal.operation)
                   for pid,private in profile_rows):
            raise ValueError('refusal request lies outside selected frozen Customer authority')

    if composition.protocol=='open-dialogue-composition-v4':
        followups=composition.follow_up_requests
        if len(followups)!=len(task.refusals):
            raise ValueError('v4 requires one public Customer follow-up for each refusal objective')
        declared=[row.requested_action for row in followups]
        expected=[adapter.Operation.model_validate(row.operation).name for row in task.refusals]
        if any(not action for action in declared) or sorted(declared)!=sorted(expected):
            raise ValueError('v4 follow-up action metadata must match the registered refusal operations')
        for refusal in task.refusals:
            operation=adapter.Operation.model_validate(refusal.operation).model_dump(mode='json')
            if not any(operation in [adapter.Operation.model_validate(raw).model_dump(mode='json')
                                     for raw in adapter.candidate_intents(world, private)]
                       for _,private in profile_rows if adapter.intent_allowed(world,private,operation)):
                raise ValueError('v4 refusal request is not an action the selected Customer can explicitly request')
    elif composition.follow_up_requests:
        raise ValueError('Customer follow-up requests require open-dialogue-composition-v4')

    from usami_benchmark.public_policy import catalogue
    if not catalogue(adapter.name,world.version)['errors']:
        raise ValueError('native Reference exposes no registered refusal vocabulary')

    for requirement in composition.information_requirements:
        if requirement.profile_id not in scope_by_profile:
            raise ValueError('information requirement uses an unselected Customer profile')
        private=next(p for pid,p in profile_rows if pid==requirement.profile_id)
        if len(set(requirement.fact_keys))!=len(requirement.fact_keys) or set(requirement.fact_keys)-set(private.facts):
            raise ValueError('information requirement is not a fact in its frozen Customer profile')

    if composition.protocol=='open-dialogue-composition-v4':
        required_by_profile={row.profile_id:set(row.fact_keys) for row in composition.information_requirements}
        resolution_by_profile={}
        for goal in task.goals:
            if goal.collection=='lines' and goal.field=='status':
                terminal_to_option={'refunded':'refund','exchanged':'exchange','reshipped':'reship'}
                for profile_id,private in profile_rows:
                    if goal.entity==private.facts.get('line_id'):
                        resolution_by_profile.setdefault(profile_id,set()).update(
                            terminal_to_option[value] for value in goal.accepted_values if value in terminal_to_option)
        for profile_id,values in resolution_by_profile.items():
            if values & {'exchange','reship'} and 'deadline' not in required_by_profile.get(profile_id,set()):
                raise ValueError('v4 delivery-remedy tasks must register the Customer deadline information requirement')

    seen_preferences=set()
    for requirement in composition.preference_requirements:
        key=(requirement.profile_id,requirement.preference)
        if requirement.profile_id not in scope_by_profile or key in seen_preferences:
            raise ValueError('preference requirement must identify one selected profile/key exactly once')
        seen_preferences.add(key)
        private=next(p for pid,p in profile_rows if pid==requirement.profile_id)
        allowed=private.preference_space.get(requirement.preference)
        if not allowed or len(set(map(repr,requirement.accepted_values)))!=len(requirement.accepted_values) or set(map(repr,requirement.accepted_values))-set(map(repr,allowed)):
            raise ValueError('preference requirement is outside the frozen Customer choices')

    for raw in composition.confirmation_requirements:
        op=adapter.Operation.model_validate(raw)
        if op.is_query or op.model_dump(mode='json')!=raw:
            raise ValueError('confirmation requirement must be an exact canonical business write')
        if not any(op.name in private.confirmable_actions and op.resource_id in private.protected_objects
                   and adapter.intent_allowed(world,private,raw) for _,private in profile_rows):
            raise ValueError('confirmation requirement is outside selected Customer authority')

    if task.external_schedule:
        for event_id in task.external_schedule:
            event=world.external_events[event_id]
            if not any(value in scope for value in _ids(event.model_dump(mode='json'))):
                raise ValueError('external event is outside the selected Customer objects')

    # The natural-language request may describe the selected scenario, but must
    # not leak private evaluator concepts, native witnesses or Reference labels.
    request=composition.customer_request
    if re.search(r'(?i)\b(?:oracle|reference answer|expected answer|witness|hidden answer)\b|<\s*(?:system|developer|vast_)',request):
        raise ValueError('Customer request contains evaluator-only instructions')
    all_ids={key for collection in world.model_dump(mode='json').values() if isinstance(collection,dict)
             for key,row in collection.items() if isinstance(row,dict) and row.get('id')==key}
    out_of_scope={value for value in all_ids-scope if re.search(r'(?<![\w-])'+re.escape(value)+r'(?![\w-])',request)}
    if out_of_scope:raise ValueError('Customer request mentions identifiers outside the selected frozen profile scope')
    for followup in composition.follow_up_requests:
        if re.search(r'(?i)\b(?:oracle|reference answer|expected answer|witness|hidden answer)\b|<\s*(?:system|developer|vast_)',followup.message):
            raise ValueError('Customer follow-up contains evaluator-only instructions')
        followup_out_of_scope={value for value in all_ids-scope
            if re.search(r'(?<![\w-])'+re.escape(value)+r'(?![\w-])',followup.message)}
        if followup_out_of_scope:
            raise ValueError('Customer follow-up mentions identifiers outside the selected frozen profile scope')
    return task
