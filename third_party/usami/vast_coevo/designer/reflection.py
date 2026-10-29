"""Designer learning concerns test choice and attribution, not business execution."""
import copy
import json
from vast_coevo.models import ModelCall
from vast_coevo.skills.models import SkillPatch
from vast_coevo.reflection.scope import DomainPolicy, resolve_scope

OBJECTIVE = (
    'You are the Designer Skill Reflector. Improve how to select the NEXT informative test, not scene verbosity. '
    'Learn from real verified failures: isolate suspected triggers, remove irrelevant conditions and test legal expansions. '
    'Learn from correct cases: formulate unverified weakness hypotheses and flip a few comparable prerequisites. '
    'Use actual construction, trigger, Oracle and reproduction feedback; separate fingerprints from confirmed rule roots. '
    'Prefer informative tests within cost limits and avoid repeated tests without a new purpose. '
    'Never claim a hypothesis or an unused challenge is a failure. Customer policy/skills remain fixed. '
    'Do not modify business rules, tools, Engine, Oracle or permissions. Skills are declarative, no code or specific IDs. '
    'Return one submit_designer_patch tool call (or NOOP). Role designer, skill type design. '
    'ADD/UPDATE include id, version, trigger, selection, transformations, interpretation, stop. '
    'Scope metadata is program-managed: omit domains/hash; ADD starts at1, UPDATE increments once. '
    'Cite only provided trial IDs. Frozen base weights; no retries.'
)


def _patch_arguments(response):
    """Validate the single submission without depending on a business runtime."""
    try:
        choices = response['choices']
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError
        choice = choices[0]
        message = choice['message']
        calls = message['tool_calls']
        if (choice.get('finish_reason') not in {'stop', 'tool_calls'} or
                message.get('role') != 'assistant' or
                (message.get('content') is not None and not isinstance(message['content'], str)) or
                not isinstance(calls, list) or len(calls) != 1):
            raise ValueError
        call = calls[0]
        function = call['function']
        if (call.get('type') != 'function' or not isinstance(call.get('id'), str) or not call['id'] or
                function.get('name') != 'submit_designer_patch' or
                not isinstance(function.get('arguments'), str)):
            raise ValueError
    except (KeyError, TypeError, ValueError, AttributeError):
        raise ValueError('one complete submit_designer_patch call required') from None

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON field: ' + key)
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError('non-finite JSON value: ' + value)

    data = json.loads(function['arguments'], object_pairs_hook=unique, parse_constant=nonfinite)
    if not isinstance(data, dict):
        raise ValueError('Designer patch must be a JSON object')
    return data


def reflect(client, model, library, feedback, prefix, objective=OBJECTIVE, domain_policy=None):
    schema=SkillPatch.model_json_schema();schema['properties']['role']['const']='designer'
    for name in ('DesignerSkill',):
        schema['$defs'][name]['required']=[k for k in schema['$defs'][name]['required'] if k!='scope']
    scope=schema['$defs']['Scope'];scope['required']=[k for k in scope.get('required',[]) if k!='domains']
    request={**model.options,'model':model.model,'messages':[{'role':'system','content':objective},
             {'role':'user','content':json.dumps({'library':library.model_dump(mode='json'),'verified_feedback':feedback},ensure_ascii=False,sort_keys=True)}],
             'tools':[{'type':'function','function':{'name':'submit_designer_patch','description':'One Designer Skill update.', 'parameters':schema}}],
             'tool_choice':'auto','n':1,'stream':False}
    fields=dict(id=prefix+'.designer_reflector',role='designer_reflector',model=model,base_request=request,request=request)
    try:response=client.complete(copy.deepcopy(request))
    except Exception as exc:
        return {'call':ModelCall(**fields,error={'type':type(exc).__name__,'message':str(exc)}).model_dump(mode='json'),'patch':None,'error':'reflection_transport_error'}
    call=ModelCall(**fields,response=response)
    try:
        raw=_patch_arguments(response)
        data,_=resolve_scope(raw,library,domain_policy or DomainPolicy(add_domains=('cloudops',)))
        patch=SkillPatch.model_validate(data)
        if patch.role!='designer' or patch.base_snapshot_hash!=library.snapshot_hash:raise ValueError('wrong Designer parent or role')
        if set(patch.evidence_ids)-{f['id'] for f in feedback}:raise ValueError('unavailable evidence citation')
        return {'call':call.model_dump(mode='json'),'patch':patch.model_dump(mode='json'),'error':None}
    except (ValueError,TypeError,KeyError) as exc:
        return {'call':call.model_dump(mode='json'),'patch':None,'error':str(exc)}
