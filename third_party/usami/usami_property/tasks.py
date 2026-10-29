"""Native repair, scheduling, complaint and refusal goals over an unchanged world."""
from usami_dialogue.tasks import DialogueTask,Goal,FrameRule,RefusalGoal,checked_task
from .models import Operation
from .rules import ticket_key,complaint_key,cost_terms,contract
from .customer import private_state


def frames(world,private,*,refusal_only=False,inspection=False):
    aid=private.facts['asset_id'];tid=ticket_key(aid)
    rows=[FrameRule(collection='tickets',selector_field='asset_id',selector_value=aid,mutable_fields=['status','severity','assessment_version','version','approval_requested','inspection_requested','appointment_id','booked_slot_id','repair_notice','closure_notice'],allow_create=True),
        FrameRule(collection='approvals',selector_field='$key',selector_value=aid,allow_create=True)]
    if not refusal_only:rows += [FrameRule(collection='plans',selector_field='ticket_id',selector_value=tid,mutable_fields=['status'],allow_create=True),
        FrameRule(collection='appointments',selector_field='ticket_id',selector_value=tid,mutable_fields=['status'],allow_create=True),
        FrameRule(collection='complaints',selector_field='ticket_id',selector_value=tid,mutable_fields=['status','review_notice'],allow_create=True),
        FrameRule(collection='charges',selector_field='ticket_id',selector_value=tid,allow_create=True)]
    fields=['condition'] if not refusal_only else []
    if inspection:fields+=['simulated_hazard','revision','inspection_notice']
    if fields:rows.append(FrameRule(collection='assets',selector_value=aid,mutable_fields=fields))
    return rows


def compose(world,profile_id,*,family='repair',slot_ids=None,mixed=False,inspection=False,refusal_operation=None,refusal_code=None):
    from . import adapter
    private=private_state(world,profile_id);aid=private.facts['asset_id'];asset=world.assets[aid];unit=world.units[asset.unit_id];tid=ticket_key(aid)
    choices=list(slot_ids) if slot_ids is not None else [private.initial_preferences['slot_id']]
    if not choices or len(set(choices))!=len(choices) or set(choices)-set(private.preference_space['slot_id']):raise ValueError('slot choice outside frozen resident preferences')
    refusal=[];goals=[];pure=family.startswith('refuse_')
    if pure:
        if mixed or refusal_operation is None or refusal_code is None:raise ValueError('refusal needs exact native operation/reason')
        refusal=[RefusalGoal(operation=refusal_operation,code=refusal_code)]
    elif family in {'repair','schedule','complaint'}:
        if family=='repair':goals=[Goal(collection='tickets',entity=tid,field='status',accepted_values=['closed']),Goal(collection='assets',entity=aid,field='condition',accepted_values=['working'])]
        elif family=='schedule':goals=[Goal(collection='tickets',entity=tid,field='status',accepted_values=['scheduled'])]
        else:
            outcome=world.external_events['technician_visit_'+aid].outcome
            reason='missed_visit' if outcome=='no_show' else 'unresolved_repair'
            goals=[Goal(collection='complaints',entity=complaint_key(tid,reason,private.identity),field='status',accepted_values=['escalated'])]
        goals.append(Goal(collection='tickets',entity=tid,field='booked_slot_id',accepted_values=choices))
        if mixed:
            if family!='repair':raise ValueError('only completed repair has a general duplicate-closure refusal')
            operation=Operation(name='close_ticket',scope_id=unit.property_id,resource_id=unit.id,params={'ticket_id':tid})
            refusal=[RefusalGoal(operation=operation.model_dump(mode='json'),code='TICKET_ALREADY_CLOSED')]
    else:raise ValueError('unregistered property task family')
    scheduled=[]
    if not pure or family=='refuse_visit':
        cost,payer,_=cost_terms(world,asset,private.identity);c=contract(world,unit)
        if payer==unit.owner_id and private.identity!=unit.owner_id and cost>c.resident_approval_limit_cents:scheduled.append('owner_review_'+aid)
    if inspection:scheduled.append('inspection_'+aid)
    if family in {'repair','complaint'}:scheduled.append('technician_visit_'+aid)
    task=DialogueTask(domain='property',rules_version=world.version,world_hash=world.state_hash,profile_id=profile_id,private_hash=private.private_hash,
        family=family,goals=goals,refusals=refusal,frames=frames(world,private,refusal_only=pure,inspection=inspection),external_schedule=scheduled,require_dependency=family!='refuse_open')
    return checked_task(adapter,world,task)


def validate_task(world,private,task):
    aid=private.facts['asset_id'];tid=ticket_key(aid);unit=world.units[private.facts['unit_id']]
    pure=task.family in {'refuse_open','refuse_visit','refuse_close'};inspection='inspection_'+aid in task.external_schedule
    if task.family not in {'repair','schedule','complaint','refuse_open','refuse_visit','refuse_close'}:raise ValueError('unregistered property family')
    if task.frames!=frames(world,private,refusal_only=pure,inspection=inspection):raise ValueError('task frame exceeds native property scope')
    if any(world.external_events[e].asset_id!=aid for e in task.external_schedule if e in world.external_events):raise ValueError('external event outside this asset')
    if pure:
        if task.goals or len(task.refusals)!=1:raise ValueError('refusal requires one real native probe')
        r=task.refusals[0];op=Operation.model_validate(r.operation)
        expected={'refuse_open':'open_ticket','refuse_visit':'quote_visit','refuse_close':'close_ticket'}[task.family]
        from .customer import intent_allowed
        codes={'ASSET_ALREADY_WORKING','TECHNICIAN_SCOPE_MISMATCH','TECHNICIAN_SKILL_REQUIRED','SLOT_UNAVAILABLE','EMERGENCY_RESPONDER_REQUIRED','OUTSIDE_WORKING_HOURS','RESPONSE_DEADLINE_EXCEEDED','CALENDAR_CONFLICT','CUSTOMER_CONSTRAINT_UNMET','REPAIR_PROOF_REQUIRED'}
        if op.name!=expected or not intent_allowed_for_static(world,private,op) or r.code not in codes:raise ValueError('refusal target/code outside frozen property scope')
        if task.require_dependency!=(task.family!='refuse_open'):raise ValueError('refusal dependency declaration differs')
        return
    if not task.require_dependency:raise ValueError('core property task requires actual dialogue dependency')
    if not task.goals or task.goals[-1].collection!='tickets' or task.goals[-1].entity!=tid or task.goals[-1].field!='booked_slot_id':raise ValueError('task must bind the selected visit time')
    choices=task.goals[-1].accepted_values
    if not choices or len(set(choices))!=len(choices) or set(choices)-set(private.preference_space['slot_id']):raise ValueError('task slot outside frozen preference space')
    if task.family=='repair':prefix=[Goal(collection='tickets',entity=tid,field='status',accepted_values=['closed']),Goal(collection='assets',entity=aid,field='condition',accepted_values=['working'])]
    elif task.family=='schedule':prefix=[Goal(collection='tickets',entity=tid,field='status',accepted_values=['scheduled'])]
    else:
        event=world.external_events['technician_visit_'+aid];reason='missed_visit' if event.outcome=='no_show' else 'unresolved_repair'
        prefix=[Goal(collection='complaints',entity=complaint_key(tid,reason,private.identity),field='status',accepted_values=['escalated'])]
    if task.goals[:-1]!=prefix:raise ValueError('task business goal differs from native contract')
    if task.refusals:
        op=Operation(name='close_ticket',scope_id=unit.property_id,resource_id=unit.id,params={'ticket_id':tid})
        if task.family!='repair' or task.refusals!=[RefusalGoal(operation=op.model_dump(mode='json'),code='TICKET_ALREADY_CLOSED')]:raise ValueError('unregistered mixed refusal')


def intent_allowed_for_static(world,private,op):
    # The initial ticket need not exist yet; only source identifiers are resolved.
    aid=private.facts['asset_id'];unit=world.units[private.facts['unit_id']]
    if op.scope_id!=unit.property_id or op.resource_id!=unit.id:return False
    if op.name=='open_ticket':return op.params=={'asset_id':aid,'symptom':private.facts['symptom']}
    if op.params.get('ticket_id')!=ticket_key(aid):return False
    if op.name=='quote_visit':return op.params['slot_id'] in private.preference_space['slot_id'] and op.params['technician_id'] in world.technicians and world.technicians[op.params['technician_id']].property_id==unit.property_id
    return op.name=='close_ticket'


def search(world,task,*,max_nodes=1024,max_depth=12,max_paths=2):
    from . import adapter
    from usami_dialogue.planner import search as native_search
    private=private_state(world,task.profile_id)
    if task.family=='refuse_visit':choices=[task.refusals[0].operation['params']['slot_id']]
    elif task.family.startswith('refuse_'):choices=[private.initial_preferences['slot_id']]
    else:choices=task.goals[-1].accepted_values
    options=[{'slot_id':s,'feedback':'dissatisfied' if task.family=='complaint' else 'satisfied'} for s in choices]
    return native_search(adapter,world,task,options,max_nodes=max_nodes,max_depth=max_depth,max_paths=max_paths)


def certified_tasks(world,profile_id,*,max_nodes=1024,max_paths=2,**kwargs):
    from . import adapter
    from usami_dialogue.planner import certified_paths
    task=compose(world,profile_id,**kwargs);result=search(world,task,max_nodes=max_nodes,max_paths=max_paths)
    return task,result,certified_paths(adapter,world,task,result)


def customer_task_brief(world,private,task):
    import json
    prefix='Your unit is '+private.facts['unit_id']+' and the asset is '+private.facts['asset_id']+'. '
    if task.family.startswith('refuse_'):
        op=task.refusals[0].operation
        return prefix+'You want this specific service: '+json.dumps({'operation':op['name'],'parameters':{k:v for k,v in op['params'].items() if k!='ticket_id' or v in world.tickets}},sort_keys=True)+'. Discuss the real response without inventing a fault or changing the goal.'
    need={'repair':'Obtain a real repair and decide whether to confirm closure after actual work evidence.','schedule':'Arrange a suitable technician visit.','complaint':'Seek follow-up for the repair; if an actual visit is missed or fails to resolve the problem, file the corresponding complaint and request escalation.'}[task.family]
    text=prefix+need+' Your acceptable visit choices for this request are '+json.dumps(task.goals[-1].accepted_values)+'. Keep your actual access and budget constraints; clarify and compare real offers.'
    if task.refusals:text+=' You also want to request closure again after the ticket has already been closed.'
    return text


def task_intent_allowed(private,task,intent):
    if task.family=='refuse_visit':allowed=[task.refusals[0].operation['params']['slot_id']]
    elif task.family.startswith('refuse_'):allowed=private.preference_space['slot_id']
    else:allowed=task.goals[-1].accepted_values
    if 'slot_id' in intent.preferences and intent.preferences['slot_id'] not in allowed:return False
    for op in intent.requested_operations:
        if op.get('name')=='quote_visit' and op.get('params',{}).get('slot_id') not in allowed:return False
    return True
