"""Frozen profile task composition and bounded native Reference reachability.

This planner creates private certificates only. The production Service does not
receive its path and is scored on terminal goals, refusal claims and frame rules.
"""
from usami_dialogue.tasks import DialogueTask,Goal,FrameRule,RefusalGoal,checked_task,certify
from .models import Operation
from .rules import case_key
from .customer import private_state


TERMINAL={'refund':'refunded','exchange':'exchanged','reship':'reshipped'}


def frame_contract(world,private):
    line=world.lines[private.facts['line_id']];cid=case_key(line.order_id,line.id,private.facts['issue'])
    return [FrameRule(collection='lines',selector_value=line.id,mutable_fields=['status']),
        FrameRule(collection='products',selector_value=line.product_id,mutable_fields=['stock']),
        FrameRule(collection='cases',selector_field='line_id',selector_value=line.id,mutable_fields=['status','evidence_id','evidence_notice','review_requested','agreement_quote','agreement_revision'],allow_create=True),
        FrameRule(collection='quotes',selector_field='case_id',selector_value=cid,mutable_fields=['status'],allow_create=True),
        FrameRule(collection='returns',selector_field='case_id',selector_value=cid,mutable_fields=['status','receipt_notice'],allow_create=True),
        FrameRule(collection='refunds',selector_field='line_id',selector_value=line.id,allow_create=True),
        FrameRule(collection='replacements',selector_field='line_id',selector_value=line.id,allow_create=True),
        FrameRule(collection='approvals',selector_field='$key',selector_value=line.id,allow_create=True)]


def compose(world,profile_id,*,resolutions=None,mixed=False):
    from . import adapter
    private=private_state(world,profile_id);line=world.lines[private.facts['line_id']];order=world.orders[line.order_id]
    choices=list(resolutions) if resolutions is not None else list(private.preference_space['resolution'])
    if not choices or len(set(choices))!=len(choices) or set(choices)-set(private.preference_space['resolution']):raise ValueError('resolution outside frozen Customer choices')
    op=Operation(name='open_case',scope_id=order.merchant_id,resource_id=order.id,params={'line_id':line.id,'issue':private.facts['issue']})
    task=DialogueTask(domain=world.domain,rules_version=world.version,world_hash=world.state_hash,profile_id=profile_id,private_hash=private.private_hash,
        family='settlement_with_duplicate_refusal' if mixed else 'dispute_resolution',
        goals=[Goal(collection='lines',entity=line.id,field='status',accepted_values=[TERMINAL[c] for c in choices])],
        refusals=[RefusalGoal(operation=op.model_dump(mode='json'),code='ALREADY_RESOLVED')] if mixed else [],
        frames=frame_contract(world,private),external_schedule=[e.id for e in world.external_events.values() if e.line_id==line.id])
    return checked_task(adapter,world,task)


def validate_task(world,private,task):
    line=world.lines[private.facts['line_id']];order=world.orders[line.order_id]
    if len(task.goals)!=1 or task.goals[0].collection!='lines' or task.goals[0].entity!=line.id or task.goals[0].field!='status':raise ValueError('unsupported or out-of-scope aftersales goal')
    goal=task.goals[0]
    if not goal.accepted_values or set(goal.accepted_values)-{TERMINAL[c] for c in private.preference_space['resolution']}:raise ValueError('unregistered terminal outcome')
    if task.frames!=frame_contract(world,private):raise ValueError('task frame exceeds the native per-line contract')
    if len(task.refusals)>1:raise ValueError('unsupported refusal scope')
    if task.refusals:
        expected=Operation(name='open_case',scope_id=order.merchant_id,resource_id=order.id,params={'line_id':line.id,'issue':private.facts['issue']})
        if task.refusals[0].operation!=expected.model_dump(mode='json') or task.refusals[0].code!='ALREADY_RESOLVED':raise ValueError('refusal is not a real duplicate-resolution probe')
    if any(world.external_events[e].line_id!=line.id for e in task.external_schedule if e in world.external_events):raise ValueError('external event outside task line')


def search(world,task,*,max_nodes=1024,max_depth=12,max_paths=2):
    from . import adapter
    from usami_dialogue.planner import search as native_search
    private=private_state(world,task.profile_id)
    options=[{'resolution':option} for option in private.preference_space['resolution'] if TERMINAL[option] in task.goals[0].accepted_values]
    return native_search(adapter,world,task,options,max_nodes=max_nodes,max_depth=max_depth,max_paths=max_paths)


def execute_witness(world,task,path):
    from . import adapter
    from usami_dialogue.planner import execute_witness as native_execute
    return native_execute(adapter,world,task,path)


def certified_tasks(world,profile_id,*,resolutions=None,mixed=False,max_nodes=1024,max_paths=2):
    from . import adapter
    task=compose(world,profile_id,resolutions=resolutions,mixed=mixed)
    result=search(world,task,max_nodes=max_nodes,max_paths=max_paths)
    certificates=[]
    for path in result['paths']:
        trace=execute_witness(world,task,path)
        certificates.append({'trace':trace,'certificate':certify(adapter,world,task,trace)})
    return task,result,certificates


def customer_task_brief(world,private,task):
    # A user goal, not the hidden Reference answer, witness or refusal reason.
    options={'refunded':'refund','exchanged':'exchange','reshipped':'replacement shipment'}
    result='Seek help for order '+private.facts['order_id']+', item '+private.facts['line_id']+'. Resolve the actual issue with one of: '+', '.join(options[v] for v in task.goals[0].accepted_values)+'. Compare real offers against your frozen needs; provide facts when appropriate.'
    if task.refusals:result+=' You also want to request a further resolution for this same item after it has been resolved. Discuss the actual Service response.'
    return result


def task_intent_allowed(private,task,intent):
    values={'refunded':'refund','exchanged':'exchange','reshipped':'reship'}
    allowed={values[v] for v in task.goals[0].accepted_values}
    if 'resolution' in intent.preferences and intent.preferences['resolution'] not in allowed:return False
    return all(op.get('name')!='quote_resolution' or op.get('params',{}).get('option') in allowed for op in intent.requested_operations)
