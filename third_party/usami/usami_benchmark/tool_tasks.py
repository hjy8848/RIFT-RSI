"""Task-only native proof adapter for the four existing business runtimes.

Worlds are loaded from independently sealed registrations. Native Rules, actual
Engine, Oracle and Replay remain authoritative; no conditioned make_task is used.
"""
import importlib
from collections import deque
from typing import Literal,Optional
from types import SimpleNamespace
from pydantic import Field
from usami_harness.models import Model,digest
from usami_harness.goals import JointGoal
from usami_benchmark.worlds import FrozenWorld
from usami_benchmark.splits import verified_world

TOOL_DOMAINS=('saas','cloudops','fulfillment','stayops')


def binding(domain,world=None):
    from integrations.usami.examiner import binding as original,goal_adapter
    if domain not in TOOL_DOMAINS:raise ValueError('not a tool-centric domain')
    b=original(domain)
    if domain=='stayops' and world is not None and getattr(world,'version',None)=='stayops-0.2.0':
        from usami_benchmark.native import stayops_v2 as adapter
        from usami_harness.goals import execute
        b=SimpleNamespace(World=adapter.World,Operation=adapter.Operation,invariants=adapter.invariants,decide=adapter.decide,execute=lambda w,i,p:execute(adapter,w,i,p))
    return b,goal_adapter(domain,world)


def load_world(frozen):
    domain=frozen.spec.domain
    if domain=='stayops' and frozen.spec.rules_version=='stayops-0.2.0':
        from usami_benchmark.native.stayops_v2 import World
        return World.model_validate(frozen.world)
    b,_=binding(domain);return b.World.model_validate(frozen.world)


CREATE_IDS={
 'saas':{},
 'cloudops':{'create_instance':{'resource_id'}},
 'fulfillment':{'create_order':{'order_id','id'},'receive_inventory':{'lot_id'},'dispatch_transfer':{'transfer_id','target_lot_id'},'reserve_inventory':{'reservation_id'},'create_shipment':{'shipment_id'},'request_return':{'return_id'}},
 'stayops':{'create_hold':{'reservation_id'}},
}


def visible_ids(domain,world,identity):
    if domain=='saas':
        from usami.customer.observation import initial_observation
        view=initial_observation(world,identity)
    else:
        module=importlib.import_module('usami_'+domain+'.observations')
        view=module.observation(world,identity)
    # Include nested order-line IDs and public reference/catalog IDs, not just
    # top-level rows. No private world collection is traversed here.
    return {value for key,value in _id_fields(view)}|{identity}


def _id_fields(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if (key=='id' or key.endswith('_id')) and isinstance(item,str):yield key,item
            elif key.endswith('_ids') and isinstance(item,list):
                for v in item:
                    if isinstance(v,str):yield key,v
            else:
                for pair in _id_fields(item):yield pair
    elif isinstance(value,list):
        for item in value:
            for pair in _id_fields(item):yield pair


def _validate_scope(domain,world,identity,task):
    permitted=visible_ids(domain,world,identity)
    all_ids={key for values in world.model_dump(mode='json').values() if isinstance(values,dict) for key,value in values.items() if isinstance(value,dict) and value.get('id')==key}
    created=set()
    for raw in task.capabilities:
        creates=CREATE_IDS[domain].get(raw['name'],set())
        created.update(value for key,value in _id_fields(raw) if key in creates and value not in all_ids)
    for raw in task.capabilities:
        if any(value not in permitted|created for key,value in _id_fields(raw)):raise ValueError('capability references a hidden or unregistered entity')
    if task.goal:
        for atom in task.goal.atoms:
            if atom.entity in all_ids and atom.entity not in permitted:raise ValueError('goal references an invisible entity')
            if any(part in {'notice','gateway_notice','signature','attestation'} for part in atom.field.split('.')):raise ValueError('goal requests evaluator attestation material')
            if atom.field in {'payments','refunds'}:raise ValueError('financial goals need public business fields, not full proof records')
        for key in task.goal.allowed_changes:
            entity=key.split('/',1)[1]
            if entity in all_ids and entity not in permitted:raise ValueError('frame refers to hidden state')
    # Requests cannot smuggle hidden object IDs, certificates or evaluator commands.
    import re
    if any(t in task.message.casefold() for t in ('witness','reference answer','<vast_','<system','<developer')) or task.world_hash in task.message or task.registration_hash in task.message:raise ValueError('private evaluator material in business request')
    if any(re.search(r'(?<!\w)'+re.escape(key)+r'(?!\w)',task.message) for key in all_ids-permitted-created):raise ValueError('business message exposes hidden entity')


def refusal_proof(domain,world,identity,raw):
    b,_=binding(domain,world);op=b.Operation.model_validate(raw);decision=b.decide(world,identity,op)
    if decision.allowed:raise ValueError('refusal proof requested for allowed action')
    if domain=='saas':
        from usami.world.engine import WorldEngine,replay_trace
        from usami.protocol import make_request
        from usami.evaluation.oracle import evaluate_trace
        engine=WorldEngine(world,1);req=make_request('refusal-proof',identity,op);engine.register_request(req)
        event=engine.execute(identity,op,req.request_id);trace=engine.finish()
        report=evaluate_trace(trace,{req.request_id:req}).model_dump(mode='json')
        verified=replay_trace(trace).state==world
        requests={req.request_id:req.model_dump(mode='json')}
    elif domain=='cloudops':
        from usami_cloudops.engine import Engine,replay_trace
        from usami_cloudops.models import Request
        from usami_cloudops.oracle import evaluate
        engine=Engine(world,1);req=Request(id='refusal-proof',identity=identity,operation=op);engine.register(req)
        event=engine.execute(identity,op,req.id);trace=engine.finish();report=evaluate(trace,engine.requests)
        verified=replay_trace(trace,engine.requests).state==world
        requests={req.id:req.model_dump(mode='json')}
    else:
        from usami_harness.engine import Engine,replay_trace
        from usami_harness.models import Request
        from usami_harness.oracle import evaluate
        adapter=importlib.import_module('usami_benchmark.native.stayops_v2' if domain=='stayops' and world.version=='stayops-0.2.0' else 'usami_'+domain+'.adapter')
        engine=Engine(adapter,world,1);req=Request(id='refusal-proof',identity=identity,operation=op.model_dump(mode='json'));engine.register(req)
        event=engine.execute(identity,op,req.id);trace=engine.finish();report=evaluate(adapter,trace,engine.requests)
        verified=replay_trace(adapter,trace,engine.requests).state==world
        requests={req.id:req.model_dump(mode='json')}
    result=event.payload['result']
    if result['success'] or result['error_code']!=decision.error_code or report['status']!='pass' or not verified:raise ValueError('native refusal Engine/Oracle/Replay disagreement')
    return {'operation':op.model_dump(mode='json'),'reference_code':decision.error_code,'requests':requests,'trace':trace.model_dump(mode='json'),'oracle':report,'replay_verified':verified}


class ToolTask(Model):
    protocol:Literal['native-tool-task-v1']='native-tool-task-v1'
    domain:Literal['saas','cloudops','fulfillment','stayops']
    registration_hash:str=Field(pattern=r'^[0-9a-f]{64}$')
    world_hash:str=Field(pattern=r'^[0-9a-f]{64}$')
    identity:str
    family:str=Field(min_length=1)
    message:str=Field(min_length=1,max_length=6000)
    goal:Optional[JointGoal]=None
    refusal_operations:list[dict]=Field(default_factory=list,max_length=4)
    capabilities:list[dict]=Field(min_length=1,max_length=16)
    task_hash:str=''
    def computed_hash(self):return digest(self.model_dump(mode='json',exclude={'task_hash'}))


class ToolCertificate(Model):
    protocol:Literal['native-tool-certificate-v1']='native-tool-certificate-v1'
    task:ToolTask
    max_depth:int=Field(ge=1,le=16,strict=True)
    max_nodes:int=Field(ge=1,le=20000,strict=True)
    max_paths:int=Field(ge=1,le=8,strict=True)
    search_nodes:int=Field(ge=1,strict=True)
    search_exhausted:bool
    witnesses:list[dict]
    certificate_hash:str=''
    def computed_hash(self):return digest(self.model_dump(mode='json',exclude={'certificate_hash'}))


def checked_task(frozen,task):
    frozen=FrozenWorld.model_validate(frozen.model_dump(mode='json') if isinstance(frozen,FrozenWorld) else frozen).checked()
    world=load_world(frozen);b,fields=binding(frozen.spec.domain,world)
    task=ToolTask.model_validate(task.model_dump(mode='json') if isinstance(task,ToolTask) else task)
    if task.domain!=frozen.spec.domain or task.world_hash!=frozen.world_hash or task.registration_hash!=frozen.registration_hash or task.identity not in world.users:raise ValueError('task differs from sealed world/identity')
    if b.invariants(world):raise ValueError('invalid registered initial state')
    ops=[b.Operation.model_validate(raw) for raw in task.capabilities]
    if any(op.is_query for op in ops) or len({digest(op) for op in ops})!=len(ops):raise ValueError('exact distinct business capabilities required')
    if [op.model_dump(mode='json') for op in ops]!=task.capabilities:raise ValueError('capability parameters must be canonical')
    if task.goal is None and not task.refusal_operations:raise ValueError('objective goal or policy refusal required')
    if task.goal is None and task.capabilities!=task.refusal_operations:raise ValueError('pure refusal must contain only its exact requested probes')
    if task.goal is not None:
        task.goal.validate_for(fields,world)
        if task.goal.satisfied(world):raise ValueError('completion goal already satisfied')
    if len({digest(o) for o in task.refusal_operations})!=len(task.refusal_operations):raise ValueError('duplicate refusal request')
    if any(o not in task.capabilities for o in task.refusal_operations):raise ValueError('refusal request outside frozen capabilities')
    _validate_scope(task.domain,world,task.identity,task)
    value=task.computed_hash()
    if task.task_hash and task.task_hash!=value:raise ValueError('task hash differs')
    return task.model_copy(update={'task_hash':value},deep=True)


def expected_goal(domain,world,after):
    """Derive a default one-transition business goal only from a real decision."""
    _,fields=binding(domain,world);left=world.model_dump(mode='json');right=after.model_dump(mode='json');atoms=[];changes={}
    for collection,allowed in fields.GOAL_FIELDS.items():
        for key in sorted(set(left[collection])|set(right[collection])):
            a=left[collection].get(key);b=right[collection].get(key)
            if a==b:continue
            if b is None:
                atoms.append({'collection':collection,'entity':key,'comparison':'absent'});changes[collection+'/'+key]=['*'];continue
            differing=[field for field in allowed if a is None or a.get(field)!=b.get(field)]
            if not differing:continue
            changes[collection+'/'+key]=['*'] if a is None else differing
            for field in differing:
                value=b.get(field)
                if not isinstance(value,(list,dict)) and value is not None:atoms.append({'collection':collection,'entity':key,'field':field,'value':value})
                elif isinstance(value,list) and value and isinstance(value[-1],dict) and value[-1].get('outcome') is not None:
                    atoms.append({'collection':collection,'entity':key,'field':field+'.'+str(len(value)-1)+'.outcome','value':value[-1]['outcome']})
    if not atoms:raise ValueError('operation has no supported nontrivial business goal')
    return JointGoal(atoms=atoms,allowed_changes=changes)


def _refusals(b,state,task):
    results=[]
    for raw in task.refusal_operations:
        decision=b.decide(state,task.identity,b.Operation.model_validate(raw));code=decision.error_code
        if decision.allowed or not code or code=='INVALID_WORLD' or code.endswith('NOT_FOUND') or code.startswith('UNKNOWN_'):return None
        results.append({'operation':raw,'reference_code':code})
    return results


def _causal(b,world,task,path):
    prefixes=[];state=world.clone()
    for op in path:prefixes.append(state);state=b.decide(state,task.identity,op).expected_state
    omissions=[];edges=[]
    for omitted in range(len(path)):
        other=prefixes[omitted].clone();first=None
        for later in range(omitted+1,len(path)):
            d=b.decide(other,task.identity,path[later])
            if d.allowed:other=d.expected_state
            elif first is None:first={'from':omitted,'to':later,'code':d.error_code};edges.append(first)
        reached=(task.goal is None or task.goal.satisfied(other)) and (task.goal is None or task.goal.frame_preserved(world,other)) and _refusals(b,other,task) is not None
        omissions.append({'step':omitted,'first_failure':first,'objective_after_remaining_legal_steps':reached})
    levels=[0]*len(path)
    for e in sorted(edges,key=lambda x:(x['to'],x['from'])):levels[e['to']]=max(levels[e['to']],levels[e['from']]+1)
    return {'direct_edges':edges,'omissions':omissions,'causal_depth':max(levels,default=0)}


def certify(frozen,task,*,max_depth=8,max_nodes=512,max_paths=2):
    # Validate generation independently, not just a mutable in-memory world hash.
    if type(max_paths) is not int or not 1<=max_paths<=8 or type(max_depth) is not int or not 1<=max_depth<=16 or type(max_nodes) is not int or not 1<=max_nodes<=20000:raise ValueError('bounded proof budget required')
    frozen=verified_world(frozen);task=checked_task(frozen,task);world=load_world(frozen);b,_=binding(task.domain,world)
    queue=deque([(world,[],frozenset())]);seen=set();paths=[];nodes=0
    operations=[b.Operation.model_validate(o) for o in task.capabilities]
    while queue and nodes<max_nodes and len(paths)<max_paths:
        state,path,used=queue.popleft();key=(state.state_hash,used)
        if key in seen:continue
        seen.add(key);nodes+=1
        goal=task.goal is None or task.goal.satisfied(state) and task.goal.frame_preserved(world,state)
        refusals=_refusals(b,state,task)
        if goal and refusals is not None:
            causal=_causal(b,world,task,path)
            if path and any(x['objective_after_remaining_legal_steps'] for x in causal['omissions']):continue
            trace=b.execute(world,task.identity,path) if path else None
            final=state if trace is None else b.World.model_validate(trace.final_state.model_dump(mode='json') if hasattr(trace.final_state,'model_dump') else trace.final_state)
            if final!=state:raise ValueError('actual native execution differs from Reference search')
            paths.append({'operations':[o.model_dump(mode='json') for o in path],'trace':trace.model_dump(mode='json') if trace is not None else None,'final_state_hash':state.state_hash,'refusals':[refusal_proof(task.domain,state,task.identity,r['operation']) for r in refusals],**causal});continue
        if len(path)>=max_depth:continue
        for i,op in enumerate(operations):
            if i in used:continue
            d=b.decide(state,task.identity,op)
            if d.allowed and d.expected_state!=state and not b.invariants(d.expected_state):queue.append((d.expected_state,path+[op],used|{i}))
    if not paths:raise ValueError('bounded native search did not certify a task; not an impossibility proof')
    result=ToolCertificate(task=task,max_depth=max_depth,max_nodes=max_nodes,max_paths=max_paths,search_nodes=nodes,search_exhausted=not queue,witnesses=paths)
    return result.model_copy(update={'certificate_hash':result.computed_hash()})


def verify(frozen,certificate):
    c=ToolCertificate.model_validate(certificate.model_dump(mode='json') if isinstance(certificate,ToolCertificate) else certificate)
    if c.certificate_hash!=c.computed_hash():raise ValueError('tool certificate hash differs')
    rebuilt=certify(frozen,c.task,max_depth=c.max_depth,max_nodes=c.max_nodes,max_paths=c.max_paths)
    if rebuilt!=c:raise ValueError('native tool proof differs')
    return True
