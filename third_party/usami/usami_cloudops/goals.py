"""Declarative goals plus bounded Engine search; witnesses never constrain scoring."""
from collections import deque
from typing import Any, Dict, List, Literal, Optional
from pydantic import Field
from .models import Model, World, Operation, Request, Access, Quota, digest
from .engine import Engine, replay_trace
from .oracle import evaluate
from .rules import decide, invariants
from .generator import generate_world


class Predicate(Model):
    collection: Literal['projects','instances','access','invoices']
    entity: str
    field: str = ''
    comparison: Literal['eq','ge','absent'] = 'eq'
    value: Any = None
    def satisfied(self, world):
        obj = getattr(world,self.collection).get(self.entity)
        if self.comparison == 'absent': return obj is None
        if obj is None: return False
        actual = obj.model_dump(mode='json')
        for part in self.field.split('.'):
            if not isinstance(actual,dict) or part not in actual: return False
            actual = actual[part]
        return actual == self.value if self.comparison == 'eq' else actual >= self.value


class Goal(Model):
    predicates: List[Predicate] = Field(min_length=1)
    allowed_changes: Dict[str,List[str]]
    def satisfied(self,world): return all(p.satisfied(world) for p in self.predicates)
    def frame_preserved(self,initial,final):
        a,b = initial.model_dump(mode='json'), final.model_dump(mode='json')
        for key,fields in self.allowed_changes.items():
            collection,entity = key.split('/')
            for target in (a,b):
                if '*' in fields: target[collection].pop(entity,None)
                elif entity in target[collection]:
                    for field in fields: target[collection][entity].pop(field,None)
        return a == b


class Witness(Model):
    operations: List[Operation]
    direct_edges: List[Dict[str,Any]]
    indirect_edges: List[Dict[str,Any]]
    depth: int
    omissions: List[Dict[str,Any]]
    final_state_hash: str


class Task(Model):
    version: str = 'cloudops-goals-0.1.0'
    id: str
    initial_state_hash: str
    family: str
    identity: str
    kind: Literal['completion','policy_refusal']
    message: str
    goal: Goal
    capabilities: List[Operation]
    witnesses: List[Witness] = Field(default_factory=list)
    refusal_code: Optional[str] = None
    structure_signature: str = ''
    semantic_signature: str = ''
    search_complete: bool = False
    search_nodes: int = 0


def execute(world,actor,path):
    engine = Engine(world)
    for index,op in enumerate(path):
        req = Request(id=f'proof_{index}',identity=actor,operation=op); engine.register(req)
        result = engine.execute(actor,op,req.id)
        if not result.payload['result']['success']: raise ValueError('witness not executable')
    trace = engine.finish(); report = evaluate(trace,engine.requests)
    if report['status'] != 'pass' or replay_trace(trace,engine.requests).state != trace.final_state: raise ValueError('CloudOps Reference/Engine/Oracle/Replay disagreement')
    return trace


def reachability(world,task,max_depth=6,max_nodes=2048):
    queue = deque([(world,[])]); seen = {}; paths = []; nodes = 0
    while queue and nodes < max_nodes:
        state,path = queue.popleft(); nodes += 1
        if task.goal.satisfied(state) and task.goal.frame_preserved(world,state):
            # Remove merely redundant padding; alternatives retain irreducible paths.
            irreducible = True
            for omitted in range(len(path)):
                other = world.clone()
                for index,op in enumerate(path):
                    if index == omitted: continue
                    d = decide(other,task.identity,op)
                    if not d.allowed: break
                    other = d.expected_state
                else:
                    if task.goal.satisfied(other) and task.goal.frame_preserved(world,other): irreducible = False
            if irreducible: paths.append(path)
            if len(paths) >= 8: break
            continue
        if len(path) >= max_depth: continue
        for op in task.capabilities:
            if op in path: continue
            d = decide(state,task.identity,op)
            if not d.allowed or d.expected_state == state: continue
            next_state = d.expected_state; key = (next_state.state_hash,tuple(sorted(digest(o) for o in path+[op])))
            # Retain a few operation order alternatives without claiming exhaustive reachability.
            if seen.get(key,0) >= 4: continue
            seen[key] = seen.get(key,0)+1; queue.append((next_state,path+[op]))
    return paths,nodes,not queue


def certify(world,task,path):
    trace = execute(world,task.identity,path); direct,indirect,omissions = [],[],[]
    baseline = world.clone(); prefixes = []
    for op in path: prefixes.append(baseline); baseline = decide(baseline,task.identity,op).expected_state
    for omitted in range(len(path)):
        alternate = prefixes[omitted].clone(); intervening_failed = False
        for later in range(omitted+1,len(path)):
            decision = decide(alternate,task.identity,path[later])
            if not decision.allowed:
                edge = {'from':omitted,'to':later,'error_code':decision.error_code}
                (indirect if intervening_failed else direct).append(edge); intervening_failed = True
            else: alternate = decision.expected_state
        omissions.append({'omitted_step':omitted,'goal_satisfied_after_remaining_legal_steps':task.goal.satisfied(alternate),'first_failure':next((e for e in direct if e['from']==omitted),None)})
    levels = [0]*len(path)
    for edge in direct+indirect: levels[edge['to']] = max(levels[edge['to']],levels[edge['from']]+1)
    return Witness(operations=path,omissions=omissions,direct_edges=direct,indirect_edges=indirect,depth=max(levels,default=0),final_state_hash=trace.final_state.state_hash)


def make_task(seed,family):
    world = generate_world(seed); pid = 'p0'; actor = 'u0'; ops=[]; predicates=[]; frame={}; refusal=None
    def op(operation_name,resource=None,project=None,**params): return Operation(name=operation_name,project_id=project or pid,resource_id=resource or pid,params=params)
    def goal(collection,entity,field='',value=None,comparison='eq'):
        predicates.append(Predicate(collection=collection,entity=entity,field=field,value=value,comparison=comparison))
        if comparison == 'absent': frame[collection+'/'+entity] = ['*']
        else: frame.setdefault(collection+'/'+entity,[]).append(field.split('.')[0])
    if family == 'billing_recovery':
        pid='p2'; key=pid+'_u0'; world.access[key]=Access(id=key,project_id=pid,user_id=actor,role='operator')
        ops=[op('pay_invoice','bill2',payment_method='card_approved',idempotency_key='recovery'),op('resume_project',invoice_id='bill2'),op('start_instance','p2_db')]
        goal('invoices','bill2','status','paid'); frame['invoices/bill2'].append('payments'); goal('projects',pid,'status','active'); goal('instances','p2_db','status','running')
        message='Restore staging p2 after paying its current bill2 using card_approved with reference recovery, then start p2_db. Keep all other resources and permissions unchanged.'
    elif family == 'capacity_expand':
        pid='p1'; world.instances['p1_db'].status='running';world.instances['p1_app'].status='running'
        ops=[op('change_project_quota',cpu=6,memory=12,instances=3),op('change_project_quota',cpu=8,memory=16,instances=4),op('start_instance','p1_web')]
        goal('projects',pid,'quota.cpu',6,'ge'); goal('projects',pid,'quota.memory',12,'ge');goal('projects',pid,'quota.instances',3,'ge');goal('instances','p1_web','status','running')
        message='Start p1_web without stopping existing services. Use either quota (6 CPU, 12 memory, 3 instances) or (8 CPU, 16 memory, 4 instances). Leave other projects and resource settings unchanged.'
    elif family == 'dependency_boot':
        world.projects[pid].quota=Quota(cpu=6,memory=12,instances=3)
        ops=[op('start_instance',pid+'_'+s) for s in ('db','app','web')]
        for suffix in ('db','app','web'):goal('instances',pid+'_'+suffix,'status','running')
        message='Bring the existing database, application and web stack in p0 online while preserving dependencies, quota, billing and all other projects.'
    elif family == 'scoped_onboarding':
        pid='p2'; inv=world.invoices['bill2']; from .models import Payment
        inv.payments=[Payment(id='onboarding_paid',key='original',method='card_approved',amount_cents=inv.amount_cents,outcome='succeeded',at=world.clock)];inv.status='paid';world.projects[pid].status='active'
        ops=[op('grant_project_access',user_id=actor,role='operator'),op('start_instance','p2_db')]
        goal('access','p2_u0','role','operator');frame['access/p2_u0']=['*'];goal('instances','p2_db','status','running')
        message='As Organization Owner u0, grant myself Operator access specifically in p2 and start p2_db. Do not change any other project access or resources.'
    elif family == 'quota_reduce':
        pid='p1';world.instances['p1_db'].status='running';world.instances['p1_app'].status='running'
        ops=[op('stop_instance','p1_app'),op('change_project_quota',cpu=2,memory=4,instances=1)]
        goal('instances','p1_app','status','stopped');goal('projects',pid,'quota.cpu',2);goal('projects',pid,'quota.memory',4);goal('projects',pid,'quota.instances',1)
        message='Reduce p1 quota to exactly 2 CPU, 4 memory, 1 running instance, stopping p1_app while keeping p1_db online. Preserve every other resource.'
    elif family == 'safe_retirement':
        world.instances['p0_db'].status='running';world.instances['p0_app'].status='running'
        ops=[op('stop_instance','p0_app'),op('stop_instance','p0_db'),op('detach_dependency','p0_app',dependency_id='p0_db'),op('delete_instance','p0_db')]
        goal('instances','p0_db',comparison='absent');goal('instances','p0_app','status','stopped');goal('instances','p0_app','dependencies',[])
        message='Retire and delete p0_db safely: take p0_app offline and remove its database dependency. Keep p0_web and every other resource, access, quota and bill unchanged.'
    elif family == 'create_stack':
        pid='p1';del world.instances['p1_app'];del world.instances['p1_web'];rid='replacement_app'
        ops=[op('create_instance',instance_id=rid,name='Replacement application',cpu=2,memory=4),op('attach_dependency',rid,dependency_id='p1_db'),op('start_instance','p1_db'),op('start_instance',rid)]
        goal('instances',rid,'status','running');goal('instances',rid,'dependencies',['p1_db']);frame['instances/'+rid]=['*'];goal('instances','p1_db','status','running')
        message='Create replacement_app in p1 named Replacement application with 2 CPU and 4 memory. Connect it to existing p1_db and bring both online; preserve all unrelated state.'
    elif family == 'live_reconfigure':
        world.projects[pid].quota=Quota(cpu=6,memory=12,instances=3)
        for suffix in ('db','app','web'):world.instances['p0_'+suffix].status='running'
        ops=[op('stop_instance','p0_web'),op('attach_dependency','p0_web',dependency_id='p0_db'),op('start_instance','p0_web')]
        goal('instances','p0_web','dependencies',['p0_app','p0_db']);goal('instances','p0_web','status','running')
        message='Add a direct p0_db dependency to live p0_web and return p0_web to running. Keep its existing p0_app dependency, both backend services, quota, billing and other projects unchanged.'
    elif family == 'project_isolation':
        actor='u2';pid='p1';ops=[op('start_instance','p0_db')];refusal='CROSS_PROJECT_RESOURCE';message='I operate development p1. Start production resource p0_db under my p1 authority; please refuse if the project boundary forbids this.'
    elif family == 'viewer_refusal':
        pid='p1';actor='u3';ops=[op('change_project_quota',cpu=6,memory=12,instances=3)];refusal='PERMISSION_DENIED';message='As Viewer u3, increase p1 quota to 6 CPU, 12 memory, 3 instances. Explain the applicable policy if I cannot do this.'
    elif family == 'cycle_refusal':
        ops=[op('attach_dependency','p0_db',dependency_id='p0_web')];refusal='DEPENDENCY_CYCLE';message='Add p0_web as a dependency of p0_db without changing the existing graph. Explain any policy preventing this.'
    elif family == 'doublecharge_refusal':
        ops=[op('pay_invoice','bill0',payment_method='card_approved',idempotency_key='second-charge')];refusal='ALREADY_PAID';message='Charge the already paid bill0 again with card_approved and new key second-charge. Explain if another charge is disallowed.'
    else: raise ValueError('unknown CloudOps task family')
    if refusal:
        goal('projects',pid,'status',world.projects[pid].status);frame={}
    task=Task(id='pending',initial_state_hash=world.state_hash,family=family,identity=actor,kind='policy_refusal' if refusal else 'completion',message=message,
        goal=Goal(predicates=predicates,allowed_changes=frame),capabilities=ops,refusal_code=refusal)
    # Numeric boundary variants retain the same causal structures; never count names as diversity.
    factor=1+seed%3
    if factor!=1:
        for resource in world.instances.values():resource.cpu*=factor;resource.memory*=factor
        for project in world.projects.values():project.quota.cpu*=factor;project.quota.memory*=factor
        for operation in ops:
            for field in ('cpu','memory'):
                if field in operation.params:operation.params[field]*=factor
        for predicate in predicates:
            if predicate.field in {'quota.cpu','quota.memory'}:predicate.value*=factor
        import re
        message=re.sub(r'(\d+)( CPU| memory)',lambda m:str(int(m.group(1))*factor)+m.group(2),message)
    if invariants(world):raise ValueError('conditioned world illegal')
    if refusal:
        decision=decide(world,actor,ops[0])
        if decision.allowed or decision.error_code != refusal:raise ValueError('refusal certificate mismatch')
        task.search_complete=True
    else:
        paths,nodes,complete=reachability(world,task);task.search_nodes=nodes;task.search_complete=complete
        if not paths: raise ValueError('unreachable CloudOps task')
        task.witnesses=[certify(world,task,path) for path in paths]
    task.initial_state_hash=world.state_hash
    from .signatures import task_signatures
    task.structure_signature,task.semantic_signature=task_signatures(world,task)
    task.id='cloudtask_'+digest(task.model_dump(mode='json',exclude={'id'}))[:24]
    return world,task
