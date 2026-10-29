"""Task-only composer. Uses native reachability/certify against exact pool bytes."""
from typing import Literal, Optional, Tuple
from pydantic import Field
from vast_coevo.schema import FrozenModel
from vast_coevo.designer.storage import digest
from vast_coevo.skills.models import Hash
from vast_coevo.designer.models import Branch
from usami_cloudops.models import World,Operation,TOOLS
from usami_cloudops.rules import decide,POLICY
from usami_cloudops.observations import observation,visible_ids
from usami_cloudops.goals import Task,Goal,Predicate,reachability,certify,execute
from usami_harness.goals import JointGoal
from integrations.usami.designer.compiler import Anchor,goal_from_transition,expectation
from integrations.usami.designer.dialogue import CustomerBrief
from integrations.usami.examiner import goal_adapter
from vast_coevo.skills.validation import _unsafe_instruction


class TaskSpec(FrozenModel):
    kind: Literal['completion','policy_refusal']
    target: Operation
    goal: Optional[JointGoal]=None
    constraints: Tuple[str,...]=Field(default=(),max_length=8)


class TaskProposal(FrozenModel):
    world_id: str
    world_hash: Hash
    identity: str
    source_evidence: Optional[Hash]=None
    branch: Branch
    hypothesis: str=Field(min_length=1,max_length=3000)
    task_spec: TaskSpec
    candidate_operations: Tuple[Operation,...]=Field(min_length=1,max_length=10)
    customer_brief: str=Field(min_length=1,max_length=3000)
    @property
    def target(self):return self.task_spec.target
    @property
    def capabilities(self):return self.candidate_operations
    @property
    def goal(self):return self.task_spec.goal


class FrozenTask(FrozenModel):
    protocol: Literal['frozen-task-discovery-v1']='frozen-task-discovery-v1'
    proposal: TaskProposal
    anchor: Optional[Anchor]=None
    world: dict
    pool_hash: Hash
    world_structure_hash: Hash
    native_task: dict
    certificate: dict
    witness: Tuple[dict,...]
    brief: dict
    reference_before: dict
    reference_after: dict
    diff: Tuple[dict,...]=() # Always empty; retained only for common feedback display.
    search_nodes: int
    scene_hash: str=''
    def checked(self):
        value=digest(self.model_dump(mode='json',exclude={'scene_hash'}))
        if self.scene_hash and value!=self.scene_hash:raise ValueError('frozen task hash mismatch')
        return self.model_copy(update={'scene_hash':value})


class TaskComposer:
    def __init__(self,pool,budget):self.pool=pool;self.budget=budget;self.stats={}
    def compile(self,anchor,p):
        self.stats={'reference_search_nodes':0,'native_proof_operations':0}
        p=TaskProposal.model_validate(p.model_dump(mode='json'));entry=self.pool.load(p.world_id,p.world_hash)
        world=World.model_validate(entry.world);before=world.state_hash
        if p.source_evidence is not None and (anchor is None or p.source_evidence!=anchor.evidence_hash or p.branch!=anchor.branch):raise ValueError('task inspiration must match verified branch/source')
        if p.source_evidence is None and anchor is not None:raise ValueError('undeclared source')
        if p.identity not in world.users:raise ValueError('unknown execution identity')
        visible=visible_ids(observation(world,p.identity))
        if p.target not in p.capabilities or len({digest(o) for o in p.capabilities})!=len(p.capabilities):raise ValueError('target must be a distinct exact candidate operation')
        for op in p.capabilities:
            if op.is_query or op.name not in TOOLS:raise ValueError('candidate must be a native business write')
            if op.name=='create_instance':raise ValueError('new resource IDs are outside existing-entity task composition')
            collection=world.instances if op.name in {'start_instance','stop_instance','delete_instance','attach_dependency','detach_dependency'} else world.invoices if op.name=='pay_invoice' else world.projects
            if op.project_id not in world.projects or op.resource_id not in collection:raise ValueError('task operation references nonexistent entity')
            ids=[op.project_id,op.resource_id]+[op.params[k] for k in ('dependency_id','invoice_id','user_id') if k in op.params]
            if any(key not in visible for key in ids):raise ValueError('task refers to Customer-invisible entity')
        text=p.customer_brief+' '.join(p.task_spec.constraints)
        hidden={k for c in ('users','projects','instances','invoices','access','organizations') for k in getattr(world,c)}-visible
        import re
        if (_unsafe_instruction(text) or any(t in text.casefold() for t in ('oracle','witness','hypothesis','error_amplification','correct_to_failure','<vast_')) or
                any(code in text for code in POLICY) or p.world_hash in text or self.pool.pool_hash in text or (p.source_evidence and p.source_evidence in text)):
            raise ValueError('private evaluator material or authority instruction in Customer brief')
        if any(re.search(r'(?<!\w)'+re.escape(k)+r'(?!\w)',text) for k in hidden):raise ValueError('hidden entity in Customer brief')
        if p.task_spec.kind=='completion' and p.goal is None:
            d=decide(world,p.identity,p.target)
            if not d.allowed:raise ValueError('multi-step completion requires an explicit goal')
            p=p.model_copy(update={'task_spec':p.task_spec.model_copy(update={'goal':goal_from_transition(world,d.expected_state)})})
        native=native_task(world,p,self.budget,self.stats)
        if world.state_hash!=before or world.model_dump(mode='json')!=entry.world:raise ValueError('task composer changed frozen world')
        # The proof is private. Customer receives neither path, expected refusal code nor goal certificate.
        brief=CustomerBrief(need=p.customer_brief,constraints=p.task_spec.constraints,capabilities=p.capabilities).model_dump(mode='json')
        paths=native.witnesses
        witness=paths[0].operations if paths else []
        if p.task_spec.kind=='completion' and p.goal is None:
            joint=JointGoal(atoms=[dict(collection=a.collection,entity=a.entity,field=a.field,comparison=a.comparison,value=a.value) for a in native.goal.predicates],allowed_changes=native.goal.allowed_changes)
            p=p.model_copy(update={'task_spec':p.task_spec.model_copy(update={'goal':joint})})
        certificate={'protocol':'frozen-task-proof-v1','world_hash':before,'pool_hash':self.pool.pool_hash,'kind':native.kind,
            'reference_code':native.refusal_code,'native_task_hash':digest(native),'proof_hash':digest(paths),
            'causal_depth':max((w.depth for w in paths),default=0),'search_complete':native.search_complete}
        return FrozenTask(proposal=p,anchor=anchor,world=entry.world,pool_hash=self.pool.pool_hash,world_structure_hash=entry.structure_hash,
            native_task=native.model_dump(mode='json'),certificate=certificate,witness=tuple(o.model_dump(mode='json') for o in witness),
            brief=brief,reference_before=anchor.facts['expected'] if anchor else expectation(world,p.identity,p.target),
            reference_after=expectation(world,p.identity,p.target),search_nodes=native.search_nodes).checked()


def native_task(world,p,budget,stats=None):
    """Never calls make_task: no conditioned world or model-authored witness."""
    stats=stats if stats is not None else {};goal=p.goal
    if p.task_spec.kind=='policy_refusal':
        d=decide(world,p.identity,p.target)
        if len(p.capabilities)!=1 or goal is not None or d.allowed or not d.error_code or d.error_code=='INVALID_WORLD' or d.error_code.endswith('NOT_FOUND'):
            raise ValueError('refusal requires one existing-object policy boundary')
        pid=p.target.project_id
        task=Task(id='frozen-'+digest(p)[:20],initial_state_hash=world.state_hash,family='frozen_refusal',identity=p.identity,kind='policy_refusal',
            message=p.customer_brief,goal=Goal(predicates=[Predicate(collection='projects',entity=pid,field='status',value=world.projects[pid].status)],allowed_changes={}),
            capabilities=list(p.capabilities),refusal_code=d.error_code,search_complete=True)
        from usami_cloudops.engine import Engine,replay_trace
        from usami_cloudops.models import Request
        from usami_cloudops.oracle import evaluate
        engine=Engine(world);req=Request(id='refusal-proof',identity=p.identity,operation=p.target);engine.register(req)
        stats['native_proof_operations']=1
        event=engine.execute(p.identity,p.target,req.id);trace=engine.finish()
        if event.payload['result']['success'] or evaluate(trace,engine.requests)['status']!='pass' or replay_trace(trace,engine.requests).state!=world:
            raise ValueError('Reference refusal proof failed')
        return task
    if goal is None:
        d=decide(world,p.identity,p.target)
        if not d.allowed:raise ValueError('multi-step completion requires an explicit goal')
        goal=goal_from_transition(world,d.expected_state)
    goal.validate_for(goal_adapter('cloudops',world),world)
    visible=visible_ids(observation(world,p.identity))
    created_access={o.project_id+'_'+o.params['user_id'] for o in p.capabilities if o.name=='grant_project_access'}
    for key in goal.allowed_changes:
        collection,entity=key.split('/')
        if entity not in visible and not (collection=='access' and entity in created_access):raise ValueError('frame refers to invisible or nonexistent entity')
    for atom in goal.atoms:
        if atom.entity not in getattr(world,atom.collection):raise ValueError('goal references nonexistent entity')
        if atom.entity not in visible_ids(observation(world,p.identity)):raise ValueError('goal references invisible entity')
    if goal.satisfied(world):raise ValueError('goal already satisfied')
    if any(a.comparison not in {'eq','ge','absent'} for a in goal.atoms):raise ValueError('unsupported native goal comparison')
    task=Task(id='frozen-'+digest(p)[:20],initial_state_hash=world.state_hash,family='frozen_composition',identity=p.identity,kind='completion',message=p.customer_brief,
        goal=Goal(predicates=[Predicate(**a.model_dump(mode='json')) for a in goal.atoms],allowed_changes=goal.allowed_changes),capabilities=list(p.capabilities))
    paths,nodes,complete=reachability(world,task,max_depth=min(6,budget.max_service_tool_calls),max_nodes=512)
    stats['reference_search_nodes']=nodes
    if not paths:raise ValueError('bounded native search cannot prove task reachability')
    for path in paths:
        task.witnesses.append(certify(world,task,path))
        stats['native_proof_operations']=stats.get('native_proof_operations',0)+len(path)
    task.search_nodes=nodes;task.search_complete=complete
    return task


def semantic_signatures(scene):
    """Native anonymous task signatures; authored list order is not diversity."""
    from usami_cloudops.signatures import task_signatures
    task=Task.model_validate(scene.native_task)
    task.capabilities=sorted(task.capabilities,key=lambda o:(o.name,o.project_id,o.resource_id,digest(o.params)))
    task.goal.predicates=sorted(task.goal.predicates,key=lambda a:(a.collection,a.entity,a.field,a.comparison,digest(a.value)))
    return task_signatures(World.model_validate(scene.world),task)
