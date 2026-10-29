"""Bounded generator-independent task index/audit; not the Designer's output space."""
from usami_cloudops.models import World,Operation
from usami_cloudops.rules import decide,owner,role
from usami_cloudops.observations import observation,visible_ids
from usami_harness.goals import JointGoal
from .composer import TaskProposal,TaskSpec,TaskComposer,semantic_signatures


def candidates(entry):
    world=World.model_validate(entry.world)
    def proposal(uid,ops,goal=None,need=None):
        d=decide(world,uid,ops[-1]);kind='completion' if goal or d.allowed else 'policy_refusal'
        return TaskProposal(world_id=entry.spec.id,world_hash=entry.world_hash,identity=uid,branch='correct_to_failure',
            hypothesis='Independent pool coverage probe; no Service failure assumed.',
            task_spec=TaskSpec(kind=kind,target=ops[-1],goal=goal),candidate_operations=tuple(ops),
            customer_brief=need or f'Please perform {ops[-1].name} for {ops[-1].resource_id} and explain the actual result.')
    def op(name,pid,rid=None,**params):return Operation(name=name,project_id=pid,resource_id=rid or pid,params=params)
    for uid in sorted(world.users):
        visible=visible_ids(observation(world,uid))
        for rid,i in sorted(world.instances.items()):
            if rid not in visible:continue
            yield proposal(uid,[op('start_instance',i.project_id,rid)])
            yield proposal(uid,[op('delete_instance',i.project_id,rid)])
            for dependency in i.dependencies:
                yield proposal(uid,[op('detach_dependency',i.project_id,rid,dependency_id=dependency)])
        for pid,p in sorted(world.projects.items()):
            if pid not in visible:continue
            yield proposal(uid,[op('change_project_quota',pid,cpu=6,memory=12,instances=3)])
            resources=[i for i in world.instances.values() if i.project_id==pid]
            roots=[i for i in resources if not i.dependencies]
            if len(resources)==3:
                ordered=sorted(resources,key=lambda i:len(i.dependencies))
                # Native DAG order is searched, never prescribed to Customer.
                writes=[op('change_project_quota',pid,cpu=6,memory=12,instances=3)]+[op('start_instance',pid,i.id) for i in ordered]
                goal=JointGoal(atoms=[dict(collection='instances',entity=i.id,field='status',value='running') for i in ordered],
                    allowed_changes={f'projects/{pid}':['quota'],**{f'instances/{i.id}':['status'] for i in ordered}})
                yield proposal(uid,writes,goal,f'Bring the full existing stack in {pid} online; increase quota to 6 CPU, 12 memory, 3 instances if needed, preserving unrelated state.')
            for root in roots:
                for child in resources:
                    if root.id not in child.dependencies:continue
                    ops=[op('start_instance',pid,root.id),op('start_instance',pid,child.id)]
                    goal=JointGoal(atoms=[dict(collection='instances',entity=i.id,field='status',value='running') for i in (root,child)],
                        allowed_changes={f'instances/{i.id}':['status'] for i in (root,child)})
                    yield proposal(uid,ops,goal,f'Bring {root.id} and {child.id} online, preserving billing, quota and unrelated resources.')
            # Billing recovery and optional legitimate access setup are operations,
            # not conditioning edits to the initial generator world.
            if p.invoice_id in visible and p.status=='suspended' and roots:
                ops=[];frame={}
                if role(world,uid,pid) not in {'admin','operator'}:
                    ops.append(op('grant_project_access',pid,user_id=uid,role='operator'));frame[f'access/{pid}_{uid}']=['*']
                ops.extend([op('pay_invoice',pid,p.invoice_id,payment_method='card_approved',idempotency_key='pool-recovery'),
                            op('resume_project',pid,invoice_id=p.invoice_id),op('start_instance',pid,roots[0].id)])
                frame.update({f'invoices/{p.invoice_id}':['status','payments'],f'projects/{pid}':['status'],f'instances/{roots[0].id}':['status']})
                goal=JointGoal(atoms=[dict(collection='invoices',entity=p.invoice_id,field='status',value='paid'),dict(collection='projects',entity=pid,field='status',value='active'),
                    dict(collection='instances',entity=roots[0].id,field='status',value='running')],allowed_changes=frame)
                yield proposal(uid,ops,goal,f'Restore {pid}: arrange necessary project access, pay its current bill with card_approved and reference pool-recovery, resume it and start {roots[0].id}.')


def audit(pool,budget):
    compiler=TaskComposer(pool,budget);summaries=[];registered=[];search_nodes=0;proof_operations=0;shapes=set();semantics=set()
    for entry in pool.entries.values():
        kinds={};codes={};roles={};ops=set();chains=set();attempts=0;failures={};maxdepth=0;multi=0;local_semantics=set()
        for p in candidates(entry):
            attempts+=1
            try:s=compiler.compile(None,p)
            except ValueError as exc:
                search_nodes+=compiler.stats.get('reference_search_nodes',0);proof_operations+=compiler.stats.get('native_proof_operations',0)
                message=str(exc);failures[message]=failures.get(message,0)+1;continue
            search_nodes+=compiler.stats.get('reference_search_nodes',0);proof_operations+=compiler.stats.get('native_proof_operations',0)
            registered.append(s)
            shape,semantic=semantic_signatures(s);shapes.add(shape);semantics.add(semantic);local_semantics.add(semantic)
            kinds[s.certificate['kind']]=kinds.get(s.certificate['kind'],0)+1
            if s.certificate['reference_code']:codes[s.certificate['reference_code']]=codes.get(s.certificate['reference_code'],0)+1
            world=World.model_validate(entry.world);r=role(world,p.identity,p.target.project_id) or ('owner_without_project_role' if owner(world,p.identity,p.target.project_id) else 'none')
            roles[r]=roles.get(r,0)+1;ops.update(o.name for o in p.capabilities)
            if len(s.witness)>1:multi+=1;chains.add(tuple(o['name'] for o in s.witness))
            maxdepth=max(maxdepth,s.certificate['causal_depth'])
        summaries.append({'world_id':entry.spec.id,'world_hash':entry.world_hash,'structure_hash':entry.structure_hash,
            'attempted_templates':attempts,'verified_templates':sum(kinds.values()),'kinds':kinds,'reference_codes':codes,'roles':roles,
            'operations':sorted(ops),'multi_step_tasks':multi,'max_causal_depth':maxdepth,'chains':[list(c) for c in sorted(chains)],'construction_gaps':failures,
            'unique_semantic_tasks':len(local_semantics)})
    return {'protocol':'world-pool-coverage-v1','pool_hash':pool.pool_hash,'registered_worlds':len(pool.entries),
        'reference_search_nodes':search_nodes,'native_proof_operations':proof_operations,'model_calls':0,
        'effective_structural_worlds':len({e.structure_hash for e in pool.entries.values()}),'worlds':summaries,
        'unique_task_structures':len(shapes),'unique_task_semantics':len(semantics),
        'limitations':['Native CloudOps seeds randomize names but retain the same business graph, roles, statuses and quotas.',
            'Coverage uses bounded existing-entity task templates; it is a lower bound, not exhaustive enumeration.',
            'Initial running-resource states, alternative DAG topologies and quota distributions are absent.',
            'Creating new resource IDs and compound multi-write refusal goals are outside this first composer.',
            'Terminal goals cannot certify nontrivial temporal loops that return to the initial state.'],
        'research_readiness':'limited_generator_diversity; no cross-structure generalization claim'},registered
