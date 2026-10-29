"""Small curated release builder, structural audit and readable reviewer samples."""
from collections import Counter
from pathlib import Path
from usami.service.models import ModelSettings
from .models import digest
from .generator import generate_world
from .benchmark import Budget, Instance, Manifest, runtime_hashes, validate, write_json
from .goals import make_task
from .rules import decide

TRAIN=('billing_recovery','capacity_expand','dependency_boot','scoped_onboarding','quota_reduce','project_isolation','viewer_refusal','cycle_refusal','doublecharge_refusal')
TEST=('safe_retirement','create_stack','live_reconfigure')


def build_release(out):
    out=Path(out);model=ModelSettings(model='offline-contract',base_url='https://example.invalid/v1',options={'max_tokens':4096,'temperature':0})
    manifests=[]
    for split,families,replicas,base in (('train',TRAIN,3,10000),('validation',TRAIN,1,20000),('test-demo',TEST,3,30000)):
        instances=[]
        for n,family in enumerate(families):
            for r in range(replicas):
                world,task=make_task(base+n*100+r,family)
                instances.append(Instance(id=task.id,world=world,initial_hash=world.state_hash,identity=task.identity,task=task))
        m=Manifest(suite_id='cloudops-v0.1-'+split,track='service_task',split=split,
            budget=Budget(max_customer_turns=1,max_customer_llm_calls=0,max_service_llm_calls=16,max_service_tool_calls=32),
            service_model=model,instances=instances,source_hashes=runtime_hashes()).seal()
        validate(m);write_json(out/f'service-{split}.json',m);manifests.append(m)
    instances=[Instance(id='cloudworld_'+digest([seed,identity])[:20],world=generate_world(seed),initial_hash=generate_world(seed).state_hash,identity=identity) for seed,identity in ((40001,'u0'),(40002,'u2'))]
    d=Manifest(suite_id='cloudops-v0.1-discovery',track='customer_discovery',split='discovery',mode='sandbox',service_model=model,
        customer_model=model,budget=Budget(max_service_llm_calls=24,max_service_tool_calls=48),trials=['r0','r1','r2'],instances=instances,source_hashes=runtime_hashes()).seal()
    validate(d);write_json(out/'discovery-demo.json',d)
    audit=coverage(manifests);write_json(out/'coverage.json',audit)
    examples=[]
    for m in manifests:
        selected={}
        for i in m.instances:
            if len(selected.get(i.task.family,[]))>=2:continue
            selected.setdefault(i.task.family,[]).append(i.id)
            task=i.task
            examples.append({'instance_id':i.id,'split':m.split,'family':task.family,'identity':i.identity,'request':task.message,
                'initial_world':i.world.model_dump(mode='json'),'goal':task.goal.model_dump(mode='json'),
                'rejection_code':task.refusal_code,'legal_paths':[w.model_dump(mode='json') for w in task.witnesses],
                'wrong_result':'Missing goal, frame change or independent Oracle violations does not score success.'})
        if len(examples)>=18:break
    by_family={}
    for m in manifests:
        for instance in m.instances:
            by_family.setdefault(instance.task.family,[]).append(instance)
    chosen=[items[0] for items in by_family.values()] + [items[1] for items in list(by_family.values())[:6]]
    examples=[{'instance_id':i.id,'family':i.task.family,'identity':i.identity,'request':i.task.message,
        'split':next(m.split for m in manifests if i in m.instances),'initial_world':i.world.model_dump(mode='json'),
        'goal':i.task.goal.model_dump(mode='json'),'rejection_code':i.task.refusal_code,
        'legal_paths':[w.model_dump(mode='json') for w in i.task.witnesses],
        'wrong_result':'Reference rejects unsafe ordering; goal/frame/Oracle must all pass.'} for i in chosen]
    write_json(out/'review-examples.json',examples)
    text=['# CloudOps reviewer examples','', 'Status: AWAITING EXTERNAL HUMAN REVIEW. Software witnesses are not human business-quality ratings.','']
    for index,e in enumerate(examples):
        text += [f"## {index+1}. {e['family']} ({e['split']})",'',f"Identity: {e['identity']}. Request: {e['request']}",'',
            'Initial state / terminal goal / legal conditions / causal omission evidence / wrong-result boundary:', '', '```json',__import__('json').dumps(e,ensure_ascii=False,indent=2),'```','']
    (out/'TASK_EXAMPLES.md').write_text('\n'.join(text),encoding='utf-8')
    return audit


def coverage(manifests):
    tasks=[i.task for m in manifests for i in m.instances];structures=Counter(t.structure_signature for t in tasks);semantics=Counter(t.semantic_signature for t in tasks)
    families=Counter(t.family for t in tasks);ops=Counter();guards=Counter();depths=Counter();direct=Counter();indirect=Counter()
    routes=0
    for m in manifests:
        for i in m.instances:
            for op in i.task.capabilities:
                ops[op.name]+=1;guards[op.name+'/'+(decide(i.world,i.identity,op).error_code or 'ALLOWED')]+=1
            depths[max((w.depth for w in i.task.witnesses),default=0)]+=1
            for w in i.task.witnesses:
                routes+=1
                for edge in w.direct_edges:direct[w.operations[edge['from']].name+'->'+w.operations[edge['to']].name]+=1
                for edge in w.indirect_edges:indirect[w.operations[edge['from']].name+'->'+w.operations[edge['to']].name]+=1
    overlap={}
    for a,b in (('train','validation'),('train','test-demo'),('validation','test-demo')):
        left=next(m for m in manifests if m.split==a);right=next(m for m in manifests if m.split==b)
        overlap[a+'/'+b]={'structures':len({i.task.structure_signature for i in left.instances}&{i.task.structure_signature for i in right.instances}),
            'semantics':len({i.task.semantic_signature for i in left.instances}&{i.task.semantic_signature for i in right.instances})}
    cross=sum(any(o.resource_id in i.world.instances and i.world.instances[o.resource_id].project_id!=o.project_id for o in i.task.capabilities) for m in manifests for i in m.instances)
    subsystem=lambda name:'billing' if name in {'pay_invoice','resume_project'} else 'capacity' if name=='change_project_quota' else 'access' if 'project_access' in name else 'compute'
    compound=sum(len({subsystem(o.name) for o in t.capabilities})>1 for t in tasks)
    return {'domain':'cloudops','version':'0.1.0','service_tasks':len(tasks),'split_counts':{m.split:len(m.instances) for m in manifests},'business_families':dict(families),
        'independent_structures':len(structures),'semantic_decision_signatures':len(semantics),'structure_frequencies':dict(structures),'semantic_frequencies':dict(semantics),
        'operation_coverage':dict(ops),'actual_initial_decision_coverage':dict(guards),'policy_refusal_tasks':sum(t.kind=='policy_refusal' for t in tasks),
        'cross_project_tasks':cross,'cross_project_ratio':cross/len(tasks),'cross_subsystem_tasks':compound,'cross_subsystem_ratio':compound/len(tasks),
        'causal_depth_distribution':dict(sorted(depths.items())),'multi_path_tasks':sum(len(t.witnesses)>1 for t in tasks),'verified_witnesses':routes,
        'direct_first_failure_edges':dict(direct),'indirect_after_failed_intervening_step_edges':dict(indirect),'split_overlap':overlap,
        'human_review':'AWAITING EXTERNAL REVIEW','limitations':['Numeric sizing variants and reordered legal witnesses are not independent task structures.','Train/validation are in-distribution; public test is a three-family composition demo, never a blind/private test.','Omission depth describes the verified bounded witnesses, not exhaustive all-plan causal depth.','Operation rule coverage must be supplemented with boundary tests; initial first-error guards are not exhaustive predicate coverage.']}
