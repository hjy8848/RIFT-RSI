"""Unconditioned structural distribution; leaves legacy generate_world untouched."""
from usami.randomness import StableRandom
from usami_benchmark.contracts import GeneratorConfig
from usami_cloudops.models import World,Identity,Organization,Project,Quota,Access,Instance,Invoice,Payment,digest
from usami_cloudops.rules import invariants

GENERATOR_VERSION='cloudops-structural-v1'
RULES_VERSION='cloudops-0.1.0'


def generate_world(seed=42,config=None,*,diagnostics=None):
    cfg=GeneratorConfig.model_validate(config or {})
    rng=StableRandom(digest({'version':GENERATOR_VERSION,'seed':seed,'config':cfg.model_dump(mode='json')}))
    users={f'u{i}':Identity(id=f'u{i}',name=f'User {i}') for i in range(10)}
    orgs={f'o{i}':Organization(id=f'o{i}',name=f'Organization {i}',owner_id=f'u{i}') for i in range(rng.randint(2,3))}
    projects={};access={};instances={};invoices={};clock=World.model_fields['clock'].default
    for n in range(rng.randint(cfg.min_scopes,cfg.max_scopes)):
        pid=f'p{n}';oid=rng.choice(tuple(orgs));iid=f'bill{n}';active=rng.random()<.7
        billing='paid' if active else rng.choice(['paid','unpaid','failed'])
        amount=rng.choice([1000,1500,3000,7000]);payments=[]
        if billing!='unpaid':payments=[Payment(id='opening'+str(n),key='opening',method='card_approved' if billing=='paid' else 'card_declined',amount_cents=amount,outcome='succeeded' if billing=='paid' else 'failed',at=clock)]
        invoices[iid]=Invoice(id=iid,project_id=pid,amount_cents=amount,status=billing,payments=payments)
        prior=[];shape=rng.choice(['chain','star','layered','independent']);running=[]
        for k in range(rng.randint(cfg.min_resources,cfg.max_resources)):
            rid=f'{pid}_r{k}';deps=[]
            if prior and shape=='chain':deps=[prior[-1]]
            elif prior and shape=='star':deps=[prior[0]]
            elif prior and shape=='layered':deps=rng.sample(prior,rng.randint(0,min(2,len(prior))))
            status='running' if active and all(d in running for d in deps) and rng.random()<.5 else 'stopped'
            instances[rid]=Instance(id=rid,name=f'Resource {k}',project_id=pid,cpu=rng.choice([1,2,4]),memory=rng.choice([1,2,4,8]),status=status,dependencies=deps)
            prior.append(rid)
            if status=='running':running.append(rid)
        cpu=sum(instances[r].cpu for r in running);memory=sum(instances[r].memory for r in running)
        projects[pid]=Project(id=pid,name=f'Project {n}',org_id=oid,status='active' if active else 'suspended',invoice_id=iid,
            quota=Quota(cpu=min(64,cpu+rng.choice([0,1,4,8])),memory=min(256,memory+rng.choice([0,2,8,16])),instances=min(16,len(running)+rng.choice([0,1,2,4]))))
        for uid in rng.sample(list(users),rng.randint(2,6)):
            aid=pid+'_'+uid;access[aid]=Access(id=aid,project_id=pid,user_id=uid,role=rng.choice(['admin','operator','viewer']))
    world=World(seed=seed,users=users,organizations=orgs,projects=projects,access=access,instances=instances,invoices=invoices)
    if invariants(world):raise ValueError('structural generation violates CloudOps invariants: '+str(invariants(world)))
    return world
