"""Anonymous causal signatures, not seed/name/identifier diversity."""
from .models import digest
from .rules import decide, owner, role, usage


def facts(world,actor,op):
    p=world.projects.get(op.project_id);i=world.instances.get(op.resource_id);bill=world.invoices.get(op.resource_id)
    return {'owner':bool(p and owner(world,actor,p.id)), 'role':role(world,actor,p.id),
        'status':p.status if p else None,'quota_usage_relation':{k:'full' if v==getattr(p.quota,k) else 'over' if v>getattr(p.quota,k) else 'free' for k,v in usage(world,p.id).items()} if p else {},
        'resource_same_project':bool((i and i.project_id==op.project_id) or (bill and bill.project_id==op.project_id) or op.resource_id==op.project_id),
        'instance_state':i.status if i else None,'dependency_count':len(i.dependencies) if i else 0,
        'billing_status':bill.status if bill else None,'reference_code':decide(world,actor,op).error_code}


def task_signatures(world,task):
    # Root IDs mapped by semantic appearances, never lexicographic ID order.
    aliases={}; counters={}
    def alias(collection,key):
        token=(collection,key)
        if token not in aliases:
            counters[collection]=counters.get(collection,0)+1;aliases[token]=collection+str(counters[collection])
        return aliases[token]
    predicates=[]
    for p in task.goal.predicates:
        value=p.value
        if isinstance(value,list):value=[alias('instances',v) if v in world.instances else v for v in value]
        if isinstance(value,(int,float)) and not isinstance(value,bool):value='integer'
        predicates.append([p.collection,alias(p.collection,p.entity),p.field,p.comparison,value])
    types=[]
    for op in task.capabilities:
        collection='instances' if op.resource_id in world.instances or op.name in {'start_instance','attach_dependency'} else 'invoices' if op.resource_id in world.invoices else 'projects'
        types.append([op.name,alias('projects',op.project_id),alias(collection,op.resource_id),sorted(op.params)])
    edge_types=sorted({(w.operations[e['from']].name,w.operations[e['to']].name,e['error_code'],'direct' if e in w.direct_edges else 'indirect') for w in task.witnesses for e in w.direct_edges+w.indirect_edges})
    shape={'kind':task.kind,'predicates':predicates,'operations':types,'causal_edges':edge_types}
    semantic={'shape':shape,'decisions':[facts(world,task.identity,op) for op in task.capabilities]}
    return digest(shape),digest(semantic)


def finding_signature(world,actor,op,codes):
    return digest({'operation':op.name,'facts':facts(world,actor,op),'codes':sorted(c for c in codes if c!='UNAUTHORIZED_OPERATION')})
