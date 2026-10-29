"""Seeded random baseline: same fields, proposal grammar, compiler and budgets."""
import copy
import random
from usami_cloudops.models import World
from usami_cloudops.rules import invariants
from .compiler import ChallengeProposal, SceneEdit


def random_proposal(anchors, seed, with_stats=False):
    rng=random.Random(seed);anchor=rng.choice(anchors);world=World.model_validate(anchor.world)
    raw=world.model_dump(mode='json');options=[];checks=0
    def option(changes):
        nonlocal checks
        if checks>=512:return
        changes=[(c,e,f,v) for c,e,f,v in changes if value_at(raw[c][e],f)!=v]
        if not changes or len(changes)>12:return
        checks+=1;candidate=copy.deepcopy(raw);edits=[]
        for collection,entity,field,value in changes:
            obj=candidate[collection][entity];parts=field.split('.')
            for part in parts[:-1]:obj=obj[part]
            before=obj[parts[-1]];obj[parts[-1]]=value
            edits.append(SceneEdit(collection=collection,entity=entity,field=field,expected_before=before,value=value))
        try:
            valid=World.model_validate(candidate)
            if not invariants(valid):options.append(edits)
        except ValueError:pass
    for project in world.projects.values():
        for status in ('active','suspended'):option([('projects',project.id,'status',status)])
        for field in ('cpu','memory','instances'):
            current=getattr(project.quota,field)
            for value in sorted({1,max(1,current//2),current+1,current*2}):option([('projects',project.id,'quota.'+field,value)])
    for access in world.access.values():
        for role in ('admin','operator','viewer'):option([('access',access.id,'role',role)])
        for uid in world.users:option([('access',access.id,'user_id',uid)])
        for pid in world.projects:option([('access',access.id,'project_id',pid)])
    for instance in world.instances.values():
        for status in ('stopped','running'):option([('instances',instance.id,'status',status)])
        for field in ('cpu','memory'):
            current=getattr(instance,field)
            for value in sorted({1,max(1,current//2),current+1,current*2}):option([('instances',instance.id,field,value)])
        for pid in world.projects:option([('instances',instance.id,'project_id',pid)])
        choices=[[]]+[[i.id] for i in world.instances.values() if i.id!=instance.id and i.project_id==instance.project_id]
        for deps in choices:option([('instances',instance.id,'dependencies',deps)])
    # Legal paired accounting edits are explicit proposals, never compiler repair.
    for invoice in world.invoices.values():
        stopped=[('instances',i.id,'status','stopped') for i in world.instances.values() if i.project_id==invoice.project_id]
        option([('projects',invoice.project_id,'status','suspended'),('invoices',invoice.id,'status','unpaid'),('invoices',invoice.id,'payments',[])]+stopped)
    if not options:raise ValueError('no invariant-preserving random edit within the common 512-check ceiling')
    edits=rng.choice(options)
    # All auxiliary source writes remain eligible, just as for a Designer proposal.
    capabilities=[anchor.target]
    for raw_op in anchor.facts['source_capabilities']:
        from usami_cloudops.models import Operation
        op=Operation.model_validate(raw_op)
        if op not in capabilities and len(capabilities)<10 and rng.choice((False,True)):capabilities.append(op)
    # Refusal certification deliberately requires a single exact request; the
    # common compiler will reject an inappropriate multi-write/refusal proposal.
    p=ChallengeProposal(anchor_hash=anchor.anchor_hash,branch=anchor.branch,purpose='expand' if anchor.branch=='error_amplification' else 'boundary',
        hypothesis='Seeded random local state variation; no failure claimed.',edits=tuple(edits),identity=anchor.identity,
        customer_need='Please perform my requested business operation and explain its verified result.',target=anchor.target,
        capabilities=tuple(capabilities),goal=None)
    stats={'random_invariant_checks':checks,'random_legal_options':len(options)}
    return (p,stats) if with_stats else p


def value_at(obj,path):
    for part in path.split('.'):obj=obj[part]
    return obj
