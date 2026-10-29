"""Hotel workflow/state strata, without seed or numeric task replicas."""
from collections import Counter,defaultdict
import json
from pathlib import Path
from usami.service.models import ModelSettings
from usami_harness.models import Budget,digest
from usami_harness.goals import execute
from . import adapter
from .goals import make_task,progress_task
from .generator import generate_world
from .models import TOOLS,VERSION
from .benchmark import Instance,Manifest,runtime_hashes,write_json

TRAIN=(
    'booking_confirmation','booking_confirmation:guest','booking_confirmation:suite','booking_confirmation:prepaid',
    'arrival_preparation','guest_check_in','guest_check_in:frontdesk','guest_check_in:choice','room_swap',
    'departure_turnover','booking_date_change','booking_upgrade','booking_downgrade','booking_cancellation',
    'booking_cancellation:late','booking_cancellation:prepaid','cancellation_refund','cancellation_refund:late',
    'deposit_completion','hold_release','hold_expiration','maintenance_recovery','calendar_reopening','calendar_capacity',
    'room_reassignment','booking_reallocation','hold_reallocation','expired_hold_reallocation','guest_identity',
    'room_assignment','room_unassignment','room_cleaning','room_cleaning:housekeeping','maintenance_scheduling',
    'sales_restriction','checkout','occupied_room_link','balance_payment','rate_date_requote',
    'booking_confirmation:assigned','booking_date_change:arrival','booking_downgrade:arrival',
    'hold_release:refund','hold_release:guest_refund','maintenance_recovery:arrival',
    'booking_reallocation:arrival','expired_hold_reallocation:arrival',
    'deposit_refusal','deposit_refusal:prepaid','refund_refusal','overpay_refusal','payment_key_refusal',
    'expired_confirmation_refusal','early_expiration_refusal','occupied_clean_refusal','dirty_assignment_refusal',
    'maintenance_assignment_refusal','overbooking_refusal','overlap_refusal','scope_refusal','staff_role_refusal',
    'staff_role_refusal:housekeeping','guest_staff_refusal','occupancy_refusal','early_checkin_refusal','identity_refusal',
    'identity_refusal:declined','balance_refusal','date_conflict_refusal','calendar_reduction_refusal','calendar_closure_refusal',
    'maintenance_lock_refusal','cancel_stay_refusal','swap_conflict_refusal','cross_type_refusal','unassign_stay_refusal',
)
HELDOUT=('complete_stay','complete_stay:choice','rate_date_requote:capacity','arrival_turnover_rebooking',
         'calendar_reopening:arrival','calendar_capacity:arrival','booking_reallocation:refund_arrival')


def candidates(seed=51001):
    rows=[];seen=set()
    for heldout,scenarios in [(False,TRAIN),(True,HELDOUT)]:
        for scenario in scenarios:
            world,task=make_task(seed,scenario);variants=[(scenario,world,task)]
            prefixes=set()
            for route,witness in enumerate(task.witnesses):
                for phase in range(1,len(witness.operations)):
                    initial=execute(adapter,world,task.identity,witness.operations[:phase]).final_state
                    key=digest(initial)
                    if key in prefixes:continue
                    prefixes.add(key)
                    state,remaining=progress_task(world,task,route,phase)
                    variants.append((f'{scenario}@{route}@{phase}',state,remaining))
            for recipe,state,remaining in variants:
                key=digest({'world':state.model_dump(mode='json'),'identity':remaining.identity,'goal':remaining.goal.model_dump(mode='json'),
                    'capabilities':remaining.capabilities,'kind':remaining.kind,'refusal':remaining.refusal_code})
                if key in seen:continue
                seen.add(key);rows.append((heldout,Instance(id=remaining.id,world=state.model_dump(mode='json'),initial_hash=state.state_hash,
                    identity=remaining.identity,task=remaining,generation_recipe=recipe)))
            print(f'{scenario}: {len(variants)} distinct certified initial progress states; cumulative {len(rows)}',flush=True)
    return rows


def coverage(manifests,candidate_count):
    instances=[i for m in manifests for i in m.instances];tasks=[i.task for i in instances]
    base=[i.task for i in instances if '@' not in i.generation_recipe]
    structures=Counter(t.structure_signature for t in base);semantics=Counter(t.semantic_signature for t in base)
    families=Counter(t.family for t in tasks);depths=Counter(max((w.causal_depth for w in t.witnesses),default=0) for t in tasks)
    measures={'frozen_service_tasks':len(tasks),'business_families':len(families),'independent_structures':len(structures),
        'semantic_decision_signatures':len(semantics),'three_linked_transition_tasks':sum(n for depth,n in depths.items() if depth>=2),
        'multi_path_tasks':sum(len({digest(w.operations) for w in t.witnesses})>1 for t in tasks),'max_causal_depth':max(depths,default=0)}
    targets={'frozen_service_tasks':278,'business_families':21,'independent_structures':28,'semantic_decision_signatures':47,
        'three_linked_transition_tasks':46,'multi_path_tasks':80,'max_causal_depth':4}
    train=next(m for m in manifests if m.split=='train');test=next(m for m in manifests if m.split=='test-demo')
    abilities=lambda m:{o['name'] for i in m.instances for o in i.task.capabilities}
    def atoms(m):
        # Relationship targets are anonymous entity references, not new abilities
        # whenever a second reservation creates a differently named Stay.
        def value(a):
            if a.field=='current_stay_id' and isinstance(a.value,str):return '<stay_reference>'
            return '<integer>' if type(a.value) is int else digest(a.value)
        return {(a.collection,a.field,a.comparison,value(a)) for i in m.instances for a in i.task.goal.atoms}
    direct,mediated=Counter(),Counter()
    for task in tasks:
        for witness in task.witnesses:
            for name,counter in [('direct_edges',direct),('mediated_failures',mediated)]:
                for edge in getattr(witness,name):counter[witness.operations[edge['from']]['name']+' -> '+witness.operations[edge['to']]['name']]+=1
    return {'domain':'stayops','version':VERSION,'candidate_count':candidate_count,'metrics':measures,'targets':targets,
        'quantity_gaps':{k:max(0,targets[k]-measures[k]) for k in targets},'base_goal_cases':len(base),
        'workflow_state_instance_structures':len({t.structure_signature for t in tasks}),
        'workflow_state_instance_semantics':len({t.semantic_signature for t in tasks}),
        'split_counts':{m.split:len(m.instances) for m in manifests},'family_counts':dict(families),
        'structure_frequencies':dict(structures),'semantic_frequencies':dict(semantics),
        'operation_coverage':dict(Counter(o['name'] for t in tasks for o in t.capabilities)),
        'causal_depth_distribution':dict(sorted(depths.items())),'verified_witnesses':sum(len(t.witnesses) for t in tasks),
        'policy_refusal_tasks':sum(t.kind=='policy_refusal' for t in tasks),
        'train_test_structure_overlap':sorted({i.task.structure_signature for i in train.instances}&{i.task.structure_signature for i in test.instances}),
        'test_operations_unseen_in_train':sorted(abilities(test)-abilities(train)),
        'test_atomic_goals_unseen_in_train':[list(x) for x in sorted(atoms(test)-atoms(train))],
        'direct_first_failure_edges':dict(direct),'mediated_after_failed_step_edges':dict(mediated),
        'human_review':'AWAITING HUMAN REVIEW',
        'counting_policy':'All instances use one seed. Proper actual workflow progress counts as instances, never new business families or base structures. No ID/name/number/date-only replicas.',
        'limitations':['Bounded witness omission depth is not exhaustive necessity over unlisted operations.',
            'Public test is a structural composition demo, not a blind test.','Family taxonomy awaits independent business review.']}


def review_examples(out,manifests):
    allrows=[(m.split,i) for m in manifests for i in m.instances]
    allrows.sort(key=lambda row:('@' in row[1].generation_recipe,row[1].id))
    chosen=[];seen=set()
    for key_function in [lambda s,i:(s,i.task.family),lambda s,i:(s,i.identity,max((w.causal_depth for w in i.task.witnesses),default=0),i.task.kind),lambda s,i:(i.id,)]:
        layers=set()
        for split,instance in allrows:
            key=key_function(split,instance)
            if key in layers:continue
            layers.add(key)
            if instance.id in seen:continue
            chosen.append((split,instance));seen.add(instance.id)
            if len(chosen)>=55:break
        if len(chosen)>=55:break
    examples=[]
    for split,i in chosen:
        examples.append({'id':i.id,'recipe':i.generation_recipe,'split':split,'family':i.task.family,'identity':i.identity,
            'request':i.task.message,'initial_state':i.world,'goal':i.task.goal.model_dump(mode='json'),
            'refusal_code':i.task.refusal_code,'legal_witnesses':[w.model_dump(mode='json') for w in i.task.witnesses],
            'wrong_result':'Missing joint goal, changed protected frame, room/date conflict, invalid authority or independent Oracle failure does not pass.'})
    write_json(out/'review-examples.json',examples)
    lines=['# StayOps stratified reviewer examples','','AWAITING HUMAN REVIEW. Software proofs are not human business ratings.','']
    for index,e in enumerate(examples):lines += [f'## {index+1}. {e["family"]} — {e["split"]}', '', '```json',json.dumps(e,ensure_ascii=False,indent=2),'```','']
    (out/'TASK_EXAMPLES.md').write_text('\n'.join(lines),encoding='utf-8')
    return len(examples)


def build_release(out):
    out=Path(out);rows=candidates();train_signatures={i.task.structure_signature for heldout,i in rows if not heldout}
    train_candidates=[];test=[]
    for heldout,i in rows:
        (test if heldout and i.task.structure_signature not in train_signatures else train_candidates).append(i)
    strata=defaultdict(list)
    for i in train_candidates:strata[(i.task.structure_signature,i.task.semantic_signature)].append(i)
    train=[];validation=[]
    for key,items in sorted(strata.items()):
        items.sort(key=lambda i:('@' in i.generation_recipe,i.id))
        if len(items)>1:validation.append(items[-1]);items=items[:-1]
        train.extend(items)
    if not validation:
        validation=[i for i in train if '@' in i.generation_recipe][::12];ids={i.id for i in validation};train=[i for i in train if i.id not in ids]
    contract=ModelSettings(model='offline-contract',base_url='https://example.invalid/v1',options={'max_tokens':4096,'temperature':0})
    manifests=[]
    for split,instances in [('train',train),('validation',validation),('test-demo',test)]:
        manifest=Manifest(domain='stayops',domain_version=VERSION,suite_id='stayops-v0.1-'+split,track='service_task',split=split,
            tools=list(TOOLS),budget=Budget(max_customer_turns=1,max_customer_llm_calls=0,max_service_llm_calls=24,max_service_tool_calls=48),
            service_model=contract,instances=sorted(instances,key=lambda i:i.id),source_hashes=runtime_hashes()).seal()
        write_json(out/f'service-{split}.json',manifest);manifests.append(manifest)
    world=generate_world(59001)
    discovery=Manifest(domain='stayops',domain_version=VERSION,suite_id='stayops-v0.1-discovery',track='customer_discovery',split='discovery',mode='sandbox',
        tools=list(TOOLS),budget=Budget(),service_model=contract,customer_model=contract,customer_method='adaptive',trials=['r0','r1','r2'],
        instances=[Instance(id='stayops_world_59001_u0',world=world.model_dump(mode='json'),initial_hash=world.state_hash,identity='u0')],source_hashes=runtime_hashes()).seal()
    write_json(out/'discovery-demo.json',discovery)
    audit=coverage(manifests,len(rows));audit['human_review_ready_examples']=review_examples(out,manifests);write_json(out/'coverage.json',audit)
    return audit
