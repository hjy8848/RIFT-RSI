"""Stratified executable workflow-state tasks; never seed-only replicas."""
from collections import Counter, defaultdict
from pathlib import Path
import json
from usami.service.models import ModelSettings
from usami_harness.models import Budget, digest
from .goals import make_task, progress_task
from .generator import generate_world
from .benchmark import Instance, Manifest, runtime_hashes, write_json
from .models import TOOLS, VERSION

TRAIN = (
    'paid_fulfillment','paid_fulfillment:picker','paid_fulfillment:delivered','paid_fulfillment:created',
    'supply_replenishment','quality_release','reservation_reallocation','cancel_reallocation','label_reallocation',
    'regional_transfer','packed_cancellation','return_restock','return_restock:refund','return_restock:receiving',
    'return_refund','return_refund:merchant_fault','return_withdrawal','draft_amendment',
    'partial_cancel_fulfillment','capacity_recovery','receiving_recovery','capacity_turnover','delivery_confirmation',
    'partial_fulfillment','order_cancellation','order_cancellation:allocated','order_cancellation:paid',
    'order_cancellation:refund','allocation_release','payment_completion','multi_line_fulfillment','split_fulfillment',
    'warehouse_scope_refusal','staff_role_refusal','staff_role_refusal:buyer','stock_refusal','packing_refusal',
    'packing_refusal:picked','shipped_cancel_refusal','shipped_cancel_refusal:delivered','return_quantity_refusal',
    'unearned_refund_refusal','unearned_refund_refusal:paid','payment_key_refusal','committed_edit_refusal',
    'quality_refusal','transfer_scope_refusal','transfer_access_refusal','receiving_refusal','capacity_refusal',
    'product_admission_refusal','return_window_refusal','nonreturnable_refusal','damaged_refund_refusal',
    'undelivered_return_refusal','capacity_reduction_refusal','picked_release_refusal','closed_charge_refusal',
)
HELDOUT = ('fulfillment_return','return_replacement','draft_amendment:fulfill','regional_transfer:closed',
           'regional_transfer:capacity','packed_cancellation:refund','partial_cancel_fulfillment:refund',
           'inbound_recovery','return_restock:capacity')


def candidates(seed=41001):
    """Base joint goals plus real partial-progress states of certified workflows.

    All candidates use one seed: changed labels/IDs/numbers never increase task
    count. Alternative prefixes are deduplicated by exact world, goal, authority
    and capabilities. They are operational phase cases, not new business families.
    """
    rows,seen=[],set()
    for heldout,scenarios in [(False,TRAIN),(True,HELDOUT)]:
        for scenario in scenarios:
            world,task=make_task(seed,scenario)
            variants=[(scenario,world,task)]
            if task.kind=='completion':
                prefixes=set()
                for route,witness in enumerate(task.witnesses):
                    from . import adapter
                    from usami_harness.goals import execute
                    for phase in range(1,len(witness.operations)):
                        trace=execute(adapter,world,task.identity,witness.operations[:phase])
                        key=digest(trace.final_state)
                        if key in prefixes:
                            continue
                        prefixes.add(key)
                        recipe=f'{scenario}@{route}@{phase}'
                        initial,conditioned=progress_task(world,task,route,phase)
                        variants.append((recipe,initial,conditioned))
            for recipe,initial,conditioned in variants:
                key=digest({'world':initial.model_dump(mode='json'),'identity':conditioned.identity,
                    'goal':conditioned.goal.model_dump(mode='json'),'capabilities':conditioned.capabilities,
                    'kind':conditioned.kind,'refusal':conditioned.refusal_code})
                if key in seen:
                    continue
                seen.add(key)
                rows.append((heldout,Instance(id=conditioned.id,world=initial.model_dump(mode='json'),
                    initial_hash=initial.state_hash,identity=conditioned.identity,task=conditioned,generation_recipe=recipe)))
            print(f'{scenario}: {len(variants)} verified candidate states; cumulative {len(rows)}',flush=True)
    return rows


def coverage(manifests,candidate_count=None):
    tasks=[i.task for m in manifests for i in m.instances]
    base_tasks=[i.task for m in manifests for i in m.instances if '@' not in i.generation_recipe]
    structures=Counter(t.structure_signature for t in base_tasks)
    semantics=Counter(t.semantic_signature for t in base_tasks)
    phase_structures=Counter(t.structure_signature for t in tasks)
    phase_semantics=Counter(t.semantic_signature for t in tasks)
    families=Counter(t.family for t in tasks)
    depths=Counter(max((w.causal_depth for w in t.witnesses),default=0) for t in tasks)
    operation_counts=Counter(op['name'] for t in tasks for op in t.capabilities)
    direct,mediated=Counter(),Counter()
    for task in tasks:
        for witness in task.witnesses:
            for collection,counter in [('direct_edges',direct),('mediated_failures',mediated)]:
                for edge in getattr(witness,collection):
                    counter[witness.operations[edge['from']]['name']+' -> '+witness.operations[edge['to']]['name']]+=1
    train=next(m for m in manifests if m.split=='train')
    test=next(m for m in manifests if m.split=='test-demo')
    def abilities(manifest):
        return {op['name'] for i in manifest.instances for op in i.task.capabilities}
    test_unknown=abilities(test)-abilities(train)
    overlap={i.task.structure_signature for i in train.instances}&{i.task.structure_signature for i in test.instances}
    measures={'frozen_service_tasks':len(tasks),'business_families':len(families),
        'independent_structures':len(structures),'semantic_decision_signatures':len(semantics),
        'three_linked_transition_tasks':sum(n for depth,n in depths.items() if depth>=2),
        'multi_path_tasks':sum(len(t.witnesses)>1 for t in tasks),'max_causal_depth':max(depths,default=0)}
    targets={'frozen_service_tasks':278,'business_families':21,'independent_structures':28,
        'semantic_decision_signatures':47,'three_linked_transition_tasks':46,'multi_path_tasks':80,'max_causal_depth':4}
    return {'domain':'fulfillment','version':VERSION,'candidate_count':candidate_count or len(tasks),
        'metrics':measures,'targets':targets,'quantity_gaps':{k:max(0,targets[k]-measures[k]) for k in targets},
        'split_counts':{m.split:len(m.instances) for m in manifests},'family_counts':dict(families),
        'structure_frequencies':dict(structures),'semantic_frequencies':dict(semantics),
        'base_goal_cases':len(base_tasks),'workflow_state_instance_structures':len(phase_structures),
        'workflow_state_instance_semantics':len(phase_semantics),
        'operation_coverage':dict(operation_counts),'causal_depth_distribution':dict(sorted(depths.items())),
        'direct_first_failure_edges':dict(direct),'mediated_after_failed_step_edges':dict(mediated),
        'verified_witnesses':sum(len(t.witnesses) for t in tasks),
        'policy_refusal_tasks':sum(t.kind=='policy_refusal' for t in tasks),
        'train_test_structure_overlap':sorted(overlap),'test_operations_unseen_in_train':sorted(test_unknown),
        'human_review':'AWAITING HUMAN REVIEW',
        'counting_policy':'Proper initial workflow progress is counted as an executable instance, never a new family. No seed/ID/amount-only replicas. Query reordering is never a causal edge.',
        'limitations':['Causal depth counts first failure omission edges in certified bounded paths; mediated downstream failures are separate.',
            'Search is explicitly finite and not a proof about arbitrary unlisted operations.',
            'Public test is a structural composition demo, not a private blind evaluation.',
            'Business-family labels are authored workflow categories awaiting independent human review.']}


def build_release(out):
    out=Path(out);rows=candidates()
    # Assign held-out joint compositions as a group; signatures also independently
    # prohibit a phase of the same structure from leaking into training.
    train_signatures={i.task.structure_signature for heldout,i in rows if not heldout}
    groups=defaultdict(list)
    for heldout,instance in rows:
        if heldout and instance.task.structure_signature not in train_signatures:
            groups['test-demo'].append(instance)
        else:
            groups['train'].append(instance)
    # Validation is sampled by actual decision/structure strata, not seed changes.
    strata=defaultdict(list)
    for instance in groups['train']:
        strata[(instance.task.structure_signature,instance.task.semantic_signature)].append(instance)
    train=[];validation=[]
    for key,instances in sorted(strata.items()):
        instances.sort(key=lambda i:('@' in i.generation_recipe,i.id))
        if len(instances)>1:
            validation.append(instances[-1]);instances=instances[:-1]
        train.extend(instances)
    if not validation:
        # A few actual state variants leave their atomic skills represented in train.
        phase_candidates=[i for i in train if '@' in i.generation_recipe]
        validation=phase_candidates[::12]
        selected={i.id for i in validation};train=[i for i in train if i.id not in selected]
    groups['train'],groups['validation']=train,validation
    contract=ModelSettings(model='offline-contract',base_url='https://example.invalid/v1',options={'max_tokens':4096,'temperature':0})
    manifests=[]
    for split in ['train','validation','test-demo']:
        manifest=Manifest(domain='fulfillment',domain_version=VERSION,suite_id='fulfillment-v0.1-'+split,
            track='service_task',split=split,tools=list(TOOLS),budget=Budget(max_customer_turns=1,max_customer_llm_calls=0,
                max_service_llm_calls=24,max_service_tool_calls=48),service_model=contract,
            instances=sorted(groups[split],key=lambda i:i.id),source_hashes=runtime_hashes()).seal()
        write_json(out/f'service-{split}.json',manifest);manifests.append(manifest)
    world=generate_world(49001)
    discovery=Manifest(domain='fulfillment',domain_version=VERSION,suite_id='fulfillment-v0.1-discovery',
        track='customer_discovery',split='discovery',mode='sandbox',tools=list(TOOLS),budget=Budget(),
        service_model=contract,customer_model=contract,customer_method='adaptive',trials=['r0','r1','r2'],
        instances=[Instance(id='fulfillment_world_49001_u0',world=world.model_dump(mode='json'),
                            initial_hash=world.state_hash,identity='u0')],source_hashes=runtime_hashes()).seal()
    write_json(out/'discovery-demo.json',discovery)
    audit=coverage(manifests,len(rows));write_json(out/'coverage.json',audit)
    # Layered selection prioritizes business families, role/decision strata and
    # both split types. No invented human score is assigned.
    selected=[];seen=set()
    for selector in [lambda m,i:(i.task.family,m.split),lambda m,i:(i.task.semantic_signature,m.split),
                     lambda m,i:(max((w.causal_depth for w in i.task.witnesses),default=0),i.identity,m.split)]:
        layers=set()
        for manifest in manifests:
            for instance in manifest.instances:
                layer=selector(manifest,instance)
                if layer not in layers and instance.id not in seen:
                    selected.append((manifest.split,instance));seen.add(instance.id);layers.add(layer)
        if len(selected)>=55:break
    examples=[]
    for split,instance in selected:
        examples.append({'id':instance.id,'split':split,'family':instance.task.family,'identity':instance.identity,
            'request':instance.task.message,'initial_state':instance.world,'goal':instance.task.goal.model_dump(mode='json'),
            'certificate':{'refusal_code':instance.task.refusal_code,'legal_witnesses':[w.model_dump(mode='json') for w in instance.task.witnesses]},
            'wrong_result':'Missing terminal goal, changed protected frame, invalid authority or independently verified business violation fails.'})
    write_json(out/'review-examples.json',examples)
    lines=['# FulfillmentOps layered reviewer examples','','AWAITING HUMAN REVIEW. Software proofs are not human business-quality ratings.','']
    for index,item in enumerate(examples):
        lines += [f'## {index+1}. {item["family"]} — {item["split"]}', '', '```json', json.dumps(item,ensure_ascii=False,indent=2),'```','']
    (out/'TASK_EXAMPLES.md').write_text('\n'.join(lines),encoding='utf-8')
    audit['human_review_ready_examples']=len(examples);write_json(out/'coverage.json',audit)
    return audit
