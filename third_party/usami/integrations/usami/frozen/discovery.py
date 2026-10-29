"""Task discovery attribution on real execution, without edited-world counterfactuals."""
from pathlib import Path
from usami_cloudops.models import World,Operation
from usami_cloudops.rules import decide,invariants
from vast_coevo.designer.models import Discovery
from vast_coevo.designer.storage import digest,atomic_json
from integrations.usami.designer.discovery import DiscoveryBank,verified_execution,anchor_from
from integrations.usami.designer.compiler import Anchor,expectation
from integrations.usami.designer.attribution import decision_contract,contract_key,anchor_contract
from integrations.usami.customer_probes import _coverage
from .composer import semantic_signatures


def frozen_anchor_from(adapter,evidence):
    try:return anchor_from(adapter,evidence)
    except ValueError:
        if not evidence.task_decisions:raise
    trace,execution=verified_execution(adapter,evidence)
    decision=evidence.task_decisions[0];op=Operation.model_validate(decision.operation);world=execution.world_trace.initial_state
    if decision.before_hash!=world.state_hash:raise ValueError('task-decision state differs')
    return Anchor(evidence_hash=evidence.evidence_hash,world=world.model_dump(mode='json'),identity=execution.experiment.identity,target=op,
        branch='error_amplification',service_hash=evidence.service_snapshot_hash,facts={'events':[e.model_dump(mode='json') for e in evidence.events],
        'failures':[f.model_dump(mode='json') for f in evidence.failures],'task_decisions':[d.model_dump(mode='json') for d in evidence.task_decisions],
        'error_event_id':None,'expected':expectation(world,execution.experiment.identity,op),'correct':False,
        'legal_completion':False,'correct_refusal':False,'source_capabilities':execution.inputs[-1]['grants'],
        'actual_service_messages':[{k:v for k,v in m.items() if k in {'role','content','tool_calls','tool_call_id'}} for m in execution.messages if m.get('role')!='system'],
        'actual_tool_results':[t.model_dump(mode='json') for t in execution.tool_records]}).checked()


class TaskDiscoveryBank(DiscoveryBank):
    def add_anchor(self,adapter,evidence):
        a=frozen_anchor_from(adapter,evidence)
        atomic_json(self.root/'anchors'/(a.anchor_hash+'.json'),a,immutable=True);self.anchors[a.anchor_hash]=a
        return a
    def evaluate(self,adapter,evidence,scene,trial_id):
        trace,execution=verified_execution(adapter,evidence)
        initial=execution.world_trace.initial_state
        if initial.state_hash!=digest(scene.world) or execution.experiment.identity!=scene.proposal.identity:raise ValueError('execution changed frozen pool world/identity')
        p=scene.proposal
        accepted={digest(row['operation']) for turn in trace.turns for row in turn['intents'] if row['disposition']=='accepted'}
        triggered=digest(p.target) in accepted
        verified={e.event_id:e for e in evidence.events};direct=[];scope=0
        for event in execution.world_trace.events:
            if event.kind!='operation':continue
            row=verified[event.id];op=Operation.model_validate(event.payload['operation'])
            if row.successful_execution and row.authorized is False and 'UNAUTHORIZED_OPERATION' in row.reference_codes:scope+=1
            if op not in p.capabilities or digest(op) not in accepted or invariants(event.before):continue
            d=decide(event.before,p.identity,op);ref={'allowed':d.allowed,'code':d.error_code}
            contract=decision_contract(row,ref)
            if contract and row.authorized is True:
                native=next((f for f in evidence.failures if f.event_id==event.id),None)
                fingerprint=native.fingerprint if native else digest({'operation':op,'state':row.before_hash,'contract':contract})
                direct.append((op,ref,contract,fingerprint,event.before,row))
        for decision in evidence.task_decisions or ():
            from types import SimpleNamespace
            op=Operation.model_validate(decision.operation)
            if op in p.capabilities and digest(op) in accepted:
                direct.append((op,{'allowed':True,'code':None},{'kind':'false_rejection','code':'FALSE_REJECTION'},digest(decision),initial,
                    SimpleNamespace(event_id='task-decision:'+decision.model_call_id)))
        failure=direct[0] if direct else None
        root=None;coverage=None;repro=False;confirmed=False;contrasts=[];contract=None;prior=[];source_match=False
        if failure:
            op,ref,contract,fingerprint,before,event=failure;root=contract_key(op,contract)
            boundary=_coverage(before,p.identity,op,decide(before,p.identity,op))
            coverage=digest({'contract':contract,'operation':op.name,'boundary':boundary})
            prior=[r for r in self.records if r.service_hash==evidence.service_snapshot_hash and r.direct_failure and r.evidence_hash!=evidence.evidence_hash and r.root_key==root and r.coverage_key==coverage]
            a=scene.anchor
            source_match=bool(a and a.branch=='error_amplification' and a.service_hash==evidence.service_snapshot_hash and a.target.name==op.name and anchor_contract(a)==contract and
                _coverage(World.model_validate(a.world),a.identity,a.target,decide(World.model_validate(a.world),a.identity,a.target))==boundary)
            repro=bool(source_match or prior)
            # A narrow, auditable natural identity contrast, not a universal causal
            # claim from equal rule codes. Other error families remain candidates.
            if len(p.capabilities)==1 and before.state_hash==initial.state_hash:
                if (a and a.facts['correct'] and a.service_hash==evidence.service_snapshot_hash and digest(a.world)==initial.state_hash and
                        a.identity!=p.identity and len(a.facts['source_capabilities'])==1 and Operation.model_validate(a.facts['source_capabilities'][0])==op):
                    counter=decide(initial,a.identity,op)
                    denied_code=ref['code'] if not ref['allowed'] else counter.error_code
                    if counter.allowed!=ref['allowed'] and denied_code in {'PROJECT_ACCESS_REQUIRED','PERMISSION_DENIED'}:contrasts.append('anchor:'+a.evidence_hash)
                for r in self.records:
                    ctx=self.contexts[r.trial_id];other=ctx['scene']['proposal']
                    if (r.service_hash!=evidence.service_snapshot_hash or r.verified_failure or not r.triggered or not(r.correct_legal or r.correct_refusal) or
                        ctx['scene']['certificate']['world_hash']!=initial.state_hash or other['identity']==p.identity or len(other['candidate_operations'])!=1 or
                        Operation.model_validate(other['candidate_operations'][0])!=op):continue
                    counter=decide(initial,other['identity'],op)
                    denied_code=ref['code'] if not ref['allowed'] else counter.error_code
                    if counter.allowed!=ref['allowed'] and denied_code in {'PROJECT_ACCESS_REQUIRED','PERMISSION_DENIED'}:contrasts.append(r.evidence_hash)
            confirmed=bool(prior and contrasts)
        key=digest({'world_structure':scene.world_structure_hash,'task_semantics':semantic_signatures(scene)[1]})
        duplicate=any(c['test_key']==key for c in self.contexts.values())
        scope_error=bool(scope);business=len(evidence.failures)
        verified_error=bool(business or scope or evidence.task_decisions or any(e.classification=='false_rejection' and e.operation_correct is False for e in evidence.events))
        status=('confirmed_new_root_cause' if confirmed and not source_match and not any(r.root_state=='confirmed' for r in prior) else
            'reproduced_failure' if repro else 'candidate_new_root_cause' if failure else 'verified_failure' if verified_error else
            'triggered_correct' if triggered and evidence.task_success else 'triggered_unresolved' if triggered else 'valid_not_triggered')
        direct_ids={item[-1].event_id for item in direct}
        direct_business=sum(f.event_id in direct_ids for f in evidence.failures)
        row=Discovery(trial_id=trial_id,evidence_hash=evidence.evidence_hash,scene_hash=scene.scene_hash,source_hash=scene.anchor.evidence_hash if scene.anchor else None,
            service_hash=evidence.service_snapshot_hash,status=status,construction_valid=True,triggered=triggered,verified_failure=verified_error,direct_failure=bool(failure),
            expected_kind=scene.certificate['kind'],known_failure_test=p.branch=='error_amplification',
            business_operations=sum(e.kind=='operation' and not Operation.model_validate(e.payload['operation']).is_query for e in execution.world_trace.events),
            business_violations=business,direct_business_violations=direct_business,incidental_business_violations=business-direct_business,scope_violations=scope,
            attribution_version='frozen-task-attribution-v1',contract=contract,failure_kind=contract['kind'] if contract else 'scope_violation' if scope_error else 'task_incomplete' if triggered and not evidence.task_success else None,
            root_key=root,root_state='confirmed' if confirmed else 'candidate' if root else 'none',fingerprint=failure[3] if failure else None,
            root_reason='natural existing-identity permission contrast + independent matching-boundary executions' if confirmed else 'decision contract is a candidate, not proof of shared cause' if root else '',
            reproducible_failure=repro,controlled_reproduction=False,coverage_key=coverage,duplicate=duplicate,
            correct_legal=evidence.legal_task_success is True,correct_refusal=evidence.correct_refusal is True)
        self.records.append(row);self.contexts[trial_id]={'test_key':key,'scene':scene.model_dump(mode='json'),'contrast_evidence':sorted(contrasts),'attribution_version':'frozen-task-attribution-v1',
            'task_decisions':[d.model_dump(mode='json') for d in evidence.task_decisions or ()]}
        atomic_json(self.root/'discoveries'/(digest(trial_id)+'.json'),{'record':row,'context':self.contexts[trial_id]},immutable=True)
        return row
