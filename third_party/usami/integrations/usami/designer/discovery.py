"""Post-run event attribution. Hypotheses never become findings by assertion."""
from pathlib import Path
from types import SimpleNamespace
from usami_cloudops.models import World, Operation
from usami_cloudops.rules import POLICY, decide
from integrations.usami.adapter import NativeArtifact
from integrations.usami.domains import execution_of, native_trace
from vast_coevo.designer.storage import atomic_json, digest
from vast_coevo.designer.models import Discovery
from .compiler import Anchor, expectation
from .attribution import decision_contract, contract_key, boundary_contrast, source_contract, test_coverage


def verified_execution(adapter, evidence):
    raw = NativeArtifact.model_validate_json(Path(evidence.trace.path).read_text())
    check = adapter.replay(evidence, raw.customer_snapshot, raw.service_snapshot)
    if evidence.config.partition != 'development' or not evidence.replay.verified or not check.verified:
        raise ValueError('source must be independently replay-verified development evidence')
    trace = native_trace('cloudops', raw.native_trace, raw.track)
    return trace, execution_of('cloudops', trace, raw.track)


def anchor_from(adapter, evidence):
    trace, execution = verified_execution(adapter, evidence)
    writes = [e for e in execution.world_trace.events if e.kind=='operation' and not Operation.model_validate(e.payload['operation']).is_query]
    bad = next((e for e in writes if any(f.event_id==e.id for f in evidence.failures) or any(v.event_id==e.id and v.operation_correct is False and (v.classification=='false_rejection' or (v.successful_execution and v.authorized is False and 'UNAUTHORIZED_OPERATION' in v.reference_codes)) for v in evidence.events)), None)
    if bad is not None:
        world = bad.before; target = Operation.model_validate(bad.payload['operation']); branch='error_amplification'
    elif evidence.task_success is True and not evidence.failures and execution.oracle['status']=='pass':
        world = execution.world_trace.initial_state
        intents = [Operation.model_validate(op) for turn in execution.inputs for op in turn['grants']]
        if not intents: raise ValueError('correct source has no actual business request')
        target = intents[0]; branch='correct_to_failure'
    else: raise ValueError('source is neither verified business failure nor correctly handled task')
    world = World.model_validate(world.model_dump(mode='json'))
    return Anchor(evidence_hash=evidence.evidence_hash,world=world.model_dump(mode='json'),
        identity=execution.experiment.identity,target=target,branch=branch,service_hash=evidence.service_snapshot_hash,
        facts={'error_event_id':bad.id if bad is not None else None,'events':[e.model_dump(mode='json') for e in evidence.events],
               'failures':[f.model_dump(mode='json') for f in evidence.failures],
               'expected':expectation(world,execution.experiment.identity,target),'correct':branch=='correct_to_failure',
               'legal_completion':evidence.legal_task_success,'correct_refusal':evidence.correct_refusal,
               'source_capabilities':[op.model_dump(mode='json') for op in {digest(raw):Operation.model_validate(raw) for turn in execution.inputs for raw in turn['grants']}.values()],
               'actual_service_messages':[{k:v for k,v in m.items() if k in {'role','content','tool_calls','tool_call_id'}} for m in execution.messages if m.get('role')!='system'],
               'actual_tool_results':[t.model_dump(mode='json') for t in execution.tool_records]}).checked()


class DiscoveryBank:
    def __init__(self, root):
        self.root=Path(root); self.anchors={}; self.records=[]; self.contexts={}

    def add_anchor(self, adapter, evidence):
        anchor=anchor_from(adapter,evidence)
        atomic_json(self.root/'anchors'/(anchor.anchor_hash+'.json'),anchor,immutable=True)
        self.anchors[anchor.anchor_hash]=anchor
        return anchor

    def evaluate(self, adapter, evidence, scene, trial_id):
        trace, execution=verified_execution(adapter,evidence)
        if (execution.world_trace.initial_state.state_hash != digest(scene.world) or
                execution.experiment.identity != scene.proposal.identity):
            raise ValueError('execution changed compiled world or identity')
        states={execution.world_trace.initial_state.state_hash:execution.world_trace.initial_state}
        for event in execution.world_trace.events:
            if event.kind!='operation':continue
            states[event.before.state_hash]=event.before; states[event.after.state_hash]=event.after
        target=scene.proposal.target; triggered=False
        for turn in getattr(trace,'turns',[]):
            action=turn['action']; accepted=any(Operation.model_validate(x['operation'])==target and x['disposition']=='accepted' for x in turn['intents'])
            if target in [Operation.model_validate(o) for o in action.get('requested_operations',[])] and accepted:
                state=states[turn['before_hash']]
                if expectation(state,scene.proposal.identity,target)==scene.reference_after: triggered=True
        direct=[];direct_business=0;contracts=[]
        for event in execution.world_trace.events:
            if event.kind!='operation':continue
            if (Operation.model_validate(event.payload['operation'])==target and
                    expectation(event.before,scene.proposal.identity,target)==scene.reference_after):
                business_errors=[f for f in evidence.failures if f.event_id==event.id and f.permission_scope.get('authorized') is True]
                direct.extend(business_errors);direct_business+=len(business_errors)
                for verified in evidence.events:
                    if verified.event_id==event.id:
                        contract=decision_contract(verified,scene.reference_after)
                        if contract:contracts.append(contract)
                    if (verified.event_id==event.id and verified.classification=='false_rejection'
                            and verified.operation_correct is False and verified.authorized is True):
                        direct.append(SimpleNamespace(fingerprint=digest({'operation':target,'prestate':verified.before_hash,'reference_codes':verified.reference_codes,'classification':'false_rejection'})))
        failure=direct[0] if direct and triggered else None
        # An outcome contract is a candidate grouping, never an independent cause by itself.
        contract=contracts[0] if failure and contracts else None
        root=contract_key(target,contract)
        test_key=digest({'world':scene.world,'identity':scene.proposal.identity,'target':target})
        duplicate=any(ctx['test_key']==test_key for ctx in self.contexts.values())
        prior=[r for r in self.records if root and r.root_key==root and r.service_hash==evidence.service_snapshot_hash
               and r.direct_failure and r.evidence_hash!=evidence.evidence_hash]
        contrasts=boundary_contrast(scene,evidence.service_snapshot_hash,self.records,self.contexts,contract) if contract else []
        confirmed=bool(root and prior and contrasts)
        source=source_contract(scene)
        coverage=test_coverage(scene,contract) if failure else None
        source_matches=bool(failure and source==contract and contract and scene.anchor.branch=='error_amplification' and
            scene.anchor.service_hash==evidence.service_snapshot_hash and scene.anchor.identity==scene.proposal.identity and
            scene.anchor.target==target and {digest(op) for op in scene.proposal.capabilities}=={digest(op) for op in scene.anchor.facts['source_capabilities']})
        controlled=bool(failure and contract and scene.anchor.branch=='error_amplification' and
            source==contract and scene.anchor.service_hash==evidence.service_snapshot_hash and
            scene.anchor.identity==scene.proposal.identity and scene.anchor.target==target and
            {digest(op) for op in scene.proposal.capabilities}=={digest(op) for op in scene.anchor.facts['source_capabilities']} and
            scene.diff and scene.reference_before==scene.reference_after)
        repeated=bool(failure and (source_matches or controlled or prior or
            any(r.fingerprint==failure.fingerprint and r.service_hash==evidence.service_snapshot_hash for r in self.records)))
        reproducible=bool(failure and (source_matches or any(r.coverage_key==coverage for r in prior)))
        root_state='confirmed' if confirmed else 'candidate' if root else 'none'
        business=len(evidence.failures)
        scope=sum(e.successful_execution and e.authorized is False and 'UNAUTHORIZED_OPERATION' in e.reference_codes for e in evidence.events)
        verified_error=bool(business or scope or any(e.classification=='false_rejection' and e.operation_correct is False for e in evidence.events))
        status=('confirmed_new_root_cause' if confirmed and not source_matches and not any(r.root_state=='confirmed' for r in prior) else
                'reproduced_failure' if repeated or (confirmed and any(r.root_state=='confirmed' for r in prior)) else 'candidate_new_root_cause' if root else
                'verified_failure' if verified_error else 'triggered_correct' if triggered and evidence.task_success else 'triggered_unresolved' if triggered else 'valid_not_triggered')
        row=Discovery(trial_id=trial_id,evidence_hash=evidence.evidence_hash,scene_hash=scene.scene_hash,status=status,
            expected_kind=scene.certificate['kind'],known_failure_test=scene.anchor.branch=='error_amplification',
            business_operations=sum(e.kind=='operation' and not Operation.model_validate(e.payload['operation']).is_query for e in execution.world_trace.events),
            construction_valid=True,triggered=triggered,verified_failure=verified_error,direct_failure=bool(failure),
            fingerprint=failure.fingerprint if failure else None,root_key=root,root_state=root_state,
            attribution_version='decision-contract-v2',contract=contract,
            scope_violations=scope,
            failure_kind=contract['kind'] if contract else 'scope_violation' if any(e.authorized is False and e.successful_execution for e in evidence.events) else 'task_incomplete' if triggered and not evidence.task_success else None,
            controlled_reproduction=controlled,reproducible_failure=reproducible,coverage_key=coverage,
            root_reason='same decision contract + executed one-field opposite Reference contrast + independent same-Service reproduction' if confirmed else
                        'fingerprint and policy labels are hypotheses; controlled contrast and reproduction still required' if root else '',
            duplicate=duplicate,service_hash=evidence.service_snapshot_hash,correct_legal=evidence.legal_task_success is True,
            correct_refusal=evidence.correct_refusal is True,business_violations=business,
            direct_business_violations=direct_business if triggered else 0,incidental_business_violations=business-(direct_business if triggered else 0),source_hash=scene.anchor.evidence_hash)
        self.records.append(row); self.contexts[trial_id]={'test_key':test_key,'contrast_evidence':contrasts,'scene':scene.model_dump(mode='json')}
        atomic_json(self.root/'discoveries'/(digest(trial_id)+'.json'),{'record':row,'context':self.contexts[trial_id]},immutable=True)
        return row

    def fork(self, root):
        bank=type(self)(root); bank.anchors=dict(self.anchors); bank.records=list(self.records); bank.contexts=dict(self.contexts)
        return bank
