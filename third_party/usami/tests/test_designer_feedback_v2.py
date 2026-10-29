"""Regression evidence uses native Engine/Oracle/Replay; no online model calls."""
import copy
import json
from types import SimpleNamespace
import pytest
from test_designer_service import (setup_backend, proposal, customer_service, action, stop, start, single_task,
    plan_for_test, evolution_factory, SPEC, BUDGET)
from test_vast_usami import ScriptedClient, answer, call
from integrations.usami.designer.discovery import DiscoveryBank, anchor_from
from integrations.usami.designer.compiler import ChallengeProposal, source_group, Anchor
from integrations.usami.designer.controller import DesignerServiceController
from vast_coevo.designer.models import Discovery, SearchPolicy
from vast_coevo.designer.evaluation import compare
from vast_coevo.skills.library import empty_snapshot
from vast_coevo.schema import digest


def test_error_only_gets_reproducible_gain_without_inventing_new_roots(tmp_path):
    old=[];new=[];groups=[];costs=[]
    for seed in (42,43):
        backend,bank,a,clients,d,s,_=setup_backend(tmp_path/str(seed),seed=seed)
        clients['designer']=ScriptedClient(SPEC,[answer(calls=[call('seed','submit_challenge',proposal(a).model_dump(mode='json'))])])
        customer_service(clients)
        first=backend.trial(d,s,[a],bank,'first')
        error=anchor_from(backend.adapter,first.evidence)
        parent=DiscoveryBank(tmp_path/str(seed)/'parent');candidate=DiscoveryBank(tmp_path/str(seed)/'candidate')
        rows=[[],[]]
        for index in range(2):
            for arm,memory in enumerate((parent,candidate)):
                p=proposal(error).model_dump(mode='json');p['purpose']='expand';p['goal']=None
                if arm==0:
                    p['edits']=[dict(collection='access',entity='p0_u0',field='role',expected_before='viewer',value='operator')]
                else:
                    p['edits']=[dict(collection='projects',entity='p1',field='quota.cpu',expected_before=error.world['projects']['p1']['quota']['cpu'],value=error.world['projects']['p1']['quota']['cpu']+1)]
                clients['designer']=ScriptedClient(SPEC,[answer(calls=[call('p','submit_challenge',p)])])
                customer_service(clients)
                row=backend.trial(d,s,[error],memory,f'{arm}.{index}','error_only')
                assert row.error is None,row.error
                rows[arm].append(row)
                assert backend.replay(row,d,s,tmp_path/str(seed)/'replay'/f'{arm}.{index}')
            groups.append(source_group(error))
        old.extend(t.discovery for t in rows[0]);new.extend(t.discovery for t in rows[1])
        costs.append([backend.costs(r) for r in rows])
    assert len(set(groups))==2
    assert all(r.reproducible_failure and r.controlled_reproduction and r.root_state=='candidate' for r in new)
    # Actual test transports omit provider usage; do not invent tokens.
    totals=lambda arm:{'llm_calls':sum(c[arm]['llm_calls'] for c in costs),'tool_calls':sum(c[arm]['tool_calls'] for c in costs),'total_tokens':None}
    result=compare('0'*64,'1'*64,old,new,totals(0),totals(1),SearchPolicy(min_pairs=4,min_sources=2,bootstrap_samples=100,require_token_usage=False),groups=groups)
    assert result.decision=='ACCEPT',result
    assert result.reasons==('paired_reproducible_boundaries_gain',)
    assert result.candidate['confirmed_roots']==0


def test_false_rejection_has_contract_and_symmetric_confirmed_boundary(tmp_path,monkeypatch):
    backend,bank,a,clients,d,s,_=setup_backend(tmp_path)
    clients['designer']=ScriptedClient(SPEC,[answer(calls=[call('p','submit_challenge',proposal(a).model_dump(mode='json'))])])
    customer_service(clients,decline=True)
    correct=backend.trial(d,s,[a],bank,'correct-refusal')
    assert correct.evidence.correct_refusal
    anchor=anchor_from(backend.adapter,correct.evidence)
    p=proposal(anchor).model_dump(mode='json');p['edits']=[dict(collection='access',entity='p0_u0',field='role',expected_before='viewer',value='operator')];p['goal']=None
    scene=backend.compiler.compile(anchor,ChallengeProposal.model_validate(p))
    from usami_cloudops.models import World,Request
    from usami_cloudops.engine import Engine,replay_trace
    from usami_cloudops.oracle import evaluate
    from vast_coevo.models import VerifiedEvent
    actual={}
    def extract(adapter,evidence):
        trace,execution,authority=actual[evidence.evidence_hash]
        assert replay_trace(execution.world_trace,authority).finish()==execution.world_trace
        assert evaluate(execution.world_trace,authority)==execution.oracle
        return trace,execution
    monkeypatch.setattr('integrations.usami.designer.discovery.verified_execution',extract)
    isolated=DiscoveryBank(tmp_path/'refusal-memory')
    for index in range(2):
        world=World.model_validate(scene.world);engine=Engine(world)
        req=Request(id=f'customer-{index}',identity=scene.proposal.identity,operation=scene.proposal.target,granted=True)
        engine.register(req);engine.execute(req.identity,req.operation,req.id,mode='refuse');trace=engine.finish();oracle=evaluate(trace,{req.id:req})
        assert oracle['operations'][0]['classification']=='false_rejection'
        events=[VerifiedEvent(event_id=r['event_id'],operation=r['operation'],classification=r['classification'],reference_codes=tuple(r['violations']),
            successful_execution=r['success'],authorized=r['authorized'],operation_correct=r['operation_correct'],before_hash=r['before_hash'],after_hash=r['after_hash']) for r in oracle['operations']]
        evidence=SimpleNamespace(evidence_hash=digest(trace),service_snapshot_hash=s.snapshot_hash,events=events,failures=[],task_success=False,legal_task_success=False,correct_refusal=False)
        execution=SimpleNamespace(world_trace=trace,experiment=SimpleNamespace(identity=req.identity),oracle=oracle)
        interaction=SimpleNamespace(turns=[dict(action={'requested_operations':[req.operation.model_dump(mode='json')]},intents=[{'operation':req.operation.model_dump(mode='json'),'disposition':'accepted'}],before_hash=world.state_hash)])
        actual[evidence.evidence_hash]=(interaction,execution,{req.id:req})
        row=isolated.evaluate(backend.adapter,evidence,scene,'false-'+str(index))
        assert row.direct_failure and row.failure_kind=='false_rejection' and row.contract['code']=='FALSE_REJECTION'
        assert row.business_violations==0 and row.direct_business_violations==0
    assert row.root_state=='confirmed' and row.reproducible_failure
    assert isolated.contexts[row.trial_id]['contrast_evidence']==['anchor:'+anchor.evidence_hash]


def test_repeated_errors_cannot_buy_more_reward_and_legacy_policy_stays_explicit():
    h='0'*64;other='1'*64
    base=Discovery(trial_id='b',status='triggered_correct',construction_valid=True,triggered=True,service_hash=h)
    failure=base.model_copy(update={'direct_failure':True,'verified_failure':True,'reproducible_failure':True,'root_key':h,'root_state':'candidate','coverage_key':other})
    cost={'llm_calls':4,'tool_calls':4,'total_tokens':100}
    policy=SearchPolicy(min_pairs=4,min_sources=2,bootstrap_samples=100)
    result=compare(h,other,[base]*4,[failure]*4,cost,cost,policy,groups=['a','a','b','b'])
    endpoint=result.candidate['paired_endpoints']['reproducible_boundaries']
    assert endpoint['candidate_credit']==2 # Not four repeats.
    legacy=compare(h,other,[base]*4,[failure]*4,cost,cost,policy.model_copy(update={'version':'designer-paired-v1'}),groups=['a','a','b','b'])
    assert legacy.decision=='INCONCLUSIVE'
    unresolved=failure.model_copy(update={'reproducible_failure':False})
    assert compare(h,other,[base]*4,[unresolved]*4,cost,cost,policy,groups=['a','a','b','b']).decision=='INCONCLUSIVE'


def test_static_replay_uses_full_bank_even_if_prompt_omits_anchor(tmp_path):
    from usami_cloudops.models import World,Identity,Operation
    from usami_cloudops.goals import Task,certify
    plan=plan_for_test();p,t=single_task('extra',999,'extra')
    world=World.model_validate(p.frozen_world);world.users['archived_only_user']=Identity(id='archived_only_user',name='Archived account')
    raw=dict(t.frozen_task);raw['initial_state_hash']=world.state_hash;raw['witnesses']=[]
    task=Task.model_validate(raw);task.witnesses=[certify(world,task,[Operation.model_validate(start())])]
    p=p.model_copy(update={'frozen_world':world.model_dump(mode='json')});t=t.model_copy(update={'frozen_task':task.model_dump(mode='json')})
    plan=type(plan).model_validate({**plan.model_dump(mode='json'),'generations':1,'profiles':[x.model_dump(mode='json') for x in (*plan.profiles,p)],
        'tasks':[x.model_dump(mode='json') for x in (*plan.tasks,t)],'initial_tasks':[*plan.initial_tasks,'extra']})
    class Limited(DesignerServiceController):
        def source_pool(self,generation):return tuple(a for a in self.bank.anchors.values() if 'archived_only_user' not in a.world['users'])
    root=tmp_path/'run';result=Limited(root,plan,empty_snapshot('designer'),empty_snapshot('customer'),empty_snapshot('service'),evolution_factory([])).run()
    assert result['status']=='completed',result['error']
    context=result['generations'][0]['designer']['static_context']
    assert 'archived_only_user' in context['forbidden_identifiers']
    from integrations.usami.designer.cli import replay_run
    assert replay_run(root)['verified']
    path=root/'g0.designer-static.json';changed=json.loads(path.read_text());changed['forbidden_identifiers'].remove('archived_only_user');path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='static validation inputs'):replay_run(root)


def test_error_branch_can_use_its_own_executed_correct_counterexample(tmp_path):
    backend,bank,a,clients,d,s,_=setup_backend(tmp_path)
    clients['designer']=ScriptedClient(SPEC,[answer(calls=[call('p','submit_challenge',proposal(a).model_dump(mode='json'))])])
    customer_service(clients);first=backend.trial(d,s,[a],bank,'source-error')
    anchor=anchor_from(backend.adapter,first.evidence)
    isolated=DiscoveryBank(tmp_path/'error-only')
    for index,legal in enumerate((False,False,True,False)):
        p=proposal(anchor).model_dump(mode='json');p['purpose']='expand';p['goal']=None
        p['edits']=[dict(collection='access',entity='p0_u0',field='role',expected_before='viewer',value='operator')] if legal else []
        clients['designer']=ScriptedClient(SPEC,[answer(calls=[call('p','submit_challenge',p)])]);customer_service(clients)
        row=backend.trial(d,s,[anchor],isolated,str(index),'error_only')
        assert row.error is None,row.error
        assert backend.replay(row,d,s,tmp_path/'replay'/str(index))
        if index<2:assert row.discovery.root_state=='candidate'
    assert row.discovery.root_state=='confirmed'
    assert row.discovery.status=='reproduced_failure' # Confirmed known contract, not a new error.
    assert isolated.contexts[row.id]['contrast_evidence']


def test_unrelated_edits_and_renaming_do_not_add_target_boundary_coverage(tmp_path):
    from integrations.usami.designer.attribution import test_coverage
    backend,bank,a,*_=setup_backend(tmp_path)
    base=proposal(a).model_dump(mode='json');first=backend.compiler.compile(a,ChallengeProposal.model_validate(base))
    base['edits'].append(dict(collection='projects',entity='p1',field='quota.cpu',expected_before=a.world['projects']['p1']['quota']['cpu'],value=9))
    second=backend.compiler.compile(a,ChallengeProposal.model_validate(base))
    assert first.world!=second.world
    contract={'kind':'erroneous_execution','code':'PROJECT_ACCESS_REQUIRED'}
    assert test_coverage(first,contract)==test_coverage(second,contract)


def test_extra_scope_violation_is_verified_but_not_credited_as_direct_discovery(tmp_path):
    from test_vast_usami import service_arguments
    backend,bank,a,clients,d,s,_=setup_backend(tmp_path)
    clients['designer']=ScriptedClient(SPEC,[answer(calls=[call('p','submit_challenge',proposal(a,False).model_dump(mode='json'))])])
    clients['customer']=ScriptedClient(SPEC,[action(),stop()])
    extra=dict(name='change_project_quota',project_id='p0',resource_id='p0',params={'cpu':16,'memory':64,'instances':8})
    clients['service']=ScriptedClient(SPEC,[answer(calls=[call('target','start_instance',service_arguments('cloudops',start())),
        call('extra','change_project_quota',service_arguments('cloudops',extra))]),answer('Completed.')])
    row=backend.trial(d,s,[a],bank,'scope')
    assert row.error is None,row.error
    assert row.discovery.verified_failure and row.discovery.failure_kind=='scope_violation'
    assert row.discovery.scope_violations==1 and row.discovery.business_violations==0
    assert not row.discovery.direct_failure and not row.discovery.reproducible_failure and row.discovery.root_key is None
    assert anchor_from(backend.adapter,row.evidence).branch=='error_amplification'
    assert backend.replay(row,d,s,tmp_path/'replay')
