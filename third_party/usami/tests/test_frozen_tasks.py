"""Offline actors; original CloudOps Engine/Reference/Oracle/Replay provide all business evidence."""
import copy
import json
from pathlib import Path
import pytest
from test_designer_service import SPEC,BUDGET,FunctionalClient,action,stop,start,evolution_factory
from test_vast_usami import ScriptedClient,answer,call,service_arguments
from vast_coevo.skills.library import empty_snapshot
from vast_coevo.designer.storage import digest
from integrations.usami.frozen.pool import WorldSpec,WorldPool
from integrations.usami.frozen.composer import TaskProposal,TaskSpec,TaskComposer
from integrations.usami.frozen.controller import FrozenPlan,FrozenController,preparation
from integrations.usami.frozen.backend import FrozenSearchBackend
from integrations.usami.frozen.discovery import TaskDiscoveryBank
from integrations.usami.designer.discovery import anchor_from


def plan(**updates):
    data=dict(protocol='frozen-task-discovery-v1',designer_quality_scoring=True,designer_dynamic_gate=True,service_dynamic_gate=True,world_specs=[{'id':'dev42','seed':42}],designer_model=SPEC.model_dump(mode='json'),
        customer_model=SPEC.model_dump(mode='json'),service_model=SPEC.model_dump(mode='json'),budget=BUDGET.model_dump(mode='json'),
        generations=1,trials_per_generation=2,designer_validation_pairs=4,max_llm_calls=400,max_tool_calls=1000,
        search_policy={'version':'designer-paired-v2','min_pairs':4,'min_sources':2,'bootstrap_samples':100})
    data.update(updates);return FrozenPlan.model_validate(data)


def task(pool,identity='u0',world_id='dev42',source=None,branch='correct_to_failure'):
    entry=pool.entries[world_id]
    return TaskProposal(world_id=world_id,world_hash=entry.world_hash,identity=identity,source_evidence=source,branch=branch,
        hypothesis='Test an existing identity and project authority.',task_spec=TaskSpec(kind='completion' if identity=='u0' else 'policy_refusal',target=start()),
        candidate_operations=(start(),),customer_brief='Please start the project database and explain the result.')


def factory(log,patch=False,malformed=False):
    base=evolution_factory(log,service_patch=patch)
    def create(model,role,phase):
        if role in {'service_reflector','designer_reflector'}:
            original=base(model,role,phase)
            if role=='service_reflector' and not patch:return original
            if role=='designer_reflector':
                def reflect(req):
                    v=json.loads(req['messages'][-1]['content']);log.append((role,phase,copy.deepcopy(req)))
                    p={'action':'ADD','role':'designer','base_snapshot_hash':v['library']['snapshot_hash'],'rationale':'Prefer untested existing project roles.',
                       'skill':{'id':'designer.existing_roles','version':1,'type':'design','trigger':{'keywords':['world_window']},
                       'selection':['Select an existing identity with different project access.'],'transformations':['Compose existing operations and goals without changing the initial world.'],
                       'interpretation':['Use actual Oracle errors and distinguish repetitions from causes.'],'stop':['Stop at the registered budget.']}}
                    return answer(calls=[call('patch','submit_designer_patch',p)])
                return FunctionalClient(model,reflect)
            return original
        if role=='designer':
            def design(req):
                log.append((role,phase,copy.deepcopy(req)));v=json.loads(req['messages'][-1]['content']);entry=v['world_window'][0]
                source=next((a for a in v['anchors'] if a['facts']['correct']),v['anchors'][0])
                p=dict(world_id=entry['spec']['id'],world_hash=entry['world_hash'],identity='u1',source_evidence=source['evidence_hash'],branch=source['branch'],
                    hypothesis='Compare an existing viewer with the observed successful administrator.',
                    task_spec={'kind':'policy_refusal','target':start()},candidate_operations=[start()],customer_brief='Please start the database in my project.')
                if malformed:p['edits']=[{'field':'role','value':'viewer'}]
                return answer(calls=[call('task','submit_challenge',p)])
            return FunctionalClient(model,design)
        return base(model,role,phase)
    return create


def backend_fixture(tmp_path,specs=None):
    p=plan(world_specs=specs or [{'id':'dev42','seed':42}]);clients={}
    c=FrozenController(tmp_path/'run',p,empty_snapshot('designer'),empty_snapshot('customer'),empty_snapshot('service'),lambda m,r,ph:clients[r])
    c.root.mkdir();c.prepare_run()
    ref=p.initial_tasks[0];case=c.task_case(ref)
    clients['service']=ScriptedClient(SPEC,[answer(calls=[call('s','start_instance',service_arguments('cloudops',start()))]),answer('Done.')])
    e=c.adapter.run_service_task(case.config,ref,c.customer,c.service,'source');assert e.task_success
    c.backend.register_source(e);a=c.bank.add_anchor(c.adapter,e)
    return c,a,clients


def submit(c,a,clients,p,tid,query=False,stop_early=False,service_ops=None,decline=False,bank=None,strategy='bidirectional'):
    clients['designer']=ScriptedClient(SPEC,[answer(calls=[call('t','submit_challenge',p.model_dump(mode='json'))])])
    queries=[action('What is my project status?',False)] if query else []
    requested=answer(json.dumps({'type':'send_message','message':p.customer_brief,'requested_operations':[o.model_dump(mode='json') for o in p.capabilities],'reason':''}))
    clients['customer']=ScriptedClient(SPEC,[stop()] if stop_early else queries+[requested,stop()])
    replies=[answer(calls=[call('q','get_project',{'project_id':'p0','resource_id':'p0','params':{}})]),answer('The project is active.')] if query else []
    replies+=([answer('PROJECT_ACCESS_REQUIRED')] if decline else [answer(calls=[call('s'+str(i),o.name,service_arguments('cloudops',o.model_dump(mode='json'))) for i,o in enumerate(service_ops or p.capabilities)]),answer('Done.')])
    clients['service']=ScriptedClient(SPEC,replies)
    return c.backend.trial(c.designer,c.service,[a],bank or c.bank,tid,strategy)


def test_pool_independence_and_real_coverage_gaps(tmp_path,monkeypatch):
    def forbidden(*a,**k):raise AssertionError('conditioned make_task forbidden')
    monkeypatch.setattr('usami_cloudops.goals.make_task',forbidden)
    p=plan(world_specs=[{'id':'one','seed':42},{'id':'two','seed':43}])
    data=preparation(p.world_specs,p.budget);audit=data['audit']
    assert audit['effective_structural_worlds']==1 and audit['registered_worlds']==2
    assert all(w['multi_step_tasks']>0 and w['max_causal_depth']>=1 for w in audit['worlds'])
    assert all({'completion','policy_refusal'}<=set(w['kinds']) for w in audit['worlds'])
    assert audit['model_calls']==0 and audit['limitations']
    pool=WorldPool(p.world_specs);pool.save(tmp_path);pool.verify_saved(tmp_path)
    pool.entries['one'].world['projects']['p0']['status']='suspended'
    with pytest.raises(ValueError,match='mutated'):pool.load('one',pool.entries['one'].world_hash)
    with pytest.raises(ValueError,match='heldout'):WorldPool([WorldSpec(id='blind',seed=1,partition='heldout')])


def test_schema_has_no_world_edit_or_witness_authority(tmp_path):
    pool=WorldPool([WorldSpec(id='dev42',seed=42)]);p=task(pool)
    for field in ('world','edits','expected_before','witness'):
        with pytest.raises(ValueError):TaskProposal.model_validate({**p.model_dump(mode='json'),field:{}})
    c=TaskComposer(pool,BUDGET);before=digest(pool.entries['dev42'].world)
    assert c.compile(None,p).certificate['kind']=='completion'
    assert c.compile(None,task(pool,'u1')).certificate['reference_code']=='PROJECT_ACCESS_REQUIRED'
    assert c.compile(None,task(pool,'u2')).certificate['kind']=='policy_refusal'
    assert before==digest(pool.entries['dev42'].world)
    with pytest.raises(ValueError,match='hash'):c.compile(None,p.model_copy(update={'world_hash':'0'*64}))


def test_native_multistep_search_and_execution_never_conditions_world(tmp_path):
    c,a,clients=backend_fixture(tmp_path)
    scenes=[s for s in c.catalog if len(s.witness)>=4 and s.proposal.identity=='u0' and s.proposal.target.project_id=='p0']
    assert scenes
    p=scenes[0].proposal.model_copy(update={'source_evidence':a.evidence_hash})
    # Reverse the candidate set: the proof and execution order are not the authored list.
    p=p.model_copy(update={'candidate_operations':tuple(reversed(p.capabilities))})
    frozen=c.backend.compiler.compile(a,p);assert len(frozen.witness)>=4
    from usami_cloudops.models import Operation
    operations=[Operation.model_validate(o) for o in frozen.witness]
    row=submit(c,a,clients,p,'multi',service_ops=operations)
    assert row.error is None,row.error
    assert row.evidence.task_success and row.discovery.correct_legal
    assert row.evidence.config.initial_world_hash==c.pool.entries['dev42'].world_hash
    assert c.backend.replay(row,c.designer,c.service,tmp_path/'replay')
    assert c.pool.entries['dev42'].world==row.scene['world']


def test_nonexistent_invisible_unreachable_and_private_brief_are_rejected(tmp_path):
    pool=WorldPool([WorldSpec(id='dev42',seed=42)]);composer=TaskComposer(pool,BUDGET);p=task(pool)
    raw=p.model_dump(mode='json');raw['candidate_operations'][0]['resource_id']='missing';raw['task_spec']['target']['resource_id']='missing'
    with pytest.raises(ValueError,match='nonexistent'):composer.compile(None,TaskProposal.model_validate(raw))
    raw=p.model_dump(mode='json');raw['identity']='u3'
    with pytest.raises(ValueError,match='invisible'):composer.compile(None,TaskProposal.model_validate(raw))
    raw=p.model_dump(mode='json');raw['task_spec']['goal']={'atoms':[{'collection':'instances','entity':'p0_web','field':'status','value':'running'}],'allowed_changes':{'instances/p0_web':['status']}}
    with pytest.raises(ValueError,match='reachability'):composer.compile(None,TaskProposal.model_validate(raw))
    with pytest.raises(ValueError,match='private evaluator'):composer.compile(None,p.model_copy(update={'customer_brief':'Inspect the Oracle witness.'}))


def test_ad_hoc_cross_project_composition_is_not_limited_to_catalog_templates():
    from usami_cloudops.models import Operation
    from usami_harness.goals import JointGoal
    pool=WorldPool([WorldSpec(id='dev42',seed=42)]);p=task(pool)
    other=Operation(name='start_instance',project_id='p1',resource_id='p1_db')
    goal=JointGoal(atoms=[dict(collection='instances',entity=k,field='status',value='running') for k in ('p0_db','p1_db')],
        allowed_changes={f'instances/{k}':['status'] for k in ('p0_db','p1_db')})
    p=p.model_copy(update={'candidate_operations':(other,p.target),'task_spec':p.task_spec.model_copy(update={'goal':goal}),
        'customer_brief':'Bring both of my project databases online without changing their other resources.'})
    scene=TaskComposer(pool,BUDGET).compile(None,p)
    assert len(scene.witness)==2 and len(scene.native_task['witnesses'])>=2
    assert scene.world==pool.entries['dev42'].world


def test_correct_to_failure_uses_existing_viewers_and_keeps_customer_isolated(tmp_path):
    c,a,clients=backend_fixture(tmp_path);original=copy.deepcopy(c.pool.entries['dev42'].world);rows=[]
    for i,uid in enumerate(('u1','u2')):
        p=task(c.pool,uid,source=a.evidence_hash)
        row=submit(c,a,clients,p,str(i),query=True);assert row.error is None,row.error
        assert row.discovery.direct_failure and row.discovery.verified_failure
        rows.append(row)
        first=json.dumps(clients['customer'].requests[0]);assert 'world_window' not in first and p.hypothesis not in first and 'p3_db' not in first
        assert 'The project is active.' in clients['customer'].requests[1]['messages'][-1]['content']
        assert c.backend.replay(row,c.designer,c.service,tmp_path/'replay'/str(i))
    assert rows[1].discovery.reproducible_failure and rows[1].discovery.root_state=='confirmed'
    assert c.pool.entries['dev42'].world==original


def test_error_amplification_can_select_other_frozen_world_without_editing_it(tmp_path):
    c,a,clients=backend_fixture(tmp_path,[{'id':'dev42','seed':42},{'id':'dev43','seed':43}])
    first=submit(c,a,clients,task(c.pool,'u1',source=a.evidence_hash),'first');assert first.error is None
    error=anchor_from(c.adapter,first.evidence);c.backend.register_source(first.evidence)
    isolated=TaskDiscoveryBank(tmp_path/'isolated')
    p=task(c.pool,'u2','dev43',error.evidence_hash,'error_amplification')
    row=submit(c,error,clients,p,'cross-world',bank=isolated,strategy='error_only')
    assert row.error is None,row.error
    assert row.discovery.reproducible_failure and row.discovery.root_state=='candidate'
    assert row.scene['world']==c.pool.entries['dev43'].world and row.scene['diff']==[]
    assert c.backend.replay(row,c.designer,c.service,tmp_path/'replay')


@pytest.mark.parametrize('stop_early,decline,expected',[(True,False,'valid_not_triggered'),(False,True,'triggered_correct')])
def test_untriggered_and_correct_refusal_are_not_findings(tmp_path,stop_early,decline,expected):
    c,a,clients=backend_fixture(tmp_path)
    row=submit(c,a,clients,task(c.pool,'u1',source=a.evidence_hash),'test',stop_early=stop_early,decline=decline)
    assert row.error is None,row.error
    assert row.discovery.status==expected and not row.discovery.verified_failure


def test_recursive_frozen_protocol_and_full_replay(tmp_path,monkeypatch):
    def forbidden(*a,**k):raise AssertionError('SceneEdit compiler reached')
    monkeypatch.setattr('integrations.usami.designer.compiler.CloudOpsCompiler.compile',forbidden)
    log=[];p=plan(generations=2)
    c=FrozenController(tmp_path/'run',p,empty_snapshot('designer'),empty_snapshot('customer'),empty_snapshot('service'),factory(log))
    report=c.run();assert report['status']=='completed',report['error']
    assert all(r['designer']['dynamic']['decision']=='INCONCLUSIVE' for r in report['generations'])
    assert all('insufficient_preregistered_pairs' in r['designer']['dynamic']['reasons'] for r in report['generations'])
    assert report['fixed_customer']==empty_snapshot('customer').snapshot_hash
    assert any(role=='service_reflector' and json.loads(req['messages'][-1]['content'])['evidence'][0]['failures'] for role,phase,req in log if role=='service_reflector')
    from integrations.usami.designer.cli import replay_run
    assert replay_run(tmp_path/'run')['verified']
    file=tmp_path/'run'/'world-pool.json';raw=json.loads(file.read_text());raw['entries'][0]['world']['projects']['p0']['quota']['cpu']=64;file.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='world pool changed'):replay_run(tmp_path/'run')


@pytest.mark.parametrize('strategy',['random_legal','error_only','correct_only','bidirectional','bidirectional_fixed_designer'])
def test_five_ablations_use_the_same_pool_and_validator(tmp_path,strategy):
    c,a,clients=backend_fixture(tmp_path);p=task(c.pool,source=a.evidence_hash)
    row=submit(c,a,clients,p,strategy,strategy=strategy)
    if strategy=='error_only':assert row.error and not row.calls
    else:
        assert row.scene is not None,row.error
        assert row.scene['world']==c.pool.entries['dev42'].world
        assert c.backend.replay(row,c.designer,c.service,tmp_path/'replay')


def test_billing_recovery_is_service_execution_not_initial_world_edit(tmp_path):
    c,a,clients=backend_fixture(tmp_path)
    scene=next(s for s in c.catalog if s.proposal.target.project_id=='p2' and s.certificate['kind']=='completion' and any(o.name=='pay_invoice' for o in s.proposal.capabilities))
    initial=copy.deepcopy(scene.world)
    assert initial['projects']['p2']['status']=='suspended' and initial['invoices']['bill2']['status']=='unpaid'
    assert 'p2_u0' not in initial['access']
    from usami_cloudops.goals import execute
    from usami_cloudops.models import World,Operation
    trace=execute(World.model_validate(initial),scene.proposal.identity,[Operation.model_validate(o) for o in scene.witness])
    assert trace.initial_state.model_dump(mode='json')==initial
    assert trace.final_state.projects['p2'].status=='active' and trace.final_state.instances['p2_db'].status=='running'
    assert c.pool.entries['dev42'].world==initial


def test_false_rejection_uses_natural_identity_contrast_and_real_oracle(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from usami_cloudops.models import World,Request
    from usami_cloudops.engine import Engine,replay_trace
    from usami_cloudops.oracle import evaluate
    from vast_coevo.models import VerifiedEvent
    c,a,clients=backend_fixture(tmp_path)
    correct=submit(c,a,clients,task(c.pool,'u1',source=a.evidence_hash),'correct',decline=True)
    assert correct.discovery.correct_refusal
    p=task(c.pool,'u0',source=a.evidence_hash)
    scene=c.backend.compiler.compile(a,p);actual={}
    def extract(adapter,e):
        interaction,execution,authority=actual[e.evidence_hash]
        assert replay_trace(execution.world_trace,authority).finish()==execution.world_trace
        assert evaluate(execution.world_trace,authority)==execution.oracle
        return interaction,execution
    monkeypatch.setattr('integrations.usami.frozen.discovery.verified_execution',extract)
    for index in range(2):
        world=World.model_validate(scene.world);engine=Engine(world)
        req=Request(id='refuse-'+str(index),identity='u0',operation=p.target);engine.register(req)
        engine.execute('u0',p.target,req.id,mode='refuse');trace=engine.finish();oracle=evaluate(trace,engine.requests)
        events=[VerifiedEvent(event_id=r['event_id'],operation=r['operation'],classification=r['classification'],reference_codes=tuple(r['violations']),
            successful_execution=r['success'],authorized=r['authorized'],operation_correct=r['operation_correct'],before_hash=r['before_hash'],after_hash=r['after_hash']) for r in oracle['operations']]
        e=SimpleNamespace(evidence_hash=digest(trace),service_snapshot_hash=c.service.snapshot_hash,events=events,failures=[],task_decisions=(),task_success=False,legal_task_success=False,correct_refusal=False)
        execution=SimpleNamespace(world_trace=trace,experiment=SimpleNamespace(identity='u0'))
        execution.oracle=oracle
        interaction=SimpleNamespace(turns=[{'intents':[{'operation':p.target.model_dump(mode='json'),'disposition':'accepted'}]}])
        actual[e.evidence_hash]=(interaction,execution,engine.requests)
        row=c.bank.evaluate(c.adapter,e,scene,'refusal-'+str(index))
        assert row.failure_kind=='false_rejection' and row.direct_failure and row.business_violations==0
    assert row.root_state=='confirmed' and row.reproducible_failure


def test_original_service_gate_and_budget_remain_active(tmp_path):
    log=[];p=plan(trials_per_generation=1,strategy='bidirectional_fixed_designer')
    c=FrozenController(tmp_path/'run',p,empty_snapshot('designer'),empty_snapshot('customer'),empty_snapshot('service'),factory(log,patch=True))
    result=c.run();assert result['status']=='completed',result['error']
    service=result['generations'][0]['service']
    assert service['outcome']=='rejected' and 'known_business_failure_not_repaired' in service['reasons']
    from integrations.usami.designer.cli import replay_run
    assert replay_run(tmp_path/'run')['verified']
    limited=FrozenController(tmp_path/'budget',plan(max_llm_calls=1),empty_snapshot('designer'),empty_snapshot('customer'),empty_snapshot('service'),factory([]))
    report=limited.run();assert report['status']=='blocked' and report['cost']['llm_calls']==1


def test_independent_frozen_evaluation_never_updates_snapshots(tmp_path):
    from integrations.usami.designer.evaluation import evaluate
    log=[];p=plan(trials_per_generation=1,strategy='bidirectional_fixed_designer')
    c=FrozenController(tmp_path/'train',p,empty_snapshot('designer'),empty_snapshot('customer'),empty_snapshot('service'),factory(log))
    result=c.run();assert result['status']=='completed',result['error']
    saved=(tmp_path/'train'/'g0.json').read_bytes();n=len(log)
    evaluation=plan(world_specs=[{'id':'new43','seed':43}],trials_per_generation=1,strategy='bidirectional_fixed_designer')
    report=evaluate(tmp_path/'eval',evaluation,tmp_path/'train',result['selected_designer'],result['selected_service'],factory(log))
    assert report['replay_verified'] and report['selection_changes']==0 and report['reflection_calls']==0
    assert not any('reflector' in role for role,phase,req in log[n:])
    assert saved==(tmp_path/'train'/'g0.json').read_bytes() and report['cost']['llm_calls']>0


def test_actual_explicit_false_refusal_is_a_task_decision_not_an_engine_event(tmp_path):
    c,a,clients=backend_fixture(tmp_path)
    row=submit(c,a,clients,task(c.pool,'u0',source=a.evidence_hash),'explicit',decline=True)
    assert row.error is None,row.error
    e=row.evidence
    assert len(e.task_decisions)==1 and not e.events and not e.failures
    assert row.discovery.direct_failure and row.discovery.failure_kind=='false_rejection' and row.discovery.business_violations==0
    error=c.bank.add_anchor(c.adapter,e);assert error.branch=='error_amplification'
    assert c.backend.replay(row,c.designer,c.service,tmp_path/'replay')
    from vast_coevo.reflection.reflector import ServiceReflector,ReflectionInput
    client=ScriptedClient(SPEC,[answer(calls=[call('noop','submit_skill_patch',{'action':'NOOP','role':'service','base_snapshot_hash':c.service.snapshot_hash,'rationale':'Recorded test feedback only.'})])])
    reflection=ServiceReflector(client,SPEC).reflect(ReflectionInput(generation=0,role='service',library=c.service,evidence=(e,),interaction_views=(c.adapter.reflection_view(e),)))
    assert reflection.error is None
    sent=json.loads(client.requests[0]['messages'][1]['content'])
    assert sent['evidence'][0]['task_decisions'][0]['claimed_code']=='PROJECT_ACCESS_REQUIRED'
    raw=e.model_dump(mode='json',exclude={'evidence_hash'});raw['task_decisions'][0]['response_hash']='0'*64
    from vast_coevo.models import EpisodeEvidence
    assert not c.adapter.replay(EpisodeEvidence.model_validate(raw),c.customer,c.service).verified


def test_ambiguous_prose_refusal_remains_unresolved(tmp_path):
    c,a,clients=backend_fixture(tmp_path);p=task(c.pool,'u0',source=a.evidence_hash)
    clients['designer']=ScriptedClient(SPEC,[answer(calls=[call('p','submit_challenge',p.model_dump(mode='json'))])])
    clients['customer']=ScriptedClient(SPEC,[action(),stop()]);clients['service']=ScriptedClient(SPEC,[answer('I cannot do that.')])
    row=c.backend.trial(c.designer,c.service,[a],c.bank,'ambiguous')
    assert row.error is None,row.error
    assert not row.evidence.task_decisions and not row.discovery.verified_failure and row.discovery.status=='triggered_unresolved'
    assert c.backend.replay(row,c.designer,c.service,tmp_path/'replay')


@pytest.mark.parametrize('code',[[],{},None])
def test_malformed_refusal_claim_is_unresolved_not_an_auditor_crash(code):
    from types import SimpleNamespace
    from integrations.usami.frozen.task_oracle import decisions
    e=SimpleNamespace(final_reply=json.dumps({'decision':'refuse','operation':start(),'code':code}),inputs=[{'grants':[start()]}])
    assert decisions(e,[])==()
