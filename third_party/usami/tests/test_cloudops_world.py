import copy
import pytest
from usami_cloudops import generate_world, Operation, Request
from usami_cloudops.models import Quota
from usami_cloudops.engine import Engine, replay_trace
from usami_cloudops.rules import decide, invariants
from usami_cloudops.oracle import evaluate
from usami_cloudops.goals import make_task, execute


def op(name,resource='p0',project='p0',**params):return Operation(name=name,project_id=project,resource_id=resource,params=params)


def step(engine,operation,actor='u0',mode='reference',granted=True):
    req=Request(id=f'r{len(engine.requests)}',identity=actor,operation=operation,granted=granted)
    engine.register(req);return engine.execute(actor,operation,req.id,mode)


def test_deterministic_and_domain_separation():
    world=generate_world(42)
    assert world==generate_world(42) and world!=generate_world(43) and not invariants(world)
    from usami import generate_world as saas
    with pytest.raises(ValueError):Engine(saas(42))
    with pytest.raises(ValueError):Operation.model_validate({'name':'start_instance','org_id':'org_000','resource_id':'sub_000','params':{}})


@pytest.mark.parametrize('actor,name,allowed,code',[
    ('u0','start_instance',True,None),('u1','start_instance',False,'PROJECT_ACCESS_REQUIRED'),
    ('u2','start_instance',False,'PROJECT_ACCESS_REQUIRED'),('u3','start_instance',False,'CROSS_PROJECT_ACCESS'),
    ('u1','change_project_quota',False,'PERMISSION_DENIED')])
def test_production_roles_never_borrow_development_privilege(actor,name,allowed,code):
    operation=op(name,'p0_db' if name=='start_instance' else 'p0',**({'cpu':6,'memory':12,'instances':3} if name=='change_project_quota' else {}))
    decision=decide(generate_world(42),actor,operation)
    assert (decision.allowed,decision.error_code)==(allowed,code)


@pytest.mark.parametrize('actor,name,allowed', [('u1','change_project_quota',True),('u2','start_instance',True),('u2','change_project_quota',False),('u3','get_project',True),('u3','get_invoice',False)])
def test_admin_operator_viewer_matrix(actor,name,allowed):
    resource='p1_db' if name=='start_instance' else 'bill1' if name=='get_invoice' else 'p1'
    params={'cpu':6,'memory':12,'instances':3} if name=='change_project_quota' else {}
    assert decide(generate_world(),actor,op(name,resource,'p1',**params)).allowed is allowed


def test_owner_requires_explicit_operational_access():
    w,t=make_task(42,'scoped_onboarding')
    assert decide(w,'u0',t.capabilities[1]).error_code=='PROJECT_ACCESS_REQUIRED'
    trace=execute(w,'u0',t.witnesses[0].operations)
    assert trace.final_state.instances['p2_db'].status=='running'


def test_quota_boundaries_and_schema():
    world=generate_world();e=Engine(world)
    step(e,op('start_instance','p0_db'));step(e,op('start_instance','p0_app'))
    assert decide(e.state,'u0',op('start_instance','p0_web')).error_code=='QUOTA_EXCEEDED'
    assert decide(e.state,'u0',op('change_project_quota',cpu=3,memory=8,instances=2)).error_code=='QUOTA_BELOW_USAGE'
    assert decide(e.state,'u0',op('change_project_quota',cpu=4,memory=8,instances=2)).allowed
    for params in ({'cpu':65,'memory':8,'instances':2},{'cpu':4,'memory':257,'instances':2},{'cpu':4,'memory':8,'instances':17},{'cpu':True,'memory':8,'instances':2}):
        with pytest.raises(ValueError):op('change_project_quota',**params)


def test_dag_dependency_shutdown_delete_and_detach():
    w,t=make_task(42,'dependency_boot');e=Engine(w)
    assert decide(w,'u0',op('start_instance','p0_web')).error_code=='DEPENDENCY_NOT_RUNNING'
    for operation in t.witnesses[0].operations:step(e,operation)
    assert decide(e.state,'u0',op('stop_instance','p0_db')).error_code=='DEPENDENTS_RUNNING'
    assert decide(e.state,'u0',op('detach_dependency','p0_app',dependency_id='p0_db')).error_code=='DEPENDENT_MUST_BE_STOPPED'
    assert decide(w,'u0',op('attach_dependency','p0_db',dependency_id='p0_web')).error_code=='DEPENDENCY_CYCLE'
    assert decide(w,'u0',op('attach_dependency','p0_db',dependency_id='p1_db')).error_code=='CROSS_PROJECT_RESOURCE'
    w,t=make_task(42,'safe_retirement');trace=execute(w,'u0',t.witnesses[0].operations)
    assert 'p0_db' not in trace.final_state.instances
    assert t.goal.frame_preserved(w,trace.final_state)


def test_create_conflict_and_exact_replay():
    w,t=make_task(42,'create_stack');trace=execute(w,t.identity,t.witnesses[0].operations)
    assert replay_trace(trace).state==trace.final_state
    assert decide(trace.final_state,t.identity,t.capabilities[0]).error_code=='INSTANCE_ID_CONFLICT'
    corrupted=trace.model_copy(deep=True);corrupted.events[-2].payload['actor']='u2'
    assert not evaluate(corrupted)['trace_valid']
    with pytest.raises(ValueError):replay_trace(corrupted)


def test_failed_payment_retry_conflict_and_recovery():
    w=generate_world();e=Engine(w)
    failed=op('pay_invoice','bill2','p2',payment_method='card_declined',idempotency_key='failed')
    first=step(e,failed);second=step(e,failed)
    assert first.payload['result']['data']==second.payload['result']['data']
    assert len(e.state.invoices['bill2'].payments)==1 and e.state.invoices['bill2'].status=='failed'
    assert decide(e.state,'u0',op('pay_invoice','bill2','p2',payment_method='card_approved',idempotency_key='failed')).error_code=='IDEMPOTENCY_CONFLICT'
    assert decide(e.state,'u0',op('resume_project','p2','p2',invoice_id='bill2')).error_code=='INVOICE_NOT_PAID'
    step(e,op('pay_invoice','bill2','p2',payment_method='card_approved',idempotency_key='succeeded'))
    assert e.state.projects['p2'].status=='suspended'
    step(e,op('resume_project','p2','p2',invoice_id='bill2'))
    assert e.state.projects['p2'].status=='active' and evaluate(e.finish(),e.requests)['status']=='pass'


@pytest.mark.parametrize('family', ['billing_recovery','capacity_expand','dependency_boot','scoped_onboarding','quota_reduce','safe_retirement','create_stack','live_reconfigure'])
def test_all_goal_routes_and_causal_omissions(family):
    w,t=make_task(43,family)
    for witness in t.witnesses:
        trace=execute(w,t.identity,witness.operations)
        assert t.goal.satisfied(trace.final_state) and t.goal.frame_preserved(w,trace.final_state)
        assert witness.direct_edges
        assert all(not row['goal_satisfied_after_remaining_legal_steps'] for row in witness.omissions)


@pytest.mark.parametrize('operation,code',[
    (op('start_instance','p0_web'),'DEPENDENCY_NOT_RUNNING'),
    (op('attach_dependency','p0_db',dependency_id='p0_web'),'DEPENDENCY_CYCLE'),
    (op('start_instance','p0_db','p1'),'CROSS_PROJECT_RESOURCE'),
    (op('resume_project','p2','p2',invoice_id='bill2'),'INVOICE_NOT_PAID'),
    (op('pay_invoice','bill0',payment_method='card_approved',idempotency_key='double'),'ALREADY_PAID'),
    (op('delete_instance','p0_db'),'DEPENDENCY_STILL_ATTACHED')])
def test_independent_oracle_sandbox_success_is_business_violation(operation,code):
    e=Engine(generate_world());event=step(e,operation,mode='sandbox')
    assert event.payload['result']['success']
    report=evaluate(e.finish(),e.requests);row=report['operations'][0]
    assert row['classification']=='successful_business_violation' and code in row['violations'] and row['operation_correct'] is False


def test_authorization_only_is_not_business_finding_and_read_isolation():
    e=Engine(generate_world());step(e,op('start_instance','p0_db'),mode='sandbox',granted=False)
    row=evaluate(e.finish(),e.requests)['operations'][0]
    assert row['classification']=='authorization_only'
    e=Engine(generate_world());event=step(e,op('get_invoice','bill3','p3'),actor='u0',mode='sandbox')
    assert not event.payload['result']['success'] and event.payload['result']['data']=={}


def test_oracle_catches_wrong_project_mutation_without_trusting_applier(monkeypatch):
    import usami_cloudops.engine as module
    apply=module.apply
    def broken(world,actor,operation,event):
        state,result=apply(world,actor,operation,event);state.projects['p1'].quota.cpu+=1;return state,result
    monkeypatch.setattr(module,'apply',broken)
    e=Engine(generate_world());step(e,op('start_instance','p0_db'))
    row=evaluate(e.finish(),e.requests)['operations'][0]
    assert 'EXPECTED_TRANSITION_MISMATCH' in row['violations']


def test_access_grant_revoke_and_project_admin_ceiling():
    e=Engine(generate_world())
    grant=op('grant_project_access','p1','p1',user_id='u3',role='operator')
    step(e,grant,actor='u1');assert e.state.access['p1_u3'].role=='operator'
    assert decide(e.state,'u3',op('start_instance','p1_db','p1')).allowed
    revoke=op('revoke_project_access','p1','p1',user_id='u3')
    step(e,revoke,actor='u1');assert 'p1_u3' not in e.state.access
    assert decide(e.state,'u1',op('grant_project_access','p1','p1',user_id='u3',role='admin')).error_code=='PERMISSION_DENIED'
    assert decide(e.state,'u1',op('revoke_project_access','p1','p1',user_id='u0')).error_code=='PERMISSION_DENIED'
    assert e.state.organizations['o0'].owner_id=='u0' and evaluate(e.finish(),e.requests)['status']=='pass'


@pytest.mark.parametrize('quota', [{'cpu':3,'memory':8,'instances':2},{'cpu':4,'memory':7,'instances':2},{'cpu':4,'memory':8,'instances':1}])
def test_each_quota_dimension_independently_detected(quota):
    world=generate_world();world.instances['p0_db'].status='running';world.instances['p0_app'].status='running'
    operation=op('change_project_quota',**quota)
    e=Engine(world);event=step(e,operation,mode='sandbox')
    assert event.payload['result']['success']
    assert 'QUOTA_EXCEEDED' in evaluate(e.finish(),e.requests)['operations'][0]['violations']


def test_suspended_start_without_billing_recovery_is_actual_violation():
    e=Engine(generate_world());event=step(e,op('start_instance','p2_db','p2'),actor='u2',mode='sandbox')
    assert event.payload['result']['success']
    row=evaluate(e.finish(),e.requests)['operations'][0]
    assert row['classification']=='successful_business_violation' and 'SUSPENDED_RESOURCE_RUNNING' in row['violations']


def test_access_identity_is_scoped_tuple_not_generated_id_spelling():
    world=generate_world()
    grant=world.access.pop('p1_u0');grant.id='opaque-admin-grant';world.access[grant.id]=grant
    operation=op('grant_project_access','p1','p1',user_id='u0',role='operator')
    assert decide(world,'u1',operation).error_code=='PERMISSION_DENIED'
    e=Engine(world);step(e,operation,actor='u0')
    assert e.state.access['opaque-admin-grant'].role=='operator'
    assert 'p1_u0' not in e.state.access and not invariants(e.state)
    assert evaluate(e.finish(),e.requests)['status']=='pass'
