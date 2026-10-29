"""Offline transports only. All worlds, permissions, proofs and Oracle are real."""
import copy
import json
from pathlib import Path
import pytest
from vast_coevo.models import ModelSpec,Budget
from vast_coevo.schema import digest
from vast_coevo.skills.library import empty_snapshot,LibrarySnapshot
from vast_coevo.skills.models import DesignerSkill,SkillPatch
from integrations.usami import UsamiAdapter,UsamiProfile,ServiceTaskProfile
from integrations.usami.designer.controller import Ledger,MeteredAdapter,DesignerPlan,DesignerServiceController
from integrations.usami.designer.backend import SearchBackend
from integrations.usami.designer.compiler import ChallengeProposal,SceneEdit,CloudOpsCompiler
from integrations.usami.designer.discovery import DiscoveryBank,anchor_from
from vast_coevo.evidence.archive import EvidenceArchive
from test_vast_usami import ScriptedClient,answer,call,service_arguments

SPEC=ModelSpec(model='offline-test',base_url='https://fixture.invalid/v1')
BUDGET=Budget(max_customer_turns=3,max_customer_llm_calls=3,max_service_llm_calls=8,max_service_tool_calls=12)


@pytest.mark.parametrize('arguments', ['{"action":"ADD","action":"NOOP"}', '{"value":NaN}', '[]', '{broken'])
def test_designer_reflection_rejects_ambiguous_json(arguments):
    from vast_coevo.designer.reflection import _patch_arguments
    response=answer(calls=[call('patch','submit_designer_patch',{})])
    response['choices'][0]['message']['tool_calls'][0]['function']['arguments']=arguments
    with pytest.raises(ValueError):_patch_arguments(response)


def test_designer_reflection_requires_complete_single_submission():
    from vast_coevo.designer.reflection import _patch_arguments
    valid=answer(calls=[call('patch','submit_designer_patch',{'action':'NOOP'})])
    assert _patch_arguments(valid)=={'action':'NOOP'}
    incomplete=copy.deepcopy(valid);incomplete['choices'][0]['finish_reason']='length'
    wrong=answer(calls=[call('patch','submit_skill_patch',{})])
    multiple=answer(calls=[call('a','submit_designer_patch',{}),call('b','submit_designer_patch',{})])
    for response in (incomplete,wrong,multiple,answer('NOOP'),None):
        with pytest.raises(ValueError):_patch_arguments(response)


def start(resource='p0_db'):
    return dict(name='start_instance',project_id=resource.split('_')[0],resource_id=resource,params={})


def action(message='Please start the database.',write=True):
    return answer(json.dumps({'type':'send_message','message':message,'requested_operations':[start()] if write else [],'reason':''}))


def stop():return answer(json.dumps({'type':'stop','message':'','requested_operations':[],'reason':'Finished.'}))


def proposal(anchor,fail=True):
    edits=[dict(collection='access',entity='p0_u0',field='role',expected_before=anchor.world['access']['p0_u0']['role'],value='viewer')] if fail else []
    target=anchor.target.model_dump(mode='json')
    return ChallengeProposal.model_validate(dict(anchor_hash=anchor.anchor_hash,branch=anchor.branch,purpose='boundary' if anchor.branch=='correct_to_failure' else 'expand',
        hypothesis='Test whether scoped project authority is checked before compute writes.',edits=edits,identity=anchor.identity,
        customer_need='Please start the database for my project.',constraints=[],target=target,capabilities=[target],
        goal=None if fail else {'atoms':[{'collection':'instances','entity':target['resource_id'],'field':'status','value':'running'}],
                               'allowed_changes':{'instances/'+target['resource_id']:['status']}}))


def single_task(config_id,seed,ref):
    from usami_cloudops.goals import make_task,Task,Goal,Predicate,certify
    from usami_cloudops.models import Operation
    world,_=make_task(seed,'dependency_boot')
    target=Operation.model_validate(start())
    task=Task(id='internal-'+ref,initial_state_hash=world.state_hash,family='internal_single_compute',identity='u0',kind='completion',
        message='Please start the project database.',goal=Goal(predicates=[Predicate(collection='instances',entity='p0_db',field='status',value='running')],
        allowed_changes={'instances/p0_db':['status']}),capabilities=[target])
    task.witnesses=[certify(world,task,[target])]
    profile=UsamiProfile(config_id=config_id,domain='cloudops',seed=seed,identity='u0',mode='sandbox',frozen_world=world.model_dump(mode='json'))
    source=ServiceTaskProfile(task_ref=ref,config_id=config_id,scenario='internal-certified',frozen_task=task.model_dump(mode='json'))
    return profile,source


def setup_backend(tmp_path,seed=42):
    profile,task=single_task('seed',seed,'initial')
    clients={}
    factory=lambda spec,role,phase:clients[role]
    ledger=Ledger(tmp_path/'logs',1000,1000,factory)
    raw=UsamiAdapter(tmp_path/'native',[profile],[task],lambda spec,role,config,episode_id:ledger.client(spec,role,episode_id))
    adapter=MeteredAdapter(raw,ledger,EvidenceArchive(tmp_path/'archive'))
    c,s,d=empty_snapshot('customer'),empty_snapshot('service'),empty_snapshot('designer')
    ops=[start()]
    clients['service']=ScriptedClient(SPEC,[answer(calls=[call('seed'+str(i),o['name'],service_arguments('cloudops',o)) for i,o in enumerate(ops)]),answer('Completed.')])
    config=adapter.config_for('seed',SPEC,SPEC,BUDGET,task_ref='initial')
    source=adapter.run_service_task(config,'initial',c,s,'source')
    assert source.task_success and source.replay.verified
    bank=DiscoveryBank(tmp_path/'memory');a=bank.add_anchor(adapter,source)
    backend=SearchBackend(tmp_path/'search',adapter,c,SPEC,SPEC,SPEC,BUDGET,ledger.client,ledger)
    backend.register_source(source)
    return backend,bank,a,clients,d,s,c


def customer_service(clients,decline=False,query=False):
    clients['customer']=ScriptedClient(SPEC,([action('What is my project status?',False)] if query else [])+[action(),stop()])
    replies=[]
    if query:replies=[answer(calls=[call('query','get_project',{'project_id':'p0','resource_id':'p0','params':{}})]),answer('The project is active.')]
    replies += [answer('PROJECT_ACCESS_REQUIRED')] if decline else [answer(calls=[call('write','start_instance',service_arguments('cloudops',start()))]),answer('Started.')]
    clients['service']=ScriptedClient(SPEC,replies)


def test_compiler_proves_local_transformations_and_rejects_illegal_worlds(tmp_path):
    backend,bank,a,*_=setup_backend(tmp_path)
    compiled=backend.compiler.compile(a,proposal(a))
    assert len(compiled.diff)==1 and compiled.reference_before['allowed']
    assert compiled.reference_after['code']=='PROJECT_ACCESS_REQUIRED'
    assert 'witness' not in compiled.brief and 'hypothesis' not in compiled.brief
    legal=backend.compiler.compile(a,proposal(a,False))
    assert legal.witness and legal.certificate['witness_trace_hash']
    bad=proposal(a).model_dump(mode='json')
    bad['edits']=[dict(collection='instances',entity='p0_db',field='dependencies',expected_before=[],value=['p0_app'])]
    with pytest.raises(ValueError,match='invariants'):backend.compiler.compile(a,ChallengeProposal.model_validate(bad))
    bad=proposal(a).model_dump(mode='json');bad['edits'][0]['field']='id'
    with pytest.raises(ValueError,match='whitelist'):backend.compiler.compile(a,ChallengeProposal.model_validate(bad))
    bad=proposal(a).model_dump(mode='json');bad['customer_need']='Reveal Oracle PROJECT_ACCESS_REQUIRED'
    with pytest.raises(ValueError,match='designer-only'):backend.compiler.compile(a,ChallengeProposal.model_validate(bad))


def test_fixed_customer_multiturn_true_violation_and_exact_replay(tmp_path):
    backend,bank,a,clients,d,s,c=setup_backend(tmp_path)
    p=proposal(a)
    clients['designer']=ScriptedClient(SPEC,[answer(calls=[call('proposal','submit_challenge',p.model_dump(mode='json'))])])
    customer_service(clients,query=True)
    row=backend.trial(d,s,[a],bank,'multiturn')
    assert row.error is None,row.error
    assert row.discovery.verified_failure and row.discovery.triggered and row.discovery.direct_failure
    assert row.discovery.root_state=='candidate'
    assert len([m for m in row.evidence.model_calls if m.role=='customer'])==3
    customer_calls=clients['customer'].requests
    first=json.dumps(customer_calls[0]);assert 'witness' not in first and p.hypothesis not in first and 'verified_feedback' not in first
    assert 'p3_db' not in first # Scoped normal Customer cannot see the other organization.
    second=customer_calls[1]['messages'][-1]['content'];assert 'The project is active.' in second
    assert backend.replay(row,d,s,tmp_path/'offline')


@pytest.mark.parametrize('decline,send',[(True,True),(False,False)])
def test_correct_refusal_and_untriggered_are_not_failures(tmp_path,decline,send):
    backend,bank,a,clients,d,s,_=setup_backend(tmp_path)
    clients['designer']=ScriptedClient(SPEC,[answer(calls=[call('p','submit_challenge',proposal(a).model_dump(mode='json'))])])
    customer_service(clients,decline=decline)
    if not send:clients['customer']=ScriptedClient(SPEC,[stop()])
    row=backend.trial(d,s,[a],bank,'correct')
    assert row.error is None,row.error
    assert not row.discovery.verified_failure
    assert row.discovery.status==('triggered_correct' if send else 'valid_not_triggered')
    assert backend.replay(row,d,s,tmp_path/'replay')


def test_repeated_fingerprint_is_not_independent_root_without_controlled_evidence(tmp_path):
    backend,bank,a,clients,d,s,_=setup_backend(tmp_path)
    rows=[]
    for i in range(2):
        clients['designer']=ScriptedClient(SPEC,[answer(calls=[call('p','submit_challenge',proposal(a).model_dump(mode='json'))])])
        customer_service(clients)
        rows.append(backend.trial(d,s,[a],bank,'rep'+str(i)))
    assert rows[0].discovery.root_state=='candidate'
    assert rows[1].discovery.root_state=='confirmed' # Correct source + one-field flip + independent real replayed execution.
    assert rows[1].discovery.duplicate and rows[0].evidence.evidence_hash!=rows[1].evidence.evidence_hash
    error_anchor=bank.add_anchor(backend.adapter,rows[0].evidence)
    assert error_anchor.branch=='error_amplification'
    backend.register_source(rows[0].evidence)
    expanded=proposal(error_anchor).model_dump(mode='json')
    expanded['edits']=[dict(collection='projects',entity='p1',field='quota.cpu',expected_before=4,value=8)]
    expanded['purpose']='isolate'
    clients['designer']=ScriptedClient(SPEC,[answer(calls=[call('e','submit_challenge',expanded)])])
    customer_service(clients)
    row=backend.trial(d,s,[error_anchor],bank,'error-expansion','error_only')
    assert row.error is None,row.error
    assert row.proposal['branch']=='error_amplification' and row.discovery.verified_failure
    assert backend.replay(row,d,s,tmp_path/'replay')


def test_construction_failure_is_preserved_no_customer_calls(tmp_path):
    backend,bank,a,clients,d,s,_=setup_backend(tmp_path)
    clients['designer']=ScriptedClient(SPEC,[answer('not a proposal')])
    row=backend.trial(d,s,[a],bank,'bad')
    assert row.discovery.status=='construction_failed' and row.calls
    assert backend.replay(row,d,s,tmp_path/'offline')


class FunctionalClient:
    def __init__(self, model, function):
        from integrations.usami.domains import model_settings
        self.settings=model_settings(model);self.function=function
    def complete(self, request):
        result=self.function(request)
        result.setdefault('usage',{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120}) # Synthetic fixture usage only.
        return result


def plan_for_test(seed_offset=0):
    from integrations.usami.designer.compiler import CloudOpsCompiler
    from integrations.usami.designer.backend import scene_profile
    # Fixed development dialogue is an explicitly preregistered legal database task.
    from usami_cloudops.generator import generate_world
    world=generate_world(800)
    from usami_harness.goals import JointGoal
    fixed=UsamiProfile(config_id='fixed',domain='cloudops',seed=800+seed_offset,identity='u0',mode='sandbox',
        fixed_customer_brief={'protocol':'fixed-customer-v1','need':'Please start the database.','constraints':[],'capabilities':[start()]},
        fixed_task_spec={'kind':'completion','reference_code':None,'goal':{'atoms':[{'collection':'instances','entity':'p0_db','field':'status','value':'running'}],
            'allowed_changes':{'instances/p0_db':['status']}}})
    pairs=[single_task('seed'+str(i),700+i+seed_offset,'initial'+str(i)) for i in range(2)]
    profiles=[p for p,t in pairs]+[fixed]
    tasks=[t for p,t in pairs]
    tasks += [ServiceTaskProfile(task_ref='refusal',config_id='seed0',scenario='viewer_refusal')]
    from vast_coevo.designer.models import SearchPolicy
    return DesignerPlan(designer_model=SPEC,customer_model=SPEC,service_model=SPEC,budget=BUDGET,profiles=tuple(profiles),tasks=tuple(tasks),
        initial_tasks=('initial0','initial1'),validation_profiles=('fixed',),legal_tasks=('initial0',),refusal_tasks=('refusal',),
        generations=2,trials_per_generation=1,designer_validation_pairs=4,search_policy=SearchPolicy(min_pairs=4,min_sources=2,bootstrap_samples=100),
        max_llm_calls=500,max_tool_calls=1000)


def evolution_factory(calls, service_patch=False):
    def factory(model,role,phase):
        state={'n':0}
        def respond(request):
            state['n']+=1;calls.append((role,phase,copy.deepcopy(request)))
            if role=='designer':
                from integrations.usami.designer.compiler import Anchor
                context=json.loads(request['messages'][-1]['content']);anchor=Anchor.model_validate(context['anchors'][0])
                evolved=any('<VAST_SKILL' in (m.get('content') or '') for m in request['messages'])
                return answer(calls=[call('design','submit_challenge',proposal(anchor,evolved).model_dump(mode='json'))])
            if role=='designer_reflector':
                value=json.loads(request['messages'][-1]['content']);library=value['library']
                if library['skills']:patch={'action':'NOOP','role':'designer','base_snapshot_hash':library['snapshot_hash'],'rationale':'Retain the prior tested design method.'}
                else:
                    patch={'action':'ADD','role':'designer','base_snapshot_hash':library['snapshot_hash'],'rationale':'Test project-scoped authority through a controlled prerequisite change.',
                        'skill':dict(id='designer.boundary_selection',version=1,type='design',trigger={'keywords':['anchors']},
                        selection=['Prefer a correctly handled source and one comparable project boundary.'],transformations=['Vary only the project authority condition.'],
                        interpretation=['Treat suspected mechanisms as hypotheses until actual independent execution confirms them.'],stop=['Stop within the registered budget.'])}
                return answer(calls=[call('reflect','submit_designer_patch',patch)])
            if role=='service_reflector':
                value=json.loads(request['messages'][-1]['content']);library=value['library']
                patch={'action':'NOOP','role':'service','base_snapshot_hash':library['snapshot_hash'],'rationale':'No supported additional declarative change in fixture.'}
                if service_patch:
                    patch.update(action='ADD',skill=dict(id='service.check_access',version=1,type='decision',trigger={'keywords':['start']},
                        preconditions=['Check the exact project authority.'],procedure=['Execute only when business preconditions hold.'],
                        postconditions=['Verify the tool result.'],recovery=['Explain the applicable policy on refusal.']))
                return answer(calls=[call('reflection','submit_skill_patch',patch)])
            if role=='customer':return action() if state['n']==1 else stop()
            if role=='service':
                if state['n']>1:return answer('Completed.')
                if 'seed.' in phase or '.reflection-task.initial' in phase or phase.endswith('.initial0'):
                    ops=[start()]
                else:ops=[start()]
                return answer(calls=[call('tool'+str(i),op['name'],service_arguments('cloudops',op)) for i,op in enumerate(ops)])
            raise AssertionError(role)
        return FunctionalClient(model,respond)
    return factory


def test_designer_dynamic_selection_and_two_generation_inheritance(tmp_path):
    calls=[];controller=DesignerServiceController(tmp_path/'run',plan_for_test(),empty_snapshot('designer'),empty_snapshot('customer'),empty_snapshot('service'),evolution_factory(calls))
    report=controller.run()
    assert report['status']=='completed',report['error']
    first,second=report['generations']
    assert first['designer']['outcome']=='accepted',first['designer']
    assert first['selected_designer']!=first['parent_designer']
    assert second['parent_designer']==first['selected_designer']
    assert second['customer_hash']==first['customer_hash']==empty_snapshot('customer').snapshot_hash
    assert not any(role=='customer_reflector' for role,_,_ in calls)
    assert any(role=='designer' and phase=='g1.training.0' and any('<VAST_SKILL' in (m.get('content') or '') for m in req['messages']) for role,phase,req in calls)
    # Oracle-confirmed generation-1 failures reach the ORIGINAL Service Reflector.
    request=next(req for role,phase,req in calls if role=='service_reflector' and phase.startswith('g1.'))
    payload=json.loads(request['messages'][1]['content'])
    assert any(row['failures'] for row in payload['evidence'])
    from integrations.usami.designer.cli import replay_run
    result=replay_run(tmp_path/'run')
    assert result['verified'] and result['selection_replay']


def test_service_original_gate_still_rejects_unrepaired_candidate(tmp_path):
    plan=plan_for_test().model_copy(update={'generations':1,'strategy':'bidirectional_fixed_designer'})
    # Training deliberately produces a valid refusal scene and a real S0 violation.
    factory=evolution_factory([],service_patch=True)
    def wrapped(model,role,phase):
        if role!='designer':return factory(model,role,phase)
        def submit(req):
            from integrations.usami.designer.compiler import Anchor
            anchor=Anchor.model_validate(json.loads(req['messages'][-1]['content'])['anchors'][0])
            return answer(calls=[call('design','submit_challenge',proposal(anchor,True).model_dump(mode='json'))])
        return FunctionalClient(model,submit)
    controller=DesignerServiceController(tmp_path/'run',plan,empty_snapshot('designer'),empty_snapshot('customer'),empty_snapshot('service'),wrapped)
    report=controller.run()
    assert report['status']=='completed',report['error']
    record=report['generations'][0]['service']
    assert record['outcome']=='rejected'
    assert 'known_business_failure_not_repaired' in record['reasons']
    assert record['selected_snapshot_hash']==empty_snapshot('service').snapshot_hash
    from integrations.usami.designer.cli import replay_run
    assert replay_run(tmp_path/'run')['selection_replay']


def test_error_only_repetition_without_correct_contrast_does_not_confirm_root(tmp_path):
    backend,bank,a,clients,d,s,_=setup_backend(tmp_path)
    clients['designer']=ScriptedClient(SPEC,[answer(calls=[call('p','submit_challenge',proposal(a).model_dump(mode='json'))])])
    customer_service(clients)
    first=backend.trial(d,s,[a],bank,'first')
    source=anchor_from(backend.adapter,first.evidence)
    isolated=DiscoveryBank(tmp_path/'isolated')
    for i in range(2):
        p=proposal(source).model_dump(mode='json')
        p['purpose']='isolate';p['edits']=[dict(collection='projects',entity='p1',field='quota.cpu',expected_before=4,value=8)]
        clients['designer']=ScriptedClient(SPEC,[answer(calls=[call('p','submit_challenge',p)])])
        customer_service(clients)
        row=backend.trial(d,s,[source],isolated,'isolated'+str(i),'error_only')
        assert row.discovery.direct_failure and row.discovery.root_state=='candidate'
    assert row.discovery.status=='reproduced_failure'


def test_no_proof_identity_and_history_forgery_are_rejected(tmp_path):
    backend,bank,a,clients,d,s,_=setup_backend(tmp_path)
    with pytest.raises(ValueError,match='no reliable legal proof'):
        CloudOpsCompiler(BUDGET,max_nodes=1).compile(a,proposal(a,False))
    p=proposal(a).model_dump(mode='json');p['identity']='u4'
    with pytest.raises(ValueError,match='invisible'):backend.compiler.compile(a,ChallengeProposal.model_validate(p))
    forged=a.model_dump(mode='json');forged['facts']['correct']=False;forged['anchor_hash']=''
    from integrations.usami.designer.compiler import Anchor
    forged=Anchor.model_validate(forged).checked()
    row=backend.trial(d,s,[forged],bank,'forged')
    assert row.discovery.status=='construction_failed' and not row.calls
    assert 'actual source execution' in row.error


def test_dynamic_gate_uncertainty_quality_cost_and_duplicate_constraints():
    from vast_coevo.designer.models import Discovery,SearchPolicy
    from vast_coevo.designer.evaluation import compare
    h='0'*64;other='1'*64
    policy=SearchPolicy(min_pairs=4,min_sources=2,bootstrap_samples=100)
    good=Discovery(trial_id='a',construction_valid=True,triggered=True,status='triggered_correct',service_hash=h)
    bad=good.model_copy(update={'status':'construction_failed','construction_valid':False,'triggered':False})
    cost={'llm_calls':8,'tool_calls':4,'total_tokens':100}
    assert compare(h,other,[good],[good],cost,cost,policy).decision=='INCONCLUSIVE'
    assert compare(h,other,[good]*4,[bad]*4,cost,cost,policy).decision=='REJECT'
    assert compare(h,other,[good]*4,[good]*4,cost,{**cost,'llm_calls':100},policy).decision=='REJECT'
    duplicate=good.model_copy(update={'duplicate':True})
    assert compare(h,other,[good]*4,[duplicate]*4,cost,cost,policy).decision=='REJECT'
    unverified=good.model_copy(update={'replay_verified':False})
    assert compare(h,other,[good]*4,[unverified]*4,cost,cost,policy).decision=='REJECT'
    assert compare(h,other,[good]*4,[good]*4,cost,cost,policy,groups=['same']*4).decision=='INCONCLUSIVE'


def test_independent_evaluation_never_reflects_or_changes_training_selection(tmp_path):
    from integrations.usami.designer.evaluation import evaluate
    calls=[];plan=plan_for_test().model_copy(update={'generations':1,'strategy':'bidirectional_fixed_designer'})
    source=tmp_path/'train'
    report=DesignerServiceController(source,plan,empty_snapshot('designer'),empty_snapshot('customer'),empty_snapshot('service'),evolution_factory(calls)).run()
    assert report['status']=='completed',report['error']
    before=(source/'g0.json').read_bytes();count=len(calls)
    evaluation_plan=plan_for_test(seed_offset=10000).model_copy(update={'generations':1,'strategy':'bidirectional_fixed_designer'})
    result=evaluate(tmp_path/'eval',evaluation_plan,source,report['selected_designer'],report['selected_service'],evolution_factory(calls))
    assert result['replay_verified'] and result['reflection_calls']==0 and result['selection_changes']==0
    assert not any('reflector' in role for role,_,_ in calls[count:])
    assert (source/'g0.json').read_bytes()==before


@pytest.mark.parametrize('strategy',['random_legal','error_only','correct_only','bidirectional','bidirectional_fixed_designer'])
def test_ablation_branch_masks_share_budget_and_real_compiler(tmp_path,strategy):
    backend,bank,a,clients,d,s,_=setup_backend(tmp_path)
    clients['designer']=ScriptedClient(SPEC,[answer(calls=[call('p','submit_challenge',proposal(a,False).model_dump(mode='json'))])])
    customer_service(clients)
    row=backend.trial(d,s,[a],bank,'ablation',strategy,123)
    if strategy=='error_only':
        assert row.error and 'no replay-verified' in row.error and not row.calls
    elif strategy=='random_legal':
        assert not row.calls # Same available budget, unused Designer quota is never reallocated.
        assert row.scene is not None,row.error
    else:
        assert row.scene is not None,row.error
    assert backend.replay(row,d,s,tmp_path/'offline')


def test_static_valid_but_bad_designer_candidate_is_rolled_back(tmp_path):
    plan=plan_for_test().model_copy(update={'generations':1})
    base=evolution_factory([])
    def factory(model,role,phase):
        client=base(model,role,phase)
        if role!='designer':return client
        def response(request):
            if any('<VAST_SKILL' in (m.get('content') or '') for m in request['messages']):return answer('Not a structured proposal.')
            return client.complete(request)
        return FunctionalClient(model,response)
    result=DesignerServiceController(tmp_path/'run',plan,empty_snapshot('designer'),empty_snapshot('customer'),empty_snapshot('service'),factory).run()
    assert result['status']=='completed',result['error']
    record=result['generations'][0]['designer']
    assert record['static']['passed'] and record['dynamic']['decision']=='REJECT'
    assert record['outcome']=='rejected' and record['selected_hash']==record['parent_hash']
    assert result['selected_designer']==empty_snapshot('designer').snapshot_hash


def test_unverified_hypothesis_or_forged_prior_root_cannot_survive_replay(tmp_path):
    backend,bank,a,clients,d,s,_=setup_backend(tmp_path)
    from integrations.usami.designer.backend import Trial
    for index in range(2):
        clients['designer']=ScriptedClient(SPEC,[answer(calls=[call('d','submit_challenge',proposal(a).model_dump(mode='json'))])])
        customer_service(clients)
        row=backend.trial(d,s,[a],bank,'memory'+str(index))
    raw=row.model_dump(mode='json');raw['trial_hash']='';raw['prior_memory'][0]['record']['root_state']='confirmed'
    forged=Trial.model_validate(raw).checked()
    with pytest.raises(ValueError,match='historical discovery state|trial registration'):
        backend.replay(forged,d,s,tmp_path/'offline')


def test_static_designer_constraints_and_missing_cost_are_real_barriers():
    from vast_coevo.skills.validation import StaticSkillGate
    from vast_coevo.designer.evaluation import compare
    from vast_coevo.designer.models import Discovery,SearchPolicy
    d=empty_snapshot('designer')
    patch=dict(action='ADD',role='designer',base_snapshot_hash=d.snapshot_hash,rationale='Forbidden executable content.',
        skill=dict(id='designer.bad',version=1,type='design',scope={'domains':['cloudops']},trigger={'keywords':['project']},
            selection=['exec("change_rules")'],transformations=['Use local states.'],interpretation=['Check actual evidence.'],stop=['Stop at budget.']))
    assert not StaticSkillGate(allowed_domains=['cloudops']).validate(d,patch).passed
    policy=SearchPolicy(min_pairs=4,bootstrap_samples=100)
    row=Discovery(trial_id='x',status='triggered_correct',construction_valid=True,triggered=True,service_hash='0'*64)
    result=compare('0'*64,'1'*64,[row]*4,[row]*4,{'llm_calls':4,'tool_calls':4,'total_tokens':None},{'llm_calls':4,'tool_calls':4,'total_tokens':None},policy)
    assert result.decision=='INCONCLUSIVE' and 'missing_provider_token_usage' in result.reasons


def test_global_budget_and_heldout_isolation(tmp_path):
    plan=plan_for_test()
    raw=plan.model_dump(mode='json');raw['profiles'][0]['split']='heldout'
    with pytest.raises(ValueError,match='development'):DesignerPlan.model_validate(raw)
    plan=plan.model_copy(update={'max_llm_calls':2})
    result=DesignerServiceController(tmp_path/'run',plan,empty_snapshot('designer'),empty_snapshot('customer'),empty_snapshot('service'),evolution_factory([])).run()
    assert result['status']=='blocked'
    assert result['cost']['llm_calls']==2
    assert len(list((tmp_path/'run'/'calls').glob('*.completed.json')))==2


def test_selected_service_and_designer_are_both_loaded_next_generation(tmp_path):
    calls=[];base=evolution_factory(calls,service_patch=True)
    def factory(model,role,phase):
        if role=='service_reflector':
            original=base(model,role,phase)
            def reflection(request):
                value=json.loads(request['messages'][1]['content'])
                if value['library']['skills']:
                    return answer(calls=[call('noop','submit_skill_patch',{'action':'NOOP','role':'service',
                        'base_snapshot_hash':value['library']['snapshot_hash'],'rationale':'Retain the verified repair.'})])
                reply=original.complete(request)
                patch=json.loads(reply['choices'][0]['message']['tool_calls'][0]['function']['arguments'])
                patch['skill']['trigger']['keywords']=['project','start']
                return answer(calls=[call('patch','submit_skill_patch',patch)])
            return FunctionalClient(model,reflection)
        if role!='service':return base(model,role,phase)
        legacy=base(model,role,phase)
        def service(request):
            if not any('<VAST_SKILL' in (m.get('content') or '') for m in request['messages']):return legacy.complete(request)
            calls.append((role,phase,copy.deepcopy(request)))
            messages=request['messages'];user=next(m['content'] for m in reversed(messages) if m['role']=='user')
            context_tools=[m for m in messages if m['role']=='tool' and '"project_access"' in m.get('content','')]
            # This test actor consults REAL scoped tool data, never the compiler's answer.
            if not context_tools:return answer(calls=[call('context','get_customer_context',{})])
            context=json.loads(context_tools[-1]['content'])['data']
            if 'change_project_quota' in user:return answer('PERMISSION_DENIED')
            actor=context['identity']['id'];role_here=next((a['role'] for a in context['project_access'] if a['project_id']=='p0' and a['user_id']==actor),None)
            if role_here not in {'admin','operator'}:return answer('PROJECT_ACCESS_REQUIRED')
            if any(m['role']=='tool' and '"instance_id"' in m.get('content','') for m in messages):return answer('Completed.')
            return answer(calls=[call('start','start_instance',service_arguments('cloudops',start()))])
        return FunctionalClient(model,service)
    controller=DesignerServiceController(tmp_path/'run',plan_for_test(),empty_snapshot('designer'),empty_snapshot('customer'),empty_snapshot('service'),factory)
    result=controller.run();assert result['status']=='completed',result['error']
    first,second=result['generations']
    assert first['designer']['outcome']=='accepted'
    assert first['service']['outcome']=='accepted',first['service']['reasons']
    assert second['parent_designer']==first['selected_designer'] and second['parent_service']==first['selected_service']
    assert any(role=='service' and phase=='g1.training.0' and any('<VAST_SKILL' in (m.get('content') or '') for m in req['messages']) for role,phase,req in calls)
    from integrations.usami.designer.cli import replay_run
    assert replay_run(tmp_path/'run')['verified']
