"""Actual native Engine/Oracle fixtures test experiment plumbing, not LLM gains."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest

from integrations.usami import UsamiAdapter, UsamiProfile
from integrations.usami.domains import initial_world
from vast_coevo.models import Budget
from experiments.vast_coevo.examples.compare_customer_performance import (
    SCENARIOS, episode_metrics, historical_world, make_world, semantic_state, snapshots,
)
from test_vast_usami import SPEC, ScriptedClient, answer, call, customer_action, service_arguments
from usami_cloudops.rules import invariants


def test_twenty_scenarios_have_real_distinct_valid_unseen_business_state():
    original=historical_world()
    signatures=[]
    for n in range(20):
        title,world,actor=make_world(n)
        assert title and actor in world.users and not invariants(world)
        assert world==make_world(n)[1]
        signatures.append(semantic_state(world))
        assert signatures[-1]!=semantic_state(original)
        cosmetically_changed=world.clone()
        cosmetically_changed.seed+=100
        cosmetically_changed.instances['p0_db'].name='Other Name'
        assert semantic_state(cosmetically_changed)==signatures[-1]
    assert len(set(signatures))==20 and len(SCENARIOS)==20


def test_frozen_world_is_cloned_and_legacy_profiles_keep_exact_shape():
    _,world,actor=make_world(0)
    profile=UsamiProfile(config_id='custom',domain='cloudops',seed=world.seed,identity=actor,frozen_world=world.model_dump(mode='json'))
    a,b=initial_world(profile),initial_world(profile)
    a.instances['p0_db'].status='running'
    assert b==world and initial_world(profile)==world
    legacy=UsamiProfile(config_id='legacy',domain='cloudops',identity='u0')
    assert 'frozen_world' not in legacy.model_dump(mode='json')


@pytest.mark.parametrize('tamper',['seed','identity','invariant','domain'])
def test_frozen_profiles_reject_untrusted_configuration(tamper):
    _,world,actor=make_world(0)
    data=dict(config_id='custom',domain='cloudops',seed=world.seed,identity=actor,frozen_world=world.model_dump(mode='json'))
    if tamper=='seed':data['seed']+=1
    elif tamper=='identity':data['identity']='nonexistent'
    elif tamper=='invariant':data['frozen_world']['instances']['p0_db']['dependencies']=['p0_db']
    else:data['domain']='fulfillment'
    with pytest.raises(ValueError):UsamiProfile.model_validate(data)


def test_real_c1_patch_is_exact_and_not_formally_selected():
    libraries,patch,*_=snapshots()
    assert libraries['C1'].parent_hash==libraries['C0'].snapshot_hash==patch.base_snapshot_hash
    assert libraries['C1'].skills[0]==patch.skill
    assert libraries['C0'].skills[0].version==1 and libraries['C1'].skills[0].version==2
    assert libraries['C1'].skills[0].scope.domains==libraries['C0'].skills[0].scope.domains
    assert libraries['S0'].skills[0].version==1


def fixture_adapter(tmp_path,actor='u0',barrier=None):
    _,world,_=make_world(0)
    profile=UsamiProfile(config_id='custom',domain='cloudops',seed=world.seed,identity=actor,mode='sandbox',frozen_world=world.model_dump(mode='json'))
    operation=dict(name='start_instance',project_id='p0',resource_id='p0_db',params={})
    def factory(spec,role,config,episode_id):
        responses=[customer_action('cloudops','intent','Inspect a project dependency boundary.',operation)] if role=='customer' else [answer(calls=[call('start','start_instance',service_arguments('cloudops',operation))]),answer('Actual operation result reported.')]
        client=ScriptedClient(spec,responses)
        if barrier and role=='customer':
            complete=client.complete
            def synchronized(request):
                barrier.wait(timeout=10)
                return complete(request)
            client.complete=synchronized
        return client
    adapter=UsamiAdapter(tmp_path/'native',(profile,),client_factory=factory)
    config=adapter.config_for('custom',SPEC,SPEC,Budget(max_customer_turns=1,max_customer_llm_calls=1,max_service_llm_calls=2,max_service_tool_calls=2))
    return adapter,config


def test_paired_concurrent_sessions_isolate_world_and_replay_exact_injection(tmp_path):
    libraries,*_=snapshots()
    adapter,config=fixture_adapter(tmp_path,barrier=threading.Barrier(2))
    def run(arm):return adapter.run_episode(config,libraries[arm],libraries['S0'],'pair.'+arm)
    with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(run,('C0','C1')))
    for arm,row in zip(('C0','C1'),rows):
        assert row.replay.verified and row.config.initial_world_hash==config.initial_world_hash
        assert adapter.replay(row,libraries[arm],libraries['S0']).verified
        assert {i.hits[0].version for i in row.injections if i.role=='customer' and i.hits}=={1 if arm=='C0' else 2}
    assert rows[0].trace.path!=rows[1].trace.path and len(adapter._profiles)==1
    assert rows[0].events[0].before_hash==rows[1].events[0].before_hash
    assert initial_world(adapter._profiles['custom']).state_hash==config.initial_world_hash


@pytest.mark.parametrize('actor,expected_violations,effective',[('u0',0,1),('u1',1,0)])
def test_metrics_only_score_oracle_successful_violation_and_lawful_request_lower_bound(tmp_path,actor,expected_violations,effective):
    libraries,*_=snapshots()
    adapter,config=fixture_adapter(tmp_path,actor)
    row=adapter.run_episode(config,libraries['C0'],libraries['S0'],'metrics')
    result=episode_metrics(dict(episode_id='metrics',scene_id='custom',arm='C0'),row)
    assert row.replay.verified
    assert result['business_violations']==expected_violations
    assert len(result['failure_types'])==expected_violations
    assert result['effective_tests']==effective
    assert result['cost']['missing_usage_calls']==3
