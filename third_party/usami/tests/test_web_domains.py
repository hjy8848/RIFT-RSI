"""Real independent engines and published evidence through the desktop API."""
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from web.backend.app import create_app

DOMAINS=('cloudops','fulfillment','stayops')


@pytest.fixture
def client():
    with TestClient(create_app()) as c:yield c


def initial(c,d):
    r=c.post(f'/api/domains/{d}/sessions',json={'seed':42});assert r.status_code==200,r.text;return r.json()


def write(c,d,name,scope,rid,params=None,actor='u0',mode='reference',expected=None):
    state=c.get(f'/api/domains/{d}/world').json()
    return c.post(f'/api/domains/{d}/operations',json={'actor_id':actor,'operation':{'name':name,
        'project_id' if d=='cloudops' else 'scope_id':scope,'resource_id':rid,'params':params or {}},
        'mode':mode,'expected_state_hash':expected or state['state_hash']})


def test_catalog_uses_actual_four_releases(client):
    catalog=client.get('/api/domains').json()['domains']
    assert [(d['id'],d['metrics']['frozen_service_tasks']) for d in catalog]==[('saas',278),('cloudops',45),('fulfillment',338),('stayops',302)]
    assert [(d['metrics']['business_families'],d['metrics']['independent_structures']) for d in catalog]==[(21,28),(12,12),(31,49),(37,64)]
    assert client.post('/api/domains/unknown/sessions',json={'seed':42}).status_code==404
    assert client.get('/api/domains/unknown/benchmark').status_code==404


@pytest.mark.parametrize('domain',DOMAINS)
def test_generation_reset_export_and_exact_replay_do_not_finalize_session(client,domain):
    state=initial(client,domain)
    assert state['kind']=='manual_exploration' and state['evaluation']['status']=='pass'
    trace=client.get(f'/api/domains/{domain}/export/trace').json();assert trace['events'][-1]['kind']=='final'
    assert client.get(f'/api/domains/{domain}/world').json()['events'][-1]['kind']=='generation'
    replay=client.post(f'/api/domains/{domain}/replay').json();assert replay['replayed'] and replay['matches_current_state']
    assert client.post(f'/api/domains/{domain}/world/reset').json()['state_hash']==state['state_hash']
    generated=client.post(f'/api/domains/{domain}/world/generate',json={'seed':43}).json()
    assert generated['state_hash']!=state['state_hash']
    assert client.post(f'/api/domains/{domain}/world/generate',json={'seed':42}).json()['state_hash']==state['state_hash']


def test_four_worlds_and_browser_sessions_remain_independent(client):
    saas=client.post('/api/sessions',json={'seed':42}).json()
    states={d:initial(client,d) for d in DOMAINS}
    changed=write(client,'stayops','mark_room_clean','p0','p0_standard_0').json()
    assert changed['snapshot']['state_hash']!=states['stayops']['state_hash']
    assert client.get('/api/world').json()['state_hash']==saas['state_hash']
    for d in ['cloudops','fulfillment']:assert client.get(f'/api/domains/{d}/world').json()['state_hash']==states[d]['state_hash']
    with TestClient(create_app()) as other:
        assert other.get('/api/domains/stayops/world').status_code==401
        assert initial(other,'stayops')['state_hash']==states['stayops']['state_hash']
    assert write(client,'stayops','mark_room_clean','p0','p0_standard_0',expected=states['stayops']['state_hash']).status_code==409


@pytest.mark.parametrize('domain,name,scope,rid,actor',[
    ('cloudops','start_instance','p0','p0_db','u2'),
    ('fulfillment','confirm_order','w0','order0','u5'),
    ('stayops','mark_room_clean','p0','p0_standard_0','u3')])
def test_real_permission_denial_and_no_state_change(client,domain,name,scope,rid,actor):
    state=initial(client,domain);r=write(client,domain,name,scope,rid,actor=actor);assert r.status_code==200,r.text
    result=r.json();assert not result['operation']['event']['payload']['result']['success']
    assert result['snapshot']['state_hash']==state['state_hash']
    assert result['operation']['evaluation']['reference_allowed'] is False


@pytest.mark.parametrize('domain,name,scope,rid',[
    ('cloudops','start_instance','p0','p0_web'),('fulfillment','confirm_order','w0','order0')])
def test_reference_rejects_and_sandbox_has_actual_business_violation(client,domain,name,scope,rid):
    state=initial(client,domain);denied=write(client,domain,name,scope,rid).json()
    assert not denied['operation']['event']['payload']['result']['success']
    invalid=write(client,domain,name,scope,rid,mode='sandbox').json()
    assert invalid['operation']['event']['payload']['result']['success']
    assert invalid['snapshot']['state_hash']!=state['state_hash']
    assert invalid['operation']['evaluation']['classification']=='successful_business_violation'
    assert client.post(f'/api/domains/{domain}/replay').json()['matches_current_state']


def test_hotel_deposit_confirmation_and_native_schema_errors(client):
    initial(client,'stayops')
    params={'reservation_id':'web_booking','guest_id':'g0','room_type_id':'p0_standard','rate_plan_id':'p0_flex',
        'arrival':'2026-01-15','departure':'2026-01-17','occupants':2,'hold_minutes':30}
    hold=write(client,'stayops','create_hold','p0','p0',params).json();assert hold['operation']['evaluation']['operation_correct']
    denied=write(client,'stayops','confirm_reservation','p0','web_booking').json();assert not denied['operation']['event']['payload']['result']['success']
    assert write(client,'stayops','capture_payment','p0','web_booking',{'amount_cents':'2400','key':'x','token':'card_approved'}).status_code==422
    assert write(client,'stayops','capture_payment','p0','web_booking',{'amount_cents':2400,'key':'x','token':'card_approved'}).status_code==200
    confirmed=write(client,'stayops','confirm_reservation','p0','web_booking').json()
    assert confirmed['snapshot']['world']['reservations']['web_booking']['status']=='confirmed'
    assert confirmed['snapshot']['evaluation']['status']=='pass'
    assert client.post('/api/domains/stayops/replay').json()['matches_current_state']
    body={'actor_id':'u0','operation':{'name':'verify_guest','scope_id':'p0','resource_id':'web_booking','params':{'outcome':'verified'}},'expected_state_hash':confirmed['snapshot']['state_hash']}
    assert client.post('/api/domains/stayops/operations',json={**body,'oracle':{'status':'pass'}}).status_code==422
    assert client.post('/api/domains/stayops/operations',json=body).status_code==422


@pytest.mark.parametrize('domain',DOMAINS)
def test_live_entity_graph_and_scoped_customer_view(client,domain):
    s=initial(client,domain);graph=client.get(f'/api/domains/{domain}/world/graph').json()
    nodeids={n['id'] for n in graph['nodes']}
    assert graph['state_hash']==s['state_hash'] and graph['edges']
    assert all(e['source'] in nodeids and e['target'] in nodeids for e in graph['edges'])
    assert client.get(f'/api/domains/{domain}/world/graph?scope=missing').status_code==404
    view=client.get(f'/api/domains/{domain}/observation?identity=u3');assert view.status_code==200
    assert 'simulated_identity_result' not in view.text and 'gateway_notice' not in view.text
    assert client.get(f'/api/domains/{domain}/observation?identity=missing').status_code==404
    if domain=='stayops':
        calendar=client.get('/api/domains/stayops/world/graph?calendar=true').json()
        assert len(calendar['nodes'])==len(graph['nodes'])+102


@pytest.mark.parametrize('domain,counts', [('cloudops',[27,9,9,2]),('fulfillment',[265,27,46,1]),('stayops',[234,1,67,1])])
def test_frozen_library_loading_witness_and_read_only_pilot(client,domain,counts):
    state=initial(client,domain);release=client.get(f'/api/domains/{domain}/benchmark').json()
    assert [s['instance_count'] for s in release['suites']]==counts
    assert not release['pilot']['official_full_release_score']
    assert client.get(f'/api/domains/{domain}/benchmark/suites/private').status_code==404
    items=client.get(f'/api/domains/{domain}/benchmark/suites/test-demo').json()['instances']
    item=next(i for i in items if i['task'] and i['task']['kind']=='completion')
    command={'suite':'test-demo','instance_id':item['id'],'expected_state_hash':state['state_hash']}
    loaded=client.post(f'/api/domains/{domain}/benchmark/load',json=command);assert loaded.status_code==200,loaded.text
    assert loaded.json()['state_hash']==item['initial_hash'] and loaded.json()['exploration_source']['kind']=='manual_exploration'
    command['expected_state_hash']=loaded.json()['state_hash']
    witness=client.post(f'/api/domains/{domain}/benchmark/witness',json={**command,'route':0});assert witness.status_code==200,witness.text
    assert witness.json()['model_calls']==0 and witness.json()['snapshot']['evaluation']['status']=='pass'
    assert client.post(f'/api/domains/{domain}/benchmark/witness',json={**command,'route':0}).status_code==409
    row=release['pilot']['service_report']['rows'][0]
    run=client.get(f'/api/domains/{domain}/benchmark/pilots/{row["run_hash"]}');assert run.status_code==200
    replay=client.post(f'/api/domains/{domain}/benchmark/pilots/{row["run_hash"]}/replay');assert replay.status_code==200,replay.text
    assert replay.json()==row
    assert client.get(f'/api/domains/{domain}/benchmark/pilots/not-a-path').status_code==404
    assert 'VERIFICATION.md' in release['documents']
    assert client.get(f'/api/domains/{domain}/reports/VERIFICATION.md').status_code==200
    assert client.get(f'/api/domains/{domain}/reports/secret.json').status_code==404
