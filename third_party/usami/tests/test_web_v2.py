"""Web v2 routes execute the released Engine, Oracle and recorded model journals."""
import pytest
pytest.importorskip('fastapi')
from fastapi.testclient import TestClient
from web.backend.app import create_app
from usami import generate_world, replay_trace
from usami.events import Trace
from usami.world.generator import GeneratorConfig


def config(workflow=None):
    return {'version': 'saas-generator-0.2.0', 'rules': {'version': 'saas-rules-0.2.0'}, 'workflow': workflow}


@pytest.fixture
def client():
    with TestClient(create_app()) as client:
        assert client.post('/api/sessions', json={'seed': 42, 'config': config('failed_restore')}).status_code == 200
        yield client


def state(client): return client.get('/api/world').json()


def write(client, operation, actor='user_000', mode='reference'):
    response = client.post('/api/operations', json={'actor_id': actor, 'operation': operation,
        'mode': mode, 'expected_state_hash': state(client)['state_hash']})
    assert response.status_code == 200, response.text
    return response.json()


def test_v2_payment_restore_ledger_and_replay(client):
    initial = state(client)
    assert initial['world'] == generate_world(42, GeneratorConfig.model_validate(config('failed_restore'))).model_dump(mode='json')
    inv = next(i for i in initial['world']['invoices'].values() if i['purpose'] == 'renewal' and i['org_id'] == 'org_000')
    pay = {'name': 'pay_invoice', 'org_id': 'org_000', 'resource_id': inv['id'], 'params': {'payment_method': 'card_visa', 'idempotency_key': 'web-test'}}
    denied = write(client, pay, actor='user_001')
    assert denied['operation']['event']['payload']['result']['success'] is False
    paid = write(client, pay)
    assert paid['operation']['evaluation']['operation_correct'] is True
    count = len(paid['snapshot']['world']['invoices'][inv['id']]['payments'])
    again = write(client, pay)
    assert len(again['snapshot']['world']['invoices'][inv['id']]['payments']) == count
    restored = write(client, {'name': 'reactivate_subscription', 'org_id': 'org_000', 'resource_id': 'sub_000', 'params': {'invoice_id': inv['id']}})
    assert restored['snapshot']['world']['subscriptions']['sub_000']['status'] == 'active'
    graph = client.get('/api/world/graph?org_id=org_000').json()
    payments = restored['snapshot']['world']['invoices'][inv['id']]['payments']
    assert set(p['id'] for p in payments) <= {n['id'] for n in graph['nodes'] if n['kind'] == 'payment'}
    assert any(e['source'] == inv['id'] and e['target'] == 'sub_000' for e in graph['edges'])
    trace = Trace.model_validate(client.get('/api/export/trace').json())
    assert trace.schema_version == 'usami-trace-0.2.0'
    assert replay_trace(trace).state.state_hash == restored['snapshot']['state_hash']
    assert client.post('/api/replay').json()['matches_current_state']
    assert client.post('/api/world/reset').json()['state_hash'] == initial['state_hash']


def test_v2_member_capacity_sandbox_and_scenarios(client):
    client.post('/api/world/generate', json={'seed': 42, 'config': config('seat_activate')})
    data = state(client)
    member = next(m for m in data['world']['memberships'].values() if m['org_id'] == 'org_000' and m['status'] == 'pending')
    op = {'name': 'activate_member', 'org_id': 'org_000', 'resource_id': member['id'], 'params': {}}
    denied = write(client, op)
    assert denied['operation']['reference_error_code'] == 'NO_AVAILABLE_SEATS'
    invalid = write(client, op, mode='sandbox')
    assert invalid['operation']['event']['payload']['result']['success']
    assert invalid['operation']['evaluation']['operation_correct'] is False
    client.post('/api/world/reset')
    catalog = client.get('/api/scenarios').json()['scenarios']
    witness = next(s for s in catalog if s['category'] == 'business_witness')
    result = client.post('/api/scenarios/run', json={'scenario_id': witness['scenario_id'], 'expected_state_hash': state(client)['state_hash']}).json()
    assert result['snapshot']['evaluation']['status'] == 'pass'
    assert result['snapshot']['world']['memberships'][member['id']]['status'] == 'active'


def test_v2_partial_refunds_and_returned_query_projection(client):
    client.post('/api/world/generate', json={'seed': 42, 'config': config('partial_refund')})
    inv = next(i for i in state(client)['world']['invoices'].values() if i['org_id'] == 'org_000' and i['status'] == 'paid' and i['refundable_amount_cents'] > 0)
    op = {'name': 'refund_invoice', 'org_id': 'org_000', 'resource_id': inv['id'], 'params': {'amount_cents': 1, 'payment_method': inv['payment_method'], 'idempotency_key': 'partial-web'}}
    result = write(client, op)
    assert result['operation']['evaluation']['operation_correct'] is True
    assert result['snapshot']['world']['invoices'][inv['id']]['status'] == 'partially_refunded'
    again = write(client, op)
    assert again['snapshot']['state_hash'] == result['snapshot']['state_hash']
    query = write(client, {'name': 'get_invoice', 'org_id': 'org_000', 'resource_id': inv['id'], 'params': {}})
    assert query['operation']['event']['payload']['result']['data']['refundable_amount_cents'] == inv['refundable_amount_cents'] - 1


def test_public_release_catalog_and_load_are_not_benchmark_scores(client):
    data = client.get('/api/benchmark').json()
    assert [s['instance_count'] for s in data['suites']] == [170, 68, 40, 6]
    assert data['coverage']['accepted_instances'] == 278
    assert not data['pilot']['official_full_release_score']
    assert client.get('/api/benchmark/suites/private').status_code == 404
    instance = client.get('/api/benchmark/suites/test-demo').json()['instances'][0]
    detail = client.get('/api/benchmark/suites/test-demo/instances/' + instance['instance_id']).json()
    assert detail['task']['witnesses']
    command = {'suite': 'test-demo', 'instance_id': instance['instance_id'], 'expected_state_hash': 'stale'}
    assert client.post('/api/benchmark/load', json=command).status_code == 409
    command['expected_state_hash'] = state(client)['state_hash']
    loaded = client.post('/api/benchmark/load', json=command).json()
    assert loaded['state_hash'] == instance['initial_state_hash']
    assert loaded['exploration_source']['kind'] == 'manual_exploration'
    assert loaded['exploration_source']['customer_identity'] == instance['customer_identity']
    assert loaded['operations_used'] == 0


@pytest.mark.parametrize('track', ['service_report', 'discovery_report'])
def test_every_published_pilot_replays_against_frozen_authority(client, track):
    report = client.get('/api/benchmark').json()['pilot'][track]
    for row in report['rows']:
        run_hash = row['run_hash']
        assert client.get('/api/benchmark/pilots/' + run_hash).json()['run_hash'] == run_hash
        replay = client.post('/api/benchmark/pilots/' + run_hash + '/replay')
        assert replay.status_code == 200, replay.text
        assert replay.json() == row
    assert client.get('/api/benchmark/pilots/not-a-filename').status_code == 404
