"""Read-only frozen release browsing; loaded worlds remain manual exploration."""
import json
from pathlib import Path

from fastapi import Depends, HTTPException
from usami.world.generator import GeneratorConfig, generate_world
from usami.world.generator_v2 import WORKFLOWS
from usami.world.engine import WorldEngine
from usami.world.models import Model

ROOT = Path(__file__).resolve().parents[2] / 'benchmarks' / 'v0.2.0'
SUITES = {'train': 'service-train.json', 'validation': 'service-validation.json',
          'test-demo': 'service-test-demo.json', 'discovery': 'discovery-demo.json'}


class LoadInstance(Model):
    suite: str
    instance_id: str
    expected_state_hash: str


def read(name):
    # Only server-selected, committed public files; no arbitrary path or upload.
    path = ROOT / name
    if not path.is_file():
        raise HTTPException(503, 'Published v0.2 release artifact unavailable')
    return json.loads(path.read_text(encoding='utf-8'))


def suite_for(key):
    if key not in SUITES:
        raise HTTPException(404, 'Unknown public suite')
    return read(SUITES[key])


def find_instance(suite, identifier):
    instance = next((i for i in suite['instances'] if i['instance_id'] == identifier), None)
    if instance is None:
        raise HTTPException(404, 'Instance is not in the public suite')
    return instance


def pilot_for(run_hash):
    # A filename supplied by the browser is never opened.
    for track in ('service', 'discovery'):
        for path in sorted((ROOT / 'pilot' / track).glob('*.run.json')):
            run = json.loads(path.read_text(encoding='utf-8'))
            if run['run_hash'] == run_hash:
                return run, ROOT / 'pilot' / (track + '-profile.json')
    raise HTTPException(404, 'Unknown published pilot run')


def install_release_routes(app, current, snapshot, assert_current):
    @app.get('/api/catalog')
    def catalog():
        return {'versions': ['0.1.0', '0.2.0'], 'workflows': list(WORKFLOWS)}

    @app.get('/api/benchmark')
    def benchmark():
        suites = []
        for key in SUITES:
            data = suite_for(key)
            suites.append({k: data[k] for k in ('suite_id', 'track', 'split', 'mode', 'budget',
                'tools', 'trials', 'versions', 'dataset_hash', 'manifest_hash')} |
                {'key': key, 'instance_count': len(data['instances'])})
        pilot = read('pilot-report.json')
        # Provider URLs/options are unnecessary in the overview. No credential is read.
        return {'release': '0.2.0', 'suites': suites, 'coverage': read('coverage.json'),
                'pilot': {k: pilot[k] for k in ('service_report', 'discovery_report',
                    'official_full_release_score', 'interpretation', 'correction')},
                'service_model': pilot['service_model']['model']}

    @app.get('/api/benchmark/suites/{key}')
    def suite(key: str):
        data = suite_for(key)
        return {'instances': [{k: i[k] for k in ('instance_id', 'seed', 'customer_identity', 'initial_state_hash')} |
            {'task': {k: i['task'][k] for k in ('message', 'family', 'business_family', 'actor_role', 'kind',
                'structure_signature', 'semantic_signature')} if i['task'] else None} for i in data['instances']]}

    @app.get('/api/benchmark/suites/{key}/instances/{identifier}')
    def detail(key: str, identifier: str):
        data = suite_for(key)
        return find_instance(data, identifier)

    @app.post('/api/benchmark/load')
    def load(command: LoadInstance, session=Depends(current)):
        data = suite_for(command.suite)
        instance = find_instance(data, command.instance_id)
        config = GeneratorConfig.model_validate(instance['generator'] or data['generator'])
        world = generate_world(instance['seed'], config)
        if world.state_hash != instance['initial_state_hash']:
            raise HTTPException(409, 'Frozen instance no longer matches generated world')
        with session.lock:
            assert_current(session, command.expected_state_hash)
            session.config, session.initial, session.engine = config, world.clone(), WorldEngine(world)
            session.requests, session.scenarios = {}, None
            session.exploration_source = {'suite': command.suite, 'instance_id': command.instance_id,
                'customer_identity': instance['customer_identity'], 'kind': 'manual_exploration'}
            return snapshot(session)

    @app.get('/api/benchmark/pilots/{run_hash}')
    def pilot(run_hash: str):
        run, _ = pilot_for(run_hash)
        return run

    @app.post('/api/benchmark/pilots/{run_hash}/replay')
    def replay(run_hash: str):
        from usami.benchmark.manifest import BenchmarkManifest
        from usami.benchmark.runtime import BenchmarkRun
        from usami.benchmark.scoring import score_run
        run, profile = pilot_for(run_hash)
        manifest = BenchmarkManifest.model_validate_json(profile.read_text(encoding='utf-8'))
        # Independent current Engine / Oracle / recorded LLM requests, no provider calls.
        return score_run(manifest, BenchmarkRun.model_validate(run))
