"""Native structural generators: provenance, invariants, no label-only diversity."""
import importlib
import json
import pytest
from usami_benchmark.registry import default_spec,generate,invariants
from usami_benchmark.signatures import signatures
from usami_benchmark.contracts import GeneratorConfig


@pytest.mark.parametrize('domain',['saas','cloudops','fulfillment','stayops'])
def test_structural_generator_reproducible_and_native_valid(domain):
    spec=default_spec(domain,19)
    left,right=generate(spec),generate(spec)
    assert left.model_dump_json()==right.model_dump_json()
    assert not invariants(domain,left)
    assert left.state_hash==right.state_hash
    assert signatures(domain,left)==signatures(domain,right)


@pytest.mark.parametrize('domain',['cloudops','fulfillment','stayops'])
def test_generator_config_is_strict_and_unconditioned(domain):
    module=importlib.import_module('usami_benchmark.generators.'+domain)
    with pytest.raises(ValueError):module.generate_world(42,{'task':'force a violation'})
    with pytest.raises(ValueError):GeneratorConfig(min_scopes=5,max_scopes=2)
    with pytest.raises(ValueError):generate(default_spec(domain,42).model_copy(update={'rules_version':'unregistered'}))
    assert module.generate_world(42,{'lifecycle_steps':0}).state_hash!=module.generate_world(43,{'lifecycle_steps':0}).state_hash


@pytest.mark.parametrize('domain',['cloudops','fulfillment','stayops'])
def test_legacy_generator_not_silently_replaced(domain):
    legacy=importlib.import_module('usami_'+domain+'.generator')
    world=legacy.generate_world(42)
    assert legacy.generate_world(42)==world
    assert world.state_hash!=generate(default_spec(domain,42)).state_hash


def test_signature_ignores_cloud_ids_names_and_numeric_jitter_without_boundary_change():
    from usami_cloudops.models import World
    world=generate(default_spec('cloudops',42))
    before=signatures('cloudops',world)
    raw=world.model_dump(mode='json');mapping={k:'renamed_'+k for c in ('users','organizations','projects','access','instances','invoices') for k in raw[c]}
    def rename(v):
        if isinstance(v,dict):return {mapping.get(k,k):rename(x) for k,x in v.items()}
        if isinstance(v,list):return [rename(x) for x in v]
        if isinstance(v,str):return mapping.get(v,v)
        return v
    changed=rename(raw);changed['seed']=999
    for name in ('users','organizations','projects','instances'):
        for row in changed[name].values():row['name']='Display label only'
    renamed=World.model_validate(changed)
    assert not invariants('cloudops',renamed)
    assert before==signatures('cloudops',renamed)
    for invoice in renamed.invoices.values():
        invoice.amount_cents+=1
        for payment in invoice.payments:payment.amount_cents+=1
    assert not invariants('cloudops',renamed)
    assert before==signatures('cloudops',renamed)


def test_historical_seed_hashes_remain_exact():
    from pathlib import Path
    pins=json.loads((Path(__file__).parents[1]/'benchmarks/vnext-0.1/verification/legacy-generator-baseline.json').read_text())
    for domain,record in pins['domains'].items():
        module=importlib.import_module('usami.world.generator' if domain=='saas' else 'usami_'+domain+'.generator')
        assert module.generate_world(record['seed']).state_hash==record['world_hash']


def test_original_stayops_reference_invariant_conflict_is_preserved_and_excluded_from_new_worlds():
    import gzip
    from pathlib import Path
    from usami_stayops import adapter
    from usami_benchmark.generation import NativeHistory
    path=Path(__file__).parents[1]/'benchmarks/vnext-0.1/audits/worlds-phase1/stayops-native-consistency-10009.json.gz'
    recorded=json.loads(gzip.decompress(path.read_bytes()))
    bad=next(r for r in recorded['oracle']['operations'] if not r['operation_correct'])
    event=next(e for e in recorded['trace']['events'] if e['id']==bad['event_id'])
    before=adapter.World.model_validate(event['before']);operation=adapter.Operation.model_validate(event['payload']['operation'])
    decision=adapter.decide(before,event['payload']['actor'],operation)
    assert decision.allowed and adapter.invariants(decision.expected_state)
    notes=[];history=NativeHistory(adapter,before,notes)
    assert not history.step(event['payload']['actor'],operation.name,operation.scope_id,operation.resource_id,operation.params)
    assert history.finish()==before
    assert notes[0]['kind']=='native_reference_invariant_conflict'
    assert 'DIRTY_ROOM_ASSIGNED' in {v['code'] for v in notes[0]['invariants']}


def test_frozen_pool_rejects_mutation_and_heldout():
    from usami_benchmark.worlds import freeze,WorldPool
    entry=freeze(default_spec('cloudops',25));pool=WorldPool([entry],discovery=True)
    key=entry.registration_hash;loaded=pool.load(key)
    loaded.world['seed']=999
    assert pool.load(key).world['seed']==25
    pool.entries[key].world['seed']=999
    with pytest.raises(ValueError,match='mutated'):pool.load(key)
    with pytest.raises(ValueError,match='heldout'):WorldPool([freeze(default_spec('cloudops',25,'heldout'))],discovery=True)
