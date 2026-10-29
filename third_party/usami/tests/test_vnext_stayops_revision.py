import gzip,json
from pathlib import Path
from usami_benchmark.native import stayops_v2 as revised
from usami_stayops import adapter as legacy
from usami_harness.engine import Engine,replay_trace
from usami_harness.oracle import evaluate
from usami_harness.models import Request
from usami_benchmark.contracts import WorldSpec
from usami_benchmark.registry import generate,invariants


def fixture():
    path=Path(__file__).parents[1]/'benchmarks/vnext-0.1/audits/worlds-phase1/stayops-native-consistency-10009.json.gz'
    raw=json.loads(gzip.decompress(path.read_bytes()))
    bad=next(r for r in raw['oracle']['operations'] if not r['operation_correct'])
    event=next(e for e in raw['trace']['events'] if e['id']==bad['event_id'])
    return event


def test_versioned_checkout_corrects_future_turnover_without_altering_legacy():
    event=fixture();op=legacy.Operation.model_validate(event['payload']['operation']);actor=event['payload']['actor']
    old=legacy.World.model_validate(event['before']);before=revised.World.model_validate({**event['before'],'version':revised.VERSION})
    assert legacy.invariants(legacy.decide(old,actor,op).expected_state)
    assert not revised.invariants(revised.decide(before,actor,op).expected_state)
    engine=Engine(revised,before);req=Request(id='checkout',identity=actor,operation=op.model_dump(mode='json'))
    engine.register(req);assert engine.execute(actor,op,req.id).payload['result']['success']
    trace=engine.finish();assert evaluate(revised,trace,engine.requests)['status']=='pass'
    assert replay_trace(revised,trace,engine.requests).state==engine.state
    assert engine.state.rooms[before.stays['stay_'+op.resource_id].room_id].housekeeping=='dirty'
    # A currently occupied dirty room remains an invariant error in the new rules.
    occupied=before.clone();room=occupied.rooms[occupied.stays['stay_'+op.resource_id].room_id];room.housekeeping='dirty'
    assert any(r['code']=='DIRTY_ROOM_OCCUPIED' for r in revised.invariants(occupied))


def test_revision_generation_is_explicit_deterministic_and_valid():
    spec=WorldSpec(domain='stayops',seed=10009,generator_version='stayops-structural-v2',rules_version=revised.VERSION)
    a=generate(spec);assert a==generate(spec) and a.version==revised.VERSION and not invariants('stayops',a)
    legacy_spec=spec.model_copy(update={'generator_version':'stayops-structural-v1','rules_version':'stayops-0.1.0'})
    assert generate(legacy_spec).version=='stayops-0.1.0'


def test_new_release_defaults_to_corrected_rules_but_retains_explicit_v1():
    from usami_benchmark.registry import default_spec
    from usami_benchmark.worlds import freeze
    spec=default_spec('stayops',10009)
    assert (spec.generator_version,spec.rules_version)==('stayops-structural-v2',revised.VERSION)
    assert not invariants('stayops',generate(spec))
    historical=spec.model_copy(update={'generator_version':'stayops-structural-v1','rules_version':'stayops-0.1.0'})
    assert generate(historical).version=='stayops-0.1.0' and freeze(historical)==freeze(historical)
