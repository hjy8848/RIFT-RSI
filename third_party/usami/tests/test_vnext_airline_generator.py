"""Timetable/ownership signatures exclude identifiers, display labels and raw money."""
from usami_benchmark.registry import default_spec,generate,invariants
from usami_benchmark.worlds import freeze
from usami_benchmark.signatures import signatures
from usami_airline.models import World


def test_airline_generator_registration_reproducibility_and_structural_variation():
    entries=[freeze(default_spec('airline',s)) for s in range(10)]
    assert len({e.world_hash for e in entries})==10 and len({e.structure_signature for e in entries})>=8
    for row in entries:
        assert row==freeze(row.spec)
        assert not invariants('airline',World.model_validate(row.world))


def test_airline_anonymous_structure_is_not_an_id_or_currency_counter():
    world=generate(default_spec('airline',42));raw=world.model_dump(mode='json')
    ids={r['id']:'renamed_'+r['id'] for collection in raw.values() if isinstance(collection,dict) for r in collection.values() if isinstance(r,dict) and 'id' in r}
    def rename(v):
        if isinstance(v,dict):return {ids.get(k,k):rename(x) for k,x in v.items()}
        if isinstance(v,list):return [rename(x) for x in v]
        return ids.get(v,v) if isinstance(v,str) else v
    replaced=rename(raw)
    for collection in ('users','carriers'):
        for r in replaced[collection].values():r['name']='Another label'
    assert signatures('airline',World.model_validate(replaced))==signatures('airline',world)
    scaled=world.clone()
    for u in scaled.users.values():u.payment_limit_cents*=2
    for f in scaled.flights.values():f.fares={c:v*2 for c,v in f.fares.items()}
    for r in scaled.fare_rules.values():r.change_fee_cents*=2;r.refund_fee_cents*=2;r.baggage_price_per_kg_cents*=2
    for b in scaled.bookings.values():b.initial_paid_cents*=2;b.fare_value_cents*=2
    for p in scaled.profiles.values():p.facts['max_fee_cents']*=2
    assert signatures('airline',scaled)==signatures('airline',world)
    target=next(iter(scaled.flights.values()));target.status='cancelled' if target.status!='cancelled' else 'scheduled'
    assert signatures('airline',scaled)['structure']!=signatures('airline',world)['structure']
