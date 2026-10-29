"""Business-structural diversity is invariant to identifiers and raw money scaling."""
from usami_benchmark.registry import default_spec,generate,invariants
from usami_benchmark.worlds import freeze,WorldPool
from usami_benchmark.signatures import signatures
from usami_aftersales.models import World


def test_frozen_aftersales_world_generation_is_independent_reproducible_and_scoped():
    entries=[freeze(default_spec('aftersales',seed)) for seed in range(10)]
    assert len({e.world_hash for e in entries})==10
    assert len({e.structure_signature for e in entries})>=8
    for e in entries:
        assert not invariants('aftersales',World.model_validate(e.world))
        assert e==freeze(e.spec)
    pool=WorldPool(entries,discovery=True);row=pool.load(entries[0].registration_hash)
    row.world['orders'].clear()
    assert pool.load(entries[0].registration_hash).world['orders']


def test_aftersales_structure_does_not_count_labels_ids_or_money_jitter():
    w=generate(default_spec('aftersales',42));raw=w.model_dump(mode='json')
    ids={obj['id']:'renamed_'+obj['id'] for values in raw.values() if isinstance(values,dict) for obj in values.values() if isinstance(obj,dict) and 'id' in obj}
    def rename(value):
        if isinstance(value,dict):return {ids.get(k,k):rename(v) for k,v in value.items()}
        if isinstance(value,list):return [rename(v) for v in value]
        return ids.get(value,value) if isinstance(value,str) else value
    changed=rename(raw)
    for collection in ('users','merchants','products'):
        for obj in changed[collection].values():obj['name']='A completely different display label'
    changed=World.model_validate(changed)
    assert signatures('aftersales',changed)==signatures('aftersales',w)
    scaled=w.clone()
    for o in scaled.orders.values():o.paid_cents*=2
    for l in scaled.lines.values():l.unit_price_cents*=2
    for m in scaled.merchants.values():m.refund_approval_limit_cents*=2
    assert signatures('aftersales',scaled)==signatures('aftersales',w)
    # A real policy boundary must remain visible.
    lid=next(iter(scaled.lines));scaled.lines[lid].status='return_pending'
    assert signatures('aftersales',scaled)['structure']!=signatures('aftersales',w)['structure']
