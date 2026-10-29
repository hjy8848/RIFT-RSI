"""Unit authority graphs and technician-resource boundaries have neutral signatures."""
from usami_benchmark.registry import default_spec,generate,invariants
from usami_benchmark.worlds import freeze
from usami_benchmark.signatures import signatures
from usami_property.models import World


def test_property_generator_is_native_reproducible_and_structurally_variable():
    entries=[freeze(default_spec('property',s)) for s in range(10)]
    assert len({e.world_hash for e in entries})==10 and len({e.structure_signature for e in entries})>=8
    for e in entries:
        assert e==freeze(e.spec) and not invariants('property',World.model_validate(e.world))


def test_property_signatures_ignore_contacts_ids_skill_order_and_money_scale():
    w=generate(default_spec('property',42));raw=w.model_dump(mode='json')
    mapping={r['id']:'renamed_'+r['id'] for rows in raw.values() if isinstance(rows,dict) for r in rows.values() if isinstance(r,dict) and 'id' in r}
    def rename(v):
        if isinstance(v,dict):return {mapping.get(k,k):rename(x) for k,x in v.items()}
        if isinstance(v,list):return [rename(x) for x in v]
        return mapping.get(v,v) if isinstance(v,str) else v
    other=rename(raw)
    for r in other['users'].values():r['name']='Different display';r['contact']='unrelated@example.invalid'
    for r in other['technicians'].values():r['name']='Other worker label';r['skills'].reverse()
    assert signatures('property',World.model_validate(other))==signatures('property',w)
    scaled=w.clone()
    for c in scaled.contracts.values():c.coverage_cents*=2;c.resident_approval_limit_cents*=2
    for a in scaled.assets.values():a.repair_cost_cents*=2
    for p in scaled.profiles.values():p.facts['max_budget_cents']*=2
    assert signatures('property',scaled)==signatures('property',w)
    a=next(iter(scaled.assets.values()));a.condition='working' if a.condition=='faulty' else 'faulty'
    assert signatures('property',scaled)['structure']!=signatures('property',w)['structure']
