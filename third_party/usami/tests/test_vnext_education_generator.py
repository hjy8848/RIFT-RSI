"""Enrollment authority and academic resource signatures ignore display jitter."""
from usami_benchmark.registry import default_spec,generate,invariants
from usami_benchmark.worlds import freeze
from usami_benchmark.signatures import signatures
from usami_education.models import World


def test_education_generator_has_reproducible_real_structure_and_valid_accounts():
    entries=[freeze(default_spec('education',s)) for s in range(10)]
    assert len({e.world_hash for e in entries})==10 and len({e.structure_signature for e in entries})>=8
    for e in entries:
        assert e==freeze(e.spec) and not invariants('education',World.model_validate(e.world))


def test_education_signatures_ignore_ids_display_and_qualification_list_order():
    w=generate(default_spec('education',42));raw=w.model_dump(mode='json')
    mapping={r['id']:'renamed_'+r['id'] for rows in raw.values() if isinstance(rows,dict) for r in rows.values() if isinstance(r,dict) and 'id' in r}
    def rename(v):
        if isinstance(v,dict):return {mapping.get(k,k):rename(x) for k,x in v.items()}
        if isinstance(v,list):return [rename(x) for x in v]
        return mapping.get(v,v) if isinstance(v,str) else v
    other=rename(raw)
    for r in other['users'].values():r['name']='Another label';r['contact']='other@example.invalid'
    for r in other['teachers'].values():r['name']='Teacher label';r['subjects'].reverse()
    for r in other['contracts'].values():r['extension_reasons'].reverse()
    assert signatures('education',World.model_validate(other))==signatures('education',w)
    changed=w.clone();next(iter(changed.contracts.values())).transferable=not next(iter(changed.contracts.values())).transferable
    assert signatures('education',changed)['structure']!=signatures('education',w)['structure']
