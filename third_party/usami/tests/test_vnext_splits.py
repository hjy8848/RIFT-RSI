"""Different seeds, registry relabelling and a new hash are not leakage proofs."""
import json
import pytest
from usami_benchmark.registry import default_spec
from usami_benchmark.worlds import freeze
from usami_benchmark.splits import audit_world_splits,generate_private_heldout


def test_same_business_world_cannot_be_relabelled_as_independent_test():
    a=freeze(default_spec('cloudops',42,'train'));b=freeze(default_spec('cloudops',42,'test_demo'))
    assert a.registration_hash!=b.registration_hash and a.world_hash==b.world_hash
    report=audit_world_splits([a,b]);r=report['overlaps']['test_demo / train']
    assert r['exact_worlds']==1 and r['anonymous_structures']==1 and not report['world_isolation_pass']
    assert report['task_semantic_isolation']=='NOT RUN'


def test_anonymous_signature_and_source_tampering_are_recomputed():
    a=freeze(default_spec('cloudops',42));a.structure_signature='0'*64
    with pytest.raises(ValueError,match='independent generation'):audit_world_splits([a])


def test_private_heldout_rejects_seen_world_then_uses_frozen_finite_schedule(tmp_path):
    known=[freeze(default_spec('cloudops',42,'train'))]
    specs=[default_spec('cloudops',seed,'heldout') for seed in (42,50001,50002)]
    root=tmp_path/'private';r=generate_private_heldout(specs,known,root,requested_count=2)
    assert r['complete'] and r['actual_count']==2 and r['excluded'][0]['reason']=='previously_seen_world'
    assert r['world_split_audit']['world_isolation_pass'] and r['evaluation_status']=='NOT RUN'
    assert json.loads((root/'registration.json').read_text())['candidate_schedule']==[s.model_dump(mode='json') for s in specs]
    pool=json.loads((root/'world-pool.json').read_text());assert all(x['spec']['partition']=='heldout' for x in pool['worlds'])
    with pytest.raises(FileExistsError):generate_private_heldout(specs,known,root,requested_count=2)


def test_shortfall_is_saved_without_inventing_or_resampling_worlds(tmp_path):
    known=[freeze(default_spec('cloudops',42,'train'))];r=generate_private_heldout([default_spec('cloudops',42,'heldout')],known,tmp_path/'shortfall',requested_count=1)
    assert not r['complete'] and r['actual_count']==0 and r['processed_candidates']==1
    assert not (tmp_path/'shortfall/world-pool.json').exists()
    assert json.loads((tmp_path/'shortfall/report.json').read_text())['actual_count']==0


def test_private_pool_cannot_be_written_inside_repository_or_take_train_specs():
    from pathlib import Path
    repo=Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError,match='outside the public repository'):generate_private_heldout([default_spec('cloudops',1,'heldout')],[],repo/'must-not-exist-private-heldout',requested_count=1)
    assert not (repo/'must-not-exist-private-heldout').exists()
    with pytest.raises(ValueError,match='heldout candidate'):generate_private_heldout([default_spec('cloudops',1,'train')],[],repo/'must-not-exist-private-heldout',requested_count=1)
