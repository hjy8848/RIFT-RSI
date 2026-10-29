"""Source-complete artifacts preserve native proof versus actual-model semantics."""
import json
import pytest
from usami_benchmark.registry import default_spec
from usami_benchmark.worlds import freeze
from usami_airline import adapter
from usami_airline.tasks import certified_tasks
from usami_dialogue.artifacts import save_examples,verify_examples
from usami_harness.models import digest


def write_example(root):
    frozen=freeze(default_spec('airline',0));world=adapter.World.model_validate(frozen.world)
    t,r,rows=certified_tasks(world,'profile_u0_booking1',family='refund',mixed=True,max_nodes=128,max_paths=1)
    assert rows
    return save_examples(root,'airline',[{'frozen':frozen,**rows[0]}])


def test_offline_source_bound_native_case_and_archive_integrity(tmp_path):
    root=tmp_path/'example';m=write_example(root)
    assert verify_examples(root)['verified'] and m['model_calls']==0
    path=root/m['cases'][0]['path'];data=path.read_bytes();path.write_bytes(data+b'corrupt')
    with pytest.raises(ValueError,match='compressed'):verify_examples(root)
    with pytest.raises(FileExistsError):write_example(root)


def test_proof_artifact_cannot_be_relabelled_as_real_llm_evidence(tmp_path):
    root=tmp_path/'example';manifest=write_example(root)
    manifest['model_experiments']='EXECUTED';manifest['model_calls']=12
    manifest['manifest_hash']=digest({k:v for k,v in manifest.items() if k!='manifest_hash'})
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='cannot claim'):verify_examples(root)


def test_missing_source_pin_is_not_accepted_even_with_resealed_hash(tmp_path):
    root=tmp_path/'example';m=write_example(root);p=root/'source-pins.json';pins=json.loads(p.read_text())
    pins.pop('usami_airline/rules.py');p.write_text(json.dumps(pins));m['source_pins_hash']=digest(pins);m['manifest_hash']=digest({k:v for k,v in m.items() if k!='manifest_hash'})
    (root/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='source file set'):verify_examples(root)
