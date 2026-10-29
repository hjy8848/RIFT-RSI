"""Finite native sampling never edits worlds or fabricates complete strata."""
import gzip
import json
from pathlib import Path
import pytest
from usami_harness.models import digest
from usami_benchmark.registry import default_spec
from usami_benchmark.worlds import freeze
from usami_benchmark.candidates import scheduled_selections
from usami_benchmark.datasets import DatasetPlan, build, verify_dataset, select_balanced
from usami_benchmark.tasks import ConstructionBudget, compile_task


@pytest.mark.parametrize('domain',['saas','cloudops','fulfillment','stayops','aftersales','airline','property','education'])
def test_all_eight_candidate_catalogues_use_exact_frozen_bytes_and_have_no_answer_fields(domain):
    world=freeze(default_spec(domain,42));before=world.model_dump_json()
    a=scheduled_selections(world,4);b=scheduled_selections(world,4)
    assert a==b and len(a)==4 and world.model_dump_json()==before
    for row in a:
        assert row.kind==('tool' if domain in {'saas','cloudops','fulfillment','stayops'} else 'dialogue')
        assert not {'world','reference_code','witness','expected_outcome'} & set(row.model_dump())


def test_invalid_plans_limits_and_duplicate_worlds_are_rejected():
    s=default_spec('cloudops',42)
    with pytest.raises(ValueError,match='duplicate'):DatasetPlan(worlds=[s,s])
    with pytest.raises(ValueError,match='quotas'):DatasetPlan(worlds=[s],quotas={'basic':40})
    with pytest.raises(ValueError):DatasetPlan(worlds=[s],quotas={'basic':True,'medium':0,'complex':0,'refusal':0})
    with pytest.raises(ValueError,match='exact hashes'):DatasetPlan(worlds=[s],known_world_hashes=['fake hash'])
    with pytest.raises(ValueError,match='candidate limit'):scheduled_selections(freeze(s),0)


def test_native_bank_exact_verification_and_preserved_incomplete_quotas(tmp_path):
    root=tmp_path/'bank';plan=DatasetPlan(worlds=[default_spec('cloudops',42)],candidates_per_world=8,
        construction_budget=ConstructionBudget(max_nodes=64,max_depth=4))
    m=build(plan,root)
    assert m['attempts']==8 and m['certified']>0 and m['model_calls']==0 and not m['quota_complete']
    assert m['shortfalls']['saas']==plan.quotas
    assert m['release_status']=='CANDIDATE_BANK'
    assert len(list((root/'attempts').glob('*.json')))==8
    assert verify_dataset(root)['verified']
    with pytest.raises(FileExistsError):build(plan,root)
    # Re-sealing aggregate metadata cannot manufacture full release counts.
    m['quota_complete']=True;m['manifest_hash']=digest({k:v for k,v in m.items() if k!='manifest_hash'})
    (root/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='shortfalls'):verify_dataset(root)


def test_alias_questions_do_not_fill_two_slots():
    from test_vnext_task_audit import tool_record
    r=tool_record();other=compile_task(r.frozen,r.selection.model_copy(update={'message':'Please make that same change.','family':'another_label'}),r.construction_budget)
    q={'basic':2,'medium':0,'complex':0,'refusal':0}
    result=select_balanced([r,other],q)
    assert len(result['selected'])==1 and result['shortfalls']['cloudops']['basic']==1
    assert result['excluded'][0]['reason']=='duplicate_rooted_task_structure'


def test_failed_certifications_are_rebuilt_instead_of_silently_omitted(tmp_path):
    root=tmp_path/'failed-candidates'
    plan=DatasetPlan(worlds=[default_spec('saas',42)],candidates_per_world=12,construction_budget=ConstructionBudget(max_nodes=32,max_depth=3))
    manifest=build(plan,root);assert verify_dataset(root)['verified']
    paths=sorted((root/'attempts').glob('*.json'))
    failures=[p for p in paths if json.loads(p.read_text())['status']=='construction_rejected']
    assert failures
    raw=json.loads(failures[0].read_text());raw['reason']='hidden replacement reason';failures[0].write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='outcome/failure'):verify_dataset(root)


def test_known_training_world_cannot_be_relabelled_as_test(tmp_path):
    s=default_spec('cloudops',42,partition='test_demo');world=freeze(s)
    plan=DatasetPlan(worlds=[s],known_world_hashes=[world.world_hash])
    root=tmp_path/'relabel'
    with pytest.raises(ValueError,match='relabelled'):build(plan,root)
    assert json.loads((root/'failure.json').read_text())['stage']=='world_generation'
    assert (root/'plan.json').exists() and not (root/'candidate-schedule.json').exists()


def test_cli_dataset_is_offline_and_provenance_is_not_model_evidence(tmp_path,monkeypatch):
    from usami_benchmark.__main__ import main
    import usami.service.client as client
    monkeypatch.setattr(client,'key_from_env',lambda:(_ for _ in ()).throw(AssertionError('offline')))
    path=tmp_path/'plan.json';path.write_text(DatasetPlan(worlds=[default_spec('fulfillment',42)],candidates_per_world=2).model_dump_json())
    root=tmp_path/'cli'
    assert main(['build-dataset',str(path),'--output',str(root)])==0
    assert main(['verify-dataset',str(root)])==0
    m=json.loads((root/'manifest.json').read_text());m['model_experiments']='REAL_MODELS';m['manifest_hash']=digest({k:v for k,v in m.items() if k!='manifest_hash'})
    (root/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='provenance'):verify_dataset(root)
