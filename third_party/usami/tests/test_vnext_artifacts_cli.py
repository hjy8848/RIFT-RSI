"""Source-pinned native proof bundles and network-free CLI operations."""
import json
import pytest
from test_vnext_tool_tasks import single
from usami_benchmark.tasks import TaskSelection,ConstructionBudget,compile_task
from usami_benchmark.artifacts import save_archive,verify_archive
from usami_benchmark.__main__ import main


def one():
    frozen,t=single('cloudops')
    return compile_task(frozen,TaskSelection(kind='tool',family=t.family,identity=t.identity,message=t.message,goal=t.goal,capabilities=t.capabilities),ConstructionBudget(max_nodes=32,max_depth=3))


def test_eight_domain_archive_has_actual_proof_and_source_integrity(tmp_path):
    record=one();root=tmp_path/'archive';m=save_archive(root,[record]);r=verify_archive(root)
    assert r['verified'] and r['tasks']==1 and r['episodes']==0 and m['model_experiments']=='NOT RUN'
    path=root/m['cases'][0]['path'];path.write_bytes(path.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='compressed'):verify_archive(root)
    with pytest.raises(FileExistsError):save_archive(root,[record])


def test_cli_compiles_and_verifies_without_reading_any_model_key(tmp_path,capsys,monkeypatch):
    frozen,t=single('cloudops');selection=TaskSelection(kind='tool',family=t.family,identity=t.identity,message=t.message,goal=t.goal,capabilities=t.capabilities)
    request=tmp_path/'request.json';request.write_text(json.dumps({'world_spec':frozen.spec.model_dump(mode='json'),'selection':selection.model_dump(mode='json'),'construction_budget':{'max_nodes':32,'max_depth':3}}))
    output=tmp_path/'task.json';assert main(['prove',str(request),'--output',str(output)])==0
    assert main(['verify-task',str(output)])==0
    assert '"model_calls": 0' in capsys.readouterr().out
    import usami.service.client as client
    monkeypatch.setattr(client,'key_from_env',lambda:(_ for _ in ()).throw(AssertionError('must not read credential')))
    with pytest.raises(SystemExit):main(['run',str(output),'--output',str(tmp_path/'run'),'--max-llm-calls','4'])
    assert not (tmp_path/'run').exists()


def test_archive_cannot_relabel_proofs_as_real_models_by_resealing_manifest(tmp_path):
    from usami_harness.models import digest
    root=tmp_path/'archive';m=save_archive(root,[one()]);m['model_experiments']='DECLARED_REAL_MODEL_EPISODES';m['manifest_hash']=digest({k:v for k,v in m.items() if k!='manifest_hash'})
    (root/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='cannot claim'):verify_archive(root)
