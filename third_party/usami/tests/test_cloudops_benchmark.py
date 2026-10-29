import json
import pytest
from usami_cloudops.release import build_release
from usami_cloudops.benchmark import load,validate,score,run_instance,summarize,assert_comparable,Run
from usami_cloudops.baselines import ProofClient,OneStepClient,StopCustomer
from usami_cloudops.goals import make_task
from usami_cloudops.signatures import task_signatures
from usami_cloudops.models import World,digest
from usami_cloudops.suite import run_suite


@pytest.fixture(scope='module')
def release(tmp_path_factory):
    root=tmp_path_factory.mktemp('cloud-release');build_release(root);return root


def test_frozen_split_quality_and_no_name_seed_diversity(release):
    manifests=[load(release/f'service-{s}.json') for s in ('train','validation','test-demo')]
    assert [len(m.instances) for m in manifests]==[27,9,9]
    coverage=json.loads((release/'coverage.json').read_text())
    assert coverage['service_tasks']==45 and coverage['independent_structures']==12
    assert coverage['split_overlap']['train/test-demo']['structures']==0
    assert len(json.loads((release/'review-examples.json').read_text()))==18
    for family in ('capacity_expand','quota_reduce','billing_recovery'):
        a,b=make_task(42,family),make_task(43,family)
        assert a[1].structure_signature==b[1].structure_signature and a[1].semantic_signature==b[1].semantic_signature


def test_id_renaming_signature_and_world_payload(release):
    w,t=make_task(42,'dependency_boot');data=w.model_dump(mode='json');task=t.model_dump(mode='json')
    # All domain identifiers renamed consistently; names and seed do not count.
    ids=set(w.users)|set(w.organizations)|set(w.projects)|set(w.instances)|set(w.access)|set(w.invoices)
    mapping={key:'renamed_'+key for key in ids}
    def rename(obj):
        if isinstance(obj,str):return mapping.get(obj,obj)
        if isinstance(obj,list):return [rename(x) for x in obj]
        if isinstance(obj,dict):return {mapping.get(k,k):rename(v) for k,v in obj.items()}
        return obj
    data=rename(data);task=rename(task)
    task['goal']['allowed_changes']={collection+'/'+mapping.get(entity,entity):fields for key,fields in t.goal.allowed_changes.items() for collection,entity in [key.split('/')]}
    from usami_cloudops.goals import Task
    assert task_signatures(World.model_validate(data),Task.model_validate(task))==task_signatures(w,t)


def test_all_frozen_paths_and_offline_proof_baseline(release):
    for split in ('train','validation','test-demo'):
        m=load(release/f'service-{split}.json');runs=[]
        for i in m.instances:
            run=run_instance(m,i,ProofClient(m.service_model,i.task));row=score(m,run)
            assert row['task_success'];runs.append(run)
        assert summarize(m,runs)['success_rate']==1.0
        assert summarize(m,runs[:1])['success_rate'] is None
        with pytest.raises(ValueError):summarize(m,runs+[runs[0]])


def test_discovery_has_no_goals_and_customer_strategy_comparison(release):
    m=load(release/'discovery-demo.json')
    assert all(i.task is None for i in m.instances) and len(m.trials)==3 and m.strategy==''
    other=m.model_copy(deep=True);other.strategy='Prefer broad information gathering';other.customer_model.model='another-customer';other.seal();assert_comparable(m,other)
    other.budget.max_service_tool_calls+=1;other.seal()
    with pytest.raises(ValueError):assert_comparable(m,other)
    runs=[run_instance(m,i,ProofClient(m.service_model,None),StopCustomer(m.customer_model),trial) for i in m.instances for trial in m.trials]
    assert summarize(m,runs)['success_rate']==0


def test_manifest_integrity_source_pin_and_suite_resume(release,tmp_path):
    m=load(release/'service-test-demo.json');altered=m.model_copy(deep=True);altered.instances[0].world.projects['p0'].quota.cpu+=1
    with pytest.raises(ValueError):validate(altered)
    altered=m.model_copy(deep=True);altered.source_hashes={};altered.seal()
    with pytest.raises(ValueError,match='provenance'):validate(altered)
    factory=lambda manifest,instance:(ProofClient(manifest.service_model,instance.task),None)
    partial=run_suite(m,tmp_path/'runs',factory,max_runs=1);assert not partial['complete'] and partial['success_rate'] is None
    complete=run_suite(m,tmp_path/'runs',factory);assert complete['complete'] and complete['success_rate']==1
    assert run_suite(m,tmp_path/'runs',lambda *_:(_ for _ in ()).throw(RuntimeError('must replay not retry')))==complete
    with pytest.raises(ValueError):summarize(m,[Run.model_validate({'manifest_hash':m.manifest_hash,'dataset_hash':m.dataset_hash,'instance_id':'foreign','trial':'r0'})])
