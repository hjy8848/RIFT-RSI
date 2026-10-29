"""Equal domain means, conditional denominators, immutable cohorts and missing data."""
import pytest
from usami_benchmark.evaluation import EvaluationPlan,EvaluationResult,aggregate,CAPABILITIES
from usami_benchmark.contracts import DOMAINS
from usami_harness.models import digest


def cohort():
    tasks={d:[digest([d,i]) for i in range(9 if d=='saas' else 1)] for d in DOMAINS}
    plan=EvaluationPlan(partition='test_demo',evidence_kind='scripted_transport_fixture',model_configuration_hash=digest('frozen model/config'),tasks=tasks)
    rows=[]
    for d,keys in tasks.items():
        for key in keys:
            success=d=='saas'
            rows.append(EvaluationResult(plan_hash=plan.plan_hash,domain=d,task_hash=key,world_hash=digest(d),execution_hash=digest([key,'run']),partition=plan.partition,evidence_kind=plan.evidence_kind,
                model_configuration_hash=plan.model_configuration_hash,family='test',difficulty='basic',outcome='completion',capabilities=['state_preconditions'],
                metrics={'task_success':success,'legal_completion':success,'frame_preservation':True,'budget_compliance':True,'replay_verification':True,'successful_business_violation':False,'unauthorized_side_effect':False,'false_refusal':False},customer_calls=1,service_calls=2,tool_calls=1))
    return plan,rows


def test_eight_domains_and_two_tracks_have_equal_weight_not_task_weight():
    p,rows=cohort();r=aggregate(p,rows)
    assert len(CAPABILITIES)==15 and r['complete']
    assert r['macro']['task_success']==.125 and r['tracks']['A']['metrics']['task_success']==.25 and r['tracks']['B']['metrics']['task_success']==0
    assert r['domains']['saas']['observed']==9 and r['domains']['education']['observed']==1
    assert r['macro']['correct_refusal'] is None and r['domains']['saas']['metrics']['correct_refusal']['denominator']==0
    assert r['cost']['customer_input_tokens'] is None and r['cost']['service_calls']==32
    assert r['strata']['capability']['state_preconditions']['domains']['saas']['runs']==9


def test_missing_domains_are_not_silently_renormalized():
    p,rows=cohort();r=aggregate(p,rows[:-1])
    assert not r['complete'] and r['macro']['task_success'] is None
    assert r['tracks']['B']['metrics']['task_success'] is None
    assert r['missing_tasks']['education']==p.tasks['education']


def test_evidence_kind_partition_configuration_and_duplicate_results_are_strict():
    p,rows=cohort()
    with pytest.raises(ValueError,match='duplicate'):aggregate(p,rows+[rows[0]])
    for field,value in [('evidence_kind','real_model'),('partition','train'),('model_configuration_hash',digest('other'))]:
        changed=rows[0].model_copy(update={field:value})
        with pytest.raises(ValueError,match='provenance'):aggregate(p,[changed])


def test_success_cannot_hide_unverified_safety_or_partial_mixed_failure():
    p,rows=cohort();raw=rows[0].model_dump(mode='json')
    raw['metrics']['replay_verification']=None
    with pytest.raises(ValueError,match='Replay'):EvaluationResult.model_validate(raw)
    raw=rows[0].model_dump(mode='json');raw['outcome']='mixed';raw['metrics']['correct_refusal']=False
    with pytest.raises(ValueError,match='completion/refusal'):EvaluationResult.model_validate(raw)
    raw=rows[0].model_dump(mode='json');raw['metrics']['task_success']=1
    with pytest.raises(ValueError,match='exact boolean'):EvaluationResult.model_validate(raw)


def test_plan_requires_exact_distinct_content_hashes_and_unregistered_metrics_fail():
    p,rows=cohort();raw=p.model_dump(mode='json');raw['tasks']['airline']=raw['tasks']['saas'][:1]
    with pytest.raises(ValueError,match='multiple domains'):EvaluationPlan.model_validate(raw)
    raw=rows[0].model_dump(mode='json');raw['metrics']['vibes']=True
    with pytest.raises(ValueError,match='unregistered metric'):EvaluationResult.model_validate(raw)


def test_refusal_denominators_do_not_include_completion_only_tasks():
    p,rows=cohort();raw=rows[0].model_dump(mode='json');raw['metrics']['correct_refusal']=False
    with pytest.raises(ValueError,match='no refusal denominator'):EvaluationResult.model_validate(raw)
    raw=rows[0].model_dump(mode='json');raw['metrics']['false_refusal']=True
    with pytest.raises(ValueError,match='safety failures'):EvaluationResult.model_validate(raw)


def test_optional_native_interfaces_preserve_old_plan_hash_shape_and_reject_drift():
    p,rows=cohort()
    assert 'interface_hashes' not in p.model_dump(mode='json') and 'interface_hash' not in rows[0].model_dump(mode='json')
    raw=p.model_dump(mode='json');raw['interface_hashes']={d:digest([d,'interface']) for d in p.tasks}
    p=EvaluationPlan.model_validate(raw)
    updated=[r.model_copy(update={'plan_hash':p.plan_hash,'interface_hash':p.interface_hashes[r.domain]}) for r in rows]
    assert aggregate(p,updated)['interface_validation']=='pinned'
    bad=updated[0].model_copy(update={'interface_hash':digest('changed tools')})
    with pytest.raises(ValueError,match='native interface'):aggregate(p,[bad])
