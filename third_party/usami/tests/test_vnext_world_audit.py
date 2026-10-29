"""A future simulator success is not an already repaired/completed business state."""
from usami_benchmark.audit import state_coverage,audit_domain


def test_current_latent_and_future_states_are_separate_without_numeric_jitter():
    raw={'assets':{'a':{'id':'a','condition':'faulty','simulated_hazard':True,'repair_cost_cents':1000}},
        'technicians':{'t':{'id':'t','skills':['plumbing','electrical','electrical']}},
        'external_events':{'e':{'id':'e','kind':'technician_visit','outcome':'repaired','at':'future','grant_days':30}}}
    row=state_coverage(raw)
    assert row['state_coverage']=={'assets.condition=faulty':1,'technicians.skills=electrical':1,'technicians.skills=plumbing':1}
    assert row['latent_fact_coverage']=={'assets.simulated_hazard=True':1}
    assert row['frozen_external_event_coverage']=={'external_events.kind=technician_visit':1,'external_events.outcome=repaired':1}
    raw['external_events']['e']['outcome']='missed';raw['assets']['a']['repair_cost_cents']=9999
    assert state_coverage(raw)['state_coverage']==row['state_coverage']


def test_eight_domain_audit_v2_reports_native_states_and_unrun_task_boundaries():
    report=audit_domain('education',[0,1])
    assert report['protocol']=='eight-domain-world-audit-v3' and report['valid_worlds']==2 and not report['invalid']
    assert 'enrollments.status=active' in report['state_coverage']
    assert any(k.startswith('external_events.outcome=') for k in report['frozen_external_event_coverage'])
    assert not any('external_events' in k for k in report['state_coverage'])
    assert report['reference_decision_coverage']['status']=='NOT RUN' and not report['minimum_structural_goal_met']


def test_consumed_external_events_are_historical_not_future_coverage():
    raw={'external_events':{'e':{'id':'e','kind':'attendance','outcome':'attended'}},'applied_events':['e']}
    row=state_coverage(raw)
    assert not row['frozen_external_event_coverage'] and not row['state_coverage']
    assert row['historical_external_event_coverage']=={'external_events.kind=attendance':1,'external_events.outcome=attended':1}
