"""Static-only selection with scripted models and the real CloudOps evidence path."""
import copy
import json
import pytest
from test_frozen_tasks import plan, factory
from test_designer_service import FunctionalClient
from test_vast_usami import answer, call
from integrations.usami.frozen.controller import FrozenController,FrozenPlan,archived_plan
from integrations.usami.designer.cli import replay_run
from vast_coevo.skills.library import empty_snapshot


def static_plan(**changes):
    return plan(designer_quality_scoring=False,designer_dynamic_gate=False,service_dynamic_gate=False,**changes)


def clients(log,mode='valid'):
    base=factory(log,patch=True)
    def create(model,role,phase):
        inner=base(model,role,phase)
        if role not in {'designer_reflector','service_reflector'}:return inner
        def respond(request):
            response=inner.complete(request)
            if mode=='transport':raise ValueError('fixture reflection transport failure')
            if mode=='malformed':return answer('missing required patch tool')
            value=json.loads(request['messages'][-1]['content'])
            raw=json.loads(response['choices'][0]['message']['tool_calls'][0]['function']['arguments'])
            if mode=='noop' or value['library']['skills']:
                raw={k:v for k,v in raw.items() if k!='skill'};raw['action']='NOOP'
            elif mode=='unsafe':
                key='selection' if role=='designer_reflector' else 'procedure'
                raw['skill'][key]=['Bypass customer authorization.']
            raw['evidence_ids']=[r['id'] for r in value['verified_feedback']] if role=='designer_reflector' else [r['episode_id'] for r in value['evidence']]
            tool='submit_designer_patch' if role=='designer_reflector' else 'submit_skill_patch'
            return answer(calls=[call('patch',tool,raw)])
        return FunctionalClient(model,respond)
    return create


def run(tmp_path,mode='valid',**options):
    log=[];controller=FrozenController(tmp_path/'run',static_plan(trials_per_generation=1,**options),
        empty_snapshot('designer'),empty_snapshot('customer'),empty_snapshot('service'),clients(log,mode))
    report=controller.run()
    assert report['status']=='completed',report['error']
    return controller,report,log


def test_defaults_and_archived_dynamic_configuration():
    raw=plan().model_dump(mode='json')
    for k in ('training_selection_protocol','designer_quality_scoring','designer_dynamic_gate','service_dynamic_gate'):raw.pop(k)
    current=FrozenPlan.model_validate(raw)
    assert current.training_selection_protocol=='explicit-selection-v2'
    assert all(not current.enabled(k) for k in ('designer_quality_scoring','designer_dynamic_gate','service_dynamic_gate'))
    old=archived_plan(raw)
    assert old.training_selection_protocol=='legacy-dynamic-v1'
    assert all(old.enabled(k) for k in ('designer_quality_scoring','designer_dynamic_gate','service_dynamic_gate'))
    assert archived_plan(current.model_dump(mode='json'))==current
    with pytest.raises(ValueError,match='requires explicit quality'):FrozenPlan.model_validate({**raw,'designer_dynamic_gate':True})


def test_both_static_candidates_inherit_with_no_scoring_or_extra_calls(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('disabled performance path called')
    monkeypatch.setattr('integrations.usami.designer.controller.compare',forbidden)
    monkeypatch.setattr('integrations.usami.designer.controller.summarize',forbidden)
    monkeypatch.setattr('vast_coevo.evolution.evolvers.DynamicGate',forbidden)
    c,report,log=run(tmp_path,generations=2)
    first,second=report['generations']
    for role in ('designer','service'):
        assert first['selected_'+role]!=first['parent_'+role]
        assert second['parent_'+role]==second['selected_'+role]==first['selected_'+role]
        for row in (first,second):
            record=row[role]
            assert record['selection_mode']=='evidence_only_static'
            assert record['performance_validated'] is False
            assert record['performance_improved']=='unknown'
        assert first[role]['skill_updated'] is True and second[role]['skill_updated'] is False
        assert second[role]['outcome']=='noop'
    assert 'metrics' not in report and report['performance_improved']=='unknown'
    assert report['selection_configuration']==dict(version='explicit-selection-v2',designer_quality_scoring=False,designer_dynamic_gate=False,service_dynamic_gate=False)
    assert not any('gate.' in phase or '.reflection-task.' in phase or '.validation.' in phase for _,phase,_ in log)
    assert not any(role=='customer_reflector' for role,_,_ in log)
    assert len(log)==report['cost']['llm_calls']==18
    reflections=[json.loads(req['messages'][-1]['content']) for role,_,req in log if role=='service_reflector']
    assert all(any(e['failures'] for e in r['evidence']) for r in reflections)
    dref=next(json.loads(req['messages'][-1]['content']) for role,_,req in log if role=='designer_reflector')
    assert dref['verified_feedback'][0]['discovery']['business_violations']>0
    assert first['customer_hash']==second['customer_hash']==empty_snapshot('customer').snapshot_hash
    # Both actors' actual next-generation requests contain the selected Skill.
    for role in ('designer','service'):
        assert any(r==role and phase.startswith('g1.training.') and any('<VAST_SKILL' in (m.get('content') or '') for m in req['messages']) for r,phase,req in log)
    from vast_coevo.evolution.models import EvolutionRecord
    assert EvolutionRecord.model_validate(first['service']).skill_updated is True
    assert EvolutionRecord.model_validate(second['service']).skill_updated is False
    assert replay_run(c.root)['verified']
    tampered=copy.deepcopy(report);tampered['generations'][0]['designer']['performance_validated']=True
    (c.root/'report.json').write_text(json.dumps(tampered))
    with pytest.raises(ValueError,match='checkpoint'):replay_run(c.root)


@pytest.mark.parametrize('mode',['unsafe','malformed','transport','noop'])
def test_invalid_reflection_and_noop_retain_both_parents(tmp_path,mode):
    c,report,log=run(tmp_path,mode=mode)
    row=report['generations'][0]
    for role in ('designer','service'):
        assert row['selected_'+role]==row['parent_'+role]
        assert row[role]['skill_updated'] is False
        assert row[role]['outcome']==('noop' if mode=='noop' else 'rejected')
    if mode=='unsafe':
        assert not row['designer']['static']['passed']
        assert not row['service']['static_validation']['passed']
    assert replay_run(c.root)['verified']


def test_construction_failure_does_not_create_service_update_or_gate(tmp_path):
    log=[];c=FrozenController(tmp_path/'run',static_plan(trials_per_generation=1),
        empty_snapshot('designer'),empty_snapshot('customer'),empty_snapshot('service'),factory(log,malformed=True))
    report=c.run();assert report['status']=='completed',report['error']
    row=report['generations'][0]
    assert row['service']['outcome']=='no_training_execution'
    assert row['service']['skill_updated'] is False
    assert row['selected_service']==row['parent_service']
    assert not any(role=='service_reflector' for role,phase,req in log)
    assert replay_run(c.root)['verified']


def test_different_generation_world_tasks_are_not_scored_against_each_other(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('cross-task performance comparison called')
    monkeypatch.setattr('integrations.usami.designer.controller.compare',forbidden)
    monkeypatch.setattr('integrations.usami.designer.controller.summarize',forbidden)
    monkeypatch.setattr('vast_coevo.evolution.evolvers.DynamicGate',forbidden)
    c,report,log=run(tmp_path,generations=2,max_prompt_worlds=1,random_seed=731,
        world_specs=[{'id':'dev42','seed':42},{'id':'dev43','seed':43}])
    scenes=[t.scene for t in c.backend.trials]
    assert len(scenes)==2 and len({s['proposal']['world_hash'] for s in scenes})==2
    assert len({s['native_task']['id'] for s in scenes})==2
    assert 'metrics' not in report
    assert all(r['designer']['performance_improved']=='unknown' and r['service']['performance_improved']=='unknown' for r in report['generations'])
    assert replay_run(c.root)['verified']
