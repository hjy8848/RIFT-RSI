"""Recovery policy tests. All model replies here are explicitly offline fixtures."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/vast_coevo/examples'))
from episode_resilience import ResilientController
from test_vast_controller import setup_run
from vast_coevo.models import ModelCall, ReplayResult


def invalid_customer(evidence, **changes):
    request = {'model': evidence.config.customer_model.model, 'messages': []}
    call = ModelCall(id=evidence.episode_id+'.customer.1', role='customer', model=evidence.config.customer_model,
                     base_request=request, request=request,
                     response={'choices':[{'message':{'role':'assistant','content':'invalid fixture'}}]})
    raw = evidence.model_dump(mode='json')
    raw.update(termination='CUSTOMER_OUTPUT_INVALID', model_calls=[call.model_dump(mode='json')], evidence_hash='')
    raw.update(changes)
    return type(evidence).model_validate(raw)


def test_resume_retains_failure_and_does_not_repeat_completed_operations(tmp_path):
    controller, config, adapter, c, s, libraries, archive, runs = setup_run(tmp_path, caps={"schema_version":2})
    original = adapter._run
    bad_id = 'g0.'+config.run_id+'.cross.parent_parent.explore'
    def failing(config, customer, service, episode_id, *args, **kwargs):
        row = original(config, customer, service, episode_id, *args, **kwargs)
        return invalid_customer(row) if episode_id == bad_id else row
    adapter._run = failing
    first = controller.run(config)
    assert first.status == 'failed' and not first.generations
    old_calls = len(adapter.calls), len(c.calls), len(s.calls)
    failed = next(e for e in archive.query() if e.episode_id == bad_id)
    resumed = ResilientController(adapter, c, s, libraries, archive, runs).run(config)
    assert resumed.status == 'completed', resumed.error
    assert len(resumed.generations) == 2
    assert sum(item == bad_id for item in adapter.calls) == 1
    assert len(c.calls) == old_calls[1]+1 and len(s.calls) == old_calls[2]+1
    assert next(e for e in archive.query() if e.episode_id == bad_id) == failed
    counts = len(adapter.calls),len(c.calls),len(s.calls)
    assert ResilientController(adapter,c,s,libraries,archive,runs).run(config) == resumed
    assert counts == (len(adapter.calls),len(c.calls),len(s.calls))


def test_policy_does_not_swallow_unknown_service_or_unverified_errors(tmp_path):
    controller,config,adapter,c,s,libraries,archive,runs=setup_run(tmp_path, caps={"schema_version":2})
    e=adapter.run_episode(config.exploration_cases[0].config,libraries.load(config.initial_customer_hash),
                          libraries.load(config.initial_service_hash),'fixture')
    good=invalid_customer(e)
    ResilientController._check_main_evidence((good,))
    with pytest.raises(ValueError):
        ResilientController._check_main_evidence((good.model_copy(update={'replay':ReplayResult(verified=False)}),))
    for value in [good.model_copy(update={'model_calls':()}),
                  good.model_copy(update={'termination':'MODEL_OUTPUT_INVALID'}),
                  good.model_copy(update={'termination':'provider_error'}),
                  good.model_copy(update={'model_calls':(good.model_calls[0].model_copy(update={'response':None}),)})]:
        with pytest.raises(RuntimeError): ResilientController._check_main_evidence((value,))


@pytest.mark.parametrize('stage', ['current', 'selected_customer', 'cross.parent_parent'])
def test_known_customer_error_is_local_in_each_main_phase(tmp_path, stage):
    _,config,adapter,c,s,libraries,archive,runs=setup_run(tmp_path, caps={"schema_version":2})
    original=adapter._run
    bad_id='g0.'+config.run_id+'.'+stage+'.explore'
    def failing(config, customer, service, episode_id, *args, **kwargs):
        row=original(config,customer,service,episode_id,*args,**kwargs)
        return invalid_customer(row) if episode_id==bad_id else row
    adapter._run=failing
    report=ResilientController(adapter,c,s,libraries,archive,runs).run(config)
    assert report.status=='completed',report.error
    assert len(report.generations)==2
    assert adapter.calls.count(bad_id)==1
    assert next(e for e in archive.query() if e.episode_id==bad_id).termination=='CUSTOMER_OUTPUT_INVALID'
    assert len(c.calls)==len(s.calls)==2


def test_second_turn_invalid_customer_reply_with_role_grouped_calls(tmp_path):
    _,config,adapter,c,s,libraries,archive,runs=setup_run(tmp_path, caps={"schema_version":2})
    e=adapter.run_episode(config.exploration_cases[0].config,libraries.load(config.initial_customer_hash),
                          libraries.load(config.initial_service_hash),'fixture')
    bad=invalid_customer(e)
    customer=bad.model_calls[0]
    service=customer.model_copy(update={'id':'earlier.service.1','role':'service'})
    grouped=bad.model_copy(update={'model_calls':(customer,service)})
    ResilientController._check_main_evidence((grouped,))
    with pytest.raises(RuntimeError):
        ResilientController._check_main_evidence((grouped.model_copy(update={'model_calls':(customer,service.model_copy(update={'response':None}))}),))
