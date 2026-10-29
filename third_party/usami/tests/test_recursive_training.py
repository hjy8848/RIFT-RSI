"""Offline fixtures verify protocol order; native Engine/Oracle are real where stated."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vast_coevo.evidence.archive import EvidenceArchive
from vast_coevo.evaluation.cross import evaluate_cross
from vast_coevo.evolution.controller import CoEvolutionController
from vast_coevo.evolution.models import GatePolicy, RunConfig, ValidationCase
from vast_coevo.schema import digest
from vast_coevo.skills.library import LibraryStore
from vast_coevo.storage.runs import RunStore, GenerationRecord
from test_vast_controller import setup_run, GenericAdapter
from test_vast_usami import adapter_for, libraries, SPEC, answer


def test_single_generation_is_one_interaction_then_both_learn_and_gate(tmp_path):
    controller,config,adapter,c,s,store,archive,runs=setup_run(tmp_path,generations=1,service_modes=('add',),failures=True)
    events=[]
    run=adapter._run
    def interaction(*a,**kw):
        events.append('episode:'+a[3]);return run(*a,**kw)
    adapter._run=interaction
    for client,label in ((c,'customer_reflection'),(s,'service_reflection')):
        complete=client.complete
        def wrapped(request,complete=complete,label=label):
            events.append(label);return complete(request)
        client.complete=wrapped
    report=controller.run(config)
    assert report.status=='completed',report.error
    assert config.schema_version==3 and config.training_protocol=='recursive-training-v1'
    assert events[0]=='episode:g0.test-run.current.explore'
    assert events.index('customer_reflection') < events.index('service_reflection')
    assert not any('.selected_customer.' in e or '.cross.' in e for e in events)
    assert runs.load_evolution(config.run_id,0,'service').dynamic_validation.passed
    assert not report.generations[0].crossings and not report.generations[0].selected_customer_evidence
    current=report.generations[0].current_evidence[0]
    for client in (c,s):
        payload=json.loads(client.calls[0]['messages'][1]['content'])
        assert payload['evidence'][0]['episode_id']==archive.load(current).episode_id
        assert payload['evidence'][0]['failures']
    # Every actual generation interaction/preflight was consumed; candidate Gate results are validation only.
    learned={row['episode_id'] for client in (c,s) for row in json.loads(client.calls[0]['messages'][1]['content'])['evidence']}
    assert archive.load(current).episode_id in learned


def test_schema_v4_runs_recursive_training_with_frozen_paired_gate_protocol(tmp_path):
    controller, legacy, adapter, *_rest = setup_run(tmp_path, generations=1, service_modes=("noop",))
    config = legacy.model_copy(update={
        "schema_version": 4,
        "customer_selection": "customer-static-only-v1",
        "training_protocol": "recursive-training-v1",
        "policy": GatePolicy(service_task_regression_protocol="paired-task-v1"),
    })
    report = controller.run(config)
    assert report.status == "completed", report.error
    assert report.generations[0].schema_version == 2
    assert not report.generations[0].crossings
    assert not report.generations[0].selected_customer_evidence
    assert all(".cross." not in row for row in adapter.calls)


def test_actual_cloudops_oracle_failure_reaches_both_reflections(tmp_path):
    adapter,episode,_=adapter_for(tmp_path/'native','cloudops',mode='sandbox')
    customer,service=libraries('cloudops')
    store=LibraryStore(tmp_path/'libraries');store.save(customer);store.save(service)
    calls={'customer':[],'service':[]}
    def client(role):
        def complete(request):
            value=json.loads(request['messages'][1]['content']);calls[role].append(value)
            assert value['evidence'][0]['failures']
            assert any(e['classification']=='successful_business_violation' for e in value['evidence'][0]['events'])
            return answer(json.dumps(dict(action='NOOP',role=role,base_snapshot_hash=value['library']['snapshot_hash'],rationale='Offline fixture preserving actual Oracle evidence.')))
        return SimpleNamespace(complete=complete)
    case=ValidationCase(id='cloudops',config=episode)
    config=RunConfig(run_id='native',exploration_cases=(case,),validation_cases=(case,),
        customer_reflector_model=SPEC,service_reflector_model=SPEC,
        initial_customer_hash=customer.snapshot_hash,initial_service_hash=service.snapshot_hash)
    archive=EvidenceArchive(tmp_path/'evidence');runs=RunStore(tmp_path/'runs')
    report=CoEvolutionController(adapter,client('customer'),client('service'),store,archive,runs).run(config)
    assert report.status=='completed',report.error
    assert len(calls['customer'])==len(calls['service'])==1
    e=archive.load(report.generations[0].current_evidence[0])
    assert adapter.replay(e,customer,service).verified and e.failures
    assert e.episode_id==calls['service'][0]['evidence'][0]['episode_id']
    assert runs.load_evolution('native',0,'service').reflection.input.evidence[0].evidence_hash==e.evidence_hash


def test_next_generation_uses_both_selected_versions_without_reasking_prior_generation(tmp_path):
    controller,config,adapter,c,s,store,archive,runs=setup_run(tmp_path,customer_modes=('add','add'),service_modes=('add','add'))
    report=controller.run(config)
    assert report.status=='completed',report.error
    first,second=report.generations
    assert first.selected_customer_hash!=first.parent_customer_hash
    assert first.selected_service_hash!=first.parent_service_hash
    e=archive.load(second.current_evidence[0])
    assert (e.customer_snapshot_hash,e.service_snapshot_hash)==(first.selected_customer_hash,first.selected_service_hash)
    assert second.parent_customer_hash==first.selected_customer_hash
    assert second.parent_service_hash==first.selected_service_hash
    assert not any('.selected_customer.' in x or '.cross.' in x for x in adapter.calls)


def tree(path):return {str(p.relative_to(path)):p.read_bytes() for p in path.rglob('*') if p.is_file()}


def test_independent_cross_combinations_are_read_only_and_not_train_feedback(tmp_path):
    controller,config,adapter,c,s,store,archive,runs=setup_run(tmp_path/'train',service_modes=('add',))
    report=controller.run(config);assert report.status=='completed'
    g=report.generations[0]
    pairs={label:(store.load(ch),store.load(sh)) for label,ch,sh in (
        ('C0S0',config.initial_customer_hash,config.initial_service_hash),
        ('C1S0',g.selected_customer_hash,config.initial_service_hash),
        ('C0S1',config.initial_customer_hash,g.selected_service_hash),
        ('C1S1',g.selected_customer_hash,g.selected_service_hash))}
    raw=config.exploration_cases[0].config.model_dump(mode='json');raw.update(initial_world_hash='c'*64,partition='heldout',config_hash='')
    evaluation_config=type(config.exploration_cases[0].config).model_validate(raw)
    before=tree(tmp_path/'train');calls=len(c.calls),len(s.calls)
    output=tmp_path/'evaluation';evaluator=GenericAdapter(output/'native')
    result=evaluate_cross(evaluator,pairs,(evaluation_config,),output,evaluation_id='cross',purpose='frozen-heldout',excluded_world_hashes={'a'*64})
    assert result['status']=='completed' and len(result['rows'])==4
    assert result['reflection_calls']==0 and result['selection_authority'] is False
    assert before==tree(tmp_path/'train') and calls==(len(c.calls),len(s.calls))
    with pytest.raises(ValueError,match='overlaps'):
        evaluate_cross(evaluator,pairs,(evaluation_config,),tmp_path/'bad',evaluation_id='bad',purpose='frozen-heldout',excluded_world_hashes={'c'*64})
    with pytest.raises(ValueError,match='already exists'):
        evaluate_cross(evaluator,pairs,(evaluation_config,),output,evaluation_id='cross',purpose='frozen-heldout',excluded_world_hashes={'a'*64})
    with pytest.raises(ValueError,match='partition'):
        evaluate_cross(evaluator,pairs,(evaluation_config,),tmp_path/'mixed',evaluation_id='mixed',purpose='development-diagnostic',excluded_world_hashes={'a'*64})


def test_old_protocol_is_explicit_and_new_generation_cannot_mix_comparisons(tmp_path):
    controller,old,adapter,*_=setup_run(tmp_path,caps={'schema_version':2},generations=1)
    dumped=old.model_dump(mode='json');assert 'training_protocol' not in dumped
    assert digest(RunConfig.model_validate(dumped))==digest(dumped)
    report=controller.run(old)
    assert report.status=='completed' and len(report.generations[0].crossings)==4
    raw=report.generations[0].model_dump(mode='json');raw.update(schema_version=2,record_hash='')
    with pytest.raises(ValueError,match='recursive generation'):GenerationRecord.model_validate(raw)


def test_independent_evaluation_failure_keeps_costs_and_never_retries(tmp_path):
    from vast_coevo.models import ReplayResult
    _,config,adapter,c,s,store,archive,runs=setup_run(tmp_path/'train')
    pair={'initial':(store.load(config.initial_customer_hash),store.load(config.initial_service_hash))}
    raw=config.exploration_cases[0].config.model_dump(mode='json');raw.update(initial_world_hash='c'*64,config_hash='')
    case=type(config.exploration_cases[0].config).model_validate(raw)
    evaluation=GenericAdapter(tmp_path/'failed/native')
    evaluation.replay=lambda *_:ReplayResult(verified=False,reason='offline integrity fixture')
    with pytest.raises(ValueError,match='Replay'):
        evaluate_cross(evaluation,pair,(case,),tmp_path/'failed',evaluation_id='failed',purpose='development-diagnostic',excluded_world_hashes={'a'*64})
    record=json.loads((tmp_path/'failed/result.json').read_text())
    assert record['consumption']['customer_llm_calls']==1 and record['consumption']['service_llm_calls']==1
    assert len(list((tmp_path/'failed/attempts').glob('*.completed.json')))==1
    calls=len(evaluation.calls)
    with pytest.raises(ValueError,match='already exists'):
        evaluate_cross(evaluation,pair,(case,),tmp_path/'failed',evaluation_id='failed',purpose='development-diagnostic',excluded_world_hashes={'a'*64})
    assert len(evaluation.calls)==calls


def test_cross_cli_loads_saved_hashes_and_uses_separate_real_native_session(tmp_path, monkeypatch, capsys):
    from integrations.usami import UsamiAdapter, UsamiProfile
    from integrations.usami.cli import main
    from test_vast_usami import ScriptedClient, scripts
    # Actual native Reference Engine/Oracle; every model response is a labelled offline fixture.
    root=tmp_path/'train';adapter,episode,_=adapter_for(root/'native','cloudops')
    customer,service=libraries('cloudops');store=LibraryStore(root/'libraries');store.save(customer);store.save(service)
    def reflector(role):
        def complete(request):
            payload=json.loads(request['messages'][1]['content'])
            return answer(json.dumps(dict(action='NOOP',role=role,base_snapshot_hash=payload['library']['snapshot_hash'],rationale='Offline CLI isolation fixture.')))
        return SimpleNamespace(complete=complete)
    case=ValidationCase(id='train',config=episode)
    config=RunConfig(run_id='source',exploration_cases=(case,),validation_cases=(case,),customer_reflector_model=SPEC,
                     service_reflector_model=SPEC,initial_customer_hash=customer.snapshot_hash,initial_service_hash=service.snapshot_hash)
    runs=RunStore(root/'runs');report=CoEvolutionController(adapter,reflector('customer'),reflector('service'),store,EvidenceArchive(root/'evidence'),runs).run(config)
    assert report.status=='completed'
    train_profile=next(iter(adapter._profiles.values()))
    independent=UsamiProfile(config_id='unseen',domain='cloudops',seed=train_profile.seed+1,identity='u2',mode='reference',split='heldout')
    plan=dict(profiles=[train_profile.model_dump(mode='json'),independent.model_dump(mode='json')],
        exploration=[train_profile.config_id],validation=[train_profile.config_id],customer_library='unused-c.json',service_library='unused-s.json',
        customer_model=SPEC.model_dump(mode='json'),service_model=SPEC.model_dump(mode='json'),budget=episode.budget.model_dump(mode='json'))
    plan_path=tmp_path/'plan.json';plan_path.write_text(json.dumps(plan))
    native_script=scripts('cloudops')
    def factory(path,profiles,*args):
        return UsamiAdapter(path,profiles,client_factory=lambda spec,role,*_:ScriptedClient(spec,native_script[role]))
    monkeypatch.setattr('integrations.usami.cli.UsamiAdapter',factory)
    monkeypatch.setattr('integrations.usami.cli.transport',lambda *_:pytest.fail('cross evaluation cannot call Reflection transport'))
    before=tree(root)
    out=tmp_path/'cross'
    args=['--root',str(root),'--plan',str(plan_path),'evaluate-cross','--id','after-training','--training-run','source',
          '--pair','C0S0:'+customer.snapshot_hash+':'+service.snapshot_hash,'--profile','unseen',
          '--purpose','frozen-heldout','--output',str(out)]
    assert main(args)==0
    result=json.loads(capsys.readouterr().out)
    assert result['reflection_calls']==0 and len(result['rows'])==1
    assert before==tree(root)
    assert (out/'native/episodes').exists() and not (out/'libraries').exists()
