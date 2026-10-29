"""Initial case/history distributions must come from genuine audited transitions."""
import copy
import pytest
from usami_benchmark.registry import default_spec,generate,invariants
from usami_benchmark.generators.dialogue_history import replay_history


def spec(domain,seed=0,steps=9):
    row=default_spec(domain,seed,config={'lifecycle_steps':steps});row.generator_version=domain+'-lifecycle-v2';return row


@pytest.mark.parametrize('domain',['aftersales','airline','property','education'])
def test_lifecycle_is_reproducible_legal_and_actual_compact_journal_replays(domain):
    s=spec(domain);notes=[];w=generate(s,diagnostics=notes);again=[]
    assert generate(s,diagnostics=again)==w and notes==again
    summary=notes[0]
    assert 0<summary['actual_transitions']<=9 and summary['histories']
    assert all(h['oracle_status']=='pass' and h['replay_verified'] for h in summary['histories'])
    assert not invariants(domain,w)
    assert replay_history(domain,0,s.generator_config,summary)==w
    assert w.interaction.identity=='' and not w.interaction.consents
    changed=copy.deepcopy(summary);changed['histories'][0]['initial_hash']='0'*64
    with pytest.raises(ValueError,match='initial world'):replay_history(domain,0,s.generator_config,changed)


@pytest.mark.parametrize('domain',['aftersales','airline','property','education'])
def test_zero_lifecycle_preserves_original_distribution_and_task_conditioning_is_forbidden(domain):
    old=default_spec(domain,42,config={'lifecycle_steps':0});new=spec(domain,42,steps=0);notes=[]
    w=generate(new,diagnostics=notes)
    assert w==generate(old) and notes[0]['actual_transitions']==0 and not notes[0]['histories']
    assert replay_history(domain,42,new.generator_config,notes[0])==w
    new.generator_config={'task':'engineer a failure'}
    with pytest.raises(ValueError):generate(new)


def test_lifecycle_history_is_not_mislabelled_as_a_generation_exclusion():
    from usami_benchmark.audit import audit_domain
    report=audit_domain('education',[0],generator_version='education-lifecycle-v2',rules_version='education-0.1.0')
    assert report['valid_worlds']==1 and not report['invalid']
    assert report['generation_history']['actual_business_operations']>0
    assert not any(d.get('kind')=='native_dialogue_lifecycle_v2' for d in report['generation_exclusions'])
