"""Service reflection projection and preregistered world/certificate boundaries."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from integrations.usami import ServiceTaskProfile, UsamiProfile
from integrations.usami.domains import native_task
from vast_coevo.reflection.reflector import ReflectionInput, ServiceReflector
from vast_coevo.skills import empty_snapshot
from test_vast_customer_goal import with_tool
from test_vast_reflection import MODEL, Client, completion, noop

EXAMPLES=Path(__file__).resolve().parents[1]/'experiments/vast_coevo/examples'
sys.path.insert(0,str(EXAMPLES))
spec=importlib.util.spec_from_file_location('repair_study',EXAMPLES/'continue_coevolution.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)


def test_service_compact_projection_preserves_real_selected_tool_and_oracle(tmp_path):
    row,view=with_tool(tmp_path)
    value=ReflectionInput(generation=0,role='service',library=empty_snapshot('service'),evidence=(row,),interaction_views=(view,))
    record=ServiceReflector(Client(completion(noop(value).model_dump_json())),MODEL).reflect(value)
    payload=json.loads(record.call.request['messages'][-1]['content'])
    assert record.calls==1 and record.input.evidence==(row,)
    assert 'messages' not in payload['evidence'][0]
    assert payload['evidence'][0]['interaction']['service_tools'][0]['arguments']['resource_id']=='wrong'
    assert payload['evidence'][0]['interaction']['oracle_events']==[]
    assert record.prepared_interaction_views==(view,)
    assert 'Repair verified decision failures' in record.call.request['messages'][0]['content']


def test_old_service_task_serialization_is_unchanged():
    task=ServiceTaskProfile(task_ref='legal',config_id='world',scenario='dependency_boot')
    assert task.model_dump(mode='json')==dict(task_ref='legal',config_id='world',scenario='dependency_boot')


@pytest.mark.parametrize('kind',['permission','suspension','dependency'])
def test_frozen_refusal_uses_real_reference_and_rejects_tampered_certificate(kind):
    world=study.development_world(kind,801)
    profile=UsamiProfile(config_id='dev',domain='cloudops',seed=world.seed,identity='u0',mode='sandbox',split='dev',frozen_world=world.model_dump(mode='json'))
    task=study.refusal_profile(kind,world,profile.config_id)
    pinned,certified,_=native_task(profile,task)
    assert certified.initial_state_hash==pinned.state_hash
    raw=task.model_dump(mode='json');raw['frozen_task']['refusal_code']='MADE_UP'
    with pytest.raises(ValueError,match='Reference'):
        native_task(profile,ServiceTaskProfile.model_validate(raw))


def test_twenty_worlds_are_valid_structurally_distinct_and_all_projects_visible():
    rows=[study.novel_world(i) for i in range(20)]
    assert len({study.semantic_shape(w,a) for w,a,_ in rows})==20
    for world,actor,features in rows:
        assert not study.invariants(world)
        assert len(world.projects)==4
        assert all(study.owner(world,actor,pid) or study.role(world,actor,pid) is not None for pid in world.projects)
        assert all(pid.startswith('zone') for pid in world.projects)
        assert len(set(features['topologies']))==4
        assert len(set(features['actor_roles']))>=3
        assert {'active','suspended'}==set(features['states'])
    # Seeds and cosmetic names cannot manufacture structural diversity.
    first,actor,_=rows[0];changed=first.clone();changed.seed+=100
    for project in changed.projects.values():project.name='Cosmetic only'
    assert study.semantic_shape(first,actor)==study.semantic_shape(changed,actor)
