"""Open multi-object dialogue tasks stay inside frozen native contracts."""
import pytest

from usami_aftersales import adapter as aftersales, tasks as aftersales_tasks
from usami_benchmark.registry import default_spec
from usami_benchmark.worlds import freeze
from usami_benchmark.tasks import (
    ConstructionBudget, OpenDialogueComposition, TaskSelection, compile_task, verify_task,
)
from usami_benchmark.designer import (
    _designer_tools, _selection_from_response, designer_view, design_and_compile,
    verify_designer_execution,
)
from usami_benchmark.contracts import DOMAINS
from usami_dialogue.tasks import CustomerFollowUp, Goal, PreferenceRequirement, InformationRequirement
from usami_harness.models import digest
from test_vast_usami import ScriptedClient, SPEC, answer, call


def multi_order_selection(frozen):
    world=aftersales.World.model_validate(frozen.world)
    # Pick two separately addressable orders for the same authenticated user;
    # order0/line0 has no legal witness in this seed, so use line1 explicitly.
    profile_ids=['profile_u0_order0_line1','profile_u0_order1_line0']
    assert all(pid in world.profiles and world.profiles[pid].identity=='u0' for pid in profile_ids)
    assert len({world.lines[world.profiles[pid].facts['line_id']].order_id for pid in profile_ids})==2

    frames=[];seen=set();goals=[];preferences=[];information=[];events=[]
    for profile_id,resolution in zip(profile_ids,['refund','exchange']):
        private=world.profiles[profile_id]
        line_id=private.facts['line_id']
        for frame in aftersales_tasks.frame_contract(world,private):
            key=digest(frame)
            if key not in seen:seen.add(key);frames.append(frame)
        goals.append(Goal(collection='lines',entity=line_id,field='status',
                          accepted_values=['refunded','exchanged','reshipped']))
        preferences.append(PreferenceRequirement(profile_id=profile_id,preference='resolution',accepted_values=[resolution]))
        information.append(InformationRequirement(profile_id=profile_id,fact_keys=['line_id','issue']))
        events.extend(event.id for event in world.external_events.values() if event.line_id==line_id)
    return TaskSelection(kind='dialogue',family='open_composition_v1',profile_id=profile_ids[0],composition=OpenDialogueComposition(
        profile_ids=profile_ids,
        customer_request='Please resolve the issues with both purchases using each purchase’s actual details.',
        goals=goals,frames=frames,external_schedule=events,
        information_requirements=information,preference_requirements=preferences,
    ))


def test_open_composition_certifies_multiple_objects_with_exact_replay():
    frozen=freeze(default_spec('aftersales',0))
    selection=multi_order_selection(frozen)
    record=compile_task(frozen,selection,ConstructionBudget(max_nodes=512,max_depth=16,max_paths=1))

    task=record.dialogue_witnesses[0].certificate.task
    trace=record.dialogue_witnesses[0].trace
    assert task.family=='open_composition_v1'
    assert task.composition.profile_ids==selection.composition.profile_ids
    assert len(task.goals)==2 and len({g.entity for g in task.goals})==2
    assert trace.scoped_profile_ids==selection.composition.profile_ids
    assert record.dialogue_witnesses[0].certificate.evaluation['success']
    assert record.dialogue_witnesses[0].certificate.evaluation['information_requirements']==[True,True]
    assert record.dialogue_witnesses[0].certificate.evaluation['preference_requirements']==[True,True]
    assert verify_task(record)


def _expired_open_case_refusal_selection(frozen, *, protocol='open-dialogue-composition-v2'):
    world=aftersales.World.model_validate(frozen.world)
    profile_id='profile_u0_order0_line0'
    private=world.profiles[profile_id]
    line=world.lines[private.facts['line_id']]
    order=world.orders[line.order_id]
    operation=aftersales.Operation(name='open_case',scope_id=order.merchant_id,
        resource_id=order.id,params={'line_id':line.id,'issue':private.facts['issue']})
    return TaskSelection(kind='dialogue',family='open_composition_v1',profile_id=profile_id,
        composition=OpenDialogueComposition(protocol=protocol,profile_ids=[profile_id],
            customer_request='I received this order with missing parts. Please open a case; if policy prevents it, explain why.',
            goals=[],refusal_operations=[operation.model_dump(mode='json')],
            frames=aftersales_tasks.frame_contract(world,private),
            information_requirements=[InformationRequirement(profile_id=profile_id,fact_keys=['line_id','issue'])],
            preference_requirements=[PreferenceRequirement(profile_id=profile_id,preference='resolution',accepted_values=['exchange'])]))


def test_v2_open_composition_certifies_a_native_refusal_without_fabricating_dependency():
    frozen=freeze(default_spec('aftersales',1))
    selection=_expired_open_case_refusal_selection(frozen)
    record=compile_task(frozen,selection,ConstructionBudget(max_nodes=4096,max_depth=16,max_paths=1))
    witness=record.dialogue_witnesses[0]
    assert witness.certificate.task.composition.protocol=='open-dialogue-composition-v2'
    assert witness.certificate.task.require_dependency is False
    assert witness.certificate.dependencies==[]
    assert witness.certificate.evaluation['success']
    assert witness.certificate.evaluation['refusal_subgoals']==[True]
    assert witness.certificate.oracle['status']=='pass'
    assert verify_task(record)


def test_v4_refusal_task_requires_public_followup_and_keeps_harness_label_private():
    from usami_dialogue.preflight import customer_interaction_preflight
    from usami_dialogue.user_scenario import scenario_from_task

    frozen = freeze(default_spec('aftersales', 1))
    selection = _expired_open_case_refusal_selection(frozen, protocol='open-dialogue-composition-v4')
    public_message = 'I still need a support case opened for the item in my order.'
    selection.composition.follow_up_requests = [CustomerFollowUp(
        message=public_message, requested_action='open_case')]
    record = compile_task(frozen, selection, ConstructionBudget(max_nodes=4096, max_depth=16, max_paths=1))
    task = record.dialogue_witnesses[0].certificate.task
    assert task.composition.protocol == 'open-dialogue-composition-v4'
    assert task.composition.follow_up_requests[0].requested_action == 'open_case'
    world = aftersales.World.model_validate(frozen.world)
    scenario = scenario_from_task(aftersales, world, task)
    assert scenario.follow_up_requests == (public_message,)
    assert 'requested_action' not in scenario.model_dump(mode='json')
    preflight = customer_interaction_preflight(aftersales, world, task, scenario)
    assert preflight['status'] == 'pass'
    assert verify_task(record)


def test_v4_refusal_task_rejects_missing_or_misaligned_customer_followup():
    frozen = freeze(default_spec('aftersales', 1))
    selection = _expired_open_case_refusal_selection(frozen, protocol='open-dialogue-composition-v4')
    with pytest.raises(ValueError, match='one public Customer follow-up'):
        compile_task(frozen, selection, ConstructionBudget(max_nodes=4096, max_depth=16, max_paths=1))
    selection.composition.follow_up_requests = [CustomerFollowUp(
        message='Please prepare a quote for this order.', requested_action='quote_resolution')]
    with pytest.raises(ValueError, match='action metadata must match'):
        compile_task(frozen, selection, ConstructionBudget(max_nodes=4096, max_depth=16, max_paths=1))


def test_mixed_completion_and_refusal_search_uses_the_refusal_profiles_customer_context():
    """A prior profile's missing facts must not mask the target policy refusal."""
    from usami_benchmark.designer_exploration import compile_proposal

    frozen=freeze(default_spec('aftersales',1))
    proposal={
        'profile_ids':['profile_u0_order0_line0','profile_u0_order1_line0'],
        'customer_request':(
            'Please assess each order on its own facts. The older missing-parts report may be outside its remedy '
            'window, while the recent wrong-item return may still be started.'
        ),
        'goals':[{'collection':'lines','entity':'u0_order1_line0','field':'status',
                  'accepted_values':['return_pending']}],
        'refusal_operations':[{'name':'open_case','scope_id':'m0','resource_id':'u0_order0',
                               'params':{'line_id':'u0_order0_line0','issue':'missing_parts'}}],
        'information_requirements':[
            {'profile_id':'profile_u0_order0_line0','fact_keys':['order_id','line_id','issue','evidence_id']},
            {'profile_id':'profile_u0_order1_line0','fact_keys':['order_id','line_id','issue','evidence_id','deadline','pickup_slot']},
        ],
        'preference_requirements':[
            {'profile_id':'profile_u0_order1_line0','preference':'resolution','accepted_values':['refund']},
        ],
        'confirmation_requirements':[],
        'external_schedule':[],
    }
    _,record=compile_proposal(frozen,proposal,
        ConstructionBudget(max_nodes=4096,max_depth=16,max_paths=1))

    witness=record.dialogue_witnesses[0]
    task=witness.certificate.task
    assert task.refusals[0].code=='AFTERSALES_WINDOW_EXPIRED'
    assert witness.certificate.evaluation['success']
    assert witness.certificate.evaluation['refusal_subgoals']==[True]
    assert witness.certificate.oracle['status']=='pass'
    assert verify_task(record)


def _created_case_proposal(frozen):
    world=aftersales.World.model_validate(frozen.world)
    profile_id='profile_u0_order0_line0'
    private=world.profiles[profile_id]
    return {
        'profile_ids':[profile_id],
        'customer_request':'Please open a support case for the issue on this item.',
        'created_goals':[{'profile_id':profile_id,'collection':'cases',
            'selectors':{'line_id':private.facts['line_id'],'issue':private.facts['issue']},
            'field':'status','accepted_values':['evidence_verified']}],
        'information_requirements':[{'profile_id':profile_id,'fact_keys':['line_id','issue','evidence_id']}],
    }


def test_created_object_goal_is_proven_by_native_engine_oracle_and_replay():
    from usami_benchmark.designer_exploration import PROTOCOL, compile_proposal

    frozen=freeze(default_spec('aftersales',0))
    proposal=_created_case_proposal(frozen)
    selection,record=compile_proposal(frozen,proposal,
        ConstructionBudget(max_nodes=512,max_depth=12,max_paths=1),protocol=PROTOCOL)

    witness=record.dialogue_witnesses[0]
    task=witness.certificate.task
    assert task.protocol=='certified-dialogue-task-v2'
    assert task.composition.protocol=='open-dialogue-composition-v3'
    assert task.goals==[] and len(task.created_goals)==1
    assert task.created_goals[0].selectors==proposal['created_goals'][0]['selectors']
    assert witness.certificate.evaluation['created_completion_subgoals']==[True]
    assert witness.certificate.evaluation['oracle_status']=='pass'
    assert witness.certificate.oracle['status']=='pass'
    assert verify_task(record)
    final=aftersales.World.model_validate(witness.trace.final_world)
    cases=[case for case in final.cases.values()
           if case.line_id==task.created_goals[0].selectors['line_id']
           and case.issue==task.created_goals[0].selectors['issue']]
    assert len(cases)==1 and cases[0].status=='evidence_verified'
    assert selection.composition.created_goals==task.created_goals


def test_created_object_goal_rejects_ungrounded_ids_unselected_profiles_and_missing_frame_anchor():
    from usami_benchmark.designer_exploration import PROTOCOL, compile_proposal

    frozen=freeze(default_spec('aftersales',0))
    proposal=_created_case_proposal(frozen)

    proposal['created_goals'][0]['selectors']['line_id']='invented-line-id'
    with pytest.raises(ValueError,match='not grounded in the selected Customer profile'):
        compile_proposal(frozen,proposal,ConstructionBudget(),protocol=PROTOCOL)
    proposal=_created_case_proposal(frozen)

    proposal['created_goals'][0]['selectors']={'issue':proposal['created_goals'][0]['selectors']['issue']}
    with pytest.raises(ValueError,match='not covered by a native create-enabled Customer frame'):
        compile_proposal(frozen,proposal,ConstructionBudget(),protocol=PROTOCOL)
    proposal=_created_case_proposal(frozen)

    proposal['created_goals'][0]['profile_id']='profile_u1_order0_line0'
    with pytest.raises(ValueError,match='created-object goal must be anchored to a selected Customer profile'):
        compile_proposal(frozen,proposal,ConstructionBudget(),protocol=PROTOCOL)

    proposal=_created_case_proposal(frozen)
    proposal['created_goals'][0]['field']='evidence_notice'
    with pytest.raises(ValueError,match='private or generated-ID fields'):
        compile_proposal(frozen,proposal,ConstructionBudget(),protocol=PROTOCOL)


def test_v1_refusal_dependency_semantics_remain_unchanged():
    frozen=freeze(default_spec('aftersales',1))
    selection=_expired_open_case_refusal_selection(frozen,protocol='open-dialogue-composition-v1')
    with pytest.raises(ValueError,match='claimed dialogue task has no actual information/consent dependency'):
        compile_task(frozen,selection,ConstructionBudget(max_nodes=4096,max_depth=16,max_paths=1))


def test_v2_completion_cannot_opt_out_of_process_dependency():
    from usami_dialogue.tasks import checked_task
    frozen=freeze(default_spec('aftersales',0))
    record=compile_task(frozen,multi_order_selection(frozen),ConstructionBudget(max_nodes=512,max_depth=16,max_paths=1))
    task=record.dialogue_witnesses[0].certificate.task.model_copy(update={'require_dependency':False,'task_hash':''})
    with pytest.raises(ValueError,match='dependency policy differs'):
        checked_task(aftersales,aftersales.World.model_validate(frozen.world),task)


def test_open_composition_rejects_cross_identity_profile_scope():
    frozen=freeze(default_spec('aftersales',0))
    selection=multi_order_selection(frozen)
    world=aftersales.World.model_validate(frozen.world)
    other=next(pid for pid,p in world.profiles.items() if p.identity!='u0')
    composition=selection.composition.model_copy(update={'profile_ids':[selection.profile_id,other]})
    with pytest.raises(ValueError,match='same authenticated identity'):
        compile_task(frozen,selection.model_copy(update={'composition':composition}))


def test_open_composition_rejects_selected_profile_scope_leak():
    frozen=freeze(default_spec('aftersales',0))
    selection=multi_order_selection(frozen)
    composition=selection.composition.model_copy(update={
        'customer_request':selection.composition.customer_request+' Also process u1_order0_line0.'
    })
    with pytest.raises(ValueError,match='outside the selected frozen profile scope'):
        compile_task(frozen,selection.model_copy(update={'composition':composition}))


def test_legacy_task_selection_serialization_omits_new_composition_field():
    selection=TaskSelection(kind='dialogue',family='refund',profile_id='profile_u1_enrollment0')
    assert 'composition' not in selection.model_dump(mode='json')


@pytest.mark.parametrize('domain',DOMAINS)
def test_designer_view_is_frozen_and_includes_native_contract(domain):
    frozen=freeze(default_spec(domain,0))
    before=frozen.world_hash
    view=designer_view(frozen,protocol='frozen-world-task-designer-v1')
    assert view['world_hash']==before
    assert view['world']==frozen.world
    if domain in {'saas','cloudops','fulfillment','stayops'}:
        assert view['track']=='A'
        assert view['operation_schema'] and view['native_goal_fields']
        assert view['customer_identities_and_visible_entities']
    else:
        assert view['track']=='B'
        assert view['customer_profiles_and_native_task_contracts']
        assert all(row['native_candidate_intents'] for row in view['customer_profiles_and_native_task_contracts'])


def test_designer_skill_and_verified_history_are_in_the_replayed_request():
    frozen=freeze(default_spec('aftersales',0))
    selection=multi_order_selection(frozen)
    skill='<VAST_SKILLS><VAST_SKILL id="designer.test" version="2">keep order facts separate</VAST_SKILL></VAST_SKILLS>'
    proposal={key:value for key,value in selection.model_dump(mode='json').items()
              if key in {'kind','family','profile_id','composition'}}
    selection_response=answer(calls=[call('designer-task','submit_frozen_world_task',proposal)])
    client=ScriptedClient(SPEC,[selection_response])
    context={'skill_snapshot_hash':'a'*64,'skill_rendered':skill,'evidence_ids':['task-1'],
        'hypothesis':'Test whether a second existing order can be composed safely.',
        'source_evidence':[{'episode_id':'task-1','process_errors':[{'classification':'false_rejection'}]}]}
    execution=design_and_compile(frozen,client,ConstructionBudget(max_nodes=512,max_depth=16,max_paths=1),
                                 prompt_context=context,protocol='frozen-world-task-designer-v1')
    system=execution.request['messages'][0]['content']
    evidence='\n'.join(row['content'] for row in execution.request['messages'][1:] if row['role']=='user')
    assert skill in system and context['skill_snapshot_hash'] in system
    assert 'task-1' in evidence and 'false_rejection' in evidence and context['hypothesis'] in evidence
    assert len(execution.task.selection.composition.profile_ids)==2
    assert verify_designer_execution(execution)['verified']


def test_dialogue_designer_tool_schema_only_exposes_composition_fields():
    frozen=freeze(default_spec('aftersales',0))
    schema=_designer_tools(frozen)[0]['function']['parameters']
    assert set(schema['properties'])=={'kind','family','profile_id','composition'}
    assert set(schema['required'])=={'kind','family','profile_id','composition'}
    assert schema['properties']['kind']['const']=='dialogue'
    assert schema['properties']['family']['const']=='open_composition_v1'
    world=aftersales.World.model_validate(frozen.world)
    assert schema['properties']['profile_id']['enum']==sorted(world.profiles)


def test_dialogue_designer_parser_rejects_tool_only_fields():
    frozen=freeze(default_spec('aftersales',0))
    selection=multi_order_selection(frozen)
    proposal={key:value for key,value in selection.model_dump(mode='json').items()
              if key in {'kind','family','profile_id','composition'}}
    proposal['message']='This top-level field conflicts with an open composition.'
    response=answer(calls=[call('designer-task','submit_frozen_world_task',proposal)])
    with pytest.raises(ValueError):
        _selection_from_response(response,frozen)
