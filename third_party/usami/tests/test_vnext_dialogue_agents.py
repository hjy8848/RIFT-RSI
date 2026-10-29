"""Two real agent code paths using recorded function responses, no paid model."""
import copy,json
import pytest
from types import SimpleNamespace
from jsonschema import Draft202012Validator
from test_vast_usami import ScriptedClient,answer,call,SPEC
from test_vnext_aftersales import prepared,operation
from usami_aftersales import adapter
from usami_aftersales.rules import case_key,quote_key
from usami_dialogue.agents import run,replay,RunConfig,SkillPromptContext,service_response_schema
from usami_harness.models import Budget


def configured(world,profile_id,**budget):
    limits=dict(max_customer_turns=5,max_customer_llm_calls=7,max_service_llm_calls=12,max_service_tool_calls=12,max_format_recoveries=1);limits.update(budget)
    return RunConfig(profile_id=profile_id,initial_world_hash=world.state_hash,tools=list(adapter.TOOLS),budget=Budget(**limits))


def intent(identifier,**value):return answer(calls=[call(identifier,'submit_customer_turn',{'message':'A natural response to the current business discussion.',**value})])
def service_tool(identifier,op):return answer(calls=[call(identifier,op.name,op.model_dump(mode='json',exclude={'name'}))])
def reply(identifier,message='Please clarify.',op=None):return answer(calls=[call(identifier,'respond_to_customer',{'message':message,'offered_operation':op.model_dump(mode='json') if op else None})])


def test_aftersales_response_schema_only_accepts_quote_backed_operations_or_null():
    source,order,_=prepared()
    schema=service_response_schema(adapter)['function']['parameters']
    validator=Draft202012Validator(schema)
    commit=operation(source,'commit_resolution',order,quote_id='quote-example',destination_channel=order.payment_channel)
    release=operation(source,'release_refund',order,case_id='case-example')
    valid=lambda op:{'message':'Here are the terms.','offered_operation':op.model_dump(mode='json')}

    assert validator.is_valid({'message':'No quote is needed.','offered_operation':None})
    assert validator.is_valid(valid(commit))
    assert validator.is_valid(valid(release))
    assert not validator.is_valid({'message':'Malformed empty offer.','offered_operation':{}})
    assert not validator.is_valid(valid(operation(source,'verify_evidence',order,case_id='case-example',evidence_id='evidence-example')))
    assert not validator.is_valid({'message':'Missing operation name.','offered_operation':{
        'scope_id':order.merchant_id,'resource_id':order.id,'params':{'case_id':'case-example'}}})
    assert not validator.is_valid({'message':'Unexpected parameter.','offered_operation':{
        'name':'release_refund','scope_id':order.merchant_id,'resource_id':order.id,
        'params':{'case_id':'case-example','approved':True}}})


def test_domains_without_offer_operation_registry_keep_the_archived_response_schema():
    assert service_response_schema(SimpleNamespace(name='legacy-domain'))=={
        'type':'function','function':{'name':'respond_to_customer','description':'Reply, ask or present real quote-backed terms.',
            'parameters':{'type':'object','properties':{
                'message':{'type':'string','minLength':1},
                'offered_operation':{'anyOf':[{'type':'object'},{'type':'null'}]},
                'refused_operation':{'anyOf':[{'type':'object'},{'type':'null'}]},
                'refusal_code':{'anyOf':[{'type':'string'},{'type':'null'}]},
            },'required':['message','offered_operation'],'additionalProperties':False}}}


def test_invalid_empty_offer_is_rejected_without_action_and_recovered_as_protocol_error():
    source,_,_=prepared();world=source.initial;pid=source.profile_id
    customer=CheckedClient(SPEC,[intent('c0'),intent('c1',finish=True)])
    malformed=answer(calls=[call('r0','respond_to_customer',{'message':'Here is a quote.','offered_operation':{}})])
    service=CheckedClient(SPEC,[malformed,reply('r1','No quote is needed yet.')])
    result=run(adapter,world,configured(world,pid),customer,service)

    sent_schema=service.requests[0]['tools'][-1]['function']['parameters']
    assert sent_schema==service_response_schema(adapter)['function']['parameters']
    assert result.status=='customer_finished'
    assert result.cost['tool_calls']==0
    assert len(result.trace.events)==4  # setup, customer, corrected reply, finish
    assert result.errors[0]['kind']=='protocol'
    assert 'offered_operation must be null or a well-formed quote-backed operation' in result.errors[0]['message']
    assert replay(adapter,result,world=world)['verified']


def assert_wire_legal(messages):
    pending=[]
    for message in messages:
        if pending:
            assert message['role']=='tool' and message['tool_call_id'] in pending
            pending.remove(message['tool_call_id']);continue
        assert message['role']!='tool'
        if message['role']=='assistant':pending=[c['id'] for c in message.get('tool_calls',[])]
    assert not pending


class CheckedClient(ScriptedClient):
    def complete(self,request):
        assert_wire_legal(request['messages'])
        return super().complete(request)


def test_fixed_customer_and_service_function_calls_complete_refund_and_wire_replay():
    source,order,line=prepared();world=source.initial;profile=source.private;pid=source.profile_id
    opening=operation(source,'open_case',order,line_id=line.id,issue=profile.facts['issue'])
    cid=case_key(order.id,line.id,profile.facts['issue'])
    verify=operation(source,'verify_evidence',order,case_id=cid,evidence_id=profile.facts['evidence_id'])
    quote=operation(source,'quote_resolution',order,case_id=cid,option='refund')
    commit=operation(source,'commit_resolution',order,quote_id=quote_key(cid,'refund',1),destination_channel=order.payment_channel)
    customer=CheckedClient(SPEC,[intent('c0'),intent('c1',preferences={'resolution':'refund'},requested_operations=[opening.model_dump(mode='json')]),
        intent('c2',requested_operations=[verify.model_dump(mode='json'),quote.model_dump(mode='json')]),intent('c3',confirm_operation=commit.model_dump(mode='json')),intent('c4',finish=True)])
    service=CheckedClient(SPEC,[reply('r0','Which order and line?'),service_tool('s0',opening),reply('r1','Please provide your evidence.'),
        service_tool('s1',verify),service_tool('s2',quote),reply('r2','These are the actual refund terms. Do you confirm?',commit),service_tool('s3',commit),reply('r3','The real payment gateway has issued the refund.')])
    result=run(adapter,world,configured(world,pid),customer,service)
    assert result.status=='customer_finished' and result.oracle['status']=='pass'
    assert result.trace.final_world['lines'][line.id]['status']=='refunded'
    assert result.cost['customer_calls']==5 and result.cost['service_calls']==8 and result.cost['tool_calls']==4
    assert result.cost['usage_complete'] is False
    for row in result.calls:
        if row.role=='service':
            wire=json.dumps(row.record.request)
            assert 'private_hash' not in wire and 'simulated_valid' not in wire and 'confirmable_actions' not in wire
    assert replay(adapter,result,world=world)['verified']
    assert replay(adapter,json.loads(json.dumps(result.model_dump(mode='json'),sort_keys=True)),world=world)['verified']
    changed=result.model_copy(deep=True);changed.calls[0].record.response['choices'][0]['message']['content']='tampered'
    with pytest.raises(ValueError,match='integrity'):replay(adapter,changed)


def test_protocol_recovery_is_bounded_recorded_and_does_not_create_business_error():
    source,order,line=prepared();world=source.initial;pid=source.profile_id
    customer=CheckedClient(SPEC,[{'choices':[]},intent('ok',finish=True)])
    service=CheckedClient(SPEC,[])
    result=run(adapter,world,configured(world,pid),customer,service)
    assert result.status=='customer_finished' and len(result.errors)==1 and result.cost['customer_calls']==2
    assert result.oracle['metrics']['successful_business_violation']==0 and not result.oracle['operations']
    assert replay(adapter,result)['verified']


def test_budget_stop_and_transport_failure_are_preserved_without_retry():
    source,order,line=prepared();world=source.initial;pid=source.profile_id
    customer=CheckedClient(SPEC,[intent('a')]);service=CheckedClient(SPEC,[answer('Please provide more information.')])
    result=run(adapter,world,configured(world,pid,max_customer_llm_calls=1),customer,service)
    assert result.status=='customer_llm_budget_exhausted' and result.cost['customer_calls']==1
    assert replay(adapter,result)['verified']
    broken=CheckedClient(SPEC,[RuntimeError('fixture transport failure')]);service=CheckedClient(SPEC,[])
    failed=run(adapter,world,configured(world,pid),broken,service)
    assert failed.status=='customer_provider_error' and failed.cost['customer_calls']==1
    assert replay(adapter,failed)['verified']


def test_service_skill_context_is_injected_and_bound_to_exact_replay():
    source,_,_=prepared();world=source.initial;pid=source.profile_id
    context=SkillPromptContext(snapshot_hash='a'*64,skill_ids=['aftersales.test'],
        rendered='<VAST_SKILLS><VAST_SKILL id="aftersales.test" version="2">inspect live order</VAST_SKILL></VAST_SKILLS>')
    config=configured(world,pid).model_copy(update={'skill_contexts':{'service':context}})
    customer=CheckedClient(SPEC,[intent('c0'),intent('c1',finish=True)])
    service=CheckedClient(SPEC,[reply('r0','I can help with this request.')])
    result=run(adapter,world,config,customer,service)
    service_call=next(row for row in result.calls if row.role=='service')
    assert context.rendered in service_call.record.request['messages'][0]['content']
    assert result.config.skill_contexts['service'].context_hash==context.context_hash
    assert replay(adapter,result,world=world,config=config)['verified']

    tampered=context.model_copy(update={'rendered':context.rendered+' altered'})
    bad_config=config.model_copy(update={'skill_contexts':{'service':tampered}})
    with pytest.raises(ValueError,match='Skill prompt context hash differs'):
        run(adapter,world,bad_config,customer,service)
