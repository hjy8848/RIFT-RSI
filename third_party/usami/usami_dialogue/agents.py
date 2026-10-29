"""Real function-calling Customer/Service interaction; fixtures use this exact runner."""
import copy,json
from typing import Literal,Union
from pydantic import Field,model_validator
from usami_harness.models import Model,Budget,digest
from usami_harness.tools import inline_schema,schemas,parse
from usami.service.models import LLMRecord,ModelSettings,strict_json,ServiceError
from usami.service.runner import completion_message
from .models import CustomerIntent
from .engine import DialogueSession,ExecutionTrace
from .oracle import evaluate
from .tasks import DialogueTask,TaskCertificate,verify_certificate


class SkillPromptContext(Model):
    """Exact rendered Skill text placed in an agent's system message."""
    snapshot_hash:str
    skill_ids:list[str]=Field(default_factory=list)
    rendered:str
    context_hash:str=''
    @model_validator(mode='after')
    def seal(self):
        expected=self.computed_hash()
        if self.context_hash and self.context_hash!=expected:raise ValueError('Skill prompt context hash differs')
        object.__setattr__(self,'context_hash',expected)
        return self
    def computed_hash(self):
        return digest({'snapshot_hash':self.snapshot_hash,'skill_ids':self.skill_ids,'rendered':self.rendered})
    @classmethod
    def checked(cls,value):
        row=cls.model_validate(value.model_dump(mode='json') if hasattr(value,'model_dump') else value)
        if row.context_hash and row.context_hash!=row.computed_hash():raise ValueError('Skill prompt context hash differs')
        return row.model_copy(update={'context_hash':row.computed_hash()})


class RunConfig(Model):
    protocol:Literal['business-dialogue-agents-v2']='business-dialogue-agents-v2'
    profile_id:str
    initial_world_hash:str
    mode:Literal['reference','sandbox']='reference'
    tools:list[str]
    budget:Budget=Field(default_factory=Budget)
    external_schedule:list[str]=Field(default_factory=list)
    # Optional content-addressed Skill context. Omitted when empty so archived
    # v2 configurations retain their original serialization and replay behavior.
    skill_contexts:dict[str,SkillPromptContext]=Field(default_factory=dict,exclude_if=lambda value:not value)


class TaskRunConfig(RunConfig):
    # A separate discriminator preserves the original journal/config shape.
    protocol:Literal['business-dialogue-task-agents-v2']='business-dialogue-task-agents-v2'
    task:DialogueTask
    certificate:TaskCertificate
    witness:ExecutionTrace


class AgentCall(Model):
    role:Literal['customer','service']
    record:LLMRecord


class AgentExecution(Model):
    protocol:Literal['business-dialogue-agents-v2','business-dialogue-task-agents-v2']='business-dialogue-agents-v2'
    config:Union[RunConfig,TaskRunConfig]=Field(discriminator='protocol')
    customer_model:ModelSettings
    service_model:ModelSettings
    calls:list[AgentCall]
    tools:list[dict]
    trace:ExecutionTrace
    oracle:dict
    status:str
    errors:list[dict]
    cost:dict
    execution_hash:str=''
    def computed_hash(self):return digest(self.model_dump(mode='json',exclude={'execution_hash'}))


CUSTOMER_POLICY=(
 'You are the ordinary customer, not a test Designer, Reflector or business operator. Your model and policy stay fixed. '
 'Your frozen private profile contains the only objective facts you may disclose and the reasonable preference values you may choose. '
 'Respond naturally to the actual Service reply; ask, clarify, compare, revise a permitted preference, confirm a real presented offer, or finish. '
 'Do not disclose all private facts automatically. Never fabricate facts or identities, infer hidden Oracle answers, or alter objective facts after the fact. '
 'Use submit_customer_turn once. requested_operations express exact business writes YOU actually request; only declare them from the supplied current candidate intents. '
 'These candidates disclose no policy approval and are not an execution order. You may approve multiple relevant preparatory operations. '
 'Confirm an irreversible action only by copying the actual publicly presented operation, after checking its terms against your preferences. '
 'A newly changed choice needs a suitable current offer. Stop when the request is resolved or a real refusal is explained; do not force a number of turns.'
)
SERVICE_POLICY=(
 'You provide service in a deterministic simulated business. Query and choose your own business tools and order. '
 'The authenticated customer is fixed. Only customer-declared exact write intentions authorize writes; tool arguments and your own messages cannot create authority. '
 'Before irreversible action obtain the necessary facts, chosen resolution and explicit current consent. Never invent evidence, approval, receipt or success. '
 'To ask or reply use respond_to_customer(message, offered_operation). To present binding terms, offered_operation must identify the actual quote-backed action; '
 'the host displays its real terms to the customer. Never expose private simulator state, switch actor, or assume consent. '
 'Reject inappropriate operations using respond_to_customer refused_operation and refusal_code; do not try a forbidden write merely to have the Engine reject it. Decide autonomously; no reference witness is supplied.'
)


def offered_operation_schema(adapter):
    """Build the response contract from this domain's quote-bearing operations.

    Domains without an explicit list retain the historical nullable arbitrary
    operation schema. Domains that declare OFFERABLE_OPERATIONS get a closed,
    discriminated union so an empty object or a non-offer action cannot be
    mistaken for a customer-facing quote.
    """
    names=getattr(adapter,'OFFERABLE_OPERATIONS',None)
    if names is None:
        return {'anyOf':[{'type':'object'},{'type':'null'}]}
    alternatives=[]
    for name in names:
        if name not in adapter.PARAMETERS:
            raise ValueError('offerable operation is absent from the domain schema: '+str(name))
        alternatives.append({'type':'object','properties':{
            'name':{'const':name},
            'scope_id':{'type':'string','minLength':1},
            'resource_id':{'type':'string','minLength':1},
            'params':inline_schema(adapter.PARAMETERS[name]),
        },'required':['name','scope_id','resource_id','params'],'additionalProperties':False})
    if not alternatives:
        return {'type':'null'}
    return {'anyOf':alternatives+[{'type':'null'}]}


def service_response_schema(adapter):
    names=getattr(adapter,'OFFERABLE_OPERATIONS',None)
    if names is None:
        # Preserve archived tool schemas byte-for-byte for domains that have
        # not declared their quote-bearing operation set.
        return {'type':'function','function':{'name':'respond_to_customer','description':'Reply, ask or present real quote-backed terms.',
            'parameters':{'type':'object','properties':{'message':{'type':'string','minLength':1},'offered_operation':{'anyOf':[{'type':'object'},{'type':'null'}]},'refused_operation':{'anyOf':[{'type':'object'},{'type':'null'}]},'refusal_code':{'anyOf':[{'type':'string'},{'type':'null'}]}},'required':['message','offered_operation'],'additionalProperties':False}}}
    if getattr(adapter,'name',None)=='aftersales':
        offer_description='Use null when no current quote is being presented. For after-sales, only commit_resolution or release_refund backed by a real current quote is valid.'
    else:
        offer_description='Use null when no current quote is being presented. Otherwise provide the exact quote-backed operation supported by this domain.'
    return {'type':'function','function':{
        'name':'respond_to_customer',
        'description':'Reply, ask, or present terms. Use null when there is no current quote; otherwise pass the exact quote-backed operation.',
        'parameters':{'type':'object','properties':{
            'message':{'type':'string','minLength':1},
            'offered_operation':{**offered_operation_schema(adapter),'description':offer_description},
            'refused_operation':{'anyOf':[{'type':'object'},{'type':'null'}]},
            'refusal_code':{'anyOf':[{'type':'string'},{'type':'null'}]},
        },'required':['message','offered_operation'],'additionalProperties':False}}}


def checked_offered_operation(adapter,value):
    if value is None:return None
    try:operation=adapter.Operation.model_validate(value)
    except (ValueError,TypeError,KeyError):
        raise ValueError('offered_operation must be null or a well-formed quote-backed operation') from None
    allowed=getattr(adapter,'OFFERABLE_OPERATIONS',None)
    if allowed is not None and operation.name not in allowed:
        raise ValueError('offered_operation must be null or one of the domain quote-backed offer operations')
    return operation.model_dump(mode='json')


def _service_system_content(adapter,session,config,public_policy):
    """Render the exact v2 system message, including a registered policy version."""
    content=(SERVICE_POLICY+' Identity: '+session.private.identity+'. Public policy: '+
        json.dumps(public_policy,sort_keys=True)+'. Budget: '+config.budget.model_dump_json())
    if 'service' in config.skill_contexts:
        skill_context=SkillPromptContext.checked(config.skill_contexts['service'])
        content+='\n\nApplied declarative Service Skills from snapshot '+skill_context.snapshot_hash+':\n'+skill_context.rendered
    return content


def run(adapter,world,config,customer_client,service_client,*,_public_policy_override=None):
    raw=config.model_dump(mode='json') if isinstance(config,RunConfig) else config
    config=(TaskRunConfig if raw.get('protocol')=='business-dialogue-task-agents-v2' else RunConfig).model_validate(raw)
    brief=None
    if isinstance(config,TaskRunConfig):
        from .tasks import checked_task
        config.task=checked_task(adapter,world,config.task)
        if config.certificate.task!=config.task:raise ValueError('task differs from its private native certificate')
        verify_certificate(adapter,world,config.certificate,config.witness)
        if config.task.profile_id!=config.profile_id or config.task.world_hash!=config.initial_world_hash:raise ValueError('Customer task and run source registration differ')
        if config.external_schedule!=config.task.external_schedule:raise ValueError('task external schedule differs from run registration')
        if config.task.composition is not None:
            brief=config.task.composition.customer_request
        else:
            if not callable(getattr(adapter,'task_intent_allowed',None)):raise ValueError('domain has no fixed Customer task-scope validator')
            brief=adapter.customer_task_brief(world,adapter.private_state(world,config.profile_id),config.task)
    if config.initial_world_hash!=world.state_hash or set(config.tools)-set(adapter.TOOLS):raise ValueError('frozen world/tool configuration differs')
    if len(config.tools)!=len(set(config.tools)) or len(config.external_schedule)!=len(set(config.external_schedule)):raise ValueError('duplicate tool/external registration')
    if set(config.external_schedule)-set(world.external_events):raise ValueError('unregistered external event')
    scoped_ids=config.task.composition.profile_ids if isinstance(config,TaskRunConfig) and config.task.composition is not None else None
    session=DialogueSession(adapter,world,config.profile_id,max_operations=config.budget.max_service_tool_calls,profile_ids=scoped_ids)
    cmodel=customer_client.settings.model_copy(deep=True);smodel=service_client.settings.model_copy(deep=True)
    ctools=[{'type':'function','function':{'name':'submit_customer_turn','description':'One natural Customer response grounded in frozen facts.', 'parameters':CustomerIntent.model_json_schema()}}]
    stool=schemas(adapter,config.tools)+[service_response_schema(adapter)]
    public_policy=adapter.POLICY if _public_policy_override is None else _public_policy_override
    if _public_policy_override is not None:
        from usami_benchmark.public_policy import catalogue
        registered=[adapter.POLICY,catalogue(adapter.name,world.version)]
        if public_policy not in registered:raise ValueError('v2 Replay public policy is not a registered historical renderer')
    smessages=[{'role':'system','content':_service_system_content(adapter,session,config,public_policy)}]
    if session.scoped_profile_ids:
        private_profiles=[{'profile_id':pid,'profile':private.model_dump(mode='json')} for pid,private in session.profiles.items()]
        private_context=json.dumps(private_profiles,sort_keys=True,separators=(',',':'),ensure_ascii=False)
        customer_role=' Your selected frozen customer situations are: '+private_context+'. You may handle them in any order but must not combine facts across profiles in one turn.'
    else:
        customer_role=' Your private profile: '+json.dumps(session.private.model_dump(mode='json'),sort_keys=True,separators=(',',':'),ensure_ascii=False)+'.'
    cmessages=[{'role':'system','content':CUSTOMER_POLICY+customer_role+' Business write schemas: '+json.dumps(schemas(adapter,[n for n in config.tools if n in adapter.PARAMETERS and n not in adapter.Operation.queries]),sort_keys=True)}]
    if brief is not None:cmessages[0]['content']+=' Your chosen business request: '+brief
    calls=[];tool_records=[];errors=[];seen=set();conversation=[];pending=list(config.external_schedule)
    counts={'customer':0,'service':0};recoveries={'customer':0,'service':0};status='customer_turn_budget_exhausted'
    def complete(role,messages,tools):
        client=customer_client if role=='customer' else service_client;model=cmodel if role=='customer' else smodel
        request=copy.deepcopy(model.request(messages,tools));record=LLMRecord(request=request);row=AgentCall(role=role,record=record)
        counts[role]+=1;calls.append(row)
        try:row.record.response=client.complete(copy.deepcopy(request))
        except Exception as exc:
            row.record.response=getattr(exc,'response',None);row.record.error={'stage':'transport','code':getattr(exc,'code','PROVIDER_ERROR'),'message':str(exc)}
            errors.append({'role':role,'kind':'transport','message':str(exc)})
            return None
        return row.record
    def protocol_error(role,record,exc,messages):
        record.error={'stage':'protocol','code':'OUTPUT_INVALID','message':str(exc)};errors.append({'role':role,'kind':'protocol','message':str(exc)})
        recoveries[role]+=1
        if recoveries[role]>config.budget.max_format_recoveries:return False
        messages.append({'role':'user','content':'The previous output was invalid and caused no business action: '+str(exc)+'. Correct the output within the remaining frozen budget.'})
        return True
    stopped=False
    for turn in range(config.budget.max_customer_turns):
        if session.scoped_profile_ids:
            from .scope import customer_candidates
            available=customer_candidates(adapter,session.state,list(session.profiles.items()))
        else:available=adapter.candidate_intents(session.state,session.private)
        customer_observation={'conversation':conversation,'business_view':adapter.observation(session.state,session.private.identity),
            'presented_offer':session.presented_offer,'current_candidate_intents':available}
        cmessages.append({'role':'user','content':json.dumps(customer_observation,ensure_ascii=False,sort_keys=True)})
        submitted=None
        while counts['customer']<config.budget.max_customer_llm_calls:
            record=complete('customer',cmessages,ctools)
            if record is None:status='customer_provider_error';stopped=True;break
            try:
                message,proposals=completion_message(record.response,set())
                if len(proposals)!=1 or proposals[0]['function']['name']!='submit_customer_turn':raise ValueError('one submit_customer_turn required')
                submitted=CustomerIntent.model_validate(strict_json(proposals[0]['function']['arguments']))
                if isinstance(config,TaskRunConfig):
                    if config.task.composition is not None:
                        from .scope import profile_for_intent
                        profile_for_intent(adapter,session.state,list(session.profiles.items()),submitted)
                    elif not adapter.task_intent_allowed(session.private,config.task,submitted):raise ValueError('Customer intent leaves the frozen selected task goal')
                session.customer(submitted);cmessages.append(message)
                cmessages.append({'role':'tool','tool_call_id':proposals[0]['id'],'content':json.dumps({'accepted':True,'turn':session.turn},sort_keys=True)})
                break
            except (ValueError,TypeError,KeyError,ServiceError) as exc:
                if not protocol_error('customer',record,exc,cmessages):status='customer_protocol_error';stopped=True;break
        if stopped:break
        if submitted is None:status='customer_llm_budget_exhausted';break
        conversation.append({'role':'customer','message':submitted.message})
        if submitted.finish:status='customer_finished';break
        smessages.append({'role':'user','content':submitted.message+'\nCustomer-declared exact write intentions: '+json.dumps(session.grants,sort_keys=True)})
        replied=False
        while counts['service']<config.budget.max_service_llm_calls:
            record=complete('service',smessages,stool)
            if record is None:status='service_provider_error';stopped=True;break
            try:message,actions=completion_message(record.response,seen)
            except (ValueError,TypeError,KeyError,ServiceError) as exc:
                if not protocol_error('service',record,exc,smessages):status='service_protocol_error';stopped=True;break
                continue
            # Validate a response boundary before emitting any business action.
            response_calls=[a for a in actions if a['function']['name']=='respond_to_customer']
            if response_calls:
                try:
                    if len(actions)!=1:raise ValueError('reply/offer cannot be mixed with business tools')
                    args=strict_json(response_calls[0]['function']['arguments'])
                    if not isinstance(args,dict) or not {'message','offered_operation'}<=set(args) or set(args)-{'message','offered_operation','refused_operation','refusal_code'}:raise ValueError('reply requires message and offered_operation')
                    offered=checked_offered_operation(adapter,args['offered_operation'])
                    session.reply(args['message'],offered,refused_operation=args.get('refused_operation'),refusal_code=args.get('refusal_code'))
                except (ValueError,TypeError,KeyError,ServiceError) as exc:
                    if not protocol_error('service',record,exc,smessages):status='service_protocol_error';stopped=True;break
                    continue
                smessages.append(message);smessages.append({'role':'tool','tool_call_id':response_calls[0]['id'],'content':json.dumps({'delivered':True,'offer':session.presented_offer},sort_keys=True)})
                conversation.append({'role':'service','message':args['message'],'offer':session.presented_offer});replied=True;break
            if not actions:
                text=message.get('content')
                if not isinstance(text,str) or not text.strip():
                    if not protocol_error('service',record,ValueError('empty Service response'),smessages):status='service_protocol_error';stopped=True;break
                    continue
                session.reply(text);smessages.append(message);conversation.append({'role':'service','message':text});replied=True;break
            smessages.append(message)
            for action in actions:
                if len(tool_records)>=config.budget.max_service_tool_calls:status='service_tool_budget_exhausted';stopped=True;break
                before=session.state.state_hash
                try:
                    operation=parse(adapter,action['function']['name'],action['function']['arguments'],config.tools)
                    result={'success':True,'error_code':None,'data':adapter.observation(session.state,session.private.identity)} if operation is None else session.act(operation,config.mode)
                except (ValueError,TypeError,KeyError,ServiceError) as exc:result={'success':False,'error_code':'TOOL_ARGUMENT_INVALID','data':{},'message':str(exc)}
                tool_records.append({'call_id':action['id'],'name':action['function']['name'],'arguments':action['function']['arguments'],'result':result,'before_hash':before,'after_hash':session.state.state_hash})
                smessages.append({'role':'tool','tool_call_id':action['id'],'content':json.dumps(result,ensure_ascii=False,sort_keys=True)})
                if adapter.invariants(session.state):status='business_world_invalid';stopped=True;break
                for eid in list(pending):
                    if adapter.external_ready(session.state,eid):
                        session.external(eid);pending.remove(eid)
                        smessages.append({'role':'user','content':'A pre-registered external simulator update has arrived. Query the current business record before deciding.'})
            if stopped:break
        if stopped:break
        if not replied:status='service_llm_budget_exhausted';break
    trace=session.finish();oracle=evaluate(adapter,trace,initial_world=world,private=session.private)
    tokens={'input_tokens':0,'output_tokens':0,'total_tokens':0};usage_complete=True
    by_role={role:{**tokens,'calls':counts[role],'usage_complete':True} for role in ('customer','service')}
    for row in calls:
        usage=row.record.response.get('usage',{}) if isinstance(row.record.response,dict) else {}
        if not isinstance(usage,dict):usage={}
        required=('prompt_tokens','completion_tokens','total_tokens')
        if not all(isinstance(usage.get(k),int) and not isinstance(usage[k],bool) and usage[k]>=0 for k in required):
            usage_complete=False;by_role[row.role]['usage_complete']=False;continue
        for target,key in [('input_tokens','prompt_tokens'),('output_tokens','completion_tokens'),('total_tokens','total_tokens')]:
            tokens[target]+=usage[key];by_role[row.role][target]+=usage[key]
    cost={**tokens,'usage_complete':usage_complete,'by_role':by_role,'customer_calls':counts['customer'],'service_calls':counts['service'],'tool_calls':len(tool_records),'format_recoveries':recoveries}
    result=AgentExecution(protocol=config.protocol,config=config,customer_model=cmodel,service_model=smodel,calls=calls,tools=tool_records,trace=trace,oracle=oracle,status=status,errors=errors,cost=cost)
    result.execution_hash=result.computed_hash();return result


def replay(adapter,execution,*,world=None,config=None):
    from usami.service.client import RecordedCompletionClient
    from .replay import replay as replay_trace
    raw=execution.model_dump(mode='json') if isinstance(execution,AgentExecution) else execution
    if raw.get('protocol') not in {'business-dialogue-agents-v2','business-dialogue-task-agents-v2'}:raise ValueError('agent protocol requires its archived runtime; v2 uses canonical rendered context')
    row=AgentExecution.model_validate(raw)
    if row.protocol!=row.config.protocol:raise ValueError('agent execution/config protocol differs')
    if row.execution_hash!=row.computed_hash():raise ValueError('agent execution integrity differs')
    initial=adapter.World.model_validate(row.trace.initial_world)
    if world is not None and world!=initial:raise ValueError('registered world differs')
    if config is not None and config!=row.config:raise ValueError('registered config differs')
    replay_trace(adapter,row.trace,initial_world=initial,private=row.trace.private)
    public_policy_override=None
    service_calls=[call for call in row.calls if call.role=='service']
    if service_calls:
        # V2 journals predate the current policy renderer. Select only a
        # deterministic, source-registered renderer by comparing the complete
        # recorded first system message; never replay an arbitrary journal prompt.
        scope_ids=row.config.task.composition.profile_ids if isinstance(row.config,TaskRunConfig) and row.config.task.composition is not None else None
        session=DialogueSession(adapter,initial,row.config.profile_id,
            max_operations=row.config.budget.max_service_tool_calls,profile_ids=scope_ids)
        recorded_system=service_calls[0].record.request.get('messages',[{}])[0].get('content')
        current= _service_system_content(adapter,session,row.config,adapter.POLICY)
        if recorded_system==current:
            public_policy_override=adapter.POLICY
        else:
            from usami_benchmark.public_policy import catalogue
            legacy=catalogue(adapter.name,initial.version)
            if recorded_system!=_service_system_content(adapter,session,row.config,legacy):
                raise ValueError('historical v2 Service policy context is not a registered renderer')
            public_policy_override=legacy
    c=RecordedCompletionClient(row.customer_model,[r.record for r in row.calls if r.role=='customer'])
    s=RecordedCompletionClient(row.service_model,[r.record for r in row.calls if r.role=='service'])
    rebuilt=run(adapter,initial,row.config,c,s,_public_policy_override=public_policy_override)
    if rebuilt!=row or c.index!=len(c.records) or s.index!=len(s.records):raise ValueError('actual model request/response, disclosure, tools or budget replay differs')
    return {'verified':True,'model_calls':len(row.calls),'business_events':len(row.oracle['operations'])}
