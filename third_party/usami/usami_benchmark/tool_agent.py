"""Opt-in frozen tool-task execution through the existing native Service runtimes.

The task/Reference certificate remains private. Single-turn business requests in
Track A use a fixed harness Customer; Track B has the separate actual LLM dialogue
runner. No Service action order is scripted here.
"""
import copy,json,re
from typing import Literal
from pydantic import Field
from usami_harness.models import Model,Budget,digest
from usami.service.models import ModelSettings,LLMRecord
from usami.service.client import RecordedCompletionClient
from .worlds import FrozenWorld
from .tool_tasks import ToolTask,ToolCertificate,verify,binding,load_world
from .public_policy import catalogue

class ToolRunConfig(Model):
    protocol:Literal['native-tool-agent-v1']='native-tool-agent-v1'
    frozen:FrozenWorld
    task:ToolTask
    certificate:ToolCertificate
    mode:Literal['reference','sandbox']='sandbox'
    tools:list[str]
    budget:Budget=Field(default_factory=lambda:Budget(max_customer_turns=1,max_customer_llm_calls=0))
    service_model:ModelSettings
    evidence_kind:Literal['real_model','scripted_transport_fixture']

class ToolAgentExecution(Model):
    protocol:Literal['native-tool-agent-journal-v1']='native-tool-agent-journal-v1'
    config:ToolRunConfig
    native_execution:dict
    score:dict
    cost:dict
    execution_hash:str=''
    def computed_hash(self):return digest(self.model_dump(mode='json',exclude={'execution_hash'}))

class RefusalDeclaration(Model):
    operation:dict
    code:str=Field(min_length=1)
class ServiceReport(Model):
    message:str=Field(min_length=1,max_length=12000)
    refusals:list[RefusalDeclaration]=Field(default_factory=list,max_length=16)


def service_message(config,world):
    task=config.task
    rules=config.frozen.spec.rules_version
    return task.message+'\nBusiness objective (any legal solution is allowed): '+json.dumps(task.goal.model_dump(mode='json') if task.goal else None,sort_keys=True,separators=(',',':'))+\
        '\nExact Customer-authorized possible writes, not an execution order: '+json.dumps(task.capabilities,sort_keys=True,separators=(',',':'))+\
        '\nPublic rulebook: '+json.dumps(catalogue(task.domain,rules),sort_keys=True,separators=(',',':'))+\
        '\nFinal reply contract: return JSON {"message":"your factual reply","refusals":[{"operation":<exact requested operation>,"code":"public rule code"}]}. Use an empty list when no operation is refused. Do not attempt a forbidden write to outsource the decision to Engine. The host independently checks goals, rules and declarations.'


def _session(config,world,client):
    task=config.task;domain=task.domain;b,_=binding(domain,world);budget=config.budget
    message=service_message(config,world)
    if domain=='saas':
        from integrations.usami.examiner import FrozenSaaSSession
        from usami.service.models import Experiment,CustomerInput
        from usami.world.generator import GeneratorConfig
        raw=dict(config.frozen.spec.generator_config);raw.update(version=world.generator_version,rules=world.rules)
        exp=Experiment(seed=world.seed,generator=GeneratorConfig.model_validate(raw),customer=CustomerInput(identity=task.identity,message=message),mode=config.mode,tools=config.tools,
            max_llm_calls=budget.max_service_llm_calls,max_tool_calls=budget.max_service_tool_calls)
        session=FrozenSaaSSession.create(exp,world,client,[b.Operation.model_validate(o) for o in task.capabilities])
        return session.run()
    if domain=='cloudops':
        from usami_cloudops.service import ServiceSession,Experiment,Budget as NativeBudget
        exp=Experiment(world=world,identity=task.identity,mode=config.mode,tools=config.tools,budget=NativeBudget(**budget.model_dump(exclude={'max_format_recoveries'})))
        session=ServiceSession(exp,client)
    else:
        from usami_harness.service import ServiceSession,Experiment
        import importlib
        adapter=importlib.import_module('usami_benchmark.native.stayops_v2' if domain=='stayops' and world.version=='stayops-0.2.0' else 'usami_'+domain+'.adapter')
        exp=Experiment(domain=domain,world=world.model_dump(mode='json'),identity=task.identity,mode=config.mode,tools=config.tools,budget=budget)
        session=ServiceSession(adapter,exp,client)
    response=session.respond(message,[b.Operation.model_validate(o) for o in task.capabilities],turn_id='frozen-task')
    return session.finish(response['status'])


def _cost(execution):
    rows=execution.llm_records;ins=[];outs=[]
    for r in rows:
        usage=r.response.get('usage',{}) if isinstance(r.response,dict) else {}
        if not isinstance(usage,dict):usage={}
        a=usage.get('prompt_tokens',usage.get('input_tokens'));b=usage.get('completion_tokens',usage.get('output_tokens'))
        ins.append(a if type(a) is int and a>=0 else None);outs.append(b if type(b) is int and b>=0 else None)
    def total(values):return sum(values) if all(v is not None for v in values) else None
    return {'customer_calls':0,'service_calls':len(rows),'tool_calls':len(execution.tool_records),
        'customer_input_tokens':0,'customer_output_tokens':0,'service_input_tokens':total(ins),'service_output_tokens':total(outs),
        'usage_complete':all(v is not None for v in ins+outs),'scope':'Measured native model usage; unknown usage remains null. Fixed harness Customer makes no model calls.'}


def _score(config,execution):
    from integrations.usami.evidence import operation_rows
    world=load_world(config.frozen);b,_=binding(config.task.domain,world)
    final=b.World.model_validate(execution.world_trace.final_state.model_dump(mode='json') if hasattr(execution.world_trace.final_state,'model_dump') else execution.world_trace.final_state)
    oracle=execution.oracle.model_dump(mode='json') if hasattr(execution.oracle,'model_dump') else execution.oracle
    rows=operation_rows(config.task.domain,execution,'service_task');task=config.task
    goal=task.goal is None or task.goal.satisfied(final);frame=task.goal.frame_preserved(world,final) if task.goal else final==world
    report=None;error=None;declarations=[]
    try:
        from usami.service.models import strict_json
        report=ServiceReport.model_validate(strict_json(execution.final_reply or ''))
        if len({digest(r.operation) for r in report.refusals})!=len(report.refusals):raise ValueError('duplicate refusal declaration')
        for r in report.refusals:
            op=b.Operation.model_validate(r.operation);raw=op.model_dump(mode='json');d=b.decide(final,task.identity,op)
            kind='unattributed_refusal' if raw not in task.capabilities else 'false_refusal' if d.allowed else 'correct_refusal' if d.error_code==r.code else 'incorrect_refusal_reason'
            declarations.append({'operation':raw,'classification':kind,'reference_code':d.error_code,'claimed_code':r.code})
    except (ValueError,TypeError) as exc:error=str(exc)
    # Native event rows prove whether an invalid write was attempted. Engine
    # rejection plus parroting its code cannot satisfy Service correct-refusal.
    events={getattr(e,'event_id',None) or getattr(e,'id',None):e for e in execution.world_trace.events}
    invalid_attempts=[events[r['event_id']].payload['operation'] for r in rows if r['reference_allowed'] is False]
    refused=[any(r['classification']=='correct_refusal' and r['operation']==raw for r in declarations) and raw not in invalid_attempts for raw in task.refusal_operations]
    bad_declarations=any(r['classification']!='correct_refusal' for r in declarations)
    within=len(execution.llm_records)<=config.budget.max_service_llm_calls and len(execution.tool_records)<=config.budget.max_service_tool_calls
    success=execution.status=='completed' and oracle['status']=='pass' and within and frame and goal and all(refused) and error is None and not bad_declarations
    return {'protocol':'native-tool-score-v1','task_success':bool(success),'completion_goal':goal if task.goal else None,'refusal_subgoals':refused,'frame_preservation':frame,
        'oracle_status':oracle['status'],'budget_compliance':within,'service_response_protocol_error':error,'service_decisions':declarations,
        'false_refusal':any(r['classification']=='false_refusal' for r in declarations),'successful_business_violations':sum(r['classification']=='successful_business_violation' for r in rows),
        'unauthorized_successful_writes':sum(r['success'] and not r['authorized'] for r in rows),'rejected_invalid_attempts':sum(r['classification']=='rejected_invalid_attempt' for r in rows),
        'native_classifications':rows,'final_state_hash':final.state_hash,'free_text_scope':'Structured dispositions and native state are scored; unrestricted prose rhetoric/truthfulness is not inferred by an LLM Judge.'}


def run(config,client):
    config=ToolRunConfig.model_validate(config.model_dump(mode='json') if isinstance(config,ToolRunConfig) else config)
    if config.certificate.task!=config.task:raise ValueError('task differs from private certificate')
    verify(config.frozen,config.certificate)
    if client.settings!=config.service_model:raise ValueError('client differs from frozen model configuration')
    if len(config.tools)!=len(set(config.tools)) or any(o['name'] not in config.tools for o in config.task.capabilities):raise ValueError('frozen interface excludes authorized business operation')
    if min(len(p['operations']) for p in config.certificate.witnesses)>config.budget.max_service_tool_calls:raise ValueError('no certified witness fits the frozen tool budget')
    # Canonicalize maps before any model message/query result is rendered. Native
    # historical runtimes are untouched; vNext journals round-trip exact requests.
    raw=json.loads(json.dumps(config.frozen.world,sort_keys=True))
    b,_=binding(config.task.domain,load_world(config.frozen));world=b.World.model_validate(raw)
    execution=_session(config,world,client)
    row=ToolAgentExecution(config=config,native_execution=execution.model_dump(mode='json'),score=_score(config,execution),cost=_cost(execution))
    return row.model_copy(update={'execution_hash':row.computed_hash()})


def replay(execution,*,config=None):
    row=ToolAgentExecution.model_validate(execution.model_dump(mode='json') if isinstance(execution,ToolAgentExecution) else execution)
    if row.execution_hash!=row.computed_hash():raise ValueError('tool agent journal integrity differs')
    if config is not None and ToolRunConfig.model_validate(config.model_dump(mode='json') if isinstance(config,ToolRunConfig) else config)!=row.config:raise ValueError('external frozen config differs')
    records=[LLMRecord.model_validate(r) for r in row.native_execution['llm_records']]
    client=RecordedCompletionClient(row.config.service_model,records)
    rebuilt=run(row.config,client)
    if rebuilt!=row or client.index!=len(records):raise ValueError('native execution or exact model request Replay differs')
    return {'verified':True,'model_calls':client.index,'domain':row.config.task.domain,'task_hash':row.config.task.task_hash,'evidence_kind':row.config.evidence_kind}
