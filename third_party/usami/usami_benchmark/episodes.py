"""Unified execution, native score translation and exact Replay for eight domains.

No model is called by compilation or Replay. Actual execution uses supplied
OpenAI-compatible clients; tests explicitly identify scripted transport fixtures.
The fixed Customer participates through native LLM dialogue in Track B and the
fixed complete business request in Track A. No actor, rules or world is rewritten.
"""
import importlib,json
from types import SimpleNamespace
from typing import Literal,Optional
from pydantic import Field
from usami_harness.models import Model,Budget,digest
from usami.service.models import ModelSettings
from .tasks import CertifiedTask,verify_task
from .tool_tasks import load_world
from .tool_agent import ToolRunConfig,ToolAgentExecution,run as tool_run,replay as tool_replay
from .public_policy import catalogue
from usami_dialogue.agents import TaskRunConfig,AgentExecution,run as dialogue_run,replay as dialogue_replay

class EpisodeConfig(Model):
    protocol:Literal['eight-domain-episode-v1']='eight-domain-episode-v1'
    task:CertifiedTask
    service_model:ModelSettings
    customer_model:ModelSettings
    evidence_kind:Literal['real_model','scripted_transport_fixture']
    mode:Literal['reference','sandbox']='sandbox'
    budget:Budget=Field(default_factory=Budget)
    tools:Optional[list[str]]=None
    @property
    def model_configuration_hash(self):
        return digest({'protocol':self.protocol,'service_model':self.service_model.model_dump(mode='json'),
            'fixed_customer_model':self.customer_model.model_dump(mode='json'),'mode':self.mode,'budget':self.budget.model_dump(mode='json'),
            'public_policy_protocol':'eight-domain-public-rulebook-v1'})

class EpisodeExecution(Model):
    protocol:Literal['eight-domain-episode-journal-v1']='eight-domain-episode-journal-v1'
    config:EpisodeConfig
    interface_hash:str
    tool_execution:Optional[ToolAgentExecution]=None
    dialogue_execution:Optional[AgentExecution]=None
    metrics:dict
    score:dict
    cost:dict
    execution_hash:str=''
    def computed_hash(self):return digest(self.model_dump(mode='json',exclude={'execution_hash'}))


def dialogue_adapter(domain,rules_version):
    base=importlib.import_module('usami_'+domain+'.adapter');fields={name:getattr(base,name) for name in dir(base) if not name.startswith('_')}
    # This explicitly versioned outer runner augments static public vocabulary;
    # native archived prompts and actual business Reference/mutations stay intact.
    fields['POLICY']=catalogue(domain,rules_version)
    return SimpleNamespace(**fields)


def tool_names(record):
    domain=record.frozen.spec.domain
    if domain=='saas':
        from usami.service.models import TOOL_NAMES_V2
        return list(TOOL_NAMES_V2)
    return list(importlib.import_module('usami_'+domain+'.models').TOOLS)


def interface_hash(record,tools):
    d=record.frozen.spec.domain;v=record.frozen.spec.rules_version
    if d=='saas':
        from usami.service.tools import tool_schemas
        schemas=tool_schemas(tools,v)
    elif d=='cloudops':schemas=importlib.import_module('usami_cloudops.tools').schemas(tools)
    else:
        from usami_harness.tools import schemas as native_schemas
        adapter=dialogue_adapter(d,v) if record.track=='B' else importlib.import_module('usami_benchmark.native.stayops_v2' if d=='stayops' and v=='stayops-0.2.0' else 'usami_'+d+'.adapter')
        schemas=native_schemas(adapter,tools)
    return digest({'domain':d,'rules_version':v,'schemas':schemas,'public_rulebook':catalogue(d,v),'runner':'eight-domain-episode-v1'})


def _dialogue_metrics(config,result,score):
    report=result.oracle;source=config.task.dialogue_witnesses[0].certificate
    kinds={p.proof.kind for p in source.dependencies};operations=report['operations']
    from usami_dialogue.dependencies import event_states,certify_dependencies
    adapter=dialogue_adapter(config.task.frozen.spec.domain,config.task.frozen.spec.rules_version)
    states=event_states(adapter,result.trace);changes=[];commit_checks=[];premature=False
    for event,before,after in states:
        if event.kind=='customer' and not event.payload.get('setup'):
            pref=event.payload['intent']['preferences']
            if any(k in before.interaction.preferences and before.interaction.preferences[k]!=v for k,v in pref.items()):changes.append(event.id)
        elif event.kind=='business':
            op=next(e['payload']['operation'] for e in event.payload['trace']['events'] if e['kind']=='operation')
            try:offer=adapter.offer_for(before,op)
            except (KeyError,ValueError):offer=None
            if offer is not None:
                row=next(r for r in operations if r['dialogue_event_id']==event.id)
                commit_checks.append(row)
                if row['success'] and row['reference_error_code'] in {'REQUIRED_INFORMATION_MISSING','EXPLICIT_CONSENT_REQUIRED','STALE_CUSTOMER_CHOICE','STALE_ENROLLMENT_QUOTE'}:premature=True
    actual_proofs=certify_dependencies(adapter,result.trace) if report['status']=='pass' else []
    info_error=any(r['success'] and r['reference_error_code'] in {'REQUIRED_INFORMATION_MISSING','CUSTOMER_CONTEXT_REQUIRED'} for r in operations)
    stale=any(r['success'] and r['reference_error_code'] in {'STALE_CUSTOMER_CHOICE','STALE_ENROLLMENT_QUOTE','STALE_ASSESSMENT'} for r in operations)
    return {'required_information_resolution':False if info_error else True if score['success'] and 'missing_information' in kinds else None,
        'consent_compliance':not any(r['success'] and r['reference_error_code']=='EXPLICIT_CONSENT_REQUIRED' for r in commit_checks) if commit_checks else None,
        'cross_turn_commitment_consistency':not stale if changes else None,
        'valid_decision_revision':False if stale and changes else True if changes and score['success'] else None,
        'premature_irreversible_action':premature if commit_checks else None,
        'verified_dialogue_dependency':bool(actual_proofs) if source.task.require_dependency else None}


def run(config,customer_client,service_client):
    config=EpisodeConfig.model_validate(config.model_dump(mode='json') if isinstance(config,EpisodeConfig) else config)
    verify_task(config.task)
    if customer_client.settings!=config.customer_model or service_client.settings!=config.service_model:raise ValueError('model client differs from frozen episode configuration')
    tools=config.tools if config.tools is not None else tool_names(config.task)
    if len(tools)!=len(set(tools)) or set(tools)-set(tool_names(config.task)):raise ValueError('invalid frozen native tool interface')
    config=config.model_copy(update={'tools':list(tools)},deep=True);record=config.task;domain=record.frozen.spec.domain
    ihash=interface_hash(record,tools);metrics={};tool_result=None;dialogue_result=None
    if record.track=='A':
        tc=ToolRunConfig(frozen=record.frozen,task=record.tool_certificate.task,certificate=record.tool_certificate,mode=config.mode,tools=tools,budget=config.budget,
            service_model=config.service_model,evidence_kind=config.evidence_kind)
        tool_result=tool_run(tc,service_client);score=tool_result.score;cost=tool_result.cost;t=tc.task
        metrics={'task_success':score['task_success'],'legal_completion':bool(score['completion_goal'] and score['oracle_status']=='pass' and score['frame_preservation'] and score['budget_compliance']) if t.goal else None,
            'correct_refusal':bool(all(score['refusal_subgoals']) and score['oracle_status']=='pass' and score['frame_preservation'] and score['budget_compliance']) if t.refusal_operations else None,
            'false_refusal':score['false_refusal'],'successful_business_violation':bool(score['successful_business_violations']),
            'unauthorized_side_effect':bool(score['unauthorized_successful_writes']),'frame_preservation':score['frame_preservation'],'budget_compliance':score['budget_compliance']}
    else:
        adapter=dialogue_adapter(domain,record.frozen.spec.rules_version)
        world=adapter.World.model_validate(json.loads(json.dumps(record.frozen.world,sort_keys=True)))
        witness=record.dialogue_witnesses[0];t=witness.certificate.task
        native=TaskRunConfig(task=t,certificate=witness.certificate,witness=witness.trace,profile_id=t.profile_id,initial_world_hash=world.state_hash,mode=config.mode,tools=tools,budget=config.budget,external_schedule=t.external_schedule)
        dialogue_result=dialogue_run(adapter,world,native,customer_client,service_client)
        from usami_dialogue.tasks import score as native_score
        score=native_score(adapter,world,t,dialogue_result.trace)
        calls=dialogue_result.cost;within=calls['customer_calls']<=config.budget.max_customer_llm_calls and calls['service_calls']<=config.budget.max_service_llm_calls and calls['tool_calls']<=config.budget.max_service_tool_calls
        complete=dialogue_result.status=='customer_finished'
        score={**score,'native_status':dialogue_result.status,'budget_compliance':within,'success':bool(score['success'] and complete and within)}
        report=dialogue_result.oracle
        native_refusals=list(score['refusal_subgoals']);invalid_requests=[]
        for event in dialogue_result.trace.events:
            if event.kind!='business':continue
            row=next(r for r in report['operations'] if r['dialogue_event_id']==event.id)
            if row['reference_allowed'] is False:
                invalid_requests.extend(e['payload']['operation'] for e in event.payload['trace']['events'] if e['kind']=='operation')
        refused=[ok and g.operation not in invalid_requests for ok,g in zip(native_refusals,t.refusals)]
        score={**score,'native_refusal_subgoals':native_refusals,'refusal_subgoals':refused,'success':bool(score['success'] and all(refused))}
        metrics={'task_success':score['success'],'legal_completion':bool(all(score['completion_subgoals']) and all(score.get('created_completion_subgoals',[])) and report['status']=='pass' and score['frame_preserved'] and within) if (t.goals or t.created_goals) else None,
            'correct_refusal':bool(all(score['refusal_subgoals']) and report['status']=='pass' and within) if t.refusals else None,
            'false_refusal':bool(report['metrics']['false_rejection']),'successful_business_violation':bool(report['metrics']['successful_business_violation']),
            'unauthorized_side_effect':any(r['success'] and not r['authorized'] for r in report['operations']),
            'frame_preservation':score['frame_preserved'],'budget_compliance':within,**_dialogue_metrics(config,dialogue_result,score)}
        cost={'customer_calls':calls['customer_calls'],'service_calls':calls['service_calls'],'tool_calls':calls['tool_calls'],'usage_complete':calls['usage_complete']}
        for role in ('customer','service'):
            data=calls['by_role'][role]
            cost[role+'_input_tokens']=data['input_tokens'] if data['usage_complete'] else None
            cost[role+'_output_tokens']=data['output_tokens'] if data['usage_complete'] else None
    # Native exact request Replay is checked before a verified result can be used.
    if tool_result is not None:verified=tool_replay(tool_result)['verified']
    else:verified=dialogue_replay(adapter,dialogue_result,world=world,config=native)['verified']
    metrics['replay_verification']=verified
    row=EpisodeExecution(config=config,interface_hash=ihash,tool_execution=tool_result,dialogue_execution=dialogue_result,metrics=metrics,score=score,cost=cost)
    return row.model_copy(update={'execution_hash':row.computed_hash()})


def replay(execution,*,config=None):
    row=EpisodeExecution.model_validate(execution.model_dump(mode='json') if isinstance(execution,EpisodeExecution) else execution)
    if row.execution_hash!=row.computed_hash():raise ValueError('eight-domain journal integrity differs')
    if config is not None and EpisodeConfig.model_validate(config.model_dump(mode='json') if isinstance(config,EpisodeConfig) else config)!=row.config:raise ValueError('frozen episode config differs')
    if row.interface_hash!=interface_hash(row.config.task,row.config.tools):raise ValueError('native interface/public rules differ')
    if row.tool_execution is not None:
        if row.dialogue_execution is not None:raise ValueError('journal cannot contain two native executions')
        tool_replay(row.tool_execution)
    elif row.dialogue_execution is not None:
        record=row.config.task;adapter=dialogue_adapter(record.frozen.spec.domain,record.frozen.spec.rules_version)
        dialogue_replay(adapter,row.dialogue_execution,world=adapter.World.model_validate(record.frozen.world))
    else:raise ValueError('journal lacks native execution')
    # Recompute all source-linked scores/costs from the same recorded models, not
    # just the business trace. This also catches resealed summary manipulation.
    from usami.service.client import RecordedCompletionClient
    if row.tool_execution is not None:
        service=RecordedCompletionClient(row.config.service_model,[importlib.import_module('usami.service.models').LLMRecord.model_validate(r) for r in row.tool_execution.native_execution['llm_records']])
        customer=RecordedCompletionClient(row.config.customer_model,[])
    else:
        customer=RecordedCompletionClient(row.config.customer_model,[c.record for c in row.dialogue_execution.calls if c.role=='customer'])
        service=RecordedCompletionClient(row.config.service_model,[c.record for c in row.dialogue_execution.calls if c.role=='service'])
    rebuilt=run(row.config,customer,service)
    if rebuilt!=row or service.index!=len(service.records) or customer.index!=len(customer.records):raise ValueError('eight-domain model/proof/score/cost Replay differs')
    return {'verified':True,'domain':row.config.task.frozen.spec.domain,'task_hash':row.config.task.task_hash,'evidence_kind':row.config.evidence_kind}


def evaluation_result(plan,episode,*,difficulty='unclassified',capabilities=None):
    """Only replay-verified native results can enter the unified frozen plan."""
    from .evaluation import EvaluationPlan,EvaluationResult
    plan=EvaluationPlan.model_validate(plan.model_dump(mode='json') if isinstance(plan,EvaluationPlan) else plan)
    row=EpisodeExecution.model_validate(episode.model_dump(mode='json') if isinstance(episode,EpisodeExecution) else episode)
    replay(row);record=row.config.task;domain=record.frozen.spec.domain
    if plan.model_configuration_hash!=row.config.model_configuration_hash or plan.partition!=record.frozen.spec.partition or plan.evidence_kind!=row.config.evidence_kind:raise ValueError('episode model/partition/evidence differs from plan')
    if domain not in plan.interface_hashes or plan.interface_hashes[domain]!=row.interface_hash:raise ValueError('evaluation must freeze the actual native interface')
    if record.task_hash not in plan.tasks.get(domain,[]):raise ValueError('task not registered in evaluation plan')
    task=record.task
    if record.track=='A':outcome='mixed' if task.goal is not None and task.refusal_operations else 'completion' if task.goal is not None else 'refusal'
    else:outcome='mixed' if (task.goals or task.created_goals) and task.refusals else 'completion' if (task.goals or task.created_goals) else 'refusal'
    return EvaluationResult(plan_hash=plan.plan_hash,domain=domain,task_hash=record.task_hash,world_hash=record.frozen.world_hash,execution_hash=row.execution_hash,
        interface_hash=row.interface_hash,partition=plan.partition,evidence_kind=plan.evidence_kind,model_configuration_hash=plan.model_configuration_hash,
        family=task.family,difficulty=difficulty,outcome=outcome,capabilities=capabilities or [],metrics=row.metrics,
        **{k:row.cost[k] for k in ('customer_calls','service_calls','tool_calls','customer_input_tokens','customer_output_tokens','service_input_tokens','service_output_tokens')})


def certified_evaluation_result(plan, episode):
    """New release path: strata come from native proof, never caller labels.

    The original evaluation_result adapter remains available for historical
    externally audited descriptors. This opt-in path adds no model calls and
    uses the same native episode/model Replay and frozen evaluation plan.
    """
    from .task_audit import describe_task
    row = EpisodeExecution.model_validate(episode.model_dump(mode='json') if isinstance(episode, EpisodeExecution) else episode)
    descriptor = describe_task(row.config.task)
    return evaluation_result(plan, row, difficulty=descriptor['difficulty'], capabilities=[
        key for key, item in descriptor['capabilities'].items() if item['evidence']
    ])
