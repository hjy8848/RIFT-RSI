"""Unconditioned, bounded native lifecycle simulation for dialogue world v2.

This is a generator, not a task solver or model policy. It samples legitimate
customer choices and existing operations without consulting a task, Designer,
Service, score, learned skill or error. Every committed transition is executed
through the real native runtime and verified by Oracle and Replay.
"""
import copy,importlib
from usami.randomness import StableRandom
from usami_harness.models import digest
from usami_benchmark.contracts import GeneratorConfig
from usami_dialogue.models import CustomerIntent,InteractionState
from usami_dialogue.engine import DialogueSession
from usami_dialogue.oracle import evaluate
from usami_dialogue.replay import replay

DOMAINS=('aftersales','airline','property','education')


def _fork(session):
    fork=copy.copy(session)
    fork._state=session.state;fork.trace=session.trace.model_copy(deep=True)
    fork.grants=copy.deepcopy(session.grants);fork.presented_offer=copy.deepcopy(session.presented_offer)
    return fork


def _steps(trace):
    result=[]
    for e in trace.events:
        if e.kind=='customer' and not e.payload.get('setup'):result.append({'kind':'customer','intent':e.payload['intent']})
        elif e.kind=='service':result.append({'kind':'service','payload':e.payload})
        elif e.kind=='business':
            op=next(x['payload']['operation'] for x in e.payload['trace']['events'] if x['kind']=='operation')
            result.append({'kind':'business','operation':op})
        elif e.kind=='external':result.append({'kind':'external','event_id':e.payload['event_id']})
    return result


def generate_world(domain,seed=42,config=None,*,diagnostics=None):
    if domain not in DOMAINS:raise ValueError('unknown dialogue lifecycle domain')
    cfg=GeneratorConfig.model_validate(config or {});adapter=importlib.import_module('usami_'+domain+'.adapter')
    version=domain+'-lifecycle-v2';rng=StableRandom(digest({'version':version,'seed':seed,'config':cfg.model_dump(mode='json')}))
    world=adapter.generate_world(seed,cfg.model_dump(mode='json'));base_hash=world.state_hash
    remaining=cfg.lifecycle_steps;histories=[];exclusions=[]
    ids=sorted(world.profiles);selected=rng.sample(ids,min(len(ids),3))
    for pid in selected:
        if remaining<=0:break
        world.interaction=InteractionState()
        session=DialogueSession(adapter,world,pid,max_operations=max(1,cfg.lifecycle_steps))
        private=session.private
        preferences={key:rng.choice(values) for key,values in sorted(private.preference_space.items())}
        session.customer(CustomerIntent(message='My genuine business request during the earlier lifecycle.',disclosures=private.facts,preferences=preferences))
        allowance=rng.randint(1,remaining);used=0
        for _ in range(allowance):
            state=session.state;candidates=adapter.candidate_intents(state,private)
            # Fixed finite rejection-sampling budget; no task-conditioned search.
            proposals=[('business',raw) for raw in rng.sample(candidates,min(12,len(candidates)))]
            events=[eid for eid in sorted(state.external_events) if adapter.external_ready(state,eid)]
            proposals += [('external',eid) for eid in rng.sample(events,min(2,len(events)))]
            proposals=rng.sample(proposals,len(proposals));chosen=None
            for kind,item in proposals:
                draft=_fork(session)
                if kind=='external':
                    try:expected=adapter.external_reference(state,item)
                    except ValueError:continue
                    errors=adapter.invariants(expected)
                    if errors:
                        exclusions.append({'kind':'native_external_invariant_conflict','event_id':item,'before_hash':state.state_hash,'invariants':errors});continue
                    draft.external(item);chosen=draft;break
                op=adapter.Operation.model_validate(item);offer=adapter.offer_for(draft.state,item)
                try:
                    if offer is not None:
                        draft.reply('Actual terms of the earlier service offer.',item)
                        draft.customer(CustomerIntent(message='I confirm the actual offer within my real constraints.',confirm_operation=item))
                    else:draft.customer(CustomerIntent(message='Please perform this exact earlier business action.',requested_operations=[item]))
                except ValueError:continue
                before=draft.state;decision=adapter.decide(before,private.identity,op)
                if not decision.allowed or decision.expected_state==before:continue
                errors=adapter.invariants(decision.expected_state)
                if errors:
                    exclusions.append({'kind':'native_reference_invariant_conflict','operation':item,'before_hash':before.state_hash,'invariants':errors});continue
                actual=draft.act(op)
                if not actual['success']:raise ValueError('lifecycle actual mutation disagrees with allowed Reference transition')
                chosen=draft;break
            if chosen is None:break
            session=chosen;used+=1;remaining-=1
        trace=session.finish();report=evaluate(adapter,trace)
        if report['status']!='pass' or not replay(adapter,trace)['verified']:raise ValueError('native lifecycle Oracle/Replay failure')
        world=adapter.World.model_validate(trace.final_world)
        histories.append({'profile_id':pid,'initial_hash':digest(trace.initial_world),'final_hash':world.state_hash,'trace_hash':trace.trace_hash,
            'max_operations':trace.max_operations,'actual_business_operations':len(report['operations']),
            'external_events':sum(e.kind=='external' for e in trace.events),'steps':_steps(trace),'oracle_status':report['status'],'replay_verified':True})
    # A newly generated episode starts with no prior customer grants or consent.
    # Private initial-expiry facts describe this new epoch, not the historical one.
    epoch_before=world.state_hash;world.interaction=InteractionState()
    if domain=='education':
        for pid,p in list(world.profiles.items()):
            facts={**p.facts,'initial_expires_at':world.enrollments[p.facts['enrollment_id']].expires_at.isoformat()}
            world.profiles[pid]=p.model_copy(update={'facts':facts,'private_hash':''}).checked()
    if adapter.invariants(world):raise ValueError('invalid final lifecycle world')
    summary={'kind':'native_dialogue_lifecycle_v2','domain':domain,'generator_version':version,'base_world_hash':base_hash,
        'transition_budget':cfg.lifecycle_steps,'actual_transitions':cfg.lifecycle_steps-remaining,'histories':histories,
        'excluded_native_conflicts':exclusions,'epoch_before_hash':epoch_before,'final_world_hash':world.state_hash,
        'epoch_policy':'Clear prior interaction authority; education initial-expiry fact describes generated episode epoch. No business fields are assigned manually.',
        'model_experiments':'NOT RUN'}
    if diagnostics is not None:diagnostics.append(summary)
    return world


def replay_history(domain,seed,config,summary):
    """Re-execute recorded historical customer/offer/business/external events.

    RNG choices are not trusted as execution proof: each compact journal is run
    through Session, independent Oracle and exact native Replay again.
    """
    if domain not in DOMAINS:raise ValueError('unknown lifecycle domain')
    cfg=GeneratorConfig.model_validate(config or {});adapter=importlib.import_module('usami_'+domain+'.adapter')
    if summary.get('kind')!='native_dialogue_lifecycle_v2' or summary.get('domain')!=domain or summary.get('generator_version')!=domain+'-lifecycle-v2' or summary.get('transition_budget')!=cfg.lifecycle_steps or summary.get('model_experiments')!='NOT RUN':raise ValueError('lifecycle provenance differs')
    world=adapter.generate_world(seed,cfg.model_dump(mode='json'))
    if world.state_hash!=summary['base_world_hash']:raise ValueError('historical base generation differs')
    total=0
    for h in summary['histories']:
        world.interaction=InteractionState()
        if world.state_hash!=h['initial_hash']:raise ValueError('historical initial world differs')
        session=DialogueSession(adapter,world,h['profile_id'],max_operations=h['max_operations'])
        for step in h['steps']:
            if step['kind']=='customer':session.customer(CustomerIntent.model_validate(step['intent']))
            elif step['kind']=='service':
                p=step['payload'];session.reply(p['message'],p['offered_operation'],refused_operation=p['refused_operation'],refusal_code=p['refusal_code'])
                if session.trace.events[-1].payload!=p:raise ValueError('historical offer differs from actual quote')
            elif step['kind']=='business':
                if not session.act(step['operation'])['success']:raise ValueError('historical business action no longer succeeds')
            elif step['kind']=='external':session.external(step['event_id'])
            else:raise ValueError('unknown historical event')
        trace=session.finish();oracle=evaluate(adapter,trace);business=len(oracle['operations']);external=sum(e.kind=='external' for e in trace.events)
        if trace.trace_hash!=h['trace_hash'] or digest(trace.final_world)!=h['final_hash'] or business!=h['actual_business_operations'] or external!=h['external_events'] or oracle['status']!='pass' or not replay(adapter,trace)['verified']:raise ValueError('historical trace/Oracle/Replay differs')
        world=adapter.World.model_validate(trace.final_world);total+=business+external
    if total!=summary['actual_transitions'] or total>cfg.lifecycle_steps or world.state_hash!=summary['epoch_before_hash']:raise ValueError('historical budget or epoch differs')
    world.interaction=InteractionState()
    if domain=='education':
        for pid,p in list(world.profiles.items()):
            world.profiles[pid]=p.model_copy(update={'facts':{**p.facts,'initial_expires_at':world.enrollments[p.facts['enrollment_id']].expires_at.isoformat()},'private_hash':''}).checked()
    if world.state_hash!=summary['final_world_hash'] or adapter.invariants(world):raise ValueError('generated lifecycle epoch differs')
    return world
