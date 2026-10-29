"""Executable, trace-bound interaction counterfactuals, never rewards for turn count.

Counterfactual worlds are evaluator-only copies. They cannot enter a session or
World Pool. Only the declared process intervention is permitted, and both native
Reference decisions are recalculated whenever the certificate is verified.
"""
from typing import Literal
from pydantic import Field
from usami_harness.models import Model,digest
from .models import DependencyProof
from .process import initialize,customer_event
from .replay import replay


class DependencyCertificate(Model):
    protocol:Literal['dialogue-dependency-v1']='dialogue-dependency-v1'
    trace_hash:str
    event_id:str
    proof:DependencyProof
    intervention:dict
    control_world:dict
    changed_world:dict
    certificate_hash:str=''
    def computed_hash(self):return digest(self.model_dump(mode='json',exclude={'certificate_hash'}))


def event_states(adapter,trace):
    """Recover actual pre/post states after checking Engine/process/external replay."""
    replay(adapter,trace)
    world=adapter.World.model_validate(trace.initial_world)
    profiles={pid:adapter.private_state(world,pid) for pid in (trace.scoped_profile_ids or [trace.profile_id])}
    rows=[]
    for event in trace.events:
        before=world.clone();p=event.payload
        if event.kind=='customer':
            if p.get('setup'):world.interaction=initialize(trace.private)
            else:
                from .models import CustomerIntent
                intent=CustomerIntent.model_validate(p['intent'])
                profile_id=p.get('profile_id',trace.profile_id)
                if profile_id not in profiles:raise ValueError('Customer event references an unselected profile')
                private=profiles[profile_id]
                world.interaction=customer_event(world.interaction,private,adapter.normalize_intent(private,intent),
                    turn_id=p['turn_id'],world_hash=world.state_hash,
                    offer=adapter.offer_for(world,intent.confirm_operation),consent_check=adapter.consent_allowed)
        elif event.kind=='business':world=adapter.World.model_validate(p['trace']['final_state'])
        elif event.kind=='external':world=adapter.World.model_validate(p['after'])
        rows.append((event,before,world.clone()))
    return rows


def intervene(adapter,private,control,kind,intervention):
    """A closed list of process interventions; no business state editing allowed."""
    changed=control.clone();ctx=changed.interaction
    if kind=='missing_information':
        if set(intervention)!={'fact'}:raise ValueError('invalid missing-information intervention')
        key=intervention['fact']
        if key not in ctx.known_facts or ctx.known_facts[key]!=private.facts.get(key):raise ValueError('fact was not actually disclosed')
        del ctx.known_facts[key]
    elif kind=='explicit_consent':
        if intervention:raise ValueError('invalid consent intervention')
        ctx.consents=[]
    elif kind=='preference_resolution':
        if set(intervention)!={'preference','value'}:raise ValueError('invalid preference intervention')
        key=intervention['preference'];value=intervention['value']
        if value not in private.preference_space.get(key,[]) or value==ctx.preferences.get(key):raise ValueError('not a genuine permitted alternate preference')
        ctx.preferences[key]=value;ctx.revision+=1;ctx.consents=[];ctx.commitments={}
    else:raise ValueError('unsupported process intervention')
    return changed


def _certificate(adapter,trace,event,control,changed,operation,kind,intervention):
    op=adapter.Operation.model_validate(operation);a=adapter.decide(control,trace.private.identity,op);b=adapter.decide(changed,trace.private.identity,op)
    if a.allowed==b.allowed:return None
    proof=DependencyProof(kind=kind,operation=op.model_dump(mode='json'),control_hash=control.state_hash,changed_hash=changed.state_hash,
        expected_control_code=a.error_code,expected_changed_code=b.error_code,control_allowed=a.allowed,changed_allowed=b.allowed,
        reason='Independent native Reference decisions differ under the recorded, bounded intervention.').checked()
    row=DependencyCertificate(trace_hash=trace.trace_hash,event_id=event.id,proof=proof,intervention=intervention,
        control_world=control.model_dump(mode='json'),changed_world=changed.model_dump(mode='json'))
    row.certificate_hash=row.computed_hash()
    return row


def _commitment_candidates(event,before,after):
    """An actual later Customer revision cancels an actual earlier commitment."""
    if event.kind!='customer' or event.payload.get('setup'):return []
    if after.interaction.revision<=before.interaction.revision:return []
    return [(c.operation,{'customer_event_id':event.id,'prior_consent_turn_id':c.turn_id}) for c in before.interaction.consents]


def _partial_candidates(adapter,trace,states):
    # Pair a genuine correct refusal with an earlier real, legal state transition.
    # A mere guessed reason, blocked tool call or text claim is insufficient.
    from .oracle import evaluate
    report=evaluate(adapter,trace)
    if report['status']!='pass':return {}
    legal={r['dialogue_event_id'] for r in report['operations'] if r['classification']=='legal_execution'}
    positions={e.id:i for i,(e,_,_) in enumerate(states)};result={}
    for refusal in report['service_decisions']:
        if refusal['classification']!='correct_refusal':continue
        stop=positions[refusal['event_id']]
        for event,before,after in states[:stop]:
            if event.id not in legal or event.kind!='business':continue
            op=adapter.Operation.model_validate(refusal['operation'])
            a=adapter.decide(before,trace.private.identity,op);b=adapter.decide(after,trace.private.identity,op)
            if a.allowed and not b.allowed and b.error_code==refusal['reference_code']:
                result.setdefault(event.id,[]).append((refusal['operation'],{'refusal_event_id':refusal['event_id']}))
    return result


def certify_dependencies(adapter,trace):
    result=[];states=event_states(adapter,trace);partials=_partial_candidates(adapter,trace,states)
    for event,before,after in states:
        for operation,intervention in _commitment_candidates(event,before,after):
            row=_certificate(adapter,trace,event,before,after,operation,'cross_turn_commitment',intervention)
            if row:result.append(row)
        for operation,intervention in partials.get(event.id,[]):
            row=_certificate(adapter,trace,event,before,after,operation,'partial_completion',intervention)
            if row:result.append(row)
        if event.kind=='business':
            private=adapter.private_state(adapter.World.model_validate(trace.initial_world),event.payload.get('profile_id',trace.profile_id))
            ops=[e['payload']['operation'] for e in event.payload['trace']['events'] if e['kind']=='operation']
            op=adapter.Operation.model_validate(ops[0])
            if op.is_query or not adapter.decide(before,trace.private.identity,op).allowed:continue
            interventions=[('missing_information',{'fact':key}) for key,value in before.interaction.known_facts.items() if private.facts.get(key)==value]
            interventions+=[('explicit_consent',{})]
            interventions+=[('preference_resolution',{'preference':k,'value':v}) for k,values in private.preference_space.items() for v in values if v!=before.interaction.preferences.get(k)]
            for kind,intervention in interventions:
                changed=intervene(adapter,private,before,kind,intervention)
                row=_certificate(adapter,trace,event,before,changed,op,kind,intervention)
                if row:result.append(row)
        elif event.kind=='external':
            # The paired states are actual replayed states, not invented outcomes.
            candidates=[]
            for profile_id in (trace.scoped_profile_ids or [trace.profile_id]):
                private=adapter.private_state(adapter.World.model_validate(trace.initial_world),profile_id)
                candidates+=adapter.candidate_intents(before,private)+adapter.candidate_intents(after,private)
            seen=set()
            for op in candidates:
                if digest(op) in seen:continue
                seen.add(digest(op))
                row=_certificate(adapter,trace,event,before,after,op,'state_update',{'external_event_id':event.payload['event_id']})
                if row:result.append(row)
    return sorted(result,key=lambda row:row.certificate_hash)


def verify_dependency(adapter,trace,certificate):
    c=DependencyCertificate.model_validate(certificate.model_dump(mode='json') if isinstance(certificate,DependencyCertificate) else certificate)
    if c.certificate_hash!=c.computed_hash() or c.trace_hash!=trace.trace_hash:raise ValueError('dependency certificate integrity/source mismatch')
    states=event_states(adapter,trace);rows={e.id:(e,b,a) for e,b,a in states}
    if c.event_id not in rows:raise ValueError('dependency has no source event')
    event,control,actual_after=rows[c.event_id]
    if c.proof.kind=='cross_turn_commitment':
        if (c.proof.operation,c.intervention) not in _commitment_candidates(event,control,actual_after):raise ValueError('commitment dependency lacks an actual prior consent and later Customer revision')
        changed=actual_after
    elif c.proof.kind=='partial_completion':
        if (c.proof.operation,c.intervention) not in _partial_candidates(adapter,trace,states).get(event.id,[]):raise ValueError('partial completion lacks actual legal transition and later correct refusal')
        changed=actual_after
    elif c.proof.kind=='state_update':
        if event.kind!='external' or c.intervention!={'external_event_id':event.payload['event_id']}:raise ValueError('state update lacks actual external source')
        allowed=adapter.candidate_intents(control,trace.private)+adapter.candidate_intents(actual_after,trace.private)
        if c.proof.operation not in allowed:raise ValueError('external dependency operation outside frozen Customer scope')
        changed=actual_after
    else:
        if event.kind!='business':raise ValueError('process dependency lacks business source')
        operations=[e['payload']['operation'] for e in event.payload['trace']['events'] if e['kind']=='operation']
        if c.proof.operation!=operations[0]:raise ValueError('dependency operation differs from actual source action')
        private=adapter.private_state(adapter.World.model_validate(trace.initial_world),event.payload.get('profile_id',trace.profile_id))
        changed=intervene(adapter,private,control,c.proof.kind,c.intervention)
    if c.control_world!=control.model_dump(mode='json') or c.changed_world!=changed.model_dump(mode='json'):raise ValueError('counterfactual edits exceed the declared intervention')
    rebuilt=_certificate(adapter,trace,event,control,changed,c.proof.operation,c.proof.kind,c.intervention)
    if rebuilt is None or rebuilt!=c:raise ValueError('Reference dependency does not reproduce')
    return True
