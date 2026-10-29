"""Independent process audit. Business correctness uses Reference, not mutations."""
from usami_harness.models import Request,digest
from usami_harness.engine import Trace
from usami_harness.oracle import evaluate as business_oracle
from .models import CustomerIntent
from .process import initialize,customer_event


def evaluate(adapter,trace,*,initial_world=None,private=None):
    errors=[];business=[];decisions=[];disclosures=[];consents=[];previous='';operations=0;turn=0;grants=[];presented_offer=None
    def fail(code,index=None):errors.append({'code':code,'event_id':index})
    if trace.protocol!='business-dialogue-v1' or trace.domain!=adapter.name or trace.trace_hash!=trace.computed_hash():fail('DIALOGUE_TRACE_INTEGRITY')
    world=adapter.World.model_validate(trace.initial_world)
    if initial_world is not None and initial_world!=world:fail('REGISTERED_WORLD_MISMATCH')
    expected_private=adapter.private_state(world,trace.profile_id)
    adapter.validate_private(world,expected_private)
    if trace.scoped_profile_ids:
        from .scope import load_profiles
        profile_rows=load_profiles(adapter,world,trace.profile_id,trace.scoped_profile_ids)
    else:profile_rows=[(trace.profile_id,expected_private)]
    profiles=dict(profile_rows);active_profile_id=trace.profile_id
    if trace.private!=expected_private or (private is not None and private!=expected_private):fail('PRIVATE_SOURCE_MISMATCH')
    if world.version!=trace.rules_version or adapter.invariants(world):fail('INITIAL_DOMAIN_INVALID')
    for index,event in enumerate(trace.events):
        if event.id!='dialogue_'+str(index) or event.previous_hash!=previous or event.event_hash!=event.computed_hash() or event.before_hash!=world.state_hash:fail('DIALOGUE_CHAIN_MISMATCH',event.id)
        previous=event.event_hash;p=event.payload
        try:
            if event.kind=='customer':
                if p.get('setup'):
                    if index!=0 or p!={'setup':True,'private_hash':expected_private.private_hash}:raise ValueError('invalid setup')
                    world.interaction=initialize(expected_private)
                else:
                    intent=CustomerIntent.model_validate(p['intent']);turn+=1;tid='turn_'+str(turn)
                    if p['turn_id']!=tid:raise ValueError('turn identity differs')
                    offer=adapter.offer_for(world,intent.confirm_operation)
                    if intent.confirm_operation is not None and (offer is None or offer!=presented_offer):raise ValueError('offer was not actually presented')
                    if trace.scoped_profile_ids:
                        from .scope import profile_for_intent
                        selected_id,selected_private=profile_for_intent(adapter,world,profile_rows,intent)
                        if p.get('profile_id')!=selected_id:raise ValueError('Customer profile selection differs from frozen scope')
                    else:
                        selected_id,selected_private=trace.profile_id,expected_private
                        if 'profile_id' in p:raise ValueError('legacy customer event declares an unregistered profile')
                    active_profile_id=selected_id
                    normalized=adapter.normalize_intent(selected_private,intent)
                    context=customer_event(world.interaction,selected_private,normalized,turn_id=tid,world_hash=world.state_hash,offer=offer,consent_check=adapter.consent_allowed)
                    declarations=list(intent.requested_operations)+([intent.confirm_operation] if intent.confirm_operation else [])
                    grants=[];seen=set()
                    for raw in declarations:
                        op=adapter.Operation.model_validate(raw)
                        if not adapter.intent_allowed(world,selected_private,op.model_dump(mode='json')):raise ValueError('customer intent outside scope')
                        if digest(op) not in seen:grants.append(op.model_dump(mode='json'));seen.add(digest(op))
                    if grants!=p['grants']:raise ValueError('trusted grants differ')
                    disclosures.append({'event_id':event.id,'keys':sorted(intent.disclosures)})
                    if intent.confirm_operation:consents.append({'event_id':event.id,'operation':intent.confirm_operation,'revision':context.revision})
                    world.interaction=context;presented_offer=None
            elif event.kind=='service':
                if set(p)!={'message','offered_operation','offer','refused_operation','refusal_code'} or not isinstance(p['message'],str):raise ValueError('invalid reply event')
                offered=adapter.offer_for(world,p['offered_operation'])
                if offered!=p['offer'] or (p['offered_operation'] is not None and offered is None):raise ValueError('Service quote disclosure differs')
                if (p['refused_operation'] is None)!=(p['refusal_code'] is None) or (p['refused_operation'] is not None and p['offered_operation'] is not None):raise ValueError('invalid refusal declaration')
                if p['refused_operation'] is not None:
                    op=adapter.Operation.model_validate(p['refused_operation']);d=adapter.decide(world,expected_private.identity,op)
                    attributable=op.model_dump(mode='json') in grants
                    kind='unattributed_refusal' if not attributable else 'false_rejection' if d.allowed else 'correct_refusal' if p['refusal_code']==d.error_code else 'incorrect_refusal_reason'
                    decisions.append({'event_id':event.id,'operation':op.model_dump(mode='json'),'classification':kind,'reference_allowed':d.allowed,'reference_code':d.error_code,'claimed_code':p['refusal_code'],'before_hash':world.state_hash})
                presented_offer=offered;grants=[]
            elif event.kind=='business':
                native=Trace.model_validate(p['trace'])
                if native.initial_state!=world.model_dump(mode='json') or p['turn_id']!='turn_'+str(turn):raise ValueError('business prestate/turn differs')
                if trace.scoped_profile_ids:
                    if p.get('profile_id')!=active_profile_id:raise ValueError('business action detached from the active scoped Customer turn')
                elif 'profile_id' in p:raise ValueError('legacy business event declares an unregistered profile')
                native_ops=[e for e in native.events if e.kind=='operation']
                if len(native_ops)!=1:raise ValueError('one event must contain exactly one business operation')
                op=adapter.Operation.model_validate(native_ops[0].payload['operation']);canonical=op.model_dump(mode='json')
                authorized=op.is_query
                if not op.is_query and canonical in grants:grants.remove(canonical);authorized=True
                expected=Request(id='action_'+str(operations),identity=expected_private.identity,operation=canonical,granted=authorized)
                trusted={k:Request.model_validate(v) for k,v in p['requests'].items()}
                if trusted!={expected.id:expected} or native_ops[0].payload['actor']!=expected_private.identity:raise ValueError('independent Customer authority differs')
                report=business_oracle(adapter,native,trusted)
                if not report['trace_valid']:fail('NATIVE_TRACE_INTEGRITY',event.id)
                business.extend({**r,'dialogue_event_id':event.id} for r in report['operations'])
                world=adapter.World.model_validate(native.final_state);operations+=1
            elif event.kind=='external':
                # Native external Reference is deliberately separate from actual external_apply.
                expected=adapter.external_reference(world,p['event_id'])
                if expected.model_dump(mode='json')!=p['after']:raise ValueError('external event differs from frozen simulator')
                world=expected
            else:raise ValueError('unknown event')
            if event.after_hash!=world.state_hash:fail('DIALOGUE_AFTER_HASH_MISMATCH',event.id)
        except (KeyError,ValueError,TypeError,AttributeError) as exc:fail('DIALOGUE_EVENT_INVALID:'+str(exc),event.id)
    if not trace.events or not trace.events[0].payload.get('setup'):fail('MISSING_CUSTOMER_SETUP')
    if operations>trace.max_operations:fail('OPERATION_BUDGET_EXCEEDED')
    if trace.final_world!=world.model_dump(mode='json'):fail('DIALOGUE_FINAL_WORLD_MISMATCH')
    final_invariants=adapter.invariants(world)
    metrics={kind:sum(r['classification']==kind for r in business) for kind in ('successful_business_violation','authorization_only','false_rejection','legal_execution','rejected_invalid_attempt','unscored')}
    metrics['false_rejection']+=sum(r['classification']=='false_rejection' for r in decisions)
    metrics['correct_refusal']=sum(r['classification']=='correct_refusal' for r in decisions)
    return {'protocol':'business-dialogue-oracle-v1','status':'fail' if errors or final_invariants or any(r['classification'] in {'false_rejection','incorrect_refusal_reason'} for r in decisions) or any(r['operation_correct'] is False for r in business) else 'unknown' if any(r['operation_correct'] is None for r in business) else 'pass',
        'trace_valid':not errors,'errors':errors,'final_invariants':final_invariants,'operations':business,'service_decisions':decisions,'disclosures':disclosures,'consents':consents,'metrics':metrics,'final_world_hash':world.state_hash}
