"""Bounded private Reference search shared by native dialogue domains.

This is proof construction, never a production Customer or Service policy.
Search failure at a bound is not a claim of task impossibility.
"""
from collections import deque
from itertools import product
from .models import CustomerIntent
from .process import initialize,customer_event
from .engine import DialogueSession
from .tasks import checked_task,certify


def _context(adapter,world,private,intent):
    new=world.clone()
    new.interaction=customer_event(world.interaction,private,adapter.normalize_intent(private,intent),turn_id='certificate_search',world_hash=world.state_hash,
        offer=adapter.offer_for(world,intent.confirm_operation),consent_check=adapter.consent_allowed)
    return new


def search(adapter,world,task,preference_options,*,max_nodes=1024,max_depth=12,max_paths=2,refusal_code_options=None):
    task=checked_task(adapter,world,task);private=adapter.private_state(world,task.profile_id)
    from .tasks import frame_preserved
    codes=None
    if refusal_code_options is not None:
        if task.goals or len(task.refusals)!=1 or not refusal_code_options:raise ValueError('code discovery requires one pure native refusal probe')
        codes=set(refusal_code_options)
        for code in codes:
            candidate=task.model_copy(deep=True);candidate.refusals[0].code=code;candidate.task_hash=''
            checked_task(adapter,world,candidate)
    queue=deque();seen=set();paths=[];nodes=0;matched=[]
    for preferences in preference_options:
        state=world.clone();state.interaction=initialize(private)
        intent=CustomerIntent(message='I am providing my actual facts and choosing an acceptable option.',disclosures=private.facts,preferences=preferences)
        state=_context(adapter,state,private,intent)
        queue.append((state,[{'kind':'customer','intent':intent.model_dump(mode='json')}],0))
    while queue and nodes<max_nodes and len(paths)<max_paths:
        state,path,depth=queue.popleft();key=state.state_hash
        if key in seen:continue
        seen.add(key);nodes+=1
        decisions=[adapter.decide(state,private.identity,adapter.Operation.model_validate(g.operation)) for g in task.refusals]
        refusal_ready=all(not d.allowed and d.error_code in (codes if codes is not None else {g.code}) for g,d in zip(task.refusals,decisions))
        if all(g.satisfied(state) for g in task.goals) and refusal_ready and (codes is None or frame_preserved(world,state,task.frames)):
            paths.append(path);matched.append([d.error_code for d in decisions]);continue
        if depth>=max_depth:continue
        for raw in adapter.candidate_intents(state,private):
            op=adapter.Operation.model_validate(raw)
            offer=adapter.offer_for(state,raw)
            if offer is not None:
                intent=CustomerIntent(message='I confirm these actual terms.',confirm_operation=raw)
                try:prepared=_context(adapter,state,private,intent)
                except ValueError:continue
                steps=[{'kind':'service','operation':raw},{'kind':'customer','intent':intent.model_dump(mode='json')}]
            else:
                intent=CustomerIntent(message='Please perform this exact preparatory action.',requested_operations=[raw])
                prepared=_context(adapter,state,private,intent)
                steps=[{'kind':'customer','intent':intent.model_dump(mode='json')}]
            decision=adapter.decide(prepared,private.identity,op)
            if not decision.allowed or decision.expected_state==prepared:continue
            if codes is None or frame_preserved(world,decision.expected_state,task.frames):queue.append((decision.expected_state,path+steps+[{'kind':'business','operation':raw}],depth+1))
        for eid in task.external_schedule:
            if adapter.external_ready(state,eid):
                following=adapter.external_reference(state,eid)
                if not adapter.invariants(following) and (codes is None or frame_preserved(world,following,task.frames)):queue.append((following,path+[{'kind':'external','event_id':eid}],depth+1))
    result={'paths':paths,'nodes':nodes,'exhausted':not queue,'max_nodes':max_nodes,'max_depth':max_depth}
    if codes is not None:result['matched_refusal_codes']=matched
    return result


def _scoped_context(adapter,world,profiles,intent):
    from .scope import profile_for_intent
    profile_id,private=profile_for_intent(adapter,world,profiles,intent)
    new=world.clone()
    normalized=adapter.normalize_intent(private,intent)
    new.interaction=customer_event(world.interaction,private,normalized,turn_id='certificate_search',world_hash=world.state_hash,
        offer=adapter.offer_for(world,intent.confirm_operation),consent_check=adapter.consent_allowed)
    return new


def search_composed(adapter,world,task,*,max_nodes=1024,max_depth=12,max_paths=2,refusal_code_options=None):
    """Bounded native search for Designer-composed, multi-profile objectives.

    It enumerates only Customer facts/preferences and business intents exposed by
    the selected frozen profiles. Every terminal path is later executed through
    the ordinary Engine and independently certified by Oracle and Replay.
    """
    from .scope import load_profiles
    from .tasks import frame_preserved
    task=checked_task(adapter,world,task)
    if task.composition is None:raise ValueError('open composition search requires its versioned task scope')
    profiles=load_profiles(adapter,world,task.profile_id,task.composition.profile_ids)
    codes=None
    if refusal_code_options is not None:
        if len(refusal_code_options)!=len(task.refusals):raise ValueError('one Reference code set is required per refusal objective')
        codes=[set(x) for x in refusal_code_options]
        if any(not x for x in codes):raise ValueError('every refusal needs a nonempty native code vocabulary')
    elif task.refusals:codes=[{g.code} for g in task.refusals]
    else:codes=[]

    def refusal_decisions(state):
        """Evaluate each refusal in the same Customer context as witness replay.

        A composed path may end after operating on a different selected profile
        than the one named by a refusal objective. Inspecting ``state`` directly
        can then mistake missing facts from the previous profile for the reason
        the requested write should be refused. The executable witness explicitly
        submits the refused operation as a new Customer turn, so Reference
        search must construct that exact turn before comparing native codes.
        """
        scoped=state
        decisions=[]
        for goal in task.refusals:
            intent=CustomerIntent(message='Please perform this exact requested action.',
                requested_operations=[goal.operation])
            scoped=_scoped_context(adapter,scoped,profiles,intent)
            from .scope import profile_for_intent
            _,private=profile_for_intent(adapter,scoped,profiles,intent)
            decisions.append(adapter.decide(scoped,private.identity,
                adapter.Operation.model_validate(goal.operation)))
        return decisions
    preference_choices=[]
    for profile_id,private in profiles:
        requirements=[r for r in task.composition.preference_requirements if r.profile_id==profile_id]
        keys=[r.preference for r in requirements]
        values=[r.accepted_values for r in requirements]
        options=[]
        for selected in product(*values) if values else [()]:
            preferences=dict(private.initial_preferences);preferences.update(dict(zip(keys,selected)))
            options.append(preferences)
        preference_choices.append(options)
    combinations=1
    for options in preference_choices:combinations*=len(options)
    combination_limit=max(1,min(max_nodes,512))
    truncated=combinations>combination_limit
    starts=[];seen=set();paths=[];matched=[];nodes=0
    # Freeze real facts and one policy-permitted preference choice into each
    # proof path. The Cartesian choice space is explicit and bounded.
    for selected_preferences in list(product(*preference_choices))[:combination_limit]:
        initial=world.clone();initial.interaction=initialize(profiles[0][1]);initial_path=[]
        for (profile_id,private),preferences in zip(profiles,selected_preferences):
            intent=CustomerIntent(message='I am sharing my actual situation and preferences.',
                disclosures=private.facts,preferences=preferences)
            following=_scoped_context(adapter,initial,profiles,intent)
            if following.state_hash!=initial.state_hash:
                initial=following;initial_path.append({'kind':'customer','intent':intent.model_dump(mode='json')})
        selected_by_profile={pid:dict(prefs) for (pid,_),prefs in zip(profiles,selected_preferences)}
        starts.append((initial,initial_path,len(initial_path),selected_by_profile))
    stack=list(reversed(starts))
    order={
        'aftersales':{'open_case':0,'verify_evidence':1,'request_manager_review':2,'request_return':3,'schedule_return':3,'quote_resolution':4,'commit_resolution':5,'release_refund':6},
        'airline':{'verify_passenger':0,'quote_change':1,'quote_refund':1,'quote_baggage':1,'commit_quote':2},
        'property':{'open_ticket':0,'triage_ticket':1,'request_owner_review':2,'request_inspection':2,'quote_visit':3,'book_visit':4,'file_complaint':5,'close_ticket':6,'escalate_complaint':7},
        'education':{'open_request':0,'request_extension_review':1,'quote_refund':2,'quote_transfer':2,'quote_extension':2,'quote_lesson':2,'commit_quote':3,'cancel_lesson':4},
    }.get(adapter.name,{})
    while stack and nodes<max_nodes and len(paths)<max_paths:
        state,path,depth,selected_by_profile=stack.pop();key=state.state_hash
        if key in seen:continue
        seen.add(key);nodes+=1
        decisions=refusal_decisions(state)
        refusal_ready=all(not decision.allowed and decision.error_code in codes[index] for index,decision in enumerate(decisions))
        if (all(g.satisfied(state) for g in task.goals)
                and all(g.satisfied(world,state) for g in task.created_goals)
                and refusal_ready and frame_preserved(world,state,task.frames)):
            paths.append(path);matched.append([d.error_code for d in decisions]);continue
        if depth>=max_depth:continue
        # Expand only exact native writes attached to selected frozen profiles;
        # no designer-invented entity or business operation enters the search.
        children=[]
        for profile_id,private in profiles:
            for raw in adapter.candidate_intents(state,private):
                if not adapter.intent_allowed(state,private,raw):continue
                op=adapter.Operation.model_validate(raw);offer=adapter.offer_for(state,raw)
                if offer is not None:
                    intent=CustomerIntent(message='I confirm the actual terms that were presented.',confirm_operation=raw)
                    try:prepared=_scoped_context(adapter,state,profiles,intent)
                    except (KeyError,TypeError,ValueError):continue
                    steps=[{'kind':'service','operation':raw},{'kind':'customer','intent':intent.model_dump(mode='json')}]
                else:
                    intent=CustomerIntent(message='Please perform this exact action for the selected situation.',
                        disclosures=private.facts,preferences=selected_by_profile[profile_id],requested_operations=[raw])
                    try:prepared=_scoped_context(adapter,state,profiles,intent)
                    except (KeyError,TypeError,ValueError):continue
                    steps=[{'kind':'customer','intent':intent.model_dump(mode='json')}]
                decision=adapter.decide(prepared,profiles[0][1].identity,op)
                if not decision.allowed or decision.expected_state==prepared or adapter.invariants(decision.expected_state):continue
                if frame_preserved(world,decision.expected_state,task.frames):
                    children.append((order.get(op.name,50),profile_id,decision.expected_state,
                        path+steps+[{'kind':'business','operation':raw}],depth+1,selected_by_profile))
        for event_id in task.external_schedule:
            if adapter.external_ready(state,event_id):
                following=adapter.external_reference(state,event_id)
                if not adapter.invariants(following) and frame_preserved(world,following,task.frames):
                    children.append((25,event_id,following,path+[{'kind':'external','event_id':event_id}],depth+1,selected_by_profile))
        children.sort(key=lambda row:(row[0],row[1]))
        for _,_,following,following_path,following_depth,profile_choices in reversed(children):
            stack.append((following,following_path,following_depth,profile_choices))
    result={'paths':paths,'nodes':nodes,'exhausted':not stack and not truncated,'max_nodes':max_nodes,'max_depth':max_depth,
        'preference_combinations':combinations,'preference_combinations_truncated':truncated}
    if codes is not None:result['matched_refusal_codes']=matched
    return result


def execute_witness(adapter,world,task,path):
    task=checked_task(adapter,world,task);session=DialogueSession(adapter,world,task.profile_id,
        profile_ids=task.composition.profile_ids if task.composition is not None else None)
    for step in path:
        if step['kind']=='customer':session.customer(CustomerIntent.model_validate(step['intent']))
        elif step['kind']=='service':session.reply('Actual binding offer for the selected option.',step['operation'])
        elif step['kind']=='business':
            result=session.act(step['operation'])
            if not result['success']:raise ValueError('actual Engine disagrees with private Reference witness: '+str(result))
        elif step['kind']=='external':session.external(step['event_id'])
        else:raise ValueError('unknown witness step')
    for goal in task.refusals:
        session.customer(CustomerIntent(message='Please perform this exact requested action.',requested_operations=[goal.operation]))
        decision=adapter.decide(session.state,session.private.identity,adapter.Operation.model_validate(goal.operation))
        if decision.allowed or decision.error_code!=goal.code:raise ValueError('mixed refusal goal is not established')
        session.reply('This requested action is disallowed for the stated business reason.',refused_operation=goal.operation,refusal_code=goal.code)
    return session.finish()


def certified_paths(adapter,world,task,search_result):
    rows=[]
    for path in search_result['paths']:
        trace=execute_witness(adapter,world,task,path)
        rows.append({'trace':trace,'certificate':certify(adapter,world,task,trace)})
    return rows
