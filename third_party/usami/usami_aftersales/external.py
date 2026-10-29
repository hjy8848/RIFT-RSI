"""Pre-registered manager/carrier simulator events, with independent expected changes."""
from usami_harness.models import signed_notice
from .models import Return


def expected(world,event_id):
    event=world.external_events.get(event_id)
    if not event or event_id in world.applied_events:raise ValueError('unknown or consumed external event')
    cases=[c for c in world.cases.values() if c.line_id==event.line_id]
    if len(cases)!=1:raise ValueError('external event requires an unambiguous real case')
    case=cases[0];new=world.clone();new.clock=max(world.clock,event.at)
    if event.kind=='manager_review':
        if not case.review_requested:raise ValueError('review has not been requested')
        binding={'line_id':event.line_id,'approved':event.outcome=='approved','event_id':event.id}
        new.approvals[event.line_id]={'approved':binding['approved'],'notice':signed_notice(world.domain,'merchant-manager',binding,new.clock)}
    else:
        ret=world.returns.get('return_'+case.id)
        if not ret or ret.status!='scheduled':raise ValueError('carrier cannot receive an unscheduled return')
        if event.outcome=='received':
            new.returns[ret.id].status='received';binding={'case_id':case.id,'return_id':ret.id,'received':True}
            new.returns[ret.id].receipt_notice=signed_notice(world.domain,'carrier',binding,new.clock)
    new.applied_events.append(event.id)
    return new


def apply(world,event_id):
    # Separate mutation implementation; Oracle calls expected(), never this function.
    item=world.external_events.get(event_id)
    if not item or event_id in world.applied_events:raise ValueError('unknown or consumed external event')
    matching=[c for c in world.cases.values() if c.line_id==item.line_id]
    if len(matching)!=1:raise ValueError('ambiguous external target')
    case=matching[0];new=world.clone();new.clock=max(item.at,new.clock)
    if item.kind=='manager_review':
        if not case.review_requested:raise ValueError('review has not been requested')
        answer=item.outcome=='approved';proof={'line_id':item.line_id,'approved':answer,'event_id':item.id}
        new.approvals[item.line_id]={'approved':answer,'notice':signed_notice(new.domain,'merchant-manager',proof,new.clock)}
    elif item.kind=='return_received':
        identifier='return_'+case.id
        if identifier not in new.returns or new.returns[identifier].status!='scheduled':raise ValueError('return is not scheduled')
        if item.outcome=='received':
            new.returns[identifier].status='received'
            new.returns[identifier].receipt_notice=signed_notice(new.domain,'carrier',{'case_id':case.id,'return_id':identifier,'received':True},new.clock)
    new.applied_events.append(event_id);return new
