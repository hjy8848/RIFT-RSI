"""Exact state/authority replay; no online model call and no modified initial world."""
from .engine import DialogueSession,ExecutionTrace
from .oracle import evaluate


def replay(adapter,trace,*,initial_world=None,private=None):
    trace=ExecutionTrace.model_validate(trace.model_dump(mode='json') if isinstance(trace,ExecutionTrace) else trace)
    report=evaluate(adapter,trace,initial_world=initial_world,private=private)
    if not report['trace_valid']:raise ValueError('dialogue evidence invalid: '+str(report['errors']))
    world=adapter.World.model_validate(trace.initial_world)
    session=DialogueSession(adapter,world,trace.profile_id,max_operations=trace.max_operations,profile_ids=trace.scoped_profile_ids or None)
    for event in trace.events[1:]:
        if event.kind=='customer':session.customer(event.payload['intent'])
        elif event.kind=='service':session.reply(event.payload['message'],event.payload['offered_operation'],refused_operation=event.payload['refused_operation'],refusal_code=event.payload['refusal_code'])
        elif event.kind=='external':session.external(event.payload['event_id'])
        elif event.kind=='business':
            native=event.payload['trace'];operation=next(e for e in native['events'] if e['kind']=='operation')['payload']
            session.act(operation['operation'],operation['mode'])
    rebuilt=session.finish()
    if rebuilt!=trace:raise ValueError('actual dialogue/business reconstruction differs')
    return {'verified':True,'events':len(trace.events),'operations':len(report['operations']),'oracle':report}
