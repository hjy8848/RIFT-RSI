"""Conservative explicit-refusal audit. Never invents an Engine event from prose."""
from usami.service.models import strict_json
from usami_cloudops.models import Operation
from usami_cloudops.rules import POLICY,decide
from vast_coevo.models import TaskDecision
from vast_coevo.schema import digest


def decisions(execution,calls):
    reply=execution.final_reply
    if not isinstance(reply,str) or not execution.inputs:return ()
    turn=execution.inputs[-1];grants=turn['grants']
    if len(grants)!=1:return ()
    operation=Operation.model_validate(grants[0]);code=reply.strip()
    if code not in POLICY:
        try:body=strict_json(code)
        except (ValueError,TypeError):return ()
        if (not isinstance(body,dict) or set(body)!={'decision','operation','code'} or body['decision']!='refuse' or
                body['operation']!=operation.model_dump(mode='json') or not isinstance(body['code'],str) or body['code'] not in POLICY):return ()
        code=body['code']
    if any(e.kind=='operation' and not Operation.model_validate(e.payload['operation']).is_query for e in execution.world_trace.events):return ()
    state=execution.world_trace.initial_state
    if not decide(state,execution.experiment.identity,operation).allowed:return ()
    service=[c for c in calls if c.role=='service' and c.response is not None and not c.error]
    if not service:return ()
    final=service[-1]
    if final.response['choices'][0]['message'].get('content')!=reply:return ()
    return (TaskDecision(operation=operation.model_dump(mode='json'),before_hash=state.state_hash,claimed_code=code,
        model_call_id=final.id,response_hash=digest(final.response),turn_id=turn['turn_id']),)
