"""Domain-local hash chained events and actual Reference/Sandbox execution."""
from typing import Any, Dict, List, Literal, Optional
from pydantic import Field
from .models import Model, Operation, Request, Result, World, VERSION, digest
from .rules import decide, invariants
from .transitions import apply


class Event(Model):
    id: str
    kind: Literal['generation','request','operation','final']
    payload: Dict[str, Any]
    before: Optional[World] = None
    after: Optional[World] = None
    previous_hash: str = ''
    event_hash: str = ''
    def computed_hash(self): return digest(self.model_dump(mode='json', exclude={'event_hash'}))


class Trace(Model):
    domain: Literal['cloudops'] = 'cloudops'
    version: Literal['cloudops-0.1.0'] = VERSION
    initial_state: World
    operation_budget: int = Field(strict=True, ge=1, le=1000)
    events: List[Event] = Field(default_factory=list)
    final_state: World
    trace_hash: str = ''
    def computed_hash(self): return digest(self.model_dump(mode='json', exclude={'trace_hash'}))
    def append(self, kind, payload, before=None, after=None):
        event = Event(id=f'event_{len(self.events):06d}', kind=kind, payload=payload,
            before=before.clone() if before else None, after=after.clone() if after else None,
            previous_hash=self.events[-1].event_hash if self.events else '')
        event.event_hash = event.computed_hash(); self.events.append(event)
        return event


class Engine:
    def __init__(self, world, operation_budget=100):
        world = World.model_validate(world.model_dump())
        if invariants(world): raise ValueError('illegal initial CloudOps world')
        self.initial = world.clone(); self._state = world.clone(); self.requests = {}; self.used = set()
        self.trace = Trace(initial_state=world.clone(), final_state=world.clone(), operation_budget=operation_budget)
        self.trace.append('generation', {'domain': 'cloudops', 'version': VERSION, 'seed': world.seed, 'initial_hash': world.state_hash})
    @property
    def state(self): return self._state.clone()
    def register(self, request):
        if self.trace.events[-1].kind == 'final': raise ValueError('closed engine')
        if request.id in self.requests: raise ValueError('duplicate trusted request ID')
        self.requests[request.id] = request.model_copy(deep=True)
        self.trace.append('request', {'request': request.model_dump(mode='json')})
    def execute(self, actor, operation, request_id, mode='reference'):
        op = Operation.model_validate(operation.model_dump())
        if mode not in {'reference','sandbox','refuse'}: raise ValueError('unknown mode')
        if self.trace.events[-1].kind == 'final': raise ValueError('closed engine')
        if request_id not in self.requests: raise ValueError('missing trusted request')
        if sum(e.kind == 'operation' for e in self.trace.events) >= self.trace.operation_budget: raise ValueError('operation budget exhausted')
        before = self.state; req = self.requests[request_id]; decision = decide(before, actor, op)
        if mode == 'refuse': result = Result(success=False, error_code='SERVICE_REFUSAL')
        elif mode == 'reference' or op.is_query:
            code = 'UNAUTHORIZED_OPERATION' if not req.authorizes(actor, op) else 'REQUEST_ALREADY_USED' if request_id in self.used else decision.error_code if not decision.allowed else None
            if code: result = Result(success=False, error_code=code)
            else: self._state, result = apply(before, actor, op, f'event_{len(self.trace.events):06d}')
        else: self._state, result = apply(before, actor, op, f'event_{len(self.trace.events):06d}')
        self.used.add(request_id)
        event = self.trace.append('operation', {'actor': actor, 'operation': op.model_dump(mode='json'),
            'request_id': request_id, 'mode': mode, 'result': result.model_dump(mode='json')}, before, self.state)
        self.trace.final_state = self.state
        return event.model_copy(deep=True)
    def finish(self):
        if self.trace.events[-1].kind != 'final': self.trace.append('final', {'state_hash': self.state.state_hash})
        self.trace.trace_hash = self.trace.computed_hash()
        return self.trace.model_copy(deep=True)
    def reset(self):
        self.__dict__.update(Engine(self.initial, self.trace.operation_budget).__dict__)
        return self.state


def replay_trace(trace, trusted_requests=None):
    trace = Trace.model_validate(trace.model_dump())
    if trace.trace_hash != trace.computed_hash(): raise ValueError('CloudOps trace hash mismatch')
    engine = Engine(trace.initial_state, trace.operation_budget)
    if trace.events[0] != engine.trace.events[0]: raise ValueError('CloudOps generation provenance differs')
    for event in trace.events[1:]:
        if event.kind == 'request':
            req = Request.model_validate(event.payload['request'])
            if trusted_requests is not None and trusted_requests.get(req.id) != req: raise ValueError('external CloudOps authority differs')
            engine.register(req)
        elif event.kind == 'operation':
            p = event.payload
            engine.execute(p['actor'], Operation.model_validate(p['operation']), p['request_id'], p['mode'])
        elif event.kind == 'final': engine.finish()
        else: raise ValueError('unexpected generation event')
        if engine.trace.events[-1] != event: raise ValueError('actual CloudOps replay differs from recorded event')
    if engine.finish() != trace: raise ValueError('CloudOps terminal trace differs')
    return engine
