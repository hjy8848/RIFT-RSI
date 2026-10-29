"""Actual event execution and exact replay parameterized by an explicit adapter."""
from typing import Any, Dict, List, Literal, Optional
from pydantic import Field
from .models import Model, Request, Result, digest


class Event(Model):
    id: str
    kind: Literal['generation', 'request', 'operation', 'final']
    payload: Dict[str, Any]
    before: Optional[Dict[str, Any]] = None
    after: Optional[Dict[str, Any]] = None
    previous_hash: str = ''
    event_hash: str = ''

    def computed_hash(self):
        return digest(self.model_dump(mode='json', exclude={'event_hash'}))


class Trace(Model):
    domain: str
    version: str
    initial_state: Dict[str, Any]
    operation_budget: int = Field(strict=True, ge=1, le=2000)
    events: List[Event] = Field(default_factory=list)
    final_state: Dict[str, Any]
    trace_hash: str = ''

    def computed_hash(self):
        return digest(self.model_dump(mode='json', exclude={'trace_hash'}))

    def append(self, kind, payload, before=None, after=None):
        event = Event(id=f'event_{len(self.events):06d}', kind=kind, payload=payload,
                      before=before.model_dump(mode='json') if before is not None else None,
                      after=after.model_dump(mode='json') if after is not None else None,
                      previous_hash=self.events[-1].event_hash if self.events else '')
        event.event_hash = event.computed_hash()
        self.events.append(event)
        return event


class Engine:
    def __init__(self, adapter, world, operation_budget=100):
        self.adapter = adapter
        world = adapter.World.model_validate(world.model_dump())
        if world.domain != adapter.name or adapter.invariants(world):
            raise ValueError('invalid initial state or foreign domain')
        self.initial = world.clone()
        self._state = world.clone()
        self.requests, self.used = {}, set()
        self.trace = Trace(domain=adapter.name, version=world.version,
                           initial_state=world.model_dump(mode='json'),
                           final_state=world.model_dump(mode='json'), operation_budget=operation_budget)
        self.trace.append('generation', {'domain': world.domain, 'version': world.version,
                                        'seed': world.seed, 'initial_hash': world.state_hash})

    @property
    def state(self):
        return self._state.clone()

    def register(self, request):
        if self.trace.events[-1].kind == 'final' or request.id in self.requests:
            raise ValueError('closed engine or reused request ID')
        operation = self.adapter.Operation.model_validate(request.operation)
        if request.operation != operation.model_dump(mode='json'):
            raise ValueError('noncanonical authority')
        self.requests[request.id] = request.model_copy(deep=True)
        self.trace.append('request', {'request': request.model_dump(mode='json')})

    def execute(self, actor, operation, request_id, mode='reference'):
        operation = self.adapter.Operation.model_validate(operation.model_dump())
        if mode not in {'reference', 'sandbox', 'refuse'}:
            raise ValueError('unknown execution mode')
        if self.trace.events[-1].kind == 'final' or request_id not in self.requests:
            raise ValueError('closed engine or missing independent authority')
        if sum(e.kind == 'operation' for e in self.trace.events) >= self.trace.operation_budget:
            raise ValueError('operation budget exhausted')
        before, request = self.state, self.requests[request_id]
        decision = self.adapter.decide(before, actor, operation)
        if mode == 'refuse':
            result = Result(success=False, error_code='SERVICE_REFUSAL')
        elif mode == 'reference' or operation.is_query:
            code = ('UNAUTHORIZED_OPERATION' if not request.authorizes(actor, operation)
                    else 'REQUEST_ALREADY_USED' if request_id in self.used
                    else decision.error_code if not decision.allowed else None)
            if code:
                result = Result(success=False, error_code=code)
            else:
                self._state, result = self.adapter.apply(before, actor, operation, len(self.trace.events))
        else:
            self._state, result = self.adapter.apply(before, actor, operation, len(self.trace.events))
        self.used.add(request_id)
        event = self.trace.append('operation', {'actor': actor, 'operation': operation.model_dump(mode='json'),
                                  'request_id': request_id, 'mode': mode,
                                  'result': result.model_dump(mode='json')}, before, self.state)
        self.trace.final_state = self.state.model_dump(mode='json')
        return event.model_copy(deep=True)

    def finish(self):
        if self.trace.events[-1].kind != 'final':
            self.trace.append('final', {'state_hash': self.state.state_hash})
        self.trace.trace_hash = self.trace.computed_hash()
        return self.trace.model_copy(deep=True)

    def reset(self):
        self.__dict__.update(Engine(self.adapter, self.initial, self.trace.operation_budget).__dict__)
        return self.state


def replay_trace(adapter, trace, trusted_requests=None):
    trace = Trace.model_validate(trace.model_dump())
    if trace.domain != adapter.name or trace.trace_hash != trace.computed_hash():
        raise ValueError('foreign domain or trace integrity mismatch')
    engine = Engine(adapter, adapter.World.model_validate(trace.initial_state), trace.operation_budget)
    if not trace.events or trace.events[0] != engine.trace.events[0]:
        raise ValueError('generation provenance differs')
    for event in trace.events[1:]:
        if event.kind == 'request':
            request = Request.model_validate(event.payload['request'])
            if trusted_requests is not None and trusted_requests.get(request.id) != request:
                raise ValueError('external request authority differs')
            engine.register(request)
        elif event.kind == 'operation':
            p = event.payload
            engine.execute(p['actor'], adapter.Operation.model_validate(p['operation']), p['request_id'], p['mode'])
        elif event.kind == 'final':
            engine.finish()
        else:
            raise ValueError('unexpected generation event')
        if engine.trace.events[-1] != event:
            raise ValueError('actual event reconstruction differs')
    if engine.finish() != trace:
        raise ValueError('terminal trace differs')
    if trusted_requests is not None and engine.requests != trusted_requests:
        raise ValueError('authority set differs')
    return engine
