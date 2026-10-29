"""Deterministic, task-independent lifecycle sampling with original execution proof."""
from usami_harness.models import Request
from usami_harness.engine import Engine, replay_trace
from usami_harness.oracle import evaluate


class NativeHistory:
    def __init__(self,adapter,world,diagnostics=None):
        self.diagnostics=diagnostics if diagnostics is not None else []
        self.adapter=adapter;self.engine=Engine(adapter,world,operation_budget=2000)
    @property
    def state(self):return self.engine.state
    def step(self,actor,name,scope,resource,params=None):
        op=self.adapter.Operation(name=name,scope_id=scope,resource_id=resource,params=params or {})
        # Rejection sampling is on independent simulator choices, never a task or LLM outcome.
        decision=self.adapter.decide(self.state,actor,op)
        if not decision.allowed:return False
        errors=self.adapter.invariants(decision.expected_state)
        if errors:
            self.diagnostics.append({'kind':'native_reference_invariant_conflict','actor':actor,'operation':op.model_dump(mode='json'),
                'before_hash':self.state.state_hash,'expected_hash':decision.expected_state.state_hash,'invariants':errors})
            return False
        req=Request(id='generation-'+str(len(self.engine.requests)),identity=actor,operation=op.model_dump(mode='json'))
        self.engine.register(req);event=self.engine.execute(actor,op,req.id)
        if not event.payload['result']['success']:raise ValueError('generator transition disagrees with Reference')
        return True
    def finish(self):
        trace=self.engine.finish()
        if evaluate(self.adapter,trace,self.engine.requests)['status']!='pass':raise ValueError('generation history failed original Oracle')
        if replay_trace(self.adapter,trace,self.engine.requests).state!=self.state:raise ValueError('generation history replay differs')
        return self.state
