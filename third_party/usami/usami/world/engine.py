"""Trusted strict execution and explicit permissive in-memory sandbox execution.

Sandbox applies syntactically valid operations without consulting ReferenceRules.
It is an evaluation mechanism, not an intentionally vulnerable business backend.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional, Union

from usami.events import Event, Trace
from usami.protocol import ExecutionResult, Operation, RequestContext
from usami.world.models import Refund, WorldState, digest
from usami.world.rules import ReferenceRules, check_invariants


class WorldEngine:
    def __init__(self, world: WorldState, operation_budget: int = 100):
        if check_invariants(world):
            raise ValueError("initial world must satisfy reference invariants")
        self._initial = world.clone()
        self._state = world.clone()
        self._requests: Dict[str, RequestContext] = {}
        self.operation_budget = operation_budget
        self.trace = Trace(schema_version="usami-trace-0.2.0" if world.rules.version == "saas-rules-0.2.0" else "usami-trace-0.1.0", initial_state=world.clone(), initial_state_hash=world.state_hash,
                           rules_hash=digest(world.rules), operation_budget=operation_budget,
                           final_state=world.clone(), final_state_hash=world.state_hash)
        self.trace.append("generation", {"generator_version": world.generator_version, "seed": world.seed,
                                         "config_hash": world.config_hash, "initial_state_hash": world.state_hash,
                                         "rules_hash": digest(world.rules)})

    @property
    def state(self) -> WorldState:
        return self._state.clone()

    def register_request(self, request: RequestContext) -> Event:
        request = RequestContext.model_validate(request.model_dump(mode="json"))
        if self.trace.events[-1].kind == "final":
            raise ValueError("finalized engine must be reset before further execution")
        if request.request_id in self._requests:
            raise ValueError("request_id already registered")
        self._requests[request.request_id] = request.model_copy(deep=True)
        return self.trace.append("request", {"context": request.model_dump(mode="json")})

    def execute(self, actor_id: str, operation: Operation, request_id: str,
                mode: Literal["reference", "sandbox", "refuse"] = "reference",
                refusal_code: str = "SERVICE_REFUSAL") -> Event:
        operation = Operation.model_validate(operation.model_dump(mode="json"))
        if request_id not in self._requests:
            raise ValueError("trusted request must be registered before execution")
        if self.trace.events[-1].kind == "final":
            raise ValueError("finalized engine must be reset before further execution")
        if mode not in {"reference", "sandbox", "refuse"}:
            raise ValueError("unknown execution mode")
        if sum(e.kind == "operation" for e in self.trace.events) >= self.operation_budget:
            raise ValueError("operation budget exhausted")
        before = self._state.clone()
        event_id = f"event_{len(self.trace.events):06d}"
        if mode == "refuse":
            result = ExecutionResult(success=False, error_code=refusal_code)
        elif mode == "reference":
            request = self._requests[request_id]
            decision = ReferenceRules().decide(before, actor_id, operation, event_id)
            if not request.authorizes(actor_id, operation):
                result = ExecutionResult(success=False, error_code="UNAUTHORIZED_OPERATION")
            elif any(e.kind == "operation" and e.payload["request_id"] == request_id for e in self.trace.events):
                result = ExecutionResult(success=False, error_code="REQUEST_ALREADY_EXECUTED")
            elif not decision.allowed:
                result = ExecutionResult(success=False, error_code=decision.error_code)
            else:
                if before.rules.version == "saas-rules-0.2.0":
                    from usami.world.transitions_v2 import apply_v2
                    self._state, result = apply_v2(before, actor_id, operation, event_id)
                else:
                    self._state = decision.expected_state.clone()
                    result = ExecutionResult(success=True, data=decision.expected_data)
        else:
            self._state, result = sandbox_apply(before, actor_id, operation, event_id)
        event = self.trace.append("operation", {"request_id": request_id, "actor_id": actor_id,
                                                "operation": operation.model_dump(mode="json"),
                                                "result": result.model_dump(mode="json"), "mode": mode,
                                                "refusal_code": refusal_code}, before, self._state)
        self.trace.final_state = self._state.clone()
        self.trace.final_state_hash = self._state.state_hash
        return event

    def reset(self) -> WorldState:
        fresh = WorldEngine(self._initial, self.operation_budget)
        self.__dict__.update(fresh.__dict__)
        return self.state

    def clone(self) -> WorldEngine:
        import copy
        return copy.deepcopy(self)

    def export_snapshot(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self._state.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def import_snapshot(cls, path: Union[str, Path], operation_budget: int = 100) -> WorldEngine:
        return cls(WorldState.model_validate_json(Path(path).read_text(encoding="utf-8")), operation_budget)

    def finish(self, include_evaluation: bool = True) -> Trace:
        if not self.trace.events or self.trace.events[-1].kind != "final":
            if include_evaluation:
                from usami.evaluation.oracle import evaluate_trace
                report = evaluate_trace(self.trace)
                self.trace.append("oracle", {"report": report.model_dump(mode="json")})
            self.trace.append("final", {"final_state_hash": self._state.state_hash})
        return self.trace.model_copy(deep=True)


def sandbox_apply(world: WorldState, actor_id: str, op: Operation, event_id: str):
    """Only schema/existence checks. All business validation belongs to the oracle."""
    if world.rules.version == "saas-rules-0.2.0":
        from usami.world.transitions_v2 import apply_v2
        return apply_v2(world, actor_id, op, event_id)
    new = world.clone()
    if actor_id not in world.users:
        return new, ExecutionResult(success=False, error_code="UNKNOWN_ACTOR")
    if op.org_id not in world.organizations:
        return new, ExecutionResult(success=False, error_code="ORG_NOT_FOUND")
    name, rid, params = op.name, op.resource_id, op.params
    if name in {"change_plan", "change_seats", "cancel_renewal", "get_subscription"}:
        resource = new.subscriptions.get(rid)
    elif name in {"refund_invoice", "get_invoice"}:
        resource = new.invoices.get(rid)
    elif name == "change_member_role":
        resource = new.memberships.get(rid)
    else:
        resource = new.organizations.get(rid)
    if resource is None:
        return new, ExecutionResult(success=False, error_code="RESOURCE_NOT_FOUND")
    data = {}
    if name == "change_plan":
        resource.plan_id = params["plan_id"]
    elif name == "change_seats":
        resource.seats = params["seats"]
    elif name == "cancel_renewal":
        resource.auto_renew = False
    elif name == "refund_invoice":
        resource.status = "refunded"
        resource.refunds.append(Refund(id=f"refund_{event_id}", amount_cents=params["amount_cents"],
                                      payment_method=params["payment_method"], refunded_at=world.clock))
    elif name == "change_member_role":
        resource.role = params["role"]
    elif name == "list_members":
        data = {"memberships": [m.model_dump(mode="json") for m in new.memberships.values() if m.org_id == rid]}
    else:
        data = resource.model_dump(mode="json")
    return new, ExecutionResult(success=True, data=data)


def clone_world(world: Union[WorldState, WorldEngine]):
    return world.clone()


def reset_world(engine: WorldEngine) -> WorldState:
    return engine.reset()


def execute_operation(engine: WorldEngine, actor_id: str, operation: Operation, request_id: str,
                      mode: str = "reference") -> Event:
    return engine.execute(actor_id, operation, request_id, mode)


def replay_trace(trace: Trace) -> WorldEngine:
    if trace.integrity_errors():
        raise ValueError("cannot replay corrupt trace")
    engine = WorldEngine(trace.initial_state, trace.operation_budget)
    if not trace.events or trace.events[0] != engine.trace.events[0]:
        raise ValueError("generation provenance mismatch")
    for recorded in trace.events[1:]:
        if recorded.kind == "request":
            actual = engine.register_request(RequestContext.model_validate(recorded.payload["context"]))
        elif recorded.kind == "operation":
            p = recorded.payload
            actual = engine.execute(p["actor_id"], Operation.model_validate(p["operation"]), p["request_id"],
                                    p["mode"], p.get("refusal_code", "SERVICE_REFUSAL"))
        elif recorded.kind == "oracle":
            from usami.evaluation.oracle import evaluate_trace
            actual = engine.trace.append("oracle", {"report": evaluate_trace(engine.trace).model_dump(mode="json")})
        elif recorded.kind == "final":
            actual = engine.trace.append("final", {"final_state_hash": engine.state.state_hash})
        else:
            raise ValueError("unexpected generation event")
        if actual != recorded:
            raise ValueError(f"non-deterministic or altered event: {recorded.event_id}")
    if engine.state.state_hash != trace.final_state_hash:
        raise ValueError("final state mismatch")
    return engine
