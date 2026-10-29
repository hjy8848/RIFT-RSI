"""A bounded two-party conversation using the existing persistent Service harness."""
from pathlib import Path
from typing import Optional, Union

from usami import Operation, replay_trace
from usami.customer import CustomerAgent, initial_observation
from usami.interaction.authorization import IntentAdapter
from usami.interaction.models import (CUSTOMER_VERSION, INTERACTION_VERSION, InteractionConfig,
                                      InteractionTrace, InteractionTurn, OperationAudit)
from usami.service import CustomerInput, Experiment, ServiceSession
from usami.service.client import RecordedCompletionClient
from usami.service.models import SERVICE_VERSION, ServiceError
from usami.world.rules import ReferenceRules


def audit_operations(execution, turns):
    events = {e.event_id: e for e in execution.world_trace.events if e.kind == "operation"}
    oracle = {r.request_id: r for r in execution.oracle.requests}
    audits = []
    for turn in turns:
        for tool in execution.tool_records[turn.service.tool_start:turn.service.tool_end]:
            if not tool.event_id:
                continue
            event = events[tool.event_id]
            payload = event.payload
            operation = Operation.model_validate(payload["operation"])
            decision = ReferenceRules().decide(event.before, payload["actor_id"], operation, event.event_id)
            evaluation = oracle[payload["request_id"]]
            allowed = None if decision.error_code in {"INVALID_WORLD", "UNSUPPORTED_RULES_VERSION"} else decision.allowed
            success = payload["result"]["success"]
            # Primary evidence must hold even without interpreting natural language.
            business = (success and allowed is not None and evaluation.operation_correct is False
                        and (not allowed or event.after != decision.expected_state
                             or payload["result"]["data"] != decision.expected_data))
            if allowed is None or evaluation.operation_correct is None:
                category = "unscored"
            elif business:
                category = "successful_business_violation"
            elif not success and not allowed:
                category = "rejected_invalid_attempt"
            elif any(v.code == "UNAUTHORIZED_OPERATION" for v in evaluation.violations) or not execution.trusted_requests[payload["request_id"]].authorizes(payload["actor_id"], operation):
                category = "authorization_only"
            elif not success:
                category = "false_rejection"
            else:
                category = "legal_execution"
            audits.append(OperationAudit(customer_identity=payload["actor_id"], turn_id=turn.prepared.turn_id,
                call_id=tool.call_id, event_id=event.event_id, request_id=payload["request_id"], operation=operation,
                reference_allowed=allowed, reference_error_code=decision.error_code,
                expected_state_hash=decision.expected_state.state_hash if decision.expected_state else None,
                oracle=evaluation, classification=category))
    return audits


class InteractionSession:
    def __init__(self, config: InteractionConfig, customer_client, service_client):
        self._config = config.model_copy(deep=True)
        budget = self._config.budget
        experiment = Experiment(seed=config.seed, generator=config.generator, mode=config.mode, tools=config.tools,
            customer=CustomerInput(identity=config.customer_identity, message="Multi-turn Customer exploration"),
            max_llm_calls=budget.max_service_llm_calls, max_tool_calls=budget.max_service_tool_calls)
        self.service = ServiceSession(experiment, service_client, defer_initial_turn=True)
        self._view = initial_observation(self.service.state, config.customer_identity)
        self.customer = CustomerAgent(self._view, customer_client, strategy=config.exploration_strategy, tools=config.tools)
        self.reset()

    def reset(self):
        self.service.reset()
        self.customer.reset()
        self._adapter = IntentAdapter(self._view)
        self._turns, self._actions = [], []
        self._finished = False
        return self.service.state

    def _remaining(self):
        budget, usage = self._config.budget, self.service.usage
        return {"customer_turns": budget.max_customer_turns - len(self._turns),
                "customer_llm_calls": budget.max_customer_llm_calls - len(self.customer.records),
                "service_llm_calls": budget.max_service_llm_calls - usage["llm_calls"],
                "service_tool_calls": budget.max_service_tool_calls - usage["tool_calls"]}

    def run(self, on_turn=None):
        if self._finished:
            raise ValueError("episode already ran; call reset() before another episode")
        status, reason, stop_reason = "budget_exhausted", "customer_turn_budget_exhausted", ""
        while True:
            remaining = self._remaining()
            exhausted = next((key for key, value in remaining.items() if value <= 0), None)
            if exhausted:
                reason = {"customer_turns": "customer_turn_budget_exhausted", "customer_llm_calls": "customer_llm_budget_exhausted",
                          "service_llm_calls": "service_llm_budget_exhausted", "service_tool_calls": "service_tool_budget_exhausted"}[exhausted]
                break
            try:
                action = self.customer.next_action(remaining)
            except ServiceError as exc:
                status = "interrupted" if exc.code == "PROVIDER_INTERRUPTED" else "customer_error"
                reason = "customer_" + exc.code.lower()
                break
            self._actions.append(action)
            if action.type == "stop":
                status, reason, stop_reason = "completed", "customer_stopped", action.reason
                break
            turn_id = f"turn_{len(self._turns) + 1:03d}"
            prepared = self._adapter.prepare(action, self.service.state, self._config.customer_identity, turn_id)
            # Prepare and record the Customer input independently BEFORE any Service call.
            response = self.service.respond(prepared.customer, turn_id)
            self._turns.append(InteractionTurn(prepared=prepared, service=response))
            if on_turn is not None:
                on_turn(self._turns[-1].model_copy(deep=True))
            self._adapter.receive_reply(self.service.state, response.final_reply)
            self.customer.receive(prepared.customer.message, response.final_reply)
            if response.status != "completed":
                status = ("budget_exhausted" if response.status.endswith("budget_exhausted") else
                          "interrupted" if response.status == "interrupted" else "service_error")
                reason = "service_" + response.status
                break
        execution = self.service.finish(status)
        audits = audit_operations(execution, self._turns)
        intents = [i for t in self._turns for i in t.prepared.intents]
        metrics = {"customer_turns": len(self._turns), "customer_llm_calls": len(self.customer.records),
            "service_llm_calls": execution.metrics["llm_calls"], "service_tool_calls": execution.metrics["dispatched_tool_calls"],
            "engine_operations": execution.metrics["engine_operations"],
            "invalid_operation_attempts": sum(a.reference_allowed is False for a in audits),
            "correctly_rejected_invalid_attempts": sum(a.classification == "rejected_invalid_attempt" and a.oracle.operation_correct is True for a in audits),
            "successful_business_violations": sum(a.classification == "successful_business_violation" for a in audits),
            "authorization_only_attempts": sum(a.classification == "authorization_only" for a in audits),
            "successful_authorization_only_operations": sum(a.classification == "authorization_only" and next(e for e in execution.world_trace.events if e.event_id == a.event_id).payload["result"]["success"] for a in audits),
            "unscored_operations": sum(a.classification == "unscored" for a in audits),
            "tool_errors": sum(t.dispatched and not t.event_id and not t.response["success"] for t in execution.tool_records),
            "accepted_intents": sum(i.disposition == "accepted" for i in intents),
            "duplicate_intents": sum(i.disposition == "duplicate" for i in intents),
            "unverified_intents": sum(i.disposition == "unverified_object" for i in intents),
            "unverified_free_text_turns": sum(t.prepared.natural_language_status == "unverified_free_text" for t in self._turns),
            "expired_request_ids": [rid for t in self._turns for rid in t.service.expired_request_ids],
            "unfulfilled_business_requests": execution.metrics["unfulfilled_authorizations"]}
        trace = InteractionTrace(config=self._config, customer_model=self.customer.settings, service_model=execution.model,
            versions={"interaction": INTERACTION_VERSION, "customer": CUSTOMER_VERSION, "service": SERVICE_VERSION,
                      "generator": execution.world_trace.initial_state.generator_version,
                      "rules": execution.world_trace.initial_state.rules.version, "world_trace": execution.world_trace.schema_version},
            initial_observation=self._view, customer_records=self.customer.records, customer_actions=self._actions,
            customer_messages=self.customer.messages, turns=self._turns, service_execution=execution,
            operation_audits=audits, status=status, termination_reason=reason, stop_reason=stop_reason, metrics=metrics)
        trace.interaction_hash = trace.computed_hash()
        self._finished = True
        return trace.model_copy(deep=True)


def replay_interaction(trace: Union[InteractionTrace, str, Path], config: Optional[InteractionConfig] = None,
                       expected_hash: Optional[str] = None):
    """Rebuild both conversations, intent adaptation and the original core events offline."""
    if not isinstance(trace, InteractionTrace):
        trace = InteractionTrace.load(trace)
    if trace.schema_version != INTERACTION_VERSION or trace.interaction_hash != trace.computed_hash():
        raise ValueError("unsupported or corrupt interaction trace")
    if expected_hash is not None and expected_hash != trace.interaction_hash:
        raise ValueError("interaction differs from the external trusted hash")
    if config is not None and config != trace.config:
        raise ValueError("interaction differs from the external trusted configuration")
    if trace.service_execution.execution_hash != trace.service_execution.computed_hash():
        raise ValueError("corrupt nested Service execution")
    engine = replay_trace(trace.service_execution.world_trace)
    customer = RecordedCompletionClient(trace.customer_model, trace.customer_records)
    service = RecordedCompletionClient(trace.service_model, trace.service_execution.llm_records)
    regenerated = InteractionSession(trace.config, customer, service).run()
    if (regenerated != trace or customer.index != len(trace.customer_records)
            or service.index != len(trace.service_execution.llm_records)
            or regenerated.service_execution.world_trace.final_state != engine.state):
        raise ValueError("altered interaction, observation, request authorization, tool result or Oracle evidence")
    return regenerated
