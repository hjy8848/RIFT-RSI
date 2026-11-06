"""Run one retail episode: a Service agent against the native Engine, judged by the golden final state.

The interface matches ``run_episode``: the Service object exposes ``decide(ServicePrompt) -> ServiceAction``
(and ``uses_model``), so the stepwise bridge that drives model Services can drive these tasks too. Every
tool call goes through the real Engine, Oracle and Replay; the customer's request grants authority over the
customer's own lines, as the structured request would.
"""
import json
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from usami_v2.runtime.agents import AgentMessage, ServicePrompt
from usami_v2.runtime.runner import _execution, _public_ids
from usami_v2.scenarios.retail_tasks import RetailTask, delegated, evaluate_retail_task


class ScriptedRetailCustomer:
    """Confirms the first proposal, then ends. Preferences already live in the task, not in the dialogue."""

    uses_model = False

    def __init__(self, confirmation="Yes, please go ahead.", farewell="Thanks, that is all."):
        self.confirmation, self.farewell = confirmation, farewell

    def respond(self, request, conversation, service_message, turn):
        return (self.confirmation, False) if turn == 0 else (self.farewell, True)


@dataclass
class RetailEpisode:
    status: str
    terminal_reason: str
    success: bool
    state_match: bool
    told_total: bool
    service_decisions: int
    tool_calls: int
    transcript: List[Any] = field(default_factory=list)
    tool_log: List[Tuple[str, dict]] = field(default_factory=list)
    final_world: Any = None


def run_retail_episode(world, task: RetailTask, service, customer=None, *, max_service_decisions=24,
                       max_tool_calls=16, max_customer_turns=6, patience=None, mode="reference"):
    task = task.checked()
    if world.state_hash != task.world_hash:
        raise ValueError("task belongs to another world")
    customer = customer or ScriptedRetailCustomer()
    current = delegated(world, task.customer_id)
    conversation = [AgentMessage(role="user", content=task.customer_request)]
    observed = {task.customer_id}
    service_texts, tool_log = [], []
    decisions = tools = turns = 0
    status, reason = "budget_exhausted", "service_decision_budget_exhausted"
    while decisions < max_service_decisions and tools < max_tool_calls:
        if patience is not None and decisions >= patience:
            status, reason = "customer_left", "customer_patience_exhausted"
            break
        prompt = ServicePrompt(customer_id=task.customer_id, customer_request=task.customer_request,
                               conversation=tuple(conversation),
                               available_merchant_scope_ids=tuple(sorted(world.merchants)),
                               observed_subject_ids=tuple(sorted(observed)))
        try:
            action = service.decide(prompt)
        except Exception as exc:
            status, reason = "agent_error", type(exc).__name__ + ": " + str(exc)
            break
        decisions += 1
        if action.kind == "operation" and action.operation is not None:
            current, row = _execution(current, task.customer_id, action.operation, "retail_request_%d" % tools, mode)
            tools += 1
            tool_log.append((action.operation.name, row.result))
            observed.update(_public_ids(row.result))
            if row.result.get("success"):
                observed.add(action.operation.resource_id)
            if action.call:
                raw = action.call.response.get("choices", [{}])[0].get("message", {})
                conversation.append(AgentMessage(role="assistant", content=raw.get("content"),
                                                 tool_calls=tuple(raw.get("tool_calls", []))))
            conversation.append(AgentMessage(role="tool", tool_call_id=action.call_id or "retail_request_%d" % (tools - 1),
                                             content=json.dumps(row.result, ensure_ascii=False, sort_keys=True)))
        elif action.kind == "reply":
            service_texts.append(action.message)
            conversation.append(AgentMessage(role="assistant", content=action.message))
            if turns >= max_customer_turns:
                status, reason = "budget_exhausted", "customer_turn_budget_exhausted"
                break
            text, finish = customer.respond(task.customer_request, tuple(conversation), action.message, turns)
            turns += 1
            conversation.append(AgentMessage(role="user", content=text))
            if finish:
                status, reason = "completed", "customer_finished"
                break
        else:
            status, reason = "service_protocol_error", "invalid Service decision"
            break
    outcome = evaluate_retail_task(world, current, task, service_texts)
    return RetailEpisode(status=status, terminal_reason=reason, success=outcome["success"] and status == "completed",
                         state_match=outcome["state_match"], told_total=outcome["told_total"],
                         service_decisions=decisions, tool_calls=tools, transcript=conversation, tool_log=tool_log,
                         final_world=current)
