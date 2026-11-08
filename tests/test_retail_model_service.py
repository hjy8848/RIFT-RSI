"""A model-backed Service on the retail profile reaches the new tools; scripted chat client, no network."""
import json

from usami_v2.runtime.agents import OpenAIServiceAgent
from usami_v2.runtime.retail_runner import run_retail_episode
from usami_v2.scenarios.retail_tasks import generate_retail_task, gold_operations


def tool_message(name, arguments, index):
    return {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_%d" % index, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}}]}


def text_message(text):
    return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": text}}]}


class Settings:
    def model_copy(self, deep=False):
        return self

    def request(self, messages, tools):
        return {"messages": messages, "tools": tools}


class ScriptedClient:
    """Plays the model: asks first, then performs the golden operation through a real tool call."""

    settings = Settings()

    def __init__(self, operation):
        self.operation, self.requests = operation, []

    def complete(self, request):
        self.requests.append(request)
        tools = [m for m in request["messages"] if m["role"] == "tool"]
        confirmed = any(m["role"] == "user" and "go ahead" in (m.get("content") or "") for m in request["messages"])
        if not confirmed:
            return text_message("I will change it. Shall I proceed?")
        if not tools:
            return tool_message(self.operation.name, self.arguments(), 1)
        return text_message("Done.")

    def arguments(self):
        op = self.operation
        if op.name == "update_default_address":
            return dict(op.params)
        return {"order_id": op.resource_id, **op.params}


def test_a_model_service_on_the_retail_profile_can_use_the_new_tools_end_to_end():
    for family in ("address_stated", "default_from_order", "items_simple"):
        world, task, _ = generate_retail_task(4, family)
        operation = gold_operations(world, task)[0][0]
        client = ScriptedClient(operation)
        episode = run_retail_episode(world, task, OpenAIServiceAgent(client, tool_profile="retail"))
        assert episode.success, (family, episode.status, episode.terminal_reason, episode.tool_log)
        sent = {row["function"]["name"] for row in client.requests[0]["tools"]}
        assert {"modify_order_address", "modify_order_items", "get_customer_profile"} <= sent


def test_the_full9_profile_still_refuses_the_new_tools():
    world, task, _ = generate_retail_task(4, "address_stated")
    operation = gold_operations(world, task)[0][0]
    episode = run_retail_episode(world, task, OpenAIServiceAgent(ScriptedClient(operation), tool_profile="full9"))
    assert episode.status == "agent_error" and not episode.success
