"""CPU tests for Service tool-turn sampling and schema reminders."""
from types import SimpleNamespace

import pytest

from rift.verl.agent_loop import single_tool_turn_sampling_params
from rift.verl.usami_session import service_tool_protocol_reminder


def test_hermes_stop_preserves_existing_settings_and_closing_tag():
    parser = SimpleNamespace(tool_call_end_token="</tool_call>")
    original = {"temperature": 0.8, "seed": 39, "stop": ["<|im_end|>"]}

    configured = single_tool_turn_sampling_params(original, parser)

    assert original == {"temperature": 0.8, "seed": 39, "stop": ["<|im_end|>"]}
    assert configured["stop"] == ["<|im_end|>", "</tool_call>"]
    assert configured["include_stop_str_in_output"] is True
    assert configured["seed"] == 39
    assert single_tool_turn_sampling_params(configured, parser)["stop"] == configured["stop"]


def test_hermes_stop_accepts_string_and_none_and_is_idempotent():
    parser = SimpleNamespace(tool_call_end_token="</tool_call>")

    assert single_tool_turn_sampling_params({"stop": "<|im_end|>"}, parser)["stop"] == [
        "<|im_end|>", "</tool_call>"]
    assert single_tool_turn_sampling_params({}, parser)["stop"] == ["</tool_call>"]
    existing = {"stop": "</tool_call>"}
    assert single_tool_turn_sampling_params(existing, parser)["stop"] == ["</tool_call>"]


def test_hermes_stop_capture_setting_is_accepted_as_on_policy():
    from rift.verl.rollout_evidence import unmodified_sampling

    parser = SimpleNamespace(tool_call_end_token="</tool_call>")
    configured = single_tool_turn_sampling_params({
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }, parser)

    assert unmodified_sampling(configured)
    assert not unmodified_sampling({**configured, "include_stop_str_in_output": False})


def test_other_parser_keeps_generation_settings_unchanged():
    parser = SimpleNamespace(tool_call_end_token="<tool_call|>")
    original = {"temperature": 0.4, "stop": ["<|im_end|>"]}

    assert single_tool_turn_sampling_params(original, parser) == original
    assert "include_stop_str_in_output" not in original


@pytest.mark.parametrize("stop", [1, ["valid", ""], ["valid", None], {"stop": "bad"}])
def test_hermes_stop_rejects_invalid_stop_settings(stop):
    parser = SimpleNamespace(tool_call_end_token="</tool_call>")

    with pytest.raises(ValueError, match="vLLM stop"):
        single_tool_turn_sampling_params({"stop": stop}, parser)


def test_protocol_reminder_lists_schema_empty_params_tools_only():
    tools = [
        {"function": {"name": "request_policy_review", "parameters": {
            "type": "object", "properties": {"params": {
                "type": "object", "properties": {}, "additionalProperties": False}}}}},
        {"function": {"name": "search_catalog", "parameters": {
            "type": "object", "properties": {"params": {
                "type": "object", "properties": {"query": {"type": "string"}},
                "additionalProperties": False}}}}},
    ]

    reminder = service_tool_protocol_reminder(tools)

    assert "exactly ONE complete tool call" in reminder
    assert "Never output multiple <tool_call> blocks" in reminder
    assert "params field MUST be the empty object {}: request_policy_review." in reminder
    assert "search_catalog" not in reminder


def test_protocol_reminder_matches_native_service_tool_schemas():
    from usami_v2.runtime.agents import service_tools

    reminder = service_tool_protocol_reminder(service_tools())

    assert "get_case_status" in reminder
    assert "request_policy_review" in reminder
    assert "search_catalog" not in reminder
