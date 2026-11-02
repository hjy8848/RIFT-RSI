"""Bind a Probe protocol to its actual Service tools and system prompt."""
from hashlib import sha256
import json


def service_profile_descriptor(tool_profile):
    from usami_v2.runtime.agents import OpenAIServiceAgent, service_tools
    from rift.verl.usami_session import service_tool_protocol_reminder
    tools = service_tools(tool_profile)
    system = OpenAIServiceAgent.system_for_profile(tool_profile) + '\n\n' + service_tool_protocol_reminder(tools)
    encoded_tools = json.dumps(tools, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return {'tool_profile': tool_profile, 'system_prompt_sha256': sha256(system.encode('utf-8')).hexdigest(),
            'tool_schema_sha256': sha256(encoded_tools.encode('utf-8')).hexdigest()}


def bind_service_protocol(protocol, tool_profile):
    if not isinstance(protocol, dict):
        raise TypeError('evaluation protocol must be a mapping')
    descriptor = service_profile_descriptor(tool_profile)
    if 'service_profile' in protocol and protocol['service_profile'] != descriptor:
        raise ValueError('protocol already binds a different Service profile')
    return {**protocol, 'service_profile': descriptor}


def service_protocol_digest(protocol, tool_profile):
    from rift.banks.bootstrap import _protocol_digest
    return _protocol_digest(bind_service_protocol(protocol, tool_profile))
