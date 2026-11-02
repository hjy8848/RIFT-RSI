import asyncio
from types import SimpleNamespace

import pytest

from rift.runtime.service_protocol import service_profile_descriptor, service_protocol_digest
from rift.verl.host_provider import NativeHostProvider
from rift.verl.usami_session import NativeUsamiSession
from usami_v2.runtime.agents import OpenAIServiceAgent, service_tools
from usami_v2.runtime.runner import EpisodeBudget, replay_episode


def test_profile_tools_prompts_and_protocols_are_distinct():
    four = service_tools('core4')
    nine = service_tools('full9')
    assert len(four) == 4 and len(nine) == 9
    assert {row['function']['name'] for row in four} == {
        'list_customer_orders', 'get_order_details', 'return_delivered_order_items', 'respond_to_customer'}
    assert OpenAIServiceAgent.system_for_profile('full9') == OpenAIServiceAgent.SYSTEM
    assert 'cancel_order' not in OpenAIServiceAgent.system_for_profile('core4')
    assert service_profile_descriptor('core4')['system_prompt_sha256'] != service_profile_descriptor('full9')['system_prompt_sha256']
    assert service_protocol_digest({'budget': 4}, 'core4') != service_protocol_digest({'budget': 4}, 'full9')


def test_core_profile_rejects_unavailable_task_before_certification():
    from test_families import host_for
    from rift.benchmark.families import FAMILIES
    instance = FAMILIES['F6'].instance({}, 92)
    host = host_for(instance)
    host.tool_profile = 'core4'
    host.capability_options = {'tool_profile': 'core4'}
    result = host.certify_training_plan(instance.proposal(), world_seed=92)
    assert not result.tasks
    assert result.challenge_results[0]['attempts'][0]['reason_code'] == 'TOOL_PROFILE_UNSUPPORTED'
    refund = FAMILIES['F1'].instance({}, 93)
    host = host_for(refund)
    host.tool_profile = 'core4'
    host.capability_options = {'tool_profile': 'core4'}
    assert len(host.certify_training_plan(refund.proposal(), world_seed=93).tasks) == 1
    for family in ('F5', 'F8', 'F9'):
        instance = FAMILIES[family].instance({}, 94)
        host = host_for(instance)
        host.tool_profile = 'core4'
        host.capability_options = {**(host.capability_options or {}), 'tool_profile': 'core4'}
        result = host.certify_training_plan(instance.proposal(), world_seed=94)
        assert not result.tasks
        assert result.challenge_results[0]['attempts'][0]['reason_code'] == 'TOOL_PROFILE_UNSUPPORTED'


def test_core_profile_rejects_full_profile_probe_digest():
    provider = NativeHostProvider(object(), object(), object(), object(), object(), object(),
        budget=object(), parser_format='hermes', tool_profile='core4', evaluation_protocol={'budget': 4})
    with pytest.raises(ValueError, match='active Service tool profile'):
        asyncio.run(provider.open_probe('task', object(), 1, 'protocol', None,
            protocol_digest=service_protocol_digest({'budget': 4}, 'full9')))


def test_core_profile_session_rejects_unexposed_tool_even_if_parser_returns_it():
    from rift_usami_fixtures import frozen, refund_task, refund_profile, ConfirmingCustomer
    world, aliases = frozen()
    class Parser:
        async def extract_tool_calls(self, token_ids, schemas):
            return '', [SimpleNamespace(name='cancel_order', arguments='{"order_id":"x"}', tool_call_id='call')]
    async def run():
        session = NativeUsamiSession(world, refund_task(world, aliases), refund_profile(aliases),
            ConfirmingCustomer(), Parser(), budget=EpisodeBudget(), tool_profile='core4')
        try:
            messages, tools = await session.start()
            assert len(tools) == 4 and 'cancel_order' not in messages[0]['content']
            _, _, done = await session.advance([1], [2], {'temperature': 1.0})
            assert done
            episode = await session.finish()
            assert episode.status == 'service_protocol_error'
            assert not episode.tool_executions and replay_episode(episode).verified
            assert episode.model_calls[0].request['tool_profile'] == 'core4'
            assert episode.model_calls[0].request['system_prompt_sha256'] == service_profile_descriptor('core4')['system_prompt_sha256']
        finally:
            await session.abort()
    asyncio.run(run())


def test_profile_is_forwarded_into_service_runtime_factory():
    from rift.runtime.resources import ExclusivePolicyResources
    from rift.schemas import Checkpoint
    checkpoint = Checkpoint('service', '0', 'tokenizer', 'weights')
    seen = []
    class Runtime:
        role = 'service'
        def __init__(self): self.checkpoint = checkpoint
        def verify_loaded_state(self, expected): return expected == checkpoint
        def close(self): pass
    def factory(cp, *, tool_profile, evaluation_protocol):
        seen.append((cp, tool_profile, evaluation_protocol))
        return Runtime()
    resources = ExclusivePolicyResources(factory, lambda _: None)
    resources.configure_service_profile('core4', {'budget': 4})
    with resources.acquire('service', checkpoint):
        pass
    assert seen == [(checkpoint, 'core4', {'budget': 4})]


def test_profile_config_requires_matching_worker_manifest(tmp_path):
    import json
    from rift.runtime.ascend_910b3 import Ascend910B3RoleRuntime
    from rift.schemas import Checkpoint
    manifest = tmp_path / 'worker.json'
    manifest.write_text(json.dumps({'tool_profile': 'full9'}), encoding='utf-8')
    with pytest.raises(ValueError, match='worker manifest'):
        Ascend910B3RoleRuntime('service', Checkpoint('service', '0', 'tok', 'weights'),
            model_path=tmp_path, run_root=tmp_path, checkpoint_store=object(), serving_attestor=object(),
            open_agent_loop=True, agent_loop_config='loop.yaml', worker_manifest_path=manifest,
            tool_profile='core4', evaluation_protocol={'budget': 4})


def test_model_customer_decline_is_expressible_in_actual_tool_schema():
    import json
    from usami.service.models import ModelSettings
    from usami_v2.runtime.agents import CustomerPrompt, OpenAICustomerAgent
    from usami_v2.runtime.customer import CustomerProfileV3
    class Client:
        settings = ModelSettings(model='fake', base_url='https://example.invalid/v1')
        def complete(self, request):
            self.request = request
            args = {'message': 'No.', 'finish': True, 'declines_proposal': True}
            return {'choices': [{'finish_reason': 'tool_calls', 'message': {'role': 'assistant',
                'tool_calls': [{'id': 'one', 'type': 'function', 'function': {
                    'name': 'customer_turn', 'arguments': json.dumps(args)}}]}}]}
    client = Client()
    profile = CustomerProfileV3(customer_id='customer', intent_bundle_hash='a' * 64,
                                accepts_proposals=False)
    decision = OpenAICustomerAgent(client).decide(CustomerPrompt(public_request='Please help.'), profile)
    assert decision.declines_proposal is True
    schema = client.request['tools'][0]['function']['parameters']
    assert schema['properties']['declines_proposal']['type'] == 'boolean'
    assert 'declines_proposal' in schema['required']
