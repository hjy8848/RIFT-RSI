import json
import sys
import unittest

sys.path[:0] = ['src', 'third_party/usami']

import httpx

from rift.runtime import llm_customer as L
from usami.service.models import ServiceError
from usami_v2.runtime.agents import CustomerPrompt, OpenAICustomerAgent


def completion(arguments, name='customer_turn'):
    return {'choices': [{'message': {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': 'call_1', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}]},
        'finish_reason': 'tool_calls'}], 'usage': {'prompt_tokens': 50, 'completion_tokens': 12}}


class Profile:
    """Just enough profile for the adapter: it only reads these."""
    accepts_proposals = True
    decline_line_ids = ()

    def checked(self):
        return self

    def model_dump(self, mode='json'):
        return {'accepts_proposals': True}


def prompt():
    return CustomerPrompt(public_request='I want a refund for the watch.', last_service_message='Shall I refund it?')


ENV = {'RIFT_CUSTOMER_URL': 'http://127.0.0.1:8100/v1', 'RIFT_CUSTOMER_MODEL': 'customer'}


class SettingsTests(unittest.TestCase):
    def test_off_unless_a_url_is_given(self):
        self.assertFalse(L.llm_customer_enabled({}))
        scripted = object()
        self.assertIs(L.build_customer_factory(scripted, {}), scripted)

    def test_settings_carry_no_credentials_and_turn_thinking_off(self):
        settings = L.customer_settings(ENV)
        self.assertEqual((settings.model, settings.base_url), ('customer', 'http://127.0.0.1:8100/v1'))
        self.assertFalse(settings.options['chat_template_kwargs']['enable_thinking'])
        self.assertFalse(settings.trust_env)                       # the lab proxy variables are never used


class CustomerTests(unittest.TestCase):
    def _factory(self, handler, **kwargs):
        client = L.retrying_client(L.customer_settings(ENV), transport=httpx.MockTransport(handler),
                                   sleep=lambda seconds: None, **kwargs)
        return L.LLMCustomerFactory(client)

    def test_the_model_customer_answers_through_the_tool_call_and_records_the_call(self):
        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json=completion({'message': 'Yes, please refund it.', 'finish': False}))
        agent = self._factory(handler).create('task-1', seed=7)
        decision = agent.decide(prompt(), Profile())
        self.assertEqual((decision.message, decision.finish), ('Yes, please refund it.', False))
        body = seen[0]
        self.assertEqual(body['seed'], 7)
        self.assertEqual(body['tool_choice']['function']['name'], 'customer_turn')
        self.assertIn('PRIVATE CUSTOMER PROFILE', body['messages'][0]['content'])
        self.assertIsNotNone(agent.last_call)                        # the runner needs it for replay
        self.assertTrue(agent.uses_model)

    def test_each_episode_gets_its_own_seed(self):
        seeds = []
        factory = self._factory(lambda request: (seeds.append(json.loads(request.content)['seed']),
                                                 httpx.Response(200, json=completion({'message': 'ok', 'finish': True})))[1])
        for seed in (3, 4):
            factory.create('t', seed=seed).decide(prompt(), Profile())
        self.assertEqual(seeds, [3, 4])
        with self.assertRaises(ValueError):
            factory.create('t', seed=-1)

    def test_transient_failures_are_retried_but_a_bad_answer_is_not(self):
        calls = []

        def flaky(request):
            calls.append(1)
            return httpx.Response(503, text='busy') if len(calls) < 3 else httpx.Response(
                200, json=completion({'message': 'ok', 'finish': True}))
        self.assertEqual(self._factory(flaky).create('t', seed=1).decide(prompt(), Profile()).message, 'ok')
        self.assertEqual(len(calls), 3)

        bad = []

        def wrong_tool(request):
            bad.append(1)
            return httpx.Response(200, json=completion({'message': 'ok', 'finish': True}, name='something_else'))
        with self.assertRaises(ServiceError):
            self._factory(wrong_tool).create('t', seed=1).decide(prompt(), Profile())
        self.assertEqual(len(bad), 1)                                # a malformed decision is final

    def test_it_gives_up_after_a_few_attempts(self):
        calls = []
        with self.assertRaises(ServiceError):
            self._factory(lambda request: (calls.append(1), httpx.Response(500, text='down'))[1]).create(
                't', seed=1).decide(prompt(), Profile())
        self.assertEqual(len(calls), 3)

    def test_a_decision_outside_the_contract_is_refused(self):
        factory = self._factory(lambda request: httpx.Response(200, json=completion({'message': 'hi', 'finish': 'maybe', 'extra': 1})))
        with self.assertRaises(ServiceError):
            factory.create('t', seed=1).decide(prompt(), Profile())

    def test_factory_is_picked_by_the_environment(self):
        factory = L.build_customer_factory(object(), ENV)
        self.assertIsInstance(factory, L.LLMCustomerFactory)


if __name__ == '__main__':
    unittest.main()
