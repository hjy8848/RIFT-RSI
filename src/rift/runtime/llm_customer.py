"""A language-model Customer for the Service to talk to, in place of the scripted one.

The scripted customer says "Yes, please go ahead." once and then "Thanks, that is all.": it never chats,
changes its mind or holds anything back, so the Service never has to handle a real conversation. Usami
already has a model-backed customer (``usami_v2.runtime.agents.OpenAICustomerAgent``: it plays its private
profile, answers in its own words and may decline); this module only supplies the pieces to run it on
the training hosts:

- ``RIFT_CUSTOMER_URL`` (an OpenAI-compatible API base, for example a local vLLM server) switches it on;
  ``RIFT_CUSTOMER_MODEL`` names the served model. The customer model is frozen: it must never be the
  policy being trained.
- a client that retries transient failures a few times, which the stock client deliberately does not do;
- one customer per episode, seeded from the episode's customer seed.

A customer call that still fails ends the episode as a ``customer_protocol_error``; measure how often
that happens before relying on the results (see docs/PLAN_LLM_CUSTOMER.md).
"""
import os
import time

RETRY_CODES = ('PROVIDER_TIMEOUT', 'PROVIDER_CONNECTION_ERROR')


def llm_customer_enabled(environ=None):
    return bool((os.environ if environ is None else environ).get('RIFT_CUSTOMER_URL'))


def customer_settings(environ=None):
    """Request settings of the customer model from the environment (no credentials in them)."""
    from usami.service.models import ModelSettings
    env = os.environ if environ is None else environ
    options = {'max_tokens': int(env.get('RIFT_CUSTOMER_MAX_TOKENS', '256')),
               'temperature': float(env.get('RIFT_CUSTOMER_TEMPERATURE', '0.7')),
               'top_p': 1.0,
               # Qwen3 would otherwise think before every reply; the customer only has to answer.
               'chat_template_kwargs': {'enable_thinking': False}}
    return ModelSettings(model=env.get('RIFT_CUSTOMER_MODEL', 'customer'), base_url=env['RIFT_CUSTOMER_URL'],
                         timeout_seconds=float(env.get('RIFT_CUSTOMER_TIMEOUT', '60')), options=options)


def retrying_client(settings, api_key='', *, transport=None, attempts=3, pause=1.0, sleep=time.sleep):
    """The stock OpenAI-compatible client plus bounded retries of timeouts, connection errors and HTTP 5xx."""
    from usami.service.client import OpenAICompatibleClient
    from usami.service.models import ServiceError

    class RetryingClient(OpenAICompatibleClient):
        def complete(self, request):
            for attempt in range(attempts):
                try:
                    return super().complete(request)
                except ServiceError as exc:
                    status = (exc.response or {}).get('status_code') if isinstance(exc.response, dict) else None
                    transient = exc.code in RETRY_CODES or (exc.code == 'PROVIDER_HTTP_ERROR' and status is not None
                                                            and status >= 500)
                    if not transient or attempt == attempts - 1:
                        raise
                    sleep(pause * (attempt + 1))
    return RetryingClient(settings, api_key, transport=transport)


class LLMCustomerFactory:
    """Same ``create(task_ref, seed=...)`` contract as the scripted factory; one model customer per episode."""

    def __init__(self, client):
        self.client = client

    def create(self, task_ref, *, seed):
        if not task_ref or type(seed) is not int or seed < 0:
            raise ValueError('certified task and non-negative customer seed required')
        from usami_v2.runtime.agents import OpenAICustomerAgent
        agent = OpenAICustomerAgent(self.client)
        # Same episode seed, same sampling stream: the customer's wording is reproducible given the Service's.
        options = {**agent.settings.options, 'seed': seed % (2 ** 31)}
        agent.settings = agent.settings.model_copy(update={'options': options})
        return agent


def build_customer_factory(scripted_factory, environ=None):
    """The model customer when ``RIFT_CUSTOMER_URL`` is set, otherwise the scripted one."""
    if not llm_customer_enabled(environ):
        return scripted_factory
    return LLMCustomerFactory(retrying_client(customer_settings(environ)))
