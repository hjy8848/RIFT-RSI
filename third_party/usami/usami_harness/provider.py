"""Role-specific environment resolution without hidden protocol rewrites."""
import os
from usami.service.client import settings_from_env, OpenAICompatibleClient
from usami.service.models import strict_json


def provider_environment(role,env=None):
    if role not in {'customer','service'}:
        raise ValueError('unknown provider role')
    env=os.environ if env is None else env
    resolved={k:v for k,v in env.items() if k.startswith(('USAMI_LLM_','OPENAI_'))}
    prefix='USAMI_'+role.upper()+'_LLM_'
    for key,value in env.items():
        if key.startswith(prefix):resolved['USAMI_LLM_'+key[len(prefix):]]=value
    if env.get(prefix+'KEYCHAIN_SERVICE') and not env.get(prefix+'API_KEY'):
        resolved.pop('USAMI_LLM_API_KEY',None);resolved.pop('OPENAI_API_KEY',None)
    options=strict_json(resolved.get('USAMI_LLM_OPTIONS','{"max_tokens":4096,"temperature":0}'))
    if not isinstance(options,dict):raise ValueError('provider options require an object')
    if role=='customer' and 'response_format' in options:
        raise ValueError('Customer protocol uses native function tools; remove response_format before preregistration')
    return resolved


def model_settings(role,env=None):
    return settings_from_env(provider_environment(role,env))


def client(role,env=None):
    return OpenAICompatibleClient.from_env(provider_environment(role,env))
