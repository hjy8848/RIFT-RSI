"""Versioned generator routing. Never dispatches an old seed to a new distribution."""
import importlib
from usami_harness.models import digest
from .contracts import WorldSpec


GENERATORS={'education':'education-generator-v1','property':'property-generator-v1','airline':'airline-generator-v1','aftersales':'aftersales-generator-v1','cloudops':'cloudops-structural-v1','fulfillment':'fulfillment-structural-v1','stayops':'stayops-structural-v2','saas':'saas-generator-0.2.0'}
RULES={'education':'education-0.1.0','property':'property-0.1.0','airline':'airline-0.1.0','aftersales':'aftersales-0.1.0','cloudops':'cloudops-0.1.0','fulfillment':'fulfillment-0.1.0','stayops':'stayops-0.2.0','saas':'saas-rules-0.2.0'}


def default_spec(domain,seed,partition='train',config=None):
    if domain not in GENERATORS:raise ValueError('domain generator is not registered: '+domain)
    return WorldSpec(domain=domain,seed=seed,generator_version=GENERATORS[domain],rules_version=RULES[domain],generator_config=config or {},partition=partition)


def generate(spec,*,diagnostics=None):
    spec=WorldSpec.model_validate(spec.model_dump(mode='json') if isinstance(spec,WorldSpec) else spec)
    if spec.domain in {'aftersales','airline','property','education'} and spec.generator_version==spec.domain+'-lifecycle-v2' and spec.rules_version==RULES[spec.domain]:
        from .generators.dialogue_history import generate_world
        return generate_world(spec.domain,spec.seed,spec.generator_config,diagnostics=diagnostics)
    if spec.domain=='stayops' and (spec.generator_version,spec.rules_version)==('stayops-structural-v2','stayops-0.2.0'):
        from .generators.stayops_v2 import generate_world
        return generate_world(spec.seed,spec.generator_config,diagnostics=diagnostics)
    if spec.domain=='stayops' and (spec.generator_version,spec.rules_version)==('stayops-structural-v1','stayops-0.1.0'):
        from .generators.stayops import generate_world
        return generate_world(spec.seed,spec.generator_config,diagnostics=diagnostics)
    if spec.domain not in GENERATORS or spec.generator_version!=GENERATORS[spec.domain] or spec.rules_version!=RULES[spec.domain]:raise ValueError('unregistered generator/rules version')
    if spec.domain=='saas':
        from usami.world.generator import GeneratorConfig,generate_world
        from usami.world.models import RulesConfig
        raw=dict(spec.generator_config)
        if raw.get('workflow') is not None:raise ValueError('Discovery registration forbids task conditioning')
        if 'version' in raw and raw['version']!=spec.generator_version:raise ValueError('generator version conflict')
        if 'rules' in raw and raw['rules']!=RulesConfig(version=spec.rules_version).model_dump(mode='json'):raise ValueError('rules config must be registered separately')
        raw.update(version=spec.generator_version,rules=RulesConfig(version=spec.rules_version))
        return generate_world(spec.seed,GeneratorConfig.model_validate(raw))
    module='usami_'+spec.domain+'.generator' if spec.domain in {'aftersales','airline','property','education'} else 'usami_benchmark.generators.'+spec.domain
    return importlib.import_module(module).generate_world(spec.seed,spec.generator_config,diagnostics=diagnostics)


def invariants(domain,world):
    if domain=='stayops' and world.version=='stayops-0.2.0':
        from .native.stayops_v2.rules import invariants as check
        return check(world)
    module=importlib.import_module('usami.world.rules' if domain=='saas' else 'usami_'+domain+'.rules')
    return (module.check_invariants if domain=='saas' else module.invariants)(world)


def canonical_world(spec):
    world=generate(spec)
    if invariants(spec.domain,world):raise ValueError('invalid generated world')
    return world.model_dump(mode='json'),digest(world)


def source_modules(spec):
    if spec.domain in {'aftersales','airline','property','education'} and spec.generator_version==spec.domain+'-lifecycle-v2':
        return importlib.import_module('usami_benchmark.generators.dialogue_history'),importlib.import_module('usami_'+spec.domain+'.rules')
    if spec.domain in {'aftersales','airline','property','education'}:
        return importlib.import_module('usami_'+spec.domain+'.generator'),importlib.import_module('usami_'+spec.domain+'.rules')
    if spec.domain=='stayops' and spec.rules_version=='stayops-0.2.0':
        return importlib.import_module('usami_benchmark.generators.stayops_v2'),importlib.import_module('usami_benchmark.native.stayops_v2.rules')
    return (importlib.import_module('usami.world.generator_v2' if spec.domain=='saas' else 'usami_benchmark.generators.'+spec.domain),
            importlib.import_module('usami.world.rules_v2' if spec.domain=='saas' else 'usami_'+spec.domain+'.rules'))
