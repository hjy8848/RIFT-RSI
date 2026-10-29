"""Native function interfaces with inline operation-specific parameter schemas."""
import copy
from usami.service.models import strict_json


def inline_schema(model):
    root = model.model_json_schema()
    definitions = root.get('$defs', {})
    def expand(value):
        if isinstance(value, list):
            return [expand(v) for v in value]
        if not isinstance(value, dict):
            return value
        if '$ref' in value:
            ref = value['$ref']
            if not ref.startswith('#/$defs/'):
                raise ValueError('unsupported external schema reference')
            return expand(copy.deepcopy(definitions[ref.split('/')[-1]]))
        return {k: expand(v) for k, v in value.items() if k not in {'$defs', 'title'}}
    return expand(root)


def operation_shape(adapter, name, customer=False):
    properties = {'scope_id': {'type': 'string', 'minLength': 1},
                  'resource_id': {'type': 'string', 'minLength': 1},
                  'params': inline_schema(adapter.PARAMETERS[name])}
    if customer:
        properties['message'] = {'type': 'string', 'minLength': 1}
    return {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}


def schemas(adapter, names):
    if len(names) != len(set(names)) or set(names) - set(adapter.TOOLS):
        raise ValueError('tool set differs from the domain interface')
    return [{'type': 'function', 'function': {'name': name,
             'description': adapter.DESCRIPTIONS[name],
             'parameters': ({'type': 'object', 'properties': {}, 'additionalProperties': False}
                            if name == 'get_customer_context' else operation_shape(adapter, name))}}
            for name in names]


def parse(adapter, name, arguments, names):
    if name not in names:
        raise ValueError('tool absent from frozen domain interface')
    args = strict_json(arguments)
    if name == 'get_customer_context':
        if args != {}:
            raise ValueError('context tool accepts an empty object only')
        return None
    if not isinstance(args, dict) or set(args) != {'scope_id', 'resource_id', 'params'}:
        raise ValueError('expected scope_id, resource_id and a typed params object')
    return adapter.Operation(name=name, **args)


def customer_schemas(adapter, names):
    result = []
    for name in names:
        if name in adapter.PARAMETERS and name not in adapter.Operation.queries:
            result.append({'type': 'function', 'function': {
                'name': 'intend_' + name,
                'description': 'Declare an exact customer intention before Service acts. ' + adapter.DESCRIPTIONS[name],
                'parameters': operation_shape(adapter, name, customer=True)}})
    for name, field, description in [
        ('ask_service', 'message', 'Ask a question or discuss a request without granting any write.'),
        ('stop_exploration', 'reason', 'Voluntarily end exploration; no business operation occurs.')]:
        result.append({'type': 'function', 'function': {'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': {field: {'type': 'string', 'minLength': 1}},
                           'required': [field], 'additionalProperties': False}}})
    return result
