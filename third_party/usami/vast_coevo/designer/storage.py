"""Normalize nested Pydantic values before using existing immutable storage."""
from pydantic import BaseModel
from vast_coevo.schema import atomic_json as _atomic_json, digest as _digest


def json_value(value):
    if isinstance(value,BaseModel):return value.model_dump(mode='json')
    if isinstance(value,dict):return {key:json_value(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):return [json_value(item) for item in value]
    return value


def atomic_json(path,value,immutable=False):return _atomic_json(path,json_value(value),immutable=immutable)
def digest(value):return _digest(json_value(value))
