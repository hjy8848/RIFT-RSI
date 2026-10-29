"""Source registration is independent of tasks, actors and observed performance."""
from typing import Literal
from pydantic import Field, model_validator
from usami_harness.models import Model, digest

DOMAINS=('saas','cloudops','fulfillment','stayops','aftersales','airline','property','education')
TRACKS={d:('A' if i<4 else 'B') for i,d in enumerate(DOMAINS)}


class GeneratorConfig(Model):
    version: Literal['structural-generator-v1']='structural-generator-v1'
    min_scopes: int=Field(default=2,strict=True,ge=2,le=5)
    max_scopes: int=Field(default=4,strict=True,ge=2,le=8)
    min_resources: int=Field(default=2,strict=True,ge=1,le=4)
    max_resources: int=Field(default=5,strict=True,ge=2,le=8)
    lifecycle_steps: int=Field(default=9,strict=True,ge=0,le=12)
    @model_validator(mode='after')
    def bounds(self):
        if self.min_scopes>self.max_scopes or self.min_resources>self.max_resources:raise ValueError('inverted generator bounds')
        return self


class WorldSpec(Model):
    domain: Literal['saas','cloudops','fulfillment','stayops','aftersales','airline','property','education']
    seed: int=Field(strict=True)
    generator_version: str
    rules_version: str
    generator_config: dict=Field(default_factory=dict)
    partition: Literal['train','validation','test_demo','heldout']='train'
    generation_mode: Literal['unconditioned']='unconditioned'
    @property
    def spec_hash(self):return digest(self)
