"""Private customer facts are separate from Service-visible business state."""
from typing import Literal,Optional
from pydantic import ConfigDict,Field,model_validator
from usami_harness.models import Model,WorldBase,digest


class CustomerPrivateState(Model):
    model_config=ConfigDict(extra='forbid',frozen=True)
    identity: str
    opening_request: str=Field(min_length=1)
    facts: dict
    preference_space: dict[str,list]
    initial_preferences: dict=Field(default_factory=dict)
    confirmable_actions: tuple[str,...]
    protected_objects: tuple[str,...]
    private_hash: str=''
    @model_validator(mode='after')
    def preference_values(self):
        if not self.confirmable_actions or not self.protected_objects:raise ValueError('customer needs explicit action/object scope')
        if any(k not in self.preference_space or v not in self.preference_space[k] for k,v in self.initial_preferences.items()):raise ValueError('invalid initial preference')
        return self
    def checked(self):
        value=digest(self.model_dump(mode='json',exclude={'private_hash'}))
        if self.private_hash and self.private_hash!=value:raise ValueError('customer private state was mutated')
        return self.model_copy(update={'private_hash':value})


class ConsentRecord(Model):
    operation: dict
    revision: int=Field(strict=True,ge=0)
    quote_hash: str
    world_hash: str
    turn_id: str
    statement: str=Field(min_length=1)
    @property
    def operation_hash(self):return digest(self.operation)


class InteractionState(Model):
    identity: str=''
    known_facts: dict=Field(default_factory=dict)
    preferences: dict=Field(default_factory=dict)
    revision: int=Field(default=0,strict=True,ge=0)
    consents: list[ConsentRecord]=Field(default_factory=list)
    commitments: dict=Field(default_factory=dict)
    @property
    def state_hash(self):return digest(self)


class DialogueWorldBase(WorldBase):
    interaction: InteractionState=Field(default_factory=InteractionState)


class CustomerIntent(Model):
    message: str=Field(min_length=1,max_length=8000)
    disclosures: dict=Field(default_factory=dict)
    preferences: dict=Field(default_factory=dict)
    requested_operations: list[dict]=Field(default_factory=list,max_length=32)
    confirm_operation: Optional[dict]=None
    reject_offer: bool=False
    finish: bool=False


class DialogueEvent(Model):
    id: str
    kind: Literal['customer','service','business','external']
    payload: dict
    before_hash: str
    after_hash: str
    previous_hash: str=''
    event_hash: str=''
    def computed_hash(self):return digest(self.model_dump(mode='json',exclude={'event_hash'}))


class DependencyProof(Model):
    kind: Literal['missing_information','preference_resolution','explicit_consent','state_update','cross_turn_commitment','partial_completion']
    operation: dict
    control_hash: str
    changed_hash: str
    expected_control_code: Optional[str]
    expected_changed_code: Optional[str]
    control_allowed: bool
    changed_allowed: bool
    reason: str
    def checked(self):
        if self.control_hash==self.changed_hash or self.control_allowed==self.changed_allowed:
            raise ValueError('dialogue dependency does not change the actual Reference decision')
        return self
