"""Frozen customer knowledge and preferences; no evaluator answers or workflow."""
import copy
from typing import Literal

from pydantic import ConfigDict, Field, JsonValue, model_validator
from usami_harness.models import Model, digest


class UserScenario(Model):
    model_config = ConfigDict(extra='forbid', frozen=True)
    protocol: Literal['aftersales-user-scenario-v1'] = 'aftersales-user-scenario-v1'
    scenario_id: str = Field(min_length=1)
    version: Literal[1] = 1
    domain: Literal['aftersales'] = 'aftersales'
    identity: str
    profile_id: str
    world_hash: str
    task_hash: str
    profile_hash: str
    reason_for_call: str = Field(min_length=1, max_length=6000)
    known_info: dict[str, JsonValue]
    # Labels only. Never attach the unknown values to a simulator request.
    unknown_info: tuple[str, ...] = ('internal_policy', 'approval_outcome', 'reference_answer', 'oracle_result')
    preference_space: dict[str, tuple[JsonValue, ...]]
    initial_preferences: dict[str, JsonValue]
    follow_up_requests: tuple[str, ...] = Field(default=(), max_length=16,
                                                 exclude_if=lambda value: not value)
    scenario_hash: str = ''

    @model_validator(mode='after')
    def seal(self):
        if set(self.known_info) & set(self.unknown_info):
            raise ValueError('known and unknown information overlap')
        if any(not values for values in self.preference_space.values()):
            raise ValueError('scenario preference space cannot be empty')
        if any(key not in self.preference_space or value not in self.preference_space[key]
               for key, value in self.initial_preferences.items()):
            raise ValueError('initial preference outside scenario')
        expected = digest(self.model_dump(mode='json', exclude={'scenario_hash'}))
        if self.scenario_hash and self.scenario_hash != expected:
            raise ValueError('user scenario hash differs')
        object.__setattr__(self, 'scenario_hash', expected)
        return self

    def checked(self):
        # Revalidation catches mutations of nested dicts, including after a
        # caller used model_copy(update=...) without normal Pydantic validation.
        return type(self).model_validate(self.model_dump(mode='json'))


def scenario_from_task(adapter, world, task):
    if adapter.name != 'aftersales' or task.composition is None:
        raise ValueError('scenario simulator requires an After-sales composed task')
    if task.composition.profile_ids != [task.profile_id] or task.world_hash != world.state_hash:
        raise ValueError('scenario must bind one frozen identity and world')
    private = adapter.private_state(world, task.profile_id)
    adapter.validate_private(world, private)
    if task.private_hash != private.private_hash or task.task_hash != task.computed_hash():
        raise ValueError('scenario task/profile integrity differs')
    preferences = copy.deepcopy(private.preference_space)
    for requirement in task.composition.preference_requirements:
        if requirement.profile_id != task.profile_id:
            raise ValueError('scenario preference crosses profile scope')
        values = preferences.get(requirement.preference, [])
        preferences[requirement.preference] = [v for v in values if v in requirement.accepted_values]
    initial = {key: copy.deepcopy(value) for key, value in private.initial_preferences.items()
               if value in preferences.get(key, [])}
    binding = dict(world_hash=world.state_hash, task_hash=task.task_hash,
                   profile_id=task.profile_id, profile_hash=private.private_hash)
    return UserScenario(scenario_id='aftersales_' + digest(binding)[:24], identity=private.identity,
        reason_for_call=task.composition.customer_request, known_info=copy.deepcopy(private.facts),
        preference_space=preferences, initial_preferences=initial,
        follow_up_requests=tuple(row.message for row in task.composition.follow_up_requests), **binding)


def validate_scenario(adapter, world, task, scenario):
    scenario = UserScenario.model_validate(scenario.model_dump(mode='json'))
    if scenario != scenario_from_task(adapter, world, task):
        raise ValueError('scenario differs from frozen task/profile projection')
    return scenario
