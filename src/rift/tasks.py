from dataclasses import dataclass
from enum import Enum
import json

class CertificationStatus(str, Enum):
    CERTIFIED = "CERTIFIED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"

@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    hypothesis: str
    public_request: str = ""
    abstract_spec: str = ""
    ability_id: str = "NEW"
    diagnosis_task_refs: tuple[str, ...] = ()
    remove_failure_task_refs: tuple[str, ...] = ()
    world_draft: object | None = None
    challenges: tuple[object, ...] = ()
    def __post_init__(self):
        if not self.proposal_id or not self.hypothesis or not self.ability_id:
            raise ValueError("proposal ID, hypothesis and explicit ability choice required")
        if (not self.ability_id or len(set(self.diagnosis_task_refs)) != len(self.diagnosis_task_refs)
                or len(set(self.remove_failure_task_refs)) != len(self.remove_failure_task_refs)
                or any(not ref for ref in self.diagnosis_task_refs + self.remove_failure_task_refs)):
            raise ValueError("explicit ability choice and distinct task references required")
        if self.world_draft is None and not self.challenges:
            if not self.public_request or not self.abstract_spec:
                raise ValueError("legacy single-task proposals require a request and abstract challenge")
            return
        if self.world_draft is None or not self.challenges or self.public_request or self.abstract_spec:
            raise ValueError("a training-plan Proposal contains one native World draft and its challenges")

        # Store JSON-compatible forms of the pinned Usami contracts. This keeps
        # capture files inert and portable while still validating the exact
        # native schemas at the Designer boundary.
        from usami_v2.scenarios.designer import ScenarioDraft
        from usami_v2.scenarios.grounding import AbstractChallenge

        def payload(value):
            if isinstance(value, str):
                value = json.loads(value)
            elif hasattr(value, "model_dump"):
                value = value.model_dump(mode="json", exclude={"record_hash"})
            if not isinstance(value, dict):
                raise ValueError("native Usami output must be a structured object")
            return value

        world = ScenarioDraft.model_validate(payload(self.world_draft)).checked()
        challenges = tuple(AbstractChallenge.model_validate(payload(row)).checked()
                           for row in self.challenges)
        task_ids = [row.task_id for row in challenges]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("training-plan challenges need distinct native task IDs")
        object.__setattr__(self, "world_draft",
                           world.model_dump(mode="json", exclude={"record_hash"}))
        object.__setattr__(self, "challenges", tuple(
            row.model_dump(mode="json", exclude={"record_hash"}) for row in challenges))

    @property
    def is_training_plan(self):
        return self.world_draft is not None

@dataclass(frozen=True)
class Certification:
    proposal_id: str
    status: CertificationStatus
    task_ref: str | None = None
    def __post_init__(self):
        if not self.proposal_id or not isinstance(self.status, CertificationStatus):
            raise ValueError("invalid certification")
        if (self.status == CertificationStatus.CERTIFIED) != bool(self.task_ref):
            raise ValueError("only certified tasks may be bound")
