from dataclasses import dataclass
from enum import Enum

class CertificationStatus(str, Enum):
    CERTIFIED = "CERTIFIED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"

@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    hypothesis: str
    public_request: str
    abstract_spec: str
    ability_id: str = "NEW"
    diagnosis_task_refs: tuple[str, ...] = ()
    remove_failure_task_refs: tuple[str, ...] = ()
    def __post_init__(self):
        if not all((self.proposal_id, self.hypothesis, self.public_request, self.abstract_spec)):
            raise ValueError("proposal fields required")
        if (not self.ability_id or len(set(self.diagnosis_task_refs)) != len(self.diagnosis_task_refs)
                or len(set(self.remove_failure_task_refs)) != len(self.remove_failure_task_refs)
                or any(not ref for ref in self.diagnosis_task_refs + self.remove_failure_task_refs)):
            raise ValueError("explicit ability choice and distinct task references required")

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
