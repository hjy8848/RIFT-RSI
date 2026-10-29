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
    def __post_init__(self):
        if not all((self.proposal_id, self.hypothesis, self.public_request, self.abstract_spec)):
            raise ValueError("proposal fields required")

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
