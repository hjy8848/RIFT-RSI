from dataclasses import dataclass
from typing import Protocol, Sequence
from rift.benchmark.public_feedback import PublicEpisode
from rift.schemas import Identity, Split
from rift.tasks import Proposal

@dataclass(frozen=True)
class TrainingExperience:
    identity: Identity
    episode: PublicEpisode
    def __post_init__(self):
        if self.identity.split is not Split.TRAIN or not isinstance(self.episode, PublicEpisode):
            raise ValueError('Designer observations must come from the training split')

class Designer(Protocol):
    def propose(self, experiences: Sequence[PublicEpisode]) -> Sequence[Proposal]: ...

def propose(designer: Designer, experiences, registry):
    """Gate Designer input through the Host's registered training split."""
    if not experiences or any(not isinstance(x, TrainingExperience) or not registry.contains(x.identity) for x in experiences):
        raise ValueError('registered public training experiences required')
    proposals = tuple(designer.propose(tuple(x.episode for x in experiences)))
    if not proposals or any(not isinstance(p, Proposal) for p in proposals):
        raise ValueError('structured proposals required')
    if len({p.proposal_id for p in proposals}) != len(proposals): raise ValueError('proposal IDs must be distinct')
    return proposals

class Service(Protocol):
    """Native episode runner owns execution; a policy is supplied by the caller."""
    def decide(self, prompt): ...
