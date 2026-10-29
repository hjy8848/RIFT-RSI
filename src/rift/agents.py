from dataclasses import dataclass, field
from typing import Protocol, Sequence
from rift.benchmark.public_feedback import PublicEpisode
from rift.schemas import Identity, Split
from rift.tasks import Proposal

@dataclass(frozen=True)
class TrainingExperience:
    identity: Identity
    record: object = field(repr=False)
    def __post_init__(self):
        self.check_binding()
    def check_binding(self):
        if (self.identity.split is not Split.TRAIN
                or self.record.task.partition != 'train'
                or self.record.task.record_hash != self.identity.task
                or self.record.initial_world.state_hash != self.identity.world):
            raise ValueError('Designer observations must bind a training episode to its registered identity')

class Designer(Protocol):
    def propose(self, experiences: Sequence[PublicEpisode]) -> Sequence[Proposal]: ...

def propose(designer: Designer, experiences, registry):
    """Gate Designer input through the Host's registered training split."""
    if not experiences or any(not isinstance(x, TrainingExperience) or not registry.contains(x.identity) for x in experiences):
        raise ValueError('registered public training experiences required')
    for item in experiences:
        item.check_binding()
    # Replay and projection are repeated at the boundary; never hand native
    # records, task objects, private scores or provenance metadata to Designer.
    from rift.benchmark.usami import public_episode
    observations = tuple(public_episode(x.record) for x in experiences)
    proposals = tuple(designer.propose(observations))
    if not proposals or any(not isinstance(p, Proposal) for p in proposals):
        raise ValueError('structured proposals required')
    if len({p.proposal_id for p in proposals}) != len(proposals): raise ValueError('proposal IDs must be distinct')
    return proposals

class Service(Protocol):
    """Native episode runner owns execution; a policy is supplied by the caller."""
    def decide(self, prompt): ...
