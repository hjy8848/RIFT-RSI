from dataclasses import dataclass
from rift.schemas import Outcome

@dataclass(frozen=True)
class PublicEvent:
    kind: str
    content: str

@dataclass(frozen=True)
class PublicEpisode:
    events: tuple[PublicEvent, ...]
    outcome: Outcome

def service_reward(outcome: Outcome) -> float | None:
    if outcome is Outcome.SUCCESS: return 1.0
    if outcome is Outcome.FAILURE: return 0.0
    if outcome is Outcome.UNKNOWN: return None
    raise ValueError("unknown outcome type")
