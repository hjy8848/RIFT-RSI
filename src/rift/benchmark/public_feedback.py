from dataclasses import dataclass
from rift.schemas import Outcome

@dataclass(frozen=True)
class PublicEvent:
    kind: str
    content: str

@dataclass(frozen=True)
class PublicSummary:
    events: tuple[PublicEvent, ...]
    outcome: Outcome

# Compatibility for callers using the previous name. This is a summary, not
# a model prompt or an exact token-level policy trace.
PublicEpisode = PublicSummary

def service_reward(outcome: Outcome) -> float | None:
    if outcome is Outcome.SUCCESS: return 1.0
    if outcome is Outcome.FAILURE: return 0.0
    if outcome is Outcome.UNKNOWN: return None
    raise ValueError("unknown outcome type")
