"""Business-only reward boundary for trusted Usami episode evidence."""
from rift.schemas import Outcome
from rift.benchmark.public_feedback import service_reward

class UnknownBusinessOutcome(ValueError):
    pass

def verified_business_reward(episode, verify_episode):
    """The callback returns only a replay-verified public Outcome, never Oracle data."""
    outcome = verify_episode(episode)
    if not isinstance(outcome, Outcome): raise ValueError('verified public outcome required')
    value = service_reward(outcome)
    if value is None: raise UnknownBusinessOutcome('UNKNOWN cannot become a numerical training reward')
    return value
