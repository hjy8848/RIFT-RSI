from dataclasses import dataclass
from rift.schemas import Identity, Outcome, Split, Tokens

@dataclass(frozen=True)
class Demonstration:
    sample_id: str
    identity: Identity
    episode_id: str
    source_checkpoint: str
    tokens: Tokens
    outcome: Outcome
    replay_verified: bool
    evidence_kind: str
    def __post_init__(self):
        if not all((self.sample_id, self.episode_id, self.source_checkpoint)):
            raise ValueError("missing replay provenance")
        if self.identity.split is not Split.REPLAY or self.outcome is not Outcome.SUCCESS or self.replay_verified is not True or self.evidence_kind != "real_model":
            raise ValueError("replay needs independently verified real-model business success")

class DecisionEvidenceVerifier:
    """Trusted future capture boundary; implementation must verify provenance and visibility.

    verify_source checks a real model rollout and exact checkpoint. tokens_for_call
    returns the exact generated tokens only after verifying request serialization,
    tokenizer, public visibility, response alignment and action mask against the
    original generation record. Missing evidence must raise, never reconstruct it.
    """
    def verify_source(self, episode, checkpoint):
        raise NotImplementedError
    def tokens_for_call(self, call, checkpoint):
        raise NotImplementedError

def extract_demonstrations(record, identity, checkpoint, registry, verifier):
    """Host-only extraction: requires native replay and a trusted token recorder."""
    from usami_v2.runtime.runner import replay_episode
    from rift.benchmark.private_evaluation import public_business_outcome
    if identity.split is not Split.REPLAY or not registry.contains(identity):
        raise ValueError("registered replay identity required")
    if identity.world != record.initial_world.state_hash or identity.task != record.task.record_hash:
        raise ValueError("episode does not match registered world and task")
    verification = replay_episode(record)
    if not verification.verified or public_business_outcome(record, verification.verified) is not Outcome.SUCCESS:
        raise ValueError("verified business success required")
    calls = [c for c in record.model_calls if c.role == 'service']
    if not calls or verifier.verify_source(record, checkpoint) is not True:
        raise ValueError("real Service model provenance required")
    rows = []
    for call in calls:
        tokens = verifier.tokens_for_call(call, checkpoint)
        if not isinstance(tokens, Tokens): raise ValueError("verified tokens required")
        rows.append(Demonstration(call.record_hash, identity, record.record_hash, checkpoint.weights_digest, tokens, Outcome.SUCCESS, True, 'real_model'))
    return tuple(rows)
