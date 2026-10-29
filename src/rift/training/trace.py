"""Exact generation records supplied by a future model-serving backend.

No tokenizer or model is invoked here. Hash binding detects mismatched recorded
requests/responses; the injected attestor must establish that tokens and policy
identity were captured at generation time. A hash alone cannot establish that.
"""
from dataclasses import dataclass
import hashlib
import json
from typing import Protocol
from rift.schemas import Checkpoint, Tokens


def digest(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class ExactPolicyTrace:
    role: str
    rollout_id: str
    decision_id: str
    policy: Checkpoint
    tokens: Tokens
    request_digest: str
    response_digest: str
    serialization: str
    evidence_ref: str
    package_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.policy, Checkpoint) or not isinstance(self.tokens, Tokens):
            raise ValueError('checkpoint and token records required')
        if self.role not in ('service', 'designer') or not all((self.rollout_id, self.decision_id, self.request_digest, self.response_digest, self.serialization, self.evidence_ref)):
            raise ValueError('complete exact trace provenance required')
        if self.role == 'designer' and not self.package_id:
            raise ValueError('Designer trace requires package binding')
        for value in (self.request_digest, self.response_digest):
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('request and response digests must be SHA-256')


class CaptureAttestor(Protocol):
    def verify_capture(self, trace: ExactPolicyTrace, request, response) -> bool: ...


class RecordedDecisionEvidenceVerifier:
    """Verify native Service call bindings and delegate token origin to an attestor."""
    def __init__(self, traces, attestor):
        self.traces = tuple(traces)
        self.attestor = attestor
        self._authorized = {}

    def verify_source(self, episode, checkpoint):
        self._authorized = {}
        calls = [c for c in episode.model_calls if c.role == 'service']
        if not calls:
            return False
        pending = {}
        for call in calls:
            matches = [t for t in self.traces if t.role == 'service' and t.rollout_id == episode.record_hash
                       and t.decision_id == call.record_hash and t.policy == checkpoint]
            if len(matches) != 1:
                return False
            trace = matches[0]
            if (trace.request_digest != digest(call.request) or trace.response_digest != digest(call.response)
                    or self.attestor.verify_capture(trace, call.request, call.response) is not True):
                return False
            pending[(call.record_hash, checkpoint)] = trace
        self._authorized = pending
        return True

    def tokens_for_call(self, call, checkpoint):
        trace = self._authorized.get((call.record_hash, checkpoint))
        if (trace is None or trace.request_digest != digest(call.request)
                or trace.response_digest != digest(call.response)
                or self.attestor.verify_capture(trace, call.request, call.response) is not True):
            raise ValueError('call was not verified for this checkpoint')
        return trace.tokens

class RecordedDesignerActionVerifier:
    """Verify a Designer package action against a separately stored call."""
    def __init__(self, traces, call_resolver, attestor):
        self.traces = tuple(traces)
        self.call_resolver = call_resolver
        self.attestor = attestor

    def verify_designer_action(self, tokens, checkpoint, package_id, rollout_id):
        matches = [t for t in self.traces if t.role == 'designer' and t.rollout_id == rollout_id
                   and t.package_id == package_id and t.policy == checkpoint and t.tokens == tokens]
        if len(matches) != 1:
            return False
        trace = matches[0]
        call = self.call_resolver(trace.decision_id)
        if call is None:
            return False
        request, response = call
        return (trace.request_digest == digest(request) and trace.response_digest == digest(response)
                and self.attestor.verify_capture(trace, request, response) is True)
