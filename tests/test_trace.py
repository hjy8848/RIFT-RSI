import pytest
from types import SimpleNamespace
from dataclasses import replace
from rift.schemas import Checkpoint, Tokens
from rift.training.trace import ExactPolicyTrace, RecordedDecisionEvidenceVerifier, RecordedDesignerActionVerifier, digest
from rift.benchmark.public_feedback import PublicSummary


def test_exact_capture_binding_is_distinct_from_summary():
    policy = Checkpoint('service', 'v1', 'tokenizer-v1', 'weights')
    tokens = Tokens((1, 2), (3, 4), (True, False))
    call = SimpleNamespace(role='service', record_hash='call1', request={'messages': ['public']}, response={'action': 'query'})
    episode = SimpleNamespace(record_hash='episode1', model_calls=[call])
    trace = ExactPolicyTrace('service', 'episode1', 'call1', policy, tokens,
        digest(call.request), digest(call.response), 'chat-template-v1', 'backend-evidence')
    class FakeAttestor:
        def verify_capture(self, trace, request, response): return True
    verifier = RecordedDecisionEvidenceVerifier([trace], FakeAttestor())
    with pytest.raises(ValueError): verifier.tokens_for_call(call, policy)
    assert verifier.verify_source(episode, policy)
    assert verifier.tokens_for_call(call, policy) == tokens
    assert not isinstance(trace, PublicSummary)
    call.response = {'action': 'changed'}
    assert not verifier.verify_source(episode, policy)
    with pytest.raises(ValueError): verifier.tokens_for_call(call, policy)
    class Reject:
        def verify_capture(self, trace, request, response): return False
    call.response = {'action': 'query'}
    assert not RecordedDecisionEvidenceVerifier([trace], Reject()).verify_source(episode, policy)


def test_designer_capture_is_bound_to_package_policy_and_call():
    policy = Checkpoint('designer', 'v1', 'tok', 'weights')
    tokens = Tokens((1,), (2,), (True,))
    request, response = {'prompt': 'design'}, {'proposal': 'task'}
    trace = ExactPolicyTrace('designer', 'rollout', 'call', policy, tokens,
        digest(request), digest(response), 'template-v1', 'evidence', 'package')
    class FakeAttestor:
        def verify_capture(self, trace, request, response): return True
    verifier = RecordedDesignerActionVerifier([trace], lambda call: (request, response), FakeAttestor())
    assert verifier.verify_designer_action(tokens, policy, 'package', 'rollout')
    assert not verifier.verify_designer_action(tokens, policy, 'other', 'rollout')
    assert not verifier.verify_designer_action(tokens, replace(policy, revision='v2'), 'package', 'rollout')
    response['proposal'] = 'changed'
    assert not verifier.verify_designer_action(tokens, policy, 'package', 'rollout')
