import json
from types import SimpleNamespace as NS
import pytest
from rift.schemas import Checkpoint, Identity, Outcome, Split, Tokens
from rift.benchmark.public_feedback import service_reward
from rift.benchmark.private_evaluation import project_events, public_certification
from rift.replay.trajectory import Demonstration
from rift.replay.bank import ReplayBank
from rift.replay.splits import SplitRegistry
from rift.replay.reward import score_update
from rift.training.contracts import CompletedUpdate, UpdateRequest
from rift.training.records import service_rollout, designer_rollout, comparable_designer_rollouts
from rift.tasks import Certification, CertificationStatus
from rift.loop import LoopState, Stage

T = Tokens((1, 2), (3, 4, 5), (True, False, True))
BEFORE = Checkpoint('service', 'one', 'tok', 'weights1')
AFTER = Checkpoint('service', 'two', 'tok', 'weights2')
TRAIN = Identity('w1', 't1', 'f1', 'l1', Split.TRAIN)
REPLAY = Identity('w2', 't2', 'f2', 'l2', Split.REPLAY)

def demo(identity=REPLAY, **kwargs):
    return Demonstration('sample', identity, 'episode', BEFORE, T, kwargs.get('outcome', Outcome.SUCCESS), kwargs.get('verified', True), kwargs.get('kind', 'real_model'))

def completed():
    row = service_rollout(T, BEFORE, 't1', 'episode1', Outcome.SUCCESS, response_id='response1')
    return CompletedUpdate(UpdateRequest(BEFORE, 'package', (row,), 'pg'), AFTER, 'run1', 'evidence')

class Store:
    def verify(self, checkpoint): return checkpoint in (BEFORE, AFTER)
class Verifier:
    def verify_update(self, update): return True  # fake verifier, CPU tests only
    def verify_reward(self, reward, update): return True
    def verify_demonstration(self, sample): return True
    def verify_designer_action(self, tokens, checkpoint, package_id, rollout_id, group_id): return True
class Scorer:
    def score(self, checkpoint, batch):
        return [[-2., -10., -4.] if checkpoint == BEFORE else [-1., -100., -2.] for _ in batch]

def test_public_projection_and_business_reward():
    events = [NS(kind='customer_request', payload={'request': 'refund', 'oracle': 'SECRET'}),
        NS(kind='service_tool', payload={'operation': {'name':'query'}, 'result': {'error':'EVIDENCE_REQUIRED'}, 'oracle':'SECRET'}),
        NS(kind='customer_transition', payload={'transition_kind':'decision','decision':{'message':'yes','hidden':'SECRET'},'authority_proofs':['SECRET']}),
        NS(kind='service_reply', payload={'message': {'text':'done','hidden':'SECRET'}})]
    record = NS(events=events, outcome=NS(protocol='commerce-outcome-v2', business_goal_success=True, strict_task_success=False))
    public = project_events(record, True)
    assert 'SECRET' not in repr(public) and 'EVIDENCE_REQUIRED' in repr(public)
    assert public.outcome == Outcome.SUCCESS and service_reward(public.outcome) == 1
    assert service_reward(Outcome.UNKNOWN) is None
    with pytest.raises(ValueError): project_events(record, False)
    record.outcome.protocol = 'commerce-outcome-v1'
    assert project_events(record, True).outcome is Outcome.UNKNOWN

def test_certification_and_state():
    cert = NS(status='CERTIFIED', full_task_reachability_certified=True, task_hash='t1', blockers=['secret'])
    result = public_certification('p1', cert, verified=True)
    assert result == Certification('p1', CertificationStatus.CERTIFIED, 't1') and 'secret' not in repr(result)
    state = LoopState('package', ('p1',))
    with pytest.raises(ValueError): state.certify([public_certification('p1', cert, verified=False)])
    ready = state.certify([result]).ready(completed().request)
    assert ready.stage is Stage.READY_FOR_TRAINING
    assert not hasattr(ready, 'policy_after')

def test_package_credit_requires_distinct_tasks_and_complete_rollout_coverage():
    state = LoopState('package', ('p1', 'p2'))
    first = Certification('p1', CertificationStatus.CERTIFIED, 't1')
    with pytest.raises(ValueError, match='distinct'):
        state.certify([first, Certification('p2', CertificationStatus.CERTIFIED, 't1')])
    certified = state.certify([first, Certification('p2', CertificationStatus.CERTIFIED, 't2')])
    with pytest.raises(ValueError, match='cover'):
        certified.ready(completed().request)

def test_replay_provenance_and_separation(tmp_path):
    for opts in ({'outcome':Outcome.FAILURE}, {'outcome':Outcome.UNKNOWN}, {'verified':False}, {'kind':'scripted_transport_fixture'}):
        with pytest.raises(ValueError): demo(**opts)
    bank = ReplayBank([demo()], verifier=Verifier()); path = tmp_path / 'bank.json'; bank.save(path)
    bank = ReplayBank.load(path, verifier=Verifier())
    assert bank.freeze([TRAIN], SplitRegistry([TRAIN, REPLAY])).verify()
    for key in ('world', 'task', 'family', 'lineage'):
        vals = dict(world='w3', task='t3', family='f3', lineage='l3', split=Split.TRAIN)
        vals[key] = getattr(REPLAY, key)
        with pytest.raises(ValueError): SplitRegistry([REPLAY, Identity(**vals)])
    with pytest.raises(ValueError): bank.freeze([TRAIN], SplitRegistry([TRAIN, REPLAY]), min_samples=2)
    with pytest.raises(ValueError): bank.freeze([REPLAY], SplitRegistry([REPLAY]))
    class RejectedSource:
        def verify_demonstration(self, sample): return False
    with pytest.raises(ValueError, match='demonstration evidence'):
        ReplayBank([demo()], verifier=RejectedSource())
    with pytest.raises(ValueError, match='demonstration evidence'):
        ReplayBank.load(path, verifier=RejectedSource())

def test_reward_and_binding():
    snap = ReplayBank([demo()], verifier=Verifier()).freeze([TRAIN], SplitRegistry([TRAIN, REPLAY])); update = completed()
    r = score_update(update, snap, Scorer(), Store(), update_verifier=Verifier(), replay_verifier=Verifier())
    assert r.mean == 3 and r.samples[0].before == -6 and r.samples[0].after == -3
    assert score_update(update, snap, Scorer(), Store(), update_verifier=Verifier(), replay_verifier=Verifier(), normalization='mean').mean == 1.5
    from dataclasses import replace
    bad_sample = replace(demo(), source_checkpoint=Checkpoint('service', 'old', 'different-tokenizer', 'old-weights'))
    bad_snapshot = ReplayBank([bad_sample], verifier=Verifier()).freeze([TRAIN], SplitRegistry([TRAIN, REPLAY]))
    with pytest.raises(ValueError, match='tokenizer'):
        score_update(update, bad_snapshot, Scorer(), Store(), update_verifier=Verifier(), replay_verifier=Verifier())
    with pytest.raises(ValueError): score_update(None, snap, Scorer(), Store(), update_verifier=Verifier(), replay_verifier=Verifier())
    class Bad:
        def score(self, checkpoint, batch): return [[float('nan'), -1, -1]]
    with pytest.raises(ValueError): score_update(update, snap, Bad(), Store(), update_verifier=Verifier(), replay_verifier=Verifier())
    d = Checkpoint('designer', 'v1', 'tok', 'd1')
    row = designer_rollout(T, d, 'package', r, update, designer_rollout_id='designer1', update_verifier=Verifier(), reward_verifier=Verifier(), action_verifier=Verifier())
    comparable_designer_rollouts([row])
    with pytest.raises(ValueError): designer_rollout(T, d, 'package', r, None, designer_rollout_id='designer1', update_verifier=Verifier(), reward_verifier=Verifier(), action_verifier=Verifier())
    with pytest.raises(ValueError): CompletedUpdate(update.request, BEFORE, 'run', 'evidence')
    with pytest.raises(ValueError): service_rollout(T, BEFORE, 't1', 'ep', Outcome.UNKNOWN, response_id='response')

def test_imports_are_inert():
    import subprocess, sys
    program = '''
import importlib, pkgutil, socket, sys, rift
def forbidden(*args, **kwargs):
    raise AssertionError("import attempted a network connection")
socket.create_connection = forbidden
socket.socket.connect = forbidden
for module in pkgutil.walk_packages(rift.__path__, rift.__name__ + "."):
    importlib.import_module(module.name)
assert "rsi_core" not in sys.modules
assert "usami_v2" not in sys.modules
assert "ray" not in sys.modules and "verl" not in sys.modules
'''
    subprocess.run([sys.executable, '-c', program], check=True)

def test_split_registry_and_checkpoint_groups(tmp_path):
    from rift.replay.splits import SplitRegistry
    registry = SplitRegistry([TRAIN, REPLAY])
    path = tmp_path / 'splits.json'; registry.save(path)
    registry = SplitRegistry.load(path)
    with pytest.raises(ValueError): registry.register(Identity('w1', 'other', 'other', 'other', Split.TEST))
    with pytest.raises(ValueError): Tokens((1,), (2,), (False,))
    with pytest.raises(ValueError): UpdateRequest(BEFORE, 'p', (service_rollout(T, BEFORE, 't1', 'e', Outcome.SUCCESS, response_id='response'),), 'grpo')
    update = completed(); reward = score_update(update, ReplayBank([demo()], verifier=Verifier()).freeze([TRAIN], SplitRegistry([TRAIN, REPLAY])), Scorer(), Store(), update_verifier=Verifier(), replay_verifier=Verifier())
    d = Checkpoint('designer', 'v1', 'tok', 'd1')
    one = designer_rollout(T, d, 'package', reward, update, designer_rollout_id='designer1', update_verifier=Verifier(), reward_verifier=Verifier(), action_verifier=Verifier())
    from dataclasses import replace
    with pytest.raises(ValueError): comparable_designer_rollouts([one, replace(one, reward_source='replay:other:snapshot')])

def test_resume_pipeline_without_training(tmp_path):
    from rift.training.records import DesignerFeedback
    cert = Certification('p1', CertificationStatus.CERTIFIED, 't1')
    update = completed()
    state = LoopState('package', ('p1',)).certify([cert]).ready(update.request)
    path = tmp_path / 'state.json'
    state.save(path); state = LoopState.load(path)
    assert state.stage is Stage.READY_FOR_TRAINING and state.service_request == update.request
    with pytest.raises(ValueError): state.service_completed(None, update_verifier=Verifier())
    state = state.service_completed(update, update_verifier=Verifier())
    reward = score_update(update, ReplayBank([demo()], verifier=Verifier()).freeze([TRAIN], SplitRegistry([TRAIN, REPLAY])), Scorer(), Store(), update_verifier=Verifier(), replay_verifier=Verifier())
    state = state.replay_scored(reward, reward_verifier=Verifier())
    designer = Checkpoint('designer', 'v1', 'tok', 'd1')
    row = designer_rollout(T, designer, 'package', reward, update, designer_rollout_id='designer1', update_verifier=Verifier(), reward_verifier=Verifier(), action_verifier=Verifier())
    req = UpdateRequest(designer, 'package', (row,), 'pg')
    state = state.designer_ready(req, [DesignerFeedback(T, update, reward, 'designer1')], update_verifier=Verifier(), reward_verifier=Verifier(), action_verifier=Verifier())
    state.save(path); assert LoopState.load(path, update_verifier=Verifier(), reward_verifier=Verifier(), action_verifier=Verifier()) == state
    with pytest.raises(ValueError): state.designer_completed(None, update_verifier=Verifier())

def test_next_round_binds_both_prior_checkpoints(tmp_path):
    """All update and verification records in this test are explicit fakes."""
    from rift.training.records import DesignerFeedback, prepare_designer_update
    update = completed()
    state = LoopState('package', ('p1',)).certify([Certification('p1', CertificationStatus.CERTIFIED, 't1')]).ready(update.request)
    with pytest.raises(ValueError, match='both policy'):
        state.start_next_round('next', ('p2',), update_verifier=Verifier())
    state = state.service_completed(update, update_verifier=Verifier())
    snapshot = ReplayBank([demo()], verifier=Verifier()).freeze([TRAIN], SplitRegistry([TRAIN, REPLAY]))
    reward = score_update(update, snapshot, Scorer(), Store(), update_verifier=Verifier(), replay_verifier=Verifier())
    designer_before = Checkpoint('designer', 'v1', 'tok', 'designer-1')
    designer_after = Checkpoint('designer', 'v2', 'tok', 'designer-2')
    feedback = [DesignerFeedback(T, update, reward, 'designer-rollout')]
    req = prepare_designer_update('batch', designer_before, feedback, update_verifier=Verifier(), reward_verifier=Verifier(), action_verifier=Verifier())
    state = state.replay_scored(reward, reward_verifier=Verifier()).designer_ready(req, feedback, update_verifier=Verifier(), reward_verifier=Verifier(), action_verifier=Verifier())
    designer_update = CompletedUpdate(req, designer_after, 'designer-run', 'fake-evidence')
    state = state.designer_completed(designer_update, update_verifier=Verifier())
    next_round = state.start_next_round('next', ('p2',), update_verifier=Verifier())
    assert next_round.expected_service == AFTER and next_round.expected_designer == designer_after
    path = tmp_path / 'next-round.json'; next_round.save(path)
    assert LoopState.load(path, update_verifier=Verifier()) == next_round
    with pytest.raises(ValueError, match='evidence'):
        LoopState.load(path)
    next_round = next_round.certify([Certification('p2', CertificationStatus.CERTIFIED, 't1')])
    old_request = UpdateRequest(BEFORE, 'next', update.request.rollouts, 'pg')
    with pytest.raises(ValueError, match='prior checkpoint'):
        next_round.ready(old_request)

def test_designer_cannot_observe_heldout_or_private_records(monkeypatch):
    from rift.agents import TrainingExperience, propose
    from rift.benchmark.public_feedback import PublicEpisode, PublicEvent
    from rift.tasks import Proposal
    episode = PublicEpisode((PublicEvent('user','question'),), Outcome.FAILURE)
    record = NS(task=NS(partition='train', record_hash=TRAIN.task), initial_world=NS(state_hash=TRAIN.world), oracle='SECRET')
    monkeypatch.setattr('rift.benchmark.usami.public_episode', lambda rec: episode)
    class Designer:
        def propose(self, experiences):
            assert experiences == (episode,)
            return [Proposal('p', 'hypothesis', 'request', '{}')]
    assert propose(Designer(), [TrainingExperience(TRAIN, record)], SplitRegistry([TRAIN]))[0].proposal_id == 'p'
    with pytest.raises(ValueError): TrainingExperience(REPLAY, record)
    with pytest.raises(ValueError): propose(Designer(), [TrainingExperience(TRAIN, record)], SplitRegistry([]))
    heldout = NS(task=NS(partition='heldout', record_hash=TRAIN.task), initial_world=NS(state_hash=TRAIN.world))
    with pytest.raises(ValueError): TrainingExperience(TRAIN, heldout)
    wrong_world = NS(task=NS(partition='train', record_hash=TRAIN.task), initial_world=NS(state_hash='another-world'))
    with pytest.raises(ValueError): TrainingExperience(TRAIN, wrong_world)

def test_orchestrator_stops_on_uncertified_proposal(monkeypatch):
    from rift.agents import TrainingExperience
    from rift.benchmark.public_feedback import PublicEpisode
    from rift.orchestrator import prepare_round
    from rift.tasks import Proposal
    class Designer:
        def propose(self, experiences): return [Proposal('p','h','request','{}')]
    class Host:
        def certify_proposal(self, proposal): return Certification('p', CertificationStatus.UNKNOWN), None
        def rollout(self, task, service): raise AssertionError('must not roll out')
    registry = SplitRegistry([TRAIN])
    host = Host(); host.registry = registry
    record = NS(task=NS(partition='train', record_hash=TRAIN.task), initial_world=NS(state_hash=TRAIN.world))
    monkeypatch.setattr('rift.benchmark.usami.public_episode', lambda rec: PublicEpisode((), Outcome.UNKNOWN))
    with pytest.raises(ValueError, match='certified'):
        prepare_round('package', [TrainingExperience(TRAIN, record)], registry, Designer(), host, None, BEFORE, None)

def test_orchestrator_rejects_invalid_resume_before_designer_call():
    from rift.orchestrator import prepare_round
    class Designer:
        def propose(self, experiences): raise AssertionError('must not invoke Designer')
    registry = SplitRegistry([TRAIN])
    host = NS(registry=registry)
    prior = LoopState('old', ('p1',))
    with pytest.raises(ValueError, match='resumed round'):
        prepare_round('new', [], registry, Designer(), host, None, BEFORE, None, initial_state=prior)

def test_frozen_snapshot_persistence_and_tamper(tmp_path):
    from rift.replay.bank import Snapshot
    snap = ReplayBank([demo()], verifier=Verifier()).freeze([TRAIN], SplitRegistry([TRAIN, REPLAY]))
    path = tmp_path / 'snapshot.json'
    snap.save(path)
    assert Snapshot.load(path, verifier=Verifier()) == snap
    data = json.loads(path.read_text(encoding='utf-8'))
    data['rows'][0]['tokens']['response'][0] = 100
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='snapshot'): Snapshot.load(path, verifier=Verifier())

def test_replay_identity_is_stable_across_different_training_packages():
    bank = ReplayBank([demo()], verifier=Verifier())
    other = Identity('w3','t3','f3','l3',Split.TRAIN)
    registry = SplitRegistry([TRAIN, other, REPLAY])
    first, second = bank.freeze([TRAIN], registry), bank.freeze([other], registry)
    assert first.digest != second.digest
    assert first.replay_digest == second.replay_digest
    mismatched = bank.freeze([TRAIN, other], registry)
    with pytest.raises(ValueError, match='identities do not match'):
        score_update(completed(), mismatched, Scorer(), Store(), update_verifier=Verifier(), replay_verifier=Verifier())

def test_unverified_update_cannot_score_or_supply_designer_credit(tmp_path):
    from rift.training.records import designer_rollout
    class Rejected:
        def verify_update(self, update): return False
    update = completed()
    snap = ReplayBank([demo()], verifier=Verifier()).freeze([TRAIN], SplitRegistry([TRAIN, REPLAY]))
    with pytest.raises(ValueError, match='evidence'):
        score_update(update, snap, Scorer(), Store(), update_verifier=Rejected(), replay_verifier=Verifier())
    reward = score_update(update, snap, Scorer(), Store(), update_verifier=Verifier(), replay_verifier=Verifier())
    class RejectedSource:
        def verify_demonstration(self, sample): return False
    with pytest.raises(ValueError, match='demonstration evidence'):
        score_update(update, snap, Scorer(), Store(), update_verifier=Verifier(), replay_verifier=RejectedSource())
    designer = Checkpoint('designer', 'v1', 'tok', 'd1')
    with pytest.raises(ValueError, match='evidence'):
        designer_rollout(T, designer, 'package', reward, update, designer_rollout_id='designer1', update_verifier=Rejected(), reward_verifier=Verifier(), action_verifier=Verifier())
    class RejectedReward:
        def verify_reward(self, reward, update): return False
    class RejectedAction:
        def verify_designer_action(self, tokens, checkpoint, package_id, rollout_id, group_id): return False
    with pytest.raises(ValueError, match='reward evidence'):
        designer_rollout(T, designer, 'package', reward, update, designer_rollout_id='designer1', update_verifier=Verifier(), reward_verifier=RejectedReward(), action_verifier=Verifier())
    with pytest.raises(ValueError, match='action evidence'):
        designer_rollout(T, designer, 'package', reward, update, designer_rollout_id='designer1', update_verifier=Verifier(), reward_verifier=Verifier(), action_verifier=RejectedAction())
    state = LoopState('package', ('p1',)).certify([Certification('p1', CertificationStatus.CERTIFIED, 't1')]).ready(update.request)
    with pytest.raises(ValueError, match='evidence'):
        state.service_completed(update, update_verifier=Rejected())
    state = state.service_completed(update, update_verifier=Verifier())
    with pytest.raises(ValueError, match='reward evidence'):
        state.replay_scored(reward, reward_verifier=RejectedReward())
    path = tmp_path / 'state.json'; state.save(path)
    with pytest.raises(ValueError, match='evidence'):
        LoopState.load(path)
    with pytest.raises(ValueError, match='evidence'):
        LoopState.load(path, update_verifier=Rejected())

def test_designer_batch_accumulates_comparable_package_feedback(tmp_path):
    from rift.training.records import DesignerFeedback, prepare_designer_update
    from dataclasses import replace
    update = completed()
    other_train = Identity('w3', 't3', 'f3', 'l3', Split.TRAIN)
    registry = SplitRegistry([TRAIN, other_train, REPLAY])
    bank = ReplayBank([demo()], verifier=Verifier())
    reward = score_update(update, bank.freeze([TRAIN], registry), Scorer(), Store(), update_verifier=Verifier(), replay_verifier=Verifier())
    other_row = service_rollout(T, BEFORE, 't3', 'ep3', Outcome.SUCCESS, response_id='response3')
    other_update = CompletedUpdate(UpdateRequest(BEFORE, 'package2', (other_row,), 'pg'), AFTER, 'run2', 'evidence2')
    class OtherScorer:
        def score(self, checkpoint, batch):
            return [[-2., -10., -4.] if checkpoint == BEFORE else [-1.5, -10., -3.] for _ in batch]
    reward2 = score_update(other_update, bank.freeze([other_train], registry), OtherScorer(), Store(), update_verifier=Verifier(), replay_verifier=Verifier())
    designer = Checkpoint('designer', 'v1', 'tok', 'd1')
    items = [DesignerFeedback(T, update, reward, 'designer1', 'g'), DesignerFeedback(T, other_update, reward2, 'designer2', 'g')]
    request = prepare_designer_update('batch', designer, items, update_verifier=Verifier(), reward_verifier=Verifier(), action_verifier=Verifier(), algorithm='grpo')
    assert [r.task_id for r in request.rollouts] == ['package', 'package2']
    assert request.rollouts[0].reward != request.rollouts[1].reward
    state = LoopState('package', ('p1',)).certify([Certification('p1', CertificationStatus.CERTIFIED, 't1')]).ready(update.request)
    state = state.service_completed(update, update_verifier=Verifier()).replay_scored(reward, reward_verifier=Verifier()).designer_ready(request, items, update_verifier=Verifier(), reward_verifier=Verifier(), action_verifier=Verifier())
    assert state.stage is Stage.DESIGNER_READY
    path = tmp_path / 'batch-state.json'
    state.save(path)
    assert LoopState.load(path, update_verifier=Verifier(), reward_verifier=Verifier(), action_verifier=Verifier()) == state
    class RejectedReward:
        def verify_reward(self, reward, update): return False
    with pytest.raises(ValueError, match='reward evidence'):
        LoopState.load(path, update_verifier=Verifier(), reward_verifier=RejectedReward(), action_verifier=Verifier())
    raw = json.loads(path.read_text(encoding='utf-8'))
    raw['designer_feedback'][1]['update']['run_id'] = 'different-run'
    path.write_text(json.dumps(raw), encoding='utf-8')
    with pytest.raises(ValueError, match='reward and Service update differ'):
        LoopState.load(path, update_verifier=Verifier(), reward_verifier=Verifier(), action_verifier=Verifier())
    with pytest.raises(ValueError, match='duplicate'):
        prepare_designer_update('batch', designer, [items[0], items[0]], update_verifier=Verifier(), reward_verifier=Verifier(), action_verifier=Verifier())
    with pytest.raises(ValueError, match='duplicate Designer'):
        prepare_designer_update('batch', designer, [items[0], replace(items[1], designer_rollout_id='designer1')], update_verifier=Verifier(), reward_verifier=Verifier(), action_verifier=Verifier())
    wrong = replace(reward2, normalization='mean')
    with pytest.raises(ValueError, match='mixed'):
        prepare_designer_update('batch', designer, [items[0], DesignerFeedback(T, other_update, wrong, 'designer2')], update_verifier=Verifier(), reward_verifier=Verifier(), action_verifier=Verifier())
