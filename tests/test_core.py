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
    return Demonstration('sample', identity, 'episode', 'source', T, kwargs.get('outcome', Outcome.SUCCESS), kwargs.get('verified', True), kwargs.get('kind', 'real_model'))

def completed():
    row = service_rollout(T, BEFORE, 't1', 'episode1', Outcome.SUCCESS)
    return CompletedUpdate(UpdateRequest(BEFORE, 'package', (row,), 'pg'), AFTER, 'run1', 'evidence')

class Store:
    def verify(self, checkpoint): return checkpoint in (BEFORE, AFTER)
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

def test_replay_provenance_and_separation(tmp_path):
    for opts in ({'outcome':Outcome.FAILURE}, {'outcome':Outcome.UNKNOWN}, {'verified':False}, {'kind':'scripted_transport_fixture'}):
        with pytest.raises(ValueError): demo(**opts)
    bank = ReplayBank([demo()]); path = tmp_path / 'bank.json'; bank.save(path)
    bank = ReplayBank.load(path)
    assert bank.freeze([TRAIN], SplitRegistry([TRAIN, REPLAY])).verify()
    for key in ('world', 'task', 'family', 'lineage'):
        vals = dict(world='w3', task='t3', family='f3', lineage='l3', split=Split.TRAIN)
        vals[key] = getattr(REPLAY, key)
        with pytest.raises(ValueError): SplitRegistry([REPLAY, Identity(**vals)])
    with pytest.raises(ValueError): bank.freeze([TRAIN], SplitRegistry([TRAIN, REPLAY]), min_samples=2)
    with pytest.raises(ValueError): bank.freeze([REPLAY], SplitRegistry([REPLAY]))

def test_reward_and_binding():
    snap = ReplayBank([demo()]).freeze([TRAIN], SplitRegistry([TRAIN, REPLAY])); update = completed()
    r = score_update(update, snap, Scorer(), Store())
    assert r.mean == 3 and r.samples[0].before == -6 and r.samples[0].after == -3
    assert score_update(update, snap, Scorer(), Store(), normalization='mean').mean == 1.5
    with pytest.raises(ValueError): score_update(None, snap, Scorer(), Store())
    class Bad:
        def score(self, checkpoint, batch): return [[float('nan'), -1, -1]]
    with pytest.raises(ValueError): score_update(update, snap, Bad(), Store())
    d = Checkpoint('designer', 'v1', 'tok', 'd1')
    row = designer_rollout(T, d, 'package', r, update)
    comparable_designer_rollouts([row])
    with pytest.raises(ValueError): designer_rollout(T, d, 'package', r, None)
    with pytest.raises(ValueError): CompletedUpdate(update.request, BEFORE, 'run', 'evidence')
    with pytest.raises(ValueError): service_rollout(T, BEFORE, 't1', 'ep', Outcome.UNKNOWN)

def test_imports_are_inert():
    import subprocess, sys
    program = 'import sys; import rift.loop, rift.benchmark.usami, rift.replay.reward; assert "rsi_core" not in sys.modules; assert "usami_v2" not in sys.modules'
    subprocess.run([sys.executable, '-c', program], check=True)

def test_split_registry_and_checkpoint_groups(tmp_path):
    from rift.replay.splits import SplitRegistry
    registry = SplitRegistry([TRAIN, REPLAY])
    path = tmp_path / 'splits.json'; registry.save(path)
    registry = SplitRegistry.load(path)
    with pytest.raises(ValueError): registry.register(Identity('w1', 'other', 'other', 'other', Split.TEST))
    with pytest.raises(ValueError): Tokens((1,), (2,), (False,))
    with pytest.raises(ValueError): UpdateRequest(BEFORE, 'p', (service_rollout(T, BEFORE, 't1', 'e', Outcome.SUCCESS),), 'grpo')
    update = completed(); reward = score_update(update, ReplayBank([demo()]).freeze([TRAIN], SplitRegistry([TRAIN, REPLAY])), Scorer(), Store())
    d = Checkpoint('designer', 'v1', 'tok', 'd1')
    one = designer_rollout(T, d, 'package', reward, update)
    from dataclasses import replace
    with pytest.raises(ValueError): comparable_designer_rollouts([one, replace(one, reward_source='replay:other:snapshot')])

def test_resume_pipeline_without_training(tmp_path):
    cert = Certification('p1', CertificationStatus.CERTIFIED, 't1')
    update = completed()
    state = LoopState('package', ('p1',)).certify([cert]).ready(update.request)
    path = tmp_path / 'state.json'
    state.save(path); state = LoopState.load(path)
    assert state.stage is Stage.READY_FOR_TRAINING and state.service_request == update.request
    with pytest.raises(ValueError): state.service_completed(None)
    state = state.service_completed(update)
    reward = score_update(update, ReplayBank([demo()]).freeze([TRAIN], SplitRegistry([TRAIN, REPLAY])), Scorer(), Store())
    state = state.replay_scored(reward)
    designer = Checkpoint('designer', 'v1', 'tok', 'd1')
    row = designer_rollout(T, designer, 'package', reward, update)
    req = UpdateRequest(designer, 'package', (row,), 'pg')
    state = state.designer_ready(req)
    state.save(path); assert LoopState.load(path) == state
    with pytest.raises(ValueError): state.designer_completed(None)

def test_designer_cannot_observe_heldout_or_private_records():
    from rift.agents import TrainingExperience, propose
    from rift.benchmark.public_feedback import PublicEpisode, PublicEvent
    from rift.tasks import Proposal
    episode = PublicEpisode((PublicEvent('user','question'),), Outcome.FAILURE)
    class Designer:
        def propose(self, experiences):
            assert experiences == (episode,)
            return [Proposal('p', 'hypothesis', 'request', '{}')]
    assert propose(Designer(), [TrainingExperience(TRAIN, episode)], SplitRegistry([TRAIN]))[0].proposal_id == 'p'
    with pytest.raises(ValueError): TrainingExperience(REPLAY, episode)
    with pytest.raises(ValueError): propose(Designer(), [TrainingExperience(TRAIN, episode)], SplitRegistry([]))

def test_orchestrator_stops_on_uncertified_proposal():
    from rift.agents import TrainingExperience
    from rift.benchmark.public_feedback import PublicEpisode
    from rift.orchestrator import prepare_round
    from rift.tasks import Proposal
    class Designer:
        def propose(self, experiences): return [Proposal('p','h','request','{}')]
    class Host:
        def certify_proposal(self, proposal): return Certification('p', CertificationStatus.UNKNOWN), None
        def rollout(self, task, service): raise AssertionError('must not roll out')
    with pytest.raises(ValueError, match='certified'):
        prepare_round('package', [TrainingExperience(TRAIN, PublicEpisode((), Outcome.UNKNOWN))], SplitRegistry([TRAIN]), Designer(), Host(), None, BEFORE, None)

def test_frozen_snapshot_persistence_and_tamper(tmp_path):
    from rift.replay.bank import Snapshot
    snap = ReplayBank([demo()]).freeze([TRAIN], SplitRegistry([TRAIN, REPLAY]))
    path = tmp_path / 'snapshot.json'
    snap.save(path)
    assert Snapshot.load(path) == snap
    data = json.loads(path.read_text(encoding='utf-8'))
    data['rows'][0]['tokens']['response'][0] = 100
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='snapshot'): Snapshot.load(path)

def test_replay_identity_is_stable_across_different_training_packages():
    bank = ReplayBank([demo()])
    other = Identity('w3','t3','f3','l3',Split.TRAIN)
    registry = SplitRegistry([TRAIN, other, REPLAY])
    first, second = bank.freeze([TRAIN], registry), bank.freeze([other], registry)
    assert first.digest != second.digest
    assert first.replay_digest == second.replay_digest
