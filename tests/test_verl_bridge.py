import asyncio
import sys
import types
from types import SimpleNamespace
import numpy as np
import pytest
from rift.schemas import Checkpoint, Outcome
from rift.verl.capture import ContinuousCapture
from rift.verl.reward import verified_business_reward, UnknownBusinessOutcome


def test_capture_rejects_retokenization_and_incorrect_masks():
    checkpoint = Checkpoint('service', 'v1', 'tok', 'weights')
    cap = ContinuousCapture([1], checkpoint)
    with pytest.raises(ValueError, match='changed generated'):
        cap.add_generation([1], [2], None, [1, 3], [1])
    cap.add_generation([1], [2], [-.5], [1, 2], [1])
    with pytest.raises(ValueError, match='masked'):
        cap.add_context([1, 2, 3], [1, 1])
    cap.add_context([1, 2, 3], [1, 0])
    cap.add_generation([1, 2, 3], [4], [-.1], [1, 2, 3, 4], [1, 0, 1])
    assert cap.decisions[1].tokens.prompt == (1, 2, 3)
    assert cap.output().response == (2, 3, 4)
    assert cap.output().mask == (True, False, True)
    with pytest.raises(UnknownBusinessOutcome): verified_business_reward(None, lambda _: Outcome.UNKNOWN)


def test_agent_loop_captures_server_tokens_and_business_only_reward(monkeypatch):
    """Stub server and session exercise the extension; no VERL/NPU execution."""
    class Base:
        def __init__(self): self.rollout_config = SimpleNamespace(response_length=20, n=4); self.tokenizer = None
        async def ct_build_initial_tokens(self, messages, tools=None): return [1]
        async def ct_merge_assistant_token(self, old, output, mask, logs, assistant_logprobs=None):
            if assistant_logprobs is not None and logs is None:
                raise ValueError('response_logprobs is required when assistant_logprobs is provided')
            merged_logs = logs
            if assistant_logprobs is not None:
                merged_logs = logs + assistant_logprobs
            return SimpleNamespace(token_ids=old + output), mask + [1] * len(output), merged_logs
        async def ct_merge_context_msg(self, prev, new, old, mask, logs, tools=None):
            merged_logs = None if logs is None else logs + [0.0]
            return SimpleNamespace(token_ids=old + [7]), mask + [0], merged_logs
    class Output:
        def __init__(self, **fields): self.__dict__.update(fields)
    for name in ('verl', 'verl.experimental', 'verl.experimental.agent_loop'):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    module = types.ModuleType('verl.experimental.agent_loop.agent_loop')
    module.AgentLoopBase = Base; module.AgentLoopOutput = Output
    monkeypatch.setitem(sys.modules, module.__name__, module)
    import rift.verl.agent_loop as extension
    monkeypatch.delattr(extension, 'UsamiServiceAgentLoop', raising=False)
    cls = extension.UsamiServiceAgentLoop
    loop = cls()
    loop.rollout_config.prompt_length = 3
    assert loop._cap_text_prompt_length([1, 2, 3]) == [1, 2, 3]
    with pytest.raises(ValueError, match='initial Service prompt exceeds'):
        loop._cap_text_prompt_length([1, 2, 3, 4])
    class Server:
        def __init__(self): self.inputs = []; self.sampling = []
        async def generate(self, request_id, *, prompt_ids, sampling_params):
            self.inputs.append(prompt_ids)
            self.sampling.append(dict(sampling_params))
            return SimpleNamespace(token_ids=[2, 3] if len(self.inputs)==1 else [4],
                log_probs=[-0.1, -0.2] if len(self.inputs)==1 else [-0.3],
                extra_fields={'global_steps':1,'min_global_steps':1,'max_global_steps':1})
    loop.server_manager = Server()
    class Session:
        def __init__(self, rollout_id='fake-rollout', group_id='group', group_size=4,
                     sampling_seed=7, customer_seed=11):
            self.rollout_id, self.group_id = rollout_id, group_id
            self.sampling_seed, self.customer_seed = sampling_seed, customer_seed
            self.sampling_scope = f'test:{rollout_id}'
            self.n=0; self.expected_group_size=group_size; self.cancelled=False
        async def start(self): return ([{'role':'user','content':'refund'}], [])
        async def advance(self, tokens, prefix, sampling_params, token_logprobs, generation_id, serving_fields=None):
            self.n += 1
            return ({'role':'assistant','content':'action'}, [{'role':'tool','content':'EVIDENCE_REQUIRED'}], self.n == 2)
        async def cancel_for_budget(self): self.cancelled = True
        async def finish(self): return SimpleNamespace(record_hash='e' * 64, replay_verified=True, cancelled=self.cancelled)
        def public_outcome(self, episode): return Outcome.SUCCESS
    class Verifier:
        def verify(self, checkpoint): return True
        def verify_serving(self, checkpoint): return True
        def verify_generation(self, checkpoint, fields): return fields == {'global_steps':1,'min_global_steps':1,'max_global_steps':1}
        def record_generation(self, *args): pass
    class Sink:
        attestor = Verifier()
        async def persist(self, episode, checkpoint, decisions, tokens): self.decisions = decisions
        def evidence_ref(self, episode): return 'host-private-evidence-ref'
    monkeypatch.setattr('rift.benchmark.usami.public_episode', lambda episode: SimpleNamespace(outcome=Outcome.SUCCESS))
    sink = Sink()
    class Host:
        def __init__(self): self.sessions=[]; self.session_ids=[]
        async def open_rollout(self, task_ref, group_id, session_id, tokenizer, *, sampling_seed,
                               customer_seed, rollout_id, expected_checkpoint, group_size):
            self.session_ids.append(session_id)
            session = Session(rollout_id, group_id, group_size, sampling_seed, customer_seed)
            self.sessions.append(session)
            return session, Checkpoint('service','v1','tok','weights'), Verifier(), sink
    host = Host()
    loop.rift_host_provider = host
    with pytest.raises(ValueError, match='private host objects'):
        asyncio.run(loop.run({}, rift_task_ref='opaque', uid='group', rift_session=Session()))
    with pytest.raises(ValueError, match='RIFT group size must exactly match VERL rollout.n'):
        asyncio.run(loop.run({}, rift_task_ref='opaque', uid='group', session_id=0,
            rift_request_id='request-bad', rift_mode='train', rift_group_size=2,
            rift_rollout_id='rollout-bad', rift_sampling_seed=3, rift_customer_seed=4,
            rift_expected_checkpoint={'policy':'service','revision':'v1','tokenizer':'tok','weights_digest':'weights'}))
    result = asyncio.run(loop.run({}, rift_task_ref='opaque', uid='group', session_id=np.int64(0),
        rift_request_id='request-good', rift_mode='train', rift_group_size=4,
        rift_rollout_id='rollout-good', rift_sampling_seed=3, rift_customer_seed=4,
        rift_expected_checkpoint={'policy':'service','revision':'v1','tokenizer':'tok','weights_digest':'weights'}))
    assert loop.server_manager.inputs == [[1], [1, 2, 3, 7]]
    assert [params['max_tokens'] for params in loop.server_manager.sampling] == [20, 17]
    assert result.response_ids == [2, 3, 7, 4] and result.response_mask == [1, 1, 0, 1]
    assert result.response_logprobs == [-0.1, -0.2, 0.0, -0.3]
    assert result.reward_score == 1 and sink.decisions[1].tokens.prompt == (1, 2, 3, 7)
    assert host.sessions[-1].expected_group_size == 4
    assert host.session_ids[-1] == 0 and type(host.session_ids[-1]) is int
    # A tool result that leaves no room is never added to the model's trace.
    monkeypatch.setattr('rift.benchmark.usami.public_episode', lambda episode:
        SimpleNamespace(outcome=Outcome.UNKNOWN if episode.cancelled else Outcome.SUCCESS))
    loop.server_manager = Server()
    loop.rollout_config.response_length = 3
    limited = asyncio.run(loop.run({}, rift_task_ref='opaque', uid='group', session_id=0,
        rift_request_id='request-limited', rift_mode='train', rift_group_size=4,
        rift_rollout_id='rollout-limited', rift_sampling_seed=3, rift_customer_seed=4,
        rift_expected_checkpoint={'policy':'service','revision':'v1','tokenizer':'tok','weights_digest':'weights'}))
    assert host.sessions[-1].cancelled is True
    assert limited.response_ids == [2, 3] and limited.response_mask == [1, 1]
    assert limited.reward_score is None and limited.extra_fields['rift_outcome'] == Outcome.UNKNOWN.value
    assert len(loop.server_manager.inputs) == 1
    assert loop.server_manager.sampling[0]['max_tokens'] == 3
    monkeypatch.delattr(extension, 'UsamiServiceAgentLoop', raising=False)

def test_checkpoint_store_checks_all_rank_files_and_weights(tmp_path, monkeypatch):
    from rift.verl import checkpoints
    from rift.verl.checkpoints import FSDPCheckpointStore, model_digest
    checkpoint_path = tmp_path / 'checkpoint'; checkpoint_path.mkdir()
    for rank in range(2):
        for kind in ('model', 'optim', 'extra_state'):
            (checkpoint_path / f'{kind}_world_size_2_rank_{rank}.pt').write_bytes(f'{kind}-{rank}'.encode())
    checkpoint = Checkpoint('service', 'v1', 'tok', model_digest(checkpoint_path, 2))
    full_hash_calls = {'content': 0}
    original_content_digests = checkpoints._checkpoint_content_digests
    def counted_content_digests(*args):
        full_hash_calls['content'] += 1
        return original_content_digests(*args)
    monkeypatch.setattr(checkpoints, '_checkpoint_content_digests', counted_content_digests)
    store = FSDPCheckpointStore({checkpoint: checkpoint_path}, world_size=2)
    assert full_hash_calls['content'] == 1
    assert store.verify(checkpoint)
    assert full_hash_calls['content'] == 1
    assert store.verify(checkpoint)
    assert full_hash_calls['content'] == 1
    manifest = tmp_path / 'checkpoints.json'; store.save_manifest(manifest)
    assert FSDPCheckpointStore.load_manifest(manifest).verify(checkpoint)
    assert full_hash_calls['content'] == 1
    fresh_store = FSDPCheckpointStore({checkpoint: checkpoint_path}, world_size=2)
    assert fresh_store.verify(checkpoint)
    assert full_hash_calls['content'] == 1
    (checkpoint_path / 'optim_world_size_2_rank_1.pt').write_bytes(b'changed')
    assert not store.verify(checkpoint)
    with pytest.raises(ValueError, match='manifest'):
        FSDPCheckpointStore.load_manifest(manifest)
    (checkpoint_path / 'optim_world_size_2_rank_1.pt').write_bytes(b'optim-1')
    assert store.verify(checkpoint)
    (checkpoint_path / 'model_world_size_2_rank_0.pt').write_bytes(b'changed')
    assert not store.verify(checkpoint)
    (checkpoint_path / 'optim_world_size_2_rank_1.pt').unlink()
    assert not store.verify(checkpoint)


def test_checkpoint_publish_hashes_model_and_artifacts_once(tmp_path, monkeypatch):
    from rift.verl import checkpoints
    checkpoint_path = tmp_path / 'published'; checkpoint_path.mkdir()
    for kind in ('model', 'optim', 'extra_state'):
        (checkpoint_path / f'{kind}_world_size_1_rank_0.pt').write_bytes(kind.encode())
    calls = {'content': 0}
    original = checkpoints._checkpoint_content_digests
    def counted(*args):
        calls['content'] += 1
        return original(*args)
    monkeypatch.setattr(checkpoints, '_checkpoint_content_digests', counted)
    store = checkpoints.FSDPCheckpointStore({}, world_size=1)
    checkpoint = store.publish('service', '2', 'tok', checkpoint_path)
    assert calls['content'] == 1
    monkeypatch.setattr(checkpoints, 'model_digest', lambda *_: pytest.fail('model shards rehashed'))
    monkeypatch.setattr(checkpoints, 'artifact_digest', lambda *_: pytest.fail('checkpoint rehashed'))
    assert store.verify(checkpoint)
    assert calls['content'] == 1

def test_replay_batch_padding_preserves_context_and_action_mask():
    from rift.verl.scorer import padded_batch
    from rift.schemas import Tokens
    rows = [Tokens((1,2),(3,4,5),(True,False,True)), Tokens((9,),(8,),(True,))]
    batch = padded_batch(rows, 0)
    assert batch['input_ids'] == [[1,2,3,4,5], [0,9,8,0,0]]
    assert batch['attention_mask'] == [[1,1,1,1,1], [0,1,1,0,0]]
    assert batch['response_mask'] == [[1,0,1], [1,0,0]]
    assert batch['position_ids'][1] == [0,0,1,0,0]

def test_grpo_request_requires_real_attested_source_and_comparable_designer_reward():
    from rift.schemas import Tokens
    from rift.training.contracts import Rollout, UpdateRequest
    from rift.verl.actor_backend import validate_request
    cp = Checkpoint('service', 'v1', 'tok', 'weights')
    tokens = Tokens((1,), (2,), (True,))
    rows = tuple(Rollout(tokens, cp, 'task', f'episode{i}', float(i), 'business_outcome_v2',
        'group', f'call{i}', f'episode{i}') for i in range(2))
    request = UpdateRequest(cp, 'package', rows, 'grpo')
    class Deny:
        def verify_rollout(self, row, checkpoint): return False
    class FakeVerifier:
        def verify_rollout(self, row, checkpoint): return True
        def verify_group(self, rows, checkpoint): return True
    with pytest.raises(ValueError, match='on-policy'):
        validate_request(request, 'service', Deny())
    assert validate_request(request, 'service', FakeVerifier()) is request
    with pytest.raises(ValueError, match='business-only'):
        validate_request(UpdateRequest(cp, 'package', tuple(
            Rollout(tokens, cp, 'task', f'episode{i}', 1., 'strict_task_success',
                'group', f'call{i}', f'episode{i}') for i in range(2)), 'grpo'), 'service', FakeVerifier())

def test_replay_reward_verifier_recomputes_saved_scores():
    from rift.verl.replay_evidence import RecomputedReplayRewardVerifier
    from test_core import TRAIN, REPLAY, Scorer, Store, Verifier, completed, demo
    from rift.replay.bank import ReplayBank
    from rift.replay.splits import SplitRegistry
    from rift.replay.reward import score_update
    verifier = Verifier()
    snapshot = ReplayBank([demo()], verifier=verifier).freeze([TRAIN], SplitRegistry([TRAIN, REPLAY]))
    update = completed()
    reward = score_update(update, snapshot, Scorer(), Store(), update_verifier=verifier, replay_verifier=verifier)
    evidence = RecomputedReplayRewardVerifier(lambda ref: snapshot if ref == snapshot.digest else None, Scorer(), Store(), verifier, verifier)
    assert evidence.verify_reward(reward, update)
    assert not RecomputedReplayRewardVerifier(lambda ref: None, Scorer(), Store(), verifier, verifier).verify_reward(reward, update)

def test_wrong_verl_version_rejected_without_importing_trainer(monkeypatch):
    from rift.verl import version
    monkeypatch.setattr(version, 'version', lambda name: '0.7.0')
    with pytest.raises(RuntimeError, match='expected VERL 0.9.1'):
        version.require_verl_version()

def test_verl_weight_sync_is_bound_to_generation_global_steps(tmp_path):
    from rift.verl.serving import VerlServingAttestor, VerlCheckpointWeightSync
    from rift.verl.generation_ledger import GenerationLedger
    from rift.training.trace import ExactPolicyTrace, digest
    from rift.schemas import Tokens
    worker = object()
    class Manager:
        actor_wg = worker
        def __init__(self): self.steps = []
        async def update_weights(self, *, global_steps):
            self.steps.append(global_steps)
            return {}  # Official v0.9.1 manager does not return a model hash.
    cp = Checkpoint('service', '7', 'tok', 'weights-digest')
    class CheckpointStore:
        artifact_digests = {cp:'artifact-7'}
        def verify(self, checkpoint): return checkpoint == cp
    checkpoint_store = CheckpointStore()
    attestor = VerlServingAttestor(tmp_path / 'sync-receipts', checkpoint_store)
    manager = Manager(); sync = VerlCheckpointWeightSync(manager, attestor)
    assert sync.synchronize(worker, cp) is True and manager.steps == [7]
    assert attestor.verify_serving(cp)
    fields = {'global_steps':7, 'min_global_steps':7, 'max_global_steps':7}
    assert attestor.verify_generation(cp, fields)
    assert not attestor.verify_generation(cp, {**fields, 'max_global_steps':8})
    assert not attestor.verify_generation(cp, {'global_steps':None, 'min_global_steps':7, 'max_global_steps':7})
    with pytest.raises(ValueError, match='actor worker group'):
        sync.synchronize(object(), cp)
    with pytest.raises(ValueError, match='revision'):
        sync.synchronize(worker, Checkpoint('service','v8','tok','another-digest'))

    ledger = GenerationLedger(tmp_path / 'generation-ledger', attestor)
    generation_id = 'a' * 32
    request = {'generation_id':generation_id, 'rollout_id':'rollout-a', 'group_id':'g',
        'prompt_ids':[1], 'sampling_params':{'temperature':1.0,'seed':4}}
    response = {'generated_token_ids':[2], 'log_probs':[-0.1], 'serving_fields':fields}
    ledger.record_generation(generation_id, cp, 'rollout-a', 'g', [1],
        request['sampling_params'], [2], [-0.1], fields)
    trace = ExactPolicyTrace('service','rollout-a','decision-a',cp,Tokens((1,),(2,),(True,)),
        digest(request),digest(response),'fixture','evidence')
    assert ledger.verify_capture(trace, request, response)
    response['serving_fields'] = {**fields, 'global_steps':6, 'min_global_steps':6, 'max_global_steps':6}
    assert not ledger.verify_capture(trace, request, response)

def test_mock_coordinator_recovers_without_repeating_service_update():
    """All backend results here are fake test records, not trained weights."""
    from test_core import TRAIN, REPLAY, T, Scorer, Store, Verifier, completed, demo
    from rift.loop import LoopState, Stage
    from rift.tasks import Certification, CertificationStatus
    from rift.replay.bank import ReplayBank
    from rift.replay.splits import SplitRegistry
    from rift.training.records import DesignerFeedback
    from rift.training.contracts import CompletedUpdate
    from rift.verl.coordinator import run_round
    service_update = completed(); verifier = Verifier()
    state = LoopState('package', ('p',)).certify([Certification('p', CertificationStatus.CERTIFIED, 't1')]).ready(service_update.request)
    snapshot = ReplayBank([demo()], verifier=verifier).freeze([TRAIN], SplitRegistry([TRAIN, REPLAY]))
    class FakeService:
        calls = 0
        def train(self, request): self.calls += 1; return service_update
    class FakeDesigner:
        def train(self, request):
            after = Checkpoint('designer', 'v2', 'tok', 'fake-after')
            return CompletedUpdate(request, after, 'fake-designer-run', 'fake-evidence')
    class BadScorer:
        def score(self, checkpoint, batch): return [[float('nan')] * len(row.response) for row in batch]
    service = FakeService(); saved = []
    kwargs = dict(service_backend=service, designer_backend=FakeDesigner(), replay_snapshot=snapshot,
        checkpoint_store=Store(), update_verifier=verifier, replay_verifier=verifier,
        reward_verifier=verifier, action_verifier=verifier,
        designer_checkpoint=Checkpoint('designer','v1','tok','fake-before'),
        feedback_provider=lambda state: [DesignerFeedback(T, service_update, state.replay_reward, 'fake-rollout')],
        save_state=saved.append, designer_batch_id='batch', designer_algorithm='policy_gradient')
    with pytest.raises(ValueError, match='log probability'):
        run_round(state, scorer=BadScorer(), **kwargs)
    assert saved[-1].stage is Stage.SERVICE_TRAINING_COMPLETED and service.calls == 1
    result = run_round(saved[-1], scorer=Scorer(), **kwargs)
    assert result.stage is Stage.DESIGNER_TRAINING_COMPLETED and service.calls == 1

def test_native_usami_session_steps_through_real_runner_without_model():
    import json
    from rift_usami_fixtures import frozen, refund_task, refund_profile, ScriptedCustomer, ScriptedRefundService
    from usami_v2.runtime.runner import EpisodeBudget
    from rift.benchmark.usami import public_episode
    from rift.verl.usami_session import NativeUsamiSession
    world, aliases = frozen()
    script = ScriptedRefundService(world, aliases)
    class FakeParser:
        async def extract_tool_calls(self, token_ids, schemas):
            action = script.decide(None)
            if action.kind == 'operation':
                op = action.operation
                return '', [SimpleNamespace(name=op.name, arguments=json.dumps({'scope_id':op.scope_id,
                    'resource_id':op.resource_id, 'params':op.params}), tool_call_id=None)]
            return '', [SimpleNamespace(name='respond_to_customer', arguments=json.dumps({'message':action.message,
                'present_quote_id':action.present_quote_id, 'present_for':action.present_for,
                'subject_ids':list(action.subject_ids)}), tool_call_id=None)]
    async def exercise():
        session = NativeUsamiSession(world, refund_task(world, aliases), refund_profile(aliases), ScriptedCustomer(),
            FakeParser(), budget=EpisodeBudget(max_customer_turns=3,max_service_decisions=10,max_service_tool_calls=8,max_model_calls=10))
        try:
            messages, tools = await session.start()
            assert messages[0]['role'] == 'system' and tools
            done = False; step = 0
            while not done:
                _, additions, done = await session.advance([step + 10], [1, step + 1], {'temperature':1.0})
                if not done: assert isinstance(additions, list)
                step += 1
            episode = await session.finish()
            assert public_episode(episode).outcome is Outcome.SUCCESS
            assert len(episode.model_calls) == step
            assert episode.model_calls[0].response['generated_token_ids'] == [10]
        finally:
            await session.abort()
    asyncio.run(exercise())

def test_native_budget_cancellation_is_unknown_and_replayable():
    import json
    from rift_usami_fixtures import frozen, refund_task, refund_profile, ScriptedCustomer, ScriptedRefundService
    from usami_v2.runtime.runner import EpisodeBudget, replay_episode
    from rift.benchmark.usami import public_episode
    from rift.verl.usami_session import NativeUsamiSession
    world, aliases = frozen()
    script = ScriptedRefundService(world, aliases)
    class Parser:
        async def extract_tool_calls(self, tokens, schemas):
            op = script.decide(None).operation
            return '', [SimpleNamespace(name=op.name, arguments=json.dumps({
                'scope_id': op.scope_id, 'resource_id': op.resource_id, 'params': op.params}), tool_call_id=None)]
    async def exercise():
        session = NativeUsamiSession(world, refund_task(world, aliases), refund_profile(aliases),
            ScriptedCustomer(), Parser(), budget=EpisodeBudget(max_customer_turns=3,
            max_service_decisions=10, max_service_tool_calls=8, max_model_calls=10))
        try:
            await session.start()
            _, _, done = await session.advance([10], [1], {'max_tokens': 1})
            assert not done
            await session.cancel_for_budget()
            episode = await session.finish()
            assert episode.status == 'cancelled'
            assert replay_episode(episode).verified
            assert public_episode(episode).outcome is Outcome.UNKNOWN
            assert episode.model_calls[0].response['generated_token_ids'] == [10]
        finally:
            await session.abort()
    asyncio.run(exercise())

def test_native_capture_builds_two_distinct_grouped_trajectories(tmp_path, monkeypatch):
    """Fake parser/attestor only; native Engine and Replay are real CPU code."""
    import json
    from rift_usami_fixtures import frozen, refund_task, refund_profile, ScriptedCustomer, ScriptedRefundService
    from usami_v2.runtime.runner import EpisodeBudget
    from rift.verl.usami_session import NativeUsamiSession
    from rift.verl.capture_store import NativeCaptureStore
    from rift.training.service_data import prepare_service_trajectory_update
    world, aliases = frozen(); task = refund_task(world, aliases); cp = Checkpoint('service','v1','tok','weights')
    class FakeAttestor:
        def verify_capture(self, trace, request, response): return True
    store = NativeCaptureStore(tmp_path / 'captures', serialization='fixture', attestor=FakeAttestor())
    async def one(seed):
        script = ScriptedRefundService(world, aliases)
        class FakeParser:
            async def extract_tool_calls(self, tokens, schemas):
                action = script.decide(None)
                if action.kind == 'operation':
                    op = action.operation
                    return '', [SimpleNamespace(name=op.name, arguments=json.dumps({'scope_id':op.scope_id,
                        'resource_id':op.resource_id, 'params':op.params}), tool_call_id=None)]
                return '', [SimpleNamespace(name='respond_to_customer', arguments=json.dumps({'message':action.message,
                    'present_quote_id':action.present_quote_id,'present_for':action.present_for,
                    'subject_ids':list(action.subject_ids)}), tool_call_id=None)]
        session = NativeUsamiSession(world, task, refund_profile(aliases), ScriptedCustomer(), FakeParser(),
            group_id='same-task-group', budget=EpisodeBudget(max_customer_turns=3,max_service_decisions=10,
                max_service_tool_calls=8,max_model_calls=10), group_size=2)
        cap = ContinuousCapture([1], cp)
        try:
            await session.start(); done = False; step = 0
            while not done:
                prefix = list(cap.current); action = [step + 10]
                cap.add_generation(prefix, action, None, prefix + action, list(cap.mask) + [1])
                _, _, done = await session.advance(action, prefix, {'temperature':1.0, 'seed':seed})
                if not done: cap.add_context(list(cap.current) + [99], list(cap.mask) + [0])
                step += 1
            episode = await session.finish()
            await store.persist(episode, cp, tuple(cap.decisions), cap.output())
            assert store.verify_trajectory(episode, cp)
            return episode
        finally:
            await session.abort()
    async def both(): return [await one(1042), await one(1043)]
    episodes = asyncio.run(both())
    assert episodes[0].record_hash != episodes[1].record_hash
    update = prepare_service_trajectory_update('package', {task.record_hash}, episodes, cp, store)
    assert len(update.rollouts) == 2 and all(r.group_id == 'same-task-group' for r in update.rollouts)
    assert any(not bit for bit in update.rollouts[0].tokens.mask)
    with pytest.raises(ValueError, match='incomplete Service GRPO group'):
        prepare_service_trajectory_update('package', {task.record_hash}, episodes[:1], cp, store)
    monkeypatch.setattr('rift.training.service_data.public_business_outcome', lambda episode, verified: Outcome.UNKNOWN)
    with pytest.raises(ValueError, match='complete known-outcome GRPO group'):
        prepare_service_trajectory_update('package', {task.record_hash}, episodes, cp, store)
    monkeypatch.undo()
    from dataclasses import replace
    from rift.verl.rollout_evidence import NativeServiceRolloutVerifier, unmodified_sampling
    from rift.verl.actor_backend import validate_request
    episode_map = {episode.record_hash: episode for episode in episodes}
    provenance = NativeServiceRolloutVerifier(episode_map.get, store)
    assert validate_request(update, 'service', provenance) == update
    assert not provenance.verify_rollout(replace(update.rollouts[0], group_id='forged'), cp)
    assert not unmodified_sampling({'temperature':0.7})
    from rift.schemas import Identity, Split
    from rift.replay.splits import SplitRegistry
    from rift.replay.trajectory import extract_demonstrations
    # Test extraction in isolation; this deliberately shared-world fixture must
    # never be admitted to a training/replay snapshot as independent evidence.
    replay_identity = Identity(world.state_hash, task.record_hash, 'replay-family', 'replay-lineage', Split.REPLAY)
    decisions = extract_demonstrations(episodes[0], replay_identity, cp, SplitRegistry([replay_identity]),
        store.decision_verifier(episodes[0], cp))
    assert decisions and decisions[0].tokens.prompt == (1,)
    capture_path = store._path(episodes[0])
    raw = json.loads(capture_path.read_text(encoding='utf-8'))
    raw['trajectory']['response'][0] += 1
    capture_path.write_text(json.dumps(raw), encoding='utf-8')
    assert not store.verify_trajectory(episodes[0], cp)

def test_verl_extension_with_real_usami_session_and_fake_server(tmp_path, monkeypatch):
    """The native business runner is real; token generation and templates are test doubles."""
    import json
    from rift_usami_fixtures import frozen, refund_task, refund_profile, ScriptedCustomer, ScriptedRefundService
    from usami_v2.runtime.runner import EpisodeBudget
    from rift.verl.usami_session import NativeUsamiSession
    from rift.verl.capture_store import NativeCaptureStore
    world, aliases = frozen(); script = ScriptedRefundService(world, aliases)
    class FakeParser:
        async def extract_tool_calls(self, tokens, schemas):
            action = script.decide(None)
            if action.kind == 'operation':
                op = action.operation
                return '', [SimpleNamespace(name=op.name, arguments=json.dumps({'scope_id':op.scope_id,'resource_id':op.resource_id,'params':op.params}), tool_call_id=None)]
            return '', [SimpleNamespace(name='respond_to_customer', arguments=json.dumps({'message':action.message,'present_quote_id':action.present_quote_id,'present_for':action.present_for,'subject_ids':list(action.subject_ids)}), tool_call_id=None)]
    class Base:
        def __init__(self): self.rollout_config = SimpleNamespace(response_length=100, n=4); self.tokenizer = None
        async def ct_build_initial_tokens(self, messages, tools=None): return [1]
        async def ct_merge_assistant_token(self, old, output, mask, logs, assistant_logprobs=None):
            return SimpleNamespace(token_ids=old + output), mask + [1] * len(output), None
        async def ct_merge_context_msg(self, prev, new, old, mask, logs, tools=None):
            return SimpleNamespace(token_ids=old + [99]), mask + [0], None
    class Output:
        def __init__(self, **fields): self.__dict__.update(fields)
    for name in ('verl','verl.experimental','verl.experimental.agent_loop'):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    module = types.ModuleType('verl.experimental.agent_loop.agent_loop')
    module.AgentLoopBase = Base; module.AgentLoopOutput = Output
    monkeypatch.setitem(sys.modules, module.__name__, module)
    import rift.verl.agent_loop as extension
    monkeypatch.delattr(extension, 'UsamiServiceAgentLoop', raising=False)
    loop = extension.UsamiServiceAgentLoop()
    class FakeServer:
        def __init__(self): self.i = 0; self.sampling = []
        async def generate(self, request_id, *, prompt_ids, sampling_params):
            self.i += 1
            self.sampling.append(dict(sampling_params))
            return SimpleNamespace(token_ids=[self.i+10], log_probs=None,
                extra_fields={'global_steps':1,'min_global_steps':1,'max_global_steps':1})
    loop.server_manager = FakeServer()
    class FakeVerifier:
        def verify(self, checkpoint): return True
        def verify_capture(self, trace, request, response): return True
        def verify_serving(self, checkpoint): return True
        def verify_generation(self, checkpoint, fields): return fields == {'global_steps':1,'min_global_steps':1,'max_global_steps':1}
        def record_generation(self, *args): pass
    cp = Checkpoint('service','v1','tok','fake-weights'); verifier = FakeVerifier()
    from rift.verl.generation_ledger import GenerationLedger
    ledger = GenerationLedger(tmp_path / 'generation-ledger', verifier)
    sink = NativeCaptureStore(tmp_path / 'captures', serialization='fake-template', attestor=ledger)
    session = NativeUsamiSession(world, refund_task(world,aliases), refund_profile(aliases), ScriptedCustomer(), FakeParser(),
        group_id='group', sampling_seed=23, sampling_scope='paired-probe-scope',
        rollout_id='native-rollout', group_size=4, task_ref='opaque', rift_mode='train', customer_seed=91,
        budget=EpisodeBudget(max_customer_turns=3,max_service_decisions=10,max_service_tool_calls=8,max_model_calls=10))
    class Host:
        async def open_rollout(self, task_ref, group_id, session_id, tokenizer, **kwargs):
            return session, cp, verifier, sink
    loop.rift_host_provider = Host()
    async def run():
        output = await loop.run({'temperature':1.0}, rift_task_ref='opaque', uid='group', session_id=0,
            rift_request_id='request-1', rift_mode='train', rift_group_size=4,
            rift_rollout_id='native-rollout', rift_sampling_seed=23, rift_customer_seed=91,
            rift_expected_checkpoint={'policy':'service','revision':'v1','tokenizer':'tok','weights_digest':'fake-weights'})
        episode = await session.finish()
        assert output.reward_score == 1.0 and sink.verify_trajectory(episode, cp)
        assert len(episode.model_calls) == loop.server_manager.i
        assert 0 in output.response_mask and 1 in output.response_mask
    asyncio.run(run())
    from rift.verl.sampling import rollout_seed
    assert [row['seed'] for row in loop.server_manager.sampling] == [
        rollout_seed(23, 'paired-probe-scope', turn) for turn in range(loop.server_manager.i)]
    script = ScriptedRefundService(world, aliases)
    session = NativeUsamiSession(world, refund_task(world,aliases), refund_profile(aliases), ScriptedCustomer(), FakeParser(),
        group_id='group', sampling_seed=23, sampling_scope='paired-probe-scope',
        rollout_id='native-over-budget', group_size=4, task_ref='opaque', rift_mode='train', customer_seed=91,
        budget=EpisodeBudget(max_customer_turns=3,max_service_decisions=10,max_service_tool_calls=8,max_model_calls=10))
    loop.server_manager = FakeServer()
    loop.rollout_config.response_length = 2
    async def overflow():
        output = await loop.run({'temperature':1.0}, rift_task_ref='opaque', uid='group', session_id=0,
            rift_request_id='request-over-budget', rift_mode='train', rift_group_size=4,
            rift_rollout_id='native-over-budget', rift_sampling_seed=23, rift_customer_seed=91,
            rift_expected_checkpoint={'policy':'service','revision':'v1','tokenizer':'tok','weights_digest':'fake-weights'})
        episode = await session.finish()
        assert output.reward_score is None and output.extra_fields['rift_outcome'] == Outcome.UNKNOWN.value
        assert output.response_ids == [11] and output.response_mask == [1]
        assert sink.verify_trajectory(episode, cp)
    asyncio.run(overflow())
    monkeypatch.delattr(extension, 'UsamiServiceAgentLoop', raising=False)

def test_designer_on_policy_capture_precedes_delayed_reward(tmp_path):
    """Simulated server and attestor validate wiring, not real policy provenance."""
    import json
    from rift.benchmark.public_feedback import PublicSummary, PublicEvent
    from rift.banks.records import FailureObservation
    from rift.tasks import Proposal
    from rift.verl.designer import DesignerCaptureStore, DesignerSampler
    class FakeServer:
        async def generate(self, request_id, *, prompt_ids, sampling_params):
            assert prompt_ids == [1, 2]
            return SimpleNamespace(token_ids=[3, 4], log_probs=[-.3, -.4],
                extra_fields={'global_steps':1,'min_global_steps':1,'max_global_steps':1})
    class FakeVerifier:
        def verify(self, checkpoint): return True
        def verify_capture(self, trace, request, response): return True
        def verify_serving(self, checkpoint): return True
        def verify_generation(self, checkpoint, fields): return fields == {'global_steps':1,'min_global_steps':1,'max_global_steps':1}
        def record_generation(self, *args): pass
    async def encode(failures, ability_options):
        assert failures[0].task_ref == 'F001'
        assert isinstance(failures[0].summary, PublicSummary)
        assert ability_options == ()
        return [1, 2]
    async def parse(tokens): return [Proposal('p', 'hypothesis', 'request', '{}',
        'NEW', ('F001',))]
    from rift.verl.generation_ledger import GenerationLedger
    verifier = FakeVerifier(); ledger = GenerationLedger(tmp_path / 'generation-ledger', verifier)
    store = DesignerCaptureStore(tmp_path, attestor=ledger)
    sampler = DesignerSampler(FakeServer(), encode, parse, verifier, store, serialization='fixture')
    cp = Checkpoint('designer', 'v1', 'tok', 'weights')
    failure = FailureObservation('F001', PublicSummary((PublicEvent('user','request'),), Outcome.FAILURE))
    pending = asyncio.run(sampler.sample([failure], (), cp,
        'package', {'temperature':1.0}, group_id='group'))
    assert pending.trace.tokens.prompt == (1, 2) and pending.trace.tokens.response == (3, 4)
    assert store.verify_designer_action(pending.trace.tokens, cp, 'package', pending.rollout_id, 'group')
    assert not store.verify_designer_action(pending.trace.tokens, cp, 'package', pending.rollout_id, 'forged-group')
    assert not store.verify_designer_action(pending.trace.tokens, cp, 'other', pending.rollout_id, 'group')
    assert store.verify_designer_proposal(pending.rollout_id, pending.proposals[0],
        visible_failure_refs=('F001',), ability_options=())
    from dataclasses import replace
    assert not store.verify_designer_proposal(pending.rollout_id,
        replace(pending.proposals[0], diagnosis_task_refs=('F002',)))
    from rift.abilities.records import AbilityPrompt
    async def encode_target(failures, ability_options):
        assert ability_options[0].description == 'refund status lookup'
        return [1, 2]
    async def parse_target(tokens): return [Proposal('p2', 'refund status lookup',
        'request', '{}', 'A000001', (), ('F001',))]
    targeted_sampler = DesignerSampler(FakeServer(), encode_target, parse, verifier,
        store, serialization='fixture')
    targeted_sampler.proposal_parser = parse_target
    option = AbilityPrompt('A000001', 'refund status lookup', ('F001',))
    targeted = asyncio.run(targeted_sampler.sample([failure], (option,), cp,
        'package-targeted', {'temperature':1.0}))
    assert targeted.proposals[0].ability_id == 'A000001'
    assert store.load_pending(targeted.rollout_id) == targeted
    assert store.verify_designer_proposal(targeted.rollout_id, targeted.proposals[0],
        visible_failure_refs=('F001',), ability_options=(option,))
    from test_core import TRAIN, REPLAY, Scorer, Store, Verifier, completed, demo
    from rift.replay.bank import ReplayBank
    from rift.replay.splits import SplitRegistry
    from rift.replay.reward import score_update
    from rift.training.records import designer_rollout
    from rift.verl.rollout_evidence import DesignerRolloutVerifier
    source = Verifier(); update = completed()
    snapshot = ReplayBank([demo()], verifier=source).freeze([TRAIN], SplitRegistry([TRAIN, REPLAY]))
    reward = score_update(update, snapshot, Scorer(), Store(), update_verifier=source, replay_verifier=source)
    row = designer_rollout(pending.trace.tokens, cp, 'package', reward, update, 'group',
        designer_rollout_id=pending.rollout_id, update_verifier=source, reward_verifier=source, action_verifier=store)
    evidence = DesignerRolloutVerifier(store, lambda ref: (update, reward) if ref == update.run_id else None, source, source)
    assert evidence.verify_rollout(row, cp)
    path = store._path(pending.rollout_id)
    record = json.loads(path.read_text(encoding='utf-8'))
    record['proposals'][0]['hypothesis'] = 'tampered'
    path.write_text(json.dumps(record), encoding='utf-8')
    assert not store.verify_designer_action(pending.trace.tokens, cp, 'package', pending.rollout_id, 'group')

def test_native_session_abort_unblocks_waiting_runner():
    from rift_usami_fixtures import frozen, refund_task, refund_profile, ScriptedCustomer
    from usami_v2.runtime.runner import EpisodeBudget
    from rift.verl.usami_session import NativeUsamiSession
    world, aliases = frozen()
    async def exercise():
        session = NativeUsamiSession(world, refund_task(world, aliases), refund_profile(aliases),
            ScriptedCustomer(), object(), budget=EpisodeBudget(max_model_calls=10))
        await session.start()
        await asyncio.wait_for(session.abort(), timeout=5)
        assert session._worker.done()
    asyncio.run(exercise())

def test_host_provider_resolves_private_task_inside_worker(monkeypatch):
    from rift.verl.host_provider import NativeHostProvider
    from rift.verl.usami_session import NativeUsamiSession
    world, task, profile = object(), SimpleNamespace(partition='train'), object()
    class Resolver:
        def resolve(self, ref):
            assert ref == 'opaque'; return world, task, profile, ()
    class Authorizer:
        def verify_task(self, ref, w, t, p): return w is world and t is task and p is profile
    class Customers:
        def create(self, ref, *, seed): return ('native-customer', seed)
    cp = Checkpoint('service','v1','tok','weights')
    class Source:
        def current(self): return cp
    class Store:
        def verify(self, value): return value == cp
    class Sink: pass
    marker = SimpleNamespace(group_id='group')
    monkeypatch.setattr(NativeUsamiSession, 'with_verl_parser', lambda *a, **kw: marker)
    provider = NativeHostProvider(Resolver(), Authorizer(), Customers(), Source(), Store(), Sink(),
        budget=object(), parser_format='qwen')
    session, checkpoint, _, _ = asyncio.run(provider.open_rollout('opaque','group',0,None,
        sampling_seed=5, customer_seed=9, rollout_id='rollout', expected_checkpoint=cp,
        group_size=4))
    assert session is marker and checkpoint == cp
    task.partition = 'heldout'
    with pytest.raises(ValueError, match='training task'):
        asyncio.run(provider.open_rollout('opaque','group',0,None, sampling_seed=5,
            customer_seed=9, rollout_id='rollout', expected_checkpoint=cp, group_size=4))

def test_probe_seed_reaches_customer_and_is_checkpoint_independent(monkeypatch):
    from rift.verl.host_provider import NativeHostProvider
    from rift.verl.usami_session import NativeUsamiSession
    from rift.schemas import DataPurpose, Checkpoint
    task = SimpleNamespace(partition='train')
    world, profile = object(), object()
    class Resolver:
        def resolve(self, ref): return world, task, profile, ()
        def purpose(self, ref): return DataPurpose.BANK_CANDIDATE
    class Authorizer:
        def verify_task(self, ref, w, t, p, *, purpose=None): return purpose is DataPurpose.BANK_CANDIDATE
    class Customers:
        def __init__(self): self.seeds = []
        def create(self, ref, *, seed):
            self.seeds.append(seed)
            return SimpleNamespace(seed=seed)
    class Store:
        def verify(self, checkpoint): return True
    class Source: pass
    class Sink: pass
    captures = []
    def make_session(*args, **kwargs):
        session = SimpleNamespace(**kwargs)
        captures.append(session)
        return session
    monkeypatch.setattr(NativeUsamiSession, 'with_verl_parser', make_session)
    customers = Customers()
    provider = NativeHostProvider(Resolver(), Authorizer(), customers, Source(), Store(), Sink(),
        budget=object(), parser_format='qwen')
    before = Checkpoint('service','before','tok','weights-before')
    after = Checkpoint('service','after','tok','weights-after')
    first = asyncio.run(provider.open_probe('failure-F1', before, 41, 'eval-v1', None,
        protocol_digest='eval-digest'))[0]
    second = asyncio.run(provider.open_probe('failure-F1', after, 41, 'eval-v1', None,
        protocol_digest='eval-digest'))[0]
    assert customers.seeds == [41, 41]
    assert first.rollout_id != second.rollout_id
    assert first.sampling_seed == second.sampling_seed == 41
    assert first.sampling_scope == second.sampling_scope

def test_parser_runtime_failure_is_retained_as_unknown_native_episode(tmp_path):
    from rift_usami_fixtures import frozen, refund_task, refund_profile, ScriptedCustomer
    from usami_v2.runtime.runner import EpisodeBudget
    from rift.benchmark.usami import public_episode
    from rift.verl.usami_session import NativeUsamiSession
    from rift.verl.capture import ContinuousCapture
    from rift.verl.capture_store import NativeCaptureStore
    world, aliases = frozen()
    class BrokenParser:
        async def extract_tool_calls(self, token_ids, schemas):
            raise RuntimeError('tokenizer parser runtime failed')
    class Attestor:
        def verify_capture(self, trace, request, response): return True
    checkpoint = Checkpoint('service','v1','tok','weights')
    store = NativeCaptureStore(tmp_path / 'captures', serialization='fake-template', attestor=Attestor())
    async def run():
        session = NativeUsamiSession(world, refund_task(world, aliases), refund_profile(aliases),
            ScriptedCustomer(), BrokenParser(), budget=EpisodeBudget(max_model_calls=10))
        try:
            await session.start()
            capture = ContinuousCapture([1], checkpoint)
            capture.add_generation([1], [42], None, [1, 42], [True])
            _, _, done = await session.advance([42], [1], {'temperature':1.0, 'seed':9},
                generation_id='a' * 32, serving_fields={'global_steps':1,'min_global_steps':1,'max_global_steps':1})
            assert done
            episode = await session.finish()
            await store.persist(episode, checkpoint, tuple(capture.decisions), capture.output())
            return episode
        finally:
            await session.abort()
    episode = asyncio.run(run())
    assert episode.status == 'cancelled' and episode.replay_verified
    assert len(episode.model_calls) == 1
    assert episode.model_calls[0].response['generated_token_ids'] == [42]
    assert public_episode(episode).outcome is Outcome.UNKNOWN
    assert store.verify_trajectory(episode, checkpoint)

def test_malformed_service_tool_is_a_native_business_failure_not_runtime_abort():
    from rift_usami_fixtures import frozen, refund_task, refund_profile, ScriptedCustomer
    from usami_v2.runtime.runner import EpisodeBudget
    from rift.benchmark.usami import public_episode
    from rift.verl.usami_session import NativeUsamiSession
    world, aliases = frozen()
    class MalformedParser:
        async def extract_tool_calls(self, token_ids, schemas):
            return '', [SimpleNamespace(name='not_a_usami_tool', arguments='{}', tool_call_id='bad')]
    async def run():
        session = NativeUsamiSession(world, refund_task(world, aliases), refund_profile(aliases),
            ScriptedCustomer(), MalformedParser(), budget=EpisodeBudget(max_model_calls=10))
        try:
            await session.start()
            _, _, done = await session.advance([17], [1], {'temperature':1.0}, generation_id='b' * 32)
            assert done
            return await session.finish()
        finally:
            await session.abort()
    episode = asyncio.run(run())
    assert episode.status == 'service_protocol_error' and episode.replay_verified
    assert len(episode.model_calls) == 1
    assert public_episode(episode).outcome is Outcome.FAILURE

def test_update_ledger_detects_checkpoint_and_receipt_tampering(tmp_path):
    """Dummy files and mocked metrics only test ledger integrity."""
    from rift.verl.checkpoints import FSDPCheckpointStore, model_digest
    from rift.verl.update_ledger import UpdateLedger
    from rift.training.contracts import UpdateRequest, CompletedUpdate
    from rift.training.records import service_rollout
    from rift.schemas import Tokens
    paths=[]
    for tag in ('before','after'):
        path=tmp_path / tag; path.mkdir(); paths.append(path)
        for kind in ('model','optim','extra_state'):
            (path / f'{kind}_world_size_1_rank_0.pt').write_bytes((tag + kind).encode())
    before=Checkpoint('service','v1','tok',model_digest(paths[0],1))
    after=Checkpoint('service','v2','tok',model_digest(paths[1],1))
    store=FSDPCheckpointStore({before:paths[0],after:paths[1]},world_size=1)
    row=service_rollout(Tokens((1,),(2,),(True,)),before,'task','episode',Outcome.SUCCESS,response_id='call')
    request=UpdateRequest(before,'package',(row,),'policy_gradient')
    ledger=UpdateLedger(tmp_path / 'receipts',store)
    ref=ledger.record_update('run1',request,after,{'mock':True})
    update=CompletedUpdate(request,after,'run1',ref)
    assert ledger.verify_update(update)
    (paths[1] / 'optim_world_size_1_rank_0.pt').write_bytes(b'tampered')
    assert not ledger.verify_update(update)
