"""Offline worker mock: no VERL, NPU, real model, or training is executed."""
import sys
import types
import pytest

torch = pytest.importorskip('torch')
from rift.schemas import Checkpoint, Tokens
from rift.training.contracts import Rollout, UpdateRequest
from rift.verl.actor_backend import VerlServiceRLBackend, VerlDesignerRLBackend
from rift.verl.checkpoints import FSDPCheckpointStore, model_digest

@pytest.mark.parametrize('resident_state', [False, True])
@pytest.mark.parametrize('role,algorithm', [('service', 'grpo'), ('designer', 'grpo'), ('designer', 'reinforce_plus_plus')])
def test_mock_actor_worker_wiring(tmp_path, monkeypatch, role, algorithm, resident_state):
    import numpy as np
    monkeypatch.setattr('rift.verl.actor_backend.require_verl_version', lambda: None)
    class Proto:
        @classmethod
        def from_dict(cls, tensors):
            instance = cls(); instance.data = dict(tensors); return instance
        def to_tensordict(self): return dict(self.data)
    modules = {}
    for name in ['verl','verl.trainer','verl.trainer.ppo','verl.trainer.ppo.core_algos',
                 'verl.utils','verl.utils.tensordict_utils','verl.workers','verl.workers.utils','verl.workers.utils.padding']:
        modules[name] = types.ModuleType(name); monkeypatch.setitem(sys.modules, name, modules[name])
    modules['verl'].DataProto = Proto
    modules['verl.trainer.ppo.core_algos'].compute_grpo_outcome_advantage = lambda rewards, mask, index: (rewards * mask, rewards * mask)
    modules['verl.trainer.ppo.core_algos'].compute_reinforce_plus_plus_outcome_advantage = lambda rewards, mask, config: (rewards * mask, rewards * mask)
    tu = modules['verl.utils.tensordict_utils']
    tu.assign_non_tensor = lambda batch, **kwargs: batch.update(kwargs)
    tu.get = lambda batch, key: batch[key]
    modules['verl.utils'].tensordict_utils = tu
    modules['verl.workers.utils.padding'].left_right_2_no_padding = lambda batch: batch
    modules['verl.workers.utils.padding'].no_padding_2_padding = lambda tensor, batch: tensor
    def write_fake_shards(path, tag):
        path.mkdir()
        for kind in ('model','optim','extra_state'):
            (path / f'{kind}_world_size_1_rank_0.pt').write_bytes((tag+kind).encode())
    before_dir = tmp_path / 'before'; write_fake_shards(before_dir, 'before')
    checkpoint = Checkpoint(role, '1', 'tok', model_digest(before_dir, 1))
    store = FSDPCheckpointStore({checkpoint: before_dir}, world_size=1)
    class FakeWorker:
        def __init__(self): self.load_calls = 0
        def load_checkpoint(self, path): self.loaded = path; self.load_calls += 1
        def compute_log_prob(self, batch): return {'log_probs': torch.zeros((2, 1))}
        def update_actor(self, batch):
            assert 'advantages' in batch and 'old_log_probs' in batch
            return {'mock': True}
        def save_checkpoint(self, path, global_step):
            assert global_step == 2
            write_fake_shards(__import__('pathlib').Path(path), 'after')
    class Verifier:
        def verify_rollout(self, row, checkpoint): return True
        def verify_group(self, rows, checkpoint): return True
        def verify_resident_training_state(self, role, worker, checkpoint): return resident_state
        def verify_runtime_environment(self, role, worker): return True
        def verify_runtime(self, role, worker, checkpoint): return True
        def verify_update(self, update): return True
        def synchronize(self, worker, checkpoint): return True
        def record_update(self, run_id, request, after, metrics): return 'fake-test-evidence'
    tokens = Tokens((1,), (2,), (True,))
    source = 'business_outcome_v2' if role == 'service' else 'replay:baseline:snapshot:sum'
    rows = tuple(Rollout(tokens, checkpoint, 'task' if role == 'service' else f'package{i}',
        f'episode{i}', float(i), source, 'group' if algorithm == 'grpo' else None, f'call{i}', f'episode{i}') for i in range(2))
    request = UpdateRequest(checkpoint, 'package', rows, algorithm)
    verifier = Verifier()
    cls = VerlServiceRLBackend if role == 'service' else VerlDesignerRLBackend
    backend = cls(FakeWorker(), store, verifier, verifier, verifier, verifier, verifier,
        output_dir=tmp_path / 'after', after_revision='2', run_id='fake-run',
        pad_token_id=0, mini_batch_size=2, epochs=2)
    worker = backend.worker_group
    result = backend.train(request)
    assert result.request == request and result.policy_after.weights_digest != checkpoint.weights_digest
    assert result.evidence_ref == 'fake-test-evidence'
    assert worker.load_calls == (0 if resident_state else 1)


def test_performance_overlay_uses_verl_dynamic_batch_with_measured_budget():
    yaml = pytest.importorskip('yaml')
    from pathlib import Path
    overlay = yaml.safe_load(Path('configs/verl/performance_trial.yaml').read_text())
    actor = overlay['actor_rollout_ref']['actor']
    assert actor['strategy'] == 'fsdp2' and actor['use_dynamic_bsz'] is True
    assert actor['ppo_max_token_len_per_gpu'] == '${oc.env:RIFT_PPO_MAX_TOKEN_LEN_PER_GPU}'
    assert overlay['actor_rollout_ref']['rollout']['n'] == 4

def test_mock_replay_scorer_worker_preserves_action_context(monkeypatch):
    """Fake worker output is not a real checkpoint likelihood."""
    from rift.verl.scorer import VerlPolicyScorer
    monkeypatch.setattr('rift.verl.scorer.require_verl_version', lambda: None)
    class Proto:
        @classmethod
        def from_dict(cls, tensors):
            instance=cls(); instance.data=dict(tensors); return instance
        def to_tensordict(self): return self.data
    modules={}
    for name in ['verl','verl.utils','verl.utils.tensordict_utils','verl.workers','verl.workers.utils','verl.workers.utils.padding']:
        modules[name]=types.ModuleType(name); monkeypatch.setitem(sys.modules,name,modules[name])
    modules['verl'].DataProto=Proto
    tu=modules['verl.utils.tensordict_utils']; tu.assign_non_tensor=lambda batch,**kw: batch.update(kw); tu.get=lambda batch,key: batch[key]
    modules['verl.utils'].tensordict_utils=tu
    padding=modules['verl.workers.utils.padding']; padding.left_right_2_no_padding=lambda batch: batch
    padding.no_padding_2_padding=lambda tensor,batch: tensor
    class Worker:
        def load_checkpoint(self,path): self.loaded=path
        def compute_log_prob(self,batch):
            assert batch['input_ids'].tolist()==[[1,2,3,4]]
            assert batch['response_mask'].tolist()==[[1,0,1]]
            return {'log_probs':torch.tensor([[-0.1,-0.2,-0.3]])}
    class Store:
        def verify(self,checkpoint): return True
        def resolve(self,checkpoint): return 'fake-checkpoint'
    cp=Checkpoint('service','v1','tok','weights'); worker=Worker()
    scores=VerlPolicyScorer(worker, Store(), pad_token_id=0).score(cp,[Tokens((1,),(2,3,4),(True,False,True))])
    assert scores[0] == pytest.approx([-0.1,-0.2,-0.3]) and worker.loaded=='fake-checkpoint'
