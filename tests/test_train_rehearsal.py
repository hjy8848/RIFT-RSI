"""Rehearse train() for two rounds on CPU: real worlds, episodes, packages, stores and ledger.

Only the NPU is faked: the runtime (rollouts and the optimizer step). It checks the wiring
that the hardware run depends on: one runtime for every round, the next round starting from
the checkpoint the update produced, and a rerun recovering instead of training again.
"""
import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path[:0] = ['src', 'third_party/usami', 'tests']

from rift.evolution.matched_control import UpdateBudget
from rift.runtime import curriculum_run as run
from rift.schemas import Checkpoint, Tokens
from rift.training.contracts import CompletedUpdate, UpdateRequest
from rift.training.package import TrainingPackageConfig
from rift.training.records import service_rollout
from rift.benchmark.private_evaluation import public_business_outcome
from usami_v2.runtime.runner import EpisodeBudget, replay_episode, run_episode
from test_usami_phase3 import ScriptedService, YesCustomer, gather


def _script(skilled, member):
    def script(ctx):
        if not skilled:
            yield ctx.say('Sorry, I am unable to help with that. (attempt %d)' % member)
            return
        yield from gather(ctx)
        request = ctx.prompt.customer_request.lower()
        matches = [row for row in ctx.lines if all(
            re.search(r'\b%s\b' % re.escape(str(v).replace('_', ' ').lower()), request)
            for v in row['sku']['attributes'].values())]
        assert len(matches) == 1, 'request does not identify exactly one line: %d' % len(matches)
        yield ctx.say('I can refund that item to your original card. Shall I proceed? (attempt %d)' % member)
        yield ctx.op('return_delivered_order_items', matches[0]['id'], quantity=1, payment_method='original')
        yield ctx.say('Your refund has been issued to the original card.')
    return script


class FakeBackend:
    def __init__(self, context, runtime, request, kwargs):
        self.context, self.runtime, self.request, self.kwargs = context, runtime, request, kwargs

    def train(self, request):
        context, kwargs = self.context, self.kwargs
        directory = Path(kwargs['output_dir'])
        if kwargs.get('persist', True):
            directory.mkdir(parents=True)
            for kind in ('model', 'optim', 'extra_state'):
                for rank in range(8):
                    (directory / f'{kind}_world_size_8_rank_{rank}.pt').write_bytes(
                        kwargs['after_revision'].encode() + bytes([rank]) * 64)
            after = context.checkpoint_store.publish('service', kwargs['after_revision'],
                                                     request.policy_before.tokenizer, directory)
        else:
            from dataclasses import asdict
            from rift.training.trace import digest
            after = context.checkpoint_store.publish_virtual(request.policy_before,
                kwargs['after_revision'], digest(asdict(request)))
        context.checkpoint_store.save_manifest(kwargs['checkpoint_manifest_path'])
        self.runtime._loaded_checkpoint = after
        self.runtime.trained.append(request.policy_before)
        bindings = self.runtime.backend_bindings
        evidence = bindings['run_recorder'].record_update_with_budget(
            kwargs['run_id'], request, after, {'ok': 1}, context.update_budget)
        return CompletedUpdate(request, after, kwargs['run_id'], evidence)


class FakeRuntime:
    opened = []

    def __init__(self, context, loaded):
        self.context, self.checkpoint, self._loaded_checkpoint = context, loaded, loaded
        self.backend_bindings, self.trained, self.closed = {}, [], False
        self.service_batch, self.evaluated = self, []
        FakeRuntime.opened.append(self)

    def generate(self, requests):
        """Stands in for the Agent Loop on the in-place held-out evaluation."""
        self.evaluated.append((self._loaded_checkpoint, len(requests)))
        return tuple({'rift_request_id': row['rift_request_id'],
                      'rift_outcome': 'SUCCESS' if index % 3 else 'FAILURE',
                      'rift_replay_verified': True} for index, row in enumerate(requests))

    def create_update_backend(self, request, **kwargs):
        return FakeBackend(self.context, self, request, kwargs)

    def close(self):
        self.closed = True


class TrainRehearsal(unittest.TestCase):
    def _initial(self, root):
        """Give the run directory the starting checkpoint the NPU would have written."""
        from rift.verl.checkpoints import FSDPCheckpointStore
        directory = Path(root) / 'checkpoints' / 'service-initial'
        directory.mkdir(parents=True)
        for kind in ('model', 'optim', 'extra_state'):
            for rank in range(8):
                (directory / f'{kind}_world_size_8_rank_{rank}.pt').write_bytes(b'base' + bytes([rank]) * 64)
        store = FSDPCheckpointStore({}, world_size=8)
        checkpoint = store.publish('service', '0', 'tok', directory)
        store.save_manifest(Path(root) / 'checkpoints' / 'initial.json')
        (Path(root) / 'initial-checkpoint.json').write_text(json.dumps(
            {'policy': 'service', 'revision': '0', 'tokenizer': 'tok',
             'weights_digest': checkpoint.weights_digest}), encoding='utf-8')

    def _collector(self, context, skilled=True):
        class Collector:
            def __init__(self, checkpoint, index):
                self.checkpoint, self.index = checkpoint, index

            def collect(self, package, checkpoint):
                context.ensure_runtime(checkpoint)       # what LazyBatch does before generating
                registry = context.world.task_registry
                rows = []
                for group in package.task_groups:
                    record = registry.resolve_record(group.task_ref)
                    skill = int(hashlib.sha256(group.task_ref.encode()).hexdigest(), 16) % 5 if skilled else 0
                    for member in range(group.group_size):
                        episode = run_episode(record.world, record.task, record.profile, YesCustomer(1),
                            ScriptedService(_script(member < skill, member)),
                            budget=EpisodeBudget(max_customer_turns=6, max_service_decisions=20,
                                                 max_service_tool_calls=12, max_model_calls=0),
                            external_events=record.external_events)
                        outcome = public_business_outcome(episode, replay_episode(episode).verified)
                        digits = hashlib.sha256(episode.record_hash.encode()).digest()
                        prompt = hashlib.sha256(group.task_ref.encode()).digest()[0]
                        tokens = Tokens((1 + prompt,), (1 + digits[1], 1 + digits[2]), (True, True))
                        # Scripted failures can record identically; GRPO needs distinct members.
                        member_id = hashlib.sha256(('%s:%d' % (episode.record_hash, member)).encode()).hexdigest()
                        rows.append(service_rollout(tokens, checkpoint, group.task_ref, member_id,
                                                    outcome, group.group_id, response_id=member_id))
                request = UpdateRequest(checkpoint, package.package_id, tuple(rows), 'grpo')
                record = getattr(package, 'record_service_request', None)
                return request, (record(request) if record is not None else None)
        return Collector

    def test_a_run_can_start_from_another_runs_base_checkpoint(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as root:
            self._initial(source)
            _, result = self._train(root, init_from_run=source)
            self.assertEqual(result['rounds'], 2)
            self.assertTrue((Path(root) / 'initial-checkpoint.json').exists())
            self.assertTrue((Path(root) / 'checkpoints' / 'initial.json').exists())
            _, again = self._train(root, init_from_run=source)       # resumes from its own manifests
            self.assertEqual(again['final_checkpoint'], result['final_checkpoint'])

    def test_rounds_between_persist_points_write_no_checkpoint_files(self):
        with tempfile.TemporaryDirectory() as root:
            self._initial(root)
            context, result = self._train(root, rounds=3, persist_every=2)
            steps = Path(root) / 'checkpoints' / 'service'
            self.assertFalse((steps / 'global-step-00000001').exists())   # in memory only
            self.assertTrue((steps / 'global-step-00000002').exists())
            self.assertTrue((steps / 'global-step-00000003').exists())    # the last round persists
            self.assertEqual({cp.revision for cp in context.checkpoint_store.virtual}, {'1'})
            self.assertEqual(result['final_checkpoint']['revision'], '3')
            self.assertEqual(len(FakeRuntime.opened), 1)
            final = json.loads((Path(root) / 'final.json').read_text(encoding='utf-8'))
            self.assertEqual(final['checkpoint']['revision'], '3')

    def test_zero_means_only_the_last_round_is_written(self):
        with tempfile.TemporaryDirectory() as root:
            self._initial(root)
            context, result = self._train(root, rounds=3, persist_every=0)
            steps = Path(root) / 'checkpoints' / 'service'
            self.assertEqual(sorted(path.name for path in steps.iterdir()), ['global-step-00000003'])
            self.assertEqual({cp.revision for cp in context.checkpoint_store.virtual}, {'1', '2'})
            self.assertEqual(result['final_checkpoint']['revision'], '3')

    def test_pool_training_runs_rounds_from_single_task_worlds_without_a_package(self):
        from rift.runtime.ascend_910b3 import CHAT_TEMPLATE_KWARGS   # noqa: F401  (import path check)
        with tempfile.TemporaryDirectory() as root:
            self._initial(root)
            FakeRuntime.opened = []
            context = run.ExperimentContext(root=root, service_model='m', verl_config_dir='.',
                episode_budget=EpisodeBudget(), update_budget=UpdateBudget(8, 1, 1e-6),
                runtime_factory=lambda checkpoint: FakeRuntime(context, checkpoint), persist_every=0)
            context._worker_manifest = lambda checkpoint: None
            collector = self._collector(context)
            result = run.train_pool(context,
                train_specs=[{'family': 'F1', 'knobs': {'distractor_orders': 3}, 'instances': 6, 'seed': 7100},
                             {'family': 'F2', 'knobs': {'purchased_quantity': 5, 'refund_quantity': 2},
                              'instances': 6, 'seed': 7200}],
                rounds=3, tasks_per_round=4, seed=11,
                collector_factory=lambda loaded, index: collector(loaded, index))
            self.assertEqual(result['rounds'], 3)
            self.assertEqual(len(FakeRuntime.opened), 1)
            self.assertEqual(result['final_checkpoint']['revision'], '3')
            steps = Path(root) / 'checkpoints' / 'service'
            self.assertEqual(sorted(path.name for path in steps.iterdir()), ['global-step-00000003'])
            rows = [json.loads(line) for line in (Path(root) / 'rounds.jsonl').read_text(
                encoding='utf-8').splitlines()]
            self.assertEqual([row['round'] for row in rows], [0, 1, 2])
            self.assertEqual(sorted(rows[0]['train_success']), ['F1:{"distractor_orders":3}',
                'F2:{"purchased_quantity":5,"refund_quantity":2}'])
            self.assertEqual(rows[0]['rollouts'], 16)               # 4 tasks x 4 samples
            with self.assertRaises(FileExistsError):                # a finished run is not overwritten
                run.train_pool(context, train_specs=[], rounds=1, tasks_per_round=2, seed=1)

    def test_collect_failures_reuses_a_base_checkpoint_and_refuses_a_trained_run(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as root:
            self._initial(source)
            context = run.ExperimentContext(root=root, service_model='m', verl_config_dir='.',
                episode_budget=EpisodeBudget(), update_budget=UpdateBudget(8, 1, 1e-6),
                runtime_factory=lambda checkpoint: FakeRuntime(context, checkpoint), persist_every=0)
            specs = [{'generator': 'F1', 'instances': 1, 'seed': 0}]
            with mock.patch.object(run, 'sample_rows', return_value=[]), \
                    mock.patch('rift.designer.failures.failure_cases', return_value=[]):
                result = run.collect_failures(context, specs=specs, groups=1, seed=1, max_cases=2,
                                              init_from_run=source)
            self.assertEqual(result['status'], 'failures_collected')
            self.assertTrue((Path(root) / 'initial-checkpoint.json').exists())
            self.assertTrue((Path(root) / 'checkpoints' / 'initial.json').exists())
            self.assertFalse((Path(root) / 'checkpoints' / 'service-initial').exists())   # nothing rewritten
            self.assertEqual(json.loads((Path(root) / 'initial-checkpoint.json').read_text()),
                             json.loads((Path(source) / 'initial-checkpoint.json').read_text()))
            (Path(source) / 'final.json').write_text('{}')                          # a run that trained
            with self.assertRaisesRegex(ValueError, 'trained a policy'):
                run.collect_failures(context, specs=specs, groups=1, seed=1, max_cases=2, init_from_run=source)
            with self.assertRaisesRegex(ValueError, 'alternatives'):
                run.collect_failures(context, specs=specs, groups=1, seed=1, max_cases=2,
                                     init_from_run=source, from_run=source)

    def test_eval_reports_designer_tasks_as_their_own_class(self):
        from rift.designer.stage0 import dry_run_generate, run as stage0
        with tempfile.TemporaryDirectory() as plans_root, tempfile.TemporaryDirectory() as root:
            stage0(run.WorldKit(plans_root), lambda prompts: dry_run_generate(prompts, seed=31), [],
                   samples=3, seed=31, out_dir=plans_root, signatures=False)
            self._initial(root)
            FakeRuntime.opened = []
            context = run.ExperimentContext(root=root, service_model='m', verl_config_dir='.',
                episode_budget=EpisodeBudget(), update_budget=UpdateBudget(8, 1, 1e-6),
                runtime_factory=lambda checkpoint: FakeRuntime(context, checkpoint), persist_every=0)
            context._worker_manifest = lambda checkpoint: None
            plans = Path(plans_root) / 'designer-plans.jsonl'
            result = run.evaluate(context, specs=[{'generator': 'F4', 'instances': 2, 'seed': 70, 'balanced': True}],
                                  groups=1, seed=5, label='t', designer_plans=plans)
            self.assertEqual(sorted(result['families']), ['designer', 'gen:F4:{"balanced":true}'])
            self.assertEqual(result['families']['designer']['tasks'], 3)
            self.assertEqual(result['families']['gen:F4:{"balanced":true}']['tasks'], 2)
            with self.assertRaisesRegex(ValueError, 'nothing to evaluate'):
                run.evaluate(context, specs=[], groups=1, seed=5, label='t')

    def test_stage1_trains_on_designer_plans_with_a_balanced_mix(self):
        from rift.designer.stage0 import dry_run_generate, run as stage0
        with tempfile.TemporaryDirectory() as plans_root, tempfile.TemporaryDirectory() as root:
            stage0(run.WorldKit(plans_root), lambda prompts: dry_run_generate(prompts, seed=21), [],
                   samples=6, seed=21, out_dir=plans_root, signatures=False)
            self._initial(root)
            FakeRuntime.opened = []
            context = run.ExperimentContext(root=root, service_model='m', verl_config_dir='.',
                episode_budget=EpisodeBudget(), update_budget=UpdateBudget(8, 1, 1e-6),
                runtime_factory=lambda checkpoint: FakeRuntime(context, checkpoint), persist_every=0,
                patience_slack=2)
            context._worker_manifest = lambda checkpoint: None
            collector = self._collector(context, skilled=False)
            result = run.train_pool(context,
                train_specs=[{'generator': 'F5', 'instances': 4, 'seed': 0, 'balanced': True}],
                rounds=2, tasks_per_round=4, seed=3, designer_plans=Path(plans_root) / 'designer-plans.jsonl',
                mix_fraction=0.25, heldout_specs=[{'generator': 'F4', 'instances': 2, 'seed': 50, 'balanced': True}],
                collector_factory=lambda loaded, index: collector(loaded, index))
            rows = [json.loads(line) for line in (Path(root) / 'rounds.jsonl').read_text(
                encoding='utf-8').splitlines()]
            self.assertEqual(result['rounds'], 2)
            self.assertEqual(sorted(rows[0]['train_success']), ['designer', 'gen:F5:{"balanced":true}'])
            self.assertEqual(rows[0]['rollouts'], 16)
            self.assertIn('gen:F4:{"balanced":true}', result['heldout']['families'])

    def _policy_run(self, root, policy_name, seed):
        from rift.evolution.curriculum import Selection
        FakeRuntime.opened = []
        context = run.ExperimentContext(root=root, service_model='m', verl_config_dir='.',
            episode_budget=EpisodeBudget(), update_budget=UpdateBudget(8, 1, 1e-6),
            runtime_factory=lambda checkpoint: FakeRuntime(context, checkpoint),
            persist_every=0, persist_final=False)
        context._worker_manifest = lambda checkpoint: None
        collector = self._collector(context)
        arms = [{'family': 'F1', 'knobs': {'distractor_orders': 3}, 'instances': 6, 'seed': 8100},
                {'family': 'F2', 'knobs': {'purchased_quantity': 5, 'refund_quantity': 2},
                 'instances': 6, 'seed': 8200},
                {'family': 'F4', 'knobs': {'age_days': 20}, 'instances': 6, 'seed': 8300}]
        selections = tuple(Selection(row['family'], row['knobs'], 4) for row in arms)
        policy = run.build_policy(policy_name, selections, seed=seed, window=5)
        result = run.train_pool(context, train_specs=arms, rounds=6, tasks_per_round=4, seed=seed,
            collector_factory=lambda loaded, index: collector(loaded, index), policy=policy,
            heldout_specs=[{'family': 'F1', 'knobs': {}, 'instances': 3, 'seed': 8400},
                           {'family': 'F7', 'knobs': {}, 'instances': 3, 'seed': 8500}],
            heldout_groups=2)
        return context, result

    def test_a_curriculum_policy_picks_one_class_per_round_and_evaluates_in_place(self):
        for policy_name in ('random', 'bandit'):
            with tempfile.TemporaryDirectory() as root:
                self._initial(root)
                context, result = self._policy_run(root, policy_name, seed=5)
                rows = [json.loads(line) for line in (Path(root) / 'rounds.jsonl').read_text(
                    encoding='utf-8').splitlines()]
                self.assertEqual(len(rows), 6)
                for row in rows:
                    self.assertEqual(list(row['train_success']), [row['arm']])     # one class per round
                    self.assertTrue(0 <= row['learnability'] <= 1)
                self.assertEqual(len(FakeRuntime.opened), 1)
                steps = Path(root) / 'checkpoints' / 'service'
                self.assertFalse(steps.exists() and any(steps.iterdir()))          # no checkpoint files
                final = json.loads((Path(root) / 'final.json').read_text(encoding='utf-8'))
                self.assertTrue(final['in_memory_only'])
                runtime = FakeRuntime.opened[0]
                self.assertEqual(runtime.evaluated, [(runtime._loaded_checkpoint, 2 * 4 * 6)])
                report = result['heldout']['families']
                self.assertEqual(sorted(report), ['F1:{}', 'F7:{}'])
                self.assertEqual(sum(v['success'] + v['failure'] for v in report.values()), 48)
                self.assertEqual(result['heldout']['checkpoint']['revision'], '6')

    def test_the_bandit_stops_choosing_a_class_that_gives_no_signal(self):
        """With one class that is always solved, learnability is 0 there and the bandit moves on."""
        from rift.evolution.curriculum import BanditPolicy, Observation, Selection
        arms = tuple(Selection(f, {}, 4) for f in ('F1', 'F4', 'F7'))
        policy = BanditPolicy(arms, seed=3, window=10)
        history = []
        picks = []
        for index in range(60):
            choice = policy.propose(index, tuple(history))
            picks.append(choice.family_id)
            learn = 0.0 if choice.family_id == 'F4' else 0.75        # F4 is saturated: no signal
            history.append(Observation(choice, 0.9 if learn == 0 else 0.5, learn, 4))
        self.assertLess(picks[30:].count('F4'), picks[30:].count('F1') + picks[30:].count('F7'))
        self.assertLess(picks[30:].count('F4'), 8)

    def test_a_new_process_cannot_resume_from_an_in_memory_version(self):
        from rift.schemas import Checkpoint
        with tempfile.TemporaryDirectory() as root:
            self._initial(root)
            context, _ = self._train(root, rounds=3, persist_every=2)
            virtual = next(iter(context.checkpoint_store.virtual))
            context.runtime = None
            with self.assertRaises(ValueError):
                context.ensure_runtime(virtual)

    def _train(self, root, init_from_run=None, rounds=2, persist_every=1):
        from rift.runtime import evaluator
        FakeRuntime.opened = []
        context = run.ExperimentContext(root=root, service_model='m', verl_config_dir='.',
            episode_budget=EpisodeBudget(), update_budget=UpdateBudget(8, 1, 1e-6),
            runtime_factory=lambda checkpoint: FakeRuntime(context, checkpoint),
            persist_every=persist_every)
        context._worker_manifest = lambda checkpoint: None
        original = evaluator.ProductionServiceRolloutCollector
        collector = self._collector(context)
        evaluator.ProductionServiceRolloutCollector = lambda batch, **kwargs: collector(
            batch.checkpoint, kwargs['round_index'])
        try:
            return context, run.train(context, policy_name='bandit',
                arms=[{'family_id': 'F1', 'knobs': {}}, {'family_id': 'F4', 'knobs': {'age_days': 20}}],
                bench_specs=[{'family': 'F1', 'knobs': {}, 'instances': 4, 'seed': 900000},
                             {'family': 'F4', 'knobs': {'age_days': 20}, 'instances': 4, 'seed': 901000}],
                rounds=rounds, seed=2000, window=5, init_from_run=init_from_run,
                package_config=TrainingPackageConfig(tasks_per_round=8, designer_fraction=.5,
                                                     benchmark_fraction=.5, sampling_seed=7, group_size=4))
        finally:
            evaluator.ProductionServiceRolloutCollector = original

    def test_two_rounds_share_one_runtime_and_chain_checkpoints(self):
        with tempfile.TemporaryDirectory() as root:
            self._initial(root)
            context, result = self._train(root)
            self.assertEqual(result['status'], 'curriculum_completed')
            self.assertEqual(result['rounds'], 2)
            self.assertEqual(len(FakeRuntime.opened), 1)              # one start for both rounds
            runtime = FakeRuntime.opened[0]
            self.assertEqual([cp.revision for cp in runtime.trained], ['0', '1'])
            self.assertEqual(result['final_checkpoint']['revision'], '2')
            self.assertTrue(runtime.closed)                          # released when the run ended
            final = json.loads((Path(root) / 'final.json').read_text(encoding='utf-8'))
            self.assertEqual(final['checkpoint']['revision'], '2')
            self.assertTrue(Path(final['manifest']).name.startswith('service-step-2'))
            rows = [json.loads(line) for line in
                    (Path(root) / 'curriculum' / 'rounds.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual([row['round_index'] for row in rows], [0, 1])

    def test_a_rerun_recovers_the_finished_rounds_without_training(self):
        with tempfile.TemporaryDirectory() as root:
            self._initial(root)
            _, first = self._train(root)
            _, second = self._train(root)
            self.assertEqual(second['final_checkpoint'], first['final_checkpoint'])
            self.assertEqual([item for runtime in FakeRuntime.opened for item in runtime.trained], [])


if __name__ == '__main__':
    unittest.main()
