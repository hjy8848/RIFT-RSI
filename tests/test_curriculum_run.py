import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path[:0] = ['src', 'third_party/usami']

from rift.benchmark.families import FAMILIES, FAMILY_IDS
from rift.evolution.matched_control import UpdateBudget
from rift.runtime import curriculum_run as run
from rift.schemas import Checkpoint


class SpecTests(unittest.TestCase):
    def test_specs_are_validated(self):
        parsed = run.parse_task_specs([{'family': 'F1', 'instances': 2, 'seed': 5}])
        self.assertEqual(parsed[0]['knobs'], {})
        for bad in ([], [{'family': 'F1', 'instances': 0, 'seed': 1}],
                    [{'family': 'F1', 'instances': 1, 'seed': -1}],
                    [{'family': 'F1', 'instances': 1, 'seed': 1, 'extra': 1}]):
            with self.assertRaises(ValueError):
                run.parse_task_specs(bad)

    def test_inline_json_and_file(self):
        self.assertEqual(run.load_json_argument('[{"a": 1}]'), [{'a': 1}])
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'specs.json'
            path.write_text('{"a": 2}', encoding='utf-8')
            self.assertEqual(run.load_json_argument(str(path)), {'a': 2})

    def test_summary_keeps_unknown_apart(self):
        rows = [('F1:{}', 't1', 'success')] * 3 + [('F1:{}', 't1', 'failure'), ('F1:{}', 't2', 'unknown')]
        report = run.summarize_outcomes(rows)['F1:{}']
        self.assertEqual((report['success'], report['failure'], report['unknown']), (3, 1, 1))
        self.assertAlmostEqual(report['success_rate'], 0.75)
        self.assertLess(report['wilson95'][0], 0.75)
        self.assertGreater(report['wilson95'][1], 0.75)

    def test_summary_accepts_the_real_outcome_values(self):
        """The Agent Loop reports Outcome.value, which is upper case."""
        rows = [('k', 't', 'SUCCESS')] * 3 + [('k', 't', 'FAILURE'), ('k', 't', 'UNKNOWN')]
        report = run.summarize_outcomes(rows)['k']
        self.assertEqual((report['success'], report['failure'], report['unknown']), (3, 1, 1))

    def test_wilson_bounds(self):
        low, high = run.wilson_interval(0, 10)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)
        self.assertEqual(run.wilson_interval(0, 0), (0.0, 1.0))


class WorldTests(unittest.TestCase):
    def test_every_family_certifies_as_a_designer_arm(self):
        """Arms use n_tasks > 1; each family must certify that way before a run depends on it."""
        with tempfile.TemporaryDirectory() as root:
            kit = run.WorldKit(root)
            failures = {}
            for family in FAMILY_IDS:
                try:
                    instance = FAMILIES[family].instance({}, 3000 + FAMILY_IDS.index(family), n_tasks=4)
                    result = kit.host(instance).certify_training_plan(
                        instance.proposal(n_tasks=4), world_seed=instance.world_seed)
                    if result.build.status != 'SAT' or len(result.tasks) != 4:
                        failures[family] = result.build.status
                except Exception as error:   # report all families, not only the first
                    failures[family] = repr(error)[:120]
            self.assertEqual(failures, {})

    def test_registry_survives_restart(self):
        with tempfile.TemporaryDirectory() as root:
            kit = run.WorldKit(root)
            refs = kit.certify_specs([{'family': 'F1', 'knobs': {}, 'instances': 2, 'seed': 10}])
            kit.save()
            again = run.WorldKit(root)
            self.assertEqual(set(refs['F1:{}']), set(again.task_registry.task_refs))


class EvalRequestTests(unittest.TestCase):
    def test_groups_are_complete_and_ids_unique(self):
        checkpoint = Checkpoint('service', '0', 'tok', 'w0')
        requests, owners = run.build_eval_requests({'F1:{}': ['a' * 64, 'b' * 64]}, checkpoint,
            groups=2, group_size=4, seed=7, label='x0')
        self.assertEqual(len(requests), 16)
        self.assertEqual(len({row['rift_request_id'] for row in requests}), 16)
        self.assertEqual(len({row['session_id'] for row in requests}), 16)
        by_group = {}
        for row in requests:
            by_group.setdefault(row['uid'], []).append(row)
        self.assertEqual({len(rows) for rows in by_group.values()}, {4})
        for rows in by_group.values():
            self.assertEqual(len({row['rift_task_ref'] for row in rows}), 1)
            self.assertEqual(len({row['rift_sampling_seed'] for row in rows}), 4)
            self.assertEqual(len({row['rift_customer_seed'] for row in rows}), 1)
        self.assertEqual(set(owners), {row['rift_request_id'] for row in requests})

    def test_labels_separate_runs(self):
        checkpoint = Checkpoint('service', '0', 'tok', 'w0')
        first, _ = run.build_eval_requests({'k': ['a' * 64]}, checkpoint, groups=1, group_size=4, seed=7, label='x0')
        second, _ = run.build_eval_requests({'k': ['a' * 64]}, checkpoint, groups=1, group_size=4, seed=7, label='x1')
        self.assertFalse({r['rift_request_id'] for r in first} & {r['rift_request_id'] for r in second})


class PolicyTests(unittest.TestCase):
    def test_every_named_policy_builds(self):
        from rift.evolution.curriculum import Selection
        arms = (Selection('F1', {}, 4), Selection('F4', {'age_days': 20}, 4))
        for name in run.POLICIES:
            policy = run.build_policy(name, arms, seed=1, window=5, fixed_family='F1')
            self.assertEqual(policy.propose(0, ()).n_tasks, 4)
        with self.assertRaises(ValueError):
            run.build_policy('fixed', arms, seed=1, window=5)
        with self.assertRaises(ValueError):
            run.build_policy('nope', arms, seed=1, window=5)


class FakeStore:
    def verify(self, checkpoint):
        return True


class RecoveryTests(unittest.TestCase):
    def test_no_receipt_means_train_and_mismatch_is_refused(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as root:
            ledger = SimpleNamespace(root=Path(root), verify_update=lambda update: False)
            context = SimpleNamespace(ledger=ledger)
            recover = run.build_recover_update(context)
            self.assertIsNone(recover(0, object()))
            after = Checkpoint('service', '1', 'tok', 'w1')
            (Path(root) / (run.run_id_for(0) + '.json')).write_text(
                json.dumps({'after': {'policy': 'service', 'revision': '1', 'tokenizer': 'tok',
                                      'weights_digest': 'w1'}}), encoding='utf-8')
            request = SimpleNamespace(policy_before=Checkpoint('service', '0', 'tok', 'w0'))
            with self.assertRaises(ValueError):
                recover(0, request)
            ledger.verify_update = lambda update: True
            update = recover(0, request)
            self.assertEqual(update.policy_after, after)
            self.assertEqual(update.run_id, run.run_id_for(0))


class X1ConfigTests(unittest.TestCase):
    def test_shipped_screening_specs_are_valid_and_unique(self):
        specs = run.parse_task_specs(run.load_json_argument('configs/experiments/x1_screen.json'))
        self.assertEqual(len({run.spec_key(spec) for spec in specs}), len(specs))
        seeds = [(spec['seed'], spec['seed'] + spec['instances']) for spec in specs]
        for index, (low, high) in enumerate(seeds):
            for other_low, other_high in seeds[index + 1:]:
                self.assertTrue(high <= other_low or other_high <= low, 'World seed ranges overlap')
        from usami_v2.runtime.runner import EpisodeBudget
        EpisodeBudget.model_validate(run.load_json_argument('configs/experiments/x1_budget.json'))

    def test_shipped_smoke_arms_and_bench_fit_a_small_round(self):
        from rift.evolution.curriculum import Selection
        from rift.training.package import TrainingPackageConfig
        config = TrainingPackageConfig(tasks_per_round=8, sampling_seed=2000, group_size=4)
        arms = run.load_json_argument('configs/experiments/smoke_arms.json')
        bench = run.parse_task_specs(run.load_json_argument('configs/experiments/smoke_bench.json'))
        selections = [Selection(row['family_id'], row['knobs'], config.designer_task_count) for row in arms]
        self.assertEqual(len({item.arm_key for item in selections}), len(selections))
        pool_size = sum(spec['instances'] for spec in bench)
        self.assertGreaterEqual(pool_size, config.benchmark_task_count)
        self.assertEqual(config.designer_task_count * config.group_size % 8, 0)   # whole mini-batches

    def test_eval_run_offers_its_initial_checkpoint(self):
        from types import SimpleNamespace
        from usami_v2.runtime.runner import EpisodeBudget
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as root:
            (Path(source) / 'checkpoints').mkdir()
            (Path(source) / 'checkpoints' / 'initial.json').write_text('{}', encoding='utf-8')
            expected = Checkpoint('service', '0', 'tok', 'w0')
            (Path(source) / 'initial-checkpoint.json').write_text(json.dumps(
                {'policy': 'service', 'revision': '0', 'tokenizer': 'tok', 'weights_digest': 'w0'}),
                encoding='utf-8')
            context = run.ExperimentContext(root=root, service_model='m', verl_config_dir='.',
                episode_budget=EpisodeBudget(), update_budget=UpdateBudget(8, 1, 1e-6))
            adopted = []
            context.adopt_manifest = adopted.append
            context.checkpoint_store = SimpleNamespace(verify=lambda checkpoint: True)
            self.assertEqual(context.checkpoint_from_run(source), expected)
            self.assertEqual(adopted, [Path(source) / 'checkpoints' / 'initial.json'])


class EvaluateRehearsalTests(unittest.TestCase):
    """Run evaluate() end to end with a fake runtime, as a rehearsal for the NPU host."""
    def _source_run(self, root):
        """A finished eval run directory holding a real (tiny) checkpoint and manifest."""
        from rift.verl.checkpoints import FSDPCheckpointStore
        directory = Path(root) / 'checkpoints' / 'service-initial'
        directory.mkdir(parents=True)
        for kind in ('model', 'optim', 'extra_state'):
            for rank in range(8):
                (directory / f'{kind}_world_size_8_rank_{rank}.pt').write_bytes(b'w' * 64 + bytes([rank]))
        store = FSDPCheckpointStore({}, world_size=8)
        checkpoint = store.publish('service', '0', 'tok', directory)
        store.save_manifest(Path(root) / 'checkpoints' / 'initial.json')
        from dataclasses import asdict
        (Path(root) / 'initial-checkpoint.json').write_text(json.dumps(asdict(checkpoint)), encoding='utf-8')
        return checkpoint

    def test_every_request_is_sent_once_and_outcomes_are_counted(self):
        from usami_v2.runtime.runner import EpisodeBudget
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as root:
            checkpoint = self._source_run(source)
            calls = []

            class FakeRuntime:
                def __init__(self, loaded):
                    self.checkpoint, self.closed = loaded, False
                    self.service_batch = self

                def generate(self, requests):
                    calls.append(len(requests))
                    return tuple({'rift_request_id': row['rift_request_id'],
                                  'rift_outcome': 'SUCCESS' if index % 4 else 'FAILURE',
                                  'rift_replay_verified': True}
                                 for index, row in enumerate(requests))

                def close(self):
                    self.closed = True

            context = run.ExperimentContext(root=root, service_model='m', verl_config_dir='.',
                episode_budget=EpisodeBudget(), update_budget=UpdateBudget(8, 1, 1e-6),
                runtime_factory=FakeRuntime)
            result = run.evaluate(context,
                specs=[{'family': 'F1', 'knobs': {}, 'instances': 3, 'seed': 77},
                       {'family': 'F4', 'knobs': {'age_days': 20}, 'instances': 3, 'seed': 88}],
                groups=2, seed=5, label='rehearsal', from_run=source)
            self.assertEqual(calls, [48])            # one call: 2 specs x 3 tasks x 2 groups x 4
            self.assertEqual(result['checkpoint']['weights_digest'], checkpoint.weights_digest)
            for key in ('F1:{}', 'F4:{"age_days":20}'):
                row = result['families'][key]
                self.assertEqual((row['tasks'], row['success'] + row['failure'], row['unknown']), (3, 24, 0))
            self.assertIsNone(context.runtime)       # the runtime was released

    def test_runtime_is_released_when_generation_fails(self):
        from usami_v2.runtime.runner import EpisodeBudget
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as root:
            self._source_run(source)
            runtimes = []

            class Broken:
                def __init__(self, loaded):
                    self.checkpoint, self.closed = loaded, False
                    self.service_batch = self
                    runtimes.append(self)

                def generate(self, requests):
                    raise RuntimeError('vLLM went away')

                def close(self):
                    self.closed = True

            context = run.ExperimentContext(root=root, service_model='m', verl_config_dir='.',
                episode_budget=EpisodeBudget(), update_budget=UpdateBudget(8, 1, 1e-6),
                runtime_factory=Broken)
            with self.assertRaises(RuntimeError):
                run.evaluate(context, specs=[{'family': 'F1', 'knobs': {}, 'instances': 1, 'seed': 9}],
                             groups=1, seed=5, label='x', from_run=source)
            self.assertTrue(runtimes and all(item.closed for item in runtimes))


class RuntimeLifetimeTests(unittest.TestCase):
    """The live runtime is reused across rounds and rebuilt whenever it cannot be trusted."""
    def _context(self, root, keep=True):
        from usami_v2.runtime.runner import EpisodeBudget
        self.opened = []
        parent = self

        class Fake:
            def __init__(self, loaded):
                self.checkpoint, self._loaded_checkpoint, self.closed = loaded, loaded, False
                parent.opened.append(self)

            def close(self):
                self.closed = True

        context = run.ExperimentContext(root=root, service_model='m', verl_config_dir='.',
            episode_budget=EpisodeBudget(), update_budget=UpdateBudget(8, 1, 1e-6),
            runtime_factory=Fake, keep_runtime=keep)
        context._worker_manifest = lambda checkpoint: self.published.append(checkpoint)
        self.published = []
        return context

    def test_one_runtime_serves_every_round(self):
        first, second = Checkpoint('service', '0', 't', 'w0'), Checkpoint('service', '1', 't', 'w1')
        with tempfile.TemporaryDirectory() as root:
            context = self._context(root)
            runtime = context.ensure_runtime(first)
            self.assertIs(context.ensure_runtime(first), runtime)
            runtime._loaded_checkpoint = second          # what an update leaves behind
            self.assertIs(context.ensure_runtime(second), runtime)
            self.assertEqual(len(self.opened), 1)
            self.assertEqual(self.published, [first, second])   # workers were told each round

    def test_a_runtime_holding_another_checkpoint_is_replaced(self):
        first, other = Checkpoint('service', '0', 't', 'w0'), Checkpoint('service', '5', 't', 'w5')
        with tempfile.TemporaryDirectory() as root:
            context = self._context(root)
            old = context.ensure_runtime(first)
            fresh = context.ensure_runtime(other)
            self.assertIsNot(fresh, old)
            self.assertTrue(old.closed)

    def test_restart_each_round_never_reuses(self):
        first = Checkpoint('service', '0', 't', 'w0')
        with tempfile.TemporaryDirectory() as root:
            context = self._context(root, keep=False)
            context.ensure_runtime(first)
            context.ensure_runtime(first)
            self.assertEqual(len(self.opened), 2)

    def test_a_failed_update_releases_the_runtime(self):
        from types import SimpleNamespace
        first = Checkpoint('service', '0', 't', 'w0')
        with tempfile.TemporaryDirectory() as root:
            context = self._context(root)
            runtime = context.ensure_runtime(first)

            def explode(*args, **kwargs):
                raise RuntimeError('NaN loss')
            runtime.create_update_backend = explode
            request = SimpleNamespace(policy_before=first)
            with self.assertRaises(RuntimeError):
                run.RoundBackend(context, 0, first).train(request)
            self.assertTrue(runtime.closed and context.runtime is None)


class HostProviderReuseTests(unittest.TestCase):
    def _worker(self, root):
        """A worker manifest on disk plus the stand-in registries it points at."""
        import os
        root = Path(root)
        (root / 'splits.json').write_text('{}', encoding='utf-8')
        (root / 'tasks.json').write_text('{}', encoding='utf-8')

        def write(checkpoint):
            (root / 'manifest.json').write_text(json.dumps({
                'split_registry': str(root / 'splits.json'), 'task_registry': str(root / 'tasks.json'),
                'active_checkpoint': checkpoint}), encoding='utf-8')
        write({'revision': '0'})
        os.environ['RIFT_ASCEND910B3_RUN_MANIFEST'] = str(root / 'manifest.json')
        return write

    def test_the_worker_rebuilds_its_provider_only_when_the_round_changes(self):
        import os
        from rift.runtime import ascend_910b3 as ascend
        built = []
        original = ascend._build_service_host_provider
        ascend._build_service_host_provider = lambda manifest, checkpoint: built.append(
            checkpoint) or object()
        ascend._HOST_PROVIDERS.clear()
        saved = os.environ.get('RIFT_ASCEND910B3_RUN_MANIFEST')
        try:
            with tempfile.TemporaryDirectory() as root:
                write = self._worker(root)
                first = [ascend.service_host_provider_from_config(config=None, tokenizer=None)
                         for _ in range(44)]
                self.assertEqual(len(built), 1)                    # 44 samples, one build
                self.assertEqual(len({id(item) for item in first}), 1)
                (Path(root) / 'tasks.json').write_text('{"grown": true}', encoding='utf-8')
                ascend.service_host_provider_from_config(config=None, tokenizer=None)
                self.assertEqual(len(built), 2)                    # new tasks were certified
                write({'revision': '1'})
                ascend.service_host_provider_from_config(config=None, tokenizer=None)
                self.assertEqual(len(built), 3)                    # the policy was updated
                ascend.service_host_provider_from_config(config=None, tokenizer=None)
                self.assertEqual(len(built), 3)
                self.assertEqual(len(ascend._HOST_PROVIDERS), 1)   # old rounds are dropped
        finally:
            ascend._build_service_host_provider = original
            ascend._HOST_PROVIDERS.clear()
            if saved is None:
                os.environ.pop('RIFT_ASCEND910B3_RUN_MANIFEST', None)
            else:
                os.environ['RIFT_ASCEND910B3_RUN_MANIFEST'] = saved


class ChatTemplateTests(unittest.TestCase):
    def test_runs_use_the_non_thinking_template(self):
        self.assertEqual(run.CHAT_TEMPLATE_KWARGS, {'enable_thinking': False})


class BackendBudgetTests(unittest.TestCase):
    def test_backend_reports_the_shared_budget(self):
        from types import SimpleNamespace
        budget = UpdateBudget(8, 1, 1e-6)
        backend = run.RoundBackend(SimpleNamespace(update_budget=budget), 0, None)
        self.assertEqual(backend.budget_for(None), budget)


if __name__ == '__main__':
    unittest.main()

