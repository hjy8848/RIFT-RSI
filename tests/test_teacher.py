import asyncio
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

sys.path[:0] = ['src', 'third_party/usami']

from rift.designer.stage0 import dry_run_generate
from rift.runtime import teacher as T
from rift.runtime.curriculum_run import WorldKit
from rift.schemas import Checkpoint
from rift.training.contracts import Rollout, UpdateRequest
from rift.schemas import Tokens
from rift.verl.actor_backend import validate_request

BEFORE = Checkpoint('service', '3', 't', 'w3')
SAMPLED = Checkpoint('service', '1', 't', 'w1')


def fake_sample(sid, group, *, text='{}', response=(5, 6, 7), refs=(), selected=None, status='certified',
                copy=0.0, checkpoint=SAMPLED):
    row = {'status': status, 'task_refs': list(refs), 'copy_score': copy,
           'selected': list(refs and (selected if selected is not None else [True] * len(refs)))}
    return T.TeacherSample(sid, group, checkpoint, (1, 2, 3), tuple(response), tuple(-0.5 for _ in response),
                           text, 'teacher-' + sid, 7, row)


class LearnabilityTests(unittest.TestCase):
    def test_peaks_at_even_odds(self):
        self.assertEqual(T.learnability(2, 4), 1.0)
        self.assertEqual(T.learnability(0, 4), 0.0)
        self.assertEqual(T.learnability(4, 4), 0.0)
        self.assertAlmostEqual(T.learnability(1, 4), 0.75)
        with self.assertRaises(ValueError):
            T.learnability(0, 0)


class GenerationTests(unittest.TestCase):
    def test_check_generation_rejects_unusable_output(self):
        good = SimpleNamespace(token_ids=[4, 5], log_probs=[-0.1, -2.0])
        self.assertEqual(T.check_generation(good), ([4, 5], [-0.1, -2.0]))
        for bad in (SimpleNamespace(token_ids=[], log_probs=[]),
                    SimpleNamespace(token_ids=[4], log_probs=None),
                    SimpleNamespace(token_ids=[4, 5], log_probs=[-0.1]),
                    SimpleNamespace(token_ids=[4], log_probs=[0.5]),
                    SimpleNamespace(token_ids=[4], log_probs=[float('nan')])):
            with self.assertRaises(ValueError):
                T.check_generation(bad)

    def test_a_group_is_independent_samples_of_one_prompt(self):
        calls = []

        class Client:
            async def generate(self, request_id, *, prompt_ids, sampling_params):
                calls.append((request_id, tuple(prompt_ids), dict(sampling_params)))
                return SimpleNamespace(token_ids=[9, len(calls)], log_probs=[-1.0, -0.5])
        marker = object()
        out = asyncio.run(T.generate_group(Client(), [1, 2, 3], count=3, seed=5, group_id='g',
                                           structured_schema={'type': 'object'},
                                           structured_factory=lambda schema: (marker, schema)))
        self.assertEqual(len(out), 3)
        self.assertEqual(len({c[0] for c in calls}), 3)                 # distinct request ids
        self.assertEqual({c[1] for c in calls}, {(1, 2, 3)})
        self.assertTrue(all(c[2]['logprobs'] is True and c[2]['structured_outputs'] == (marker, {'type': 'object'})
                            for c in calls))

    def test_structured_outputs_become_a_vllm_object_when_vllm_is_present(self):
        class Params:
            def __init__(self, json):
                self.json = json
        module = SimpleNamespace(StructuredOutputsParams=Params)
        with mock.patch.dict(sys.modules, {'vllm': SimpleNamespace(sampling_params=module),
                                           'vllm.sampling_params': module}):
            self.assertEqual(T.vllm_structured_outputs({'a': 1}).json, {'a': 1})
        with mock.patch.dict(sys.modules, {'vllm': None, 'vllm.sampling_params': None}):
            self.assertEqual(T.vllm_structured_outputs({'a': 1}), {'json': {'a': 1}})

    def test_prompt_ids_come_from_the_designer_prompt(self):
        class Tokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
                assert not tokenize and add_generation_prompt and not enable_thinking
                return ' '.join(m['content'][:3] for m in messages)

            def encode(self, text, add_special_tokens):
                assert not add_special_tokens
                return [ord(c) for c in text[:10]]
        ids = T.teacher_prompt_ids(Tokenizer(), [], seed=3)
        self.assertTrue(ids and all(isinstance(i, int) for i in ids))


class RewardTests(unittest.TestCase):
    def test_reward_is_mean_learnability_of_new_tasks(self):
        a = fake_sample('a', 'g', refs=['t1', 't2'])
        b = fake_sample('b', 'g', refs=['t3'])
        counts = {'t1': (2, 4), 't2': (0, 4), 't3': (1, 4)}
        rewards = T.plan_rewards([a, b], counts)
        self.assertAlmostEqual(rewards['a'], 0.5)             # (1.0 + 0.0) / 2
        self.assertAlmostEqual(rewards['b'], 0.75)

    def test_invalid_copied_and_repeated_plans(self):
        invalid = fake_sample('i', 'g', status='format_rejected')
        copied = fake_sample('c', 'g', refs=['t1'], copy=0.9)
        repeat = fake_sample('r', 'g', refs=['t2'], selected=[False])
        unmeasured = fake_sample('u', 'g', refs=['t9'])
        rewards = T.plan_rewards([invalid, copied, repeat, unmeasured], {'t1': (2, 4), 't2': (2, 4)},
                                 invalid_reward=-0.1)
        self.assertEqual(rewards, {'i': -0.1, 'c': -0.1, 'r': 0.0, 'u': None})


class UpdateRequestTests(unittest.TestCase):
    def _samples(self):
        return [fake_sample('g-p%d' % i, 'g', response=(5, 6, 7 + i), refs=['t%d' % i]) for i in range(4)]

    def test_request_groups_plans_by_prompt_and_binds_behavior_policy(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = T.TeacherLedger(root)
            samples = self._samples()
            for sample in samples:
                ledger.record(sample)
            rewards = {s.sample_id: r for s, r in zip(samples, (1.0, 0.5, 0.0, -0.1))}
            request = T.build_teacher_update(samples, rewards, BEFORE, ledger)
            self.assertEqual(request.algorithm, 'teacher_grpo')
            self.assertEqual(request.behavior_policies, (SAMPLED,))
            self.assertEqual(len(request.rollouts), 4)
            self.assertTrue(all(r.policy == SAMPLED and r.policy != BEFORE for r in request.rollouts))

    def test_groups_of_one_and_unmeasured_plans_are_left_out(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = T.TeacherLedger(root)
            samples = self._samples() + [fake_sample('h-p0', 'h', refs=['x'])]
            for sample in samples:
                ledger.record(sample)
            rewards = {s.sample_id: 0.5 for s in samples}
            rewards['g-p3'] = None
            request = T.build_teacher_update(samples, rewards, BEFORE, ledger)
            self.assertEqual(sorted(r.episode_id for r in request.rollouts), ['g-p0', 'g-p1', 'g-p2'])
            with self.assertRaises(ValueError):
                T.build_teacher_update(samples[:1], {'g-p0': 0.5}, BEFORE, ledger)

    def test_contract_rejects_bad_teacher_updates(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = T.TeacherLedger(root)
            samples = self._samples()
            for sample in samples:
                ledger.record(sample)
            rewards = {s.sample_id: float(i) for i, s in enumerate(samples)}
            request = T.build_teacher_update(samples, rewards, BEFORE, ledger)
            future = Checkpoint('service', '9', 't', 'w9')
            with self.assertRaises(ValueError):          # behavior policy newer than the weights being updated
                replace(request, behavior_policies=(future,))
            with self.assertRaises(ValueError):          # rollouts not from a declared behavior policy
                replace(request, behavior_policies=(Checkpoint('service', '2', 't', 'w2'),))
            with self.assertRaises(ValueError):          # wrong reward protocol
                replace(request, reward_protocol='dual_bank_outcome_v1')
            bare = request.rollouts[0]
            with self.assertRaises(ValueError):
                replace(request, rollouts=(replace(bare, behavior_logprobs=None),) + request.rollouts[1:])
            with self.assertRaises(ValueError):          # log-probabilities that do not line up with the tokens
                replace(bare, behavior_logprobs=(-1.0,))

    def test_ordinary_updates_cannot_carry_behavior_log_probabilities(self):
        row = Rollout(Tokens((1, 2), (3, 4), (True, True)), BEFORE, 't', 'e', 1.0, 'business_outcome_v2',
                      group_id='g', behavior_logprobs=(-1.0, -1.0))
        with self.assertRaises(ValueError):
            UpdateRequest(BEFORE, 'p', (row,), 'grpo')

    def test_backend_validation_and_ledger_verifier(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = T.TeacherLedger(root)
            samples = self._samples()
            for sample in samples:
                ledger.record(sample)
            rewards = {s.sample_id: float(i) for i, s in enumerate(samples)}
            request = T.build_teacher_update(samples, rewards, BEFORE, ledger)
            verifier = T.TeacherRolloutVerifier(ledger)
            self.assertIs(validate_request(request, 'service', verifier), request)
            with self.assertRaises(ValueError):
                validate_request(request, 'designer', verifier)
            tampered = replace(request.rollouts[0], tokens=Tokens((1, 2, 3), (9, 9, 9), (True,) * 3))
            self.assertFalse(verifier.verify_rollout(tampered, SAMPLED))
            wrong_source = replace(request, rollouts=(replace(request.rollouts[0], reward_source='business_outcome_v2'),)
                                   + request.rollouts[1:])
            with self.assertRaises(ValueError):
                validate_request(wrong_source, 'service', verifier)


class CertificationTests(unittest.TestCase):
    def test_example_plans_certify_and_copies_are_flagged(self):
        with tempfile.TemporaryDirectory() as root:
            kit = WorldKit(root)
            texts = dry_run_generate([[] for _ in range(4)], seed=11)
            samples = [fake_sample('c-p%d' % i, 'c', text=text) for i, text in enumerate(texts)]
            for index, sample in enumerate(samples):
                sample.plan_id, sample.world_seed = 'teacher-c-p%d' % index, 11 + index
            T.certify_samples(kit, samples, prompt_seed=11)
            certified = [s for s in samples if s.row['status'] in ('certified', 'partly_certified')]
            self.assertTrue(certified)
            # dry-run output is the generator's own example plans, so the copy detector must notice it
            self.assertTrue(all(s.row['copy_score'] is not None for s in samples))
            garbage = fake_sample('c-p9', 'c', text='not json')
            T.certify_samples(kit, [garbage], prompt_seed=11)
            self.assertEqual(garbage.row['status'], 'format_rejected')
            self.assertEqual(T.plan_rewards([garbage], {}, invalid_reward=-0.1), {'c-p9': -0.1})

    def test_signatures_seen_in_earlier_rounds_are_not_rewarded_again(self):
        with tempfile.TemporaryDirectory() as root:
            kit = WorldKit(root)
            text = dry_run_generate([[]], seed=21)[0]
            first, second = fake_sample('d-p0', 'd', text=text), fake_sample('d-p1', 'd', text=text)
            for sample in (first, second):
                sample.plan_id, sample.world_seed = 'teacher-' + sample.sample_id, 21
            seen = set()
            T.certify_samples(kit, [first], prompt_seed=999, copy_threshold=2.0, seen=seen)
            T.certify_samples(kit, [second], prompt_seed=999, copy_threshold=2.0, seen=seen)
            if first.selected_refs():
                self.assertEqual(second.selected_refs(), [])


if __name__ == '__main__':
    unittest.main()
