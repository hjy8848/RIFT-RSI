"""Two settlements on one line must replay identically after the record is stored and reloaded.

Stored records re-sort their dictionaries, so goal evidence that followed settlement insertion
order differed after a reload; replay then reported "task outcome evaluation differs" for an
episode whose verdict was identical. A Service that refunded the same two items twice hit it.
"""
import json
import re
import sys
import tempfile
import unittest

sys.path[:0] = ['src', 'third_party/usami', 'tests']

from rift.runtime import curriculum_run as run
from test_usami_phase3 import ScriptedService, YesCustomer, gather
from usami_v2.runtime.runner import EpisodeBudget, EpisodeRecord, replay_episode, run_episode


def _refund_twice(ctx):
    yield from gather(ctx)
    request = ctx.prompt.customer_request.lower()
    row = [r for r in ctx.lines if all(
        re.search(r'\b%s\b' % re.escape(str(v).replace('_', ' ').lower()), request)
        for v in r['sku']['attributes'].values())][0]
    yield ctx.say('I will refund 2 items. Shall I proceed?')
    yield ctx.op('return_delivered_order_items', row['id'], quantity=2, payment_method='original')
    yield ctx.op('return_delivered_order_items', row['id'], quantity=2, payment_method='original')
    yield ctx.say('Done.')


class DoubleRefundReplayTests(unittest.TestCase):
    def test_a_repeated_refund_replays_after_a_store_and_reload(self):
        budget = EpisodeBudget(max_customer_turns=8, max_service_decisions=30,
                               max_service_tool_calls=24, max_model_calls=40)
        with tempfile.TemporaryDirectory() as root:
            kit = run.WorldKit(root, episode_budget=budget)
            for seed in range(6000, 6006):
                refs = kit.certify_specs([{'family': 'F2', 'knobs': {'purchased_quantity': 5, 'refund_quantity': 2},
                                           'instances': 1, 'seed': seed}])
                record = kit.task_registry.resolve_record(next(iter(refs.values()))[0])
                episode = run_episode(record.world, record.task, record.profile, YesCustomer(1),
                                      ScriptedService(_refund_twice), budget=budget,
                                      external_events=record.external_events)
                settled = [key for key in episode.final_world.settlements
                           if key not in episode.initial_world.settlements]
                self.assertEqual(len(settled), 2)                     # the case under test
                stored = json.loads(json.dumps(episode.model_dump(mode='json'), sort_keys=True))
                reloaded = EpisodeRecord.model_validate(stored)
                self.assertTrue(replay_episode(reloaded).verified, 'seed %d' % seed)


if __name__ == '__main__':
    unittest.main()
