import sys
import unittest

sys.path[:0] = ['src', 'third_party/usami']

from rift.runtime.curriculum_run import fit_request_to_batches
from rift.schemas import Checkpoint, Outcome, Tokens
from rift.training.contracts import UpdateRequest
from rift.training.records import service_rollout


def _request(group_outcomes, size=4):
    """group_outcomes: one list of Outcome values per GRPO group."""
    checkpoint = Checkpoint('service', '0', 'tok', 'w0')
    rows = []
    for g, outcomes in enumerate(group_outcomes):
        assert len(outcomes) == size
        for m, outcome in enumerate(outcomes):
            rows.append(service_rollout(Tokens((1,), (2, 3 + m), (True, True)), checkpoint, 'task%d' % g,
                                        'ep-%d-%d' % (g, m), outcome, 'group%d' % g, response_id='ep-%d-%d' % (g, m)))
    return UpdateRequest(checkpoint, 'pkg', tuple(rows), 'grpo')


S, F = Outcome.SUCCESS, Outcome.FAILURE
MIXED, ALLOK, ALLBAD = [S, F, S, F], [S] * 4, [F] * 4


class FitRequestTests(unittest.TestCase):
    def test_a_request_that_already_fits_is_untouched(self):
        request = _request([MIXED] * 4)                     # 16 rollouts, mini-batch 16
        self.assertIs(fit_request_to_batches(request, 16), request)

    def test_uniform_groups_are_dropped_before_informative_ones(self):
        # 9 groups = 36 rollouts: one group must go. Groups 1, 3 and 6 have no gradient.
        request = _request([MIXED, ALLOK, MIXED, ALLBAD, MIXED, MIXED, ALLOK, MIXED, MIXED])
        fitted = fit_request_to_batches(request, 16)
        kept = {row.group_id for row in fitted.rollouts}
        self.assertEqual(len(fitted.rollouts), 32)                       # 8 whole groups
        self.assertTrue({'group0', 'group2', 'group4', 'group5', 'group7', 'group8'} <= kept)
        self.assertEqual(len(kept & {'group1', 'group3', 'group6'}), 2)  # only a no-gradient group left
        self.assertEqual(len({row.group_id for row in fitted.rollouts}), 8)

    def test_the_failure_seen_in_x2_forty_four_rollouts_become_thirty_two(self):
        request = _request([MIXED] * 8 + [ALLOK, ALLBAD, MIXED])        # 11 groups = 44 rollouts
        fitted = fit_request_to_batches(request, 16)
        self.assertEqual(len(fitted.rollouts) % 16, 0)
        self.assertEqual(len(fitted.rollouts), 32)
        self.assertEqual({row.group_id for row in fitted.rollouts} & {'group8', 'group9'}, set())

    def test_less_than_one_batch_is_an_error(self):
        with self.assertRaises(ValueError):
            fit_request_to_batches(_request([MIXED] * 3), 16)           # 12 rollouts, nothing fits


if __name__ == '__main__':
    unittest.main()
