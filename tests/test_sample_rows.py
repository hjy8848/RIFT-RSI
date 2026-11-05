import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path[:0] = ['src', 'third_party/usami']

from rift.runtime import curriculum_run as run


def requests(n):
    reqs = [{'rift_request_id': 'r%d' % i} for i in range(n)]
    return reqs, {r['rift_request_id']: ('K', 't%d' % (i % 3)) for i, r in enumerate(reqs)}


class SampleRowsTests(unittest.TestCase):
    def _sample(self, n, workers):
        seen = {}

        def generate(self_, reqs):
            seen['n'] = len(reqs)
            return [{'rift_request_id': r['rift_request_id'], 'rift_outcome': 'success',
                     'rift_replay_verified': True, 'rift_episode_id': 'e' + r['rift_request_id']} for r in reqs]
        context = SimpleNamespace(group_size=4)
        with mock.patch.dict(os.environ, {'RIFT_AGENT_WORKERS': str(workers)}), \
                mock.patch.object(run, 'build_eval_requests', lambda *a, **k: requests(n)), \
                mock.patch.object(run.LazyBatch, 'generate', generate):
            rows = run.sample_rows(context, {}, None, groups=1, seed=0, label='x')
        return seen['n'], rows

    def test_an_uneven_batch_loses_only_its_last_episodes(self):
        sent, rows = self._sample(516, 8)           # 129 tasks x 4 samples over 8 workers
        self.assertEqual((sent, len(rows)), (512, 512))
        self.assertEqual(rows[-1]['episode_id'], 'er511')

    def test_an_even_batch_is_untouched(self):
        self.assertEqual(self._sample(768, 8)[0], 768)

    def test_a_batch_smaller_than_the_worker_count_is_refused(self):
        with self.assertRaises(ValueError):
            self._sample(5, 8)


if __name__ == '__main__':
    unittest.main()
