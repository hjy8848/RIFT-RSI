import sys
import unittest
from types import SimpleNamespace

sys.path[:0] = ['src', 'third_party/usami']

from rift.schemas import Checkpoint, Tokens
from rift.training import service_data
from usami_v2.runtime import runner


def _episode(name, task, group, verified):
    call = SimpleNamespace(role='service', request={'group_id': group, 'group_size': 2})
    return SimpleNamespace(record_hash=name, task=SimpleNamespace(record_hash=task),
        model_calls=(call,), verified=verified,
        outcome=SimpleNamespace(protocol='commerce-outcome-v2', business_goal_success=True))


class CaptureStore:
    def verify_trajectory(self, episode, checkpoint):
        return True

    def trajectory_for(self, episode, checkpoint):
        return Tokens((1,), (2, 3), (True, True))

    def group_for(self, episode, checkpoint):
        return episode.model_calls[0].request['group_id']


class UnverifiedReplayTests(unittest.TestCase):
    def test_a_replay_that_disagrees_drops_its_group_instead_of_ending_the_run(self):
        original = runner.replay_episode
        runner.replay_episode = lambda episode: SimpleNamespace(verified=episode.verified)
        try:
            checkpoint = Checkpoint('service', '0', 'tok', 'w0')
            episodes = [_episode('a1', 'task-a', 'group-a', True), _episode('a2', 'task-a', 'group-a', True),
                        _episode('b1', 'task-b', 'group-b', True), _episode('b2', 'task-b', 'group-b', False)]
            request = service_data.prepare_service_trajectory_update('pkg', {'task-a', 'task-b'}, episodes,
                checkpoint, CaptureStore(), algorithm='grpo', allow_partial_groups=True)
        finally:
            runner.replay_episode = original
        self.assertEqual({row.task_id for row in request.rollouts}, {'task-a'})   # group b dropped whole
        self.assertEqual(len(request.rollouts), 2)

    def test_nothing_usable_is_still_an_error(self):
        original = runner.replay_episode
        runner.replay_episode = lambda episode: SimpleNamespace(verified=False)
        try:
            checkpoint = Checkpoint('service', '0', 'tok', 'w0')
            episodes = [_episode('a1', 'task-a', 'group-a', False), _episode('a2', 'task-a', 'group-a', False)]
            with self.assertRaises(ValueError):
                service_data.prepare_service_trajectory_update('pkg', {'task-a'}, episodes, checkpoint,
                    CaptureStore(), algorithm='grpo', allow_partial_groups=True)
        finally:
            runner.replay_episode = original


if __name__ == '__main__':
    unittest.main()
