import asyncio
import sys
import threading
import unittest

sys.path[:0] = ['src', 'third_party/usami']

from rift.verl.usami_session import episode_executor


class EpisodeExecutorTests(unittest.TestCase):
    def test_blocked_episodes_do_not_starve_the_default_pool(self):
        """More waiting episodes than the default pool has threads must not stop other jobs."""
        loop = asyncio.new_event_loop()
        release = threading.Event()
        count = 64   # more than the default pool's 32 threads
        try:
            async def scenario():
                running = asyncio.get_running_loop()
                blocked = [running.run_in_executor(episode_executor(), release.wait)
                           for _ in range(count)]
                # Stands in for the tokenisation jobs that use the loop's default pool.
                result = await asyncio.wait_for(running.run_in_executor(None, lambda: 'tokenised'), 5)
                release.set()
                await asyncio.gather(*blocked)
                return result
            self.assertEqual(loop.run_until_complete(scenario()), 'tokenised')
        finally:
            release.set()
            loop.close()

    def test_one_shared_executor(self):
        self.assertIs(episode_executor(), episode_executor())


if __name__ == '__main__':
    unittest.main()
