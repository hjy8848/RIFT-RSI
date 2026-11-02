"""Rehearse an Ascend worker's load on CPU: many concurrent native sessions.

A worker holds one session per sample (44 in the X1 screen). Each session occupies a
thread while it waits for a model reply, and between replies the Agent Loop uses the
event loop's default pool for tokenisation. This reproduces that mix so a thread-pool
or event-loop deadlock fails here, not after a 40 minute NPU start-up.
"""
import asyncio
import sys

sys.path[:0] = ['src', 'third_party/usami', 'tests']

from rift.schemas import Outcome


def test_many_concurrent_sessions_finish_while_tokenisation_uses_the_default_pool():
    from rift_usami_fixtures import (frozen, refund_task, refund_profile, ConfirmingCustomer,
        PublicPathRefundService, flat_tool_call)
    from usami_v2.runtime.runner import EpisodeBudget
    from rift.benchmark.usami import public_episode
    from rift.verl.usami_session import NativeUsamiSession

    world, aliases = frozen()
    sessions = 100   # far above the default pool's 32 threads

    async def one(index):
        script = PublicPathRefundService(world, aliases)

        class Parser:
            async def extract_tool_calls(self, token_ids, schemas):
                return flat_tool_call(script.decide(None))

        session = NativeUsamiSession(world, refund_task(world, aliases), refund_profile(aliases),
            ConfirmingCustomer(), Parser(), budget=EpisodeBudget(max_customer_turns=3,
                max_service_decisions=10, max_service_tool_calls=8, max_model_calls=10))
        loop = asyncio.get_running_loop()
        try:
            await session.start()
            await loop.run_in_executor(None, lambda: None)      # stands in for ct_build_initial_tokens
            done, step = False, 0
            while not done:
                _, _, done = await session.advance([step + 10], [1, step + 1], {'temperature': 1.0})
                if not done:
                    await loop.run_in_executor(None, lambda: None)  # stands in for ct_merge_context_msg
                step += 1
            return public_episode(await session.finish()).outcome
        finally:
            await session.abort()

    async def everything():
        return await asyncio.wait_for(asyncio.gather(*(one(i) for i in range(sessions))), 60)

    outcomes = asyncio.run(everything())
    assert outcomes == [Outcome.SUCCESS] * sessions
