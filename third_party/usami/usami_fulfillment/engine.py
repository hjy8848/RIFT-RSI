from usami_harness.engine import Engine as DomainEngine, Trace
from usami_harness.engine import replay_trace as replay
from . import adapter


class Engine(DomainEngine):
    def __init__(self, world, operation_budget=100):
        super().__init__(adapter,world,operation_budget)


def replay_trace(trace, trusted_requests=None):
    return replay(adapter,trace,trusted_requests)
