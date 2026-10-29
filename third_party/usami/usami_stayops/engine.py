from usami_harness.engine import Trace,Engine as _Engine,replay_trace as _replay
from . import adapter


class Engine(_Engine):
    def __init__(self,world,operation_budget=100):super().__init__(adapter,world,operation_budget)


def replay_trace(trace,trusted_requests=None):return _replay(adapter,trace,trusted_requests)
