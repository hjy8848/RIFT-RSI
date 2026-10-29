from usami_harness.oracle import evaluate as _evaluate
from . import adapter


def evaluate(trace,trusted_requests=None):return _evaluate(adapter,trace,trusted_requests)
