from usami_harness.oracle import evaluate as audit
from . import adapter


def evaluate(trace, trusted_requests=None):
    return audit(adapter,trace,trusted_requests)
