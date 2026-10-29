from usami_harness.suite import run_suite as _run
from . import adapter


def run_suite(manifest,out,clients,max_runs=None):return _run(adapter,manifest,out,clients,max_runs)
