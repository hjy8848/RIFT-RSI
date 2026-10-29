from usami_harness.benchmark import Instance, Manifest, Run, write_json
from usami_harness import benchmark as harness
from . import adapter


def runtime_hashes():return harness.runtime_hashes(adapter)
def validate(manifest, **kwargs):return harness.validate(adapter,manifest,**kwargs)
def load(path, **kwargs):return harness.load(adapter,path,**kwargs)
def run_instance(manifest,instance,service,customer=None,trial='r0'):return harness.run_instance(adapter,manifest,instance,service,customer,trial)
def replay_run(manifest,run):return harness.replay_run(adapter,manifest,run)
def score(manifest,run):return harness.score(adapter,manifest,run)
def summarize(manifest,runs):return harness.summarize(adapter,manifest,runs)
def assert_comparable(left,right):return harness.assert_comparable(adapter,left,right)
