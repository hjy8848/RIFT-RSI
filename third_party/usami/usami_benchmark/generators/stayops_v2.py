"""Independent StayOps distribution under the explicit 0.2 rule revision."""
from usami_benchmark.native import stayops_v2 as adapter
from .stayops import _generate

GENERATOR_VERSION='stayops-structural-v2'
RULES_VERSION='stayops-0.2.0'


def generate_world(seed=42,config=None,*,diagnostics=None):
    return _generate(seed,config,diagnostics=diagnostics,world_type=adapter.World,
        native_adapter=adapter,distribution_version=GENERATOR_VERSION)
