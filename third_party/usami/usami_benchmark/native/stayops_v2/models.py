"""Schema revision for prospective room-turnover semantics."""
from typing import Literal
from usami_stayops.models import World as LegacyWorld

VERSION='stayops-0.2.0'


class World(LegacyWorld):
    version: Literal['stayops-0.2.0']=VERSION
