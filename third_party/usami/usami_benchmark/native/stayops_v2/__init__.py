"""Adapter for an explicit rule revision, sharing original independent mutations."""
from .models import World,VERSION
from .rules import decide,invariants,POLICY
from usami_stayops.models import Operation,PARAMETERS,TOOLS
from usami_stayops.transitions import apply
from usami_stayops.observations import observation,visible_ids
from usami_stayops.tools import DESCRIPTIONS
from usami_stayops.goals import GOAL_FIELDS,FRAME_FIELDS,DELETABLE

name='stayops'
version=VERSION
