"""Commerce 2.0 adapter for the unchanged native Engine, Oracle and Replay."""
from . import VERSION
from .models import World, Operation, PARAMETERS
from .invariants import invariants
from .reference import decide
from .transitions import apply

name = "commerce"
version = VERSION
TOOLS = tuple(PARAMETERS)
DESCRIPTIONS = {name: name.replace("_", " ") for name in TOOLS}
