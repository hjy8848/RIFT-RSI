"""An independent date-inventory hotel operations business domain."""
from .models import World,Operation
from .generator import generate_world

__all__=['World','Operation','generate_world']
