from .models import World,Operation,PARAMETERS,TOOLS,VERSION
from .generator import generate_world
from .rules import decide,invariants,POLICY
from .transitions import apply
from .observations import observation,visible_ids
from .customer import private_state,validate_private,intent_allowed,normalize_intent,offer_for,consent_allowed,candidate_intents,external_ready
from .external import expected as external_reference,apply as external_apply
name='property'
version=VERSION
DESCRIPTIONS={n:n.replace('_',' ')+'. Authenticated owner/resident with explicit property and unit scope; never invent approvals or work completion.' for n in TOOLS}
from .tasks import validate_task,customer_task_brief,task_intent_allowed
