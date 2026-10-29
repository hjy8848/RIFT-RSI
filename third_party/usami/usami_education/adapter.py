from .models import World,Operation,PARAMETERS,TOOLS,VERSION
from .generator import generate_world
from .rules import decide,invariants,POLICY
from .transitions import apply
from .observations import observation,visible_ids
from .customer import private_state,validate_private,intent_allowed,normalize_intent,offer_for,consent_allowed,candidate_intents,external_ready
from .external import expected as external_reference,apply as external_apply
name='education'
version=VERSION
DESCRIPTIONS={n:n.replace('_',' ')+'. Authenticated student, school/enrollment scope, actual credit ledger and contract; never invent attendance, discounts or approval.' for n in TOOLS}
from .tasks import validate_task,customer_task_brief,task_intent_allowed
