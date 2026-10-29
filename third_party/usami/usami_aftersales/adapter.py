from .models import World,Operation,PARAMETERS,TOOLS,VERSION
from .generator import generate_world
from .rules import decide,invariants,POLICY
from .transitions import apply
from .observations import observation,visible_ids
from .customer import private_state,validate_private,offer_for,consent_allowed,intent_allowed,normalize_intent,candidate_intents,external_ready
from .external import expected as external_reference,apply as external_apply

name='aftersales'
version=VERSION
OFFERABLE_OPERATIONS=('commit_resolution','release_refund')
DESCRIPTIONS={n:n.replace('_',' ')+'. Operations use authenticated buyer and explicit merchant/order scope.' for n in TOOLS}
from .tasks import validate_task,customer_task_brief,task_intent_allowed
