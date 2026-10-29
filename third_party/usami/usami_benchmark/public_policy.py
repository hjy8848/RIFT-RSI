"""World-independent public rule/code catalogue for the new benchmark runner.

Only static business source and existing public policy descriptions are read.
No world, Reference decision, private profile, witness or future outcome is input.
Historical native prompt assemblers are not changed by this opt-in renderer.
"""
import ast,importlib,re,json
from functools import lru_cache
from pathlib import Path
from usami_harness.models import digest
from .contracts import DOMAINS

RULEBOOK={
 'saas':[
  'Membership and billing capabilities are organization-scoped; a role in another organization grants no authority.',
  'Active membership, available seats, subscription state, current invoices and paid funding constrain activation, renewal and reactivation.',
  'Protect the last owner. Respect exact requested changes and original payment channels, refund windows, refundable balances and idempotency keys.'
 ],
 'cloudops':[
  'Organization ownership manages billing/quota/access but does not imply project Admin/Operator permission for compute.',
  'Starting an instance requires active project, authorized project role, running dependencies and available CPU/memory/instance quota.',
  'Do not break running dependents. Restoring a suspended project requires its actual invoice to be paid; exact customer authority is independent of roles.'
 ],
 'fulfillment':[
  'Warehouse roles and organization boundaries govern stock, fulfillment, finance and receiving operations.',
  'Conserve physical, reserved and picked stock and finite warehouse volume. Receiving/inspection, payment, confirmation, reservation, picking, packing, shipment and delivery have native prerequisites.',
  'Carrier/receipt/inspection proofs are actual simulator records. Return windows, delivered quantities, restocking and original payment/refund limits govern reverse logistics.'
 ],
 'stayops':[
  'Manager/frontdesk/housekeeping/finance and guest rights are property-scoped; guest identity must be verified where required.',
  'Stay dates are half-open intervals. Real rooms, daily sale limits, active overlapping bookings, cleaning and maintenance constrain allocation and check-in.',
  'Honor hold expiry, deposits, paid balances, cancellation timing and refund limits. Checkout may dirty a room for a future booking; it does not make an occupied dirty room legal.'
 ],
 'aftersales':[
  'Authenticate the order buyer and merchant/order/line scope. Ask for the actual order, issue, evidence and relevant customer deadline before committing a remedy.',
  'A case can be opened only for an eligible delivered line inside its product return window. Opening a case and offering or agreeing to a remedy all use the actual delivery date and the same window; manager approval does not extend it.',
  'Product returnability, verified evidence, settlement state, merchant approval limit and replacement stock further constrain refund, exchange and reship.',
  'Return-required refunds need actual receipt; pickup, current remedy choice and irreversible settlement require the applicable exact customer confirmation. Use the original payment channel and remaining paid value.'
 ],
 'airline':[
  'Authenticate the booking passenger with the actual document; do not cancel or rebook based on an ambiguous request.',
  'Fare change/refund restrictions, notice windows, current flight disruption, cabin seats, connection time, baggage limits and payment capacity constrain offers.',
  'Clarify destination, departure/arrival constraints and fee budget; honor the current chosen itinerary and exact current quote consent. Airline cancellation differs from voluntary cancellation.'
 ],
 'property':[
  'Authenticate the unit owner/resident; protect other residents and contacts. Establish the actual asset, reported symptom, hazard, acceptable time and budget.',
  'Sparks require emergency triage. Contract responsibility, coverage, severity response windows, technician skill/on-call availability and conflicting calendars constrain visits.',
  'Obtain actual owner approval when needed. A booking is not proof of repair; only real work receipts establish completion. Inspection supersedes stale plans. Closure requires separate current customer consent.'
 ],
 'education':[
  'Authenticate the school-scoped enrollment owner; clarify the actual reason, budget, study needs and available hours.',
  'Refunds use the original contract window and cumulative consumed units divided by original contracted units; transfer/top-up never reset this progress limit. Refund unused credit minus contractual fee to the original channel.',
  'Transfers stay in the package/school, meet verified placement level, finite cohort seats, valid dated classes, student calendars and minimum remaining study units. Top-ups and case charges must fit actual customer authority/budget.',
  'Extensions respect reason, grace, cumulative day limit and minimum validity. Beyond self-service authority requires an actual approval bound to days and enrollment version; no model may invent approval or discounts.',
  'Tutoring needs qualified available teacher, student availability, sufficient unreserved credit and actual validity/duration. Booking reserves value; actual attendance or contract-governed no-show consumes it; teacher cancellation consumes none.',
  'Current exact quote consent is required. Enrollment or customer-choice changes invalidate stale quotes. Cancel pending lessons before refund/transfer; do not fabricate progress or double-spend reserved credit.'
 ]}


@lru_cache(maxsize=16)
def _cached_catalogue(domain,rules_version):
    if domain not in DOMAINS:raise ValueError('unknown public business domain')
    if domain=='saas':
        if rules_version!='saas-rules-0.2.0':raise ValueError('unregistered public SaaS rulebook')
        from usami.world.rules_v2 import public_policy_catalog
        module=importlib.import_module('usami.world.rules_v2');declared=public_policy_catalog()
    else:
        module=importlib.import_module('usami_benchmark.native.stayops_v2.rules' if domain=='stayops' and rules_version=='stayops-0.2.0' else 'usami_'+domain+'.rules')
        expected=getattr(importlib.import_module('usami_'+domain+'.models'),'VERSION')
        if not (rules_version==expected or domain=='stayops' and rules_version=='stayops-0.2.0'):raise ValueError('unregistered public rulebook version')
        declared=dict(module.POLICY)
    codes=set()
    for node in ast.walk(ast.parse(Path(module.__file__).read_text())):
        if isinstance(node,ast.Return):
            codes.update(x.value for x in ast.walk(node) if isinstance(x,ast.Constant) and isinstance(x.value,str) and re.fullmatch(r'[A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+',x.value))
    errors={code:declared.get(code,code.replace('_',' ').capitalize()+'. Consult the applicable current public contract and resource facts.') for code in sorted(codes|set(declared))}
    body={'protocol':'eight-domain-public-rulebook-v1','domain':domain,'rules_version':rules_version,'business_rules':RULEBOOK[domain],'errors':errors,
        'scope':'Static public policy and code vocabulary; not the expected answer for any particular world or task.'}
    return json.dumps({**body,'catalogue_hash':digest(body)},sort_keys=True,separators=(',',':'))


def catalogue(domain,rules_version):
    return json.loads(_cached_catalogue(domain,rules_version))
