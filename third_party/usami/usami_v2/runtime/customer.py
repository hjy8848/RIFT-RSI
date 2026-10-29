"""Customer-side intent, delegation and exact offer-consent lifecycle.

Private Customer policy is held by this runtime and must never be projected to
the Service. Service tool calls cannot create or widen a delegation/consent.
"""
from datetime import timedelta
import re
from typing import Dict, Literal, Optional, Tuple

from pydantic import Field, model_validator

from usami_harness.models import digest
from usami_v2.contracts import FactEvidence, FactKind, Scalar, Sealed, SourceRef
from usami_v2.commerce.invariants import invariants
from usami_v2.commerce.models import Consent, Delegation, PREPARATIONS, World
from usami_v2.scenarios.tasks import TaskSpecV3


class CustomerProfile(Sealed):
    """Private, fixed customer-side behavioral limits for one episode."""

    protocol: Literal["commerce-customer-profile-v1"] = "commerce-customer-profile-v1"
    customer_id: str
    delegated_line_ids: Tuple[str, ...] = ()
    allowed_preparations: Tuple[str, ...] = ("open_remedy_case", "verify_submitted_evidence",
                                            "request_policy_review", "prepare_remedy_quote")
    accepted_remedies: Tuple[Literal["refund", "exchange", "reship"], ...] = (
        "refund", "exchange", "reship")
    maximum_extra_charge_cents: int = Field(default=0, ge=0, strict=True)
    latest_arrival_days: Optional[int] = Field(default=None, ge=0, strict=True)
    return_pickup_slot: Optional[str] = None
    disclose_fact_ids: Tuple[str, ...] = ()
    private_facts: Tuple["PrivateCustomerFact", ...] = ()
    delegation_lifetime_hours: int = Field(default=24, ge=1, le=720, strict=True)
    consent_lifetime_hours: int = Field(default=1, ge=1, le=24, strict=True)

    @model_validator(mode="after")
    def unique_scopes(self):
        if len(set(self.delegated_line_ids)) != len(self.delegated_line_ids):
            raise ValueError("Customer profile repeats a delegated line")
        if len(set(self.allowed_preparations)) != len(self.allowed_preparations):
            raise ValueError("Customer profile repeats a preparation")
        if set(self.allowed_preparations) - PREPARATIONS:
            raise ValueError("Customer profile contains an unregistered preparation")
        if len(set(self.accepted_remedies)) != len(self.accepted_remedies):
            raise ValueError("Customer profile repeats a remedy")
        if len(set(self.disclose_fact_ids)) != len(self.disclose_fact_ids):
            raise ValueError("Customer profile repeats a discloseable fact")
        private_ids = {row.fact_id for row in self.private_facts}
        if len(private_ids) != len(self.private_facts) or not set(self.disclose_fact_ids).issubset(private_ids):
            raise ValueError("discloseable Customer facts must exist in the private profile")
        return self


class CustomerProfileV2(CustomerProfile):
    """Customer policy whose delegations cite public product wording in the request.

    The v2 field name is retained for archived Episode compatibility. New
    profiles store a line-specific subject mention there, allowing one public
    request to authorize several distinct owned lines without matching every
    product in the full sentence against each line.
    """

    protocol: Literal["commerce-customer-profile-v2"] = "commerce-customer-profile-v2"
    delegation_request_spans: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def all_delegations_have_public_intent(self):
        if set(self.delegation_request_spans) != set(self.delegated_line_ids):
            raise ValueError("each delegated line needs exactly one public-request span")
        if any(not " ".join(span.casefold().split()) for span in self.delegation_request_spans.values()):
            raise ValueError("delegation request spans cannot be blank")
        return self


class CustomerProfileV3(CustomerProfile):
    """Customer authority bound to the compiler-generated structured intent."""

    protocol: Literal["commerce-customer-profile-v3"] = "commerce-customer-profile-v3"
    intent_bundle_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


def bind_customer_profile(profile: CustomerProfile, task: TaskSpecV3):
    """Bind an existing fixed Customer policy to the compiler-issued task intent."""
    profile, task = profile.checked(), task.checked()
    if profile.customer_id != task.customer_id:
        raise ValueError("Customer policy identity differs from structured task intent")
    expected_lines = tuple(goal.line_id for goal in task.remedy_goals)
    if profile.delegated_line_ids and profile.delegated_line_ids != expected_lines:
        raise ValueError("Customer policy line scope differs from structured task intent")
    return CustomerProfileV3(
        customer_id=profile.customer_id, delegated_line_ids=expected_lines,
        allowed_preparations=profile.allowed_preparations, accepted_remedies=profile.accepted_remedies,
        maximum_extra_charge_cents=profile.maximum_extra_charge_cents,
        latest_arrival_days=profile.latest_arrival_days, return_pickup_slot=profile.return_pickup_slot,
        disclose_fact_ids=profile.disclose_fact_ids, private_facts=profile.private_facts,
        delegation_lifetime_hours=profile.delegation_lifetime_hours,
        consent_lifetime_hours=profile.consent_lifetime_hours,
        intent_bundle_hash=task.request_bundle.record_hash)


class HumanCustomerProfileV2(CustomerProfileV2):
    """A person's explicit acceptance also confirms the displayed return slot.

    Only the authenticated Human Playground creates this profile. Simulator
    profiles keep their fixed pickup preference. The required field preserves
    the hashes and meaning of existing v1/v2 profiles and archived Episodes.
    """

    return_pickup_policy: Literal["accept_presented_quote"]


class PrivateCustomerFact(Sealed):
    fact_id: str
    subject_id: str
    kind: FactKind
    value: Scalar
    unit: str = ""


CustomerProfile.model_rebuild()
CustomerProfileV2.model_rebuild()
CustomerProfileV3.model_rebuild()
HumanCustomerProfileV2.model_rebuild()


class CustomerDecision(Sealed):
    """Structured Customer-side intent produced by a simulator or a person."""

    message: str = ""
    delegate_line_ids: Tuple[str, ...] = ()
    allowed_preparations: Tuple[str, ...] = ()
    accept_quote_id: Optional[str] = None
    accept_operation: Optional[Literal["commit_accepted_quote", "release_accepted_refund"]] = None
    reject_quote_id: Optional[str] = None
    withdraw_delegation_ids: Tuple[str, ...] = ()
    disclose_fact_ids: Tuple[str, ...] = ()
    finish: bool = False

    @model_validator(mode="after")
    def coherent_decision(self):
        if self.accept_quote_id and self.reject_quote_id:
            raise ValueError("a Customer cannot accept and reject offers in one turn")
        if self.accept_quote_id is None and self.accept_operation is not None:
            raise ValueError("consent operation requires an accepted quote")
        if self.accept_quote_id is not None and self.accept_operation is None:
            raise ValueError("quote consent must name the exact business operation")
        if self.allowed_preparations and not self.delegate_line_ids:
            raise ValueError("preparations require delegated business lines")
        for values, label in ((self.delegate_line_ids, "delegated lines"),
                              (self.allowed_preparations, "preparations"),
                              (self.withdraw_delegation_ids, "withdrawals"),
                              (self.disclose_fact_ids, "facts")):
            if len(set(values)) != len(values):
                raise ValueError("Customer decision repeats " + label)
        return self


class OfferPresentation(Sealed):
    """An actual quote rendered to the Customer before its consent can exist."""

    quote_id: str
    quote_hash: str
    operation: Literal["commit_accepted_quote", "release_accepted_refund"]
    event_id: str
    event_index: int = Field(ge=0, strict=True)
    service_message: str = Field(min_length=1)
    rendered_terms: str = Field(min_length=1)


class AuthorityProof(Sealed):
    kind: Literal["delegation", "consent", "withdrawal"]
    customer_id: str
    source_customer_event_hash: str
    target_id: str
    terms_hash: str = ""
    issued_at: str
    profile_hash: str


class CustomerEvent(Sealed):
    event_id: str
    event_index: int = Field(ge=0, strict=True)
    kind: Literal["initial_request", "customer_turn"]
    customer_id: str
    message: str = ""
    decision_hash: str
    before_state_hash: str
    after_state_hash: str
    authority_proof_hashes: Tuple[str, ...] = ()
    disclosed_fact_ids: Tuple[str, ...] = ()
    finish_requested: bool = False
    finish_allowed: bool = True
    rejection_reasons: Tuple[str, ...] = ()


class CustomerTransition(Sealed):
    world: World
    event: CustomerEvent
    authority_proofs: Tuple[AuthorityProof, ...] = ()
    disclosed_facts: Tuple[FactEvidence, ...] = ()


def render_offer(world: World, quote_id: str, operation: str, event_id: str,
                 event_index: int, service_message: str, *, expected_customer_id: str) -> OfferPresentation:
    quote = world.quotes.get(quote_id)
    allowed_status = "prepared" if operation == "commit_accepted_quote" else "committed"
    if quote is None or quote.status != allowed_status:
        raise ValueError("offer status does not match the requested Customer consent operation")
    if quote.customer_id not in world.customers or quote.customer_id != expected_customer_id:
        raise ValueError("quote belongs to a different or absent Customer")
    if operation == "release_accepted_refund" and quote.remedy != "refund":
        raise ValueError("refund-release consent requires a refund quote")
    terms = {
        "quote_id": quote.id, "remedy": quote.remedy, "quantity": quote.quantity,
        "target_sku_id": quote.target_sku_id, "refund_cents": quote.refund_cents,
        "extra_charge_cents": quote.extra_charge_cents,
        "promised_arrival": quote.promised_arrival.isoformat() if quote.promised_arrival else None,
        "return_required": quote.return_required, "pickup_slot": quote.pickup_slot,
        "payment_channel": quote.payment_channel, "terms_hash": quote.terms_hash,
    }
    rendered = "Offer terms: " + str(terms)
    return OfferPresentation(quote_id=quote.id, quote_hash=digest(quote), operation=operation,
                             event_id=event_id, event_index=event_index,
                             service_message=service_message, rendered_terms=rendered)


def _explicit_consent_intent(message: str) -> Optional[bool]:
    """Conservative lexical check for structured quote consent.

    This is intentionally narrow: ambiguity returns None and cannot create
    authority. It is not a general natural-language understanding score.
    """
    text = " ".join(message.casefold().split())
    negative = (
        r"\b(?:i\s+)?(?:do\s+not|don't|cannot|can't|will\s+not|won't)\s+"
        r"(?:accept|agree|confirm|approve|authorize|consent|proceed)\b",
        r"\b(?:no|reject|decline)\b",
        r"不接受|不同意|拒绝|不要|不确认|不授权",
    )
    if any(re.search(pattern, text) for pattern in negative):
        return False
    affirmative = (
        r"\b(?:i\s+)?(?:accept|agree|confirm|approve|authorize|consent)\b",
        r"\b(?:yes|approved|confirmed)\b",
        r"我接受|我同意|我确认|同意|接受|确认|批准",
    )
    if any(re.search(pattern, text) for pattern in affirmative):
        return True
    return None


class CustomerRuntime:
    """Apply Customer-originated business events; never trusts Service requests as consent."""

    def __init__(self, profile: CustomerProfile, *, allow_legacy: bool = False, task=None):
        self.profile = profile.checked()
        self.allow_legacy = allow_legacy
        self.task = task.checked() if task is not None else None
        if self.profile.protocol == "commerce-customer-profile-v3":
            if not isinstance(self.task, TaskSpecV3):
                raise ValueError("CustomerProfileV3 requires its structured TaskSpecV3")
            if self.profile.intent_bundle_hash != self.task.request_bundle.record_hash:
                raise ValueError("Customer profile is not bound to this task's structured request bundle")
            expected_lines = tuple(goal.line_id for goal in self.task.remedy_goals)
            if self.profile.customer_id != self.task.customer_id or self.profile.delegated_line_ids != expected_lines:
                raise ValueError("Customer profile scope differs from structured task intent")

    def initialize(self, world: World, public_request: str, *, event_index: int = 0) -> CustomerTransition:
        if not public_request.strip():
            raise ValueError("the public Customer request is required")
        if self.profile.customer_id not in world.customers:
            raise ValueError("fixed Customer identity is absent from the frozen world")
        owned_orders = {row.id for row in world.orders.values()
                        if row.customer_id == self.profile.customer_id}
        owned_subjects = {self.profile.customer_id, *owned_orders}
        owned_subjects.update(row.id for row in world.lines.values() if row.order_id in owned_orders)
        if any(row.subject_id not in owned_subjects for row in self.profile.private_facts):
            raise ValueError("private Customer fact is outside the authenticated identity scope")
        for line_id in self.profile.delegated_line_ids:
            line = world.lines.get(line_id)
            order = world.orders.get(line.order_id) if line else None
            if not order or order.customer_id != self.profile.customer_id:
                raise ValueError("Customer profile delegation crosses identity scope")
        if self.profile.delegated_line_ids and self.profile.protocol not in (
                "commerce-customer-profile-v2", "commerce-customer-profile-v3"):
            if not self.allow_legacy:
                raise ValueError("new Episodes require a versioned Customer request binding")
        if self.profile.protocol == "commerce-customer-profile-v2":
            normalized_request = " ".join(public_request.casefold().split())
            for line_id, request_span in self.profile.delegation_request_spans.items():
                span = " ".join(request_span.casefold().split())
                if span not in normalized_request:
                    raise ValueError("Customer delegation wording is not grounded in the public request text")
                # New Research Studio profiles store a line-specific product
                # phrase and use the complete public request only as issue
                # context. The fallback preserves older v2 profiles that
                # stored the whole quoted request in this field.
                resolved = ()
                from usami_v2.scenarios.tasks import public_line_match_ids
                resolved = public_line_match_ids(world, self.profile.customer_id, span, public_request)
                if resolved != (line_id,):
                    resolved = public_line_match_ids(world, self.profile.customer_id, span)
                if resolved != (line_id,):
                    raise ValueError("Customer delegation wording must uniquely identify its frozen business line")
        before = world
        new = world.clone()
        event_id = "customer_initial_" + digest([self.profile.record_hash, public_request])[:20]
        if self.profile.protocol == "commerce-customer-profile-v1" and self.allow_legacy:
            # Preserve byte-for-byte Replay semantics for archived v1 Episodes.
            request_hash = digest({"customer_id": self.profile.customer_id, "message": public_request,
                                   "profile_hash": self.profile.record_hash})
        else:
            request_hash = digest({"customer_id": self.profile.customer_id, "message": public_request,
                                   "event_index": event_index, "kind": "initial_request"})
        proofs = ()
        if self.profile.delegated_line_ids:
            did = "delegation_" + digest([event_id, self.profile.delegated_line_ids,
                                          self.profile.allowed_preparations])[:20]
            delegation = Delegation(
                id=did, customer_id=self.profile.customer_id, line_ids=self.profile.delegated_line_ids,
                allowed_preparations=self.profile.allowed_preparations,
                expires_at=world.clock + timedelta(hours=self.profile.delegation_lifetime_hours),
                source_customer_event_hash=request_hash)
            new.delegations[did] = delegation
            proofs = (AuthorityProof(kind="delegation", customer_id=self.profile.customer_id,
                                     source_customer_event_hash=request_hash, target_id=did,
                                     issued_at=world.clock.isoformat(), profile_hash=self.profile.record_hash),)
        errors = invariants(new)
        if errors:
            raise ValueError("Customer initialization violates world invariants: " + str(errors[0]))
        event = CustomerEvent(
            event_id=event_id, event_index=event_index, kind="initial_request", customer_id=self.profile.customer_id,
            message=public_request, decision_hash=request_hash, before_state_hash=before.state_hash,
            after_state_hash=new.state_hash, authority_proof_hashes=tuple(row.record_hash for row in proofs))
        return CustomerTransition(world=new, event=event, authority_proofs=proofs)

    def apply(self, world: World, decision: CustomerDecision, *,
              presentation: Optional[OfferPresentation] = None, event_index: int) -> CustomerTransition:
        decision, profile = decision.checked(), self.profile
        if decision.delegate_line_ids and not set(decision.delegate_line_ids).issubset(profile.delegated_line_ids):
            raise ValueError("Customer attempted to delegate a line outside its private profile")
        if decision.allowed_preparations and not set(decision.allowed_preparations).issubset(profile.allowed_preparations):
            raise ValueError("Customer attempted to delegate an unapproved preparation")
        if decision.disclose_fact_ids and not set(decision.disclose_fact_ids).issubset(profile.disclose_fact_ids):
            raise ValueError("Customer attempted to disclose a fact outside its private profile")
        if decision.delegate_line_ids and profile.protocol == "commerce-customer-profile-v2":
            normalized_message = " ".join(decision.message.casefold().split())
            for line_id in decision.delegate_line_ids:
                span = " ".join(profile.delegation_request_spans[line_id].casefold().split())
                if span not in normalized_message:
                    raise ValueError("Customer delegation must be supported by its visible turn text")
        if decision.accept_quote_id or decision.reject_quote_id:
            quote_id = decision.accept_quote_id or decision.reject_quote_id
            if presentation is None or presentation.quote_id != quote_id:
                raise ValueError("Customer can answer only an actually presented quote")
            quote = world.quotes.get(quote_id)
            if quote is None or digest(quote) != presentation.quote_hash or quote.customer_id != profile.customer_id:
                raise ValueError("presented quote no longer matches the live Customer/quote binding")
            expected_presentation = render_offer(
                world, presentation.quote_id, presentation.operation, presentation.event_id,
                presentation.event_index, presentation.service_message,
                expected_customer_id=profile.customer_id)
            if presentation != expected_presentation:
                raise ValueError("presented quote terms differ from the actual business quote")
            if profile.protocol == "commerce-customer-profile-v2":
                consent_intent = _explicit_consent_intent(decision.message)
                if decision.accept_quote_id and consent_intent is not True:
                    raise ValueError("structured quote acceptance lacks clear affirmative Customer text")
                if decision.reject_quote_id and consent_intent is not False:
                    raise ValueError("structured quote rejection lacks clear negative Customer text")
            if decision.accept_quote_id:
                if profile.protocol == "commerce-customer-profile-v3":
                    case = world.cases.get(quote.case_id)
                    goal = next((row for row in self.task.remedy_goals
                                 if case is not None and row.line_id == case.line_id), None)
                    if goal is None:
                        raise ValueError("quote is outside the Customer's structured task intent")
                    if quote.quantity != goal.quantity or quote.remedy not in goal.allowed_remedies:
                        raise ValueError("quote terms differ from the Customer's structured task intent")
                    if (goal.maximum_extra_charge_cents is not None
                            and quote.extra_charge_cents > goal.maximum_extra_charge_cents):
                        raise ValueError("quote exceeds the structured intent's extra-charge limit")
                    if goal.latest_arrival is not None and quote.promised_arrival is not None:
                        if quote.promised_arrival > goal.latest_arrival:
                            raise ValueError("quote exceeds the structured intent's arrival deadline")
                    if goal.required_attributes:
                        sku = world.skus.get(quote.target_sku_id or "")
                        if sku is None or any(sku.attributes.get(row.name) != row.value
                                              for row in goal.required_attributes):
                            raise ValueError("quote does not meet the structured intent's required attributes")
                if decision.accept_operation != presentation.operation:
                    raise ValueError("Customer consent operation differs from the presented request")
                if quote.remedy not in profile.accepted_remedies:
                    raise ValueError("quote remedy violates private Customer preference")
                if quote.extra_charge_cents > profile.maximum_extra_charge_cents:
                    raise ValueError("quote exceeds private Customer payment limit")
                if profile.latest_arrival_days is not None and quote.promised_arrival is not None:
                    if quote.promised_arrival > world.clock + timedelta(days=profile.latest_arrival_days):
                        raise ValueError("quote exceeds private Customer arrival limit")
                if (quote.return_required and not isinstance(profile, HumanCustomerProfileV2)
                        and quote.pickup_slot != profile.return_pickup_slot):
                    raise ValueError("quote requires a return the Customer has not agreed to")

        before, new = world, world.clone()
        event_id = "customer_turn_" + digest([profile.record_hash, event_index, decision.record_hash])[:20]
        authority = []
        customer_source_hash = digest({"event_id": event_id, "decision_hash": decision.record_hash,
                                       "message": decision.message, "profile_hash": profile.record_hash})
        disclosed = []
        private_facts = {row.fact_id: row for row in profile.private_facts}
        for fact_id in decision.disclose_fact_ids:
            fact = private_facts[fact_id]
            source = SourceRef(source_id=event_id + "-" + fact_id, source_kind="customer_message",
                               source_hash=customer_source_hash, event_index=event_index)
            disclosed.append(FactEvidence(
                fact_id="customer-disclosure-" + digest([event_id, fact_id])[:20],
                subject_id=fact.subject_id, kind=fact.kind, value=fact.value, unit=fact.unit,
                valid_from=event_index, source=source))
        if decision.delegate_line_ids:
            did = "delegation_" + digest([event_id, decision.delegate_line_ids,
                                          decision.allowed_preparations])[:20]
            new.delegations[did] = Delegation(
                id=did, customer_id=profile.customer_id, line_ids=decision.delegate_line_ids,
                allowed_preparations=decision.allowed_preparations,
                expires_at=world.clock + timedelta(hours=profile.delegation_lifetime_hours),
                source_customer_event_hash=customer_source_hash)
            authority.append(AuthorityProof(kind="delegation", customer_id=profile.customer_id,
                                            source_customer_event_hash=customer_source_hash, target_id=did,
                                            issued_at=world.clock.isoformat(), profile_hash=profile.record_hash))
        if decision.accept_quote_id:
            quote = new.quotes[decision.accept_quote_id]
            cid = "consent_" + digest([event_id, quote.id, quote.terms_hash, decision.accept_operation])[:20]
            consent = Consent(
                id=cid, customer_id=profile.customer_id, quote_id=quote.id, terms_hash=quote.terms_hash,
                operation=decision.accept_operation, presented_offer_event_hash=presentation.record_hash,
                source_customer_event_hash=customer_source_hash,
                expires_at=world.clock + timedelta(hours=profile.consent_lifetime_hours))
            new.consents[cid] = consent
            authority.append(AuthorityProof(kind="consent", customer_id=profile.customer_id,
                                            source_customer_event_hash=customer_source_hash, target_id=cid,
                                            terms_hash=quote.terms_hash, issued_at=world.clock.isoformat(),
                                            profile_hash=profile.record_hash))
        if decision.withdraw_delegation_ids:
            for did in decision.withdraw_delegation_ids:
                row = new.delegations.get(did)
                if not row or row.customer_id != profile.customer_id:
                    raise ValueError("Customer can withdraw only its own existing delegation")
                row.withdrawn = True
                authority.append(AuthorityProof(kind="withdrawal", customer_id=profile.customer_id,
                                                source_customer_event_hash=customer_source_hash, target_id=did,
                                                issued_at=world.clock.isoformat(), profile_hash=profile.record_hash))
        if decision.reject_quote_id:
            # A rejection is recorded as a Customer-originated event, with no consent created.
            pass
        if invariants(new):
            raise ValueError("Customer event violates native business invariants")
        pending_consent = any(row.customer_id == profile.customer_id and not row.withdrawn
                              and row.consumed_by is None and row.expires_at >= new.clock
                              for row in new.consents.values())
        reasons = ("unconsumed_customer_authority_requires_service_opportunity",) if decision.finish and pending_consent else ()
        event = CustomerEvent(
            event_id=event_id, event_index=event_index, kind="customer_turn", customer_id=profile.customer_id,
            message=decision.message, decision_hash=decision.record_hash, before_state_hash=before.state_hash,
            after_state_hash=new.state_hash, authority_proof_hashes=tuple(row.record_hash for row in authority),
            disclosed_fact_ids=decision.disclose_fact_ids, finish_requested=decision.finish,
            finish_allowed=not pending_consent or not decision.finish, rejection_reasons=reasons)
        return CustomerTransition(world=new, event=event, authority_proofs=tuple(authority),
                                  disclosed_facts=tuple(disclosed))

    def public_view(self, profile_details: Optional[dict] = None):
        """Customer-only model context. Callers must not pass this to Service."""
        return {"customer_id": self.profile.customer_id,
                "allowed_preparations": list(self.profile.allowed_preparations),
                "accepted_remedies": list(self.profile.accepted_remedies),
                "maximum_extra_charge_cents": self.profile.maximum_extra_charge_cents,
                "latest_arrival_days": self.profile.latest_arrival_days,
                "return_pickup_slot": self.profile.return_pickup_slot,
                "private_facts": [row.model_dump(mode="json") for row in self.profile.private_facts],
                "private_profile_details": profile_details or {}}
