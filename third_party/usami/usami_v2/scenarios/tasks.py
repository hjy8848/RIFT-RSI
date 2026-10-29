"""Open, multi-line business objectives; evaluation accepts any compliant SKU."""
from datetime import datetime
import re
from typing import Literal, Optional, Tuple

from pydantic import Field, field_validator, model_validator

from vast_coevo.schema import digest
from usami_v2.contracts import (
    CommunicationObligation, FactEvidence, FactKind, PublicRequest, Sealed,
    ServiceMessage, SourceRef, aware,
)
from usami_v2.commerce.invariants import invariants
from usami_v2.commerce.models import unit_credit
from usami_v2.evaluation.outcomes import GoalResult
from usami_v2.evaluation.communication import evaluate_communication
from .schema import Attribute


REMEDY_REQUEST_PATTERNS = {
    "refund": (
        r"\b(?:please\s+)?refund\s+(?:my|this|the|me|it)\b",
        r"\b(?:i\s+(?:want|need|request|would\s+like)|i['’]d\s+like)\s+"
        r"(?:a\s+)?(?:full\s+)?(?:refund|reimbursement)\b",
        r"\b[a-z][a-z -]{1,64}\s+(?:needs?|requires?)\s+"
        r"(?:a\s+)?(?:full\s+)?(?:refund|reimbursement)\b",
        r"\b(?:give|send)\s+me\s+(?:a\s+)?refund\b",
        r"\b(?:i\s+(?:want|need))\s+my\s+money\s+back\b",
        r"(?:给我退款|请.{0,8}退款|我(?:要|想|需要|要求|希望).{0,4}退款|退钱给我)",
    ),
    "exchange": (
        r"\b(?:please\s+)?(?:exchange|replace|swap)\s+(?:my|this|the|it)\b",
        r"\b(?:i\s+(?:want|need|request|would\s+like))\s+(?:an?\s+)?(?:exchange|replacement)\b",
        r"\b(?:i\s+(?:want|need|request|would\s+like))\s+(?:my|this|the)\b.{1,48}\b(?:replaced|replacement)\b",
        r"(?:给我换货|请.{0,8}(?:换货|更换)|我(?:要|想|需要|要求|希望).{0,4}(?:换货|更换))",
    ),
    "reship": (
        r"\b(?:i\s+(?:want|need|request|would\s+like))\s+(?:an?\s+)?replacement\b",
        r"\b(?:please\s+)?(?:reship|send|ship)\s+(?:it|this|the|my|another|a\s+replacement)\b",
        r"\b(?:please\s+)?(?:send|ship)\s+me\s+(?:the\s+)?(?:correct|right|replacement)\b",
        r"\b(?:please\s+)?replace\s+(?:my|this|the|it)\b",
        r"\b(?:i\s+(?:want|need|request|would\s+like))\s+(?:it\s+)?(?:resent|reshipped|sent\s+again)\b",
        r"\b(?:i\s+(?:want|need|request|would\s+like))\s+(?:my|this|the)\b.{1,48}\b(?:replaced|replacement)\b",
        r"(?:给我补发|请.{0,8}(?:补发|重寄)|我(?:要|想|需要|要求|希望).{0,4}(?:补发|重寄))",
    ),
}

# Public language often names the thing a customer bought rather than an
# internal catalog family. These aliases are deliberately small and
# deterministic; SKU attribute values (for example brand, model, or color)
# provide additional world-grounded descriptions.
_FAMILY_PUBLIC_TERMS = {
    "device": ("gadget", "electronic device", "computer", "laptop", "notebook", "phone",
               "smartphone", "tablet", "headphones", "headset", "earbuds", "mouse",
               "keyboard", "monitor", "camera"),
    "electronics": ("electronic device", "computer", "laptop", "notebook", "phone",
                    "smartphone", "tablet", "headphones", "headset", "earbuds", "mouse",
                    "keyboard", "monitor", "camera"),
    "consumer electronics": ("electronic device", "computer", "laptop", "notebook", "phone",
                             "smartphone", "tablet", "headphones", "headset", "earbuds",
                             "mouse", "keyboard", "monitor", "camera"),
    "audio": ("headphones", "headset", "earbuds", "earphones", "speaker"),
    "cable": ("cord", "charging cable", "usb cable"),
}

_ISSUE_PUBLIC_TERMS = {
    "damaged": ("damaged", "damage", "broken", "defective", "faulty", "cracked", "not working"),
    "wrong_item": ("wrong item", "incorrect item", "wrong product", "different item",
                   "not what i ordered", "got something else", "received something else",
                   "ordered black but got", "ordered white but got", "ordered blue but got"),
    "missing_parts": ("missing part", "missing parts", "missing component", "missing components",
                      "part is missing", "parts are missing", "component is missing"),
}

_INFORMATION_FACT_TERMS = {
    "refund_amount": ("refund amount", "amount refunded", "how much refund", "how much was refunded",
                      "money back", "退款金额", "退了多少钱", "退多少钱"),
    "extra_charge": ("extra charge", "additional charge", "extra cost", "additional cost",
                     "how much more", "pay extra", "补差", "额外费用", "还要付多少"),
    "arrival_date": ("arrival date", "delivery date", "when will it arrive", "when will it be delivered",
                     "什么时候送到", "何时送达", "预计送达"),
    "refund_status": ("refund status", "status of my refund", "was my refund", "has my refund",
                      "退款状态", "退款进度", "退了吗"),
    "shipment_status": ("shipment status", "shipping status", "tracking status", "where is my package",
                        "物流状态", "配送状态", "包裹到哪"),
    "quantity": ("how many", "quantity", "number of items", "how many items", "多少件", "数量"),
    "currency": ("currency", "which currency", "what currency", "币种", "什么货币"),
}

_SMALL_NUMBERS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                  "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _normalized_public_text(value):
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", str(value).casefold()).strip()


def _contains_public_phrase(text, phrase):
    text, phrase = _normalized_public_text(text), _normalized_public_text(phrase)
    if not phrase:
        return False
    if re.search(r"[\u4e00-\u9fff]", phrase):
        return phrase in text
    return " " + phrase + " " in " " + text + " "


def _issue_is_named(text, issue, product_terms=()):
    terms = _ISSUE_PUBLIC_TERMS.get(issue, (issue,))
    if issue == "wrong_item":
        terms += tuple("wrong " + term for term in product_terms if term)
    return any(_contains_public_phrase(text, term) for term in (issue,) + terms)


def _number_value(value):
    value = value.casefold()
    return int(value) if value.isdigit() else _SMALL_NUMBERS.get(value)


def _public_day_limits(text):
    pattern = re.compile(
        r"\b(?:within|in|no later than|by)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+days?\b",
        re.I)
    return tuple(_number_value(match.group(1)) for match in pattern.finditer(text))


def _public_charge_limit(text):
    normalized = _normalized_public_text(text)
    if any(phrase in " " + normalized + " " for phrase in (
            " no extra charge ", " no additional charge ", " without extra charge ",
            " at no extra cost ", " at no additional cost ", " free of charge ")):
        return 0
    amounts = re.findall(
        r"(?:no more than|up to|at most|within)\s*(?:\$\s*)?"
        r"(\d+(?:\.\d{1,2})?)\s*(?:usd|dollars?)?\s*(?:extra|additional|more|extra charge|additional cost)\b",
        text, flags=re.I)
    amounts.extend(re.findall(
        r"maximum\s+(?:extra\s+charge|extra)\s+of\s*(?:\$\s*)?"
        r"(\d+(?:\.\d{1,2})?)\s*(?:usd|dollars?)?\b",
        text, flags=re.I))
    amounts.extend(re.findall(
        r"(?:extra|additional)\s+(?:charge|cost)\s+(?:of|up to|no more than|at most)\s*"
        r"(?:\$\s*)?(\d+(?:\.\d{1,2})?)\s*(?:usd|dollars?)?\b",
        text, flags=re.I))
    values = []
    for amount in amounts:
        try:
            values.append(round(float(amount) * 100))
        except ValueError:
            continue
    return tuple(values)


def _public_quantity(text):
    pattern = re.compile(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
                         r"(?:items?|units?|pieces?|pairs?|headphones?|earbuds?|earphones?|headsets?|"
                         r"devices?|laptops?|computers?|tablets?|phones?|mice|mouse|keyboards?|"
                         r"cables?|chargers?)\b", re.I)
    for match in pattern.finditer(text):
        return _number_value(match.group(1))
    return None


def _validate_public_remedy_bounds(goal, initial_clock):
    """Reject score conditions that cannot be traced to the quoted request."""
    text = goal.public_request_span
    if goal.quantity > 1:
        quantity = _public_quantity(text)
        if quantity != goal.quantity:
            raise ValueError("remedy quantity must be stated in the public request")
    elif (quantity := _public_quantity(text)) not in (None, goal.quantity):
        raise ValueError("remedy quantity differs from the public request")

    if goal.required_attributes:
        for attribute in goal.required_attributes:
            if not _contains_public_phrase(text, attribute.value):
                raise ValueError("required product attributes must appear in the public request")

    day_limits = _public_day_limits(text)
    if day_limits and goal.latest_arrival is None:
        raise ValueError("a public arrival deadline must be included in the remedy goal")
    if goal.latest_arrival is not None:
        if not day_limits:
            raise ValueError("an arrival deadline may only be scored when the public request states it in days")
        seconds = (goal.latest_arrival - initial_clock).total_seconds()
        if seconds < 0 or seconds % 86400 or int(seconds // 86400) not in day_limits:
            raise ValueError("arrival deadline differs from the public request")

    charge_limits = _public_charge_limit(text)
    if charge_limits and goal.maximum_extra_charge_cents is None:
        raise ValueError("a public extra-charge limit must be included in the remedy goal")
    if goal.maximum_extra_charge_cents is not None:
        if not charge_limits:
            raise ValueError("an extra-charge cap may only be scored when the public request states it")
        if goal.maximum_extra_charge_cents not in charge_limits:
            raise ValueError("extra-charge cap differs from the public request")


def _positive_remedy_request(text, remedy):
    negative_prefix = re.compile(
        r"(?:\b(?:no|not|never|without|don't|do\s+not|didn't|cannot|can't|won't|wouldn't|shouldn't)\b)"
        r"(?:\s+[a-z]+){0,4}\s*$", re.I)
    negative_suffix = re.compile(r"^\s*(?:not|isn't|is\s+not|not\s+wanted|not\s+needed|not\s+accepted)\b", re.I)
    negative_cjk_prefix = re.compile(r"(?:不接受|不同意|不要|不想|不需要|无需|拒绝|不愿|不希望|别|禁止|请勿).{0,10}$")
    negative_cjk_suffix = re.compile(r"^.{0,8}(?:不接受|不同意|不要|不想|不需要|无需|拒绝|别|禁止|请勿)")
    negative_inside = re.compile(
        r"\b(?:no|not|never|don't|do\s+not|doesn't|does\s+not|didn't|did\s+not|"
        r"cannot|can't|will\s+not|won't|wouldn't|would\s+not|shouldn't|should\s+not)\b", re.I)
    text = " ".join(text.casefold().split())
    for pattern in REMEDY_REQUEST_PATTERNS[remedy]:
        for match in re.finditer(pattern, text, re.I):
            prefix = text[max(0, match.start() - 48):match.start()]
            suffix = text[match.end():match.end() + 18]
            if (negative_prefix.search(prefix) or negative_suffix.search(suffix)
                    or negative_inside.search(match.group(0))):
                continue
            if (negative_cjk_prefix.search(prefix) or negative_cjk_suffix.search(suffix)
                    or negative_cjk_prefix.search(match.group(0))
                    or negative_cjk_suffix.search(match.group(0))):
                continue
            return True
    return False


def request_names_remedy(text, remedies):
    """Check public customer intent against engine methods.

    A public replace/replacement request may be realized as an exchange or
    reship. It never implies a refund; other methods require explicit intent.
    """
    return bool(remedies) and all(_positive_remedy_request(text, remedy) for remedy in remedies)


def request_names_fact(text, fact_kind):
    """Conservatively bind an information objective to an explicit public question."""
    terms = _INFORMATION_FACT_TERMS.get(fact_kind)
    return bool(terms) and any(_contains_public_phrase(text, term) for term in terms)


def public_line_match_ids(world, customer_id, mention, request_context=None, *, require_issue=True):
    """Resolve natural product words to owned frozen lines, rejecting ambiguity.

    ``mention`` is the product phrase visible in the request. ``request_context``
    is its exact public quote and carries the natural-language issue description.
    Neither needs to contain internal SKU-family or issue codes.
    """
    request_context = mention if request_context is None else request_context
    owned_orders = {row.id for row in world.orders.values() if row.customer_id == customer_id}
    candidates = []
    for line in world.lines.values():
        if line.order_id not in owned_orders:
            continue
        sku = world.skus.get(line.sku_id)
        issues = {row.issue for row in world.evidence.values() if row.line_id == line.id}
        family_terms = (sku.product_family, *_FAMILY_PUBLIC_TERMS.get(
            _normalized_public_text(sku.product_family), ())) if sku is not None else ()
        attribute_terms = tuple(value for value in (sku.attributes.values() if sku else ()) if value)
        product_named = any(_contains_public_phrase(mention, term)
                            for term in family_terms + attribute_terms)
        issue_named = (not require_issue or any(
            _issue_is_named(request_context, issue, family_terms + attribute_terms) for issue in issues))
        if sku is not None and product_named and issue_named:
            candidates.append((line, sku))

    # A phrase such as "the damaged device" may match multiple lines. Publicly
    # named SKU attributes can disambiguate them without exposing internal IDs.
    attribute_names = {name for _, sku in candidates for name in sku.attributes}
    for name in sorted(attribute_names):
        values = {sku.attributes.get(name) for _, sku in candidates if sku.attributes.get(name)}
        named_values = {value for value in values if _contains_public_phrase(mention, value)}
        if named_values:
            candidates = [(line, sku) for line, sku in candidates
                          if sku.attributes.get(name) in named_values]
    return tuple(sorted(line.id for line, _ in candidates))


class RemedyGoal(Sealed):
    goal_id: str
    line_id: str
    quantity: int = Field(ge=1, strict=True)
    allowed_remedies: Tuple[Literal["refund", "exchange", "reship"], ...] = Field(min_length=1)
    required_attributes: Tuple[Attribute, ...] = ()
    maximum_extra_charge_cents: Optional[int] = Field(default=None, ge=0, strict=True)
    latest_arrival: Optional[datetime] = None
    target_stage: Literal["settled", "delivered"] = "settled"

    @model_validator(mode="after")
    def coherent_goal(self):
        if len(set(self.allowed_remedies)) != len(self.allowed_remedies):
            raise ValueError("duplicate acceptable remedies")
        if self.latest_arrival is not None:
            aware(self.latest_arrival)
        if self.target_stage == "delivered" and "refund" in self.allowed_remedies:
            raise ValueError("a refund cannot satisfy a delivered-replacement goal")
        if len({row.name for row in self.required_attributes}) != len(self.required_attributes):
            raise ValueError("duplicate required attribute")
        return self


class GroundedRemedyGoal(RemedyGoal):
    """Legacy v2 business goal linked to an exact public request span."""

    public_request_span: str = Field(min_length=1)
    subject_mention: str = Field(min_length=1)

    @field_validator("public_request_span")
    @classmethod
    def nonblank_request_span(cls, value):
        if not " ".join(value.casefold().split()):
            raise ValueError("remedy goal request span cannot be blank")
        return value


class InformationObjective(Sealed):
    goal_id: str
    subject_id: str
    subject_mention: str = Field(min_length=1)
    fact_kind: FactKind
    public_request: str = Field(min_length=1)

    @model_validator(mode="after")
    def mention_is_public(self):
        normalized = " ".join(self.public_request.casefold().split())
        mention = " ".join(self.subject_mention.casefold().split())
        if mention not in normalized:
            raise ValueError("information objective subject mention is absent from the public request")
        return self


class TaskSpec(Sealed):
    protocol: Literal["commerce-task-v1"] = "commerce-task-v1"
    task_id: str
    world_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    customer_id: str
    customer_request: str = Field(min_length=1)
    remedy_goals: Tuple[RemedyGoal, ...] = ()
    information_goals: Tuple[InformationObjective, ...] = ()
    partition: Literal["train", "validation", "heldout"]
    source_distribution_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_training_evidence_hashes: Tuple[str, ...] = ()

    @field_validator("parent_training_evidence_hashes")
    @classmethod
    def valid_parent_hashes(cls, values):
        if any(len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)
               for value in values):
            raise ValueError("training evidence references must be SHA-256 hashes")
        return values

    @model_validator(mode="after")
    def actual_goals(self):
        rows = self.remedy_goals + self.information_goals
        if not rows or len({row.goal_id for row in rows}) != len(rows):
            raise ValueError("task needs unique business/information objectives")
        normalized_request = " ".join(self.customer_request.casefold().split())
        if self.protocol == "commerce-task-v1":
            for goal in self.information_goals:
                span = " ".join(goal.public_request.casefold().split())
                if span not in normalized_request:
                    raise ValueError("information objective must be grounded in the public Customer request")
        if len({row.line_id for row in self.remedy_goals}) != len(self.remedy_goals):
            raise ValueError("combine acceptable remedies for one line rather than ambiguous duplicated goals")
        if self.partition == "heldout" and self.parent_training_evidence_hashes:
            raise ValueError("final evaluation tasks cannot be conditioned on training weakness evidence")
        return self


class TaskSpecV2(TaskSpec):
    """Versioned task with public-source binding for every remedy objective.

    The v1/v2 TaskSpec records remain readable for historical certificates and
    Episodes. New Designer output uses v3 structured request provenance.
    """

    protocol: Literal["commerce-task-v2"] = "commerce-task-v2"
    remedy_goals: Tuple[GroundedRemedyGoal, ...] = ()

    @model_validator(mode="after")
    def remedy_goals_are_public(self):
        normalized_request = " ".join(self.customer_request.casefold().split())
        for goal in self.remedy_goals:
            span = " ".join(goal.public_request_span.casefold().split())
            mention = " ".join(goal.subject_mention.casefold().split())
            if (not span or span not in normalized_request or not mention or mention not in span
                    or not request_names_remedy(span, goal.allowed_remedies)):
                raise ValueError(
                    "remedy objective must quote the public request and support every accepted remedy; "
                    "replacement wording supports exchange or reship but does not imply a refund"
                )
        for goal in self.information_goals:
            if not request_names_fact(goal.public_request, goal.fact_kind):
                raise ValueError("information fact_kind must match the explicitly quoted public question")
        return self


class StructuredInformationObjective(Sealed):
    """Machine-bound information intent; public wording is rendered by the Host."""

    goal_id: str
    subject_id: str
    fact_kind: FactKind


class CustomerRequestBundle(Sealed):
    """Provenance binding between structured Customer intent and rendered text."""

    protocol: Literal["commerce-request-bundle-v1"] = "commerce-request-bundle-v1"
    customer_id: str
    goal_ids: Tuple[str, ...]
    renderer_version: Literal["commerce-request-renderer-v1"] = "commerce-request-renderer-v1"
    intent_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def unique_goals(self):
        if not self.goal_ids or len(set(self.goal_ids)) != len(self.goal_ids):
            raise ValueError("request bundle requires unique structured goal references")
        return self


def _display_name(value):
    return " ".join(str(value).replace("_", " ").split()) or "item"


def _line_description(world, line_id):
    line = world.lines[line_id]
    sku = world.skus[line.sku_id]
    issues = sorted({row.issue for row in world.evidence.values() if row.line_id == line_id})
    issue_text = {"damaged": "damaged", "wrong_item": "incorrect", "missing_parts": "missing parts"}
    parts = [issue_text.get(issue, _display_name(issue)) for issue in issues]
    parts.extend(_display_name(sku.attributes[name]) for name in sorted(sku.attributes)
                 if name != "variant" and sku.attributes[name])
    parts.append(_display_name(sku.product_family))
    return " ".join(dict.fromkeys(parts))


def render_customer_request(world, customer_id, remedy_goals, information_goals):
    """Render a public request from the exact structured goals; never infer goals from prose."""
    remedy_goals, information_goals = tuple(remedy_goals), tuple(information_goals)
    clauses = []
    for index, goal in enumerate(remedy_goals, 1):
        if goal.line_id not in world.lines:
            raise ValueError("request renderer received a line outside the frozen world")
        line = world.lines[goal.line_id]
        order = world.orders[line.order_id]
        if order.customer_id != customer_id:
            raise ValueError("request renderer cannot describe another Customer's line")
        remedies = set(goal.allowed_remedies)
        labels = []
        if "refund" in remedies:
            labels.append("a refund")
        if remedies & {"exchange", "reship"}:
            labels.append("a replacement")
        if len(labels) == 1:
            action = labels[0]
        else:
            action = " or ".join(labels)
        subject = _line_description(world, goal.line_id)
        quantity = "" if goal.quantity == 1 else " for {} items".format(goal.quantity)
        sentence = "{}. Please arrange {}{} for the {} item in my order".format(
            index, action, quantity, subject)
        if goal.required_attributes:
            terms = ", ".join("{} {}".format(_display_name(row.value), _display_name(row.name))
                               for row in goal.required_attributes)
            sentence += " with " + terms
        if goal.latest_arrival is not None:
            days = (goal.latest_arrival - world.clock).total_seconds() / 86400
            if days < 0 or days != int(days):
                raise ValueError("structured arrival deadline must be a nonnegative whole number of days")
            sentence += " within {} days".format(int(days))
        if goal.maximum_extra_charge_cents is not None:
            amount = goal.maximum_extra_charge_cents / 100
            sentence += " with no more than CNY {:.2f} in extra charges".format(amount)
        clauses.append(sentence + ".")
    fact_labels = {
        "refund_amount": "refund amount", "extra_charge": "additional charge",
        "arrival_date": "arrival date", "refund_status": "refund status",
        "shipment_status": "shipment status", "quantity": "quantity", "currency": "currency",
    }
    for index, goal in enumerate(information_goals, len(clauses) + 1):
        if goal.subject_id in world.lines:
            subject = _line_description(world, goal.subject_id)
        elif goal.subject_id in world.orders:
            subject = "order"
        else:
            raise ValueError("request renderer received an information subject outside the frozen world")
        if goal.subject_id in world.lines:
            line = world.lines[goal.subject_id]
            if world.orders[line.order_id].customer_id != customer_id:
                raise ValueError("request renderer cannot describe another Customer's line")
        elif world.orders[goal.subject_id].customer_id != customer_id:
            raise ValueError("request renderer cannot describe another Customer's order")
        clauses.append("{}. Please tell me the {} for my {}.".format(
            index, fact_labels[goal.fact_kind], subject))
    if not clauses:
        raise ValueError("cannot render an empty Customer request")
    return "Please help me with the following request(s):\n" + "\n".join(clauses)


def request_intent_hash(customer_id, remedy_goals, information_goals):
    return digest({"customer_id": customer_id,
                   "remedy_goals": [row.model_dump(mode="json", exclude={"record_hash"})
                                    for row in remedy_goals],
                   "information_goals": [row.model_dump(mode="json", exclude={"record_hash"})
                                         for row in information_goals]})


class TaskSpecV3(TaskSpec):
    """Structured-intent task. Customer-facing text is rendered from its goals."""

    protocol: Literal["commerce-task-v3"] = "commerce-task-v3"
    remedy_goals: Tuple[RemedyGoal, ...] = ()
    information_goals: Tuple[StructuredInformationObjective, ...] = ()
    request_bundle: CustomerRequestBundle

    @model_validator(mode="after")
    def request_bundle_matches_intent(self):
        bundle = self.request_bundle
        goals = self.remedy_goals + self.information_goals
        if (bundle.customer_id != self.customer_id
                or bundle.goal_ids != tuple(row.goal_id for row in goals)
                or bundle.intent_hash != request_intent_hash(self.customer_id, self.remedy_goals,
                                                             self.information_goals)
                or bundle.request_hash != digest(self.customer_request)):
            raise ValueError("Customer request provenance differs from structured task intent")
        return self


def parse_task_spec(value):
    """Parse a saved task by its immutable protocol version."""
    protocol = value.get("protocol") if isinstance(value, dict) else None
    models = {
        "commerce-task-v1": TaskSpec,
        "commerce-task-v2": TaskSpecV2,
        "commerce-task-v3": TaskSpecV3,
    }
    model = models.get(protocol)
    if model is None:
        raise ValueError("unsupported commerce task protocol: {}".format(protocol))
    return model.model_validate(value)


def check_task_scope(world, task: TaskSpec):
    task = task.checked()
    if world.state_hash != task.world_hash or invariants(world):
        raise ValueError("task world binding or native invariants differ")
    if task.customer_id not in world.customers:
        raise ValueError("task Customer does not exist")

    owned_orders = {row.id for row in world.orders.values() if row.customer_id == task.customer_id}
    owned_lines = {row.id for row in world.lines.values() if row.order_id in owned_orders}

    if task.protocol == "commerce-task-v3":
        if not isinstance(task, TaskSpecV3):
            raise ValueError("commerce-task-v3 requires its structured task contract")
        expected_request = render_customer_request(
            world, task.customer_id, task.remedy_goals, task.information_goals)
        if task.customer_request != expected_request:
            raise ValueError("public Customer request was not rendered from its registered structured intent")
        if task.request_bundle.request_hash != digest(expected_request):
            raise ValueError("structured request bundle does not bind the rendered public request")

    for goal in task.remedy_goals:
        line = world.lines.get(goal.line_id)
        order = world.orders.get(line.order_id) if line else None
        if not order or order.customer_id != task.customer_id:
            raise ValueError("task crosses the frozen Customer identity")
        if task.protocol == "commerce-task-v2" and isinstance(goal, GroundedRemedyGoal):
            if public_line_match_ids(world, task.customer_id, goal.subject_mention,
                                     goal.public_request_span) != (line.id,):
                raise ValueError("public product mention and issue description must uniquely identify one owned frozen line")
        occupied = {unit for row in world.cases.values()
                    if row.line_id == goal.line_id and row.status not in ("resolved", "cancelled")
                    for unit in row.unit_indices}
        settled = {unit for row in world.settlements.values() if row.line_id == goal.line_id
                   for unit in row.unit_indices}
        if goal.quantity > line.quantity - len(occupied | settled):
            raise ValueError("task exceeds the uncommitted purchased quantity")

    for goal in task.information_goals:
        if goal.subject_id not in owned_orders | owned_lines:
            raise ValueError("information objective is not about an owned order/line")
        if task.protocol == "commerce-task-v2":
            if goal.subject_id in owned_lines:
                if public_line_match_ids(world, task.customer_id, goal.subject_mention,
                                         goal.public_request, require_issue=False) != (goal.subject_id,):
                    raise ValueError("public information mention must uniquely identify one owned frozen line")
            elif goal.subject_id in owned_orders and len(owned_orders) != 1:
                raise ValueError("order-level information target is ambiguous for this Customer")
        elif task.protocol == "commerce-task-v3" and goal.subject_id in owned_lines:
            if goal.subject_id not in owned_lines:
                raise ValueError("structured information target is outside the authenticated Customer scope")
    return task


def _validate_run_states(initial, final, task):
    task = check_task_scope(initial, task)
    if (final.seed != initial.seed or final.version != initial.version
            or final.domain != initial.domain):
        raise ValueError("final state belongs to a different business instance")
    if invariants(final):
        raise ValueError("final state violates native business invariants")
    immutable = ("customers", "merchants", "skus", "warehouses", "orders", "lines", "evidence")
    if any(getattr(initial, name) != getattr(final, name) for name in immutable):
        raise ValueError("final state changed frozen source business facts")
    if final.clock < initial.clock:
        raise ValueError("final business clock moved backwards")
    return task


def evaluate_state_goals(initial, final, task: TaskSpec):
    """Return only registered business-state objectives.

    Communication and out-of-scope settlement checks are separate so a
    read-only task can correctly have no state goal while still succeeding.
    """
    task = _validate_run_states(initial, final, task)
    rows = []
    for goal in task.remedy_goals:
        settlements = [row for key, row in final.settlements.items()
                       if row.line_id == goal.line_id and key not in initial.settlements]
        quantity = sum(len(row.unit_indices) for row in settlements)
        ok = quantity == goal.quantity and all(row.remedy in goal.allowed_remedies for row in settlements)
        requested_units = set()
        for settlement in settlements:
            if requested_units.intersection(settlement.unit_indices):
                ok = False
            requested_units.update(settlement.unit_indices)
            line = initial.lines[goal.line_id]
            previously_used = {unit for row in initial.settlements.values() if row.line_id == goal.line_id
                               for unit in row.unit_indices}
            previously_used.update(unit for row in initial.cases.values()
                                   if row.line_id == goal.line_id and row.status not in ("resolved", "cancelled")
                                   for unit in row.unit_indices)
            if any(unit in previously_used for unit in settlement.unit_indices):
                ok = False
            if any(type(unit) is not int or unit < 0 or unit >= line.quantity
                   for unit in settlement.unit_indices):
                ok = False
        refs = []
        extra = 0
        for settlement in settlements:
            refs.append(digest(settlement))
            quote = final.quotes.get(settlement.quote_id)
            if quote is None:
                ok = False
                continue
            ledger = [entry for entry in final.ledger.values() if entry.quote_id == quote.id]
            extra += sum(entry.amount_cents for entry in ledger if entry.kind == "extra_charge")
            if settlement.remedy == "refund":
                paid = unit_credit(initial.lines[goal.line_id], settlement.unit_indices)
                if sum(entry.amount_cents for entry in ledger if entry.kind == "refund") != paid:
                    ok = False
                if goal.required_attributes or goal.latest_arrival is not None:
                    ok = False
                order = initial.orders[initial.lines[goal.line_id].order_id]
                if any(entry.channel != order.payment_channel for entry in ledger):
                    ok = False
            else:
                deliveries = [row for row in final.fulfillments.values() if row.quote_id == quote.id]
                if len(deliveries) != 1:
                    ok = False
                    continue
                delivery = deliveries[0]
                sku = final.skus.get(delivery.sku_id)
                if not sku or any(sku.attributes.get(row.name) != row.value for row in goal.required_attributes):
                    ok = False
                if goal.target_stage == "delivered" and delivery.actual_arrival is None:
                    ok = False
                arrival = delivery.actual_arrival or delivery.promised_arrival
                if goal.latest_arrival is not None and arrival > goal.latest_arrival:
                    ok = False
                refs.append(digest(delivery))
        if goal.maximum_extra_charge_cents is not None and extra > goal.maximum_extra_charge_cents:
            ok = False
        rows.append(GoalResult(goal_id=goal.goal_id, status="PASS" if ok else "FAIL", evidence_refs=tuple(refs)))

    return tuple(rows)


def evaluate_task_compliance(initial, final, task: TaskSpec):
    """Report unrequested settlements as a compliance result, not goal progress."""
    task = _validate_run_states(initial, final, task)
    allowed_lines = {row.line_id for row in task.remedy_goals}
    unexpected = [row for key, row in final.settlements.items()
                  if key not in initial.settlements and row.line_id not in allowed_lines]
    return (GoalResult(goal_id="unrequested_settlements", status="FAIL" if unexpected else "PASS",
                       evidence_refs=tuple(digest(row) for row in unexpected)),)


def evaluate_task_communication(initial, task: TaskSpec, *, facts=(), messages=(), end_event_index=None):
    """Evaluate requests grounded in the frozen Customer text against delivered claims."""
    task = check_task_scope(initial, task)
    if task.information_goals:
        messages = tuple(messages)
        if end_event_index is None:
            end_event_index = max((row.event_index for row in messages), default=0)
        obligations = []
        for goal in task.information_goals:
            if task.protocol == "commerce-task-v3":
                public_text = task.customer_request
                request_hash = task.request_bundle.record_hash
            else:
                public_text = goal.public_request
                request_hash = digest({"customer_request": task.customer_request,
                                       "request_span": goal.public_request})
            source = SourceRef(
                source_id="customer-request-" + goal.goal_id,
                source_kind="customer_message",
                source_hash=request_hash,
                event_index=0)
            request = PublicRequest(
                request_id=goal.goal_id, subject_id=goal.subject_id, fact_kind=goal.fact_kind,
                source=source, public_text=public_text)
            obligations.append(CommunicationObligation(
                obligation_id="information-" + goal.goal_id, request=request,
                due_event_index=end_event_index, required=True))
        return evaluate_communication(facts, messages, obligations,
                                      end_event_index=end_event_index, terminal=True)
    messages = tuple(messages)
    if end_event_index is None:
        end_event_index = max((row.event_index for row in messages), default=0)
    return evaluate_communication(facts, messages, (), end_event_index=end_event_index, terminal=True)
