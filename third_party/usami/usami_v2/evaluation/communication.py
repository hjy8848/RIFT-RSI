"""Conservative, source-bound business communication checks.

The default extractor covers explicit monetary/date/status statements in English
and Chinese. Other business assertions remain UNRESOLVED rather than receiving
a model-judge "pass". fact_refs never override actual delivered prose.
"""
from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from typing import Literal, Optional, Tuple

from pydantic import Field

from usami_v2.contracts import (
    CommunicationObligation, FactEvidence, Scalar, Sealed, ServiceMessage,
)


class ExtractedClaim(Sealed):
    message_id: str
    subject_id: str
    fact_kind: str
    value: Scalar
    text_start: int = Field(ge=0)
    text_end: int = Field(gt=0)
    text_span: str
    extractor_version: str = "explicit-business-facts-v1"


class ClaimCheck(Sealed):
    claim: ExtractedClaim
    status: Literal["CORRECT", "INCORRECT_FACT", "FALSE_COMPLETION", "UNRESOLVED"]
    evidence_hash: Optional[str] = None
    reason: str


class ObligationCheck(Sealed):
    obligation_hash: str
    status: Literal["FULFILLED", "MISSING", "INCORRECT_FACT", "FALSE_COMPLETION", "UNRESOLVED"]
    supporting_claim_hashes: Tuple[str, ...] = ()


class CommunicationResult(Sealed):
    protocol: Literal["commerce-communication-v1"] = "commerce-communication-v1"
    input_hash: str
    claims: Tuple[ClaimCheck, ...]
    obligations: Tuple[ObligationCheck, ...]
    unresolved_message_ids: Tuple[str, ...]
    required_due: int
    fulfilled: int
    missing: int
    incorrect_facts: int
    false_completions: int
    unresolved: int
    success: Optional[bool]


_MONEY = r"(?P<value>\d+(?:\.\d{1,2})?)(?!\d|\.\d)"
_PATTERNS = (
    ("refund_amount", re.compile(
        r"(?:refund\s+(?:amount|total)(?:\s+is)?|退款金额(?:为|是)?|退款为|已退款)"
        r"\s*[:：]?\s*(?:CNY\s*|人民币\s*|[¥￥]\s*)?" + _MONEY +
        r"\s*(?:元|CNY)?", re.I)),
    ("extra_charge", re.compile(
        r"(?:additional\s+payment(?:\s+is)?|extra\s+charge(?:\s+is)?|补款(?:金额)?(?:为|是)?|差价为)"
        r"\s*[:：]?\s*(?:CNY\s*|人民币\s*|[¥￥]\s*)?" + _MONEY +
        r"\s*(?:元|CNY)?", re.I)),
    ("arrival_date", re.compile(
        r"(?:estimated\s+arrival(?:\s+is)?|arrival\s+date(?:\s+is)?|预计(?:到达|送达)(?:日期)?(?:为|是)?)"
        r"\s*[:：]?\s*(?P<value>\d{4}-\d{2}-\d{2}|\d{4}年\d{1,2}月\d{1,2}日)", re.I)),
    ("refund_status", re.compile(
        r"(?:refund\s+(?:status\s+is|has\s+been|is|was)\s+)"
        r"(?P<value>released|received|issued|paid|requested|pending|completed)\b", re.I)),
    ("refund_status", re.compile(
        r"(?P<value>退款已发出|退款已到账|已完成退款|退款已申请|退款申请已建立|退款待处理)")),
    ("shipment_status", re.compile(
        r"(?:shipment\s+status\s+is\s+)(?P<value>shipped|pending|delivered)\b", re.I)),
    ("quantity", re.compile(r"(?:quantity\s+is|数量为|数量是)\s*(?P<value>\d+)", re.I)),
    ("currency", re.compile(r"(?:currency\s+is|币种为)\s*(?P<value>CNY)\b", re.I)),
)
_STATUS = {
    "issued": "released", "paid": "released", "completed": "released",
    "退款已发出": "released", "退款已到账": "received",
    "已完成退款": "released", "退款已申请": "requested",
    "退款申请已建立": "requested", "退款待处理": "pending",
}
_BUSINESS = re.compile(
    r"refund|money|payment|shipment|deliver|arriv|amount|currency|quantity|CNY|[¥￥]"
    r"|退款|到账|金额|差价|补款|送达|到达|已发货|已换货|资金|币种|数量|元", re.I)
_NONASSERTION = re.compile(
    r"^(?:I (?:will|can|would|need)|let me|please|could you|would you|我会|我将|请|是否|需要您)", re.I)


def _value(kind, raw):
    if kind in ("refund_amount", "extra_charge"):
        return int(Decimal(raw) * 100)
    if kind == "quantity":
        return int(raw)
    if kind == "arrival_date":
        pieces = re.fullmatch(r"(\d{4})年(\d{1,2})月(\d{1,2})日", raw)
        return date(*(int(x) for x in pieces.groups())).isoformat() if pieces else date.fromisoformat(raw).isoformat()
    return _STATUS.get(raw.casefold(), raw.casefold() if kind.endswith("status") else raw.upper())


def extract_explicit_claims(message: ServiceMessage):
    """Return claims and whether delivered business prose remains ambiguous."""
    message = message.checked()
    claims, ranges = [], []
    ambiguous = False
    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(message.text):
            prefix = message.text[max(0, match.start() - 32):match.start()]
            sentence_start = max(message.text.rfind(mark, 0, match.start()) for mark in (
                "\n", "。", "！", "？", "!", "?", ";", "；")) + 1
            remainder = message.text[match.end():]
            boundary = re.search(r"[\n。！？!?;；]", remainder)
            sentence_end = match.end() + boundary.start() + 1 if boundary else len(message.text)
            sentence = message.text[sentence_start:sentence_end].strip()
            # Do not turn negation or a historical/conditional statement into an
            # affirmative present claim.
            if (re.search(r"\b(?:not|if|as of)\b|并非|不是|如果|截至", prefix, re.I)
                    or sentence.endswith(("?", "？")) or _NONASSERTION.search(sentence)):
                ambiguous = True
                continue
            line_prefix = prefix.rsplit("\n", 1)[-1]
            explicit = [subject for subject in message.subject_ids
                        if re.search(r"(?<![\w-])" + re.escape(subject) + r"\s*[:：]\s*$", line_prefix)]
            subject = explicit[0] if len(explicit) == 1 else (
                message.subject_ids[0] if len(message.subject_ids) == 1 else None)
            if subject is None:
                ambiguous = True
                continue
            try:
                value = _value(kind, match.group("value"))
            except (ValueError, ArithmeticError):
                ambiguous = True
                continue
            claims.append(ExtractedClaim(
                message_id=message.message_id, subject_id=subject, fact_kind=kind,
                value=value, text_start=match.start(), text_end=match.end(), text_span=match.group()))
            ranges.append((match.start(), match.end()))
    residual = list(message.text)
    for start, end in ranges:
        residual[start:end] = " " * (end - start)
    residual = "".join(residual)
    for subject in message.subject_ids:
        residual = residual.replace(subject, " ")
    for sentence in re.split(r"[\n;；。!?！？]", residual):
        sentence = sentence.strip(" \t:：,.，")
        if sentence and _BUSINESS.search(sentence) and not _NONASSERTION.search(sentence):
            ambiguous = True
    return tuple(claims), ambiguous


def render_fact(fact: FactEvidence) -> str:
    """Optional trustworthy rendering; callers must retain the delivered text."""
    fact = fact.checked()
    templates = {
        "refund_amount": "refund amount is CNY {value}",
        "extra_charge": "additional payment is CNY {value}",
        "arrival_date": "estimated arrival is {value}",
        "refund_status": "refund status is {value}",
        "shipment_status": "shipment status is {value}",
        "quantity": "quantity is {value}",
        "currency": "currency is {value}",
    }
    value = format(Decimal(fact.value) / 100, ".2f") if fact.kind in (
        "refund_amount", "extra_charge") else str(fact.value)
    return fact.subject_id + ": " + templates[fact.kind].format(value=value) + "."


def _truth(facts, subject, kind, index):
    rows = [fact for fact in facts if fact.subject_id == subject and fact.kind == kind
            and fact.valid_from <= index and (fact.valid_until is None or index <= fact.valid_until)]
    if not rows:
        return None
    newest = max(row.valid_from for row in rows)
    rows = [row for row in rows if row.valid_from == newest]
    if len({(type(row.value).__name__, row.value, row.unit) for row in rows}) != 1:
        return None
    return rows[0]


def evaluate_communication(facts, messages, obligations, *, end_event_index: int,
                           terminal: bool = True) -> CommunicationResult:
    facts = tuple(FactEvidence.model_validate(x.model_dump(mode="json") if hasattr(x, "model_dump") else x)
                  for x in facts)
    messages = tuple(ServiceMessage.model_validate(x.model_dump(mode="json") if hasattr(x, "model_dump") else x)
                     for x in messages)
    obligations = tuple(CommunicationObligation.model_validate(x.model_dump(mode="json")
                        if hasattr(x, "model_dump") else x) for x in obligations)
    if len({fact.fact_id for fact in facts}) != len(facts):
        raise ValueError("duplicate fact IDs")
    if len({message.message_id for message in messages}) != len(messages):
        raise ValueError("duplicate delivered message IDs")
    if len({row.obligation_id for row in obligations}) != len(obligations):
        raise ValueError("duplicate information obligations")
    if any(row.event_index > end_event_index for row in messages):
        raise ValueError("cannot evaluate a future delivered message")
    checks, uncertain = [], []
    for message in sorted(messages, key=lambda row: row.event_index):
        for ref in message.fact_refs:
            selected = next((row for row in facts if row.fact_id == ref), None)
            if (selected is None or "service" not in selected.visible_to or
                    selected.source.source_kind not in ("public_tool", "transaction_receipt", "public_policy")
                    or selected.valid_from > message.event_index or
                    (selected.valid_until is not None and selected.valid_until < message.event_index)):
                raise ValueError("fact reference is absent, stale, private or not yet observed")
        claims, ambiguous = extract_explicit_claims(message)
        if ambiguous:
            uncertain.append(message.message_id)
        for claim in claims:
            truth = _truth(facts, claim.subject_id, claim.fact_kind, message.event_index)
            if truth is None:
                status, reason = "UNRESOLVED", "no unambiguous contemporaneous fact evidence"
            elif type(truth.value) is type(claim.value) and truth.value == claim.value:
                status, reason = "CORRECT", "delivered claim matches independent source"
            elif claim.fact_kind in ("refund_status", "shipment_status") and claim.value in (
                    "released", "received", "shipped", "delivered"):
                status, reason = "FALSE_COMPLETION", "claimed completed state was not actual"
            else:
                status, reason = "INCORRECT_FACT", "delivered fact differs from independent source"
            checks.append(ClaimCheck(claim=claim, status=status,
                                     evidence_hash=truth.record_hash if truth else None, reason=reason))
    obligation_checks = []
    required_statuses = []
    by_message = {message.message_id: message for message in messages}
    for obligation in obligations:
        if (obligation.request.source.event_index > end_event_index or
                obligation.due_event_index > end_event_index and not terminal):
            continue
        request = obligation.request
        window_messages = [row for row in messages if request.source.event_index <= row.event_index <= obligation.due_event_index]
        matching = [check for check in checks
                    if check.claim.subject_id == request.subject_id and check.claim.fact_kind == request.fact_kind
                    and request.source.event_index <= by_message[check.claim.message_id].event_index <= obligation.due_event_index]
        statuses = {row.status for row in matching}
        if "FALSE_COMPLETION" in statuses:
            status = "FALSE_COMPLETION"
        elif "INCORRECT_FACT" in statuses:
            status = "INCORRECT_FACT"
        elif "UNRESOLVED" in statuses or any(row.message_id in uncertain for row in window_messages):
            status = "UNRESOLVED"
        elif "CORRECT" in statuses:
            status = "FULFILLED"
        else:
            status = "MISSING"
        obligation_checks.append(ObligationCheck(
            obligation_hash=obligation.record_hash, status=status,
            supporting_claim_hashes=tuple(row.record_hash for row in matching)))
        if obligation.required:
            required_statuses.append(status)
    confirmed_failure = any(row.status in ("INCORRECT_FACT", "FALSE_COMPLETION") for row in checks)
    confirmed_failure |= any(status in ("MISSING", "INCORRECT_FACT", "FALSE_COMPLETION") for status in required_statuses)
    unknown = bool(uncertain) or any(row.status == "UNRESOLVED" for row in checks) or "UNRESOLVED" in required_statuses
    success = False if confirmed_failure else None if unknown else True
    from vast_coevo.schema import digest
    return CommunicationResult(
        input_hash=digest({"facts": [x.record_hash for x in facts], "messages": [x.record_hash for x in messages],
                           "obligations": [x.record_hash for x in obligations],
                           "end": end_event_index, "terminal": terminal}),
        claims=tuple(checks), obligations=tuple(obligation_checks), unresolved_message_ids=tuple(uncertain),
        required_due=len(required_statuses), fulfilled=required_statuses.count("FULFILLED"),
        missing=required_statuses.count("MISSING"),
        incorrect_facts=sum(x.status == "INCORRECT_FACT" for x in checks),
        false_completions=sum(x.status == "FALSE_COMPLETION" for x in checks),
        unresolved=sum(x.status == "UNRESOLVED" for x in checks) + len(uncertain),
        success=success)
