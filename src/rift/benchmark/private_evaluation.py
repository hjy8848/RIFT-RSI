"""Trusted host only. Native records and certificates never cross the public boundary."""
import json
from rift.schemas import Outcome
from rift.benchmark.public_feedback import PublicEpisode, PublicEvent
from rift.tasks import Certification, CertificationStatus

def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def public_business_outcome(record, replay_verified):
    # Strict success and evaluation_status combine privileged process checks.
    # Replay integrity, independently recomputed by the host, is essential.
    if not replay_verified or record.outcome.protocol != "commerce-outcome-v2":
        return Outcome.UNKNOWN
    value = record.outcome.business_goal_success
    if value is True: return Outcome.SUCCESS
    if value is False: return Outcome.FAILURE
    return Outcome.UNKNOWN

def project_events(record, replay_verified):
    if not replay_verified:
        raise ValueError("episode must pass native replay")
    result = []
    pending_terms = None
    scheduled = {e.id: e for e in getattr(record, "scheduled_external_events", ())}
    for event in record.events:
        p = event.payload
        if event.kind == "customer_request":
            result.append(PublicEvent("user", p["request"]))
        elif event.kind == "customer_transition" and p.get("transition_kind") == "decision":
            msg = p.get("decision", {}).get("message")
            if msg: result.append(PublicEvent("user", msg))
            customer_event = p.get("customer_event", {})
            decision = p.get("decision", {})
            if customer_event.get("rejection_reasons") and not (decision.get("finish") and customer_event.get("finish_allowed")):
                result.append(PublicEvent("user", "A previously recorded Customer consent is still pending execution. Please continue the requested workflow."))
        elif event.kind == "service_tool":
            result.append(PublicEvent("tool_call", _json(p["operation"])))
            result.append(PublicEvent("tool_result", _json(p["result"])))
        elif event.kind == "external_event":
            ext = scheduled.get(p["event_id"])
            if ext is None: raise ValueError("external event absent from schedule")
            result.append(PublicEvent("tool_result", _json({"external_event": ext.kind, "status": "applied", "target_id": ext.target_id})))
        elif event.kind == "offer_presented":
            pending_terms = p["rendered_terms"]
        elif event.kind == "service_reply":
            message = p["message"]["text"]
            if pending_terms is not None: message += "\n" + pending_terms
            result.append(PublicEvent("assistant", message))
            pending_terms = None
    return PublicEpisode(tuple(result), public_business_outcome(record, replay_verified))

def public_certification(proposal_id, certificate, *, verified):
    # Certificate contains witness hashes, proof attempts and private blockers.
    if not verified: return Certification(proposal_id, CertificationStatus.UNKNOWN)
    if certificate.status == "CERTIFIED" and certificate.full_task_reachability_certified:
        return Certification(proposal_id, CertificationStatus.CERTIFIED, certificate.task_hash)
    if certificate.status == "UNSAT": return Certification(proposal_id, CertificationStatus.REJECTED)
    return Certification(proposal_id, CertificationStatus.UNKNOWN)
