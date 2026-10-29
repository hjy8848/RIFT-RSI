# Genuine multi-turn information protocol — foundation checkpoint

This document describes implemented shared protocol primitives. It does not claim the four dialogue domains or complete agent sessions are delivered yet.

## Private Customer data

`CustomerPrivateState` freezes identity, opening need, objective facts, allowed preference values, initial private preferences, confirmable operation families and protected business objects. Its hash detects even nested-dictionary changes. These fields do not automatically enter Service state.

`initialize()` creates only authenticated identity. Neither private facts nor private initial preferences are silently disclosed. `public_context()` exposes only actual Customer-provided facts, stated preferences, current revision and established commitments.

`CustomerIntent` is a structured natural response, not a fixed line in a scripted conversation. An intent can disclose selected facts, state a permitted preference, reject an offer, confirm one exact actual offer or finish. The future Customer model remains fixed and must choose these responses from the actual Service context; no forced turn count is part of these primitives.

## Trusted event reduction

`customer_event()` independently checks disclosed values against the immutable private profile. It refuses invented objects/facts, unsupported preferences, actor switching and unauthorized confirmation scope.

A preference revision or offer rejection increments the revision and invalidates old consents/commitments. Confirmation must bind an actual offered operation, quote hash, world hash and current revision. A domain-native consent callback must verify quote/object/constraint semantics; absence of that callback is rejected. Service tool arguments alone cannot create consent.

`information_code()` and `consent_code()` are deterministic process checks. Domain References must use them as real business preconditions. Those References, dialogue trace replay and per-industry consent callbacks remain subsequent work; primitive tests alone do not prove a real industry's multi-turn necessity.

## Necessity evidence

`DependencyProof` separates missing information, preference choice, explicit consent, external state update, cross-turn commitment and partial completion. A proof is invalid unless a replayable control change actually changes whether the chosen operation is Reference-allowed. Different denial text with the same allowed action set is not accepted as necessity evidence.

The final implementation must produce these proofs from actual native domain References and execution, rather than manually filling booleans. Per-task certificates and counterfactual replay are still pending.

## Actual tests

Ten primitive tests passed: selective disclosure/private separation, fabricated facts, exact current consent, preference revision/revocation, wrong world/scope, missing native consent check, nested private-state mutation and rejection of error-label-only necessity claims. These tests contain no paid model calls and do not demonstrate Agent learning.

## Executable After-sales implementation

The subsequent checkpoint adds actual shared sessions, native Oracle delegation, complete process/external Replay, and a fixed-policy Customer/Service Function Calling runner under `usami_dialogue/`. After-sales now supplies its native models, independent Reference and mutation implementations, tools, private-profile validation, quote binding, manager and carrier simulators.

`CustomerIntent.requested_operations` records the Customer's exact declared write intentions. They are independently validated against the frozen profile and consumed once. Service tool arguments cannot establish authorization. Exact Customer arguments may themselves disclose known facts; the runtime does not force an unnecessary clarification turn when the Customer has already supplied the information.

The Service sees scoped business observations and actually disclosed facts. Only the Customer receives the frozen private profile. Production dialogue is chosen by the configured LLM, with no forced turn count or witness injection. All model requests/responses, protocol failures, bounded format recovery, operation results, role-specific usage and external updates are retained. Missing provider token usage is flagged as incomplete rather than presented as a complete zero-cost result.

Structured refusal decisions are independently checked against native Reference and actual unconsumed Customer intentions. They cannot be earned by trying an illegal write and letting Engine reject it. Detailed task and necessity certificates are described in [TASK_CERTIFICATION.md](TASK_CERTIFICATION.md). The remaining three new domains and dataset-wide claims are still pending.


## Canonical v2 model context and fixed Customer goal scope

After-sales, Airline and Property now share canonical rendered private JSON, deterministic entity/candidate ordering and an exact disk-roundtrip model-journal Replay check. The model-run protocol versions are `business-dialogue-agents-v2` and `business-dialogue-task-agents-v2`; v1 archives retain their own source snapshots. These versions concern model-message rendering, not changes to historical Engine/Reference/Oracle semantics.

A selected task can narrow a Customer's already frozen acceptable business choices. Task-scoped runtime validation rejects a generated preference/request that leaves those choices, records the Customer-side protocol error/cost and does not invoke Service for that invalid turn. It does not consult hidden Reference legality or shield Service from an unlawful operation. Service actions still execute through the actual Engine, with Sandbox failures independently observable.
