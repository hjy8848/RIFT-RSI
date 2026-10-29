Status: active engineering design
Owner: RIFT-RSI maintainers
Last verified: 2026-10-29
Scope: runtime data and trust boundaries for dual-bank learning

This document describes how the dual-bank learning objective flows through the existing Usami and VERL adapters. The research decision is in [RESEARCH_SPEC.md](RESEARCH_SPEC.md); versioned bank and reward rules are in [DUAL_BANK_DESIGN.md](DUAL_BANK_DESIGN.md).

## Runtime boundaries

Usami native worlds, Customer/Engine/Tool sessions, task certificates, replay records, final business outcomes, and private evidence remain inside the Host. The Service receives real multi-turn observations and tool results. It does not receive Oracle fields, strict process labels, hidden task identities, or probe membership. A public tool failure is a normal observation, not an Oracle annotation. The trusted evaluator may award only replay-verified `business_goal_success`; `strict_task_success` is not the learning reward.

`FailureObservation` is the Designer's allowlisted input: an opaque TaskRef plus its public summary. `TaskEvidenceRecord` stays Host-private and uses the common `BANK_CANDIDATE` purpose for both failure and success observations. UNKNOWN remains explicit and never becomes a negative label.

`AbilityRegistry` is a durable Host-owned index of capability-gap hypotheses, not a source of hidden benchmark labels. Designer returns an existing ability ID or `NEW`, its hypothesis, visible failure TaskRefs, and structured task proposals. The Host binds the selected proposal to the captured on-policy generation, validates IDs and public references, allocates stable IDs for `NEW`, and records explicit association edits only. No semantic matching is performed.

## Banks, splits, and probes

`BankManager` persists checkpoint-relative records and membership events. Bootstrap failures enter `FailureBank` without ability labels; unlinked failures remain available across rounds. Recovery reads from this same bank. `SuccessBank` supplies Retention candidates and the legacy Replay LogP baseline. These are candidate stores, not the reward itself.

`SplitRegistry` keeps the legacy strict family-disjoint policy and adds explicit `instance_disjoint_within_family_v1`. Old JSON registries load as strict. Every Probe snapshot records its policy version, seed, protocol, task identities, baseline checkpoint, and content hash. Recovery first samples eligible failures linked to the selected ability, then randomly fills from other eligible failures. Designer-seen history is allowed. A current Service training task and substantive overlap are excluded. Each selected historical TaskRef needs a separately verified baseline under the current Service checkpoint and protocol; old scores are never reused as the new baseline. Retention remains an independent random sample. TEST tasks are rejected from banks, training, and reward scoring.

`ProbeRunner` reuses the verified baseline trials in the frozen snapshot and executes the same task and seeds after the update. `score_dual_bank` requires complete Checkpoint, split, protocol, seed, trial and trusted Episode evidence bindings. It rejects scripted evidence and returns no scalar when valid coverage is insufficient. Its task-equal reward is `Recovery gain - λ * max(0, Retention regression)`. Independent TEST output is researcher-only and is not passed into bank update or Designer feedback.

## Policy updates and credit

The Service remains a full-parameter VERL 0.9.1 FSDP2 policy with vLLM rollouts and GRPO. `UsamiServiceAgentLoop` drives each episode through the native Host session and records actual token IDs and response masks. A complete same-prompt group is required; UNKNOWN or failed evidence invalidates the whole group.

Designer generates a structured proposal from FailureObservation and available ability IDs, with the generating policy/checkpoint/action stored. After Usami certification, Service training and dual-bank post-update evaluation, the verified scalar is assigned to that captured Designer rollout. Sequential rewards across changing Service checkpoints do not form comparable GRPO groups. The dual-bank Designer backend uses full-parameter `reinforce_baseline` with an explicit scalar baseline, no fabricated group ID and no batch whitening. The prior Replay LogP path remains independently selectable as an experimental baseline.

`LoopState` stores and validates the protocol tag and each transition. Old state JSON without a reward tag is loaded as Replay LogP; it is never reinterpreted as a dual-bank reward. Each external operation is followed by a durable state save, and resume requires the runtime's trusted update, reward, action, snapshot, and trial verifiers.

`EvolutionScheduler` coordinates bootstrap, explicit ability selection, round plan validation, frozen Probe preparation, both policy updates, delayed Designer credit, ability history, bank refresh and optional isolated TEST callbacks. A round plan binds the selected structured proposal to the original Designer output and current visible context. `rift-evolve` invokes it through an application-owned `EvolutionRunRuntime` factory. That factory must supply production Usami stores, VERL worker groups, evidence verifiers, checkpoint store, plan factory and bank updater. Importing code starts no model or training job; mocks are not accepted as real evidence.

`CommerceHost` and the worker-side `NativeHostProvider` can share one `CertifiedTaskRegistry` for resolving and authorizing a certified task. `SplitRegistry` remains the separate source for cross-split identity checks; CommerceHost no longer keeps a second task-object cache. Native Grounding returns a deterministic, budget-capped candidate sequence, which the host tries until certification succeeds. The default Customer profile handles mixed and information-only task shapes, but Usami's current proof certifier still marks information objectives UNKNOWN without an interaction certifier.

## Current deployment boundary

The repository pins VERL v0.9.1 commit `1876b06d0a3e4e71e06230be10af14492ca8a75b` and retains the requested 8×Ascend 910B3 configuration. The real Ray/FSDP2 worker launcher, durable certified-task and Episode stores, deployment trust roots, and exclusive NPU handoff remain application-owned. CPU tests verify protocol behavior with scripted/fake evidence only. The requested NumPy 1.26.4 also conflicts with upstream VERL's `numpy>=2.0.0` requirement; the target stack is not verified. See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).
