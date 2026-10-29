Status: active engineering design
Owner: RIFT-RSI maintainers
Last verified: 2026-10-29
Scope: runtime data and trust boundaries for dual-bank learning

This document describes how the dual-bank learning objective flows through the existing Usami and VERL adapters. The research decision is in [RESEARCH_SPEC.md](RESEARCH_SPEC.md); versioned bank and reward rules are in [DUAL_BANK_DESIGN.md](DUAL_BANK_DESIGN.md).

## Runtime boundaries

Usami native worlds, Customer/Engine/Tool sessions, task certificates, replay records, final business outcomes, and private evidence remain inside the Host. The Service receives real multi-turn observations and tool results. It does not receive Oracle fields, strict process labels, hidden task identities, or probe membership. A public tool failure is a normal observation, not an Oracle annotation. The trusted evaluator may award only replay-verified `business_goal_success`; `strict_task_success` is not the learning reward.

`PublicSummary` and `DiagnosisExperience` are allowlisted Designer inputs. They contain public events and the permitted final business outcome, but no TaskRef, world identity, lineage, checkpoint, or private evidence. `TaskEvidenceRecord` is a Host-private record with purpose, identity, checkpoint, protocol, seeds, outcomes, and evidence references. UNKNOWN remains an explicit status and never becomes a negative label.

`AbilityRegistry` is a durable Host-owned index of capability-gap hypotheses, not a source of hidden benchmark labels. Each captured Designer hypothesis gets a provisional entry; the runtime does not infer semantic links from matching words or assign confidence to such a match. A Host can add a failure TaskRef to the active priority set only after verifying real Service failure evidence. Older confidence-scored links remain historical and are not used for priority selection. The captured hypothesis is checked against the original on-policy generation record.

## Banks, splits, and probes

`BankManager` persists checkpoint-relative records and append-only membership events. `FailureBank` supplies public diagnosis examples and separately held Recovery candidates; `SuccessBank` supplies Retention candidates and the legacy Replay LogP baseline. Banks are candidates, not the reward itself. Their membership changes only after verified observations.

`SplitRegistry` keeps the legacy strict family-disjoint policy and adds explicit `instance_disjoint_within_family_v1`. Old JSON registries load as strict. Under the new version, different instances can share a broad family while world, task, and source lineage remain isolated. Every Probe snapshot records its policy version, selection seed, protocol digest, task identities, baseline checkpoint, and content hash. Recovery uses explicitly linked historical failures as priorities, then fills remaining slots randomly from other eligible failures. Historical failure Checkpoints do not serve as this round's baseline: each selected TaskRef must have a separately verified record at the requested current Checkpoint and protocol. Snapshot metadata records the target ability and priority task refs. Retention remains an independent random sample. The scheduler excludes every Diagnosis identity previously shown to Designer, across restarts, as well as current training and TEST identities.

`ProbeRunner` reuses the verified baseline trials in the frozen snapshot and executes the same task and seeds after the update. `score_dual_bank` requires complete Checkpoint, split, protocol, seed, trial and trusted Episode evidence bindings. It rejects scripted evidence and returns no scalar when valid coverage is insufficient. Its task-equal reward is `Recovery gain - λ * max(0, Retention regression)`. Independent TEST output is researcher-only and is not passed into bank update or Designer feedback.

## Policy updates and credit

The Service remains a full-parameter VERL 0.9.1 FSDP2 policy with vLLM rollouts and GRPO. `UsamiServiceAgentLoop` drives each episode through the native Host session and records actual token IDs and response masks. A complete same-prompt group is required; UNKNOWN or failed evidence invalidates the whole group.

Designer generates a structured package proposal from DiagnosisExperience and stores the generating policy/checkpoint/action. After Usami certification, Service training and the dual-bank post-update evaluation, the verified scalar is assigned to the exact captured Designer rollout. Sequential rewards across changing Service checkpoints do not form comparable GRPO groups. The dual-bank Designer backend uses full-parameter `reinforce_baseline` with an explicit scalar baseline, no fabricated group ID and no batch whitening. The prior Replay LogP path remains independently selectable as an experimental baseline.

`LoopState` stores and validates the protocol tag and each transition. Old state JSON without a reward tag is loaded as Replay LogP; it is never reinterpreted as a dual-bank reward. Each external operation is followed by a durable state save, and resume requires the runtime's trusted update, reward, action, snapshot, and trial verifiers.

`EvolutionScheduler` coordinates bootstrap, the persistent Ability Registry, round plan validation, frozen Probe preparation, both policy updates, delayed Designer credit, per-ability learning/evaluation history, candidate-bank refresh and optional isolated TEST callbacks. A round plan binds a registered provisional hypothesis to the captured Designer output for that round. `rift-evolve` invokes it through an application-owned `EvolutionRunRuntime` factory. That factory must supply the production Usami stores, VERL worker groups, evidence verifiers, checkpoint store, plan factory and bank updater. Importing code starts no model or training job; no built-in mock is accepted as real evidence.

## Current deployment boundary

The repository pins VERL v0.9.1 commit `1876b06d0a3e4e71e06230be10af14492ca8a75b` and retains the requested 8×Ascend 910B3 configuration. The real Ray/FSDP2 worker launcher, durable certified-task and Episode stores, deployment trust roots, and exclusive NPU handoff remain application-owned. CPU tests verify protocol behavior with scripted/fake evidence only. The requested NumPy 1.26.4 also conflicts with upstream VERL's `numpy>=2.0.0` requirement; the target stack is not verified. See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).
