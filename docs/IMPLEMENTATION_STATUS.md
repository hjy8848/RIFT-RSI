Status: implementation tracking
Owner: RIFT-RSI maintainers
Last verified: 2026-10-29
Scope: dual-bank code coverage, offline evidence, and deployment gaps

This document records which pieces of the dual-bank RIFT protocol have code and offline tests. The research objective is defined in [RESEARCH_SPEC.md](RESEARCH_SPEC.md); executable contracts are in `src/rift/` and defaults in `configs/dual_bank.yaml`.

## Implemented

- One `BANK_CANDIDATE` task purpose feeds FailureBank and SuccessBank. Bootstrap failures need no ability label; unresolved failures remain recorded. Failure and Recovery no longer form disjoint task partitions.
- AbilityRegistry stores ID, description, explicit failure TaskRefs, training/evaluation history, and explicit link revisions. Designer selects an existing ID or `NEW`; Host validates captured generation, visible IDs and TaskRefs, verifies real Failure Bank evidence, and assigns stable IDs. No semantic matcher or old unused Registry migration branch remains.
- Designer receives opaque `FailureObservation` references with public summaries and available ability choices. Previously shown failures may enter Recovery; task selection must still exclude current Service training overlap and TEST.
- Versioned split registry with unchanged legacy `strict_family_disjoint_v1` behavior and explicit `instance_disjoint_within_family_v1` support.
- Seeded Recovery selection that samples linked failures first and randomly fills from other eligible Failure Bank records, plus independent random Retention sampling. Historical failures must pair with a verified baseline at the current Service checkpoint and protocol. Frozen snapshots persist priority refs, seed, current baseline and same-task/same-seed after-evaluation contracts.
- Dual-bank task-success reward with evidence coverage threshold, before/after checkpoint checks, same training/split isolation checks, and a verifier that reloads and recomputes its evidence.
- Legacy Replay LogP remains a separate tagged reward and state protocol; legacy untagged LoopState loads as Replay LogP.
- Full-parameter VERL role adapters remain in place. Service uses GRPO complete groups. Sequential delayed Designer feedback uses ungrouped REINFORCE with explicit scalar baseline and no batch whitening.
- CommerceHost tries native Grounding candidates in its deterministic, budget-capped order until one certifies. One shared CertifiedTaskRegistry is the task resolver/authorizer for execution; SplitRegistry retains only split-overlap state.
- Default Usami Customer bounds now accept information-only and mixed TaskSpecV3 tasks. The pinned native proof certifier still returns UNKNOWN for information goals until an interaction certifier is supplied; the adapter does not bypass that boundary.
- Malformed Service tool output is recorded by the native Runner as a Service protocol failure. Tool-parser execution exceptions produce a replay-verified cancelled Episode with UNKNOWN business reward and retained token evidence.
- Probe seed is passed to CustomerFactory and seeds each Service generation from a scope stable across before/after checkpoints. Per-generation sampling settings remain in generation evidence.
- Pending Designer generation action is retained and verified through Service training so delayed reward is credited to its exact on-policy rollout.
- Recovery snapshots exclude the current training package, substantive overlap under the selected split policy, and TEST, but allow previously Designer-visible task references.
- Dual-bank stages are persisted in LoopState. `EvolutionScheduler` and the `rift-evolve` application-factory entry coordinate round execution and checkpoint handoff without embedding fake models or checkpoints.
- `IndependentTestRunner` enforces the TEST purpose boundary; its result is isolated from learning reward and bank updates by the scheduler contract.

## Offline verification

CPU tests exercise unlabelled bootstrap, UNKNOWN exclusion, public failure projection, split policies, explicit `NEW` and existing-ID choices, captured proposal/context binding, cross-round failure reuse, training-overlap exclusion, current-checkpoint baselines, reproducible random fill, TEST rejection, Registry persistence, Probe evidence storage, scripted-evidence rejection from real reward, reward arithmetic, ungrouped Designer credit, dual-bank LoopState save/restore, native information-only Customer execution, multi-candidate certification fallback, seeded probe construction, and native parse-failure outcomes. Model generation and runtime attestation remain test doubles; these tests do not demonstrate model learning.

## Still required from a deployment runtime

- A production `EvolutionRunRuntime` factory with persistent Usami task/certificate/Episode stores, authoritative identity construction, native multi-turn evaluator, real VERL FSDP2 worker groups, checkpoint store, and production update/reward/action/trial verifiers.
- A production CustomerFactory that applies the supplied probe seed to its stochastic behavior, and an evaluator that uses the same recorded Service sampling protocol before and after updates.
- A bank updater that persists real Bootstrap and subsequent task/probe observations with trusted evidence, and a plan factory that restores the exact pending Designer action on interrupted rounds and supplies current-Checkpoint baseline records for historical Recovery and Retention candidates.
- A tested exclusive lifecycle for two full-parameter policies sharing the eight NPU devices, including optimizer/RNG restore, serving weight synchronization, and safe resource release between Designer and Service phases.
- Integration validation with the configured VERL parser, real inference server, production CustomerFactory, persistent task/Episode stores, and Agent Loop token capture.
- Resolution and validation of the requested software image. The pinned VERL v0.9.1 source requires `numpy>=2.0.0`, which conflicts with requested NumPy 1.26.4; no dependency override is claimed safe.

No real training, model inference, checkpoint update, replay measurement, capability acquisition, forgetting measurement, or Ascend job has been run. A real experiment remains unverified until performed in the target environment using `configs/dual_bank.yaml` and independently frozen TEST tasks.
