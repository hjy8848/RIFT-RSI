Status: implementation tracking
Owner: RIFT-RSI maintainers
Last verified: 2026-10-29
Scope: dual-bank code coverage, offline evidence, and deployment gaps

This document records which pieces of the dual-bank RIFT protocol have code and offline tests. The research objective is defined in [RESEARCH_SPEC.md](RESEARCH_SPEC.md); executable contracts are in `src/rift/` and defaults in `configs/dual_bank.yaml`.

## Implemented

- Checkpoint-relative SuccessBank and FailureBank records, explicit UNKNOWN trials, Host-private evidence storage, diagnosis allowlisting, and append-only membership events.
- Versioned split registry with unchanged legacy `strict_family_disjoint_v1` behavior and explicit `instance_disjoint_within_family_v1` support.
- Seeded random Recovery and Retention snapshot selection, pre-update baseline binding, same-task/same-seed after evaluation contracts, and evidence-addressed probe persistence.
- Dual-bank task-success reward with evidence coverage threshold, before/after checkpoint checks, same training/split isolation checks, and a verifier that reloads and recomputes its evidence.
- Legacy Replay LogP remains a separate tagged reward and state protocol; legacy untagged LoopState loads as Replay LogP.
- Full-parameter VERL role adapters remain in place. Service uses GRPO complete groups. Sequential delayed Designer feedback uses ungrouped REINFORCE with explicit scalar baseline and no batch whitening.
- Pending Designer generation action is retained and verified through Service training so delayed reward is credited to its exact on-policy rollout.
- Dual-bank stages are persisted in LoopState. `EvolutionScheduler` and the `rift-evolve` application-factory entry coordinate round execution and checkpoint handoff without embedding fake models or checkpoints.
- `IndependentTestRunner` enforces the TEST purpose boundary; its result is isolated from learning reward and bank updates by the scheduler contract.

## Offline verification

The added tests exercise capability classification and UNKNOWN exclusion, diagnosis privacy projection, both split policies, reproducible random Probe freezing, same-seed Probe evidence storage, scripted-evidence rejection from real reward, reward arithmetic, ungrouped Designer credit, and dual-bank LoopState save/restore. Existing Service, VERL bridge, capture and Usami adapter tests are run with the full suite. These tests use scripted benchmark records and mocked verifiers/workers; they do not demonstrate model learning.

## Still required from a deployment runtime

- A production `EvolutionRunRuntime` factory with persistent Usami task/certificate/Episode stores, authoritative identity construction, native multi-turn evaluator, real VERL FSDP2 worker groups, checkpoint store, and production update/reward/action/trial verifiers.
- A bank updater that persists real Bootstrap and subsequent task/probe observations with trusted evidence, and a plan factory that restores the exact pending Designer action on interrupted rounds.
- A tested exclusive lifecycle for two full-parameter policies sharing the eight NPU devices, including optimizer/RNG restore, serving weight synchronization, and safe resource release between Designer and Service phases.
- Integration validation of UNKNOWN and failed-tool behavior against actual Usami episodes and actual Agent Loop token capture.
- Resolution and validation of the requested software image. The pinned VERL v0.9.1 source requires `numpy>=2.0.0`, which conflicts with requested NumPy 1.26.4; no dependency override is claimed safe.

No real training, model inference, checkpoint update, replay measurement, capability acquisition, forgetting measurement, or Ascend job has been run. A real experiment remains unverified until performed in the target environment using `configs/dual_bank.yaml` and independently frozen TEST tasks.
