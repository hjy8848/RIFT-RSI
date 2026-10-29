Status: active design
Owner: RIFT-RSI maintainers
Last verified: 2026-10-29
Scope: implemented first-stage boundaries and outstanding integration decisions

This document tracks which components block a reproducible RIFT experiment; the contracts are in `src/rift/` and the research decision in [RESEARCH_SPEC.md](RESEARCH_SPEC.md).

Implemented: proposal/certificate separation, host-only native API wrappers, public event and outcome projection, replay snapshot and overlap checks, token-masked reward arithmetic, trainer/scorer/checkpoint protocols, and training-request/state persistence. `orchestrator.prepare_round` composes injected Designer and benchmark actors up to the training boundary. These are scaffolding; a caller must preserve Host isolation and trusted provenance. The native certificate proves bounded business reachability and does not certify natural customer interactions.

The default commerce Customer scenario and profile support structured remedy-only tasks with a shared conservative policy; information goals, return scheduling, and review events need explicit implementations. Certification proves bounded business reachability, not natural dialogue success.

Not yet implemented: authoritative upstream split/lineage provenance, a real token recorder and verifier for exact model inputs and Service/Designer action tokens, actual model scorer and checkpoint store, full-parameter Service and Designer trainers, production demonstration/update/replay-reward evidence verifiers, automatic invocation and recovery of real training backends, independent evaluation, and cross-checkpoint normalization. A test verifier can exercise state transitions but is not evidence of a real parameter update or replay measurement. No model learning or performance result is claimed. Native API verification and offline mock tests are reported in the final handoff, not assumed here. Until the missing boundaries are implemented, this scaffold must not be used to claim a RIFT experiment.
