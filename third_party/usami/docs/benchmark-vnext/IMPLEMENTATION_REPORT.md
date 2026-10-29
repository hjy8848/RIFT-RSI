# Usami Benchmark vNext — implementation ledger

This is an active engineering ledger, not a completed release claim. The full request is the eight-domain upgrade dated 2026-10-17. No paid LLM run is authorized or required.

## Requirement trace

| Work package | Evidence required for completion | Current status |
| --- | --- | --- |
| Preserve four historical domains/releases | Old generator hashes, historical tests and Replay | Four legacy seed/source pins verified; old Web/native regression passed after namespace isolation |
| Three structural generators | 100 worlds each, native invariants, ID-neutral signatures, ≥30 meaningful structures or actual gap | Implemented under isolated vNext namespace; 100 valid audited worlds per existing domain; native StayOps consistency issue recorded |
| Four dialogue domains | Domain-specific rules/appliers, private Customer state, information/consent dependencies, proof and Replay | All four native cores, shared runtime, certificates and offline examples implemented; historical/multi-entity breadth and unified release remain pending |
| Eight-domain contracts and audit | Unified provenance, taxonomy, structural/permission/decision/task coverage | WorldSpec/FrozenWorld/pool/signatures, state census v2, six dependency types and private dialogue reducer implemented; complete task/decision coverage pending |
| Pilot 320 / Core 800 | Frozen certified tasks, truthful counts, stratified coverage and leakage audit | Pending |
| Multi-domain Frozen Discovery | Real source-bound branches, fixed Customer, static-only inheritance and exact Replay | Existing CloudOps path inspected; expansion pending |
| Evaluation and ablations | Equal domain/track weights, immutable manifests, fixed budgets and offline validation | Frozen aggregation plans, equal weights and provenance/denominator tests implemented; native all-domain result adapters and ablations pending |
| Full tests/docs/examples | Executed test logs, source pins and eleven requested documents | Education checkpoint full regression 1153 passed; subsequent vNext integration 117 passed and split/evaluation/generator set 25 passed; all eleven document paths now exist with actual limits; final release verification pending |

## Architecture findings

The repository contains four real domains, with independent business rules and mutations. Fulfillment and StayOps share `usami_harness` Engine/Oracle/Service plumbing through explicit adapters. CloudOps has an older independent runtime; SaaS has versioned rules, generators and GoalServiceSession. Existing fixed manifests and experiment archives are not migration targets.

Current native CloudOps/Fulfillment/StayOps `generate_world()` fixtures mainly vary display labels. Their replacements must therefore be separate versioned generators, not silent changes to those fixtures. The structural generator version describes the distribution; native world schema/rules versions remain explicit and unchanged when business semantics are reused.

The latest Frozen-World training defaults to `explicit-selection-v2`, with quality scoring and both dynamic gates false. This remains mandatory for the new eight-domain training path. Independent Benchmark scoring is separate from training selection.

## Work remaining

All items labelled pending remain part of the active objective. Completing Phase 1 does not constitute completion of the eight-domain Benchmark.

## Checkpoint: structural generators and dialogue primitives

Implemented independent structural generator modules for CloudOps, Fulfillment and StayOps (currently `usami_benchmark/generators/<domain>.py`), plus strict WorldSpec registration, immutable multi-domain pools and ID-neutral graph signatures. No native legacy source file was changed.

The first audit attempted 400 worlds and exited 1 because four StayOps histories revealed a native Reference/invariant conflict. All original failure evidence and exact sources are retained. Corrected admissibility excludes invalid expected states only during independent initial-state generation; it does not alter Service execution. The second audit exited 0 and produced 100 valid worlds / 100 anonymous signatures per existing domain, with four explicit StayOps exclusions. Native simulator correction and per-task boundary coverage remain outstanding.

Executed checks so far: 113 generator/native-world regressions passed; 10 private fact/preference/consent protocol tests passed. The final complete repository regression passed 1069 tests in 500.95 seconds (exit 0). World-only audit fields for task proof, Reference boundary coverage and causal depth remain NOT RUN.

The full objective remains active: four complete native dialogue domains, 320/800 certified-task targets, eight-domain coverage/weighting and strict leakage audit, multi-domain Frozen Designer execution and full evidence/selection Replay are not yet implemented. No actual LLM experiment has run.

The first full regression stopped with 909 passed / 1 failed: historical Web loading pins every Python file in native domain directories. Adding a new generator there changed the expected runtime file set. New distributions were therefore relocated to `usami_benchmark/generators/`; historical verifiers and frozen manifests were not relaxed. The earlier audit source archives retain their original implementation locations for exact replay.


The final namespace-isolated 400-world audit exited 0. All 400 complete world hashes equal the corrected pre-relocation run. Verification: `audits/namespace-migration-verification.json`. Original source archives and both earlier audits remain intact. The 41-case new-generator/dialogue/Web compatibility regression passed after relocation.

Next engineering priority: implement a genuinely versioned correction/adapter for the known StayOps rule/invariant conflict without changing published native runtime files or weakening old source verification. Then complete After-sales end to end, validate objective multi-turn necessity, and extend the proven mechanism to Airline, Property and Education. The remaining dataset, coverage, evaluation and multi-domain VAST requirements are unchanged.


## Verified checkpoint result

- `pytest -q tests/test_vnext_generators.py tests/test_cloudops_world.py tests/test_fulfillment_world.py tests/test_stayops_world.py -x`: 113 passed, exit 0.
- Dialogue protocol primitives: 10 passed, exit 0; initial Python 3.9 annotation collection failure and correction are recorded.
- Post-isolation generator/dialogue/Web compatibility: 41 passed, exit 0.
- Final `pytest -q -x`: 1069 passed in 500.95 seconds, exit 0.
- Final four-domain world audit: 400/400 valid, exit 0, 100 anonymous structures/domain, exact namespace migration verified.
- All raw outputs are under `benchmarks/vnext-0.1/verification/logs/`.
- No existing tracked business code, old benchmark data or historical experiment has been modified. No real LLM experiment has run.

This is a committed engineering checkpoint, not completion of the user objective. The four new business domains, real dialogue Engine/Oracle/certificates, complete 320/800 task releases, capability/semantic leakage audits, equal-weight scoring and eight-domain VAST evidence/selection Replay remain mandatory active work.


## Phase-2 checkpoint: native After-sales and executable dialogue

Implemented real order-line disputes, evidence verification, refund approval thresholds, original-channel settlement, replacement inventory, return pickup/receipt and duplicate-settlement boundaries. Reference expected mutations and actual mutations are separate. Manager and carrier outcomes are frozen simulator events; neither model can fabricate their outcomes. Service receives scoped observations, not private profiles or hidden simulator verification flags.

The shared dialogue runtime independently establishes Customer identity and exact requested writes, records actual quote presentation, validates truthful disclosures, revokes superseded consent and retains full business/process/external traces. A real OpenAI-compatible two-Agent runner journals exact requests, replies, tools, usage by role, budgets and bounded protocol recovery. Fixture tests use that same runner; no paid calls were made.

Native Reference search creates private reachability witnesses. Objective scoring accepts different legal outcomes/paths, enforces the domain frame and checks exact correct-refusal subgoals. Counterfactual certificates prove changes to actual Reference legality after removing facts/consent, changing allowed preferences, or receiving a frozen manager event. The mixed example completes a real remedy and refuses a duplicate request without invoking a forbidden write.

Initial added tests passed 49 cases before the serialization-roundtrip extension. CLI certification found a separate ordering issue on reloading sorted JSON: proof ordering depended on dictionary insertion order. The failed bundle/source archive and verification traceback are retained as `examples/aftersales-mixed-001/`. Proofs now have canonical content-hash ordering, a JSON-roundtrip assertion has been added, and `examples/aftersales-mixed-002/` certified and independently verified successfully. This is an offline native proof, not a model result.

World audits: `audits/aftersales-native-002/` contains 100/100 valid worlds, 100 anonymous structures and 99 ownership topologies. The first After-sales audit aborted due to a new scanner variable-shadowing error before any world registration; its source and traceback remain in `aftersales-native-001/`. A corrected new run was saved separately. `audits/stayops-v2-001/` contains 100/100 valid worlds and 100 structures/topologies under explicitly registered `stayops-structural-v2` / `stayops-0.2.0`. The old native StayOps rules remain untouched. Its versioned correction permits future assignments of a currently dirty unoccupied room while retaining current occupancy/cleanliness constraints.

### Still mandatory

- Finish After-sales domain breadth (including initial historical case distributions, richer account and cross-order scenarios) and comprehensive boundary/task audit.
- Implement distinct Airline, Property and Education native state machines and generators; do not copy the refund flow into renamed domains.
- Complete all eight native task composers, stratified Pilot/Core manifests, capability/depth/decision coverage, equal-weight scoring and semantic split/leakage audit.
- Complete cross-turn commitment/partial-completion dependency coverage beyond the present counterfactual and mixed-task examples.
- Extend Frozen Task Discovery and source-bound bidirectional evidence/Reflection/static-only version selection to eight domains.
- Finish all requested documents, datasets, final source-pinned verification and full release audit. The current work is not an eight-domain release.

### Offline example commands

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m usami_aftersales verify \
  --bundle benchmarks/vnext-0.1/examples/aftersales-mixed-002/bundle.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m usami_aftersales certify \
  --seed 1 --profile profile_u1_order0_line1 --mixed --resolution refund \
  --max-search-nodes 256 --max-paths 1 --output /tmp/usami-aftersales-new-certificate
```

Use a new output directory. Archived bundles contain exact sources and reject current source mismatches; if the runtime advances, use the corresponding archived source snapshot. Actual LLM execution is separately available via `python -m usami_aftersales run --online` with environment-configured models/credentials and frozen budgets. It is not invoked by certification or verification. Customer/Service-specific settings can override the shared environment via `USAMI_CUSTOMER_LLM_*` / `USAMI_SERVICE_LLM_*`. No key or HTTP authorization header is stored.


Full regression for the After-sales checkpoint finished with exit 0: **1094 passed in 508.69 seconds**. Raw log: `benchmarks/vnext-0.1/verification/logs/aftersales-checkpoint-full-1094.log`. This result predates the subsequent Airline implementation and cannot be used to claim validation of that later code.


## Phase-2 checkpoint: Airline and shared certified-task execution

Airline now has a distinct native ticketing model, independent expected/actual transitions, flight/cabin seat conservation, spatial and temporal connection constraints, passenger-document verification, voluntary fare restrictions versus airline cancellation, fee/refund computation, baggage limits, simulated payment capacity, current-quote consent, actual ledger accounting, scoped queries, frozen operational updates and full Replay.

The shared `usami_dialogue.planner` now handles private native Reference search for both implemented dialogue domains; it does not drive production Service decisions. Airline composers cover rebooking, refund, baggage, standalone policy refusal and appropriate mixed completion/duplicate refusal. Correct refusal certification uses a real unattempted Customer request and independently checked policy reason. Pure refusal frames preserve ticket and financial state while allowing the necessary identity-verification record.

A separate `business-dialogue-task-agents-v1` configuration binds the exact task, private certificate and witness before any paid model call. It derives a Customer-facing business request from domain-validated goals. Reference outcomes, policy refusal codes and witnesses remain host-side. The earlier `business-dialogue-agents-v1` config/journal shape is retained. The fixed Customer's weights and policy are unchanged; selected task goals guide what it wants, without changing its objective facts or revealing an answer. The After-sales online CLI now uses this certified task variant. No online invocation was performed.

Actual validation:

- `pytest -q tests/test_vnext*`: **69 passed in 44.03s**, exit 0. This includes the two native domains, original generator/StayOps tests, process/counterfactual/certificate checks, production Function Calling runner with explicitly scripted transport, exact model-request Replay, task goal isolation and rejection of invalid certification before any model call.
- Airline world audit: **100/100 valid**, **100 ID-neutral structures**, **97 ownership topologies**, no invalid worlds. Task-scale decision/depth/coverage measurements remain NOT RUN.
- Four archived Airline native examples: refund (mixed), rebook, baggage (mixed), standalone fare-refund refusal. All verified through independent Oracle and actual transition Replay.
- The entire example source snapshot was extracted to a fresh isolated directory. The four cases verified successfully using only that archived runtime (exit 0). This checks that Replay does not accidentally depend on the current checkout.
- The prior 1094-test full repository regression is the preceding After-sales checkpoint result, not a final validation claim for all subsequent work. A final full regression is still required for the full eight-domain release.

Artifacts: `audits/airline-native-001/`, `examples/airline-native-002/`, and `verification/logs/`. The earlier `airline-native-001` example is retained with its exact sources; the second example includes the later certified-task protocol and stricter pure-refusal frame. Neither is a model experiment. Source-pinned example verification intentionally rejects a modified runtime; use the corresponding archived source snapshot.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m usami_dialogue verify-examples \
  benchmarks/vnext-0.1/examples/airline-native-002
```

Remaining scope is unchanged: After-sales historical/account/cross-order expansion, complete Property and Education domains, all eight native task bindings and certified datasets, coverage/leakage/macro evaluation, fixed-Customer multi-domain Designer/Service Discovery and selection Replay, and final release verification. No paid LLM training, actual learning gain or cross-domain generalization has been demonstrated.


## Phase-2 checkpoint: Property and canonical model-journal protocol

Property (`usami_property/`) adds a genuinely different domain: buildings/units and owner/resident relationships, contractual repair responsibility and coverage, reported hazards and triage, technician skills/on-call availability, overlapping calendars, urgent response limits, actual owner review, scoped work plans, appointment consent, physical technician outcomes, charges after completed work, separate customer closure consent, failed-work complaints and escalation. A reservation is not a completed repair. A new inspection supersedes old work instructions; the simulator will not manufacture a visit from a superseded plan. These mechanisms reuse the existing native Engine, independent Oracle, process reducer, proof search and Replay, rather than a renamed ticket-refund workflow.

Completed evidence at this checkpoint:

- `pytest -q tests/test_vnext*`: **89 passed in 92.00s**, exit 0. Raw log `verification/logs/three-dialogue-domains-89.log`. Tests cover the prior generators, three new native domains, independent business violations in Sandbox, protected contacts, true owner-review dependency, false closure, conflicting appointments, consent/revision, objective task proof, model-request Replay and Customer task drift.
- Property audit: **100/100 valid worlds, 100 anonymous structures, 100 ownership topologies**. Names/contacts/identifiers, skill-list order and currency scaling do not create extra structures. Source and audit data are retained in `audits/property-native-001/`.
- Four source-pinned native examples in `examples/property-native-002/`: verified repair plus duplicate-closure refusal; scheduling; a genuinely failed visit followed by a complaint and escalation; and a correct refusal that creates no fake work order. Offline generation/Oracle/Replay verified. Earlier `property-native-001` sources/data remain intact.
- Complete production-runner scripted transport journal in `verification/fixtures/property-agent-002/`: 17 fixture completions, 6 actual business/query events, real simulated work receipt and two distinct consents. Fresh sorted-JSON/gzip load and exact Replay passed, including replay from an isolated extracted source archive. **Real API calls: 0; real-model experiment: NOT RUN.** Token usage is absent in fixtures and is not a zero-cost model-performance estimate.

### Serialization bug and explicit version boundary

The initial scripted journal `property-agent-001` passed an in-memory check but failed a later disk roundtrip: sorted JSON changed dictionary insertion order, and the old prompt renderer inserted private dictionaries as non-canonical JSON text. It could therefore construct a different model request after loading the same hashed world. The original journal, source archive, metadata and failure traceback are preserved, with an appended failure note clarifying that its initial `replay_verified` flag was an in-memory result only.

The new model protocols are `business-dialogue-agents-v2` and `business-dialogue-task-agents-v2`. Private prompt JSON, entity query/observation lists and candidate ordering are now canonical. V1 model journals require their archived runtime; they are not silently interpreted as v2. The business trace format and all four historical domain runtimes/manifests are unchanged. Sorted-JSON tests now cover all three new domain views and exact model requests.

Certified task execution additionally checks that the fixed Customer stays within the selected business goal's resolution/service/visit choices. This uses task parameters, not Reference approval or performance scores. Deviations are recorded as Customer protocol errors before any Service action, not blamed on Service. Neither Customer nor Service weights/Skill were evolved in these tests.

The new `dialogue-task-score-v2` keeps the independent Oracle's current correct-refusal check strict, while recording the certificate's intended boundary separately from the actual refusal reason. An accurate refusal after the actual context changes must not fail solely because its code differs from the witness's earlier first-applicable code. Actual discovery/coverage attribution must use the observed Reference decision, not the original task label. Prior source-pinned scores remain reproducible under their original archives; no real research results were rescored.

### Remaining mandatory scope

This is **not** completion of the eight-domain objective. Education is still unimplemented. Initial historical/lifecycle distributions and multi-entity breadth in the new domains need expansion. Complete causal/coverage reports, 320/800 stratified certified releases, semantic leakage and private Held-out isolation, equal-weight evaluation, eight-domain Frozen Discovery/Reflection/static-only inheritance and exact selection Replay, remaining requested documents and final full regression are all still required. The earlier 1094-test full regression predates these additions and is not a final release claim.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m usami_dialogue verify-examples \
  benchmarks/vnext-0.1/examples/property-native-002
```

When later source changes cause a source-pin rejection, extract that example's archived runtime into a separate directory and run with that directory as `PYTHONPATH`. The isolated fixture replay log demonstrates this procedure without relying on the current checkout.


Audit scope caveat: the current world-audit implementation has complete content/anonymous-structure/ownership counts for the new domains, but its generic state-field list still needs expansion for the new condition/skill/latent-fact/event fields. Future simulator outcomes must be reported separately from current business lifecycle states. Initial ticket/case lifecycle variation is not yet a completed generator deliverable. These gaps remain part of the full objective and are not satisfied by the structural counts above.

## Education implementation in progress

The fourth dialogue domain now has independent education contract/credit rules,
actual financial and enrollment mutations, scoped tools, frozen student profiles,
academic review and actual attendance simulators, native goal composition and
shared certificate/agent interfaces. Registry support brings the independently
executable world generators to all eight domains. This is not completion of the
full release: education test/audit outcomes are recorded separately below once
commands terminate, and the complete release still requires all-domain task
coverage/datasets, split isolation, multi-domain Discovery/Skill evidence Replay,
remaining documentation and final repository regression.

The original education test attempt is preserved in
`verification/logs/education-initial-tests.log` (10 passed, 3 failed). Failures
were test setup defects: re-closing a completed session and supplying an actual
course/slot from a different frozen student scope. The expanded attempt is also
preserved (`education-expanded-tests-failure.log`, 15 passed, 2 failed): a
zero-budget student correctly could not pay for a top-up, and a Customer could
not authorize a case-specific quote before the case existed. Tests were repaired
without weakening either production check. All examples remain synthetic native
proofs or explicitly scripted transport; no real-model experiment has run.

### Verified education checkpoint

- Education unit/generator/agent tests: **17 passed in 30.13s** (exit 0).
- Complete vNext integration set, including census v2: **108 passed in 124.08s**
  (exit 0; `verification/logs/four-dialogue-domains-108.log`).
- Entire repository regression: **1153 passed in 591.97s** (exit 0;
  `verification/logs/education-checkpoint-full-1153.log`). This verifies this
  checkpoint, not later all-domain dataset/Discovery changes.
- All four dialogue generators: **400/400 invariant-valid worlds**, 100 per
  domain. Complete v2 state censuses and anonymous grouping are archived in
  `audits/dialogue-state-v2-001/`. Reference decision/task/depth coverage is
  deliberately still NOT RUN in a world-generation-only audit.
- `examples/education-native-001/`: actual certified refund plus duplicate
  refusal, transfer, extension and tutoring traces; no model calls.
- `verification/fixtures/education-agent-001/`: complete sorted-JSON/gzip
  production-runner journal with 5 Customer and 7 Service scripted completions,
  3 actual business operations and exact request Replay. The initial verification
  command supplied a dict where Replay requires a typed config; its failure and
  a subsequent tooling import typo are retained. The same original journal
  verifies with the correct typed configuration; no result was replaced.
- No original SaaS/CloudOps/Fulfillment/StayOps business source, rules, permissions,
  Oracle, frozen manifest or historical result was changed. No new real LLM
  training, inference experiment or Held-out evaluation was started.

The eighth executable generator is a milestone, not the full requested end state.
Mandatory remaining work includes meaningful initial lifecycle histories and
cross-entity breadth; six-category interaction dependency coverage; all-eight
native task certification/320-task Pilot and 800-task Core with honest strata;
semantic/task/world leakage and private Held-out generation; equal-weight metrics;
source-bound multi-domain Frozen Discovery/Reflection/StaticSkillGate/Skill
inheritance/selection Replay; complete requested documentation and final-release
verification. No model improvement or generalization conclusion follows from the
native proofs and scripted protocol fixtures above.


## Shared dependency and evaluation/split foundation

`usami_dialogue.dependencies` now adds two actual-event confirmation paths.
Cross-turn commitment uses an earlier real consent and a later actual Customer
revision, comparing native decisions before/after that event. Partial completion
uses a real legal business transition and a later attributable Oracle-correct
refusal. Neither an invented prior agreement nor a model's guessed refusal reason
can earn the corresponding dependency. Their trace-bound intervention/source IDs
and full paired states are verified independently. The diagnostic archive
`examples/dependency-six-kinds-001/` covers all six supported dependency kinds;
it is not a model performance result or the final task dataset.

`usami_benchmark.evaluation` implements frozen plans, exact task/result provenance,
conditional completion/refusal denominators, unknown values, role-separated costs
and eight equal domain weights (1/8), with tracks weighted 1/2. Missing domains
cannot be renormalized away. This is an aggregation layer; the complete native
eight-domain result adapters and release task descriptors remain pending.

`usami_benchmark.splits` independently regenerates worlds/signatures/source pins
and detects relabelling an already registered training world as test. Its finite
private Held-out generator preregisters all candidates, excludes known world/
structure overlaps, preserves shortfalls and refuses repository output paths.
A test exposed an existing `WorldPool.save` serialization defect (Pydantic
objects were passed to a plain JSON writer); the pool now writes explicit JSON
model dumps. The original failed test log remains preserved. This changes only
the new isolated pool infrastructure, not old native business runtimes.

Actual eight-domain world-split audit: **800 independently regenerated worlds**,
567 train / 122 validation / 111 public test-demo. Exact-world and anonymous-
structure overlaps are zero for every partition pair. Source-pinned report:
`audits/eight-domain-world-splits-001/report.json`. These are **800 worlds, not
800 certified tasks**. Task-semantic and task-combination leakage remain `NOT RUN`;
no official private Held-out evaluation was run.

Validation for these later changes: 20 dependency/education tests passed in
34.14s; 117 vNext integration tests passed in 133.65s; 25 split/evaluation/generator
tests passed in 2.11s. The 1153-test full run predates the added dependency,
evaluation and split changes; a final full repository regression remains required.
All original selection flags and the existing Service/Designer training mechanisms
remain unchanged. No new model inference, Reflection or research experiment ran.

The final combined vNext set at the dependency/evaluation/split checkpoint passed
**122 tests in 132.27s** (exit 0). The 100 dependency proofs also verified from an
isolated extracted source snapshot with exit 0. Both logs are preserved.

New vNext StayOps defaults now select the already-audited structural-v2 /
stayops-0.2.0 registration. Explicit structural-v1 / stayops-0.1.0 remains
supported, and the original `usami_stayops` runtime was not edited. This avoids
routing new default worlds through the known earlier Reference turnover defect.
The focused default/historical generator/split suite passed **22 tests in 9.13s**;
the 122-test run precedes this small default-routing change. Historical manifests
and source-pinned archives keep their original semantics and must use the original
runtime when full source pinning is required.

## Native lifecycle v2 verification in progress

An opt-in four-domain lifecycle generator now uses actual native customer,
business and external-event transitions; explicit old generators remain intact.
Its compact historical journals replay through the actual runtime, Oracle and
Replay. Source-private facts/preferences are generated before episode freezing;
Designer and task goals never participate. Initial tests: **12 passed in 11.06s**.
The 400-world audit and a new current-tree full regression have been started.
They must be checked by their live execution handles/logs; this ledger does not
claim they passed or remain alive. No Python source should change while these
source-pinned checks are running. Native dialogue defaults remain v1 during this
opt-in audit, and complete historical breadth is not established by the smoke.

### Lifecycle v2 audit completed

The source-pinned run completed with exit 0: **400/400 valid worlds**, 100 per
dialogue domain, each with 100 anonymous structures. Ownership topologies were
99 / 96 / 100 / 99 (After-sales / Airline / Property / Education). Actual
historical sessions executed 1460 business operations and 961 external events,
all through native Oracle/Replay; this is simulator execution, not LLM cost.
Raw reports and full compact journals: `audits/dialogue-lifecycle-v2-001/`.

Coverage is explicitly shallow in several places: the 100 Property worlds have
2 actual bookings, and Education has only 3 irreversible quote commitments.
After-sales has 26 settlements; Airline 6 commitments. All other operations and
zero-business sessions remain visible in the reports. This is useful initial
history variation, not proof of complete deep lifecycle/task-boundary coverage.
The opt-in v2 distributions are therefore not silently substituted for existing
native v1 generators. The final task release must choose and freeze its precise
source distributions and report these gaps.

The current-tree full regression completed with exit 0: **1178 passed in 647.09s**,
log `verification/logs/lifecycle-foundation-full-1178.log`. This result precedes
subsequent common native-task adapter work. No model call or evolution experiment
was run. Historical source snapshots and all previous logs remain unchanged.

## Unified task execution checkpoint (2026-10-17; release incomplete)

The common `usami_benchmark.tasks` compiler now dispatches all eight domains to
native Reference proof construction against independently frozen worlds. Tool
track certificates retain their actual search bounds, alternative witnesses,
frame constraints and rejection proofs. Dialogue tasks use native customer
profiles and business composers; proposed tasks cannot specify an Oracle answer
or replace private facts. Pure-refusal reasons are discovered by bounded native
Reference search, with protected business state preserved.

`usami_benchmark.episodes` runs the original tool-domain Service runtimes and the
four native dialogue Customer/Service runtimes through one configuration and
journal interface. Raw model requests, tool results, actual engine events,
Oracle results, role-separated costs and exact offline model Replay remain
available. The public rulebook contains static business policy only. Private
proofs are not injected into agent messages. Native interface hashes can be
pinned in frozen evaluation plans. Immutable source-pinned archives and the
`python -m usami_benchmark` prove/verify-task/replay/verify-archive/run commands
are implemented; online execution requires explicit opt-in and a call ceiling.

These are engineering interfaces, not evidence of model improvement. New tests
use native Reference execution and explicitly scripted model transport fixtures;
no real model execution or evolution experiment has been run. Earlier full-suite
1178-pass evidence predates this checkpoint and is not a final release regression.
Pilot320/Core800 certified dataset generation, task-semantic leakage and coverage
audits, and complete eight-domain Frozen Discovery/Skill evolution integration
remain unfinished. Historical native business rules and archived experiments
have not been rewritten. This checkpoint must not be presented as the completed
eight-domain benchmark release.

Checkpoint verification: the seven compiler/tool-agent/public-policy/episode/
archive/evaluation test modules completed with exit 0, **64 passed in 168.79s**.
Raw log: `verification/logs/unified-task-execution-checkpoint-64.log`. The earlier
failed development logs are retained verbatim, including their output whitespace.

## Certificate-backed task audit checkpoint (2026-10-18; release incomplete)

The new `usami_benchmark.task_audit` rebuilds actual native certificates before
deriving coverage. Every asserted capability retains native witness or
trace-bound dependency evidence. It records real allowed/refused Reference
boundaries, objective completion/refusal/mixed subgoals, write-omission causal
edges and different certified operation mechanisms. Caller-written family names,
display labels and message paraphrases do not establish new task structures.

Rooted relation colour refinement supplies ID-neutral task grouping, while a
separate coarse decision grouping reports shared operations, actual Reference
codes and dependency types. Pairwise partition reports disclose exact-world,
world-structure, task-group, decision-group and family overlaps. Re-registering a
training world as test is detected. A single-partition sample explicitly has no
split-isolation conclusion. Grouping is conservative and is not a proof of graph
isomorphism, semantic equivalence or independent scientific novelty.

The new opt-in `episodes.certified_evaluation_result()` derives strata from this
native evidence and performs full episode/model/score/cost Replay. Its caller
cannot supply capability/difficulty labels. The original externally audited
descriptor adapter and all historical archives remain available unchanged.
Neither path modifies training selection or invokes models during Replay.

The source-pinned engineering artifact
`audits/tasks-eight-domain-001/` contains one actual certified task per domain,
including four genuine dialogue completion-plus-refusal tasks. Exact generator
specifications, selected ordinary requests, native construction budgets and
source snapshots were saved before proof execution. The archive and coverage
report independently verified with exit 0. It is a hand-selected engineering
sample, not a random cohort, 320-task Pilot, 800-task Core or model experiment.
Each domain's Pilot quantity shortfall is 39; all eight cases use train worlds.
Full task split evaluation is therefore not claimed.

The eight-case report has verified identity/request, frame, refusal, mixed,
information, preference and consent evidence. It has no certified cells yet for
time, capacity, finance or entity-isolation boundaries; these remain
`NOT_AUDITED`. Its selected cases do not exercise causal alternatives or
cross-turn commitment revisions. Those proofs are tested elsewhere, but are not
silently added to this sample's matrix. Tool examples are basic one-write tasks;
the four dialogue examples remain structurally unclassified because long
transcripts do not substitute for necessary business-step proofs.

Initial audit tests stopped before any coverage result because a test fixture
used the ordinary After-sales family name with the mixed-task flag. The native
compiler correctly rejected it; the fixture was corrected without weakening the
check. The original log remains `verification/logs/task-coverage-initial.log`.
The repaired first audit suite passed **11 tests in 48.32s** (exit 0), including
all eight native domains, sorted-JSON verification, re-sealed tamper rejection,
ID/name/paraphrase grouping, split relabelling and immutable offline CLI behavior.
Further tests and current-tree full regression were started subsequently; only
their eventual terminal logs can establish their results.

The expanded audit/descriptor set subsequently passed **14 tests in 51.41s**,
exit 0 (`verification/logs/task-coverage-expanded-14.log`). Added checks cover
independently valid causal alternatives, separate raw/group shortfalls, unknown
single-partition isolation, private Held-out output and evidence-derived episode
strata. The eight native tasks and complete coverage report also verified under
an isolated extracted source runtime, exit 0
(`verification/logs/task-coverage-isolated-replay.log`). This is actual Oracle,
business/process Replay and audit reconstruction, with zero model requests.

The current-tree full repository regression completed with exit 0:
**1250 passed in 837.08s (13:57)**. Exact command:
`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -x`.
Raw output: `verification/logs/task-coverage-current-full.log`. Source files were
unchanged from the checkpoint archive throughout this run; the archive's entire
runtime source set and hashes were checked again afterward. Terminal commands,
exit codes and raw-log hashes are retained in
`verification/task-coverage-checkpoint-verification.json`. This validates the
task-audit checkpoint, not future release code or the unfinished dataset and
Discovery requirements.

No original native World/Engine/Reference/Oracle or permission code, historical
manifest, selected Skill, dynamic selection flag or existing experiment result
was changed. Real-model execution, Reflection, learning gain and private
Held-out evaluation remain `NOT RUN`. Mandatory remaining scope includes full
boundary mappings and necessary difficulty proofs, meaningful stratified
Pilot/Core releases and their leakage audits, eight-domain Frozen Discovery and
source-bound dual Skill inheritance/selection Replay, ablations and final release
verification. This checkpoint does not complete the active eight-domain goal.

## Frozen candidate banks and strict stratified sampling (in progress)

`usami_benchmark.candidates` now enumerates finite native existing-entity choices
in all eight domains. It is evaluator-side infrastructure, not autonomous
Designer behavior. It cannot edit worlds, request a task-conditioned generator,
invent Customer facts, supply a Reference verdict, or replace the Service policy.
CloudOps also includes a joint existing-stack goal whose operation order is
searched by native Reference. These catalogues are lower bounds on task space;
additional causal/mixed families remain mandatory for the complete release.

`usami_benchmark.datasets` freezes independent world specifications, complete
candidate schedules and proof budgets before certification. Every successful
native record and exact rejected construction is retained. Balanced selection
uses certificate-derived difficulty/outcome, excludes repeated rooted task
groups and records real per-domain/stratum shortfalls. Unclassified dialogue
tasks cannot fill a difficulty quota. Artifacts remain `CANDIDATE_BANK` even if
all numerical quotas are filled; this is not the final release-quality gate.

The verifier regenerates worlds, enumerations, successful certificates and failed
attempts; checks exact record/attempt file sets; and recomputes selection and
coverage. Known training worlds/anonymous structures cannot be relabelled as
independent evaluation. The applicable known-training registry must be supplied
by the caller; an empty registry is not proof of historical independence.
Private Held-out output is forbidden inside the public repository.

Executed foundation tests: **14 passed in 17.17s**, exit 0. Following stricter
quota/hash validation, the dataset/audit/compiler/archive integration set passed
**40 tests in 94.81s**, exit 0. Logs:
`verification/logs/dataset-foundation-initial.log` and
`verification/logs/dataset-task-integration.log`.

`manifests/native-candidate-bank-001.json` registers eight independently generated
train worlds (seed 314159), 16 candidates per world and equal Pilot target
quotas, with native search limited to 128 nodes / depth 12 / one witness. This is
a finite candidate-bank construction, not a scored experiment or completed
320-task Pilot. The build and current-tree full regression have been started;
their terminal handles/logs must be inspected before any success claim. No
source code should change while these source-pinned checks are live. No model
requests, Skill Reflection or Held-out execution were started.

The bank build terminated with exit 0: **128 attempted candidates, 83 native
certificates, 45 retained construction rejections and 65 selected quota slots**.
`quota_complete` is false. All medium/complex strata remain empty, and After-sales
has no selected slots because its certified dialogue tasks are unclassified and
its current catalogue lacks a standalone refusal family. This is a concrete
coverage gap, not a reason to lower difficulty thresholds or count duplicated
questions. Additional native causal families and dialogue difficulty proofs are
required before more seeds can establish the requested balanced release.

Complete dataset verification (including rebuilding all 45 rejected candidates)
and the current-tree full regression remain live and unclaimed at this point.
Their actual tool handles/logs must be polled; do not restart a job based only on
a delayed observation or log file. The candidate bank is not a 320-task Pilot,
800-task Core or independent Held-out cohort.

Terminal verification subsequently completed: all **128 candidate outcomes**
(83 certificates and 45 construction rejections), the complete frozen pool,
selection and coverage independently rebuilt with exit 0. Raw log:
`verification/logs/native-candidate-bank-001-verify.log`; source-linked receipt:
`candidate-banks/native-001/verification.json`.

The current-tree full regression completed with exit 0: **1264 passed in
868.24s (14:28)**, command `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest
-q -x`, raw log `verification/logs/dataset-foundation-full.log`. The 268 runtime
source pins match the tree validated by this run. Exact commands, exit codes and
raw log hashes are in `verification/dataset-foundation-verification.json`.
This completes the candidate-bank infrastructure checkpoint, not the full
eight-domain Benchmark objective or the missing causal/difficulty release strata.
