# Native dialogue task certification

Status: all four dialogue domains and the common eight-domain compiler/runner
are implemented. Pilot/Core datasets and complete multi-domain Discovery remain
active work. No real LLM research run has been performed for this upgrade.

## Reachability and scoring are different operations

`usami_aftersales.tasks.compose()` binds a declarative task to an existing frozen world and private Customer profile. It cannot change an order, stock, policy, evidence outcome, user fact or simulated external outcome. The public task goal accepts one or more actual terminal order-line states. The native frame contract limits changes to that line's case, quotes, settlement, return, approval and replacement stock. A task cannot grant arbitrary frame exemptions.

The bounded Reference search enumerates available native operations and pre-registered external events. It uses Customer-side process events for truthful disclosures and real quote confirmation. A search limit without a witness is reported as exhausted budget, not proof of impossibility. Search can return distinct legal remedies for one task.

`execute_witness()` executes each resulting plan using the actual dialogue session and independent native mutation implementation. `certify()` requires the objective goal, frame, Oracle and full offline Replay to agree. The witness is a private certificate, never a production Service script or required path. `score()` evaluates an Agent's actual trace using the same terminal predicates and independent judgments; legal extra queries and different valid paths are accepted.

## Frozen-world open composition

Task construction is no longer limited to the four dialogue domains' fixed
`compose()` families. `usami_benchmark.designer.designer_view()` gives a task
Designer the complete sealed world, public rules and the native operation and
scope contracts. For dialogue worlds it also enumerates existing Customer
profiles, their permitted facts/preferences, candidate intents and frame
contracts. A proposal can select multiple profiles only when they belong to the
same authenticated identity, and can combine multiple completion goals, exact
refusal probes, information/preference requirements, confirmations and
pre-registered external events. Track A already supports multiple native
capabilities and a `JointGoal`; the Designer view exposes its registered
operations and per-identity visible entities.

The model proposal enters `compile_task()` as `TaskSelection`. It cannot edit
the world, invent IDs, choose a Reference refusal code, provide an execution
witness or override native rules. Static scope checks run first; bounded native
Reference search then finds a witness, and the ordinary Engine, independent
Oracle, task certificate and exact Replay must agree. A bounded search that
finds no witness is unproven, not proof that the task is impossible. The
versioned dialogue payload is `open-dialogue-composition-v1` and is preserved in
the task hash. `design_and_compile()` makes one Designer call without retry and
retains its raw request, response, proposal, certificate and reported token
usage; `verify_designer_execution()` reconstructs the prompt, checks that the
proposal matches the raw function call, and replays the certified task.

This opens the task language; it does not establish that an LLM Designer
reliably proposes diverse or useful challenges. This change has no real-model
Designer run.

## Mixed task

A reproducible mixed After-sales task asks for a valid remedy and then requests another resolution of the same real order line. The first subgoal requires the actual refund/exchange/reship state. The second requires a Service refusal of the exact requested operation with the native `ALREADY_RESOLVED` reason. The original order and item exist; no invented ID is needed.

An Engine-blocked attempt is not a Service correct refusal. `DialogueSession.reply()` records a structured decision separately from native operations. The independent Oracle checks that the request was actually made, the operation remained unattempted, Reference disallows it, and the refusal reason matches. A refusal of an allowed operation is a false rejection. Other cases remain unattributed or wrong-reason decisions and cannot satisfy a refusal subgoal.

## Dialogue necessity

`usami_dialogue.dependencies.certify_dependencies()` first replays the actual trace. It evaluates actual pre-operation states under a closed list of evaluator-only interventions:

- Remove one previously disclosed frozen fact.
- Remove actual explicit consent.
- Select another permitted preference, revoking superseded consent and commitments.
- Compare actual states immediately before and after a recorded frozen external event.

A certificate is emitted only when the native Reference's allowed boolean changes. Merely changing the denial message is insufficient. It stores the actual source event, both full states, operation, intervention, decisions and hashes. Verification reconstructs the source and intervention, recomputes both Reference decisions, and rejects any extra business-state edit even when hashes are re-sealed.

These counterfactual copies never enter the World Pool or a Service session. They provide evaluator evidence of information dependency; they do not edit a Designer's challenge world.

Current tests demonstrate missing-information, preference, consent and real manager-approval dependencies. They also reject claiming a refund deadline as necessary when removing it does not affect the refund decision. The explicit cross-turn commitment and full eight-domain coverage reports are not yet complete.

## Evidence limits

A private witness proves that a task is executable under its stated rules. It does not prove a real model solves the task, asks good questions or learns across generations. Fixture-driven Agent protocol tests exercise actual Function Calling, business mutations, Oracle, budget handling and exact request Replay; they are not real LLM results. Dataset-scale certification, coverage/leakage reports and the Pilot/Core releases remain outstanding.


## Shared planner and source-complete artifacts

`usami_dialogue.planner` is the common bounded search/execution implementation. Domain-specific composers supply only real goals, exact scopes, allowable preferences, frames and refusal probes. A refusal is reachable only when the requested operation is actually disallowed for the declared reason; reaching a completion goal alone is insufficient for a mixed task.

`usami_dialogue.artifacts` saves actual native traces/certificates with independently frozen generator provenance, deterministic compressed content, exact runtime source file sets/hashes and a source archive. Offline verification checks generation, source completeness, manifest/case integrity, objective scoring and Replay. Native witnesses are explicitly labelled `model_experiments: NOT RUN`; re-labelling them as real LLM evidence is rejected.

`TaskRunConfig` uses a separate task-agent protocol. It requires the task to match a private certificate and verifies the witness before invoking either model. Domain-derived Customer briefs contain the selected business need but no hidden refusal code, expected outcome or witness. The complete private certificate remains in the host journal for reproducibility, never the model request. Base v1 configurations remain available for task-independent interactions.


## Current-state refusal reasons versus coverage labels

`dialogue-task-score-v2` requires an exact requested operation and an Oracle-confirmed correct refusal at the actual pre-decision state. A witness's intended denial code remains an audit label. It is not substituted for the independent current Reference result. Scores retain both `intended_refusal_boundaries` and `observed_refusal_reasons`, so a different accurate reason cannot be falsely attributed to the intended boundary or counted as a new discovery of that rule. Wrong reasons, false refusals and Engine-blocked attempts still cannot satisfy a correct-refusal subgoal.


## Actual cross-turn and partial-completion sources

Two additional proof paths now complete the six-category schema.
`cross_turn_commitment` pairs the actual pre/post states of a Customer event
that advances the preference revision and cancels a recorded prior consent. The
operation must be exactly that earlier consent's operation, and the prior turn
ID is rechecked. The independent Reference must change its decision; merely
adding a turn does not count.

`partial_completion` pairs a real Oracle-legal business transition with a later
actual attributable correct-refusal event for an exact request. The request must
be allowed before the transition and forbidden afterward for the observed native
reason. The full trace must pass Oracle. No blocked tool attempt, guessed reason,
failed task construction or unattributed refusal can substitute.

Both proofs retain actual source IDs and full state pairs; forged links and extra
state edits fail verification. `examples/dependency-six-kinds-001` archives three
native traces whose verified proofs cover all six kinds. This demonstrates these
process properties, not full task-scale coverage or real-model skill.

## Finite candidate banks and balanced dataset selection

`usami_benchmark.datasets.DatasetPlan` freezes the complete independent world
schedule, per-world candidate ceiling, construction budget and four equal-domain
stratum quotas. Default Pilot targets are 8 basic, 12 medium, 8 complex and 12
pure-refusal tasks per domain. A mixed task is a cross attribute and must have a
certified completion difficulty to enter a completion stratum. Unclassified
dialogue tasks remain certified candidates, but cannot manufacture difficulty
quota completion. These definitions do not change any business rule or score.

The builder first generates and seals the entire World Pool. Only then does
`usami_benchmark.candidates` enumerate finite existing-entity choices. It samples
no replacement worlds after seeing a task. Candidate enumeration is deterministic
evaluator infrastructure, not autonomous Designer discovery or a Service script.
The complete bounded candidate schedule is saved before native certification.
Both successful certificates and exact construction rejections are retained as
individual immutable records. A bounded failed proof remains unproven, not an
impossibility proof or a discovered Agent failure.

Selection uses actual certified outcome/difficulty and one task per rooted
relation group in each domain. Message or family aliases cannot fill extra slots.
The exact selected hashes, excluded hashes/reasons and each domain/stratum
shortfall are saved. Quota completion is distinct from a high-quality completed
Benchmark release: these artifacts are labelled `CANDIDATE_BANK`. They still
require the complete coverage, difficulty, split and release-quality audit.

`verify_dataset()` independently regenerates worlds and the finite schedule,
reconstructs every successful proof and every rejected construction, compares
all saved records, recomputes coverage and selection, and validates runtime source
pins. A resealed summary cannot hide a failed attempt, fabricate a selected task,
complete a missing stratum or relabel native proof as a real-model run. Replay is
offline; no model credentials are accessed.

```sh
.venv/bin/python -m usami_benchmark build-dataset REGISTERED_PLAN.json \
  --output NEW_CANDIDATE_BANK
.venv/bin/python -m usami_benchmark verify-dataset NEW_CANDIDATE_BANK
```

The plan can register known training world hashes and anonymous structures.
Reusing any of them as independent validation/test/Held-out fails before task
construction. The caller must supply the complete applicable training registry;
an empty list is not proof of historical independence. Private Held-out banks
cannot be stored in this public repository. No private Held-out evaluation or
real-model experiment is performed by the builder.
