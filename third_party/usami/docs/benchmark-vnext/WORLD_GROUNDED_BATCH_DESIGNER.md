# World-Grounded Single-Shot Batch Designer v1

This protocol runs beside `frozen-world-designer-exploration-v4`; it does not
change the v4 request, Replay, task, or certificate formats. The new protocol is
`world-grounded-batch-designer-v1` and currently has a native Grounder only for
After-sales.

The current request/compiler contract is `world-grounded-contract-v2` and its
public catalogue is `world-grounded-domain-catalogue-v2`. The outer single-shot
and one-revision protocol names stay explicit. Catalogue vocabulary, native
field value types/enums, relation typing and the submission schema share the
same grammar definitions. Private-index edge labels are not public language
operators. `mixed` and `multi_object` describe compositions of atomic objectives
in an array; they are not accepted objective `kind` values.

Historical contract-v1 archives, including the original InferAI attempt, remain
unchanged. Reproduce them on source commit `cbb5da4`; current Replay refuses to
silently score those archives under contract-v2 semantics. Existing v4 Designer
requests and Replay are unchanged. The frozen-world index format is unchanged.

The Designer receives a compact, ID-free domain catalogue and proposes one
batch of abstract challenges. A deterministic local Grounder matches symbolic
constraints to independently regenerated frozen worlds. Every grounded
proposal is then sent through the existing `compile_task()` path, which uses
the native Reference, Engine, Oracle, certificate, and `verify_task()` Replay.
The model proposes what to test; it does not select concrete IDs, native
frames, answers, or witnesses. Grounding success and native certification are
reported separately. No Service execution happens in this pipeline.

The tool schema is inlined (no local `$defs`/`$ref` dependency), fixes the exact
requested batch size, and expresses field/predicate forms and atomic objective
forms. Forced `tool_choice` selects a function name; it does not establish that
an OpenAI-compatible provider enforces its arguments. Local validation remains
mandatory. Unknown enum values, wrong primitive types, unsupported operators,
missing fields and invalid relationship pairs are recorded as failures, never
silently translated into a different task. Time comparisons use timezone-aware
instants and explicitly handle absent delivery timestamps.

The private index is for evaluator-side matching only. Never attach
`grounding-index.json` to a Designer, Customer, or Service prompt. Catalogue
construction only accepts explicit `train` or `validation` partitions; it
rejects test/demo and held-out worlds.

## Build and inspect an index

```bash
python -m usami_benchmark.designer_batch build-index \
  --domain aftersales --seeds 0,1,2 --partition train \
  --output-dir /tmp/usami-batch-index
python -m usami_benchmark.designer_batch verify-index \
  --index /tmp/usami-batch-index/world-index.json
python -m usami_benchmark.designer_batch catalogue \
  --index /tmp/usami-batch-index/world-index.json --partition train \
  --output /tmp/usami-batch-index/catalogue.json
```

The index builder accepts a registered `WorldPool` and re-runs
`verified_world()` for each registration. The index commitment binds world
contents, source hashes, rules/generator versions, relationships, authenticated
profile scopes, and registered external events. It is private because it
contains the full frozen worlds and concrete IDs.

## Single-shot run and Replay

Configure the existing OpenAI-compatible client using
`USAMI_LLM_MODEL`, `USAMI_LLM_BASE_URL`, and either `USAMI_LLM_API_KEY` or the
existing Keychain variables. The default protocol performs one provider-call
attempt and never retries. It requests five candidates by default; the batch
can be set to 1–5. An optional, separately named one-revision protocol allows
at most two calls.

```bash
python -m usami_benchmark.designer_batch run \
  --index /tmp/usami-batch-index/world-index.json \
  --partition train --num-challenges 5 --online \
  --output-dir experiments/vast_coevo/results/world-grounded-batch-001
python -m usami_benchmark.designer_batch replay \
  --run-dir experiments/vast_coevo/results/world-grounded-batch-001
```

Offline transport/structured-output verification uses a response JSON file:

```bash
python -m usami_benchmark.designer_batch run \
  --index /tmp/usami-batch-index/world-index.json \
  --num-challenges 1 --scripted-response /path/to/openai-response.json \
  --output-dir /tmp/world-grounded-scripted
```

Scripted responses only test protocol behavior; they are not live-model
results. Runs archive the exact request and response, candidate attempts,
grounding sources, compilation outcomes, native certificates, usage, and Replay
status. Replay reconstructs the model request from the frozen catalogue and
protocol, reparses every raw response, reruns constraints and native proofs,
checks request/candidate/cost journals, and verifies each stored certificate.
Malformed and failed responses are retained; output directories are immutable.

Archive Replay also checks every exported file against the manifest, including
requests, raw responses, certificates and reports. It recomputes non-timing
metrics and costs from raw responses and deterministic candidate history.
Hashes detect inconsistency; they do not authenticate a manifest controlled by
an adversary. Existing output directories are rejected before a model call.
Compiler-internal errors and failed certificate Replay are distinguished from
invalid proposals and do not trigger another model call.

`COMPLETED` describes completion of the request, not success at constructing a
task. `certification_status` is `ALL_CERTIFIED`, `PARTIALLY_CERTIFIED` or
`NO_CERTIFIED_CHALLENGES`. The run/comparison CLI returns nonzero when no task is
certified. Invalid counts and duplicate IDs cannot produce certified tasks;
all raw entries remain in the record, including overfull batches.

## A/B/C proposal comparison

`experiments/vast_coevo/examples/compare_world_grounded_designer.py` compares:

* A: existing v4 Designer, run without exploration or revision so it uses one
  submission call;
* B: world-grounded batch Designer, using one batch call;
* C: a deterministic rule-combination sampling baseline, with zero LLM calls.

The strict call-budget comparison currently requires a one-world registered
pool so A and B see the same frozen world. The script records actual input and
output tokens; equal API-call count does not imply equal token cost. It
explicitly reports that token budgets are not matched in this mode. Use the
token-normalized measurements for cost accounting, not as a token-budget
controlled causal comparison. No task-generation certificate is evidence of
Service quality or of an evolutionary gain.

```bash
python -m usami_benchmark.designer_batch build-index \
  --domain aftersales --seeds 0 --partition train \
  --output-dir /tmp/usami-abc-index
python experiments/vast_coevo/examples/compare_world_grounded_designer.py \
  --index /tmp/usami-abc-index/world-index.json --online \
  --output-dir experiments/vast_coevo/results/world-grounded-abc-001
```

For an offline end-to-end comparison, pass both `--v4-response` and
`--batch-response` instead of `--online`. A successful comparison has zero
Service executions by design. It measures proposal, grounding, certification,
and local Reference cost only.

## Scope and limits

The index can preserve registered worlds for all eight domains, but this
protocol's symbolic schema and Grounder implement After-sales only. It currently
supports existing-line completion, created Case completion, correct `open_case`
refusal, mixed/multi-object goals, typed field constraints, fixed predicates,
same-identity constraints, and registered events. Other domains and native
operations return an explicit unsupported/scope result rather than a guessed
translation. Structural and semantic fingerprints omit concrete entity IDs.
Coverage fingerprints preserve objective/entity associations and relationship
direction, while ignoring alias spelling. They are coverage signatures, not
proofs of semantic equivalence.

Every objective line explicitly selects one Customer profile. Bound Customer,
order, line, evidence and event objects stay in that authenticated identity's
scope. Products must be referenced by bound lines. Information and preferences
refer to objective profiles. Inspecting an external-event alias does not
schedule an event: scheduling is requested only by explicit
`external_event_constraints` with `required=true`. `required=false` asserts
absence of that registered event, rather than removing a present event.

The deterministic rule baseline is an internal grammar-combination baseline,
not an official reproduction of another system. No performance advantage is
assumed. A real-model run must report invalid schemas, no matches, bounded
Reference failures, costs, and Replay results as observed.
