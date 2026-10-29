# Reproducing native worlds, proofs and execution

All commands below run from the repository root using the project's Python
environment. Native proof construction and Replay require no network or API key.
Recorded/scripted model transport exercises real tools and native state, but is
not evidence of a model's business competence.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_vnext*.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m usami_benchmark.audit \
  --domains aftersales airline property education --count 100 --start-seed 10000 \
  --root /tmp/usami-new-dialogue-audit
```

Audit output directories must be new. Generation is pinned by domain,
generator/rules versions, seed and configuration; task selection cannot edit a
registered world. Full world content, anonymous grouping, source hashes and
compressed pool integrity accompany each audit. A world census does not prove
certified-task coverage.

## Source-pinned native examples

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m usami_dialogue verify-examples \
  benchmarks/vnext-0.1/examples/education-native-001
```

This command requires the exact archived runtime. Later source additions or
changes intentionally cause a source-pin failure. Do not edit a manifest or
rescore the old record to make it pass. Extract that example's
`source-snapshot.zip` into a new empty directory, verify its repository-relative
paths, then run the same command with the extracted directory as working
directory and `PYTHONPATH`, using an absolute artifact path. The archived source
includes the actual native domains, common dialogue infrastructure and their
local Python dependencies; use the project's existing installed dependencies.

Native task verification recomputes independent world generation, task scopes,
Reference/actual transitions, Oracle, objective goals, frames, interaction
counterfactuals and Replay. The private witness is host-side only, and is not the
path that a production Service must follow.

## Scripted agent journals

`verification/fixtures/education-agent-001/journal.json.gz` contains the initial
world, typed task-agent configuration, complete model requests/responses and
actual execution. Decode it with sorted JSON, validate `TaskRunConfig` and
`AgentExecution`, then call `usami_dialogue.agents.replay` with the native domain
adapter. Replay reconstructs exact model requests and actual tool execution,
without contacting a model. The source snapshot procedure above also applies.
The archived isolated-replay log verifies this on the exact recorded runtime.
Missing usage in fixtures remains unknown, even when stored primitive counters
are zero; see `usage_complete: false`. It is not a measured zero-token run.

## Compatibility and verification scope

Historical four-domain runtimes, manifests and experimental records remain
unchanged. New versions occupy separate packages/registrations. Native dialogue
model journals use v2 canonical prompt serialization; v1 journals require their
original archived runtime. World audit v2 separates current, latent and future
states; v1 archives retain their own schema and sources.

The education checkpoint's 1153-test full regression is recorded with exit 0.
Later changes require their own verification; this number is not a final-release
claim. Failed test/verification attempts remain in the logs with their actual
errors and are distinguished from later successful runs. No real LLM run or
Held-out evaluation is authorized or represented by these fixtures.

## Unified task coverage audit

The common native task archive can be audited without model credentials:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m usami_benchmark audit-tasks \
  SOURCE_PINNED_ARCHIVE --output NEW_TASK_AUDIT --per-domain-target 40
```

The input archive is read-only. A new audit directory retains its source
snapshot, the exact input manifest/task commitments, actual coverage report and
verification or failure. `coverage.json` roundtrips through sorted JSON and is
independently recomputed. A single-partition audit reports split isolation as
unknown, not as a vacuous pass. Raw task count shortfalls and rooted structural
group shortfalls are separate.

To rebuild the selected eight-domain engineering cases (not a Pilot dataset):

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python \
  experiments/vast_coevo/examples/task_coverage_checkpoint.py \
  --output NEW_EIGHT_DOMAIN_COVERAGE_CHECKPOINT
```

The script registers exact independent world specifications, ordinary task
selections and finite proof budgets before execution. It does not ask any model
to answer and it never edits a generated initial world. The four dialogue cases
contain real completion and correct-refusal subgoals and actual information and
consent dependencies. No record from this hand-selected engineering sample is
an independent Held-out example or evidence of improved model performance.

## Native candidate-bank rebuild

`build-dataset` and `verify-dataset` extend the same offline CLI. A source-pinned
bank contains its exact `plan.json`, whole independent `world-pool.json`, full
`candidate-schedule.json`, every `attempts/*.json`, successful private
`records/*.json.gz`, `selection.json`, candidate coverage and manifest. Output
directories must be new; an aborted build retains a stage-specific failure and
all already completed records. No exception causes an automatic model retry or
replacement world.

Native candidate banks are not scored model datasets or completed Pilot/Core
releases. In particular, `quota_complete` does not establish coverage quality,
semantic independence, held-out isolation or actual model performance. Old source
snapshots remain the authoritative runtime for earlier artifacts after adding
new native modules.
