# Frozen-world Designer protocol v4

Track B's default exploration protocol is now
`frozen-world-designer-exploration-v4`. It retains v3's bounded read-only
exploration, native proposal checks, Reference proof, and exact Replay, and adds
goals for objects created during native execution.

## Created-object goal

A Designer can describe a result such as “open one case for this selected
customer's wrong-item report and leave it in `open` status” without guessing a
case ID that does not exist yet. The proposal supplies a selected profile, a
native collection, immutable selector fields grounded in that profile, and a
public target field with accepted values. For After-sales, the existing
`cases.line_id` create-enabled frame binds the goal to a line; the selector can
also include the profile's observed issue.

The compiler rejects selectors outside the selected profile, generated IDs,
private fields, targets without a native `allow_create` frame, and selectors
that already match an object in the frozen world. The Reference requires exactly
one matching object in the final state. It searches only customer-visible
native intents and registered external events; the Engine executes the witness,
and Oracle plus Replay independently verify the resulting trace.

The serialized task uses `certified-dialogue-task-v2` and
`open-dialogue-composition-v3` only when it contains a created-object goal.
Existing-object and refusal-only tasks keep their prior task/composition
protocols. Designer exploration v2 and v3 retain their prior prompt, schema,
query results, and replay behavior. V4's read-profile query additionally
returns that profile's native frame contract so the Designer can ground a
creation target without receiving the whole world or a Reference answer.
V4 does not attach a harness-side output token cap to Designer requests;
provider-side limits still apply. Historical v2/v3 request construction stays
unchanged for replay compatibility.

## Validation

Run the focused offline checks with:

```bash
./.venv/bin/python -m pytest -q \
  tests/test_vnext_open_compositions.py \
  tests/test_designer_exploration.py
```

These tests establish proposal legality, native reachability, certificate,
Oracle, and Replay behavior. A scripted response in a test is protocol
verification, not evidence of a real model's proposal quality. Real Designer
success, Customer/Service execution, and learning benefit require separately
recorded experiments.

An offline Codex subagent was also given the archived Designer request context
and asked to submit a task. Its first created-object proposal omitted the native
`line_id` frame anchor and was rejected; after a read-only profile lookup it
revised the proposal, which the native compiler, Reference, Oracle, and Replay
accepted. The auditable proposal and certificate are in
`experiments/vast_coevo/results/designer-subagent-v4-001/`. This is offline
protocol debugging evidence, not a live LiteLLM result.

## Independent batch protocol

`world-grounded-batch-designer-v1` is an additional task-generation protocol,
not a replacement for v4. It sends an ID-free domain catalogue once, receives a
batch of abstract challenges, deterministically binds them to a private index
of registered frozen worlds, then invokes the same native compiler and
Reference/Engine/Oracle/Replay path. Its current Grounder supports After-sales
only. See [World-Grounded Single-Shot Batch Designer](WORLD_GROUNDED_BATCH_DESIGNER.md)
for the index, single-shot run, Replay, and A/B/C comparison commands.

New batch runs use `world-grounded-contract-v2`, with a schema-derived catalogue
and exact, typed submission vocabulary. Historical batch-v1 archives reproduce
on commit `cbb5da4`. See the [offline contract audit](WORLD_GROUNDED_BATCH_AUDIT.md)
for fixes, regression coverage and the boundary before another live model run.
