# After-sales dialogue protocol v3

Protocol v3 is an opt-in After-sales interaction boundary. The v2 runner,
request schemas, serialized execution hashes, business Engine, Reference,
Oracle, and native replay remain in place. V3 calls the same
`DialogueSession`; only the LLM-facing expression of Customer authorization
and Service quote presentation changes.

## Interaction contract

Customer returns a natural message, factual disclosures, permitted preferences,
IDs from the current turn's `available_actions`, an `offer_response` enum, and a
finish flag. The host resolves those IDs to the exact native operations. IDs bind
to the frozen world, selected private profile, turn, and operation, then expire
when the turn or state changes. Unknown, stale, duplicate, or out-of-profile IDs
cause a recorded protocol error before business state changes.

```json
{
  "message": "I want a refund.",
  "disclosures": {},
  "preferences": {"resolution": "refund"},
  "authorized_action_ids": ["act_<turn-scoped-id>"],
  "offer_response": "none",
  "finish": false
}
```

An `accept` response grants only the exact operation in the current offer. It
does not accept an offer if the Customer changes their preference in the same
turn. A `reject` must refer to a current offer. Candidate actions express what
the Customer could explicitly request; they do not disclose the Oracle's
expected verdict.

Service presents a quote using a small reference:

```json
{"message": "Please review these terms.", "offer_ref": {
  "kind": "commit_resolution", "quote_id": "quote_..."
}}
```

Ordinary replies set `offer_ref` to `null`. The host resolves the reference to a
native operation only when the quote exists, belongs to the selected order and
line, and has a current Customer revision. A pre-settlement quote must still be
offered and unexpired. A `release_refund` reference must bind to the already
committed refund agreement. The existing `offer_for`, Customer consent reducer,
Engine, and Oracle continue to determine consent and business correctness.

The distinction between the two settlement operations remains explicit:

- `commit_resolution` presents and settles a current quote; if return is
  required, it starts the return process.
- `release_refund` presents the previously agreed refund again and still needs
  a new, exact Customer confirmation after actual return receipt.

The harness never creates a quote, picks a resolution, grants all candidates,
or authorizes the Service. All mapped Customer writes are passed through the
existing private-profile check and recorded as exact single-use grants.

## Frozen task, evidence, and replay

`TaskRunConfigV3` requires one selected profile, a hash-checked task, its native
certificate, and witness. The witness stays in the private run registration; it
is not included in either agent's messages. Model requests, responses, exact
resolved grants, native business events, Oracle results, costs, and the applied
Skill context are content-bound in `AgentExecutionV3`. `replay_v3` uses only the
recorded completions and reruns the exact request/tool sequence offline.

Existing v2 records remain on `usami_dialogue.agents.run/replay`. V2 replay
accepts the current public-policy renderer and the repository's registered
historical public-policy catalogue renderer; it rejects unknown recorded system
prompts. This supports older exact journals without making an archived prompt
an arbitrary source of instructions.

## Local tests and live diagnostic

Run the protocol and related v2 regression tests without contacting a model:

```bash
.venv/bin/python -m pytest \
  tests/test_vnext_dialogue_agents_v3.py \
  tests/test_vnext_dialogue_agents.py \
  tests/test_vnext_aftersales.py -q
```

Inspect the registered frozen task and verify its certificate without creating
a model client or looking up credentials:

```bash
.venv/bin/python experiments/vast_coevo/examples/after_sales_service_v3_diagnostic.py
```

The opt-in command below sends only Service requests to the configured
OpenAI-compatible endpoint. Customer decisions are made by a deterministic
local compliance driver; they are not model calls and do not consult the
Reference witness. It refuses to overwrite an existing run directory and stores
the complete request/response record, native trace, Oracle result, and offline
Replay. Maximum remote Service calls default to 20; no provider retries are
performed.

```bash
USAMI_LLM_MODEL='provider/model' \
USAMI_LLM_BASE_URL='https://gateway.example/v1' \
.venv/bin/python experiments/vast_coevo/examples/after_sales_service_v3_diagnostic.py \
  --execute-live-service --mode sandbox
```

Configure `USAMI_LLM_API_KEY` in the shell through your existing secret
manager before running; do not put a credential in this command or in the
repository.

This implementation task has not invoked that online command. The live result
is a diagnostic episode, not a Skill effectiveness or generalization claim.

## Frozen Customer preflight and local decision audit

New After-sales registrations may use the opt-in
`own-order-after-sales-progress-v1` Customer view. It adds only the authenticated
Customer's own case, quote, and return status fields. Evaluator notices, quote
hashes, and receipt notices remain hidden. Archived runs keep their original
`legacy-v1` or `own-order-line-status-v1` projection.

The deterministic Customer fixture treats a grant as a one-turn permission,
not proof that the Service executed the operation. It checks the visible case,
quote, return, and line state before advancing; if an authorized action remains
visibly incomplete, it permits one retry and then stops authorizing it. For
exchange/reship goals it discloses the frozen Customer's actual delivery
deadline. A registered information requirement must still be grounded in that
Customer profile.

`open-dialogue-composition-v4` adds public follow-up messages for requests that
must happen after the primary goal. Each refusal objective must have one such
message and a matching harness-only `requested_action`. The compiler validates
that the operation is in the selected Customer's native action scope. Only the
message is copied into `UserScenario`; the action label is never sent to the
Customer LLM. The v4 compiler also requires delivery-deadline information for
exchange/reship completion goals. Older task protocol versions remain valid for
historical records.

Before explicit Customer-mode execution, `customer_interaction_preflight` checks
that the frozen profile contains the required facts and that v4 follow-ups are
present in the public scenario. This is a consistency check, not a replacement
for the native Reference certificate or an end-to-end scripted interaction.
The diagnostic and paired-run outputs save it as
`customer-interaction-preflight.json`.

`usami_dialogue.decision_audit.analyze_execution` adds a replay-bound offline
report that keeps positive goal completion, strict task success, denied
attempts, successful sandbox violations, incorrect refusals, protocol errors,
and per-role costs separate. It lists observed blockers and stable mechanism
IDs, but marks every causal root unverified unless a separate counterfactual
test establishes it. `compare_matched_audits` refuses different task/world
hashes and reports unvisited decision opportunities separately. These reports
are diagnostic only: they do not alter Oracle results, task scores, Skill
selection, or Dynamic Gates.
