# Coverage taxonomy and evidence requirements

The versioned evaluation schema declares these fifteen independent capability
keys. A business family is domain-specific; it must not be inferred solely from
how many tools a Service calls.

| Key | Meaning | Evidence required for a tested-coverage claim |
| --- | --- | --- |
| identity_authorization | Authenticated identity, roles and exact writes | Native decision and independent trusted request audit |
| entity_isolation | Multiple entities and tenant boundaries | Actual scoped requests, protected frame and Reference result |
| state_preconditions | Lifecycle/action prerequisites | Actual prestate and rule decision |
| capacity_resources | Finite seats, stock, quota or schedules | Resource conservation/boundary certificate |
| time_deadlines | Expiry, windows and temporal feasibility | Controlled clock and native boundary decisions |
| financial_constraints | Payments, fees, refunds and balances | Actual ledger transitions and native amount checks |
| causal_dependencies | Necessary prerequisite transitions | Executable omission/causal evidence; not raw path length |
| alternative_paths | Different legal solutions to a goal | At least two independently valid paths with meaningful differences |
| side_effect_consistency | Protected unrelated state and invariants | Explicit frame plus independent full transition audit |
| refusal_decisions | Correct and false refusals | Exact requested operation and actual Reference result |
| information_resolution | Missing facts affect valid action | Replay-bound removal of a truly disclosed necessary fact |
| preference_negotiation | Choice changes the correct action | A permitted alternate preference changes native decision |
| explicit_consent | Current, exact irreversible authorization | Real offer/consent plus consent-removal counterfactual |
| cross_turn_commitments | Later action respects current agreement | Earlier real commitment and later actual revision/decision evidence |
| mixed_completion_refusal | Legal work and accurate partial refusal | Objective subgoals, actual transition and attributable correct refusal |

The matrix axes are `Capability × Business Family × Difficulty × Task Outcome`.
Outcomes are completion, refusal and mixed; mixed is not counted as a completely
successful task when only its legitimate part succeeds. Difficulty must come
from certified business structure and necessary dependencies. A bounded witness
is an upper bound on a legal path, not automatically the global minimum proof
length. The aggregator allows `unclassified` until a structural classifier has
sufficient evidence; it must not use model failure rate to assign difficulty.

## Current implementation versus final coverage

Native dialogue certificates now support all six dependency categories: missing
information, preference resolution, explicit consent, actual external update,
actual cross-turn revision and partial completion/refusal. Only a changed
independent Reference decision establishes a dependency. The last two require
actual source events: a hypothetical prior agreement or invented refusal cannot
be substituted. A successful basic example does not establish every capability
in its domain. A descriptor annotation is not coverage proof.

The 100-world-per-domain structural audits describe world diversity, not task
coverage. Anonymous world counts cannot be converted into numbers of independent
tests. The full eight-domain matrix, per-cell certified quantities, 40/100-task
per-domain sampling and split leakage reports are still being implemented; they
must be published with actual evidence and honest gaps. This document and the
fifteen schema keys do not claim those release artifacts already exist.

A genuinely inapplicable capability requires a domain-rule explanation labelled
`N/A`. Missing evidence is `NOT AUDITED` or `NOT RUN`, rather than `N/A` or zero.
Existing tool-centric domains already support Customer/Service interaction; their
coverage cannot be dismissed merely because they lack the new private-state
protocol. Their claimed dialogue necessities require equivalent actual proof.

## Native task audit implementation

`usami_benchmark.task_audit.describe_task()` independently reconstructs the
eight-domain native task certificate before deriving a descriptor. It reports
the actual completion/refusal/mixed outcome, exact executed Reference boundaries,
protected-frame evidence, real write-omission edges, distinct solution mechanisms
and trace-bound dialogue dependency certificates. Every positive capability cell
retains its source certificate/trace/event or witness index. The descriptor does
not accept caller-supplied capability labels.

The initial auditor conservatively certifies scoped identity/request checks,
protected frames, causal preconditions proved by omissions, actual correct
refusals, mixed goals, information, preference, consent and commitment dependencies,
and different proven operation mechanisms. It does not infer complete time,
financial, capacity or entity-isolation coverage from an operation name. Those
cells remain `NOT_AUDITED` until dedicated boundary evidence is mapped. A zero
cell count is a gap in this auditor's evidence, not a claim that the business
domain lacks the capability.

Tool difficulty uses `omission-certified-tool-depth-v1`: basic has a necessary
one-write witness; medium has multiple necessary writes without a two-edge
causal chain; complex requires a proved causal depth of at least two. This is a
structural classifier within the finite certified capability set, not global
minimal planning difficulty. Dialogue and pure refusal tasks remain unclassified
by this first classifier. Dialogue proof lengths and verified dependency kinds
are reported separately; a long transcript cannot manufacture complexity.

`audit_tasks()` emits the full capability/family/difficulty/outcome matrix with
per-domain counts, actual world/group counts, duplicate fractions and an explicit
equal-domain quantity shortfall. It does not fill absent strata, turn N/A into
success, or certify the 40/100-task targets solely from task content hashes.
`verify_audit()` rebuilds the native evidence and aggregates, including after a
sorted JSON roundtrip. Resealing an altered report is insufficient.

The immutable CLI entry point is:

```sh
.venv/bin/python -m usami_benchmark audit-tasks NEW_SOURCE_PINNED_TASK_ARCHIVE \
  --output NEW_AUDIT_DIRECTORY --per-domain-target 40
```

It retains the exact input manifest commitment, source snapshot, report and
verification, or a failure record. It cannot overwrite the original archive or
an earlier audit. It performs no model calls and reads no model credentials.
Historical source-pinned task archives must run with their archived source tree;
adding the auditor does not rewrite their source manifests.
