# Independent structural world generation

## Versions

| Domain | New distribution | Native rules reused |
| --- | --- | --- |
| SaaS | `saas-generator-0.2.0`, without workflow conditioning | `saas-rules-0.2.0` |
| CloudOps | `cloudops-structural-v1` | `cloudops-0.1.0` |
| Fulfillment | `fulfillment-structural-v1` | `fulfillment-0.1.0` |
| StayOps | `stayops-structural-v1` | `stayops-0.1.0` |

New structural distributions live in `usami_benchmark/generators/`, outside historical source-hash directories. Historical generator functions and fixed outputs are unchanged. `verification/legacy-generator-baseline.json` pins seed 42 and source hashes against Git commit `2319b27`.

Common structural configuration bounds scope/resource counts and lifecycle prefix depth. All random choices use the repository's SHA-256 counter `StableRandom`, seeded by distribution version, input seed and canonical configuration. A WorldSpec with an unsupported generator/rules version is rejected. No task/goal/model/history-of-errors field is accepted in the generator config.

## Native meaning of sampling

CloudOps samples organizations/projects, project-role graphs, active/suspended projects, paid/unpaid/failed bills, chain/star/layered/independent DAGs, stopped/running closure and quota-to-usage relationships. Running instances have running dependencies and fit all three quota dimensions.

Fulfillment samples organizations, warehouses, role relations, products/categories, multiple stock lots, available/quarantine stock, signed supplier receipts, receiving capacity, multi-line orders and independent lifecycle prefixes. Native tools create payment, reservations, pick/pack proofs, shipments, delivery attestations and returns. No stock is materialized by an unproved assignment to a shipped status.

StayOps samples properties/roles, room types and physical rooms, clean/dirty/maintenance states, date capacity/closure, rate plan applicability, deposits/cancellation terms, guests and attested booking histories. Native operations produce confirmation, assignments, identity checks and actual stays. Failed independently sampled booking choices do not create records.

NativeHistory verifies each admissible expected transition against invariants, executes through the original Engine and finally verifies original Oracle and exact Replay. This is an independent simulator distribution, not generation conditioned on a pending Benchmark task. Designer and Service feedback are never inputs.

## A preserved native consistency defect

The first 400-world run exited 1: StayOps generated only 96 valid worlds. In four traces an allowed native `check_out` dirtied a room with a future active assignment; native invariants then reported `DIRTY_ROOM_ASSIGNED`. Both the actual mutation and native expected transition agree, but their expected state violates the native invariant.

The failed audit, original four full traces, trusted authority, Oracle results and exact source archive remain in `audits/worlds-phase1/`. No native rule, applier or Oracle was changed.

The corrected initial-state distribution excludes expected transitions that already fail native invariants. All such exclusions are retained in each FrozenWorld and in `generation_exclusions`. This does **not** intercept Service operations, alter Sandbox, label the simulator defect as an Agent discovery or establish that all native transitions are consistent. A separate versioned business correction remains necessary before including this boundary in a public Agent-scored task.

## Actual 100-world audit

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m usami_benchmark.audit \
  --count 100 --root benchmarks/vnext-0.1/audits/worlds-phase1-final
```

Each of the four existing domains produced 100 valid worlds, 100 full hashes, 100 anonymous structural signatures and 100 permission graph signatures. StayOps records four generation exclusions; the other three record zero. Structure-based Train/Validation/Test Demo partition assignment has zero cross-partition structure overlap in this sample. This is not yet a task-level semantic leakage result.

Signatures use fixed-round relational colour refinement. Names, IDs, clock strings, raw numeric amounts, price jitter and external signature strings are excluded; roles, object relations, operational states and policy-derived inequalities remain. Permission graph signatures use scoped roles/ownership, not project status or capacity. Colour refinement is conservative and not an exact graph-isomorphism proof.

The corrected audit freezes compressed complete worlds, source snapshots, source hashes and reports. Task structures, legal/refusal proof coverage and causal depth are still explicitly `NOT RUN` in this world-only audit; they will be measured by the later task certification audit.

## Limits still requiring work

These samples are not evidence of real-world representativeness or model generalization. The distributions use bounded synthetic organizations and simulator clocks. Financial eligibility, logistics aging, alternative history graphs and additional task-conditioned strata must be separately audited. Four dialogue-domain generators are not yet implemented in this checkpoint.

## State census v2 and education registry

`eight-domain-world-audit-v2` keeps anonymous-structure and ownership grouping
separate from three descriptive state censuses: current business categories,
latent `simulated_*` facts, and pre-registered external-event kinds/outcomes.
A future successful repair or academic approval does not count as a currently
completed repair or approved request. Names, raw money, timestamps and repeated
list entries do not add state categories. This census does not claim that a
Reference boundary was exercised; task and causal-depth coverage remain separate
measurements. V1 artifacts retain their original schema and archived sources.

Education is registered as `education-generator-v1` / `education-0.1.0`. Its
unconditioned worlds vary school/course packages, verified academic levels,
student enrollment/ownership relationships, teacher qualifications, dated classes,
limited seats, actual opening consumption and study-credit balances, contract
refund/transfer/extension restrictions and calendar conflicts. Anonymous grouping
uses relations plus policy-derived refund progress/window, credit, expiry,
capacity, duration and conflict categories. Future academic/attendance outcomes
are frozen environment inputs, reported separately by the state census.

The new four-domain census is archived under
`benchmarks/vnext-0.1/audits/dialogue-state-v2-001/`, with 100 worlds per domain,
full source archive, hashes and generation specifications. All 400 generated
worlds passed native invariants. Anonymous structures: 100 each; ownership
topologies: After-sales 99, Airline 95, Property 100, Education 100. These counts
do not prove graph non-isomorphism, task independence or model generalization.
They do not replace the remaining lifecycle-history and certified-task audits.

New vNext default StayOps specifications now use structural-v2 / stayops-0.2.0,
whose independent 100-world audit is already archived. Explicit v1 specifications
remain executable. This is a change to the isolated vNext default registration;
it does not redirect historical native generators or rewrite old manifests.

## Opt-in native lifecycle generator v2 (under audit)

The four explicit registrations `<domain>-lifecycle-v2` reuse each native v1 base
world and then sample bounded legitimate customer choices, existing native
operations and frozen external events. They never consume a task, Designer,
Service result, score or Skill. Selected transitions use actual Session/Engine,
independent Oracle and Replay. Compact historical journals retain actual Customer
facts/choices, current offers/consents, business operations and simulator events;
`replay_history` independently executes those journals again. No lifecycle status,
credit, refund, repair completion or receipt is assigned directly by the generator.

At the generated episode boundary, previous interaction authority is cleared.
Education's initial-expiry fact is rebound to the actual generated epoch; business
state is preserved. Old v1 distributions remain explicitly registered and remain
the current dialogue defaults until audit/breadth assessment is complete. Setting
`lifecycle_steps: 0` yields exactly the corresponding base distribution.

Audit v3 adds actual generation-history counts and separates already applied
external events from future frozen outcomes. Historical v1/v2 audit files and
sources are not rewritten. The 12 initial lifecycle/census tests passed; the
100-world-per-domain audit is running in
`audits/dialogue-lifecycle-v2-001/`. This paragraph records an active run, not its
outcome; use its completed summaries/logs to establish the eventual result.

The opt-in lifecycle audit above has now completed: all 400 worlds are valid,
100 anonymous structures per domain, with actual history counts and sparse
commitment coverage retained in `dialogue-lifecycle-v2-001/summary.json`. Its raw
log is `verification/logs/dialogue-lifecycle-v2-400-audit.log`. Earlier wording
records when the run started; it is not an assertion that it remains alive.
