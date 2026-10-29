# Diversity & Coverage Audit

This module analyzes the existing generator and reference policy, independently of
Service/Customer LLMs and the web frontend. It does not change business rules or
benchmark scoring.

From the repository root:

```bash
python -m pip install -e '.[dev]' -c constraints-tested.txt
python -m usami.audit \
  --start-seed 0 --count 1000 \
  --config configs/standard.json \
  --split-worlds 100 --engine-checks 32 \
  --output audits/diversity-v1
python -m pytest -q
```

`--count` and `--start-seed` fix the ordered primary world sample. `--config`
accepts the existing GeneratorConfig JSON schema, including custom rules.
`--split-worlds` independently controls each audit-derived demo split: train
seeds 0…, validation 1000000…, test 2000000…, using the input config with the
existing train_single / standard / test_composed profiles. Each demo uses the
existing category-first manifest selector (12 tasks per world). All seed lists,
configs and manifest hashes are exported. These public audit samples are not a
private test set. `--validation-manifest` selects the frozen validation to analyze
separately (defaults to the shipped manifest, with its own config).

`--engine-checks` caps Engine/Oracle/replay verification, choosing the first
representative of each newly encountered template. If fewer templates exist,
fewer checks run; JSON records the actual checked count and scenario IDs.
Every primary candidate step, including those outside that subset, is recomputed
through ReferenceRules and checked against its expected label and state hash.

`results.json` includes source hashes, provenance, full frequencies, representative
signature payloads, world distributions, rule matrix, complexity and split overlap.
`REPORT.md` is generated deterministically from those measured results. See the
committed [research artifacts](../../audits/diversity-v1/REPORT.md),
[roadmap](../../audits/diversity-v1/ROADMAP.md), and
[verification](../../audits/diversity-v1/VERIFICATION.md).

## Measurement contract

- Template: anonymous typed entity graph, actor membership relationships,
  operation sequence and measured legality dependency edges. IDs, names and
  scenario labels do not enter it.
- Semantic: template plus permission/prestate facts, policy-relative parameter
  comparisons, authorization, real reference verdict, actual guard visits and
  normalized before/after changes. Absolute amount/time/seat values are removed
  only where policy comparisons agree. Changes in granularity require a new
  signature version and cannot be compared as identical measurements.
- World graph: exact anonymous user–organization incidence canonicalization for
  up to six organizations, optionally with coarsened policy state. Larger graphs
  explicitly return Unknown for whole-world canonicalization; other metrics work.
- Dependency: remove an earlier successful state-changing step and rerun the
  remaining prefix with the actual reference. An edge means a later legality/error
  changes. Edges may be mediated; path depth is not identified direct causal depth.
  Read-information and agent-planning dependencies are unmeasured.
- Order: reverse the sequence, compare verdict bags keyed by actor+operation and
  final business state, ignoring refund execution-nonce IDs. Identical repeated
  refund calls do not create spurious order sensitivity. Not all permutations run.
- Implemented guards are parsed from real rejection conditions; exercised guards
  and vetoes use line observations of the unmodified reference function. Derived
  simultaneous predicate violations include short-circuit-masked conditions, but
  do not count as exercised vetoes or causal interactions. Missing prerequisites
  are Unknown. g01–g07 are generic version/integrity/existence/scope gates;
  business co-visitation starts at g08.
- Authorized attempts and business-policy decisions are distinct. A rejected
  request is not a successful business violation. Policy evaluation on an
  unauthorized sample describes the auditor's check, not Engine execution order.

Use `run_audit(seeds, config, ...)` for a complete census or
`analyze_scenario(world, scenario, RuleObserver())` for an existing labeled
scenario. Inputs are read only; analysis fails when expected reference labels or
state hashes do not match. The line observer is synchronous and restores the
thread's trace hook; run the audit in a dedicated process if profiling or coverage
instrumentation must observe every reference invocation concurrently.

No finite sample proves exhaustive diversity or predicts LLM accuracy/exploit
novelty. The published counts describe current constructor-generated candidates.
