# World-grounded Designer: offline contract audit

The 2026-10-19 InferAI attempt exposed contradictory public vocabulary: the
catalogue advertised private-index relationship names and composition labels
that the symbolic validator rejected. This audit fixes the task-generation
chain before another online call. It does not reinterpret the failed proposal
as a valid task, modify native business rules, or claim model learning gains.

## Confirmed defects and corrections

| Boundary | Correction |
| --- | --- |
| Public catalogue / function schema | One grammar supplies operators, legal relationship pairs and atomic objective forms. Private index edges no longer define the public language. |
| Field operands | Publish native enums/types, validate them before search, reject boolean/integer confusion and unsupported ordering; compare aware dates as instants and handle null delivery dates. |
| Model submission | Inline local schema refs, encode objective/constraint/event forms, require envelope and candidate fields, fix requested batch size, and explain the actual wire keys. Provider enforcement is never assumed. |
| Grounding identity | All bound identity-bearing entities remain within one Customer scope; objective profiles are explicit and unambiguous; requirements target objective profiles/lines. |
| Event handling | Binding an event alias is read-only. Only an explicit event condition schedules it; contradictory event conditions are rejected. |
| Match provenance / budget | Binding hashes include the frozen registration, fact paths name real native collections, and explored bindings never exceed the declared cap. |
| Batch processing | Unhashable/malformed IDs do not crash; duplicate IDs cannot collapse records or keep proofs; mismatched counts cannot silently truncate and accept a batch. |
| Bounded revision | A wholly malformed envelope can recover a complete multi-candidate batch; valid failed IDs remain stable. Internal implementation/Replay failures stop revision. |
| Native proof errors | Distinguish bounded search, compiler rejection, internal errors and certificate Replay failures. Preserve original model evidence on failure. |
| Metrics / CLI | Sanitize invalid token metadata as unmeasured, rebuild diagnostics during Replay, expose explicit certification status, return nonzero for zero certified tasks. |
| Archive / offline CLI | Fix local-import scope failure in scripted runs; reject existing output before dispatch; verify exported artifact hashes as well as the authoritative record. |
| Coverage | Preserve objective-to-entity associations and directed references in fingerprints; alias spelling and symmetric edge orientation do not create new coverage. |
| Existing v2 history | Freeze the original three-line After-sales rulebook text for v2 `read_rules`; later prose clarifications must not change historical request bytes. Native rules and v3/v4 responses are unchanged. |

## Compatibility

New records carry `world-grounded-contract-v2`; the public catalogue is v2.
Top-level single-shot/one-revision names remain unchanged, and v3/v4 Designer
wire behavior is unchanged. Historical batch-v1 records require source commit
`cbb5da4`. Current Replay reports that requirement instead of interpreting old
results with new semantics. Old records and the failed live attempt are not
modified. Native frozen-world index contents remain compatible.

## Verification boundary

All checks in this change are offline. Tests exercise native World / Reference
construction, task certificates and Replay, with scripted model transports for
protocol fault cases. No InferAI or other paid model call is part of this audit.
The original malformed model output is still a failed submission. Whether a
real model now produces a certifiable challenge remains to be tested after this
audit, using a new run directory and the revised contract.

See [the protocol and commands](WORLD_GROUNDED_BATCH_DESIGNER.md) and
`tests/test_world_grounded_batch_designer.py` for executable assertions.

## Recorded offline run

The three-world index `0f957b7c779095dbb7a8bb27f13686bfe2b1cf24282e4f2904a4bf001c1897fd`
produced five certified rule-baseline challenges (four coverage signatures),
with five native certificates verified by archive Replay. The local artifact is
`/tmp/usami-world-grounded-contractv2-audit-20261019-001/`, run hash
`b4b181e70cd46e85a29f4a21b50ab4d5e5aa3d4547860bd02e88c283cb8bba8b`.
Provider calls, provider tokens and Service executions are all zero.

The initial expanded regression completed with 171 passes and two failures.
One was the historical v2 prose drift fixed above. The other was an audit-side
interrupt caught by the Service controller and surfaced as `blocked`, rather
than a Service Gate code defect; its isolated rerun passed (1/1). All failures
were retained and affected suites rerun after the correction.

Final checks:

* `pytest -q tests/test_world_grounded_batch_designer.py`: **44 passed** (24.81s).
* `pytest -q tests/test_designer_exploration.py tests/test_designer_event_links.py tests/test_frozen_tasks.py::test_original_service_gate_and_budget_remain_active`: **35 passed** (143.87s).
* Eight changed Python files passed syntax checks; `git diff --check` passed.
* The five-task rule-baseline archive passed standalone CLI Replay, with all
  five native certificates regenerated and verified.

These final runs are affected-suite verification, not a claim that the entire
repository was rerun in one all-green test invocation.
