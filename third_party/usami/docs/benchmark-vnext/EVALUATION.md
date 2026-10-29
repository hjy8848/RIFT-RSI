# Frozen evaluation and balanced reporting

`usami_benchmark.evaluation` implements `eight-domain-evaluation-v1` plans,
`eight-domain-task-result-v1` records and deterministic aggregation. This is an
aggregation layer, not a replacement for native Reference, Engine, Oracle or
Replay. Trace adapters must verify execution before constructing result records.
The native all-eight result adapter is available; the balanced dataset release
remains in progress.

A plan freezes partition, evidence kind, model-configuration content hash and the
exact task-content hashes for every included domain. A result binds that plan,
task, world and execution. Unknown tasks, duplicate results, mixed configurations,
mixed partitions and mixing native proofs/scripted transport/real models are
rejected. Independent repeated trials require separately frozen plans. Public
test-demo and private held-out are distinct partitions; this module does not by
itself establish absence of training leakage.

## Endpoints

The common boolean endpoints are task success, legal completion, correct refusal,
false refusal, successful business violation, unauthorized side effect, frame
preservation, budget compliance and Replay verification. Dialogue endpoints add
required information resolution, consent compliance, cross-turn commitment
consistency, valid decision revision, premature irreversible action and verified
dialogue dependency. Missing/unknown endpoints remain `null`. Completion-only
tasks do not enter the correct-refusal denominator; refusal-only tasks do not
enter the legal-completion denominator. A mixed task succeeds only if both
applicable components and the safety/budget/Replay requirements pass.

A producer cannot declare successful execution while omitting frame/budget/Replay
proof or hiding business violations, unauthorized effects or false refusal. This
consistency check is not itself proof that a trace is correct: native validators
remain mandatory. Descriptor capability/difficulty annotations also require their
separate certificate audit, and are not validated simply by their presence.

## Aggregation

Each domain gets weight **1/8**, and each four-domain Track gets weight **1/2**.
Per-domain means are calculated first, then averaged. Nine SaaS tasks cannot
outvote one Education task. Official eight-domain macro values require all
registered tasks in all eight domains. Missing domains are never silently removed
and weights never renormalized; an incomplete summary exposes missing task IDs.
Track summaries likewise require all four member domains to be complete.

Each endpoint retains numerator, eligible denominator, unknown/inapplicable count
and observed rate. If a domain has no eligible observations, its endpoint is
`null`; a macro requiring that endpoint is also `null`. Dialogue-only endpoints
can therefore have a Track B value while the all-eight value is inapplicable.
Capability, structural-difficulty, outcome and business-family strata retain
separate domain summaries. They do not become a single opaque score.

Costs retain Customer/Service input and output tokens separately, plus calls and
tools. Missing usage stays `null`, including for scripted fixtures; absence of
usage is never a measured zero-token model run. Native proofs, scripted fixtures
and real models must remain separate experimental cohorts.

The small deterministic test cohort in `tests/test_vnext_balanced_evaluation.py`
checks weighting, missing domains, conditional denominators, provenance, costs
and mixed-goal/safety consistency. It is not benchmark model performance. No
real-model comparison has been run for this release (`NOT RUN`).

## Certificate-backed descriptor adapter

`usami_benchmark.episodes.certified_evaluation_result(plan, episode)` is the
new release adapter for evidence-derived strata. It independently rebuilds the
native task descriptor, derives the supported capabilities and structural
difficulty from its certificate, and then performs the original complete native
model/score/cost Replay before constructing an evaluation result. There are no
caller arguments that can substitute capability or difficulty labels. The
exact task content hash and native interface remain bound to the frozen plan.

The original `evaluation_result()` remains available for historical workflows
whose descriptors were externally audited; its behavior is unchanged. Native
certificate descriptors are coverage of the task's verified witness. They do
not imply that the evaluated model exercised every capability successfully.
In particular, an unclassified dialogue task must not be silently assigned to a
desired difficulty quota. No model requests or credentials are used by either
descriptor extraction or offline Replay.
