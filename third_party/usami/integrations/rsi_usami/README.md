# Open Benchmark integration for RSI v0.2

`run_open_period()` obtains fresh frozen commerce Worlds, accepts abstract
challenges, grounds and independently certifies them, and injects the resulting
native task handles into the existing RSI `PROPOSE_CHALLENGE -> RUN_BASELINE ->
PROBE` flow. `rsi_core` and its P/K/M/φ selection, rewards and training are
unchanged. The fixed `smoke_v02_usami_period.py` remains available.

## Native reuse and the new boundary

- World generation calls the existing native World Designer through
  `design_world_slot()`. `ScenarioDraft`, `DistributionSpec`, `sample_schedule`,
  `build_world`, `verify_build`, and business invariants remain authoritative.
  `design_and_certify()` now delegates its World phase to this same function;
  its public Task phase and historical output contract remain unchanged.
- `AbstractChallenge` **extends `TaskDraft`**. It reuses `RemedyDraft`, request
  spans, quantity/attribute/arrival/extra-charge goals and native validation.
  Added line selectors express family, issue, purchased attributes, quantity,
  age and return requirements; relations express same/different orders.
  Customer and line aliases are symbolic variables, with no native IDs.
- Native v2 Grounder uses these selectors across the frozen population. All
  variables bind distinct lines owned by one Customer. Several matches are
  allowed. Candidates are deterministically ordered by the declared selection
  seed and candidate hash; compilation/proof attempts are archived in order;
  the first certified candidate is selected. Search limits/truncation are
  explicit. Grounder never consults a Reference witness or changes an objective.
- Binding produces a native `TaskDraft` and invokes the existing `compile_task`,
  `certify_task`, and `verify_task_certificate`. It does not implement another
  business compiler, Engine, Oracle, or goal evaluator. The old
  `usami_benchmark` Grounder remains unchanged for legacy After-sales schemas.
- `NativePackageVerifier` registers the native proof before a task handle is
  returned. The existing `UsamiAdapter` executes Customer/Service, Engine,
  Oracle and Replay. Proof coverage flags are preserved: business reachability
  does not assert natural dialogue success.

## Worlds across periods

`WorldPopulation(root, distribution, world_provider, seed=...)` persists the
sampling schedule, World proposals/calls, construction proofs, failures and
period history. Every call to `next_period(count)` generates new registered
slots. Reopening the same directory checks the distribution, seed and history;
subsequent periods continue instead of restarting. A World stays frozen within
its task. Exact frozen World repetition is rejected.

World structure signatures remove aliases, names, IDs and seeds. They record
ownership/order/line/SKU-family topology, warehouse counts, issues, purchased
quantities, return requirements and inventory availability. Diversity reports
must use these signatures and actual structures, not distinct World hashes alone.

The production provider can be the existing `OpenAIDesignerProvider(client)`.
`experiments/rsi/configs/open-commerce-worlds.json` defines an extensible synthetic
World distribution with business-composition strata; it contains no tasks or
answers. A deterministic provider is also accepted, useful for integration tests
and pre-generated registered populations. No provider is replaced implicitly.

## Formal API

A trusted experiment factory assembles the existing native RSI context:

```python
population = WorldPopulation(
    output / "worlds", distribution, OpenAIDesignerProvider(world_client), seed=42)

record = run_open_period(
    population=population, world_count=3,
    host=host, service_evaluator=service_evaluator,
    adaptation_evaluator=adaptation_evaluator,
    designer=designer_snapshot, service=service_snapshot,
    service_rounds=registered_rounds,
    self_edit_context=registered_self_edit_context,
    customer_binding=customer_binding, verifier=native_package_verifier,
)
# The next call obtains fresh World slots and uses the actual selected snapshots.
service_snapshot = host.store.load(record["rsi_period"]["output_service_ref"])
designer_snapshot = host.store.load(record["rsi_period"]["output_designer_ref"])
```

`customer_binding(world, task)` supplies the existing trusted Customer profile,
private proof scenario, and Customer factory. `episode_budget`, `mode` and
`proof_budget` are explicit options. `capability_options` passes the native
`task_design_capabilities()` options; return/approval capabilities can be enabled
with their trusted Customer scenario and `external_event_factory`. Unsupported
information/delivery goals retain native `UNKNOWN`/rejection behavior.

The CLI accepts a Python factory returning these API arguments, including the
World provider and actual model clients. It invokes the original full RSI period
and carries selected snapshots into subsequent periods:

```sh
PYTHONPATH=. python -m experiments.rsi.examples.open_world_usami period \
  --factory your_experiment:context --output outputs/open-rsi \
  --periods 2 --world-count 3
```

Factories provide the existing independent service_probe/select/reward task
sets and C_probe/select/reward packages. They can use callable self-edit context
construction against the actual final Service, as in `run_period()`.

## Source and feedback isolation

Training accepts `train` World distributions only. The population checks native
ScenarioSpec/distribution provenance and cannot be resumed under another split.
Fresh open Worlds must be disjoint from already registered independent task
World sources. Independent task sets/packages stay in Core's existing registries;
they are absent from the open World catalogue. Cited training evidence must be
in the Host's explicit `allowed_training_hashes`.

The model receives aggregate vocabulary and supported schema, not World JSON,
private entity IDs, private alias maps, Reference witnesses or independent test
results. Runtime feedback uses the existing declared projection, including its
permitted aggregate scores/safety signal. This integration does not change that
information or reward contract.

Failures remain separate: invalid abstract proposal, no matching binding,
truncated search, native compilation rejection, Reference UNSAT, Reference
UNKNOWN and provider failure. A model correction must be explicitly requested;
no answer, objective or World is silently repaired.

## Reproduce deterministic interface validation

```sh
PYTHONPATH=. python -m experiments.rsi.examples.open_world_usami period \
  --factory experiments.rsi.fixtures.open_world_validation:context \
  --output outputs/open-interface-check --periods 2 --world-count 3

python -m pytest tests/test_rsi_open_world.py -q
```

This factory samples changing World structures, then uses deterministic abstract
proposals and a public-tool-only Service policy. It verifies environment/RSI
wiring, not autonomous generation or learned improvement. Its synthetic model
usage metadata must not be reported as real API usage.

## Separate real Designer validation

After configuring the native model client environment and Keychain credentials:

```sh
PYTHONPATH=. python -m experiments.rsi.examples.open_world_usami validate-designer \
  --factory experiments.rsi.fixtures.open_world_validation:proposal_context \
  --output outputs/open-interface-check
```

This makes **one actual model request** against the existing frozen training
World catalogue. It records the raw native AgentCall and independently grounds,
compiles, certifies and replays the result. It does not execute an online Service,
Reflection or RL. For production Worlds, use your experiment's factory instead.
An explicit `--period-ref` selects an existing World period; `--revise-attempt`
passes only that period's recorded proposal/schema rejection back to the model.
No raw proof, binding, Oracle judgment or independent test result enters the
correction prompt. All prior failures remain in the immutable archive.

Raw World/certificate/episode records stay under `outputs/`; commit small reports
and evidence indices rather than large duplicate experimental logs.
