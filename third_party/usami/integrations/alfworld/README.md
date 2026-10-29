# ALFWorld Adapter

This integration connects the ALFWorld **TextWorld** environment to the
existing `rsi_core` `EnvironmentAdapter`, task registries, `ServiceEvaluator`,
and `run_period`. RSI Core is unchanged.

## Install ALFWorld data

The official ALFWorld setup supports Python 3.9 or newer. Install the package
and download its game files:

```bash
python -m pip install 'alfworld[full]'
python -m pip install -e '.[service]'
alfworld-download
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export ALFWORLD_CONFIG="/path/to/alfworld/configs/base_config.yaml"
```

Point `ALFWORLD_DATA` at the directory containing `json_2.1.1/`. On Apple
Silicon, use the ALFWorld project's documented x86-64 Conda environment if its
TextWorld dependencies do not install for the native architecture. The adapter
loads the official YAML config directly; obtain `configs/base_config.yaml` from
the ALFWorld source checkout or pass its path with `--config`.

## Run the native environment smoke

```bash
python -m experiments.rsi.examples.alfworld_period smoke \
  --config "$ALFWORLD_CONFIG" \
  --output experiments/rsi/results/alfworld-smoke-001
```

This starts five registered ALFWorld train games, sends actions through the
native TextWorld environment, saves episode records, and replays each action
trace. The smoke policy is fixed and uses no model. It verifies the environment
adapter, not RSI or model quality.

## Run one RSI period without weight training

Configure the existing OpenAI-compatible client variables
(`USAMI_LLM_MODEL`, `USAMI_LLM_BASE_URL`, and its API key), then run:

```bash
python -m experiments.rsi.examples.alfworld_period period \
  --config "$ALFWORLD_CONFIG" \
  --output experiments/rsi/results/alfworld-period-001
```

The period runs the existing RSI Core Service improvement and Designer
self-edit stages. `P0_alfworld.py` selects commands; `M0_alfworld.py` proposes
an ALFRED task type, has the Host bind it to a certified ALFWorld train file,
then proposes P/K and M updates. The Core period-end model training hook is
not supplied, so no verl process or parameter update runs. Reports remain
`READY_FOR_TRAINING` when both RSI groups complete.

The task registry reserves disjoint ALFWorld train files for challenge,
Service probe/selection/reward, and M probe/selection/reward. A separate
`valid_unseen` set is recorded as `meta-test` audit data and is not exposed to
the Designer or used for training feedback. `--tasks-per-set N` requires at
least `9*N + 1` supported train tasks and one `valid_unseen` task.

## ALFWorld-specific boundary

ALFWorld supplies pre-authored TextWorld games; it does not generate new
business worlds. The abstract challenge in this adapter is a task-type choice,
grounded into one existing task file by a stable hash over the registered
matching training files. The adapter records the source hashes, sends the
chosen action string unchanged to ALFWorld, scores from ALFWorld's native
`won` result, and replays the recorded actions. It never supplies ALFWorld's
expert plan to Service or M.

Use the usual pinned RSI Docker worker image for `period`; the entrypoint does
not pull it automatically.
