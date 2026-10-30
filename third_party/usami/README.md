# Usami runtime subset vendored by RIFT-RSI

This directory is the RIFT-maintained runtime subset of [Usami](https://github.com/hjy8848/Usami), pinned in [the RIFT research spec](../../docs/RESEARCH_SPEC.md). It preserves the native Usami v2 Commerce environment used by RIFT; it is not a standalone distribution of every historical Usami benchmark, domain, web interface, or experiment.

The supported path is `ScenarioDraft` and `AbstractChallenge` → frozen World build and grounding → native `TaskSpec` compilation and certification → `run_episode` → deterministic `replay_episode` and Outcome. RIFT owns model generation, RL, orchestration, and Reward. The retained Usami code supplies the real business state machine, tools, Customer runtime, Engine, Reference/Oracle checks, and outcome evaluation.

The legacy online Designer provider and quality-diagnostic workflow were removed because RIFT generates its own proposals and does not consume those diagnostics. Existing Usami business, certification, execution, replay, and outcome behavior remains covered by the retained CPU tests and RIFT regression tests.

The retained package consists of the import closure needed by RIFT and focused tests. Historical upstream research documents under `docs/benchmark-vnext/` remain as source context; they are not included as executable packages.

From the RIFT repository root, run the core integration tests with:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path + ';' + (Resolve-Path third_party/usami).Path
python -m pytest -q tests/test_usami_native.py tests/test_training_package.py tests/test_verl_bridge.py
```

The focused native Usami test files are listed in `pyproject.toml` / `MANIFEST.in`. Hypothesis-based property tests require the `dev` extra. No model training or model weights are needed for these tests.
