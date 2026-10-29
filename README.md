# RIFT-RSI

RIFT studies whether post-update changes in the likelihood of independent historical successful actions can teach a task Designer to improve a Service. The present repository is an offline research scaffold, not a trained RSI system. Read [research specification](docs/RESEARCH_SPEC.md) and [architecture](docs/ARCHITECTURE.md) before using it.

From the repository root in PowerShell:

```powershell
git submodule update --init
python -m pip install -e ".[test,usami]"
$env:PYTHONPATH = (Resolve-Path .\third_party\usami).Path
python -m pytest -q
```

The native benchmark imports its own packages from the pinned submodule. Imports perform no training, model downloads or API calls. No verl, Ray, or model weights are required.
