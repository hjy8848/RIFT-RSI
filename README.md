# RIFT-RSI

RIFT studies whether post-update changes in the likelihood of independent historical successful actions can teach a task Designer to improve a Service. The present repository is an offline research scaffold, not a trained RSI system. Read [research specification](docs/RESEARCH_SPEC.md) and [architecture](docs/ARCHITECTURE.md) before using it.

Initialize the fixed Usami submodule with `git submodule update --init`, install this package with `pip install -e ".[test,usami]"`, and run `python -m pytest`. Imports perform no training, model downloads or API calls. The native benchmark package also imports its own Python modules; `third_party/usami` must be on the Python path when using the adapter, for example set `PYTHONPATH=src;third_party/usami` in PowerShell. No verl, Ray, or model weights are required.
