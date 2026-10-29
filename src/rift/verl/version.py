"""Check the pinned VERL distribution without importing its runtime."""
from importlib.metadata import PackageNotFoundError, version

VERL_VERSION = '0.9.1'
VERL_COMMIT = '1876b06d0a3e4e71e06230be10af14492ca8a75b'

def require_verl_version():
    try:
        installed = version('verl')
    except PackageNotFoundError as exc:
        raise RuntimeError('VERL 0.9.1 is not installed') from exc
    if installed != VERL_VERSION:
        raise RuntimeError(f'expected VERL {VERL_VERSION}; found {installed}')
