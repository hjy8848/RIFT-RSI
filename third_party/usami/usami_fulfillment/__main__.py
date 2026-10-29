from usami_harness.cli import main as _main
from . import adapter
from .release import build_release


def main(argv=None):
    return _main(adapter,build_release,argv)


if __name__=='__main__':
    raise SystemExit(main())
