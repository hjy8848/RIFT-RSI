"""Offline source-pinned native example verification (no model calls)."""
import argparse,json
from .artifacts import verify_examples


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    verify=sub.add_parser('verify-examples');verify.add_argument('directory')
    args=parser.parse_args(argv)
    result=verify_examples(args.directory)
    print(json.dumps(result,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
