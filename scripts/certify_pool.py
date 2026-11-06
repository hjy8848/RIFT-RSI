"""Certify task specs on the CPU and save the registry, so training runs can start from it.

    python scripts/certify_pool.py --out DIR --specs configs/experiments/probes_big.json \
        [--from DIR/task_registry] [--drop-references delivery_age]

``--out`` receives ``task_registry/`` (splits, certified tasks, spec cache); ``--from`` is an earlier registry
whose tasks are kept and not proved again. Prints how many tasks each class has before and after dropping
the references named in ``--drop-references``. Uses no GPU.
"""
import argparse
import json
import sys
import time
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--specs', required=True)
    parser.add_argument('--from', dest='source')
    parser.add_argument('--drop-references', default='')
    args = parser.parse_args(argv)
    from rift.runtime.curriculum_run import WorldKit, _drop, load_json_argument, seed_task_registry
    out = Path(args.out)
    if args.source:
        seed_task_registry(out, args.source)
    kit = WorldKit(out)
    started = time.perf_counter()
    by_key = kit.certify_specs(load_json_argument(args.specs))
    kit.save()
    kinds = set(item for item in args.drop_references.split(',') if item)
    context = type('Context', (), {'world': kit})()
    kept = {key: _drop(context, refs, kinds) for key, refs in by_key.items()}
    print(json.dumps({'seconds': round(time.perf_counter() - started),
                      'classes': {key: [len(by_key[key]), len(kept[key])] for key in sorted(by_key)},
                      'total': sum(map(len, by_key.values())), 'kept': sum(map(len, kept.values()))}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
