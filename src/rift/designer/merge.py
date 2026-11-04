"""Merge stage-0 plan files into one training pool and select across them.

    python -m rift.designer.merge --inputs A/designer-plans.jsonl B/designer-plans.jsonl \
        --out pool/designer-plans.jsonl [--copy-threshold 1.01]

Tasks with the same decision signature keep only their first occurrence over all files. The
default copy threshold of stage 0 drops plans copied from a prompt example; a threshold above 1
keeps every certified task, which suits a pool that should be as large as possible.
"""
import json
from pathlib import Path

from rift.designer.stage0 import COPY_THRESHOLD, load_plans, select_tasks


def merge_plans(inputs, out, *, copy_threshold=COPY_THRESHOLD):
    rows, seen_ids = [], set()
    for path in inputs:
        for row in load_plans(path):
            if row['plan_id'] in seen_ids:
                raise ValueError('plan %s appears in two input files' % row['plan_id'])
            seen_ids.add(row['plan_id'])
            rows.append(row)
    certified = [row for row in rows if row['task_refs']]
    selected = select_tasks(certified, copy_threshold=copy_threshold)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as stream:
        for row in certified:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    return {'input_plans': len(rows), 'plans_kept': len(certified),
            'certified_tasks': sum(len(row['task_refs']) for row in certified), 'selected_tasks': selected}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Merge stage-0 plan files into one selected training pool')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--copy-threshold', type=float, default=COPY_THRESHOLD)
    args = parser.parse_args(argv)
    print(json.dumps(merge_plans(args.inputs, args.out, copy_threshold=args.copy_threshold), sort_keys=True))


if __name__ == '__main__':
    main()
