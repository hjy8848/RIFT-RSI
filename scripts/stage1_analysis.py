"""Before/after comparison of a stage-1 run on the same held-out tasks (standard library only).

    python stage1_analysis.py BASELINE_RUN_DIR TRAINED_RUN_DIR [OUT.md]

BASELINE_RUN_DIR is an ``eval`` run of the untrained model on the held-out specs (its
``eval-result.json`` and ``episodes/``); TRAINED_RUN_DIR is the ``train-pool`` run whose in-place
held-out evaluation wrote ``eval-result.json``. Tasks are paired by their certified task reference.
Tasks that name an item by delivery age are left out of the main comparison: the Service cannot
see today's date, so they cannot be resolved (docs/RESULTS_BASELINE_20261104.md).
"""
import collections
import glob
import json
import random
import statistics
import sys
from pathlib import Path


def reference_kinds(baseline_dir):
    """task ref -> set of reference kinds, read from the baseline episodes (cached next to them)."""
    cache = Path(baseline_dir) / 'task-reference-kinds.json'
    if cache.exists():
        return {ref: set(kinds) for ref, kinds in json.loads(cache.read_text()).items()}
    kinds = {}
    for path in glob.glob(str(Path(baseline_dir) / 'episodes' / '*.json')):
        task = json.load(open(path))['episode_record']['task']
        ref = task['record_hash']
        if ref not in kinds:
            requests = task.get('request_bundle', {}).get('requests', [])
            kinds[ref] = sorted({q.get('reference', 'attributes') for q in requests}) or ['attributes']
    cache.write_text(json.dumps(kinds))
    return {ref: set(value) for ref, value in kinds.items()}


def rates(families, skip=('designer',)):
    out, klass = {}, {}
    for key, family in families.items():
        if key in skip:
            continue
        for ref, task in family['per_task'].items():
            n = task['success'] + task['failure']
            if n:
                out[ref], klass[ref] = (task['success'] / n, n), key
    return out, klass


def bootstrap(diffs, seed=0, rounds=10000):
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(rounds))
    return means[int(0.025 * rounds)], means[int(0.975 * rounds)]


def main(baseline_dir, trained_dir, out_path=None):
    baseline_dir, trained_dir = Path(baseline_dir), Path(trained_dir)
    before, klass = rates(json.load(open(baseline_dir / 'eval-result.json'))['families'])
    after, _ = rates(json.load(open(trained_dir / 'eval-result.json'))['families'])
    kinds = reference_kinds(baseline_dir)
    paired = sorted(set(before) & set(after))
    lines = [f'# Stage 1: before vs after on the same held-out tasks', '',
             f'baseline {baseline_dir.name} | trained {trained_dir.name} | tasks in both: {len(paired)} of {len(before)}', '']
    for label, keep in (('identifiable tasks (no delivery-age naming) - MAIN', lambda r: 'delivery_age' not in kinds.get(r, ())),
                        ('all paired tasks', lambda r: True)):
        refs = [r for r in paired if keep(r)]
        if not refs:
            continue
        diffs = [after[r][0] - before[r][0] for r in refs]
        low, high = bootstrap(diffs)
        lines += [f'## {label}', '',
                  f'tasks {len(refs)} | before {statistics.fmean(before[r][0] for r in refs):.3f} | '
                  f'after {statistics.fmean(after[r][0] for r in refs):.3f} | '
                  f'mean change {statistics.fmean(diffs):+.3f} (95% bootstrap over tasks [{low:+.3f}, {high:+.3f}])',
                  f'tasks that rose {sum(d > 0 for d in diffs)}, fell {sum(d < 0 for d in diffs)}, unchanged {sum(d == 0 for d in diffs)}', '']
        by_class = collections.defaultdict(list)
        for r in refs:
            by_class[klass[r]].append(r)
        lines += ['| class | tasks | before | after | change |', '| --- | --- | --- | --- | --- |']
        for key, members in sorted(by_class.items()):
            b = statistics.fmean(before[r][0] for r in members); a = statistics.fmean(after[r][0] for r in members)
            lines.append(f'| {key[:40]} | {len(members)} | {b:.2f} | {a:.2f} | {a - b:+.2f} |')
        lines.append('')
    rounds_path = trained_dir / 'rounds.jsonl'
    if rounds_path.exists():
        lines += ['## Training rounds', '', '| round | train success | rollouts | trained | sample s | update s |', '| --- | --- | --- | --- | --- | --- |']
        for line in rounds_path.read_text().splitlines():
            row = json.loads(line)
            lines.append(f"| {row['round']} | {row['train_success_all']:.2f} | {row['rollouts']} | {row['trained_rollouts']} | "
                         f"{row['sample_seconds']} | {row['update_seconds']} |")
        lines.append('')
    text = '\n'.join(lines)
    if out_path:
        Path(out_path).write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main(*sys.argv[1:4])
