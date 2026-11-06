"""Matched training arms forked from one checkpoint and scored on the same frozen probes.

One Teacher action is credited with the difference between two arms that start from the
same weights, train for the same number of steps on packages of the same size, and are
scored on the same Recovery/Retention probes:

    R = R_dual(treatment arm) - R_dual(control arm)
    R_dual = G - lambda * max(0, -H)

G and H are task-averaged success-rate changes on the Recovery and Retention banks. The
"before" term of G cancels in the difference, so the main noise left is the sampling of
the two "after" evaluations; the paired task bootstrap below measures it.

The power check runs this with several arms (two random ones give the noise floor) before
any Teacher is trained. See docs/PLAN_SELF_TAUGHT_TEACHER.md.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import random


@dataclass(frozen=True)
class ProbeSet:
    recovery: tuple[str, ...]
    retention: tuple[str, ...]

    def __post_init__(self):
        if not self.recovery or not self.retention:
            raise ValueError('both probe banks need at least one task')
        if len(set(self.recovery)) != len(self.recovery) or len(set(self.retention)) != len(self.retention):
            raise ValueError('a probe bank lists a task twice')
        if set(self.recovery) & set(self.retention):
            raise ValueError('a task cannot be in both probe banks')

    def by_key(self):
        """The shape ``sample_rows`` takes: one class per bank."""
        return {'recovery': list(self.recovery), 'retention': list(self.retention)}


@dataclass(frozen=True)
class ArmSpec:
    name: str
    source: str           # 'random': TRAIN pool only; 'designer': Designer tasks plus the TRAIN mix
    seed: int

    def __post_init__(self):
        if not self.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-'
                                for c in self.name):
            raise ValueError('arm names are short slugs')
        if self.source not in ('random', 'designer'):
            raise ValueError('arm source must be random or designer')
        if type(self.seed) is not int:
            raise ValueError('arm seed must be an integer')


def parse_arms(rows):
    arms = tuple(ArmSpec(row['name'], row['source'], row['seed']) for row in rows)
    if len(arms) < 2:
        raise ValueError('a matched comparison needs at least two arms')
    if len({arm.name for arm in arms}) != len(arms):
        raise ValueError('arm names must be distinct')
    return arms


def parse_pairs(text, arms):
    """``"B-A,A2-A"`` -> ``(('B', 'A'), ('A2', 'A'))``; every name must be an arm."""
    names = {arm.name for arm in arms}
    pairs = []
    for item in (part.strip() for part in text.split(',') if part.strip()):
        left, separator, right = item.partition('-')
        if not separator or left not in names or right not in names or left == right:
            raise ValueError('pair %r must name two different arms as treatment-control' % item)
        pairs.append((left, right))
    if not pairs:
        raise ValueError('at least one treatment-control pair is required')
    return tuple(pairs)


# -- counting ---------------------------------------------------------------

def task_counts(rows):
    """``{task_ref: (successes, valid, attempted)}`` from ``sample_rows`` output; UNKNOWN is not valid."""
    counts = {}
    for row in rows:
        success, valid, attempted = counts.get(row['task_ref'], (0, 0, 0))
        outcome = str(row['outcome']).lower()
        counts[row['task_ref']] = (success + (outcome == 'success'),
                                   valid + (outcome in ('success', 'failure')), attempted + 1)
    return counts


def select_probes(screen, *, n_recovery, n_retention, seed, fail_max=0.5, success_min=0.75, shrink_to=None):
    """Freeze Recovery (lowest screened success) and Retention (random among mastered) banks.

    ``screen`` is ``task_counts`` of a screening evaluation at the current checkpoint. Its
    samples are not reused as the "before" measurement, so selecting on failure does not
    bias G by regression to the mean.

    As the Solver improves, fewer tasks stay failing. With ``shrink_to`` (a minimum bank size) both
    banks shrink to what the pool offers, equal in size, instead of the selection failing.
    """
    if not (0 <= fail_max < success_min <= 1):
        raise ValueError('need 0 <= fail_max < success_min <= 1')
    rng = random.Random(seed)
    rated = [(ref, success / valid) for ref, (success, valid, _) in sorted(screen.items()) if valid]
    failing = [(rate, rng.random(), ref) for ref, rate in rated if rate <= fail_max]
    mastered = sorted(ref for ref, rate in rated if rate >= success_min)
    if shrink_to is not None and (len(failing) < n_recovery or len(mastered) < n_retention):
        size = min(len(failing), len(mastered), n_recovery, n_retention)
        if size >= shrink_to:
            n_recovery = n_retention = size
    if len(failing) < n_recovery or len(mastered) < n_retention:
        raise ValueError('probe pool too small: %d failing (need %d), %d mastered (need %d)'
                         % (len(failing), n_recovery, len(mastered), n_retention))
    recovery = tuple(ref for _, _, ref in sorted(failing)[:n_recovery])
    retention = tuple(rng.sample(mastered, n_retention))
    return ProbeSet(recovery, retention)


# -- scoring ------------------------------------------------------------------

def _rate(counts, refs):
    rates = [counts[ref][0] / counts[ref][1] for ref in refs]
    return sum(rates) / len(rates)


def _usable(refs, measurements):
    """Tasks with at least one valid outcome in every measurement (before and each arm)."""
    return tuple(ref for ref in refs if all(ref in m and m[ref][1] > 0 for m in measurements))


def _coverage(counts, refs):
    attempted = sum(counts[ref][2] for ref in refs if ref in counts)
    valid = sum(counts[ref][1] for ref in refs if ref in counts)
    return valid / attempted if attempted else 0.0


def dual_bank_score(before, after, recovery, retention, *, lambda_retention):
    gain = _rate(after, recovery) - _rate(before, recovery)
    retention_delta = _rate(after, retention) - _rate(before, retention)
    return {'gain': gain, 'retention_delta': retention_delta,
            'reward': gain - lambda_retention * max(0.0, -retention_delta)}


def _percentile(values, q):
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def compare_arms(before, arms, probes, pairs, *, lambda_retention=0.5, min_valid_fraction=0.8,
                 n_boot=2000, seed=0):
    """Score every arm and every treatment-control pair, with a paired task bootstrap.

    ``before`` and each value of ``arms`` are ``task_counts``. The bootstrap resamples the
    tasks of each bank with replacement, uses the same resample for before and every arm,
    and recomputes the reward difference, so its spread is the probe-sampling noise of the
    difference itself.
    """
    measurements = [before, *arms.values()]
    coverage = {'before': {bank: _coverage(before, refs) for bank, refs
                           in (('recovery', probes.recovery), ('retention', probes.retention))}}
    for name, counts in arms.items():
        coverage[name] = {bank: _coverage(counts, refs) for bank, refs
                          in (('recovery', probes.recovery), ('retention', probes.retention))}
    low = min(value for row in coverage.values() for value in row.values())
    recovery, retention = _usable(probes.recovery, measurements), _usable(probes.retention, measurements)
    report = {'lambda_retention': lambda_retention, 'coverage': coverage,
              'usable_tasks': {'recovery': len(recovery), 'retention': len(retention)},
              'valid': low >= min_valid_fraction and bool(recovery) and bool(retention)}
    if not report['valid']:
        report['reason'] = ('valid outcome coverage %.3f is below %.3f' % (low, min_valid_fraction)
                            if low < min_valid_fraction else 'a probe bank has no usable task')
        return report
    score = lambda counts, rec, ret: dual_bank_score(before, counts, rec, ret,
                                                     lambda_retention=lambda_retention)
    report['before'] = {'recovery': _rate(before, recovery), 'retention': _rate(before, retention)}
    report['arms'] = {name: score(counts, recovery, retention) for name, counts in arms.items()}
    rng = random.Random(seed)
    draws = {pair: [] for pair in pairs}
    for _ in range(n_boot):
        rec = [rng.choice(recovery) for _ in recovery]
        ret = [rng.choice(retention) for _ in retention]
        rewards = {name: score(counts, rec, ret)['reward'] for name, counts in arms.items()}
        for left, right in pairs:
            draws[(left, right)].append(rewards[left] - rewards[right])
    report['pairs'] = {}
    for left, right in pairs:
        values = draws[(left, right)]
        mean = sum(values) / len(values)
        sd = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5 if len(values) > 1 else 0.0
        report['pairs']['%s-%s' % (left, right)] = {
            'reward_difference': report['arms'][left]['reward'] - report['arms'][right]['reward'],
            'gain_difference': report['arms'][left]['gain'] - report['arms'][right]['gain'],
            'retention_difference': (report['arms'][left]['retention_delta']
                                     - report['arms'][right]['retention_delta']),
            'bootstrap_sd': sd, 'ci95': [_percentile(values, 0.025), _percentile(values, 0.975)],
            'n_boot': n_boot}
    return report


# -- packages -----------------------------------------------------------------

def step_tasks(arm, step, *, train_pairs, designer_refs, tasks_per_step, mix_fraction):
    """The ``(source, task_ref)`` list one training step of ``arm`` uses; reproducible per arm and step."""
    from rift.runtime.curriculum_run import designer_round
    rng = random.Random('%s:%d:%d' % (arm.name, arm.seed, step))
    if arm.source == 'random':
        pool = list(train_pairs)
        return rng.sample(pool, tasks_per_step) if len(pool) >= tasks_per_step else rng.choices(pool, k=tasks_per_step)
    if not designer_refs:
        raise ValueError('arm %s needs Designer tasks' % arm.name)
    # A task may appear once per package (its rollout ids come from package, task and sample index),
    # so when the Teacher wrote fewer new tasks than its share of a step, TRAIN tasks fill the rest.
    designer_refs = list(dict.fromkeys(designer_refs))
    n_mix = round(tasks_per_step * mix_fraction) if train_pairs else 0
    share = min(tasks_per_step - n_mix, len(designer_refs))
    chosen = [('designer', ref) for ref in rng.sample(designer_refs, share)]
    pool = list(train_pairs)
    chosen += rng.sample(pool, min(tasks_per_step - share, len(pool)))
    if len(chosen) < tasks_per_step:
        raise ValueError('too few distinct tasks to fill a step of %d' % tasks_per_step)
    return chosen


def train_step(context, checkpoint, chosen, *, index, seed, persist, label):
    """Sample one GRPO package for ``chosen`` and run one real update; returns ``(after, row)``."""
    import time
    from rift.runtime.curriculum_run import (PoolBatch, PoolGroup, RoundBackend, _pool_collector,
                                             fit_request_to_batches)
    started = time.perf_counter()
    groups = [PoolGroup(ref, sha256(('rift-arm-group-v1:%s:%d:%d:%s' % (label, index, position, ref)
                                     ).encode()).hexdigest()[:32], context.group_size, source)
              for position, (source, ref) in enumerate(chosen)]
    batch = PoolBatch('arm-%s-%06d-%s' % (label, index, sha256('|'.join(g.task_ref for g in groups).encode()
                                                              ).hexdigest()[:12]), groups, seed)
    request, _ = _pool_collector(context, checkpoint, index).collect(batch, checkpoint)
    collected = len(request.rollouts)
    rewards = {}
    source = {group.group_id: group.source for group in groups}
    for row in request.rollouts:
        rewards.setdefault(source[row.group_id], []).append(row.reward)
    counts, episodes = {}, []
    for row in request.rollouts:                  # what the Solver did on each task, for plan rewards and failures
        won, seen = counts.get(row.task_id, (0, 0))
        counts[row.task_id] = (won + int(row.reward >= 1), seen + 1)
        episodes.append({'task_ref': row.task_id, 'outcome': 'success' if row.reward >= 1 else 'failure',
                         'episode_id': row.episode_id})
    sampled = time.perf_counter()
    request = fit_request_to_batches(request, context.update_budget.mini_batch_size)
    update = RoundBackend(context, index, checkpoint, persist=persist).train(request)
    return update.policy_after, {
        'index': index, 'revision': update.policy_after.revision, 'rollouts': collected,
        'trained_rollouts': len(request.rollouts), 'task_counts': counts, 'episodes': episodes,
        'train_success': {key: sum(v) / len(v) for key, v in sorted(rewards.items())},
        'sample_seconds': round(sampled - started, 1),
        'update_seconds': round(time.perf_counter() - sampled, 1)}


# -- runner -------------------------------------------------------------------

def run_matched_arms(context, *, start, arms, probe_pool, train_pairs, designer_refs, steps,
                     tasks_per_step, mix_fraction, n_recovery, n_retention, seed, pairs,
                     screen_groups=1, probe_groups=2, lambda_retention=0.5, min_valid_fraction=0.8,
                     persist_last=False, n_boot=2000, log=None, index_start=0, label_prefix='', shrink_probes_to=None):
    """Fork every arm from ``start``, train ``steps`` updates each, score all on one frozen probe set.

    ``start`` must exist on disk: every arm after the first restores it inside the live
    runtime. The last arm's weights stay in the runtime afterwards (and are written to disk
    when ``persist_last``), so a curriculum can continue from the treatment arm.
    """
    if start in context.checkpoint_store.virtual:
        raise ValueError('the fork checkpoint must be written to disk; it exists only in memory')
    if steps < 1 or tasks_per_step < 1:
        raise ValueError('each arm needs at least one step of at least one task')
    probe_refs = {ref for refs in probe_pool.values() for ref in refs}
    train_refs = {ref for _, ref in train_pairs} | set(designer_refs or ())
    if probe_refs & train_refs:
        raise ValueError('%d probe-pool tasks are also training tasks' % len(probe_refs & train_refs))
    log = log or (lambda event: None)
    from rift.runtime.curriculum_run import sample_rows
    screen_rows = sample_rows(context, probe_pool, start, groups=screen_groups, seed=seed + 1,
                              label=label_prefix + 'screen')
    probes = select_probes(task_counts(screen_rows), n_recovery=n_recovery, n_retention=n_retention, seed=seed,
                           shrink_to=shrink_probes_to)
    log({'event': 'probes_frozen', 'recovery': len(probes.recovery), 'retention': len(probes.retention)})
    context.resync(start)
    before = task_counts(sample_rows(context, probes.by_key(), start, groups=probe_groups, seed=seed,
                                     label=label_prefix + 'before'))
    log({'event': 'before_measured'})
    after, history, index = {}, {}, index_start
    for position, arm in enumerate(arms):
        if position:
            context.restore(start)
        checkpoint, history[arm.name] = start, []
        for step in range(steps):
            chosen = step_tasks(arm, step, train_pairs=train_pairs, designer_refs=designer_refs,
                                tasks_per_step=tasks_per_step, mix_fraction=mix_fraction)
            last = position == len(arms) - 1 and step == steps - 1
            if step == 0 and position == 0:
                context.resync(checkpoint)       # the "before" evaluation left vLLM asleep
            checkpoint, row = train_step(context, checkpoint, chosen, index=index, seed=seed + 1000 * step,
                                         persist=persist_last and last, label=label_prefix + arm.name)
            history[arm.name].append(row)
            log({'event': 'arm_step', 'arm': arm.name, **row})
            index += 1
        # Same customer seeds as "before" (paired); the label keeps rollout identities distinct.
        after[arm.name] = task_counts(sample_rows(context, probes.by_key(), checkpoint, groups=probe_groups,
                                                  seed=seed, label=label_prefix + 'arm-' + arm.name))
        log({'event': 'arm_measured', 'arm': arm.name, 'revision': checkpoint.revision})
    report = compare_arms(before, after, probes, pairs, lambda_retention=lambda_retention,
                          min_valid_fraction=min_valid_fraction, n_boot=n_boot, seed=seed)
    return {'status': 'matched_arms_completed', 'start': asdict(start), 'final': asdict(checkpoint),
            'arms': [asdict(arm) for arm in arms], 'steps': steps, 'tasks_per_step': tasks_per_step,
            'mix_fraction': mix_fraction, 'probes': {'recovery': list(probes.recovery),
                                                     'retention': list(probes.retention)},
            'history': history, 'report': report, 'next_index': index}
