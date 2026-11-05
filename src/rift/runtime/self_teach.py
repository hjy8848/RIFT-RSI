"""The shared-weight Self-Taught Teacher loop (one model writes tasks, then learns from them).

One action, starting from weights theta_t that are on disk:

  1. the policy, prompted with its recent training failures, writes ``groups * plans`` plans;
  2. the environment certifies them; copies and repeated decision structures are set aside;
  3. the Solver trains ``steps`` GRPO steps on the new tasks mixed with ordinary TRAIN tasks;
  4. every plan is rewarded with the learnability the Solver showed on its tasks (dense, per plan);
  5. the policy is updated on the plans, off-policy, with each group's plans judged against each other,
     which gives theta_{t+1} (written to disk).

Every ``calibrate_every`` actions step 3 also runs a random-task control arm from theta_t and scores
both arms on frozen probes (``matched_arms``). That reward is slow and noisy, so it is not used to
update the Teacher; it is logged to show whether the dense reward tracks real learning.
See docs/PLAN_SELF_TAUGHT_TEACHER.md.
"""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

from rift.runtime import teacher as T
from rift.runtime.matched_arms import ArmSpec, run_matched_arms, step_tasks, train_step


def _log_factory(path):
    def log(event):
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, sort_keys=True, default=str) + '\n')
    return log


def failures_from_rows(context, rows, train_refs, max_cases):
    """Failure cases for the next prompt, from TRAIN-pool episodes only (never probes or Teacher tasks)."""
    from rift.designer.failures import failure_cases
    kept = [row for row in rows if row['task_ref'] in train_refs]
    return failure_cases(kept, context.world.task_registry, context.capture_store, max_cases=max_cases)


def prune_forks(forks, *, keep_last, keep_every):
    """Delete old fork checkpoints, keeping the last ``keep_last`` and every ``keep_every``-th action's.

    ``forks`` is a list of ``(action, directory)``; returns the directories removed. ``keep_last == 0``
    keeps everything: nothing is deleted unless the run was started asking for it.
    """
    removed = []
    if keep_last <= 0:
        return removed
    for position, (action, directory) in enumerate(forks):
        if directory is None or position >= len(forks) - keep_last or (keep_every and action % keep_every == 0):
            continue
        if Path(directory).is_dir():
            shutil.rmtree(directory)
            removed.append(str(directory))
    return removed


def teacher_update(context, samples, rewards, policy_before, ledger, index):
    """Off-policy GRPO on the plans; the result is written to disk as the next fork."""
    import os
    from rift.runtime.curriculum_run import RoundBackend
    # Every worker takes whole micro-batches, so the batch is a multiple of workers x micro-batch size.
    unit = context.checkpoint_store.world_size * int(os.environ.get('RIFT_PPO_MICRO_BATCH', '1'))
    request = T.build_teacher_update(samples, rewards, policy_before, ledger, world_size=unit)
    backend = RoundBackend(context, index, policy_before, persist=True,
                           rollout_verifier=T.TeacherRolloutVerifier(ledger),
                           mini_batch_size=len(request.rollouts))
    return backend.train(request), len(request.rollouts)


def generate_plans(context, checkpoint, failures, *, action, seed, groups, plans, max_tokens, constrained, ledger):
    """Wake the servers on ``checkpoint``'s weights and sample ``plans`` plans for each of ``groups`` prompts."""
    import asyncio
    from rift.designer.format import plan_schema
    runtime = context.resync(checkpoint)
    tokenizer = runtime.tokenizer
    schema = plan_schema() if constrained else None
    group_ids = ['a%03d-g%d' % (action, g) for g in range(groups)]
    prompt_seeds = [seed + 1000 * action + g for g in range(groups)]
    prompts = [T.teacher_prompt_ids(tokenizer, failures, seed=s) for s in prompt_seeds]

    def batch(client):
        async def run():
            return await asyncio.gather(*(T.generate_group(client, prompt, count=plans, seed=s, group_id=gid,
                                                           max_tokens=max_tokens, structured_schema=schema)
                                          for prompt, s, gid in zip(prompts, prompt_seeds, group_ids)))
        return run()
    generations = runtime.run_teacher(batch)
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=True)
    samples = []
    for gid, s, prompt, group in zip(group_ids, prompt_seeds, prompts, generations):
        made = T.make_samples(gid, checkpoint, prompt, group, decode, seed=s)
        for sample in made:
            ledger.record(sample)
        samples.extend(made)
    return samples, prompt_seeds


def run_solver_steps(context, checkpoint, *, action, index, seed, steps, tasks_per_step, mix_fraction,
                     train_pairs, teacher_refs):
    """Plain (non-calibration) action: the Solver trains ``steps`` steps on Teacher tasks mixed with TRAIN tasks."""
    arm = ArmSpec('T', 'designer', seed + action)
    history = []
    for step in range(steps):
        chosen = step_tasks(arm, step, train_pairs=train_pairs, designer_refs=teacher_refs,
                            tasks_per_step=tasks_per_step, mix_fraction=mix_fraction)
        checkpoint, row = train_step(context, checkpoint, chosen, index=index, seed=seed + 1000 * step,
                                     persist=False, label='a%03d-T' % action)
        history.append(row)
        index += 1
    return checkpoint, history, index


def self_teach(context, *, train_specs, probe_specs, actions, groups, plans, steps, tasks_per_step, mix_fraction,
               seed, calibrate_every=5, n_recovery=64, n_retention=64, screen_groups=1, probe_groups=2,
               lambda_retention=0.5, invalid_reward=-0.1, max_tokens=4096, constrained=True, max_failures=8,
               failure_groups=1, keep_checkpoints=0, keep_every=5, drop_references=(), init_from_run=None,
               failures_file=None):
    from rift.runtime.curriculum_run import _drop, sample_rows
    from rift.runtime.matched_arms import parse_pairs
    log_path = context.root / 'self-teach.jsonl'
    if log_path.exists():
        raise FileExistsError('this run directory already has a self-teach log; resuming is not supported, use a new one')
    log = _log_factory(log_path)
    if init_from_run and not (context.root / 'initial-checkpoint.json').exists():
        checkpoint = context.adopt_initial_from(init_from_run)
    else:
        checkpoint = context.initial_checkpoint()
    kinds = set(drop_references)
    train_by_key = context.world.certify_specs(train_specs)
    train_pairs = [(key, ref) for key in sorted(train_by_key) for ref in _drop(context, train_by_key[key], kinds)]
    train_refs = {ref for _, ref in train_pairs}
    probe_pool = {}
    if probe_specs and calibrate_every:
        probe_by_key = context.world.certify_specs(probe_specs)
        probe_pool = {key: kept for key in sorted(probe_by_key) if (kept := _drop(context, probe_by_key[key], kinds))}
    context.world.save()
    ledger = T.TeacherLedger(context.root / 'teacher')
    seen, forks, index, history = set(), [(0, None)], 0, []
    if failures_file:
        # Failure cases written earlier (collect-failures) for the starting policy: skips the sampling pass.
        failures = json.loads(Path(failures_file).read_text(encoding='utf-8'))['cases'][:max_failures]
    else:
        kept_by_key = {}
        for key, ref in train_pairs:
            kept_by_key.setdefault(key, []).append(ref)
        failure_rows = sample_rows(context, kept_by_key, checkpoint, groups=failure_groups, seed=seed,
                                   label='failures0')
        failures = failures_from_rows(context, failure_rows, train_refs, max_failures)
    log({'event': 'start', 'train': len(train_pairs), 'probe_pool': sum(map(len, probe_pool.values())),
         'failures': len(failures), 'checkpoint': asdict(checkpoint)})
    try:
        for action in range(actions):
            started = time.perf_counter()
            fork = checkpoint
            samples, prompt_seeds = generate_plans(context, fork, failures, action=action, seed=seed, groups=groups,
                                                   plans=plans, max_tokens=max_tokens, constrained=constrained,
                                                   ledger=ledger)
            generated = time.perf_counter()
            for group in {s.group_id: prompt_seeds[int(s.group_id.rsplit('-g', 1)[1])] for s in samples}.items():
                members = [s for s in samples if s.group_id == group[0]]
                T.certify_samples(context.world, members, prompt_seed=group[1], seen=seen)
            teacher_refs = [ref for s in samples for ref in s.selected_refs()]
            context.world.save()
            certified = time.perf_counter()
            calibrate = bool(calibrate_every and probe_pool and action % calibrate_every == 0)
            counts, episodes, report = {}, [], None
            if teacher_refs:
                context.resync(fork)                       # the generation put the servers to sleep
                if calibrate:
                    arms = (ArmSpec('C', 'random', seed + action), ArmSpec('T', 'designer', seed + action))
                    result = run_matched_arms(context, start=fork, arms=arms, probe_pool=probe_pool,
                        train_pairs=train_pairs, designer_refs=teacher_refs, steps=steps,
                        tasks_per_step=tasks_per_step, mix_fraction=mix_fraction, n_recovery=n_recovery,
                        n_retention=n_retention, seed=seed + 7 * action, pairs=parse_pairs('T-C', arms),
                        screen_groups=screen_groups, probe_groups=probe_groups,
                        lambda_retention=lambda_retention, persist_last=False, log=log,
                        index_start=index, label_prefix='a%03d-' % action)
                    rows, report = result['history']['T'], result['report']
                    from rift.schemas import Checkpoint
                    after_solver = Checkpoint(**result['final'])
                    index = result['next_index']
                else:
                    after_solver, rows, index = run_solver_steps(context, fork, action=action, index=index, seed=seed,
                        steps=steps, tasks_per_step=tasks_per_step, mix_fraction=mix_fraction,
                        train_pairs=train_pairs, teacher_refs=teacher_refs)
                for row in rows:
                    for ref, (won, seen_n) in row['task_counts'].items():
                        old = counts.get(ref, (0, 0))
                        counts[ref] = (old[0] + won, old[1] + seen_n)
                    episodes.extend(row['episodes'])
                solver_done = time.perf_counter()
            else:
                after_solver, solver_done = fork, certified
            rewards = T.plan_rewards(samples, counts, invalid_reward=invalid_reward)
            update, trained = teacher_update(context, samples, rewards, after_solver, ledger, index)
            index += 1
            checkpoint = update.policy_after
            forks.append((action + 1, context.root / 'checkpoints' / 'service' / ('global-step-%08d' % int(checkpoint.revision))))
            removed = prune_forks(forks, keep_last=keep_checkpoints, keep_every=keep_every)
            if episodes:
                failures = failures_from_rows(context, episodes[-4 * tasks_per_step * context.group_size:],
                                              train_refs, max_failures) or failures
            statuses = {}
            for s in samples:
                statuses[s.row.get('status')] = statuses.get(s.row.get('status'), 0) + 1
            measured = [r for r in rewards.values() if r is not None]
            row = {'event': 'action', 'action': action, 'fork_revision': fork.revision,
                   'next_revision': checkpoint.revision, 'plans': len(samples), 'plan_status': statuses,
                   'teacher_tasks': len(teacher_refs), 'measured_plans': len(measured),
                   'reward_mean': sum(measured) / len(measured) if measured else None,
                   'trained_plans': trained, 'calibrated': calibrate,
                   'calibration': None if report is None else report.get('pairs', report),
                   'pruned': removed,
                   'seconds': {'generate': round(generated - started, 1), 'certify': round(certified - generated, 1),
                               'solver': round(solver_done - certified, 1),
                               'teacher_update': round(time.perf_counter() - solver_done, 1)}}
            log(row)
            history.append(row)
    finally:
        context.close_runtime()
    return {'status': 'self_teach_completed', 'actions': history, 'final_checkpoint': asdict(checkpoint),
            'run_root': str(context.root)}
