"""Stage 0: a frozen Designer writes plans; Usami certifies them; nothing is trained.

    python -m rift.designer.stage0 --run-root RUN --failures FAILURES.json \
        --designer-model /path/to/Qwen3-4B --samples 64

writes ``designer-plans.jsonl`` (every plan, its certified task refs and reasons) and
``stage0-report.json`` (certification rate, reason histogram, answer mix, signatures).
``--dry-run`` replaces the model by generator examples to rehearse everything else on CPU.
"""
from collections import Counter
import json
from pathlib import Path

from rift.designer.format import PlanError, extract_json, plan_proposal, plan_schema
from rift.designer.prompt import copy_score, designer_messages, example_plans

# A plan sharing at least this fraction of its money values with a prompt example is a copy.
COPY_THRESHOLD = 0.5


def _reason_codes(challenge):
    codes = []
    for attempt in challenge.get('attempts', ()):
        reason = attempt.get('reason_code') or attempt.get('reason') or attempt.get('native_status')
        if isinstance(reason, list):
            reason = reason[0] if reason else attempt.get('native_status')
        codes.append(str(reason).split(':')[0])
    return codes or [str(challenge.get('status'))]


def certify_designer_plan(world_kit, text, plan_id, world_seed, *, signatures=False):
    """Certify one Designer output. Returns a JSON-ready row; certified task refs in ``task_refs``."""
    from rift.runtime.curriculum_run import plan_identity
    row = {'plan_id': plan_id, 'world_seed': world_seed, 'task_refs': [], 'answers': [],
           'actions': [], 'reasons': [], 'signatures': []}
    try:
        proposal = plan_proposal(text, plan_id)
    except PlanError as exc:
        return {**row, 'status': 'format_rejected', 'reasons': [exc.code], 'message': str(exc)}
    row['plan'] = {'hypothesis': proposal.hypothesis, 'world_draft': proposal.world_draft,
                   'challenges': list(proposal.challenges)}
    result = world_kit.certify_plan(proposal, world_seed, plan_identity('designer', {'plan_id': plan_id}))
    if result.build.status != 'SAT':
        return {**row, 'status': 'world_rejected', 'reasons': [result.build.status], 'message': result.build.reason}
    for challenge in result.challenge_results:
        if challenge['status'] != 'CERTIFIED':
            row['reasons'].extend(_reason_codes(challenge))
    for task in result.tasks:
        row['task_refs'].append(task.record_hash)
        row['answers'].append('refusal' if task.refusal_goals else 'success')
        requests = getattr(task.request_bundle, 'requests', ())
        row['actions'].append(sorted({request.action for request in requests}))
        if signatures:
            from rift.probes.fingerprint import decision_fingerprint
            try:
                row['signatures'].append(decision_fingerprint(result.spec, result.build, task,
                    return_receipt_event_configured=True).record_hash)
            except (ValueError, KeyError) as exc:
                row['signatures'].append('unmeasured:' + type(exc).__name__)
    row['status'] = ('certified' if len(result.tasks) == len(result.challenge_results)
                     else 'partly_certified' if result.tasks else 'tasks_rejected')
    return row


def select_tasks(rows, copy_threshold=COPY_THRESHOLD, seen=None):
    """Mark which certified tasks to train on: not copied from an example, first of their signature.

    A task whose signature could not be measured counts as unique. Sets ``row['selected']``
    (one boolean per certified task, in ``task_refs`` order) and returns the number selected.
    ``seen`` carries signatures from earlier rounds (it is updated in place), so a Teacher is
    not credited for writing the same decision structure again.
    """
    seen = set() if seen is None else seen
    total = 0
    for row in rows:
        flags = []
        for index in range(len(row['task_refs'])):
            signature = row['signatures'][index] if index < len(row['signatures']) else None
            measured = signature is not None and not signature.startswith('unmeasured')
            keep = (row.get('copy_score') or 0) < copy_threshold and not (measured and signature in seen)
            if keep and measured:
                seen.add(signature)
            flags.append(keep)
        row['selected'] = flags
        total += sum(flags)
    return total


def summarize(rows):
    statuses = Counter(row['status'] for row in rows)
    reasons = Counter(reason for row in rows for reason in row['reasons'])
    answers = Counter(answer for row in rows for answer in row['answers'])
    signatures = {sig for row in rows for sig in row['signatures'] if not sig.startswith('unmeasured')}
    certified = sum(len(row['task_refs']) for row in rows)
    scores = [row['copy_score'] for row in rows if row.get('copy_score') is not None]
    copied = [row for row in rows if (row.get('copy_score') or 0) >= COPY_THRESHOLD]
    report = {'plans': len(rows), 'plan_status': dict(statuses),
              'plans_with_a_certified_task': sum(bool(row['task_refs']) for row in rows),
              'certified_tasks': certified, 'answers': dict(answers),
              'distinct_signatures': len(signatures), 'rejection_reasons': dict(reasons.most_common())}
    if scores:
        report.update(mean_copy_score=round(sum(scores) / len(scores), 3),
                      copied_plans=len(copied), copy_threshold=COPY_THRESHOLD,
                      certified_not_copied=sum(bool(row['task_refs']) for row in rows
                                               if (row.get('copy_score') or 0) < COPY_THRESHOLD))
    actions = Counter(action for row in rows for task_actions in row.get('actions', ()) for action in task_actions)
    multi = sum(len(task_actions) > 1 for row in rows for task_actions in row.get('actions', ()))
    report.update(actions=dict(actions), tasks_with_several_actions=multi)
    if any('selected' in row for row in rows):
        picked = [(row, index) for row in rows for index, flag in enumerate(row.get('selected', ())) if flag]
        report.update(selected_tasks=len(picked),
                      selected_answers=dict(Counter(row['answers'][index] for row, index in picked)),
                      selected_actions=dict(Counter(action for row, index in picked
                                                    for action in row['actions'][index])))
    focused = [row for row in rows if row.get('focus_hit') is not None]
    if focused:
        report.update(focus_obeyed=sum(bool(row['focus_hit']) for row in focused), focus_asked=len(focused))
    return report


def vllm_generate(model, messages_list, *, seed, temperature=1.0, max_tokens=4096, tensor_parallel_size=1,
                  constrained=True, enforce_eager=False, gpu_memory_utilization=0.9, plan_format='legacy'):
    """Generate one Designer response per prompt with vLLM, constrained to the plan schema."""
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(model), tensor_parallel_size=tensor_parallel_size, seed=seed,
              enforce_eager=enforce_eager, gpu_memory_utilization=gpu_memory_utilization)
    tokenizer = llm.get_tokenizer()
    prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False) for messages in messages_list]
    extra = {}
    if constrained:
        if plan_format == 'retail':
            from rift.designer.retail_format import retail_plan_schema
            schema = retail_plan_schema()
        else:
            schema = plan_schema()
        try:
            from vllm.sampling_params import StructuredOutputsParams
            extra = {'structured_outputs': StructuredOutputsParams(json=schema)}
        except ImportError:  # vLLM before structured_outputs
            from vllm.sampling_params import GuidedDecodingParams
            extra = {'guided_decoding': GuidedDecodingParams(json=schema)}
    params = [SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_tokens, seed=seed + index, **extra)
              for index in range(len(prompts))]
    return [output.outputs[0].text for output in llm.generate(prompts, params)]


def dry_run_generate(messages_list, *, seed, plan_format='legacy'):
    """Generator examples in place of model output, for a CPU rehearsal of everything else."""
    if plan_format == 'retail':
        from rift.designer.retail_prompt import example_retail_plans as examples
    else:
        examples = example_plans
    return [json.dumps(examples(seed + index, 1)[0].model_dump(mode='json'))
            for index in range(len(messages_list))]


def run(world_kit, generate, failures, *, samples, seed, out_dir, examples=2, signatures=True, focus=False, plan_format='legacy'):
    messages_fn, certify_fn, copy_fn, examples_fn = designer_messages, certify_designer_plan, copy_score, example_plans
    if plan_format == 'retail':
        from rift.designer.retail_prompt import retail_designer_messages, retail_copy_score, example_retail_plans
        from rift.designer.retail_stage0 import certify_retail_plan
        messages_fn, certify_fn, copy_fn, examples_fn = (retail_designer_messages, certify_retail_plan,
                                                         retail_copy_score, example_retail_plans)
    elif plan_format != 'legacy':
        raise ValueError('unknown Designer plan format')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plans_path = out_dir / 'designer-plans.jsonl'
    if plans_path.exists():
        raise FileExistsError('this run directory already has Designer plans; use a new one')
    prompts = [messages_fn(failures, seed=seed + index, examples=examples, focus=focus)
               for index in range(samples)]
    texts = generate(prompts)
    if len(texts) != len(prompts):
        raise ValueError('the generator returned a different number of responses')
    rows = []
    for index, text in enumerate(texts):
        row = certify_fn(world_kit, text, f'designer-{seed}-{index:04d}', seed + index,
                                    signatures=signatures)
        row['response'] = text
        parsed = None
        try:
            parsed = extract_json(text)
            row['copy_score'] = copy_fn(parsed, examples_fn(seed + index, examples))
        except PlanError:
            row['copy_score'] = None
        if focus and plan_format == 'legacy':
            from rift.designer.focus import focus_for, focus_hit
            row['focus'] = focus_for(seed + index)
            row['focus_hit'] = None if parsed is None else focus_hit(row['focus'], parsed)
        rows.append(row)
    select_tasks(rows)          # needs every row's signature, so the file is written afterwards
    with plans_path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    world_kit.save()
    report = summarize(rows)
    (out_dir / 'stage0-report.json').write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    return report


def load_plans(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def certify_saved_plans(world_kit, path, *, selected_only=False):
    """Re-certify the certified plans of a stage-0 file into ``world_kit``; returns their task refs.

    World construction and certification are deterministic, so the refs are the same ones.
    ``selected_only`` keeps only the tasks stage 0 marked for training (not copied, distinct signature).
    """
    from rift.runtime.curriculum_run import plan_identity
    from rift.tasks import Proposal
    refs = []
    for row in load_plans(path):
        if not row['task_refs']:
            continue
        plan = row['plan']
        if 'gift_card_balance_cents' in plan:
            from rift.designer.retail_stage0 import certify_retail_plan
            result = certify_retail_plan(world_kit, plan, row['plan_id'], row['world_seed'], signatures=False)
            found = result['task_refs']
            if result['status'] != 'certified' or found != row['task_refs']:
                raise ValueError(f"retail plan {row['plan_id']} no longer certifies to the same tasks")
            if selected_only:
                if 'selected' not in row:
                    raise ValueError('this stage-0 file has no selection; rerun stage 0 or drop --designer-selected-only')
                found = [ref for ref, keep in zip(found, row['selected']) if keep]
            refs.extend(found)
            continue
        proposal = Proposal(row['plan_id'], plan['hypothesis'], world_draft=plan['world_draft'],
                            challenges=tuple(plan['challenges']))
        result = world_kit.certify_plan(proposal, row['world_seed'],
                                        plan_identity('designer', {'plan_id': row['plan_id']}))
        found = [task.record_hash for task in result.tasks]
        if found != row['task_refs']:
            raise ValueError(f"plan {row['plan_id']} no longer certifies to the same tasks")
        if selected_only:
            if 'selected' not in row:
                raise ValueError('this stage-0 file has no selection; rerun stage 0 or drop --designer-selected-only')
            found = [ref for ref, keep in zip(found, row['selected']) if keep]
        refs.extend(found)
    return refs


def main(argv=None):
    import argparse
    from rift.runtime.curriculum_run import WorldKit
    parser = argparse.ArgumentParser(description='Stage 0: certify plans from a frozen Designer')
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--failures', help='failures.json from curriculum_run collect-failures')
    parser.add_argument('--designer-model', help='Designer weights (Qwen3-4B); omit with --dry-run')
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--seed', type=int, default=7000000)
    parser.add_argument('--examples', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--unconstrained', action='store_true', help='do not constrain decoding to the schema')
    parser.add_argument('--enforce-eager', action='store_true',
        help='skip torch.compile and cudagraph capture: faster start-up, the verified GPU path')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
        help='share of each GPU vLLM may take (lower it on a shared machine)')
    parser.add_argument('--patience-slack', type=int)
    parser.add_argument('--no-signatures', action='store_true')
    parser.add_argument('--focus', action='store_true',
        help='tell each plan what to cover (action, rule, structure, side, naming), chosen by seed')
    parser.add_argument('--dry-run', action='store_true', help='use generator examples instead of a model')
    parser.add_argument('--plan-format', choices=('legacy', 'retail'), default='legacy')
    args = parser.parse_args(argv)
    failures = json.loads(Path(args.failures).read_text(encoding='utf-8'))['cases'] if args.failures else []
    if args.dry_run:
        generate = lambda prompts: dry_run_generate(prompts, seed=args.seed, plan_format=args.plan_format)
    elif args.designer_model:
        generate = lambda prompts: vllm_generate(args.designer_model, prompts, seed=args.seed,
            temperature=args.temperature, max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size, constrained=not args.unconstrained,
            enforce_eager=args.enforce_eager, gpu_memory_utilization=args.gpu_memory_utilization,
            plan_format=args.plan_format)
    else:
        parser.error('--designer-model is required unless --dry-run')
    kit = WorldKit(args.run_root, patience_slack=args.patience_slack)
    report = run(kit, generate, failures, samples=args.samples, seed=args.seed, out_dir=args.run_root,
                 examples=args.examples, signatures=not args.no_signatures, focus=args.focus, plan_format=args.plan_format)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
