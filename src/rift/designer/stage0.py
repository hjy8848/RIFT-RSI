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

from rift.designer.format import PlanError, plan_proposal, plan_schema
from rift.designer.prompt import designer_messages, example_plans


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
           'reasons': [], 'signatures': []}
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


def summarize(rows):
    statuses = Counter(row['status'] for row in rows)
    reasons = Counter(reason for row in rows for reason in row['reasons'])
    answers = Counter(answer for row in rows for answer in row['answers'])
    signatures = {sig for row in rows for sig in row['signatures'] if not sig.startswith('unmeasured')}
    certified = sum(len(row['task_refs']) for row in rows)
    return {'plans': len(rows), 'plan_status': dict(statuses),
            'plans_with_a_certified_task': sum(bool(row['task_refs']) for row in rows),
            'certified_tasks': certified, 'answers': dict(answers),
            'distinct_signatures': len(signatures), 'rejection_reasons': dict(reasons.most_common())}


def vllm_generate(model, messages_list, *, seed, temperature=1.0, max_tokens=4096, tensor_parallel_size=1,
                  constrained=True):
    """Generate one Designer response per prompt with vLLM, constrained to the plan schema."""
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(model), tensor_parallel_size=tensor_parallel_size, seed=seed)
    tokenizer = llm.get_tokenizer()
    prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False) for messages in messages_list]
    extra = {}
    if constrained:
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


def dry_run_generate(messages_list, *, seed):
    """Generator examples in place of model output, for a CPU rehearsal of everything else."""
    return [json.dumps(example_plans(seed + index, 1)[0].model_dump(mode='json'))
            for index in range(len(messages_list))]


def run(world_kit, generate, failures, *, samples, seed, out_dir, examples=2, signatures=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plans_path = out_dir / 'designer-plans.jsonl'
    if plans_path.exists():
        raise FileExistsError('this run directory already has Designer plans; use a new one')
    prompts = [designer_messages(failures, seed=seed + index, examples=examples) for index in range(samples)]
    texts = generate(prompts)
    if len(texts) != len(prompts):
        raise ValueError('the generator returned a different number of responses')
    rows = []
    with plans_path.open('w', encoding='utf-8') as stream:
        for index, text in enumerate(texts):
            row = certify_designer_plan(world_kit, text, f'designer-{seed}-{index:04d}', seed + index,
                                        signatures=signatures)
            row['response'] = text
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    world_kit.save()
    report = summarize(rows)
    (out_dir / 'stage0-report.json').write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    return report


def load_plans(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def certify_saved_plans(world_kit, path):
    """Re-certify the certified plans of a stage-0 file into ``world_kit``; returns their task refs.

    World construction and certification are deterministic, so the refs are the same ones.
    """
    from rift.runtime.curriculum_run import plan_identity
    from rift.tasks import Proposal
    refs = []
    for row in load_plans(path):
        if not row['task_refs']:
            continue
        plan = row['plan']
        proposal = Proposal(row['plan_id'], plan['hypothesis'], world_draft=plan['world_draft'],
                            challenges=tuple(plan['challenges']))
        result = world_kit.certify_plan(proposal, row['world_seed'],
                                        plan_identity('designer', {'plan_id': row['plan_id']}))
        found = [task.record_hash for task in result.tasks]
        if found != row['task_refs']:
            raise ValueError(f"plan {row['plan_id']} no longer certifies to the same tasks")
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
    parser.add_argument('--patience-slack', type=int)
    parser.add_argument('--no-signatures', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='use generator examples instead of a model')
    args = parser.parse_args(argv)
    failures = json.loads(Path(args.failures).read_text(encoding='utf-8'))['cases'] if args.failures else []
    if args.dry_run:
        generate = lambda prompts: dry_run_generate(prompts, seed=args.seed)
    elif args.designer_model:
        generate = lambda prompts: vllm_generate(args.designer_model, prompts, seed=args.seed,
            temperature=args.temperature, max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size, constrained=not args.unconstrained)
    else:
        parser.error('--designer-model is required unless --dry-run')
    kit = WorldKit(args.run_root, patience_slack=args.patience_slack)
    report = run(kit, generate, failures, samples=args.samples, seed=args.seed, out_dir=args.run_root,
                 examples=args.examples, signatures=not args.no_signatures)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
