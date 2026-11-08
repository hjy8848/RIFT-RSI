"""Failure cases of a served model on retail TRAIN tasks, in the form the Designer prompt reads.

    python scripts/collect_retail_failures.py --url http://127.0.0.1:8200/v1 --model solver \
        --instances 8 --samples 4 --out runs/retail-failures.json

Runs the model as the Service on retail tasks from the training seed range (the same specs as
configs/experiments/retail_train.json by default), then writes ``{"cases": [...]}`` with, for the tasks it
failed most: the request, the hidden goals with their derived answers, the success rate, the steps of one
failed attempt and its last message. Use the file with ``rift.designer.stage0 --failures`` or
``self-teach --failures-file``. One case per family is taken first, so the Designer sees varied weaknesses.
Only TRAIN-range seeds are used; held-out and probe ranges never enter the prompt.
"""
import argparse
import collections
import concurrent.futures
import json
import sys


def run_one(args, family, seed, sample):
    from rift.designer.failures import agent_steps, goal_rows
    from rift.runtime.llm_customer import retrying_client
    from usami.service.models import ModelSettings
    from usami_v2.runtime.agents import OpenAIServiceAgent
    from usami_v2.runtime.retail_runner import run_retail_episode
    from usami_v2.scenarios.retail_tasks import generate_retail_task
    world, task, _ = generate_retail_task(seed, family)
    options = {'max_tokens': args.max_tokens, 'temperature': args.temperature, 'top_p': 1.0,
               'seed': (seed * 1009 + sample) % (2 ** 31)}
    if not args.thinking:
        options['chat_template_kwargs'] = {'enable_thinking': False}
    settings = ModelSettings(model=args.model, base_url=args.url, timeout_seconds=args.timeout, options=options)
    agent = OpenAIServiceAgent(retrying_client(settings), tool_profile='retail')
    result = run_retail_episode(world, task, agent, max_service_decisions=24, max_tool_calls=16)
    ok = result.status == 'completed' and result.success
    case = None
    if not ok:
        record = result.record
        messages = [row.text for row in record.service_messages]
        case = {'request': task.customer_request, 'goals': goal_rows(world, task),
                'agent_steps': agent_steps(record), 'last_message': messages[-1] if messages else None}
    return {'family': family, 'seed': seed, 'ok': ok, 'case': case}


def main(argv=None):
    from usami_v2.scenarios.retail_tasks import FAMILIES
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--families', nargs='*', default=list(FAMILIES))
    parser.add_argument('--seed-start', type=int, default=20_000_000, help='first TRAIN-range seed')
    parser.add_argument('--instances', type=int, default=8, help='tasks per family')
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--max-cases', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if args.seed_start >= 29_000_000 or args.seed_start < 20_000_000:
        sys.exit('failure cases may only come from the TRAIN seed range (20,000,000 to 28,999,999)')
    jobs = [(f, args.seed_start + i, k) for f in args.families for i in range(args.instances)
            for k in range(args.samples)]
    rows = []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_one, args, *job) for job in jobs]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            rows.append(future.result())
            if done % 100 == 0:
                print('%d/%d episodes' % (done, len(jobs)), file=sys.stderr)
    by_task = collections.defaultdict(list)
    for row in rows:
        by_task[(row['family'], row['seed'])].append(row)
    ranked = []
    for (family, seed), group in by_task.items():
        failed = [row for row in group if not row['ok']]
        if failed:
            ranked.append((sum(row['ok'] for row in group) / len(group), family, seed, failed[0]['case']))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    chosen, used = [], set()
    for round_ in range(2):
        for rate, family, seed, case in ranked:
            if len(chosen) >= args.max_cases or (family, seed) in used:
                continue
            if round_ == 0 and any(c['family'] == family for c in chosen):
                continue
            chosen.append({'family': family, 'seed': seed, 'success_rate': rate, **case})
            used.add((family, seed))
    cases = [{key: value for key, value in case.items() if key not in ('family', 'seed')} for case in chosen]
    meta = [{'family': c['family'], 'seed': c['seed'], 'success_rate': c['success_rate']} for c in chosen]
    with open(args.out, 'w', encoding='utf-8') as sink:
        json.dump({'cases': cases, 'sources': meta, 'episodes': len(rows),
                   'success_rate': sum(r['ok'] for r in rows) / len(rows)}, sink, ensure_ascii=False, indent=1)
    print('episodes %d success %.3f, %d cases from families %s' % (
        len(rows), sum(r['ok'] for r in rows) / len(rows), len(cases), sorted({m['family'] for m in meta})))


if __name__ == '__main__':
    main()
