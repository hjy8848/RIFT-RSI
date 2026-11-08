"""How hard are the retail tasks for a model? Serve it with vLLM (OpenAI-compatible) and run this.

    python scripts/eval_retail_difficulty.py --url http://127.0.0.1:8200/v1 --model solver \
        --seeds 8 --samples 4 --workers 64 --out runs/retail-difficulty.jsonl

Each (family, seed, sample) is one episode: the model is the Service (``retail`` tool profile), the customer is
the scripted one, every tool call goes through the real Engine, and success means the final state equals the
golden state. Prints a table per family: success rate, how often the engine refused a call, and how long
episodes were. Nothing here trains or changes the model.
"""
import argparse
import collections
import concurrent.futures
import json
import sys
import time


def episode(url, model, family, seed, sample, args):
    from rift.runtime.llm_customer import retrying_client
    from usami.service.models import ModelSettings
    from usami_v2.runtime.agents import OpenAIServiceAgent
    from usami_v2.runtime.retail_runner import run_retail_episode
    from usami_v2.scenarios.retail_tasks import generate_retail_task
    world, task, evidence = generate_retail_task(seed, family)
    options = {'max_tokens': args.max_tokens, 'temperature': args.temperature, 'top_p': 1.0,
               'seed': (seed * 1009 + sample) % (2 ** 31)}
    if not args.thinking:
        options['chat_template_kwargs'] = {'enable_thinking': False}
    settings = ModelSettings(model=model, base_url=url, timeout_seconds=args.timeout, options=options)
    agent = OpenAIServiceAgent(retrying_client(settings), tool_profile=args.tool_profile)
    started = time.perf_counter()
    result = run_retail_episode(world, task, agent, max_service_decisions=args.max_decisions,
                                max_tool_calls=args.max_tools, patience=args.patience)
    errors = [r.get('error_code') for _, r in result.tool_log if not r.get('success')]
    return {'family': family, 'seed': seed, 'sample': sample, 'success': result.success,
            'state_match': result.state_match, 'told_total': result.told_total, 'status': result.status,
            'reason': result.terminal_reason, 'decisions': result.service_decisions, 'tools': result.tool_calls,
            'tool_names': [n for n, _ in result.tool_log], 'errors': errors, 'has_trap': evidence['has_trap'],
            'seconds': round(time.perf_counter() - started, 1)}


def main(argv=None):
    from usami_v2.scenarios.retail_tasks import FAMILIES
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--families', nargs='*', default=list(FAMILIES))
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--seed-start', type=int, default=1000)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--tool-profile', default='retail', choices=('retail', 'full9'))
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--max-decisions', type=int, default=24)
    parser.add_argument('--max-tools', type=int, default=16)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    jobs = [(family, args.seed_start + s, k) for family in args.families for s in range(args.seeds)
            for k in range(args.samples)]
    rows, started = [], time.perf_counter()
    with open(args.out, 'w') as sink, concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(episode, args.url, args.model, f, s, k, args): (f, s, k) for f, s, k in jobs}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                row = future.result()
            except Exception as exc:
                family, seed, sample = futures[future]
                row = {'family': family, 'seed': seed, 'sample': sample, 'success': False, 'status': 'harness_error',
                       'reason': type(exc).__name__ + ': ' + str(exc)[:200], 'errors': [], 'decisions': 0,
                       'tools': 0, 'tool_names': [], 'has_trap': None, 'seconds': 0}
            rows.append(row)
            sink.write(json.dumps(row) + '\n')
            sink.flush()
            if done % 50 == 0:
                print('%d/%d episodes, %.0fs' % (done, len(jobs), time.perf_counter() - started), file=sys.stderr)
    by = collections.defaultdict(list)
    for row in rows:
        by[row['family']].append(row)
    print('%-28s %5s %8s %8s %8s %8s' % ('family', 'n', 'success', 'refused', 'decide', 'harness'))
    for family in args.families:
        group = by[family]
        n = len(group)
        print('%-28s %5d %8.3f %8.3f %8.1f %8d' % (
            family, n, sum(r['success'] for r in group) / n,
            sum(bool(r['errors']) for r in group) / n, sum(r['decisions'] for r in group) / n,
            sum(r['status'] == 'harness_error' for r in group)))
    total = len(rows)
    print('%-28s %5d %8.3f' % ('ALL', total, sum(r['success'] for r in rows) / total))
    codes = collections.Counter(code for r in rows for code in r['errors'])
    print('engine refusals:', dict(codes.most_common(8)))
    print('statuses:', dict(collections.Counter(r['status'] for r in rows)))
    print('elapsed %.0fs' % (time.perf_counter() - started))


if __name__ == '__main__':
    main()
