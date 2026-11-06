"""Does the model customer behave? Replay recorded Service messages to it and check its decisions.

    python scripts/probe_llm_customer.py --episodes RUN/episodes --limit 200

Needs RIFT_CUSTOMER_URL (and RIFT_CUSTOMER_MODEL) to point at a running customer server and uses no
Service: for each recorded episode it shows the customer the real request and the Service's last reply, with
the episode's private profile, and compares the decision with what that profile requires. It reports how
often the model gives an unusable decision (a failed call ends a real episode as a customer protocol
error, which then counts against the Service) and how often it accepts or declines against its profile.
"""
import argparse
import collections
import glob
import json
import os
import random
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', required=True, help='an episodes directory of a finished or running run')
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    from rift.runtime.llm_customer import build_customer_factory, llm_customer_enabled
    from usami_v2.runtime.agents import AgentMessage, CustomerPrompt
    from usami_v2.runtime.customer import CustomerProfileV3
    if not llm_customer_enabled():
        sys.exit('set RIFT_CUSTOMER_URL (and RIFT_CUSTOMER_MODEL)')
    factory = build_customer_factory(None)
    files = sorted(glob.glob(os.path.join(args.episodes, '*.json')))
    random.Random(args.seed).shuffle(files)
    tally, errors, started, done = collections.Counter(), collections.Counter(), time.perf_counter(), 0
    for path in files:
        if done >= args.limit:
            break
        try:
            record = json.load(open(path))['episode_record']
            profile = CustomerProfileV3.model_validate(record['profile'])
            messages = record['service_messages']
            reply = (messages[-1].get('text') if messages else '') or ''
            request = record['task']['customer_request']
        except Exception:
            continue
        if not reply:
            continue
        done += 1
        prompt = CustomerPrompt(public_request=request, last_service_message=reply, conversation=(
            AgentMessage(role='user', content=request), AgentMessage(role='assistant', content=reply)))
        agent = factory.create('probe', seed=done)
        try:
            decision = agent.decide(prompt, profile)
        except Exception as exc:
            errors[type(exc).__name__ + ': ' + str(exc)[:80]] += 1
            continue
        wants_decline = not getattr(profile, 'accepts_proposals', True) or bool(getattr(profile, 'decline_line_ids', ()))
        tally[('profile declines' if wants_decline else 'profile accepts',
               'model declined' if decision.declines_proposal else 'model did not decline')] += 1
        tally[('finish', bool(decision.finish))] += 1
    total = done
    print('episodes probed', total, 'in %.0fs' % (time.perf_counter() - started))
    print('unusable decisions: %d (%.1f%%)' % (sum(errors.values()), 100 * sum(errors.values()) / max(1, total)))
    for message, count in errors.most_common(5):
        print('   ', count, message)
    for key, count in sorted(tally.items(), key=lambda kv: str(kv[0])):
        print('   ', key, count)


if __name__ == '__main__':
    main()
