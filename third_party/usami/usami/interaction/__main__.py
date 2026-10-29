"""Real two-agent CLI; replay does not instantiate either provider client."""
import argparse
import json
import os
from pathlib import Path
import sys

from usami.interaction import InteractionConfig, InteractionSession, replay_interaction
from usami.service import OpenAICompatibleClient
from usami.service.models import strict_json


def provider_environment(role, env=None):
    if role not in {"customer", "service"}:
        raise ValueError("unsupported provider role")
    env = os.environ if env is None else env
    resolved = {k: v for k, v in env.items() if k.startswith("USAMI_LLM_") or k.startswith("OPENAI_")}
    prefix = "USAMI_" + role.upper() + "_LLM_"
    for key, value in env.items():
        if key.startswith(prefix):
            resolved["USAMI_LLM_" + key[len(prefix):]] = value
    # An explicitly chosen role Keychain must override a shared environment key.
    if env.get(prefix + "KEYCHAIN_SERVICE") and not env.get(prefix + "API_KEY"):
        resolved.pop("USAMI_LLM_API_KEY", None)
        resolved.pop("OPENAI_API_KEY", None)
    if role == "customer":
        options = strict_json(resolved.get("USAMI_LLM_OPTIONS", '{"max_tokens":2048}'))
        if not isinstance(options, dict):
            raise ValueError("Customer LLM options must be a JSON object")
        options.setdefault("response_format", {"type": "json_object"})
        resolved["USAMI_LLM_OPTIONS"] = json.dumps(options)
    return resolved


def main(argv=None):
    parser = argparse.ArgumentParser(description="Usami: autonomous multi-turn Customer–Service exploration")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("episode", type=Path)
    replay.add_argument("--config", type=Path)
    replay.add_argument("--expected-hash")
    replay.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            config = InteractionConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
            customer = OpenAICompatibleClient.from_env(provider_environment("customer"))
            try:
                service = OpenAICompatibleClient.from_env(provider_environment("service"))
                try:
                    episode = InteractionSession(config, customer, service).run(on_turn=lambda turn:
                        print(f"{turn.prepared.turn_id}: Service {turn.service.status}", file=sys.stderr, flush=True))
                finally:
                    service.close()
            finally:
                customer.close()
            episode.save(args.out)
            print(json.dumps({"status": episode.status, "termination_reason": episode.termination_reason,
                "customer_model": episode.customer_model.model, "service_model": episode.service_model.model,
                "turns": [{"turn_id": t.prepared.turn_id, "customer_message": t.prepared.customer.message,
                           "service_reply": t.service.final_reply, "service_status": t.service.status} for t in episode.turns],
                "oracle_status": episode.service_execution.oracle.status, "metrics": episode.metrics,
                "interaction_hash": episode.interaction_hash, "out": str(args.out)}, ensure_ascii=False, indent=2))
            # A bounded exploration ending at an episode budget is a normal CLI run.
            return 0 if episode.status in {"completed", "budget_exhausted"} else 1
        config = InteractionConfig.model_validate_json(args.config.read_text(encoding="utf-8")) if args.config else None
        episode = replay_interaction(args.episode, config, args.expected_hash)
        if args.out:
            episode.save(args.out)
        print(json.dumps({"replayed": True, "interaction_hash": episode.interaction_hash,
            "final_state_hash": episode.service_execution.world_trace.final_state_hash,
            "oracle_status": episode.service_execution.oracle.status, "termination_reason": episode.termination_reason,
            "customer_turns": len(episode.turns), "engine_events": len(episode.service_execution.world_trace.events)}, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        print(f"usami-interaction: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
