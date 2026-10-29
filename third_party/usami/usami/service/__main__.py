"""CLI independent of the Web Explorer: run a real model or replay its record."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from usami.service import Experiment, OpenAICompatibleClient, ServiceExecution, ServiceSession, replay_execution


def load_experiment(path):
    return Experiment.model_validate_json(Path(path).read_text(encoding="utf-8"))


def parser():
    p = argparse.ArgumentParser(description="Usami Service: autonomous LLM tools with independent customer authorization")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="call the configured real OpenAI-compatible Service LLM")
    run.add_argument("--experiment", type=Path, required=True, help="trusted customer input and fixed experiment parameters")
    run.add_argument("--message", help="override the customer message; does not change write authorizations")
    run.add_argument("--customer", help="override fixed customer identity")
    run.add_argument("--seed", type=int)
    run.add_argument("--mode", choices=["reference", "sandbox"])
    run.add_argument("--out", type=Path, required=True)
    replay = sub.add_parser("replay", help="re-execute recorded model decisions and engine events, without an API key")
    replay.add_argument("execution", type=Path)
    replay.add_argument("--experiment", type=Path, help="external original trusted customer experiment for stronger authorization verification")
    replay.add_argument("--out", type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            experiment = load_experiment(args.experiment)
            data = experiment.model_dump(mode="json")
            for field in ("seed", "mode"):
                if getattr(args, field) is not None:
                    data[field] = getattr(args, field)
            for field, source in (("identity", "customer"), ("message", "message")):
                if getattr(args, source) is not None:
                    data["customer"][field] = getattr(args, source)
            experiment = Experiment.model_validate(data)
            client = OpenAICompatibleClient.from_env()
            try:
                execution = ServiceSession(experiment, client).run()
            finally:
                client.close()
            execution.save(args.out)
            print(json.dumps({"status": execution.status, "model": execution.model.model,
                "customer": execution.experiment.customer.identity, "final_reply": execution.final_reply,
                "oracle_status": execution.oracle.status, "metrics": execution.metrics,
                "execution_hash": execution.execution_hash, "out": str(args.out)}, ensure_ascii=False, indent=2))
            return 0 if execution.status == "completed" else 1
        else:
            expected = load_experiment(args.experiment) if args.experiment else None
            execution = replay_execution(args.execution, expected)
            if args.out:
                execution.save(args.out)
            print(json.dumps({"replayed": True, "execution_hash": execution.execution_hash,
                "final_state_hash": execution.world_trace.final_state_hash, "oracle_status": execution.oracle.status,
                "llm_calls_replayed": len(execution.llm_records), "engine_events": len(execution.world_trace.events)}, indent=2))
            return 0
    except (ValueError, OSError) as exc:
        print(f"usami-service: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
