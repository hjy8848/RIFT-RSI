from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from usami.audit.report import render_report
from usami.audit.runner import ROOT, run_audit
from usami.world.generator import GeneratorConfig


def main():
    parser = argparse.ArgumentParser(description="Audit real Usami candidates; no LLM requests.")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/standard.json")
    parser.add_argument("--split-worlds", type=int, default=100)
    parser.add_argument("--validation-manifest", type=Path, default=ROOT / "configs/validation-manifest.json")
    parser.add_argument("--engine-checks", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("outputs/audit"))
    args = parser.parse_args()
    if args.count < 1 or args.split_worlds < 1 or args.engine_checks < 0:
        parser.error("count and split-worlds must be positive; engine-checks must be nonnegative")
    config = GeneratorConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    result = run_audit(range(args.start_seed, args.start_seed + args.count), config,
                       split_worlds=args.split_worlds, validation_path=args.validation_manifest,
                       engine_checks=args.engine_checks, progress=lambda s: print(s, file=sys.stderr, flush=True))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (args.output / "REPORT.md").write_text(render_report(result), encoding="utf-8")
    print(f"Saved {args.output / 'results.json'} and {args.output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
