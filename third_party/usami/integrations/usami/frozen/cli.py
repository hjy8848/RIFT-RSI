"""Offline pool audit or the shared run/replay/evaluate CLI."""
import argparse
import json
import sys
from pathlib import Path
from vast_coevo.designer.storage import atomic_json
from .controller import FrozenPlan,preparation
from .pool import WorldPool


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    if not args or args[0]!='audit':
        from integrations.usami.designer.cli import main as shared
        return shared(args)
    parser=argparse.ArgumentParser(description='Generate, freeze and audit worlds; no model requests.')
    parser.add_argument('command',choices=['audit']);parser.add_argument('--plan',required=True);parser.add_argument('--root',required=True)
    value=parser.parse_args(args);plan=FrozenPlan.model_validate_json(Path(value.plan).read_text())
    root=Path(value.root);root.mkdir(parents=True,exist_ok=False)
    pool=WorldPool(plan.world_specs);pool.save(root);data=preparation(plan.world_specs,plan.budget)
    from integrations.usami.adapter import source_hashes
    atomic_json(root/'audit-source-pins.json',source_hashes(),immutable=True)
    atomic_json(root/'audit-plan.json',plan,immutable=True);atomic_json(root/'world-pool-audit.json',data['audit'],immutable=True)
    print(json.dumps(data['audit'],ensure_ascii=False,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
