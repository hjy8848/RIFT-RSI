"""Offline Core inspection, usable without any business-domain package."""
import argparse
import json
from .skills.library import LibraryStore
from .evidence.archive import EvidenceArchive


def main(argv=None):
    parser = argparse.ArgumentParser(description="VAST-CoEvo independent Core inspection")
    parser.add_argument("command", choices=("library", "evidence"))
    parser.add_argument("--store", required=True)
    parser.add_argument("--hash")
    args = parser.parse_args(argv)
    if args.command == "library":
        store = LibraryStore(args.store)
        value = store.load(args.hash).model_dump(mode="json") if args.hash else {"hashes": store.hashes()}
    else:
        archive = EvidenceArchive(args.store)
        value = archive.load(args.hash).model_dump(mode="json") if args.hash else archive.counts()
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0
