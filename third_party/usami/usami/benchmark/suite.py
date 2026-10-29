"""Locked, atomic, resumable instance × trial execution with exact replay."""
import fcntl
import json
import os
from pathlib import Path
from usami.benchmark.manifest import validate_manifest
from usami.benchmark.runtime import BenchmarkRun, run_instance, replay_run
from usami.benchmark.scoring import summarize
from usami.benchmark.release_v2 import write_json


def run_suite(manifest, out, client_factory, *, max_runs=None, progress=None):
    validate_manifest(manifest)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    with (out / ".suite.lock").open("a") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError("another runner holds this output directory") from None
        evidence = {}
        for path in sorted(out.glob("*.run.json")):
            run = replay_run(manifest, BenchmarkRun.load(path)); key = (run.instance_id, run.trial_id)
            if key in evidence: raise ValueError("duplicate run evidence in resume directory")
            evidence[key] = run
        for path in sorted(out.glob("*.started.json")):
            started = json.loads(path.read_text())
            if started.get("manifest_hash") != manifest.manifest_hash:
                raise ValueError("unfinished attempt belongs to another frozen profile")
            key = (started.get("instance_id"), started.get("trial_id"))
            if key not in evidence:
                raise ValueError("unrecorded interrupted attempt cannot be silently retried; preserve this directory and preregister a fresh trial")
            path.unlink()
        executed = 0
        for instance in manifest.instances:
            for trial in manifest.trials:
                key = (instance.instance_id, trial)
                if key in evidence: continue
                if max_runs is not None and executed >= max_runs: break
                service, customer = client_factory(manifest, instance)
                started = out / f"{instance.instance_id}.{trial}.started.json"
                try:
                    if service.settings != manifest.service_model or (customer is not None and customer.settings != manifest.customer_model):
                        raise ValueError("client differs from frozen model profile")
                    write_json(started, {"manifest_hash": manifest.manifest_hash, "instance_id": instance.instance_id, "trial_id": trial})
                    run = run_instance(manifest, instance.instance_id, service, customer, trial)
                finally:
                    for client in (service, customer):
                        if client is not None and hasattr(client, "close"): client.close()
                # Flush and replace: an interrupted run cannot be mistaken for complete.
                target = out / f"{instance.instance_id}.{trial}.run.json"
                tmp = target.with_suffix(".tmp")
                run.save(tmp)
                with tmp.open("rb") as file: os.fsync(file.fileno())
                os.replace(tmp, target)
                started.unlink()
                evidence[key] = run; executed += 1
                if progress: progress(instance, trial, run)
        ordered = [evidence[(i.instance_id, t)] for i in manifest.instances for t in manifest.trials if (i.instance_id, t) in evidence]
        report = summarize(manifest, ordered)
        write_json(out / "summary.json", report)
        return report
