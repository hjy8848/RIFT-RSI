import json
import os
from pathlib import Path
import subprocess
import sys

from usami.events import Trace

ROOT = Path(__file__).resolve().parents[1]


def cli(tmp_path, *arguments, expected=0):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    process = subprocess.run([sys.executable, "-m", "usami", *map(str, arguments)], cwd=tmp_path,
                             env=env, text=True, capture_output=True)
    assert process.returncode == expected, process.stderr
    return process.stdout


def test_cli_complete_flow(tmp_path):
    cli(tmp_path, "generate", "--seed", 42, "--count", 2, "--out", "worlds")
    text = cli(tmp_path, "inspect", "worlds/world_0001.json")
    assert "owner=" in text and "Rules" in text and "invoice_" in text
    result = json.loads(cli(tmp_path, "scenarios", "--world", "worlds/world_0001.json", "--out", "scenarios"))
    assert result["scenarios"] > 10
    report = json.loads(cli(tmp_path, "simulate", "--world", "worlds/world_0001.json", "--scenario", "scenarios/scenario_0001.json"))
    assert report["status"] == "pass"
    independently = json.loads(cli(tmp_path, "evaluate", "outputs/trace.json", "--out", "report.json"))
    assert independently == report
    replayed = json.loads(cli(tmp_path, "replay", "outputs/trace.json", "--out", "snapshot.json"))
    assert replayed["replayed"]
    trace = Trace.load(tmp_path / "outputs/trace.json")
    assert replayed["final_state_hash"] == trace.final_state_hash
    cli(tmp_path, "manifest", "--split", "validation", "--count", 1, "--out", "validation.json")
    result = json.loads(cli(tmp_path, "run-manifest", "validation.json", "--out", "benchmark"))
    assert result["passed"] == result["scenarios"] == 12
    result = json.loads(cli(tmp_path, "validate", "--seeds", 5, "--manifest", "validation.json"))
    assert result["valid"] and result["generated_worlds"] == 15


def test_cli_invalid_inputs_exit_nonzero(tmp_path):
    cli(tmp_path, "generate", "--count", 0, "--out", "worlds", expected=2)
    cli(tmp_path, "inspect", "missing.json", expected=2)
    cli(tmp_path, "validate", "--seeds", 0, expected=2)
