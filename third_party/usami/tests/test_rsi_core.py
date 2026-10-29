"""Real container integration tests; fixtures are not model-effect experiments."""
from dataclasses import asdict, replace
import json
import subprocess

import pytest

from rsi_core.archive import Archive
from rsi_core.budget import Ledger
from rsi_core.isolation import DockerWorker
from rsi_core.programs import ProgramArtifact, ProgramStore
from rsi_core.protocol import (
    AdaptationPackage, BudgetExceeded, FrozenModels, Limits, ModelSnapshot,
    TaskResult, validate_partitions,
)
from rsi_core.runtime import ResearchHost


IMAGE = "sha256:f06d01d4ba3f8ad5af45a3ec87df19c1d317d8e8961edc0fabda61811af6f7cf"


@pytest.fixture
def host(tmp_path):
    limits = Limits(wall_seconds=60, cpu_seconds=15)
    worker = DockerWorker(IMAGE, limits)
    model = ModelSnapshot("offline-contract-fixture", "not-a-trained-model")
    return ResearchHost(ProgramStore(tmp_path / "programs"), Archive(tmp_path / "archive"),
                        worker, None, FrozenModels(model, model, model), Ledger(limits))


def researcher(body):
    return ProgramArtifact("M", "def next_step(view, state):\n" + "\n".join("    " + line for line in body.splitlines()))


def task():
    return ProgramArtifact("P", "def step(view, state):\n    return {'kind': 'END', 'state': state}\n")


def noop():
    return researcher("return {'action':'SUBMIT','state':{},'candidate_ref':view['target']['ref']}")


def test_worker_uses_real_os_boundary_and_explicit_state(host, tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_HOST_API_KEY", "PRIVATE_CREDENTIAL_SENTINEL")
    secret = tmp_path / "private-answer"
    secret.write_text("PRIVATE_SENTINEL")
    program = researcher(
        "import os\n"
        "assert not os.path.exists(" + repr(str(secret)) + ")\n"
        "assert 'RSI_HOST_API_KEY' not in os.environ\n"
        "return {'action':'SUBMIT','state':{'observed':view['public']},'candidate_ref':view['target']['ref']}"
    )
    result = host.run(program, task(), "ADAPT_TASK", {"public": "allowed"}, (), None)
    assert result.status == "NOOP"
    assert host.verify(result)["verified"]
    inputs = [row["event"]["input"] for row in result.events if row["event"]["kind"] == "decision"]
    assert "PRIVATE_SENTINEL" not in json.dumps(inputs)
    assert not any(p.name.startswith("rsi-") for p in tmp_path.iterdir())


def test_stage_probe_freezes_all_new_candidate_ids(host):
    source = task().source
    editor = researcher(
        "n=state.get('n',0)\n"
        "if n==0:\n"
        "    return {'action':'STAGE','state':{'n':1},'kind':'P','source':" + repr(source)
        + ",'parent_ref':view['target']['ref']}\n"
        "if n==1:\n"
        "    ref=view['observations'][-1]['candidate_ref']\n"
        "    return {'action':'PROBE','state':{'n':2},'candidate_ref':ref,'probe_ref':'public-task'}\n"
        "return {'action':'STAGE','state':{},'kind':'P','source':" + repr(source)
        + ",'parent_ref':view['target']['ref']}"
    )
    probes = []
    result = host.run(editor, task(), "ADAPT_TASK", {}, ("public-task",),
                      lambda candidate, ref: probes.append(ref) or {"score": 0})
    assert result.status == "FAILED" and "frozen" in result.error
    assert probes == ["public-task"] and len(result.candidates) == 1


def test_adapt_task_cannot_stage_researcher_or_probe_unavailable_private_handle(host):
    illegal = researcher(
        "return {'action':'STAGE','state':{},'kind':'M','source':view['target']['source'],"
        "'parent_ref':view['target']['ref']}"
    )
    result = host.run(illegal, task(), "ADAPT_TASK", {}, (), None)
    assert result.status == "FAILED" and "role/parent" in result.error
    private = researcher(
        "return {'action':'PROBE','state':{},'candidate_ref':view['target']['ref'],'probe_ref':'PRIVATE_QUERY'}"
    )
    result = host.run(private, task(), "ADAPT_TASK", {}, ("public",), None)
    assert result.status == "FAILED" and "private" in result.error


def test_invalid_staging_consumes_attempt_and_noop_preserves_parent(host):
    editor = researcher(
        "n=state.get('n',0)\n"
        "if n<3:\n"
        "    return {'action':'STAGE','state':{'n':n+1},'kind':'P','source':'def broken(',"
        "'parent_ref':view['target']['ref']}\n"
        "return {'action':'SUBMIT','state':{},'candidate_ref':view['target']['ref']}"
    )
    result = host.run(editor, task(), "ADAPT_TASK", {}, (), None)
    assert result.status == "FAILED" and "stage budget" in result.error
    result = host.run(noop(), task(), "ADAPT_TASK", {}, (), None)
    assert result.status == "NOOP" and result.submitted_ref == task().ref


def test_archive_detects_mutation(host):
    result = host.run(noop(), task(), "ADAPT_TASK", {}, (), None)
    path = host.archive.root / (result.archive_ref + ".json")
    data = json.loads(path.read_text())
    data["events"][0]["event"]["output"]["candidate_ref"] = "tampered"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="integrity"):
        host.verify(result)


def test_budget_is_shared_and_complete_group_reservation_is_enforced():
    ledger = Ledger(Limits(episodes=3))
    with ledger.reserve(episodes=2) as reservation:
        ledger.charge("root", episodes=1)
        with pytest.raises(BudgetExceeded):
            with ledger.reserve(episodes=1):
                pass
        with reservation.spend():
            ledger.charge("child", episodes=2)
    assert ledger.used["episodes"] == 3 and ledger.reserved["episodes"] == 0
    assert [event["scope"] for event in ledger.events] == ["root", "child"]


def package(purpose, *, family=None, lineage=None):
    return AdaptationPackage(purpose, "fixture", family or purpose, lineage or purpose,
                             "meta-train", purpose, task().ref, (), (purpose + "-public",),
                             (purpose + "-PRIVATE_QUERY",), {})


def test_two_level_partition_and_query_projection():
    groups = {("meta-train", name): (package(name),) for name in ("C_probe", "C_select", "C_reward")}
    validate_partitions(groups)
    assert "PRIVATE_QUERY" not in json.dumps(package("C_reward").public_view())
    groups[("meta-train", "C_reward")] = (package("C_reward", lineage="C_select"),)
    with pytest.raises(ValueError, match="crosses"):
        validate_partitions(groups)


def test_unknown_remains_zero_and_not_false_verified_success():
    result = TaskResult("trace", 0, "UNKNOWN", True)
    assert result.status == "UNKNOWN" and result.score == 0
    with pytest.raises(ValueError, match="unverified"):
        TaskResult("trace", 1, "UNKNOWN", True)
