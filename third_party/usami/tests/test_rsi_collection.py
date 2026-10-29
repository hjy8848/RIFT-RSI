"""Executed group/reward tests with explicit model fixtures, not real training."""
from dataclasses import replace
import json

import pytest

from rsi_core.collection import collect_sampled_self_edit_group, collect_self_edit_group, training_records
from rsi_core.packages import PackageProvider, PackageRegistration, TaskProof
from rsi_core.protocol import Completion, InfrastructureError
from rsi_core.training import group_advantages, root_training_record
from test_rsi_core import host, researcher
from test_rsi_evaluation import adapting, fixture


@pytest.fixture(autouse=True)
def group_limits(host):
    # This test registers twelve phase allocations, unlike single-run tests.
    host.ledger.limits = replace(host.ledger.limits, cpu_seconds=120)


def editing():
    return researcher(
        "if view['mode']=='ADAPT_TASK':\n"
        "    return {'action':'SUBMIT','state':{},'candidate_ref':view['target']['ref']}\n"
        "if not state:\n"
        "    return {'action':'MODEL','state':{'phase':1},'request':{'public':view},"
        "'response_schema':{'type':'object','required':['source']}}\n"
        "if state['phase']==1:\n"
        "    source=view['observations'][-1]['output']['source']\n"
        "    if source is None:\n"
        "        return {'action':'SUBMIT','state':{},'candidate_ref':view['target']['ref']}\n"
        "    return {'action':'STAGE','state':{'phase':2},'kind':'M','source':source,"
        "'parent_ref':view['target']['ref']}\n"
        "return {'action':'SUBMIT','state':{},'candidate_ref':view['observations'][-1].get('candidate_ref','invalid')}"
    )


class GroupModel:
    def __init__(self):
        self.requests = []

    def complete(self, snapshot, request, *, max_tokens, seed):
        self.requests.append(request)
        sources = [adapting().source, None, "invalid python!",
                   researcher("return {'action':'ILLEGAL','state':{}}").source]
        return Completion({"source": sources[seed % 4]}, 2, 2, (3, 4), (-1.0, -1.0), (1, 2))


BUDGETS = {
    "root": {"model_calls": 1, "model_tokens": 100, "research_steps": 6, "cpu_seconds": 5},
    "selection": {"episodes": 8, "research_steps": 12, "cpu_seconds": 8},
    "reward": {"episodes": 8, "research_steps": 12, "cpu_seconds": 8},
}


def collect(host, evaluator, packages):
    editor = editing()
    return collect_self_edit_group(host, evaluator, editor, editor,
        (packages["C_probe"],), (packages["C_select"],), (packages["C_reward"],),
        seeds=(0, 1, 2, 3), phase_budgets=BUDGETS)


def test_all_roots_receive_executed_rewards_including_noop_failure_and_unselected(host):
    evaluator, packages = fixture(host)
    host.model = GroupModel()
    group = collect(host, evaluator, packages)
    assert group.status == "USABLE"
    assert [run.status for run in group.roots] == ["SUBMITTED", "NOOP", "FAILED", "SUBMITTED"]
    assert group.selected_ref == group.roots[0].submitted_ref
    records = training_records(host, group)
    assert len(records) == 4
    assert [r.reward for r in records] == [1, 0, -1, 0]
    assert len(group_advantages(records)) == 4
    assert all(r.turns[0]["loss_mask"] == (0, 0, 1, 1) for r in records)
    assert host.ledger.used["model_calls"] == 4
    assert host.ledger.used["episodes"] > 0
    requests = json.dumps(host.model.requests)
    assert "C_select-query" not in requests and "C_reward-query" not in requests
    assert "C_select-family" not in requests and "C_reward-family" not in requests
    assert all(v == 0 for v in host.ledger.reserved.values())
    for run, row in zip(group.roots, records):
        bad = host.archive.save({**host.archive.load(row.reward_source_ref), "reward": 0.75})
        with pytest.raises(ValueError, match="receipt"):
            root_training_record(host, run, row.reward, bad, group.context_ref)
    with pytest.raises(ValueError, match="differ"):
        training_records(host, replace(group, roots=group.roots[:1]))


def test_provider_fault_discards_whole_group_without_hiding_completed_roots(host):
    evaluator, packages = fixture(host)

    class FailingModel(GroupModel):
        def complete(self, snapshot, request, *, max_tokens, seed):
            if seed == 1:
                raise InfrastructureError("explicit provider fixture failure")
            return super().complete(snapshot, request, max_tokens=max_tokens, seed=seed)

    host.model = FailingModel()
    group = collect(host, evaluator, packages)
    assert group.status == "UNUSABLE" and len(group.roots) == 1
    assert not group.reward_refs
    assert group.selected_ref == editing().ref
    assert "provider fixture failure" in host.archive.load(group.archive_ref)["error"]
    assert all(v == 0 for v in host.ledger.reserved.values())
    with pytest.raises(ValueError, match="unusable"):
        training_records(host, group)


def test_preflight_reserves_whole_group_before_first_model_request(host):
    evaluator, packages = fixture(host)
    host.model = GroupModel()
    host.ledger.limits = replace(host.ledger.limits, episodes=1)
    group = collect(host, evaluator, packages)
    assert group.status == "UNUSABLE"
    assert not group.roots and not host.model.requests
    assert host.ledger.used["model_calls"] == 0
    assert all(v == 0 for v in host.ledger.reserved.values())


@pytest.mark.parametrize("margin", [float("nan"), float("inf"), -1])
def test_invalid_selection_policy_fails_before_sampling(host, margin):
    evaluator, packages = fixture(host)
    host.model = GroupModel()
    editor = editing()
    with pytest.raises(ValueError, match="before sampling"):
        collect_self_edit_group(host, evaluator, editor, editor,
            (packages["C_probe"],), (packages["C_select"],), (packages["C_reward"],),
            seeds=(0, 1, 2, 3), phase_budgets=BUDGETS, margin=margin)
    assert not host.model.requests and not host.ledger.events


def test_duplicate_candidates_share_evaluation_but_not_training_roots(host):
    evaluator, packages = fixture(host)

    class SameCandidate(GroupModel):
        def complete(self, snapshot, request, *, max_tokens, seed):
            return super().complete(snapshot, request, max_tokens=max_tokens, seed=0)

    host.model = SameCandidate()
    group = collect(host, evaluator, packages)
    assert group.status == "USABLE"
    assert len({r.submitted_ref for r in group.roots}) == 1
    assert len(training_records(host, group)) == 4
    # One parent/candidate comparison per purpose: 1 baseline for the
    # unchanged parent and 2 query runs for the adapting child, twice.
    assert host.ledger.used["episodes"] == 6
    saved = host.archive.load(group.archive_ref)
    assert len(saved["selections"]) == len(saved["reward_comparisons"]) == 1


def test_sampled_package_provenance_is_bound_to_actual_group_execution(host):
    evaluator, packages = fixture(host)
    registrations = []
    for p in packages.values():
        proofs = tuple(TaskProof(ref, "fixture-origin-" + ref, "explicit-fixture-proof")
                       for ref in p.probe_refs + p.query_refs)
        registrations.append(PackageRegistration(p, (), proofs))
    # Core protocol fixture; native proof verification has separate real tests.
    provider = PackageProvider(registrations, host.archive, lambda domain, proof: True)
    host.model = GroupModel()
    editor = editing()
    group = collect_sampled_self_edit_group(host, evaluator, editor, editor, provider,
        package_seed=7, package_counts={"C_probe": 1, "C_select": 1, "C_reward": 1},
        seeds=(0, 1, 2, 3), phase_budgets=BUDGETS)
    assert group.status == "USABLE" and len(training_records(host, group)) == 4
    context = host.archive.load(group.context_ref)
    sampled = host.archive.load(context["sampling_ref"])
    assert sampled["manifest_ref"] == provider.manifest_ref
    with pytest.raises(ValueError, match="contents differ"):
        collect_self_edit_group(host, evaluator, editor, editor,
            (packages["C_probe"],), (packages["C_select"],),
            (replace(packages["C_reward"], query_refs=("swapped-private-query",)),),
            seeds=(4, 5, 6, 7), phase_budgets=BUDGETS, sampling_ref=context["sampling_ref"])


def test_unknown_reward_invalidates_all_exports_without_rewriting_native_status(host):
    evaluator, packages = fixture(host)
    host.model = GroupModel()
    original = evaluator.adapter.run_task

    def unknown(program, task_ref, models, ledger, worker, *, seed):
        result = original(program, task_ref, models, ledger, worker, seed=seed)
        if task_ref == "C_reward-query":
            return replace(result, status="UNKNOWN", score=0)
        return result

    evaluator.adapter.run_task = unknown
    group = collect(host, evaluator, packages)
    assert len(group.roots) == 4 and group.status == "UNUSABLE"
    assert not group.reward_refs
    saved = host.archive.load(group.archive_ref)
    comparisons = [host.archive.load(ref) for ref in saved["reward_comparisons"].values()]
    assert any(not row["usable"] for row in comparisons)
    assert any(old["unknown_fraction"] == 1 for row in comparisons for old, new in row["pairs"])
