"""Execute the nested SELF_EDIT -> ADAPT_TASK -> P chain in real containers."""
from dataclasses import asdict, replace
import json

import pytest

from rsi_core.evaluation import Evaluator, select
from rsi_core.fixtures import ArithmeticFixtureAdapter
from rsi_core.loop import recursive_run
from rsi_core.programs import ProgramArtifact
from rsi_core.protocol import AdaptationPackage
from test_rsi_core import host, noop, researcher


BAD_P = ProgramArtifact("P", "def step(view,state):\n    return {'kind':'ANSWER','value':0,'state':{}}\n")
GOOD_P = "def step(view,state):\n    return {'kind':'ANSWER','value':sum(view['numbers']),'state':{}}\n"


def adapting(source=GOOD_P):
    return researcher(
        "if not state:\n"
        "    return {'action':'STAGE','state':{'staged':True},'kind':view['target']['kind'],"
        "'source':" + repr(source) + ",'parent_ref':view['target']['ref']}\n"
        "return {'action':'SUBMIT','state':state,'candidate_ref':view['observations'][-1]['candidate_ref']}"
    )


def fixture(host):
    host.store.save(BAD_P)
    packages, tasks = {}, {}
    for purpose in ("C_probe", "C_select", "C_reward"):
        tasks[purpose + "-probe"] = {"numbers": [1, 2]}
        tasks[purpose + "-query"] = {"numbers": [7, 4]}
        packages[purpose] = AdaptationPackage(purpose, ArithmeticFixtureAdapter.domain_id,
            purpose + "-family", purpose + "-lineage", "meta-train", purpose,
            BAD_P.ref, (), (purpose + "-probe",), (purpose + "-query",), {})
    evaluator = Evaluator(host, ArithmeticFixtureAdapter(tasks, host.archive))
    return evaluator, packages


def test_nested_value_selection_and_no_query_in_candidate_inputs(host):
    evaluator, packages = fixture(host)
    parent, child = noop(), adapting()
    comparison = evaluator.paired(parent, child, (packages["C_select"],), purpose="C_select")
    assert comparison.mean_delta == 1 and comparison.usable
    assert select(parent, child, comparison)["decision"] == "ACCEPT"
    assert all(old.p0_ref == new.p0_ref == BAD_P.ref for old, new in comparison.pairs)
    for old, new in comparison.pairs:
        for result in (old, new):
            archive = host.archive.load(result.research_run_ref)
            inputs = [row["event"]["input"] for row in archive["events"] if row["event"]["kind"] == "decision"]
            assert "C_select-query" not in json.dumps(inputs)
    reward = evaluator.paired(parent, child, (packages["C_reward"],), purpose="C_reward")
    with pytest.raises(ValueError, match="reward/audit"):
        select(parent, child, reward)


def test_adaptation_failure_scores_zero_without_removing_package(host):
    evaluator, packages = fixture(host)
    broken = researcher("return {'action':'ILLEGAL','state':{}}")
    result = evaluator.value(broken, packages["C_reward"])
    assert result.status == "ADAPTATION_FAILED" and result.score == 0
    comparison = evaluator.paired(adapting(), broken, (packages["C_select"],), purpose="C_select")
    assert len(comparison.pairs) == 1 and comparison.mean_delta == -1
    assert select(adapting(), broken, comparison)["decision"] == "KEEP_PARENT"


def test_three_generations_actually_switch_editors_and_record_failed_generation(host):
    evaluator, packages = fixture(host)
    # On ADAPT_TASK this first successor learns P. On SELF_EDIT its same
    # entrypoint emits an invalid decision, testing a retained failed generation.
    successor = researcher(
        "if view['mode']=='SELF_EDIT':\n"
        "    return {'action':'INVALID_AFTER_INHERITANCE','state':{}}\n"
        "if not state:\n"
        "    return {'action':'STAGE','state':{'staged':True},'kind':'P','source':" + repr(GOOD_P)
        + ",'parent_ref':view['target']['ref']}\n"
        "return {'action':'SUBMIT','state':{},'candidate_ref':view['observations'][-1]['candidate_ref']}"
    )
    initial = researcher(
        "if view['mode']=='ADAPT_TASK':\n"
        "    return {'action':'SUBMIT','state':{},'candidate_ref':view['target']['ref']}\n"
        "if not state:\n"
        "    return {'action':'STAGE','state':{'staged':True},'kind':'M','source':" + repr(successor.source)
        + ",'parent_ref':view['target']['ref']}\n"
        "return {'action':'SUBMIT','state':{},'candidate_ref':view['observations'][-1]['candidate_ref']}"
    )
    result = recursive_run(host, evaluator, initial, (packages["C_probe"],), (packages["C_select"],))
    history = result["history"]
    assert len(history) == 3
    assert history[0]["selection"]["decision"] == "ACCEPT"
    assert history[0]["selected_ref"] != initial.ref
    assert history[1]["editor_ref"] == history[0]["selected_ref"]
    assert history[1]["target_ref"] == history[0]["selected_ref"]
    assert history[1]["selection"]["reasons"][0] == "CANDIDATE_FAILED"
    assert history[2]["editor_ref"] == history[1]["selected_ref"]
    assert all(not row["phi_changed"] for row in history)
    fixed = recursive_run(host, evaluator, initial, (packages["C_probe"],), (packages["C_select"],),
                          generations=2, fixed_editor=True)
    assert fixed["history"][1]["editor_ref"] == initial.ref
    assert fixed["history"][1]["target_ref"] == fixed["history"][0]["selected_ref"] != initial.ref


def test_pair_rejects_different_purpose_and_models_are_bound(host):
    evaluator, packages = fixture(host)
    with pytest.raises(ValueError, match="purpose"):
        evaluator.paired(noop(), adapting(), (packages["C_reward"],), purpose="C_select")
    comparison = evaluator.paired(noop(), adapting(), (packages["C_select"],), purpose="C_select")
    old, new = comparison.pairs[0]
    corrupted = replace(comparison, pairs=((old, replace(new, pair_binding="different-model-or-P0")),))
    # Binding is checked again by the selector, not trusted just because a
    # record was once produced by a trusted evaluator.
    with pytest.raises(ValueError, match="binding"):
        select(noop(), adapting(), corrupted)
