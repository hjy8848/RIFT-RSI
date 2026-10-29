"""v0.2 snapshot, direct-task, and challenge-boundary contracts."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from rsi_core.archive import Archive
from rsi_core.budget import Ledger
from rsi_core.collection import CollectedGroup
from rsi_core.evaluation import ServiceEvaluator, select_service
from rsi_core.packages import ServiceTaskProvider, ServiceTaskRegistration, TaskProof
from rsi_core.programs import (DesignerSnapshot, ExperienceArtifact, ProgramArtifact,
                               ProgramStore, ServiceSnapshot)
from rsi_core.protocol import FrozenModels, Limits, ModelSnapshot, TaskResult
from rsi_core.training import RootTrainingRecord
from test_rsi_core import host, researcher


def _service_models():
    business = ModelSnapshot("fixed-business", "business-hash")
    return FrozenModels(business, business, business)


def _service(policy_source="def step(view,state):\n    return {'state':state}\n",
             experience=None, *, parent_ref="", policy_parent="", experience_parent="",
             business_ref=None, evaluation_only=False):
    business = business_ref or _service_models().business.ref
    return ServiceSnapshot(
        business,
        ProgramArtifact("P", policy_source, policy_parent),
        ExperienceArtifact.from_value(experience or {}, experience_parent),
        parent_ref,
        evaluation_only,
    )


def test_service_and_designer_snapshots_keep_atomic_parent_lineage(tmp_path):
    store = ProgramStore(tmp_path / "programs")
    parent = _service()
    store.publish_service(parent)
    child = _service("def step(view,state):\n    return {'state': {'new': True}}\n",
                     {"learned": ["check deadline"]}, parent_ref=parent.ref,
                     policy_parent=parent.policy.ref, experience_parent=parent.experience.ref,
                     business_ref=parent.business_model_ref)
    store.publish_service(child, parent_ref=parent.ref)
    assert store.load(child.ref) == child

    bad_parent = _service("def step(view,state):\n    return {'state':state}\n",
                          {"learned": True}, parent_ref=parent.ref,
                          policy_parent=parent.policy.ref,
                          experience_parent="0" * 64,
                          business_ref=parent.business_model_ref)
    with pytest.raises(ValueError, match="experience does not inherit"):
        store.publish_service(bad_parent, parent_ref=parent.ref)
    with pytest.raises(ValueError, match="evaluation-only"):
        store.publish_service(child.evaluation_copy(), parent_ref=parent.ref)

    phi = ModelSnapshot("research-model", "phi-0")
    m0 = researcher("return {'action':'SUBMIT','state':{},'candidate_ref':view['target']['ref']}")
    d0 = DesignerSnapshot(phi, m0)
    store.publish_designer(d0)
    # A period may train phi while keeping M unchanged, or accept an M child.
    d1 = DesignerSnapshot(ModelSnapshot("research-model", "phi-1"), m0, d0.ref)
    store.publish_designer(d1, parent_ref=d0.ref)
    with pytest.raises(ValueError, match="base model and sampling stay fixed"):
        store.publish_designer(DesignerSnapshot(ModelSnapshot("changed-base", "phi-2"), m0, d1.ref),
                               parent_ref=d1.ref)
    m1 = ProgramArtifact("M", m0.source + "\n", m0.ref)
    d2 = DesignerSnapshot(d1.research_model, m1, d1.ref)
    store.publish_designer(d2, parent_ref=d1.ref)


def _direct_provider(tmp_path, *, purpose="service_select", partition="meta-train",
                     family="family-select", lineage="lineage-select"):
    archive = Archive(tmp_path / "archive")
    refs = ("task-a", "task-b")
    proofs = tuple(TaskProof(ref, "origin-" + ref,
                             archive.save({"task": ref, "certified": True})) for ref in refs)
    row = ServiceTaskRegistration("set-" + purpose, "fixture-domain", family, lineage,
                                  partition, purpose, proofs, seed=10)
    provider = ServiceTaskProvider((row,), archive,
                                   lambda domain, proof: domain == "fixture-domain"
                                   and archive.load(proof.certificate_ref) == {
                                       "task": proof.task_ref, "certified": True})
    return archive, provider, provider.task_set(row.set_id)


class _DirectAdapter:
    domain_id = "fixture-domain"

    def __init__(self):
        self.calls = []

    def run_task(self, service, task_ref, models, ledger, worker, *, seed):
        self.calls.append((service.ref, task_ref, seed))
        if service.policy.source == "unsafe":
            return TaskResult("run-unsafe-" + task_ref, 1.0, "VERIFIED", True,
                              forbidden_behavior=True)
        if service.policy.source == "candidate":
            return TaskResult("run-new-" + task_ref, 1.0, "VERIFIED", True)
        return TaskResult("run-old-" + task_ref, 0.0, "VERIFIED", True)

    def replay(self, run_ref):
        return {"verified": True}


def test_direct_service_sets_require_native_certificates_and_pair_strict_results(tmp_path):
    archive, provider, task_set = _direct_provider(tmp_path)
    assert provider.verify_task_set(task_set)
    assert "certificate" not in str(task_set.public_view()).lower()
    models = _service_models()
    host = SimpleNamespace(models=models, ledger=Ledger(Limits()), worker=object(), archive=archive)
    adapter = _DirectAdapter()
    evaluator = ServiceEvaluator(host, adapter, provider)
    parent = _service("parent", business_ref=models.business.ref)
    candidate = _service("candidate", business_ref=models.business.ref)

    comparison = evaluator.paired(parent, candidate, task_set, purpose="service_select")
    decision = select_service(parent, candidate, comparison)
    assert comparison.mean_delta == 1.0 and comparison.usable and comparison.safe
    assert decision["decision"] == "ACCEPT"
    assert len(adapter.calls) == 4

    unsafe = _service("unsafe", business_ref=models.business.ref)
    comparison = evaluator.paired(parent, unsafe, task_set, purpose="service_select")
    decision = select_service(parent, unsafe, comparison)
    assert not comparison.safe and decision["decision"] == "KEEP_PARENT"
    assert "CANDIDATE_FORBIDDEN_BEHAVIOR" in decision["reasons"]


def test_direct_service_unknown_is_not_success_and_invalidates_reward_set(tmp_path):
    archive, provider, task_set = _direct_provider(tmp_path)
    models = _service_models()
    host = SimpleNamespace(models=models, ledger=Ledger(Limits()), worker=object(), archive=archive)

    class UnknownAdapter(_DirectAdapter):
        def run_task(self, service, task_ref, models, ledger, worker, *, seed):
            self.calls.append((service.ref, task_ref, seed))
            return TaskResult("unknown-" + service.ref[:6] + task_ref, 0.0, "UNKNOWN", True)

    evaluator = ServiceEvaluator(host, UnknownAdapter(), provider, unknown_limit=0.0)
    parent = _service("parent", business_ref=models.business.ref)
    candidate = _service("candidate", business_ref=models.business.ref)
    comparison = evaluator.paired(parent, candidate, task_set, purpose="service_select")
    assert comparison.mean_delta == 0.0
    assert comparison.unknown_fraction == 1.0
    assert not comparison.usable
    assert select_service(parent, candidate, comparison)["decision"] == "KEEP_PARENT"


def test_service_challenge_candidate_cannot_probe_before_parent_baseline(host):
    from rsi_core.programs import ExperienceArtifact

    target = ServiceSnapshot(host.models.business.ref,
                            ProgramArtifact("P", "def step(view,state):\n    return {'state':state}\n"),
                            ExperienceArtifact())
    editor = researcher(
        "if state.get('phase',0)==0:\n"
        "    return {'action':'PROPOSE_CHALLENGE','state':{'phase':1},'proposal':{'task':'new'}}\n"
        "return {'action':'PROBE','state':{'phase':2},'candidate_ref':view['target']['ref'],"
        "'probe_ref':'certified-task'}"
    )
    baseline_calls, probe_calls = [], []
    certify = lambda proposal: {"status": "CERTIFIED", "task_ref": "certified-task",
                                "certificate_ref": "certificate-ref", "certification_scope": "frozen",
                                "frozen": True}
    run_baseline = lambda service, task_ref: baseline_calls.append(task_ref)
    result = host.run(editor, target, "IMPROVE_SERVICE", {}, (),
                      lambda service, task_ref: probe_calls.append(task_ref),
                      certify_challenge=certify, run_baseline=run_baseline)
    assert result.status == "FAILED"
    assert "parent before candidate probing" in result.error
    assert not baseline_calls and not probe_calls


def test_service_challenge_must_be_certified_and_baselined_before_candidate_probe(host):
    from rsi_core.programs import ExperienceArtifact

    target = ServiceSnapshot(host.models.business.ref,
                            ProgramArtifact("P", "def step(view,state):\n    return {'state':state}\n"),
                            ExperienceArtifact())
    child_policy = "def step(view,state):\n    return {'state': {'adapted': True}}\n"
    editor = researcher(
        "phase=state.get('phase',0)\n"
        "if phase==0:\n"
        "    return {'action':'PROPOSE_CHALLENGE','state':{'phase':1},'proposal':{'task':'new'}}\n"
        "if phase==1:\n"
        "    ref=view['observations'][-1]['task_ref']\n"
        "    return {'action':'RUN_BASELINE','state':{'phase':2,'task_ref':ref},'task_ref':ref}\n"
        "if phase==2:\n"
        "    return {'action':'STAGE','state':{'phase':3,'task_ref':state['task_ref']},"
        "'kind':'SERVICE_BUNDLE','policy_source':" + repr(child_policy) + ","
        "'experience':{'learned':'check evidence'},'parent_ref':view['target']['ref']}\n"
        "if phase==3:\n"
        "    ref=view['observations'][-1]['candidate_ref']\n"
        "    return {'action':'PROBE','state':{'phase':4,'candidate_ref':ref,'task_ref':state['task_ref']},"
        "'candidate_ref':ref,'probe_ref':state['task_ref']}\n"
        "return {'action':'SUBMIT','state':{},'candidate_ref':state['candidate_ref']}"
    )
    baselines, probes = [], []
    certify = lambda proposal: {"status": "CERTIFIED", "task_ref": "certified-task",
                                "certificate_ref": "certificate-ref", "certification_scope": "frozen",
                                "frozen": True}

    def baseline(service, task_ref):
        baselines.append((service.ref, task_ref))
        return {"service_ref": service.ref, "task_ref": task_ref,
                "run_ref": "baseline-run", "score": 0.0, "status": "VERIFIED",
                "feedback": {"protocol": "fixture-v0.2", "run_ref": "baseline-run",
                             "status": "VERIFIED", "score": 0.0},
                "replay_verified": True}

    def probe(service, task_ref):
        probes.append((service.ref, task_ref))
        return {"score": 1.0, "status": "VERIFIED", "feedback": {},
                "run_ref": "candidate-run", "replay_verified": True,
                "forbidden_behavior": False}

    result = host.run(editor, target, "IMPROVE_SERVICE", {}, (), probe,
                      certify_challenge=certify, run_baseline=baseline)
    assert result.status == "SUBMITTED"
    assert len(baselines) == len(probes) == 1
    assert baselines[0] == (target.ref, "certified-task")
    assert probes[0][1] == "certified-task" and probes[0][0] != target.ref
    child = host.store.load(result.submitted_ref)
    assert isinstance(child, ServiceSnapshot)
    assert child.policy.parent_ref == target.policy.ref
    assert child.experience.parent_ref == target.experience.ref


def test_period_updates_phi_once_after_both_root_groups_and_carries_final_snapshots(tmp_path, monkeypatch):
    import rsi_core.period as period

    archive = Archive(tmp_path / "archive")
    store = ProgramStore(tmp_path / "programs")
    phi = ModelSnapshot("research-model", "phi-0")
    evaluator_model = ModelSnapshot("evaluator-model", "eval-0")
    business_model = ModelSnapshot("business-model", "business-0")
    models = FrozenModels(phi, evaluator_model, business_model)
    host = SimpleNamespace(store=store, archive=archive, models=models,
                           ledger=SimpleNamespace(snapshot=lambda: {"used": {}}))
    evaluator = SimpleNamespace(host=host)

    service0 = _service(business_ref=business_model.ref)
    store.publish_service(service0)
    service1 = _service("def step(view,state):\n    return {'state': {'period': 1}}\n",
                        {"period": 1}, parent_ref=service0.ref,
                        policy_parent=service0.policy.ref,
                        experience_parent=service0.experience.ref,
                        business_ref=business_model.ref)
    store.save(service1)
    m0 = researcher("return {'action':'SUBMIT','state':{},'candidate_ref':view['target']['ref']}")
    m1 = ProgramArtifact("M", m0.source + "\n# selected self-edit\n", m0.ref)
    store.save(m1)
    d0 = DesignerSnapshot(phi, m0)
    store.save(d0)

    service_context = archive.save({"kind": "service-context"})
    service_group_ref = archive.save({"status": "USABLE", "context_ref": service_context})
    service_group = CollectedGroup(service_context, service_group_ref, "USABLE", service1.ref,
                                   (SimpleNamespace(archive_ref="service-root", status="SUBMITTED",
                                                    error="", events=()),),
                                   ("service-reward",), "r_service")
    self_context_ref = archive.save({"kind": "self-context"})
    self_group_ref = archive.save({"status": "USABLE", "context_ref": self_context_ref})
    self_group = CollectedGroup(self_context_ref, self_group_ref, "USABLE", m1.ref,
                                (SimpleNamespace(archive_ref="self-root", status="SUBMITTED",
                                                 error="", events=()),),
                                ("self-reward",), "r_self")
    events = []

    def collect_service(*args, **kwargs):
        events.append(("service", args[3].ref))
        return service_group

    def collect_self(*args, **kwargs):
        packages = tuple(args[4]) + tuple(args[5]) + tuple(args[6])
        events.append(("self", args[2].ref, args[3].ref,
                       tuple(package.p0_ref for package in packages)))
        return self_group

    monkeypatch.setattr(period, "collect_service_improvement_group", collect_service)
    monkeypatch.setattr(period, "collect_self_edit_group", collect_self)
    model_refs = (phi.ref, evaluator_model.ref, business_model.ref)

    def rows(root_type, context_ref):
        return tuple(RootTrainingRecord(
            trace_ref=f"{root_type}-{index}", model_ref=model_refs[0],
            evaluator_ref=model_refs[1], business_ref=model_refs[2],
            group_id=context_ref, reward=reward, reward_source_ref=f"reward-{root_type}-{index}",
            turns=(), root_type=root_type)
            for index, reward in enumerate((-1.0, 1.0)))

    records = {service_group_ref: rows("r_service", service_context),
               self_group_ref: rows("r_self", self_context_ref)}
    monkeypatch.setattr(period, "training_records", lambda _host, group: records[group.archive_ref])
    hook_input = []

    def phi_update(collection_groups, groups):
        assert tuple(collection_groups) == (service_group, self_group)
        events.append(("phi_update", len(groups)))
        hook_input.extend(groups)
        return ModelSnapshot("research-model", "phi-1")

    def context(final_service):
        return {"probe_packages": (SimpleNamespace(p0_ref=final_service.ref),),
                "select_packages": (), "reward_packages": (), "seeds": (0, 1),
                "phase_budgets": {"root": {}, "selection": {}, "reward": {}}}
    record = period.run_period(
        host, evaluator, evaluator, d0, service0,
        ({"probe_tasks": "probe", "select_tasks": "select", "reward_tasks": "reward",
          "seeds": (0, 1), "phase_budgets": {}, "certify_challenge": lambda _: None},),
        context, phi_update=phi_update)

    assert events == [("service", service0.ref), ("self", m0.ref, m0.ref, (service1.ref,)),
                      ("phi_update", 2)]
    assert [group["root_type"] for group in hook_input] == ["r_service", "r_self"]
    assert all(group["advantages"] == (-1.0, 1.0) for group in hook_input)
    assert hook_input[0]["records"][0].group_id != hook_input[1]["records"][0].group_id
    assert record["status"] == "COMPLETED"
    assert record["training_status"] == "UPDATED" and record["training_updates"] == 1
    assert record["training_hook_calls"] == 1
    output_service = store.load(record["output_service_ref"])
    output_designer = store.load(record["output_designer_ref"])
    assert output_service.ref == service1.ref
    assert output_designer.research_model.ref == ModelSnapshot("research-model", "phi-1").ref
    assert output_designer.research_program.ref == m1.ref
    assert host.models.behavior.ref == ModelSnapshot("research-model", "phi-1").ref


def test_verl_period_export_keeps_root_groups_and_equal_type_context_weights(tmp_path, monkeypatch):
    import rsi_core.verl_backend as backend

    archive = Archive(tmp_path / "archive")
    phi = ModelSnapshot("research-model", "phi-0")
    evaluator = ModelSnapshot("evaluator-model", "eval-0")
    business = ModelSnapshot("business-model", "business-0")
    models = FrozenModels(phi, evaluator, business)
    host = SimpleNamespace(archive=archive)
    records_by_group = {}
    groups = []

    for root_type, group_num in (("r_service", 0), ("r_service", 1), ("r_self", 0)):
        context_ref = archive.save({"models": asdict(models), "group_num": group_num,
                                    "root_type": root_type})
        group_ref = archive.save({"kind": "fixture-group", "context_ref": context_ref,
                                  "root_type": root_type, "group_num": group_num})
        records = []
        for index, reward in enumerate((-1.0, 1.0)):
            trace = f"{root_type}-{group_num}-{index}"
            records.append(RootTrainingRecord(
                trace_ref=trace, model_ref=phi.ref, evaluator_ref=evaluator.ref,
                business_ref=business.ref, group_id=context_ref, reward=reward,
                reward_source_ref="reward-" + trace,
                turns=({"input_ids": (10, 11, 12), "loss_mask": (0, 1, 1),
                        "behavior_logprobs": (0.0, -0.1, -0.2),
                        "request_hash": "request-" + trace, "generated_output": "{}"},),
                root_type=root_type))
        records_by_group[group_ref] = tuple(records)
        groups.append(CollectedGroup(context_ref, group_ref, "USABLE", "selected",
                                     tuple(SimpleNamespace(archive_ref=row.trace_ref) for row in records),
                                     tuple(row.reward_source_ref for row in records), root_type))

    monkeypatch.setattr(backend, "training_records",
                        lambda _host, group: records_by_group[group.archive_ref])
    output = backend.export_verl_period(host, tuple(groups), tmp_path / "verl-period")
    manifest = archive.load(output["manifest_ref"])
    assert output["training_started"] is False
    assert manifest["kind"] == "verl_rsi_period_v0.2"
    assert len(manifest["rollouts"]) == 6
    assert manifest["root_type_weights"] == {"r_service": 0.5, "r_self": 0.5}
    weights = {row["root_type"]: row["rsi_loss_weight"] for row in manifest["rollouts"]}
    assert weights == {"r_service": 0.125, "r_self": 0.25}
    assert all(row["rsi_advantage"] in {-1.0, 1.0} for row in manifest["rollouts"])
    assert len({row["group_id"] for row in manifest["rollouts"]}) == 3
    dataset = [json.loads(line) for line in
               (tmp_path / "verl-period" / "train.jsonl").read_text().splitlines()]
    assert len(dataset) == 6
    assert {row["uid"] for row in dataset} == {group.context_ref for group in groups}
