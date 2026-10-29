import json
from types import SimpleNamespace

from usami.service.models import ModelSettings
from usami_benchmark.registry import default_spec
from usami_benchmark.tasks import ConstructionBudget, OpenDialogueComposition, TaskSelection, compile_task
from usami_benchmark.worlds import freeze
from usami_aftersales import adapter, tasks as aftersales_tasks
from usami_dialogue.tasks import Goal, InformationRequirement, PreferenceRequirement
from vast_coevo.skills import DecisionSkill, LibrarySnapshot, Scope, Trigger

from experiments.vast_coevo.examples.aftersales_coevo_support import initial_skill_libraries
from experiments.vast_coevo.examples import aftersales_service_skill_ab_v3 as evaluator
from experiments.vast_coevo.examples import aftersales_v3_reflect_from_error as reflection_pilot


def _validation_refund_task(frozen):
    world = adapter.World.model_validate(frozen.world)
    for profile_id in world.profiles:
        private = world.profiles[profile_id]
        line_id = private.facts["line_id"]
        line = world.lines[line_id]
        facts = [key for key in ("order_id", "line_id", "issue", "evidence_id")
                 if key in private.facts]
        selection = TaskSelection(kind="dialogue", family="open_composition_v1",
            profile_id=profile_id, composition=OpenDialogueComposition(
                profile_ids=[profile_id],
                customer_request="I want a refund for the damaged item.",
                goals=[Goal(collection="lines", entity=line_id, field="status",
                            accepted_values=["refunded"])],
                frames=aftersales_tasks.frame_contract(world, private),
                external_schedule=[event.id for event in world.external_events.values()
                                   if event.line_id == line_id],
                information_requirements=[InformationRequirement(profile_id=profile_id,
                    fact_keys=facts)],
                preference_requirements=[PreferenceRequirement(profile_id=profile_id,
                    preference="resolution", accepted_values=["refund"])],
            ))
        try:
            return compile_task(frozen, selection,
                ConstructionBudget(max_nodes=2048, max_depth=16, max_paths=1))
        except ValueError:
            continue
    raise AssertionError("no certifiable validation refund task found")


def _candidate(s0):
    old = s0.skills[0].model_dump(mode="json")
    old["version"] += 1
    old["description"] = "Updated v3 paired-evaluation fixture guidance."
    old["content_hash"] = ""
    changed = DecisionSkill.model_validate(old)
    return LibrarySnapshot(role="service", skills=(changed,), parent_hash=s0.snapshot_hash,
        provenance="v3-paired-evaluation-test-candidate")


def test_v3_config_pins_scenario_and_only_changes_service_skill_context():
    frozen = freeze(default_spec("aftersales", 7, partition="validation"))
    task = _validation_refund_task(frozen)
    _, s0, _ = initial_skill_libraries()
    s1 = _candidate(s0)
    config0, scenario0 = evaluator._make_config(task, s0, mode="sandbox", turns=12,
        service_calls=20, service_tool_calls=64, customer_mode="deterministic")
    config1, scenario1 = evaluator._make_config(task, s1, mode="sandbox", turns=12,
        service_calls=20, service_tool_calls=64, customer_mode="deterministic")

    assert scenario0.scenario_hash == scenario1.scenario_hash
    assert evaluator._config_without_skills(config0) == evaluator._config_without_skills(config1)
    assert config0.protocol == "business-dialogue-task-agents-v3-finish-clarification-v1"
    assert config0.skill_contexts["service"].snapshot_hash == s0.snapshot_hash
    assert config1.skill_contexts["service"].snapshot_hash == s1.snapshot_hash
    assert config1.skill_contexts["service"].rendered


def test_v3_injection_audit_does_not_count_empty_context_as_skill_injection():
    from usami_dialogue.agents import SkillPromptContext

    rendered = '<VAST_SKILL id="refund-check" version="1">rule</VAST_SKILL>'
    empty_context = SkillPromptContext(snapshot_hash="a" * 64, skill_ids=[], rendered="")
    empty_call = SimpleNamespace(role="service", record=SimpleNamespace(request={
        "messages": [{"role": "system", "content": "Applied snapshot " + empty_context.snapshot_hash}]}))
    empty_execution = SimpleNamespace(config=SimpleNamespace(skill_contexts={"service": empty_context}),
        calls=[empty_call])
    not_retrieved = evaluator._injection_evidence(empty_execution, ["refund-check"])
    assert not_retrieved["skill_injection_status"] == "not_retrieved"
    assert not_retrieved["candidate_skill_injected"] is False

    context = SkillPromptContext(snapshot_hash="a" * 64,
        skill_ids=["refund-check"], rendered=rendered)
    call = SimpleNamespace(role="service", record=SimpleNamespace(request={
        "messages": [{"role": "system", "content": "Applied snapshot " + context.snapshot_hash}]}))
    execution = SimpleNamespace(config=SimpleNamespace(skill_contexts={"service": context}), calls=[call])

    missing = evaluator._injection_evidence(execution, ["refund-check"])
    assert missing["all_service_requests_contained_context"] is True
    assert missing["skill_injection_status"] == "missing"
    assert missing["all_service_requests_contained_skill"] is False

    call.record.request["messages"][0]["content"] += "\n" + rendered
    injected = evaluator._injection_evidence(execution, ["refund-check"])
    assert injected["skill_injection_status"] == "verified"
    assert injected["all_service_requests_contained_skill"] is True


def test_reflector_candidate_must_be_retrieved_before_paid_evaluation(tmp_path):
    frozen = freeze(default_spec("aftersales", 7, partition="validation"))
    task = _validation_refund_task(frozen)
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task.model_dump(mode="json")), encoding="utf-8")

    def candidate(keyword):
        skill = DecisionSkill(id="release_refund_consent_verification", version=1,
            scope=Scope(domains=("aftersales",)), trigger=Trigger(keywords=(keyword,)),
            description="Require exact current authorization for settlement.",
            preconditions=("A settlement is considered.",),
            procedure=("Check exact current authorization.",),
            postconditions=("No unauthorized settlement occurs.",),
            recovery=("Ask for the missing authorization.",))
        return LibrarySnapshot(role="service", skills=(skill,), provenance="retrieval-preflight-test")

    internal_name_only = reflection_pilot._retrieval_preflight([task_path], candidate("release_refund"))
    natural_language = reflection_pilot._retrieval_preflight([task_path], candidate("refund"))
    assert internal_name_only["status"] == "FAIL_NO_CANDIDATE_SKILL_RETRIEVED"
    assert internal_name_only["paid_evaluation_started"] is False
    assert natural_language["status"] == "PASS"
    assert natural_language["tasks"][0]["rendered_chars"] > 0


def test_v3_pair_runner_uses_scenario_driver_replays_and_never_reflects(tmp_path, monkeypatch):
    from experiments.vast_coevo.examples.after_sales_service_v3_diagnostic import ControlledRefundService

    frozen = freeze(default_spec("aftersales", 7, partition="validation"))
    task = _validation_refund_task(frozen)
    excluded = freeze(default_spec("aftersales", 8, partition="train"))
    _, s0, _ = initial_skill_libraries()
    s1 = _candidate(s0)
    task_path, excluded_path = tmp_path / "task.json", tmp_path / "train-world.json"
    s0_path, s1_path = tmp_path / "s0.json", tmp_path / "s1.json"
    for path, value in ((task_path, task), (excluded_path, excluded),
                        (s0_path, s0), (s1_path, s1)):
        path.write_text(json.dumps(value.model_dump(mode="json")), encoding="utf-8")

    settings = ModelSettings(model="fixture-v3-service", base_url="https://fixture.invalid/v1",
        options={"temperature": 0})
    monkeypatch.setattr(evaluator, "_role_settings", lambda role: settings)
    monkeypatch.setattr(evaluator, "key_from_env", lambda *_args: "fixture-key")
    monkeypatch.setattr(evaluator, "OpenAICompatibleClient", lambda *_args: ControlledRefundService())

    result = evaluator.run_paired_evaluation_v3(
        tmp_path / "paired-v3", task_paths=[task_path], exclude_world_paths=[excluded_path],
        s0_path=s0_path, s1_path=s1_path, mode="sandbox", customer_mode="deterministic",
        repeats=1, turns=12, service_calls=20, service_tool_calls=64, max_model_calls=64,
    )

    assert result["status"] == "EXECUTED"
    assert result["dialogue_protocol"] == "business-dialogue-task-agents-v3-finish-clarification-v1"
    assert result["reflection_or_skill_selection"] == "disabled"
    assert result["model_calls"] > 0
    assert len(result["episodes"]) == 2
    assert all(row["execution_status"] == "customer_finished" for row in result["episodes"])
    assert all(row["replay"]["verified"] for row in result["episodes"])
    assert all(row["service_skill_injection"]["all_service_requests_contained_context"]
               for row in result["episodes"])
    assert all(row["customer_fidelity_status"] in {"pass", "needs_review"}
               for row in result["episodes"])
    assert all(row["task_success"] for row in result["episodes"])


def test_v3_strict_success_requires_customer_finish_and_budget():
    class Execution:
        status = "service_provider_error"
        task_score = {"success": True}
        cost = {"customer_calls": 1, "service_calls": 1, "tool_calls": 1,
                "format_recoveries": {"customer": 0, "service": 0}}
        config = type("Config", (), {"budget": type("Budget", (), {
            "max_customer_llm_calls": 4, "max_service_llm_calls": 8,
            "max_service_tool_calls": 12, "max_format_recoveries": 0})()})()

    scored = evaluator._strict_success(Execution())
    assert scored["native_success"] is True
    assert scored["normal_completion"] is False
    assert scored["success"] is False

    Execution.status = "customer_finished"
    Execution.cost["service_calls"] = 9
    scored = evaluator._strict_success(Execution())
    assert scored["normal_completion"] is True
    assert scored["budget_compliance"] is False
    assert scored["success"] is False
