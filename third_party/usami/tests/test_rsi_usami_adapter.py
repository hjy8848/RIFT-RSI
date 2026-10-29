from dataclasses import replace
import json
from pathlib import Path

from integrations.rsi_usami.adapter import NativeTask, UsamiAdapter, baseline_request
from rsi_core.programs import ProgramArtifact
from rsi_core.protocol import FrozenModels, ModelSnapshot
from usami.service.models import ModelSettings
from usami_v2.runtime.agents import AgentMessage, OpenAIServiceAgent, ServicePrompt
from usami_v2.runtime.runner import EpisodeBudget, EpisodeRecord
from usami_v2.scenarios.task_certification import CustomerScenario, certify_task, verify_task_certificate
from test_rsi_core import host
from test_usami_v2_runtime import frozen, refund_profile, refund_task, ScriptedCustomer


def test_native_request_matches_existing_s0_including_sampling_and_public_history():
    settings = ModelSettings(model="offline-s0", base_url="http://localhost:1/v1", options={"temperature": 0.4})

    class Capture:
        def __init__(self):
            self.settings = settings

        def complete(self, request):
            self.request = request
            return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Hello"}}]}

    client = Capture()
    prompt = ServicePrompt(customer_id="public-customer", customer_request="Please help.",
                           available_merchant_scope_ids=("public-merchant",),
                           conversation=(AgentMessage(role="user", content="Please help."),))
    OpenAIServiceAgent(client).decide(prompt)
    snapshot = ModelSnapshot(settings.model, "fixture", settings.options)
    assert baseline_request(prompt, snapshot) == client.request


def test_real_certified_refund_only_uses_public_ids_and_full_native_replay(host, tmp_path):
    world, aliases = frozen()
    task, profile = refund_task(world, aliases), refund_profile(aliases)
    scenario = CustomerScenario(customer_id=task.customer_id, delegated_line_ids=profile.delegated_line_ids)
    certificate = certify_task(world, task, scenario)
    assert certificate.status == "CERTIFIED"
    assert verify_task_certificate(world, task, scenario, certificate)["verified"]
    native = NativeTask(world, task, profile, ScriptedCustomer, EpisodeBudget())
    adapter = UsamiAdapter({"opaque-task": native}, tmp_path / "native")
    source = (Path(__file__).parents[1] / "experiments/rsi/fixtures/refund_public_controller.py").read_text()
    program = ProgramArtifact("P", source)
    host.ledger.charge("evaluation", episodes=1)
    result = adapter.run_task(program, "opaque-task", host.models, host.ledger, host.worker, seed=1)
    assert result.score == 1 and result.status == "VERIFIED"
    assert result.model_calls == 0 and result.model_tokens == 0
    assert adapter.replay(result.run_ref)["verified"]
    record = EpisodeRecord.model_validate_json(adapter.runs[result.run_ref].read_text())
    assert record.outcome.strict_task_success is True
    assert record.outcome.safety_violations.status == "PASS"
    assert record.outcome.process_errors.status == "PASS"
    feedback = adapter.research_feedback(result)
    assert feedback["protocol"] == adapter.capabilities()["feedback_protocol"]
    assert feedback["status"] == result.status and feedback["score"] == result.score
    assert len(feedback["operations"]) == len(record.tool_executions)
    # The root Designer can inspect the Service's public tool history, but not
    # the Oracle labels or private task outcome that remain in native Replay.
    assert "oracle" not in json.dumps(feedback).lower()
    assert "safety_violation" not in feedback and "forbidden_behavior" not in feedback
    assert "strict_task_success" not in json.dumps(feedback)
    internally_labeled = replace(result, forbidden_behavior=True)
    assert internally_labeled.forbidden_behavior is True
    assert "safety_violation" not in adapter.research_feedback(internally_labeled)
    inputs = json.loads((adapter.root / (result.run_ref + ".public-inputs.json")).read_text())["inputs"]
    assert not any(key in inputs[0]["public_view"] for key in ("world", "task", "profile", "oracle"))
    assert aliases["damaged"] not in json.dumps(inputs[0])
    evidence = next(row.id for row in world.evidence.values() if row.line_id == aliases["damaged"])
    assert evidence not in json.dumps(inputs[:6])
    assert evidence in json.dumps(inputs[6:])
    assert host.ledger.used["tool_steps"] == len(record.tool_executions)


def test_period_history_does_not_publish_oracle_safety_label():
    from types import SimpleNamespace
    from rsi_core.period import _public_service_research_history

    feedback = {"status": "VERIFIED", "terminal_status": "customer_finished",
                "safety_violation": True,
                "operations": [{"operation": {"name": "prepare_remedy_quote"},
                                "result": {"success": True}}]}
    result = {"status": "VERIFIED", "score": 1.0, "replay_verified": True,
              "task_ref": "certified-task", "feedback": feedback}
    root = SimpleNamespace(status="SUBMITTED", error="", events=(
        {"event": {"kind": "baseline_execution", "task_ref": "certified-task", "result": result}},))
    group = SimpleNamespace(status="USABLE", roots=(root,))

    summary = _public_service_research_history((group,))
    serialized = json.dumps(summary)
    assert "safety_violation" not in serialized
    assert "forbidden_behavior" not in serialized
    assert summary[0]["roots"][0]["baseline_runs"][0]["feedback"]["operation_results"] == [
        {"operation": "prepare_remedy_quote", "result": "SUCCESS", "count": 1}]


def test_hidden_goal_metadata_does_not_change_task_program_public_input(host, tmp_path):
    world, aliases = frozen()
    task, profile = refund_task(world, aliases), refund_profile(aliases)
    changed_goal = type(task.remedy_goals[0]).model_validate({
        **task.remedy_goals[0].model_dump(mode="json", exclude={"record_hash"}),
        "goal_id": "PRIVATE_SCORER_CANARY"})
    changed = type(task).model_validate({**task.model_dump(mode="json", exclude={"record_hash"}),
                                       "remedy_goals": [changed_goal.model_dump(mode="json")]})
    program = ProgramArtifact("P", "def step(view,state):\n    return {'kind':'ACTION','state':{},'action':{'kind':'reply','message':'Hello'}}\n")
    adapter = UsamiAdapter({"one": NativeTask(world, task, profile, ScriptedCustomer, EpisodeBudget()),
                           "two": NativeTask(world, changed, profile, ScriptedCustomer, EpisodeBudget())}, tmp_path / "native")
    first = adapter.run_task(program, "one", host.models, host.ledger, host.worker, seed=0)
    second = adapter.run_task(program, "two", host.models, host.ledger, host.worker, seed=0)
    views = [json.loads((adapter.root / (result.run_ref + ".public-inputs.json")).read_text())["inputs"]
             for result in (first, second)]
    assert views[0] == views[1]
    assert "PRIVATE_SCORER_CANARY" not in json.dumps(views)
