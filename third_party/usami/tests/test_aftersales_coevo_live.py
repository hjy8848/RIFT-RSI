import json

import pytest

from experiments.vast_coevo.examples.aftersales_coevo_live import (
    BudgetedClient, CallLedger, _designer_context,
)
from experiments.vast_coevo.examples.aftersales_coevo_support import initial_skill_libraries
from usami.service.models import ModelSettings, ServiceError


class _Endpoint:
    def __init__(self):
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13}}


def test_live_call_ledger_records_exact_wire_exchange_and_stops_at_budget(tmp_path):
    settings = ModelSettings(model="fixture", base_url="https://fixture.invalid/v1",
                             options={"temperature": 0})
    endpoint = _Endpoint()
    ledger = CallLedger(tmp_path, limit=1)
    client = BudgetedClient("service", endpoint, settings, ledger)
    request = settings.request([{"role": "user", "content": "hello"}], [])

    result = client.complete(request)
    assert result["usage"]["total_tokens"] == 13
    record = json.loads((tmp_path / "model-calls/0001-service.json").read_text())
    assert record["request"] == request
    assert record["response"] == result
    assert ledger.usage()["by_role"]["service"]["total_tokens"] == 13

    with pytest.raises(ServiceError, match="budget exhausted"):
        client.complete(request)
    assert len(endpoint.requests) == 1
    assert ledger.actual_calls == 1
    assert ledger.denied == 1


def test_live_designer_context_does_not_preselect_a_task_or_expected_result():
    designer, _, _ = initial_skill_libraries()
    prior = [{"episode_id": "task-1", "trace_hash": "a" * 64,
              "process_errors": [{"classification": "false_rejection"}]}]
    context = _designer_context(designer, prior, "task-2")

    assert context["skill_snapshot_hash"] == designer.snapshot_hash
    assert "skill_rendered" in context and context["skill_rendered"]
    assert context["verified_prior_trials"] == prior
    assert "selection" not in context
    assert "expected_outcome" not in context
    from usami_aftersales import adapter
    from typing import get_args
    expected = list(get_args(adapter.World.model_fields["lines"].annotation.__args__[1].model_fields["status"].annotation))
    assert context["native_field_enums"]["lines"]["status"] == expected
    assert "resolved" not in context["native_field_enums"]["lines"]["status"]


def test_live_initial_snapshots_are_distinct_from_the_offline_smoke_provenance():
    designer, service, customer = initial_skill_libraries()
    assert designer.provenance == "aftersales-initial-designer-v1"
    assert service.provenance == "aftersales-initial-service-v1"
    assert customer.role == "customer"


def test_live_execution_returns_saved_replayable_artifacts(tmp_path):
    from pathlib import Path
    from experiments.vast_coevo.examples.aftersales_coevo_live import (
        _execute_task, _initial_world, _summary,
    )
    from experiments.vast_coevo.examples.aftersales_coevo_smoke import (
        _single_task, _task_customer_client, ServiceScriptClient,
    )
    from usami_aftersales import adapter

    frozen = _initial_world(1)
    task = _single_task(frozen)
    _, service, _ = initial_skill_libraries()
    customer = _task_customer_client(1, adapter.World.model_validate(frozen.world), "refund")
    result = _execute_task(tmp_path, 1, frozen, task, service, customer,
                           ServiceScriptClient(task_number=1), turns=5,
                           service_calls=9, service_tool_calls=16)

    assert result["evaluation_error"] is None
    assert result["agent_replay"]["verified"] is True
    for key in ("trace_path", "execution_path", "task_path"):
        assert Path(result[key]).is_file()
    status = json.loads((tmp_path / "tasks/task-1/status.json").read_text())
    assert status["status"] == "COMPLETED"
    # The existing Designer Reflector resolves citations against feedback['id'].
    assert _summary(result)["id"] == "task-1"
    assert result["score"]["native_success"] is True
    assert result["final_goal_reached"] is True
    assert result["score"]["normal_completion"] is True
    assert result["score"]["budget_compliance"] is True
    assert result["score"]["success"] is True
    assert json.loads((tmp_path / "tasks/task-1/score.json").read_text())["mode"] == "reference"


def test_goal_reached_but_abnormal_dialogue_termination_is_not_task_success(tmp_path):
    from experiments.vast_coevo.examples.aftersales_coevo_live import (
        _execute_task, _initial_world, _strict_task_score,
    )
    from experiments.vast_coevo.examples.aftersales_coevo_smoke import (
        _single_task, _task_customer_client, ServiceScriptClient,
    )
    from usami_aftersales import adapter
    from usami_dialogue.tasks import score as native_score

    frozen = _initial_world(1)
    task_record = _single_task(frozen)
    _, service, _ = initial_skill_libraries()
    customer = _task_customer_client(1, adapter.World.model_validate(frozen.world), "refund")
    result = _execute_task(tmp_path, 1, frozen, task_record, service, customer,
                           ServiceScriptClient(task_number=1), turns=5,
                           service_calls=9, service_tool_calls=16)
    execution = result["execution"].model_copy(update={"status": "customer_provider_error"})
    business_score = native_score(adapter, adapter.World.model_validate(frozen.world),
        task_record.task, execution.trace)

    strict = _strict_task_score(business_score, execution, task_record.task)
    assert business_score["success"] is True
    assert result["final_goal_reached"] is True
    assert strict["native_success"] is True
    assert strict["normal_completion"] is False
    assert strict["success"] is False

    over_budget_cost = dict(execution.cost)
    over_budget_cost["service_calls"] = execution.config.budget.max_service_llm_calls + 1
    over_budget = result["execution"].model_copy(update={"cost": over_budget_cost})
    over_budget_score = _strict_task_score(business_score, over_budget, task_record.task)
    assert over_budget_score["normal_completion"] is True
    assert over_budget_score["budget_compliance"] is False
    assert over_budget_score["success"] is False


def test_task_config_freezes_selected_engine_mode():
    from experiments.vast_coevo.examples.aftersales_coevo_live import (
        _initial_world, _make_task_config,
    )
    from experiments.vast_coevo.examples.aftersales_coevo_smoke import _single_task

    frozen = _initial_world(1)
    _, service, _ = initial_skill_libraries()
    record = _single_task(frozen)
    reference = _make_task_config(frozen, record, service, turns=4, mode="reference")
    sandbox = _make_task_config(frozen, record, service, turns=4, mode="sandbox")
    assert reference.mode == "reference"
    assert sandbox.mode == "sandbox"
    assert reference.task == sandbox.task
    assert reference.budget == sandbox.budget
