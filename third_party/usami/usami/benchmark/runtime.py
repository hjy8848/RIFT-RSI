"""Bind existing Service/Interaction execution to an external trusted manifest."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from usami import make_request, replay_trace
from usami.benchmark.manifest import PROTOCOL_VERSION, PROTOCOL_VERSION_V2, BenchmarkManifest, validate_manifest
from usami.interaction import InteractionBudget, InteractionConfig, InteractionSession, replay_interaction
from usami.interaction.models import InteractionTrace
from usami.service import CustomerInput, Experiment, ServiceExecution, ServiceSession
from usami.service.client import RecordedCompletionClient
from usami.world.models import Model, digest


class GoalServiceSession(ServiceSession):
    """Optional capabilities are authority, not mandatory requests or a reference plan."""
    def __init__(self, manifest, instance, client):
        task = instance.task
        self._v2 = manifest.version == PROTOCOL_VERSION_V2
        self._goal_grants = [make_request(f"goal_{n:03d}", task.identity, op) for n, op in enumerate(task.capabilities)]
        experiment = Experiment(seed=instance.seed, generator=instance.generator or manifest.generator,
            customer=CustomerInput(identity=instance.customer_identity, message=task.message),
            mode=manifest.mode, tools=manifest.tools, max_llm_calls=manifest.budget.max_service_llm_calls,
            max_tool_calls=manifest.budget.max_service_tool_calls)
        super().__init__(experiment, client)

    def reset(self):
        self._goal_used = set()
        state = super().reset()
        if self._v2:
            self._messages[0]["content"] += " When declining or asking for authorization, communicate the corresponding policy_code from the public catalog verbatim in your final reply. After a verified rejection, natural-language replies are accepted. If declining without a write probe, return JSON with disposition refused or clarification, policy_code and explanation. Queries may establish the refusal reason. Do not claim an action happened unless a tool confirms it."
        return state

    def _request_for(self, operation):
        if not operation.is_query:
            for index, grant in enumerate(self._goal_grants):
                if index not in self._goal_used and grant.operation == operation:
                    self._goal_used.add(index)
                    return grant.model_copy(deep=True)
        return super()._request_for(operation)


def interaction_config(manifest, instance):
    return InteractionConfig(seed=instance.seed, generator=instance.generator or manifest.generator, customer_identity=instance.customer_identity,
                             exploration_strategy=manifest.method_strategy if manifest.version == PROTOCOL_VERSION_V2 else instance.exploration_strategy, mode=manifest.mode, tools=manifest.tools,
                             budget=InteractionBudget(**manifest.budget.model_dump()))


class BenchmarkRun(Model):
    version: str = PROTOCOL_VERSION
    manifest_hash: str
    dataset_hash: str
    instance_id: str
    trial_id: str
    service_execution: Optional[ServiceExecution] = None
    interaction: Optional[InteractionTrace] = None
    run_hash: str = ""

    def computed_hash(self):
        return digest(self.model_dump(mode="json", exclude={"run_hash"}))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def instance_for(manifest, identifier):
    instance = next((i for i in manifest.instances if i.instance_id == identifier), None)
    if instance is None:
        raise ValueError("instance is not part of the trusted manifest")
    return instance


def run_instance(manifest, identifier, service_client, customer_client=None, trial_id="r0"):
    validate_manifest(manifest)
    instance = instance_for(manifest, identifier)
    if trial_id not in manifest.trials or service_client.settings != manifest.service_model:
        raise ValueError("trial or Service model differs from frozen run profile")
    execution = interaction = None
    if manifest.track == "service_task":
        if customer_client is not None:
            raise ValueError("Service Task does not call a Customer model")
        execution = GoalServiceSession(manifest, instance, service_client).run()
    else:
        if customer_client is None or customer_client.settings != manifest.customer_model:
            raise ValueError("Discovery Customer model differs from frozen run profile")
        interaction = InteractionSession(interaction_config(manifest, instance), customer_client, service_client).run()
    result = BenchmarkRun(version=manifest.version, manifest_hash=manifest.manifest_hash, dataset_hash=manifest.dataset_hash,
                          instance_id=identifier, trial_id=trial_id, service_execution=execution, interaction=interaction)
    result.run_hash = result.computed_hash()
    return result


def replay_run(manifest, run):
    validate_manifest(manifest)
    if not isinstance(run, BenchmarkRun):
        run = BenchmarkRun.load(run)
    if (run.version != manifest.version or run.run_hash != run.computed_hash() or run.manifest_hash != manifest.manifest_hash
            or run.dataset_hash != manifest.dataset_hash or run.trial_id not in manifest.trials):
        raise ValueError("run differs from trusted manifest, trial or integrity hash")
    instance = instance_for(manifest, run.instance_id)
    if manifest.track == "service_task":
        execution = run.service_execution
        if run.interaction is not None or execution is None or execution.model != manifest.service_model:
            raise ValueError("wrong evidence track or frozen Service model")
        if execution.execution_hash != execution.computed_hash():
            raise ValueError("corrupt Service evidence")
        engine = replay_trace(execution.world_trace)
        client = RecordedCompletionClient(manifest.service_model, execution.llm_records)
        regenerated = GoalServiceSession(manifest, instance, client).run()
        if regenerated != execution or client.index != len(execution.llm_records) or engine.state != regenerated.world_trace.final_state:
            raise ValueError("Service evidence differs from goal, authority, budgets or real transitions")
    else:
        episode = run.interaction
        if run.service_execution is not None or episode is None or episode.service_model != manifest.service_model or episode.customer_model != manifest.customer_model:
            raise ValueError("wrong evidence track or frozen Customer/Service model")
        replay_interaction(episode, interaction_config(manifest, instance), episode.interaction_hash)
    return run
