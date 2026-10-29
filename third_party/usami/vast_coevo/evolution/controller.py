"""Customer-first co-evolution with at-most-once external operations."""
from __future__ import annotations

import copy
from typing import Dict

from pydantic import JsonValue, TypeAdapter

from ..models import Consumption, EpisodeEvidence
from ..reflection.reflector import CustomerReflector, ServiceReflector
from ..schema import digest
from ..storage.runs import (BudgetExhausted, CrossExecution, GenerationRecord,
                            InterruptedRun, JournalError, OperationCompletion,
                            RunReport, RunStore)
from .models import EvolutionRecord, RunConfig


class RecordedCallError(Exception):
    def __init__(self, error):
        self.code = error.code
        self.original_type = error.type
        self.response = copy.deepcopy(error.response)
        self.response_recorded = error.response_recorded
        super().__init__(error.message)


class _JournalClient:
    def __init__(self, client, store, config, operation_id):
        self.client, self.store, self.config, self.operation_id = client, store, config, operation_id

    def complete(self, request):
        request = TypeAdapter(Dict[str, JsonValue]).validate_python(copy.deepcopy(request))
        input_hash = digest(request)
        reserve = Consumption(reflection_calls=1)
        cached = self.store.start(self.config.run_id, self.operation_id, "reflection", input_hash,
                                  reserve, self.config.max_llm_calls, self.config.max_tool_calls)
        if cached is not None:
            if cached.error is not None:
                raise RecordedCallError(cached.error)
            return copy.deepcopy(cached.response)
        try:
            response = TypeAdapter(JsonValue).validate_python(copy.deepcopy(self.client.complete(request)))
        except Exception as exc:
            raw_response = TypeAdapter(JsonValue).validate_python(copy.deepcopy(getattr(exc, "response", None)))
            error = JournalError(type=type(exc).__name__, message=str(exc), code=getattr(exc, "code", "client_error"),
                                 response=raw_response, response_recorded=hasattr(exc, "response"))
            self.store.complete(self.config.run_id, OperationCompletion(operation_id=self.operation_id,
                                input_hash=input_hash, consumption=reserve, error=error))
            raise RecordedCallError(error) from None
        except BaseException as exc:
            self.store.fault(self.config.run_id, self.operation_id, exc)
            raise InterruptedRun("reflection outcome is unknown: " + self.operation_id) from None
        self.store.complete(self.config.run_id, OperationCompletion(operation_id=self.operation_id,
                            input_hash=input_hash, consumption=reserve, response=response, response_recorded=True))
        return copy.deepcopy(response)


class _JournalAdapter:
    def __init__(self, adapter, store, archive, config):
        self.adapter, self.store, self.archive, self.config = adapter, store, archive, config

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def _episode(self, config, customer, service, episode_id, track, callback, source=None, task_ref=None):
        config = type(config).model_validate(config.model_dump(mode="json"))
        if config.partition != "development":
            raise ValueError("heldout episodes cannot enter co-evolution")
        data = dict(config=config.model_dump(mode="json"), customer=customer.snapshot_hash,
                    service=service.snapshot_hash, track=track, task_ref=task_ref,
                    source_evidence_hash=None if source is None else source.evidence_hash)
        input_hash = digest(data)
        budget = config.budget
        reserve = Consumption(customer_llm_calls=budget.max_customer_llm_calls if track == "discovery" else 0,
                              service_llm_calls=budget.max_service_llm_calls,
                              tool_calls=budget.max_service_tool_calls)
        cached = self.store.start(self.config.run_id, episode_id, "episode", input_hash, reserve,
                                  self.config.max_llm_calls, self.config.max_tool_calls)
        if cached is not None:
            evidence = cached.evidence
            self.archive.save(evidence)  # also verifies the linked native artifact
            replay = self.adapter.replay(evidence, customer, service)
            if not replay.verified:
                raise RuntimeError("cached episode failed exact environment Replay: " + replay.reason)
        else:
            try:
                raw = callback()
                evidence = EpisodeEvidence.model_validate(raw.model_dump(mode="json"))
                if (evidence.episode_id != episode_id or evidence.config != config or
                        evidence.customer_snapshot_hash != customer.snapshot_hash or
                        evidence.service_snapshot_hash != service.snapshot_hash or evidence.track != track):
                    raise ValueError("environment returned evidence for another operation")
                self.archive.save(evidence)
                self.store.complete(self.config.run_id, OperationCompletion(operation_id=episode_id,
                                    input_hash=input_hash, consumption=evidence.consumption, evidence=evidence))
            except BaseException as exc:
                self.store.fault(self.config.run_id, episode_id, exc)
                raise InterruptedRun("episode has no safely completed result: " + episode_id) from None
        if (evidence.consumption.customer_llm_calls > reserve.customer_llm_calls or
                evidence.consumption.service_llm_calls > reserve.service_llm_calls or
                evidence.consumption.reflection_calls > 0 or
                evidence.consumption.tool_calls > reserve.tool_calls):
            raise RuntimeError("environment exceeded its reserved episode budget")
        return evidence

    def run_episode(self, config, customer, service, episode_id):
        return self._episode(config, customer, service, episode_id, "discovery",
                             lambda: self.adapter.run_episode(config, customer, service, episode_id))

    def run_service_task(self, config, task_ref, customer, service, episode_id):
        return self._episode(config, customer, service, episode_id, "service_task",
                             lambda: self.adapter.run_service_task(config, task_ref, customer, service, episode_id),
                             task_ref=task_ref)

    def run_failure_case(self, source, customer, service, episode_id):
        if source.config.partition != "development" or not source.failures or not source.replay.verified:
            raise ValueError("failure regression requires replay-verified development failure evidence")
        return self._episode(source.config, customer, service, episode_id, "failure_regression",
                             lambda: self.adapter.run_failure_case(source, customer, service, episode_id), source=source)


class CoEvolutionController:
    def __init__(self, adapter, customer_client, service_client, library_store, archive, run_store: RunStore):
        self.adapter, self.customer_client, self.service_client = adapter, customer_client, service_client
        self.library_store, self.archive, self.run_store = library_store, archive, run_store

    @staticmethod
    def _check_main_evidence(evidence):
        for value in evidence:
            if not value.replay.verified:
                raise ValueError("main execution did not pass Replay")
            if any(word in value.termination.casefold() for word in ("invalid", "protocol", "error", "interrupted")):
                raise RuntimeError("main episode terminated with " + value.termination + ": " + value.episode_id)

    def run(self, config: RunConfig) -> RunReport:
        from .evolvers import Evolver

        config = RunConfig.model_validate(config.model_dump(mode="json"))
        if config.schema_version not in (2, 3, 4):
            raise ValueError("historical run v1 requires its original code; create a new recursive run")
        config_hash = digest(config)
        # Configuration mismatch refuses reuse before overwriting a previous report.
        self.run_store.initialize(config)
        customer = self.library_store.load(config.initial_customer_hash)
        service = self.library_store.load(config.initial_service_hash)
        if customer.role != "customer" or service.role != "service":
            raise ValueError("run initial snapshot roles do not match Agents")
        for cases in (config.exploration_cases, config.validation_cases, config.service_task_cases):
            if len({case.id for case in cases}) != len(cases):
                raise ValueError("duplicate case IDs within a run phase")
        status, error = "completed", ""
        history = ()
        try:
            with self.run_store.lock(config.run_id):
                history = self.run_store.history(config.run_id)
                if len(history) > config.generations:
                    raise ValueError("stored generations exceed frozen run configuration")
                for item in history:
                    if item.schema_version != (2 if config.schema_version in (3, 4) else 1):
                        raise ValueError("generation checkpoint belongs to another training protocol")
                    if (item.parent_customer_hash, item.parent_service_hash) != (customer.snapshot_hash, service.snapshot_hash):
                        raise ValueError("generation history parent snapshot mismatch")
                    c_record = self.run_store.load_evolution(config.run_id, item.generation, "customer")
                    s_record = self.run_store.load_evolution(config.run_id, item.generation, "service")
                    if (c_record is None or s_record is None or
                            c_record.selected_snapshot_hash != item.selected_customer_hash or
                            s_record.selected_snapshot_hash != item.selected_service_hash):
                        raise ValueError("generation checkpoint and evolution records disagree")
                    customer = self.library_store.load(item.selected_customer_hash)
                    service = self.library_store.load(item.selected_service_hash)
                if self.run_store.pending(config.run_id):
                    # An accepted Customer update can precede an interrupted
                    # selected-Customer episode. Report that actual partial
                    # selection without advancing a completed generation.
                    generation = len(history)
                    c_partial = self.run_store.load_evolution(config.run_id, generation, "customer")
                    if c_partial is not None:
                        if (c_partial.parent_customer_hash, c_partial.parent_service_hash) != (customer.snapshot_hash, service.snapshot_hash):
                            raise ValueError("partial Customer evolution parent mismatch")
                        customer = self.library_store.load(c_partial.selected_snapshot_hash)
                    s_partial = self.run_store.load_evolution(config.run_id, generation, "service")
                    if s_partial is not None:
                        if (s_partial.parent_customer_hash, s_partial.parent_service_hash) != (customer.snapshot_hash, service.snapshot_hash):
                            raise ValueError("partial Service evolution parent mismatch")
                        service = self.library_store.load(s_partial.selected_snapshot_hash)
                    raise InterruptedRun("run contains an operation with unknown outcome")
                # A completed journal is replayed through the actual adapter,
                # including a previously completed whole run. Trace hashes alone
                # cannot validate source pins, frozen configuration or skills.
                for start in self.run_store.operation_starts(config.run_id):
                    completed = self.run_store.completion(config.run_id, start.operation_id)
                    if completed.evidence is not None:
                        evidence = completed.evidence
                        self.archive.save(evidence)
                        c_version = self.library_store.load(evidence.customer_snapshot_hash)
                        s_version = self.library_store.load(evidence.service_snapshot_hash)
                        replay = self.adapter.replay(evidence, c_version, s_version)
                        if not evidence.replay.verified or not replay.verified:
                            raise RuntimeError("stored episode failed exact Replay: " + evidence.episode_id)
                journal = _JournalAdapter(self.adapter, self.run_store, self.archive, config)
                for generation in range(len(history), config.generations):
                    parent_customer, parent_service = customer, service
                    prefix = "g" + str(generation) + "." + config.run_id
                    seeds = ()
                    if hasattr(self.adapter, "prepare_examiner_generation"):
                        previous = tuple(self.archive.load(h) for h in history[-1].current_evidence) if history else ()
                        seeds = self.adapter.prepare_examiner_generation(journal, config, generation, customer, service, previous, prefix)
                    current = seeds + tuple(journal.run_episode(case.config, customer, service,
                                    prefix + ".current." + case.id) for case in config.exploration_cases)
                    self._check_main_evidence(current)
                    c_record = self.run_store.load_evolution(config.run_id, generation, "customer")
                    if c_record is None and not config.customer_updates:
                        c_record = EvolutionRecord(generation=generation, role="customer",
                            parent_customer_hash=customer.snapshot_hash, parent_service_hash=service.snapshot_hash,
                            reflection=None, selected_snapshot_hash=customer.snapshot_hash, outcome="fixed",
                            reasons=("customer_updates_disabled_by_preregistered_ablation",), consumption=Consumption(),
                            customer_selection="customer-static-only-v1")
                        c_ref = self.run_store.save_evolution(config.run_id, c_record)
                    elif c_record is None:
                        client = _JournalClient(self.customer_client, self.run_store, config, prefix + ".customer.reflection")
                        reflector = CustomerReflector(client, config.customer_reflector_model)
                        c_record = Evolver("customer", journal, reflector, self.library_store,
                                          self.archive, config.policy).update(generation, customer, service, current,
                                          config.validation_cases, config.service_task_cases, prefix + ".customer")
                        c_ref = self.run_store.save_evolution(config.run_id, c_record)
                    else:
                        c_ref = str(self.run_store.evolution_path(config.run_id, generation, "customer"))
                    if (c_record.parent_customer_hash, c_record.parent_service_hash) != (customer.snapshot_hash, service.snapshot_hash):
                        raise ValueError("Customer evolution parent version mismatch")
                    customer = self.library_store.load(c_record.selected_snapshot_hash)
                    selected = () if config.schema_version in (3, 4) else tuple(journal.run_episode(case.config, customer, service,
                                     prefix + ".selected_customer." + case.id) for case in config.exploration_cases)
                    self._check_main_evidence(selected)
                    s_record = self.run_store.load_evolution(config.run_id, generation, "service")
                    if s_record is None:
                        client = _JournalClient(self.service_client, self.run_store, config, prefix + ".service.reflection")
                        reflector = ServiceReflector(client, config.service_reflector_model)
                        s_record = Evolver("service", journal, reflector, self.library_store,
                                          self.archive, config.policy).update(generation, customer, service, current + selected,
                                          config.validation_cases, config.service_task_cases, prefix + ".service")
                        s_ref = self.run_store.save_evolution(config.run_id, s_record)
                    else:
                        s_ref = str(self.run_store.evolution_path(config.run_id, generation, "service"))
                    if (s_record.parent_customer_hash, s_record.parent_service_hash) != (customer.snapshot_hash, service.snapshot_hash):
                        raise ValueError("Service evolution parent version mismatch")
                    service = self.library_store.load(s_record.selected_snapshot_hash)
                    crossings = []
                    legacy_pairings = () if config.schema_version in (3, 4) else (
                            ("parent_parent", parent_customer, parent_service),
                            ("selected_parent", customer, parent_service),
                            ("parent_selected", parent_customer, service),
                            ("selected_selected", customer, service))
                    for pairing, c_version, s_version in legacy_pairings:
                        for case in config.exploration_cases:
                            evidence = journal.run_episode(case.config, c_version, s_version,
                                       prefix + ".cross." + pairing + "." + case.id)
                            self._check_main_evidence((evidence,))
                            crossings.append(CrossExecution(pairing=pairing, case_id=case.id,
                                             customer_hash=c_version.snapshot_hash, service_hash=s_version.snapshot_hash,
                                             evidence_hash=evidence.evidence_hash))
                    checkpoint = GenerationRecord(schema_version=2 if config.schema_version in (3, 4) else 1, generation=generation,
                                 parent_customer_hash=parent_customer.snapshot_hash, parent_service_hash=parent_service.snapshot_hash,
                                 selected_customer_hash=customer.snapshot_hash, selected_service_hash=service.snapshot_hash,
                                 customer_evolution_ref=c_ref, service_evolution_ref=s_ref,
                                 current_evidence=tuple(value.evidence_hash for value in current),
                                 selected_customer_evidence=tuple(value.evidence_hash for value in selected),
                                 crossings=tuple(crossings))
                    self.run_store.save_generation(config.run_id, checkpoint)
                    history += (checkpoint,)
        except BudgetExhausted as exc:
            status, error = "budget_exhausted", str(exc)
        except (InterruptedRun, KeyboardInterrupt) as exc:
            status, error = "interrupted", str(exc)
        except Exception as exc:
            status, error = "failed", type(exc).__name__ + ": " + str(exc)
        report = RunReport(run_id=config.run_id, config_hash=config_hash, status=status,
                           generations=history, selected_customer_hash=customer.snapshot_hash,
                           selected_service_hash=service.snapshot_hash,
                           consumption=self.run_store.consumption(config.run_id),
                           pending_operations=self.run_store.pending(config.run_id), error=error)
        return self.run_store.save_report(report)


Controller = CoEvolutionController

