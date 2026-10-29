"""Atomic, immutable operation journals and restartable generation history."""
from __future__ import annotations

import copy
import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

from pydantic import Field, JsonValue, model_validator

from ..evolution.models import EvolutionRecord, RunConfig
from ..models import Consumption, EpisodeEvidence
from ..schema import FrozenModel, atomic_json, digest
from ..skills.models import Hash


class RunHalt(BaseException):
    """A control stop must not be swallowed as an ordinary model error."""


class BudgetExhausted(RunHalt):
    pass


class InterruptedRun(RunHalt):
    pass


class JournalError(FrozenModel):
    type: str
    message: str
    code: str = "client_error"
    response: Optional[JsonValue] = None
    response_recorded: bool = False


class OperationStart(FrozenModel):
    schema_version: Literal[1] = 1
    operation_id: str = Field(min_length=1)
    kind: Literal["episode", "reflection"]
    input_hash: Hash
    reservation: Consumption
    start_hash: str = ""

    @model_validator(mode="after")
    def seal(self):
        expected = digest(self.model_dump(mode="json", exclude={"start_hash"}))
        if self.start_hash and self.start_hash != expected:
            raise ValueError("operation start hash mismatch")
        object.__setattr__(self, "start_hash", expected)
        return self


class OperationCompletion(FrozenModel):
    schema_version: Literal[1] = 1
    operation_id: str = Field(min_length=1)
    input_hash: Hash
    consumption: Consumption
    evidence: Optional[EpisodeEvidence] = None
    response: Optional[JsonValue] = None
    response_recorded: bool = False
    error: Optional[JournalError] = None
    completion_hash: str = ""

    @model_validator(mode="after")
    def seal(self):
        if sum((self.evidence is not None, self.response_recorded, self.error is not None)) != 1:
            raise ValueError("a completed operation requires exactly one typed result")
        if self.response is not None and not self.response_recorded:
            raise ValueError("a raw response requires its recorded marker")
        if self.evidence is not None and self.evidence.consumption != self.consumption:
            raise ValueError("episode completion cost differs from evidence consumption")
        expected = digest(self.model_dump(mode="json", exclude={"completion_hash"}))
        if self.completion_hash and self.completion_hash != expected:
            raise ValueError("operation completion hash mismatch")
        object.__setattr__(self, "completion_hash", expected)
        return self


class CrossExecution(FrozenModel):
    pairing: Literal["parent_parent", "selected_parent", "parent_selected", "selected_selected"]
    case_id: str
    customer_hash: Hash
    service_hash: Hash
    evidence_hash: Hash


class StoredEvolution(FrozenModel):
    schema_version: Literal[1] = 1
    record: EvolutionRecord
    record_hash: str = ""

    @model_validator(mode="after")
    def seal(self):
        expected = digest(self.record)
        if self.record_hash and self.record_hash != expected:
            raise ValueError("evolution record hash mismatch")
        object.__setattr__(self, "record_hash", expected)
        return self


class GenerationRecord(FrozenModel):
    schema_version: Literal[1, 2] = 1
    generation: int = Field(ge=0, strict=True)
    parent_customer_hash: Hash
    parent_service_hash: Hash
    selected_customer_hash: Hash
    selected_service_hash: Hash
    customer_evolution_ref: str
    service_evolution_ref: str
    current_evidence: Tuple[Hash, ...]
    selected_customer_evidence: Tuple[Hash, ...]
    crossings: Tuple[CrossExecution, ...]
    record_hash: str = ""

    @model_validator(mode="after")
    def seal(self):
        if self.schema_version == 2 and (self.selected_customer_evidence or self.crossings):
            raise ValueError("recursive generation cannot contain selected-Customer comparison or crossings")
        expected = digest(self.model_dump(mode="json", exclude={"record_hash"}))
        if self.record_hash and self.record_hash != expected:
            raise ValueError("generation record hash mismatch")
        object.__setattr__(self, "record_hash", expected)
        return self


class RunReport(FrozenModel):
    schema_version: Literal[1] = 1
    run_id: str
    config_hash: Hash
    status: Literal["completed", "budget_exhausted", "interrupted", "failed"]
    generations: Tuple[GenerationRecord, ...]
    selected_customer_hash: Hash
    selected_service_hash: Hash
    consumption: Consumption
    pending_operations: Tuple[str, ...] = ()
    error: str = ""
    report_hash: str = ""

    @model_validator(mode="after")
    def seal(self):
        expected = digest(self.model_dump(mode="json", exclude={"report_hash"}))
        if self.report_hash and self.report_hash != expected:
            raise ValueError("run report hash mismatch")
        object.__setattr__(self, "report_hash", expected)
        return self


def add_consumption(values):
    return Consumption(**{key: sum(getattr(value, key) for value in values)
                          for key in Consumption.model_fields})


class RunStore:
    def __init__(self, root):
        self.root = Path(root)

    def run_path(self, run_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", run_id):
            raise ValueError("invalid run ID")
        return self.root / run_id

    def initialize(self, config):
        config = RunConfig.model_validate(config.model_dump(mode="json"))
        atomic_json(self.run_path(config.run_id) / "config.json", config.model_dump(mode="json"), immutable=True)
        return digest(config)

    def load_config(self, run_id):
        return RunConfig.model_validate_json((self.run_path(run_id) / "config.json").read_text())

    @contextmanager
    def lock(self, run_id):
        # flock survives process crashes: the OS releases it without deleting a
        # marker that could otherwise leave a permanently locked run.
        import fcntl
        path = self.run_path(run_id)
        path.mkdir(parents=True, exist_ok=True)
        with (path / ".lock").open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise InterruptedRun("another process owns this run") from None
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _operation_path(self, run_id, operation_id, suffix):
        return self.run_path(run_id) / "operations" / (digest(operation_id) + suffix + ".json")

    def start(self, run_id, operation_id, kind, input_hash, reservation, max_llm_calls, max_tool_calls):
        path = self._operation_path(run_id, operation_id, ".started")
        if path.exists():
            previous = OperationStart.model_validate_json(path.read_text())
            if (previous.operation_id, previous.kind, previous.input_hash, previous.reservation) != (operation_id, kind, input_hash, reservation):
                raise ValueError("operation input changed during resume: " + operation_id)
            completed = self.completion(run_id, operation_id)
            if completed is None:
                raise InterruptedRun("operation was started without a completed result: " + operation_id)
            return completed
        used = self.consumption(run_id)
        if used.llm_calls + reservation.llm_calls > max_llm_calls or used.tool_calls + reservation.tool_calls > max_tool_calls:
            raise BudgetExhausted("global call budget cannot reserve operation: " + operation_id)
        start = OperationStart(operation_id=operation_id, kind=kind, input_hash=input_hash, reservation=reservation)
        atomic_json(path, start.model_dump(mode="json"), immutable=True)
        return None

    def complete(self, run_id, value):
        value = OperationCompletion.model_validate(value.model_dump(mode="json"))
        start = OperationStart.model_validate_json(self._operation_path(run_id, value.operation_id, ".started").read_text())
        if start.input_hash != value.input_hash:
            raise ValueError("completed operation input hash mismatch")
        if (start.kind == "episode") != (value.evidence is not None):
            raise ValueError("completed result disagrees with operation kind")
        if start.kind == "reflection" and value.consumption != Consumption(reflection_calls=1):
            raise ValueError("Reflection completion must consume exactly one call")
        if any(getattr(value.consumption, key) > getattr(start.reservation, key) for key in Consumption.model_fields):
            raise ValueError("operation consumption exceeds its reserved upper bound")
        atomic_json(self._operation_path(run_id, value.operation_id, ".completed"), value.model_dump(mode="json"), immutable=True)

    def completion(self, run_id, operation_id):
        path = self._operation_path(run_id, operation_id, ".completed")
        if not path.exists():
            return None
        value = OperationCompletion.model_validate_json(path.read_text())
        if value.operation_id != operation_id:
            raise ValueError("operation filename and identity mismatch")
        start = OperationStart.model_validate_json(self._operation_path(run_id, operation_id, ".started").read_text())
        if value.input_hash != start.input_hash:
            raise ValueError("operation completion does not match its journal")
        return value

    def operation_starts(self, run_id):
        result = []
        for path in sorted((self.run_path(run_id) / "operations").glob("*.started.json")):
            value = OperationStart.model_validate_json(path.read_text())
            if path != self._operation_path(run_id, value.operation_id, ".started"):
                raise ValueError("operation start filename mismatch")
            result.append(value)
        return tuple(result)

    def pending(self, run_id):
        return tuple(start.operation_id for start in self.operation_starts(run_id)
                     if self.completion(run_id, start.operation_id) is None)

    def consumption(self, run_id):
        values = []
        for start in self.operation_starts(run_id):
            completed = self.completion(run_id, start.operation_id)
            # Unknown in-flight work retains its upper bound; restarting cannot
            # release budget for calls whose remote outcome is unknown.
            values.append(start.reservation if completed is None else completed.consumption)
        return add_consumption(values)

    def fault(self, run_id, operation_id, exc):
        atomic_json(self._operation_path(run_id, operation_id, ".interrupted"),
                    JournalError(type=type(exc).__name__, message=str(exc), code=getattr(exc, "code", "interrupted")).model_dump(mode="json"), immutable=True)

    def evolution_path(self, run_id, generation, role):
        if type(generation) is not int or generation < 0 or role not in {"customer", "service"}:
            raise ValueError("invalid generation or role")
        return self.run_path(run_id) / "evolution" / ("g" + str(generation) + "." + role + ".json")

    def save_evolution(self, run_id, record):
        record = EvolutionRecord.model_validate(record.model_dump(mode="json"))
        path = self.evolution_path(run_id, record.generation, record.role)
        atomic_json(path, StoredEvolution(record=record).model_dump(mode="json"), immutable=True)
        return str(path)

    def load_evolution(self, run_id, generation, role):
        path = self.evolution_path(run_id, generation, role)
        if not path.exists():
            return None
        record = StoredEvolution.model_validate_json(path.read_text()).record
        if record.generation != generation or record.role != role:
            raise ValueError("evolution filename mismatch")
        return record

    def save_generation(self, run_id, record):
        record = GenerationRecord.model_validate(record.model_dump(mode="json"))
        atomic_json(self.run_path(run_id) / "generations" / ("g" + str(record.generation) + ".json"),
                    record.model_dump(mode="json"), immutable=True)
        return record.record_hash

    def history(self, run_id):
        records = []
        for path in (self.run_path(run_id) / "generations").glob("g*.json"):
            record = GenerationRecord.model_validate_json(path.read_text())
            if path.name != "g" + str(record.generation) + ".json":
                raise ValueError("generation filename mismatch")
            records.append(record)
        ordered = tuple(sorted(records, key=lambda record: record.generation))
        if tuple(record.generation for record in ordered) != tuple(range(len(ordered))):
            raise ValueError("generation history is not contiguous")
        return ordered

    def save_report(self, report):
        report = RunReport.model_validate(report.model_dump(mode="json"))
        atomic_json(self.run_path(report.run_id) / "report.json", report.model_dump(mode="json"))
        atomic_json(self.run_path(report.run_id) / "reports" / (report.report_hash + ".json"),
                    report.model_dump(mode="json"), immutable=True)
        return report

    def load_report(self, run_id):
        return RunReport.model_validate_json((self.run_path(run_id) / "report.json").read_text())

    def branch_config(self, source_run_id, new_run_id, generation=None, generations=None):
        if source_run_id == new_run_id:
            raise ValueError("restore must create a distinct branch run")
        source = self.load_config(source_run_id)
        history = self.history(source_run_id)
        if not history:
            raise ValueError("source run has no completed generation")
        if generation is None:
            chosen = history[-1]
        else:
            chosen = next((item for item in history if item.generation == generation), None)
            if chosen is None:
                raise ValueError("requested generation is not completed")
        payload = source.model_dump(mode="json")
        payload.update(run_id=new_run_id, initial_customer_hash=chosen.selected_customer_hash,
                       initial_service_hash=chosen.selected_service_hash)
        if generations is not None:
            payload["generations"] = generations
        config = RunConfig.model_validate(copy.deepcopy(payload))
        self.initialize(config)
        atomic_json(self.run_path(new_run_id) / "branch.json", dict(source_run_id=source_run_id,
                    generation=chosen.generation, source_record_hash=chosen.record_hash), immutable=True)
        return config

    restore = branch_config

