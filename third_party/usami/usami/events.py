from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import Field

from usami.world.models import Model, WorldState, digest


class Event(Model):
    event_id: str
    kind: Literal["generation", "request", "operation", "oracle", "final"]
    payload: Dict[str, Any] = Field(default_factory=dict)
    before: Optional[WorldState] = None
    after: Optional[WorldState] = None
    previous_hash: str = ""
    event_hash: str = ""

    def computed_hash(self) -> str:
        return digest(self.model_dump(mode="json", exclude={"event_hash"}))


class Trace(Model):
    schema_version: str = "usami-trace-0.1.0"
    initial_state: WorldState
    initial_state_hash: str
    rules_hash: str
    operation_budget: int = Field(default=100, ge=1)
    events: List[Event] = Field(default_factory=list)
    final_state: WorldState
    final_state_hash: str

    def append(self, kind: str, payload: Dict, before: Optional[WorldState] = None,
               after: Optional[WorldState] = None) -> Event:
        event = Event(event_id=f"event_{len(self.events):06d}", kind=kind, payload=payload,
                      before=before.clone() if before else None, after=after.clone() if after else None,
                      previous_hash=self.events[-1].event_hash if self.events else "")
        event.event_hash = event.computed_hash()
        self.events.append(event)
        return event

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> Trace:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def integrity_errors(self) -> List[str]:
        errors = []
        expected_schema = "usami-trace-0.2.0" if self.initial_state.rules.version == "saas-rules-0.2.0" else "usami-trace-0.1.0"
        if self.schema_version != expected_schema:
            errors.append("unsupported trace schema")
        if self.initial_state_hash != self.initial_state.state_hash:
            errors.append("initial state hash mismatch")
        if self.rules_hash != digest(self.initial_state.rules):
            errors.append("rules hash mismatch")
        if self.final_state_hash != self.final_state.state_hash:
            errors.append("final state hash mismatch")
        previous = ""
        for n, event in enumerate(self.events):
            if event.event_id != f"event_{n:06d}" or event.previous_hash != previous or event.event_hash != event.computed_hash():
                errors.append(f"event integrity mismatch: {event.event_id}")
            previous = event.event_hash
        return errors
