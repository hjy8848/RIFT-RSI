"""Inject declarative skills without changing the native session or authority."""
from __future__ import annotations

import copy
import json
from typing import Callable, Optional

from ..models import InjectionRecord, ModelCall, ModelSpec
from ..schema import digest
from ..skills.library import LibrarySnapshot
from ..skills.renderer import render_skills
from ..skills.retriever import SkillContext, SkillRetriever


def request_context(request, domain, tools):
    """Use only messages already present in the Agent's public request.

    An adapter may supply a narrower parser for native typed intentions. This
    default never receives an environment, task witness or private evaluator.
    """
    messages = [message for message in request.get("messages", []) if message.get("role") in ("user", "tool", "assistant")]
    return SkillContext(domain=domain, public_text=json.dumps(messages, ensure_ascii=False, sort_keys=True),
                        available_tools=tuple(tools))


class RuntimeClient:
    def __init__(self, inner, role, library, domain, tools=(), context_factory=None,
                 model: Optional[ModelSpec] = None, observer: Optional[Callable] = None,
                 episode_id="episode", max_skills=4, max_chars=6000):
        self.inner = inner
        self.settings = getattr(inner, "settings", None)
        self.role = role
        self.library = LibrarySnapshot.model_validate(library.model_dump(mode="json"))
        if self.library.role != role:
            raise ValueError("runtime role differs from its skill library")
        self.domain, self.tools = domain, tuple(tools)
        self.context_factory = context_factory or (lambda req: request_context(req, domain, tools))
        if model is None:
            if self.settings is None:
                raise ValueError("an explicit frozen model spec or client settings are required")
            model = ModelSpec(model=self.settings.model, base_url=self.settings.base_url,
                              options=self.settings.options, timeout_seconds=self.settings.timeout_seconds,
                              trust_env=self.settings.trust_env)
        self.model = ModelSpec.model_validate(model.model_dump(mode="json"))
        self.model_hash = digest(self.model)
        self.observer = observer
        self.episode_id = episode_id
        self.retriever = SkillRetriever(role, max_skills=max_skills, max_chars=max_chars)
        self._records, self._injections = [], []

    @property
    def records(self):
        return tuple(ModelCall.model_validate(record.model_dump(mode="json")) for record in self._records)

    @property
    def injections(self):
        return tuple(self._injections)

    def complete(self, request):
        model = ModelSpec.model_validate(self.model.model_dump(mode="json"))
        if digest(model) != self.model_hash:
            raise ValueError("runtime model changed after construction")
        base = copy.deepcopy(request)
        if base.get("model") != self.model.model:
            raise ValueError("runtime request changed its frozen model")
        context = self.context_factory(copy.deepcopy(base))
        if not isinstance(context, SkillContext):
            context = SkillContext.model_validate(context)
        if context.domain != self.domain or set(context.available_tools) - set(self.tools):
            raise ValueError("retrieval context exceeds the configured domain or tools")
        retrieved = self.retriever.retrieve(self.library, context)
        rendered = render_skills(retrieved.skills) if retrieved.skills else ""
        wire = copy.deepcopy(base)
        if rendered:
            # Insert after native system messages; neither replace them nor add tools.
            messages = wire.setdefault("messages", [])
            index = 0
            while index < len(messages) and messages[index].get("role") == "system":
                index += 1
            messages.insert(index, {"role": "system", "content": rendered})
        call_id = self.episode_id + "." + self.role + "." + str(len(self._records) + 1)
        injection = InjectionRecord(call_id=call_id, role=self.role, snapshot_hash=self.library.snapshot_hash,
                                    context_hash=digest(context.model_dump(mode="json")), hits=retrieved.hits,
                                    rendered=rendered, request_hash=digest(wire))
        self._injections.append(injection)
        pending = dict(id=call_id, role=self.role, model=self.model, base_request=base, request=wire)
        if self.observer:
            self.observer("started", ModelCall(**copy.deepcopy(pending)))
        try:
            response = self.inner.complete(copy.deepcopy(wire))
        except (Exception, KeyboardInterrupt) as exc:
            error = {"type": type(exc).__name__, "code": getattr(exc, "code", "client_error"),
                     "message": str(exc)}
            raw_response = getattr(exc, "response", None)
            try:
                json.dumps(raw_response, allow_nan=False)
            except (TypeError, ValueError):
                raw_response = None
            record = ModelCall(**pending, error=error, response=copy.deepcopy(raw_response))
            self._records.append(record)
            if self.observer:
                self.observer("completed", ModelCall.model_validate(record.model_dump(mode="json")))
            raise  # Native session owns its exact exception/termination semantics.
        record = ModelCall(**pending, response=copy.deepcopy(response))
        self._records.append(record)
        if self.observer:
            self.observer("completed", ModelCall.model_validate(record.model_dump(mode="json")))
        return response
