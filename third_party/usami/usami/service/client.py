"""Small OpenAI-compatible Chat Completions transport; no hidden retries."""
from __future__ import annotations

import copy
import json
import os
import subprocess
from typing import Any, Dict, Mapping, Optional, Protocol

from usami.service.models import ModelSettings, ServiceError, strict_json


class CompletionClient(Protocol):
    settings: ModelSettings

    def complete(self, request: Dict[str, Any]) -> Any:
        ...


class RecordedCompletionClient:
    """Offline response journal used by both Service and interaction replay."""
    def __init__(self, settings, records):
        self.settings = settings.model_copy(deep=True)
        self.records = copy.deepcopy(records)
        self.index = 0

    def complete(self, request):
        if self.index >= len(self.records):
            raise ValueError("missing recorded model response")
        record = self.records[self.index]
        self.index += 1
        if request != record.request:
            raise ValueError("model conversation or tool schema differs from recorded input")
        if record.error.get("stage") == "transport":
            raise ServiceError(record.error["code"], record.error["message"], copy.deepcopy(record.response))
        return copy.deepcopy(record.response)


def settings_from_env(env: Optional[Mapping[str, str]] = None):
    env = os.environ if env is None else env
    model = env.get("USAMI_LLM_MODEL") or env.get("OPENAI_MODEL")
    base = env.get("USAMI_LLM_BASE_URL") or env.get("OPENAI_BASE_URL")
    if not model or not base:
        raise ValueError("Set USAMI_LLM_MODEL and USAMI_LLM_BASE_URL")
    options = strict_json(env.get("USAMI_LLM_OPTIONS", '{"max_tokens":2048}'))
    if not isinstance(options, dict):
        raise ValueError("USAMI_LLM_OPTIONS must be a JSON object")
    return ModelSettings(model=model, base_url=base, options=options,
                         timeout_seconds=float(env.get("USAMI_LLM_TIMEOUT", "60")),
                         trust_env=env.get("USAMI_LLM_TRUST_ENV") == "1")


def key_from_env(env: Optional[Mapping[str, str]] = None):
    env = os.environ if env is None else env
    key = env.get("USAMI_LLM_API_KEY") or env.get("OPENAI_API_KEY")
    if not key and env.get("USAMI_LLM_KEYCHAIN_SERVICE"):
        account = env.get("USAMI_LLM_KEYCHAIN_ACCOUNT")
        if not account:
            raise ValueError("Set USAMI_LLM_KEYCHAIN_ACCOUNT with the Keychain service")
        result = subprocess.run(["security", "find-generic-password", "-s", env["USAMI_LLM_KEYCHAIN_SERVICE"],
                                 "-a", account, "-w"], capture_output=True, text=True)
        if result.returncode:
            raise ValueError("Keychain lookup failed; credential was not displayed")
        key = result.stdout.strip()
    if not key and env.get("USAMI_LLM_ALLOW_NO_AUTH") != "1":
        raise ValueError("Set USAMI_LLM_API_KEY or configure a Keychain entry (local no-auth requires USAMI_LLM_ALLOW_NO_AUTH=1)")
    return key or ""


class OpenAICompatibleClient:
    MAX_ERROR_BODY_CHARS = 4096

    def __init__(self, settings: ModelSettings, api_key: str, transport=None):
        # Import only in the optional integration layer: the benchmark has no HTTP dependency.
        try:
            import httpx
        except ImportError:
            raise ValueError("Install the optional integration: pip install -e '.[service]'") from None
        self.settings = settings.model_copy(deep=True)
        self._key = api_key
        self._http = httpx.Client(timeout=settings.timeout_seconds, trust_env=settings.trust_env,
                                  follow_redirects=False, transport=transport)

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None):
        return cls(settings_from_env(env), key_from_env(env))

    def close(self):
        self._http.close()

    def _redact(self, value):
        text = json.dumps(value, ensure_ascii=False)
        if self._key:
            text = text.replace(self._key, "[REDACTED]")
        return json.loads(text)

    def _safe_http_error(self, response):
        """Keep bounded provider diagnostics while redacting the configured key."""
        body = response.text
        truncated = len(body) > self.MAX_ERROR_BODY_CHARS
        body = body[:self.MAX_ERROR_BODY_CHARS]
        if self._key:
            body = body.replace(self._key, "[REDACTED]")
        return {"status_code": response.status_code, "body": body, "truncated": truncated}

    @staticmethod
    def _provider_payload(request):
        """Remove internal message metadata and empty optional tool-call fields.

        Audited messages may carry a record_hash and an empty tool_calls list.
        Those fields are not useful to the provider; some OpenAI-compatible
        backends reject an explicitly empty tool_calls array. Keep the
        recorded request untouched for exact offline replay.
        """
        payload = copy.deepcopy(request)
        messages = payload.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                message.pop("record_hash", None)
                if message.get("tool_calls") in ([], None):
                    message.pop("tool_calls", None)
        return payload

    def complete(self, request):
        import httpx
        # Do not save headers, exception request objects, or raw HTTP error bodies.
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = "Bearer " + self._key
        try:
            response = self._http.post(self.settings.base_url + "/chat/completions", headers=headers,
                                       json=self._provider_payload(request))
        except httpx.TimeoutException as exc:
            # Preserve only the timeout class, never the request or URL. This
            # distinguishes connect/read/write/pool timeouts in audited records.
            raise ServiceError("PROVIDER_TIMEOUT", "LLM request timed out; no retry performed",
                               {"timeout_phase": type(exc).__name__}) from None
        except httpx.HTTPError:
            raise ServiceError("PROVIDER_CONNECTION_ERROR", "LLM transport failed; no retry performed") from None
        if response.status_code != 200:
            raise ServiceError("PROVIDER_HTTP_ERROR",
                f"LLM endpoint returned HTTP {response.status_code}; no retry performed",
                self._safe_http_error(response))
        try:
            raw = strict_json(response.text)
        except ValueError:
            raise ServiceError("MODEL_OUTPUT_INVALID", "LLM response is not valid JSON", self._redact(response.text)) from None
        return self._redact(raw)
