from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
import json
import math
import os
from pathlib import Path
import shlex
import time
from dataclasses import asdict
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .interaction_runtime import BrainRequest


DEFAULT_HERMES_ENDPOINT = "http://127.0.0.1:8642/v1/chat/completions"
DEFAULT_HERMES_MODEL = "gpt-5.6-luna"
DEFAULT_HERMES_PROVIDER = "openai-codex"
DEFAULT_HERMES_REASONING_EFFORT = "low"
DEFAULT_TIMEOUT_SECONDS = 90.0
MAX_RESPONSE_BYTES = 128 * 1024
_HERMES_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
)


from .providers.decision import _SYSTEM_PROMPT


class HermesBrainError(RuntimeError):
    """Raised when the configured Hermes endpoint cannot complete a request."""


class HermesBrainProtocolError(HermesBrainError):
    """Raised when Hermes returns a response outside the expected wire contract."""


class HermesBrainAdapter:
    """Adapts one frozen interaction turn to the local Hermes API server."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        model: str = DEFAULT_HERMES_MODEL,
        provider: str = DEFAULT_HERMES_PROVIDER,
        reasoning_effort: str = DEFAULT_HERMES_REASONING_EFFORT,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self._endpoint = _validate_endpoint(endpoint)
        self._api_key = _require_nonempty(api_key, name="api_key")
        self._model = _require_nonempty(model, name="model")
        self._provider = _require_nonempty(provider, name="provider")
        self._reasoning_effort = _validate_reasoning_effort(reasoning_effort)
        self._timeout_seconds = _validate_timeout(timeout_seconds)
        self._opener = opener

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> HermesBrainAdapter:
        env = os.environ if environment is None else environment
        api_key = env.get("API_SERVER_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "API_SERVER_KEY is required; load the existing Hermes environment first"
            )
        endpoint = env.get("VRC_ARDY_HERMES_ENDPOINT", DEFAULT_HERMES_ENDPOINT)
        model = env.get("VRC_ARDY_HERMES_MODEL", DEFAULT_HERMES_MODEL)
        provider = env.get("VRC_ARDY_HERMES_PROVIDER", DEFAULT_HERMES_PROVIDER)
        reasoning_effort = env.get(
            "VRC_ARDY_HERMES_REASONING_EFFORT",
            DEFAULT_HERMES_REASONING_EFFORT,
        )
        raw_timeout = env.get(
            "VRC_ARDY_HERMES_TIMEOUT_SECONDS",
            str(DEFAULT_TIMEOUT_SECONDS),
        )
        try:
            timeout_seconds = float(raw_timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "VRC_ARDY_HERMES_TIMEOUT_SECONDS must be a number"
            ) from exc
        return cls(
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            provider=provider,
            reasoning_effort=reasoning_effort,
            timeout_seconds=timeout_seconds,
        )

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def reasoning_effort(self) -> str:
        return self._reasoning_effort

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    def snapshot(self) -> dict[str, object]:
        return {
            "endpoint": self._endpoint,
            "model": self._model,
            "provider": self._provider,
            "reasoning_effort": self._reasoning_effort,
            "timeout_seconds": self._timeout_seconds,
        }

    def propose(self, request: BrainRequest) -> object:
        payload = self._build_payload(request)
        wire_body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        http_request = Request(
            self._endpoint,
            data=wire_body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Idempotency-Key": (
                    f"brain-{request.turn_id}-v{request.turn_version}-r{request.round_index}"
                ),
            },
        )

        try:
            with self._opener(
                http_request,
                timeout=min(
                    self._timeout_seconds,
                    max(0.01, request.deadline_monotonic - time.monotonic()),
                )
                if request.deadline_monotonic is not None
                else self._timeout_seconds,
            ) as response:
                status = int(getattr(response, "status", 200))
                response_body = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            raise HermesBrainError(f"Hermes returned HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise HermesBrainError(
                f"Hermes request failed: {type(exc).__name__}: {exc}"
            ) from exc

        if status < 200 or status >= 300:
            raise HermesBrainError(f"Hermes returned HTTP {status}")
        if len(response_body) > MAX_RESPONSE_BYTES:
            raise HermesBrainProtocolError("Hermes response is too large")

        try:
            envelope = json.loads(response_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HermesBrainProtocolError("Hermes response is not valid JSON") from exc
        if request.record_model_metadata is not None and isinstance(envelope, dict):
            request.record_model_metadata({
                "usage": envelope.get("usage"),
                "model": envelope.get("model"),
                "response_id": envelope.get("id"),
                "finish_reason": next((c.get("finish_reason") for c in envelope.get("choices", [])
                                       if isinstance(c, dict)), None),
            })
        content = _assistant_content(envelope)
        try:
            plan = json.loads(content)
        except json.JSONDecodeError as exc:
            raise HermesBrainProtocolError(
                "Hermes assistant content must be one plain JSON object"
            ) from exc
        if not isinstance(plan, dict):
            raise HermesBrainProtocolError(
                "Hermes assistant content must be one plain JSON object"
            )
        if plan.get("type") != "observe_scene" and not isinstance(
            plan.get("requires_visual"), bool
        ):
            raise HermesBrainProtocolError(
                "final decision requires requires_visual boolean"
            )
        return plan

    def _build_payload(self, request: BrainRequest) -> dict[str, object]:
        if request.screenshot is not None and (
            not isinstance(request.screenshot, bytes) or not request.screenshot
        ):
            raise ValueError("request.screenshot must contain JPEG bytes when supplied")
        turn_context = {
            "turn_id": request.turn_id,
            "turn_version": request.turn_version,
            "transcript": request.transcript,
            "round_index": request.round_index,
            "observations": [asdict(item) for item in request.observations],
            "observation_history": list(request.conversation),
            "execution_feedback": request.execution_feedback,
            "runtime": {
                "body": request.body.value,
                "speech": request.speech.value,
                "current_action_ids": list(request.current_action_ids),
                "last_error": request.last_error,
            },
        }
        image_data = (
            base64.b64encode(request.screenshot).decode("ascii")
            if request.screenshot
            else None
        )
        return {
            "model": self._model,
            "provider": self._provider,
            "model_options": {
                "reasoning_effort": self._reasoning_effort,
            },
            "stream": False,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                turn_context,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                        *(
                            [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}",
                                        "detail": "high",
                                    },
                                }
                            ]
                            if image_data is not None
                            else []
                        ),
                    ],
                },
            ],
        }


def _assistant_content(envelope: object) -> str:
    if not isinstance(envelope, dict):
        raise HermesBrainProtocolError(
            "Hermes response does not contain assistant content"
        )
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices:
        raise HermesBrainProtocolError(
            "Hermes response does not contain assistant content"
        )
    first = choices[0]
    if not isinstance(first, dict):
        raise HermesBrainProtocolError(
            "Hermes response does not contain assistant content"
        )
    if first.get("finish_reason") not in (None, "stop"):
        raise HermesBrainProtocolError("Hermes completion is incomplete or unsupported")
    extra = envelope.get("hermes", {})
    if isinstance(extra, dict) and (
        extra.get("failed") or extra.get("partial") or extra.get("completed") is False
    ):
        raise HermesBrainProtocolError("Hermes completion failed or is partial")
    message = first.get("message")
    if not isinstance(message, dict):
        raise HermesBrainProtocolError(
            "Hermes response does not contain assistant content"
        )
    if message.get("tool_calls"):
        raise HermesBrainProtocolError(
            "native tool calls are unsupported on this decision route"
        )
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise HermesBrainProtocolError(
            "Hermes response does not contain assistant content"
        )
    return content


def _validate_endpoint(value: str) -> str:
    endpoint = _require_nonempty(value, name="endpoint")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("endpoint must be an HTTP or HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("endpoint must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("endpoint must not contain a query or fragment")
    if not parsed.path.endswith("/v1/chat/completions"):
        raise ValueError("endpoint must target /v1/chat/completions")
    return endpoint


def _require_nonempty(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value.strip()


def _validate_timeout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timeout_seconds must be a number")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    return timeout


def _validate_reasoning_effort(value: str) -> str:
    effort = _require_nonempty(value, name="reasoning_effort").lower()
    if effort not in _HERMES_REASONING_EFFORTS:
        supported = ", ".join(sorted(_HERMES_REASONING_EFFORTS))
        raise ValueError(f"reasoning_effort must be one of: {supported}")
    return effort


def load_hermes_environment(
    env_file: Path,
    *,
    base_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if base_environment is None else base_environment)
    if environment.get("API_SERVER_KEY", "").strip():
        return environment
    try:
        lines = env_file.expanduser().read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(
            "API_SERVER_KEY is not set and the Hermes environment file cannot be read"
        ) from exc
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, raw_value = line.partition("=")
        if separator and name.strip() == "API_SERVER_KEY":
            tokens = shlex.split(raw_value, comments=True, posix=True)
            if len(tokens) != 1 or not tokens[0].strip():
                raise RuntimeError("Hermes API_SERVER_KEY is empty or malformed")
            environment["API_SERVER_KEY"] = tokens[0]
            return environment
    raise RuntimeError("Hermes environment does not define API_SERVER_KEY")

