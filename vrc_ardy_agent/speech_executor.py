from __future__ import annotations

from collections.abc import Callable
import json
import math
import threading
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .action_contracts import (
    ControlResource,
    ExecutionCommand,
    SayAction,
)
from .device_payloads import MAX_WAV_PAYLOAD_BYTES


DEFAULT_TTS_ENDPOINT = "http://127.0.0.1:8001/v1/audio/speech"
DEFAULT_TTS_MODEL = "gpt-sovits-v2proplus"
DEFAULT_TTS_VOICE = "charlotte"
DEFAULT_TTS_TIMEOUT_SECONDS = 120.0
MAX_WAV_BYTES = MAX_WAV_PAYLOAD_BYTES


class SpeechSynthesisError(RuntimeError):
    """Raised when the configured speech service cannot produce a WAV file."""


class SpeechSynthesizer(Protocol):
    def synthesize(self, text: str) -> bytes: ...


class SpeechOutput(Protocol):
    def play_wav(
        self,
        command: ExecutionCommand,
        wav_bytes: bytes,
        cancel_event: threading.Event,
    ) -> None: ...


class GptSovitsClient:
    """OpenAI-compatible client for the configured resident voice route."""

    def __init__(
        self,
        *,
        endpoint: str = DEFAULT_TTS_ENDPOINT,
        model: str = DEFAULT_TTS_MODEL,
        voice: str = DEFAULT_TTS_VOICE,
        timeout_seconds: float = DEFAULT_TTS_TIMEOUT_SECONDS,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.endpoint = _validate_endpoint(endpoint)
        self.model = _nonempty(model, name="model")
        self.voice = _nonempty(voice, name="voice")
        self.timeout_seconds = _positive_finite(
            timeout_seconds,
            name="timeout_seconds",
        )
        self._opener = opener

    def synthesize(self, text: str) -> bytes:
        text = _nonempty(text, name="text")
        body = json.dumps(
            {
                "model": self.model,
                "input": text,
                "voice": self.voice,
                "response_format": "wav",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        request = Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "audio/wav",
            },
        )
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                status = int(getattr(response, "status", 200))
                content_type = _content_type(response)
                wav = response.read(MAX_WAV_BYTES + 1)
        except HTTPError as exc:
            raise SpeechSynthesisError(
                f"GPT-SoVITS returned HTTP {exc.code}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise SpeechSynthesisError(
                f"GPT-SoVITS request failed: {type(exc).__name__}"
            ) from exc

        if status < 200 or status >= 300:
            raise SpeechSynthesisError(f"GPT-SoVITS returned HTTP {status}")
        if content_type not in {"audio/wav", "audio/x-wav", "application/octet-stream"}:
            raise SpeechSynthesisError(
                f"GPT-SoVITS returned unexpected content type {content_type!r}"
            )
        if not wav:
            raise SpeechSynthesisError("GPT-SoVITS returned an empty WAV")
        if len(wav) > MAX_WAV_BYTES:
            raise SpeechSynthesisError("GPT-SoVITS WAV exceeds the size limit")
        return wav


class SpeechExecutor:
    """Synthesizes one say action and delegates fenced playback to Windows."""

    def __init__(
        self,
        *,
        synthesizer: SpeechSynthesizer,
        output: SpeechOutput,
    ) -> None:
        self._synthesizer = synthesizer
        self._output = output

    def run(
        self,
        command: ExecutionCommand,
        cancel_event: threading.Event,
        mark_running: Callable[[], None],
    ) -> None:
        if not isinstance(command.action, SayAction):
            raise TypeError("SpeechExecutor requires a say action")
        if command.resource != ControlResource.VOICE_OUTPUT:
            raise ValueError("say action must own VOICE_OUTPUT")

        wav = self._synthesizer.synthesize(command.action.text)
        # Synthesis itself is not cancellable. Fencing happens before the
        # cancel event is raised, so a late synthesis result must be discarded.
        if cancel_event.is_set():
            return
        mark_running()
        self._output.play_wav(command, wav, cancel_event)


def _content_type(response: object) -> str:
    headers = getattr(response, "headers", {})
    get_content_type = getattr(headers, "get_content_type", None)
    if callable(get_content_type):
        return str(get_content_type()).lower()
    get = getattr(headers, "get", None)
    if callable(get):
        return str(get("Content-Type", "")).split(";", 1)[0].strip().lower()
    return ""


def _validate_endpoint(value: str) -> str:
    endpoint = _nonempty(value, name="endpoint")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("endpoint must be an HTTP or HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("endpoint must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("endpoint must not contain a query or fragment")
    if not parsed.path.endswith("/v1/audio/speech"):
        raise ValueError("endpoint must target /v1/audio/speech")
    return endpoint


def _nonempty(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value.strip()


def _positive_finite(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result
