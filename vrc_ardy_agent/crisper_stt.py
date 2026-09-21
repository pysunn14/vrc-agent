from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
import json
import math
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import uuid
import wave


DEFAULT_STT_ENDPOINT = "http://127.0.0.1:8002/v1/audio/transcriptions"
DEFAULT_STT_MODEL = "crisperwhisper-2.0-medium"
MAX_PCM_BYTES = 32 * 1024 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024


class TranscriptionError(RuntimeError):
    """Raised when CrisperWhisper cannot produce a final transcript."""


class CrisperWhisperClient:
    """Turn-based client for the resident OpenAI-compatible STT adapter."""

    def __init__(
        self,
        *,
        endpoint: str = DEFAULT_STT_ENDPOINT,
        model: str = DEFAULT_STT_MODEL,
        language: str = "ko",
        mode: str = "intended",
        timeout_seconds: float = 120.0,
        boundary_factory: Callable[[], str] | None = None,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.endpoint = _validate_endpoint(endpoint)
        self.model = _nonempty(model, "model")
        language = _nonempty(language, "language").lower()
        if len(language) != 2 or not language.isascii() or not language.isalpha():
            raise ValueError("language must be a two-letter language code")
        if mode not in {"verbatim", "intended"}:
            raise ValueError("mode must be verbatim or intended")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite")
        self.language = language
        self.mode = mode
        self.timeout_seconds = float(timeout_seconds)
        self._boundary_factory = boundary_factory or (lambda: uuid.uuid4().hex)
        self._opener = opener

    def transcribe_pcm(self, pcm_s16le: bytes) -> str:
        wav = _pcm_to_wav(pcm_s16le)
        boundary = self._boundary_factory()
        if (
            not isinstance(boundary, str)
            or not boundary
            or not boundary.isascii()
            or any(character in boundary for character in "\r\n\"")
        ):
            raise ValueError("boundary_factory returned an invalid boundary")
        body = _multipart_body(
            boundary,
            fields={
                "model": self.model,
                "language": self.language,
                "mode": self.mode,
                "response_format": "json",
            },
            wav=wav,
        )
        request = Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Accept": "application/json",
            },
        )
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                status = int(getattr(response, "status", 200))
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            raise TranscriptionError(
                f"CrisperWhisper returned HTTP {exc.code}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise TranscriptionError(
                f"CrisperWhisper request failed: {type(exc).__name__}"
            ) from exc
        if status < 200 or status >= 300:
            raise TranscriptionError(f"CrisperWhisper returned HTTP {status}")
        if len(raw) > MAX_RESPONSE_BYTES:
            raise TranscriptionError("CrisperWhisper response is too large")
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TranscriptionError(
                "CrisperWhisper response is not valid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise TranscriptionError("CrisperWhisper response must be an object")
        transcript = payload.get("text")
        if not isinstance(transcript, str) or not transcript.strip():
            raise TranscriptionError("CrisperWhisper returned an empty transcript")
        return transcript.strip()


def _pcm_to_wav(pcm: bytes) -> bytes:
    if not isinstance(pcm, bytes) or not pcm:
        raise ValueError("pcm_s16le must not be empty")
    if len(pcm) > MAX_PCM_BYTES:
        raise ValueError("pcm_s16le exceeds the size limit")
    if len(pcm) % 2:
        raise ValueError("pcm_s16le must contain complete samples")
    output = BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(pcm)
    return output.getvalue()


def _multipart_body(
    boundary: str,
    *,
    fields: dict[str, str],
    wav: bytes,
) -> bytes:
    marker = f"--{boundary}".encode("ascii")
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.extend(
            (
                marker,
                f'Content-Disposition: form-data; name="{name}"'.encode("ascii"),
                b"",
                value.encode("utf-8"),
            )
        )
    parts.extend(
        (
            marker,
            b'Content-Disposition: form-data; name="file"; filename="utterance.wav"',
            b"Content-Type: audio/wav",
            b"",
            wav,
            marker + b"--",
            b"",
        )
    )
    return b"\r\n".join(parts)


def _validate_endpoint(value: str) -> str:
    endpoint = _nonempty(value, "endpoint")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("endpoint must be an HTTP or HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("endpoint must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("endpoint must not contain a query or fragment")
    if not parsed.path.endswith("/v1/audio/transcriptions"):
        raise ValueError("endpoint must target /v1/audio/transcriptions")
    return endpoint


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value.strip()
