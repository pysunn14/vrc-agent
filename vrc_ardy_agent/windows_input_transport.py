from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import threading
import time

from .stream_protocol import (
    AudioChunkMessage,
    HeartbeatMessage,
    MAX_AUDIO_CHUNK_BYTES,
    ScreenshotRequestMessage,
    ScreenshotResponseMessage,
    StreamMessage,
)


class WindowsInputTransportError(RuntimeError):
    """Raised when current sensory data cannot be sent to the Mac."""


@dataclass(frozen=True, slots=True)
class WindowsInputSnapshot:
    connected: bool
    session_id: str | None
    next_audio_sequence: int
    audio_chunks_sent: int
    audio_chunks_dropped: int
    screenshots_sent: int
    heartbeat_sequence: int
    heartbeat_monotonic: float
    last_error: str | None


class WindowsInputTransport:
    """Publishes only live sensory data for the current bridge session."""

    def __init__(
        self,
        *,
        screenshot_provider: Callable[[], bytes],
    ) -> None:
        if not callable(screenshot_provider):
            raise TypeError("screenshot_provider must be callable")
        self._screenshot_provider = screenshot_provider
        self._lock = threading.RLock()
        self._session_id: str | None = None
        self._sender: Callable[[StreamMessage], None] | None = None
        self._next_audio_sequence = 0
        self._audio_chunks_sent = 0
        self._audio_chunks_dropped = 0
        self._screenshots_sent = 0
        self._heartbeat_sequence = 0
        self._heartbeat_monotonic = time.monotonic()
        self._last_error: str | None = None

    def open_session(
        self,
        session_id: str,
        sender: Callable[[StreamMessage], None],
    ) -> None:
        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id must not be empty")
        if not callable(sender):
            raise TypeError("sender must be callable")
        with self._lock:
            if self._session_id is not None:
                raise RuntimeError("a Windows input session is already open")
            self._session_id = session_id
            self._sender = sender
            self._next_audio_sequence = 0
            self._heartbeat_sequence = 0
            self._last_error = None
            self._touch_locked()

    def close_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id != self._session_id:
                return False
            self._session_id = None
            self._sender = None
            self._next_audio_sequence = 0
            self._heartbeat_sequence = 0
            self._touch_locked()
            return True

    def publish_audio(
        self,
        pcm: bytes,
        *,
        captured_monotonic_ns: int,
    ) -> bool:
        if not isinstance(pcm, bytes) or not pcm:
            raise ValueError("pcm must not be empty")
        if len(pcm) > MAX_AUDIO_CHUNK_BYTES or len(pcm) % 2:
            raise ValueError("pcm must contain complete samples within the size limit")
        if captured_monotonic_ns < 0:
            raise ValueError("captured_monotonic_ns must be non-negative")
        with self._lock:
            session_id = self._session_id
            sender = self._sender
            if session_id is None or sender is None:
                self._audio_chunks_dropped += 1
                self._touch_locked()
                return False
            sequence = self._next_audio_sequence
            self._next_audio_sequence += 1
        try:
            sender(
                AudioChunkMessage(
                    session_id=session_id,
                    sequence=sequence,
                    captured_monotonic_ns=captured_monotonic_ns,
                    pcm=pcm,
                )
            )
        except Exception as exc:
            with self._lock:
                self._audio_chunks_dropped += 1
            self._record_error(
                f"audio send failed: {type(exc).__name__}: {exc}"
            )
            return False
        with self._lock:
            self._audio_chunks_sent += 1
            self._touch_locked()
        return True

    def handle_screenshot_request(
        self,
        request: ScreenshotRequestMessage,
    ) -> None:
        with self._lock:
            session_id = self._session_id
            sender = self._sender
        if request.session_id != session_id or sender is None:
            raise WindowsInputTransportError(
                "screenshot request does not belong to the current session"
            )
        try:
            jpeg = self._screenshot_provider()
            if not isinstance(jpeg, bytes) or not jpeg:
                raise RuntimeError("screenshot provider returned no JPEG data")
            response = ScreenshotResponseMessage(
                request_id=request.request_id,
                session_id=request.session_id,
                jpeg=jpeg,
                error=None,
            )
        except Exception as exc:
            error = f"screenshot capture failed: {type(exc).__name__}: {exc}"
            self._record_error(error)
            response = ScreenshotResponseMessage(
                request_id=request.request_id,
                session_id=request.session_id,
                jpeg=None,
                error=error[:1000],
            )
        try:
            sender(response)
        except Exception as exc:
            error = f"screenshot response send failed: {type(exc).__name__}: {exc}"
            self._record_error(error)
            raise WindowsInputTransportError(error) from exc
        with self._lock:
            self._screenshots_sent += 1
            self._touch_locked()

    def send_heartbeat(self) -> None:
        with self._lock:
            session_id = self._session_id
            sender = self._sender
            if session_id is None or sender is None:
                return
            sequence = self._heartbeat_sequence
            self._heartbeat_sequence += 1
        sender(
            HeartbeatMessage(
                session_id=session_id,
                sender="windows",
                sequence=sequence,
            )
        )
        with self._lock:
            self._touch_locked()

    def snapshot(self) -> WindowsInputSnapshot:
        with self._lock:
            return WindowsInputSnapshot(
                connected=self._session_id is not None,
                session_id=self._session_id,
                next_audio_sequence=self._next_audio_sequence,
                audio_chunks_sent=self._audio_chunks_sent,
                audio_chunks_dropped=self._audio_chunks_dropped,
                screenshots_sent=self._screenshots_sent,
                heartbeat_sequence=self._heartbeat_sequence,
                heartbeat_monotonic=self._heartbeat_monotonic,
                last_error=self._last_error,
            )

    def _record_error(self, error: str) -> None:
        with self._lock:
            self._last_error = error
            self._touch_locked()

    def _touch_locked(self) -> None:
        self._heartbeat_monotonic = time.monotonic()
