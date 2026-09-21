from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import threading
import time
import uuid

from .stream_protocol import (
    AudioChunkMessage,
    HeartbeatMessage,
    ScreenshotRequestMessage,
    ScreenshotResponseMessage,
    StreamMessage,
)


class InputTransportError(RuntimeError):
    """Raised when a live Windows sensory request cannot be completed."""


@dataclass(frozen=True, slots=True)
class MacInputSnapshot:
    connected: bool
    session_id: str | None
    last_audio_sequence: int | None
    audio_chunks_received: int
    missing_audio_chunks: int
    screenshots_received: int
    stale_messages_received: int
    remote_heartbeat_sequence: int | None
    heartbeat_monotonic: float
    last_error: str | None


@dataclass(slots=True)
class _PendingScreenshot:
    event: threading.Event
    response: ScreenshotResponseMessage | None = None
    error: str | None = None


class MacInputTransport:
    """Receives ordered audio and requests one current screenshot per turn."""

    def __init__(
        self,
        *,
        audio_handler: Callable[[AudioChunkMessage], None],
        screenshot_timeout_seconds: float = 5.0,
        request_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if not callable(audio_handler):
            raise TypeError("audio_handler must be callable")
        if screenshot_timeout_seconds <= 0:
            raise ValueError("screenshot_timeout_seconds must be positive")
        self._audio_handler = audio_handler
        self._screenshot_timeout_seconds = float(screenshot_timeout_seconds)
        self._request_id_factory = request_id_factory or (lambda: uuid.uuid4().hex)
        self._lock = threading.RLock()
        self._session_id: str | None = None
        self._sender: Callable[[StreamMessage], None] | None = None
        self._pending: dict[str, _PendingScreenshot] = {}
        self._last_audio_sequence: int | None = None
        self._audio_chunks_received = 0
        self._missing_audio_chunks = 0
        self._screenshots_received = 0
        self._stale_messages_received = 0
        self._remote_heartbeat_sequence: int | None = None
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
                raise RuntimeError("an input session is already open")
            self._session_id = session_id
            self._sender = sender
            self._last_audio_sequence = None
            self._remote_heartbeat_sequence = None
            self._last_error = None
            self._touch_locked()

    def close_session(self, session_id: str, *, reason: str) -> bool:
        reason = reason.strip()
        if not reason:
            raise ValueError("reason must not be empty")
        with self._lock:
            if session_id != self._session_id:
                return False
            self._session_id = None
            self._sender = None
            self._last_audio_sequence = None
            self._remote_heartbeat_sequence = None
            self._last_error = reason
            for pending in self._pending.values():
                pending.error = reason
                pending.event.set()
            self._touch_locked()
        return True

    def request_screenshot(self) -> bytes:
        request_id = self._request_id_factory()
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id_factory must return a non-empty string")
        with self._lock:
            session_id = self._session_id
            sender = self._sender
            if session_id is None or sender is None:
                raise InputTransportError("Windows input bridge is disconnected")
            if request_id in self._pending:
                raise RuntimeError("screenshot request id is already pending")
            pending = _PendingScreenshot(event=threading.Event())
            self._pending[request_id] = pending

        try:
            try:
                sender(
                    ScreenshotRequestMessage(
                        request_id=request_id,
                        session_id=session_id,
                    )
                )
            except Exception as exc:
                error = f"screenshot request send failed: {type(exc).__name__}: {exc}"
                self._record_error(error)
                raise InputTransportError(error) from exc
            if not pending.event.wait(self._screenshot_timeout_seconds):
                error = f"screenshot request timed out: {request_id}"
                self._record_error(error)
                raise InputTransportError(error)
            if pending.error is not None:
                raise InputTransportError(pending.error)
            response = pending.response
            if response is None or response.jpeg is None:
                raise InputTransportError("screenshot response was not recorded")
            return response.jpeg
        finally:
            with self._lock:
                if self._pending.get(request_id) is pending:
                    del self._pending[request_id]
                self._touch_locked()

    def receive(self, message: StreamMessage) -> bool:
        if isinstance(message, AudioChunkMessage):
            return self._receive_audio(message)
        if isinstance(message, ScreenshotResponseMessage):
            return self._receive_screenshot(message)
        if isinstance(message, HeartbeatMessage):
            return self._receive_heartbeat(message)
        with self._lock:
            self._stale_messages_received += 1
            self._touch_locked()
        return False

    def snapshot(self) -> MacInputSnapshot:
        with self._lock:
            return MacInputSnapshot(
                connected=self._session_id is not None,
                session_id=self._session_id,
                last_audio_sequence=self._last_audio_sequence,
                audio_chunks_received=self._audio_chunks_received,
                missing_audio_chunks=self._missing_audio_chunks,
                screenshots_received=self._screenshots_received,
                stale_messages_received=self._stale_messages_received,
                remote_heartbeat_sequence=self._remote_heartbeat_sequence,
                heartbeat_monotonic=self._heartbeat_monotonic,
                last_error=self._last_error,
            )

    def _receive_audio(self, message: AudioChunkMessage) -> bool:
        with self._lock:
            if message.session_id != self._session_id:
                self._stale_messages_received += 1
                self._touch_locked()
                return False
            previous = self._last_audio_sequence
            if previous is not None and message.sequence <= previous:
                self._stale_messages_received += 1
                self._touch_locked()
                return False
            if previous is not None and message.sequence > previous + 1:
                self._missing_audio_chunks += message.sequence - previous - 1
            self._last_audio_sequence = message.sequence
            self._audio_chunks_received += 1
            self._touch_locked()
        try:
            self._audio_handler(message)
        except Exception as exc:
            self._record_error(
                f"audio handler failed: {type(exc).__name__}: {exc}"
            )
            return False
        return True

    def _receive_screenshot(self, message: ScreenshotResponseMessage) -> bool:
        with self._lock:
            if message.session_id != self._session_id:
                self._stale_messages_received += 1
                self._touch_locked()
                return False
            pending = self._pending.get(message.request_id)
            if pending is None:
                self._stale_messages_received += 1
                self._touch_locked()
                return False
            pending.response = message
            pending.error = message.error
            pending.event.set()
            self._screenshots_received += 1
            self._touch_locked()
            return True

    def _receive_heartbeat(self, message: HeartbeatMessage) -> bool:
        with self._lock:
            if message.session_id != self._session_id or message.sender != "windows":
                self._stale_messages_received += 1
                self._touch_locked()
                return False
            previous = self._remote_heartbeat_sequence
            if previous is not None and message.sequence <= previous:
                self._stale_messages_received += 1
                self._touch_locked()
                return False
            self._remote_heartbeat_sequence = message.sequence
            self._touch_locked()
            return True

    def _record_error(self, error: str) -> None:
        with self._lock:
            self._last_error = error
            self._touch_locked()

    def _touch_locked(self) -> None:
        self._heartbeat_monotonic = time.monotonic()

