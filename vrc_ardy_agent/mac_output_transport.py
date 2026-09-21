from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
import threading
import time
from collections.abc import Callable
import uuid
import wave

from .action_contracts import (
    ControlResource,
    ExecutionCommand,
    OutputEnvelope,
    ResourceLease,
)
from .device_payloads import encode_pose_payload, encode_wav_payload
from .stream_protocol import (
    AckMessage,
    AuthorizeMessage,
    OutputCompletedMessage,
    OutputDataMessage,
    OutputNeutralizeMessage,
    StreamMessage,
)


class OutputTransportError(RuntimeError):
    """Raised when Windows cannot authoritatively accept or finish output."""


@dataclass(frozen=True, slots=True)
class MacOutputSnapshot:
    connected: bool
    session_id: str | None
    messages_sent: int
    acknowledgements_received: int
    completions_received: int
    stale_messages_received: int
    heartbeat_monotonic: float
    last_error: str | None


@dataclass(slots=True)
class _PendingAck:
    event: threading.Event
    response: AckMessage | None = None
    error: str | None = None


@dataclass(slots=True)
class _PendingCompletion:
    event: threading.Event
    response: OutputCompletedMessage | None = None
    error: str | None = None


class MacOutputTransport:
    """One-session synchronous facade over the persistent bridge connection."""

    def __init__(
        self,
        *,
        control_timeout_seconds: float = 5.0,
        playback_grace_seconds: float = 5.0,
        message_id_factory: Callable[[], str] | None = None,
        disconnected: Callable[[str], None] | None = None,
    ) -> None:
        if control_timeout_seconds <= 0:
            raise ValueError("control_timeout_seconds must be positive")
        if playback_grace_seconds <= 0:
            raise ValueError("playback_grace_seconds must be positive")
        self._control_timeout_seconds = float(control_timeout_seconds)
        self._playback_grace_seconds = float(playback_grace_seconds)
        self._message_id_factory = message_id_factory or (
            lambda: uuid.uuid4().hex
        )
        self._disconnected = disconnected
        self._lock = threading.RLock()
        self._session_id: str | None = None
        self._sender: Callable[[StreamMessage], None] | None = None
        self._pending_acks: dict[str, _PendingAck] = {}
        self._pending_completions: dict[
            tuple[str, str, ControlResource, int],
            _PendingCompletion,
        ] = {}
        self._messages_sent = 0
        self._acks_received = 0
        self._completions_received = 0
        self._stale_received = 0
        self._last_error: str | None = None
        self._heartbeat_monotonic = time.monotonic()

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
                raise RuntimeError("an output session is already open")
            self._session_id = session_id
            self._sender = sender
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
            self._last_error = reason
            for pending in self._pending_acks.values():
                pending.error = reason
                pending.event.set()
            for pending in self._pending_completions.values():
                pending.error = reason
                pending.event.set()
            self._touch_locked()
        if self._disconnected is not None:
            self._disconnected(reason)
        return True

    def authorize(self, lease: ResourceLease) -> None:
        session_id = self._require_session_id()
        message_id = self._next_message_id()
        self._send_control(
            AuthorizeMessage(
                message_id=message_id,
                session_id=session_id,
                lease=lease,
            )
        )

    def neutralize(
        self,
        resource: ControlResource,
        lease_token: int,
    ) -> None:
        session_id = self._require_session_id()
        message_id = self._next_message_id()
        self._send_control(
            OutputNeutralizeMessage(
                message_id=message_id,
                session_id=session_id,
                resource=resource,
                lease_token=lease_token,
            )
        )

    def send_pose(
        self,
        lease: ResourceLease,
        frame: object,
        *,
        sequence: int,
        ttl_ms: int,
    ) -> None:
        if lease.resource != ControlResource.FULL_BODY_POSE:
            raise ValueError("pose output requires FULL_BODY_POSE")
        if not lease.action_id:
            raise ValueError("pose output requires an owned resource lease")
        session_id = self._require_session_id()
        self._send(
            OutputDataMessage(
                envelope=OutputEnvelope(
                    session_id=session_id,
                    action_id=lease.action_id,
                    resource=lease.resource,
                    lease_token=lease.token,
                    sequence=sequence,
                    ttl_ms=ttl_ms,
                    payload=encode_pose_payload(frame),
                )
            )
        )

    def play_wav(
        self,
        command: ExecutionCommand,
        wav_bytes: bytes,
        cancel_event: threading.Event,
    ) -> None:
        if command.resource != ControlResource.VOICE_OUTPUT:
            raise ValueError("speech output requires VOICE_OUTPUT")
        duration = _wav_duration_seconds(wav_bytes)
        session_id = self._require_session_id()
        key = (
            session_id,
            command.action_id,
            command.resource,
            command.lease_token,
        )
        pending = _PendingCompletion(event=threading.Event())
        with self._lock:
            if key in self._pending_completions:
                raise RuntimeError("speech completion is already pending")
            self._pending_completions[key] = pending

        ttl_ms = max(1, int(math.ceil((duration + self._playback_grace_seconds) * 1000)))
        try:
            self._send(
                OutputDataMessage(
                    envelope=OutputEnvelope(
                        session_id=session_id,
                        action_id=command.action_id,
                        resource=command.resource,
                        lease_token=command.lease_token,
                        sequence=0,
                        ttl_ms=ttl_ms,
                        payload=encode_wav_payload(wav_bytes),
                    )
                )
            )
            deadline = time.monotonic() + duration + self._playback_grace_seconds
            while not pending.event.wait(timeout=0.05):
                if cancel_event.is_set():
                    return
                if time.monotonic() >= deadline:
                    error = "Windows speech playback completion timed out"
                    self._record_error(error)
                    raise OutputTransportError(error)

            if pending.error is not None:
                raise OutputTransportError(pending.error)
            completion = pending.response
            if completion is None:
                raise OutputTransportError("speech completion was not recorded")
            if completion.state == "failed":
                raise OutputTransportError(
                    completion.error or "Windows speech playback failed"
                )
            if completion.state == "cancelled" and not cancel_event.is_set():
                raise OutputTransportError("Windows cancelled speech playback")
        finally:
            with self._lock:
                if self._pending_completions.get(key) is pending:
                    del self._pending_completions[key]
                self._touch_locked()

    def receive(self, message: StreamMessage) -> bool:
        with self._lock:
            if isinstance(message, AckMessage):
                pending = self._pending_acks.get(message.message_id)
                if pending is None:
                    self._stale_received += 1
                    self._touch_locked()
                    return False
                pending.response = message
                pending.event.set()
                self._acks_received += 1
                self._touch_locked()
                return True
            if isinstance(message, OutputCompletedMessage):
                key = (
                    message.session_id,
                    message.action_id,
                    message.resource,
                    message.lease_token,
                )
                pending = self._pending_completions.get(key)
                if pending is None:
                    self._stale_received += 1
                    self._touch_locked()
                    return False
                pending.response = message
                pending.event.set()
                self._completions_received += 1
                self._touch_locked()
                return True
            self._stale_received += 1
            self._touch_locked()
            return False

    def snapshot(self) -> MacOutputSnapshot:
        with self._lock:
            return MacOutputSnapshot(
                connected=self._session_id is not None,
                session_id=self._session_id,
                messages_sent=self._messages_sent,
                acknowledgements_received=self._acks_received,
                completions_received=self._completions_received,
                stale_messages_received=self._stale_received,
                heartbeat_monotonic=self._heartbeat_monotonic,
                last_error=self._last_error,
            )

    def _send_control(
        self,
        message: AuthorizeMessage | OutputNeutralizeMessage,
    ) -> None:
        pending = _PendingAck(event=threading.Event())
        with self._lock:
            self._pending_acks[message.message_id] = pending
        try:
            self._send(message)
            if not pending.event.wait(timeout=self._control_timeout_seconds):
                error = f"Windows acknowledgement timed out for {message.message_id}"
                self._record_error(error)
                raise OutputTransportError(error)
            if pending.error is not None:
                raise OutputTransportError(pending.error)
            response = pending.response
            if response is None:
                raise OutputTransportError("Windows acknowledgement was not recorded")
            if not response.accepted:
                reason = response.reason or "rejected"
                error = f"Windows rejected {message.message_id}: {reason}"
                self._record_error(error)
                raise OutputTransportError(error)
        finally:
            with self._lock:
                if self._pending_acks.get(message.message_id) is pending:
                    del self._pending_acks[message.message_id]
                self._touch_locked()

    def _send(self, message: StreamMessage) -> None:
        with self._lock:
            sender = self._sender
            if sender is None or self._session_id is None:
                raise OutputTransportError("Windows output bridge is disconnected")
        try:
            sender(message)
        except Exception as exc:
            error = f"Windows output send failed: {type(exc).__name__}: {exc}"
            self._record_error(error)
            raise OutputTransportError(error) from exc
        with self._lock:
            self._messages_sent += 1
            self._touch_locked()

    def _require_session_id(self) -> str:
        with self._lock:
            if self._session_id is None:
                raise OutputTransportError("Windows output bridge is disconnected")
            return self._session_id

    def _next_message_id(self) -> str:
        message_id = self._message_id_factory()
        if not isinstance(message_id, str) or not message_id:
            raise ValueError("message_id_factory must return a non-empty string")
        return message_id

    def _record_error(self, error: str) -> None:
        with self._lock:
            self._last_error = error
            self._touch_locked()

    def _touch_locked(self) -> None:
        self._heartbeat_monotonic = time.monotonic()


def _wav_duration_seconds(wav_bytes: bytes) -> float:
    try:
        with wave.open(BytesIO(wav_bytes), "rb") as reader:
            rate = reader.getframerate()
            frames = reader.getnframes()
    except (EOFError, wave.Error) as exc:
        raise OutputTransportError("speech payload is not a valid WAV") from exc
    if rate <= 0 or frames <= 0:
        raise OutputTransportError("speech WAV has no playable frames")
    return frames / rate
