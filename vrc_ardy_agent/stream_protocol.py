from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from dataclasses import dataclass
import json
from typing import Any, TypeAlias

from .action_contracts import (
    ControlResource,
    OutputEnvelope,
    ResourceLease,
)


# Version 2 carries mapped face parameters. Reject mismatched peers at hello,
# before either endpoint authorizes pose output.
PROTOCOL_VERSION = 2
MAX_MESSAGE_BYTES = 16 * 1024 * 1024
MAX_AUDIO_CHUNK_BYTES = 128 * 1024
MAX_SCREENSHOT_BYTES = 8 * 1024 * 1024


class ProtocolError(ValueError):
    """Raised when a bridge message doesn't match the current wire contract."""


@dataclass(frozen=True, slots=True)
class HelloMessage:
    client_instance_id: str


@dataclass(frozen=True, slots=True)
class SessionOpenMessage:
    session_id: str


@dataclass(frozen=True, slots=True)
class AuthorizeMessage:
    message_id: str
    session_id: str
    lease: ResourceLease


@dataclass(frozen=True, slots=True)
class OutputNeutralizeMessage:
    message_id: str
    session_id: str
    resource: ControlResource
    lease_token: int


@dataclass(frozen=True, slots=True)
class OutputDataMessage:
    envelope: OutputEnvelope


@dataclass(frozen=True, slots=True)
class AckMessage:
    message_id: str
    accepted: bool
    reason: str | None


@dataclass(frozen=True, slots=True)
class OutputCompletedMessage:
    session_id: str
    action_id: str
    resource: ControlResource
    lease_token: int
    state: str
    error: str | None


@dataclass(frozen=True, slots=True)
class AudioChunkMessage:
    session_id: str
    sequence: int
    captured_monotonic_ns: int
    pcm: bytes


@dataclass(frozen=True, slots=True)
class ScreenshotRequestMessage:
    request_id: str
    session_id: str


@dataclass(frozen=True, slots=True)
class ScreenshotResponseMessage:
    request_id: str
    session_id: str
    jpeg: bytes | None
    error: str | None


@dataclass(frozen=True, slots=True)
class HeartbeatMessage:
    session_id: str
    sender: str
    sequence: int


StreamMessage: TypeAlias = (
    HelloMessage
    | SessionOpenMessage
    | AuthorizeMessage
    | OutputNeutralizeMessage
    | OutputDataMessage
    | AckMessage
    | OutputCompletedMessage
    | AudioChunkMessage
    | ScreenshotRequestMessage
    | ScreenshotResponseMessage
    | HeartbeatMessage
)


def encode_message(message: StreamMessage) -> str:
    if isinstance(message, HelloMessage):
        payload = {
            "type": "hello",
            "version": PROTOCOL_VERSION,
            "client_instance_id": _identifier(
                message.client_instance_id,
                "client_instance_id",
            ),
        }
    elif isinstance(message, SessionOpenMessage):
        payload = {
            "type": "session.open",
            "version": PROTOCOL_VERSION,
            "session_id": _identifier(message.session_id, "session_id"),
        }
    elif isinstance(message, AuthorizeMessage):
        _positive_integer(message.lease.token, "lease_token")
        action_id = _identifier(message.lease.action_id, "action_id")
        payload = {
            "type": "output.authorize",
            "version": PROTOCOL_VERSION,
            "message_id": _identifier(message.message_id, "message_id"),
            "session_id": _identifier(message.session_id, "session_id"),
            "resource": message.lease.resource.value,
            "lease_token": message.lease.token,
            "action_id": action_id,
        }
    elif isinstance(message, OutputNeutralizeMessage):
        _nonnegative_integer(message.lease_token, "lease_token")
        payload = {
            "type": "output.neutralize",
            "version": PROTOCOL_VERSION,
            "message_id": _identifier(message.message_id, "message_id"),
            "session_id": _identifier(message.session_id, "session_id"),
            "resource": message.resource.value,
            "lease_token": message.lease_token,
        }
    elif isinstance(message, OutputDataMessage):
        envelope = message.envelope
        _positive_integer(envelope.lease_token, "lease_token")
        _nonnegative_integer(envelope.sequence, "sequence")
        _positive_integer(envelope.ttl_ms, "ttl_ms")
        if not isinstance(envelope.payload, Mapping):
            raise ProtocolError("payload must be an object")
        payload = {
            "type": "output.data",
            "version": PROTOCOL_VERSION,
            "session_id": _identifier(envelope.session_id, "session_id"),
            "action_id": _identifier(envelope.action_id, "action_id"),
            "resource": envelope.resource.value,
            "lease_token": envelope.lease_token,
            "sequence": envelope.sequence,
            "ttl_ms": envelope.ttl_ms,
            "payload": dict(envelope.payload),
        }
    elif isinstance(message, AckMessage):
        if not isinstance(message.accepted, bool):
            raise ProtocolError("accepted must be a boolean")
        payload = {
            "type": "ack",
            "version": PROTOCOL_VERSION,
            "message_id": _identifier(message.message_id, "message_id"),
            "accepted": message.accepted,
            "reason": _optional_text(message.reason, "reason"),
        }
    elif isinstance(message, OutputCompletedMessage):
        state = _completion_state(message.state)
        error = _optional_text(message.error, "error")
        if state == "failed" and error is None:
            raise ProtocolError("failed completion requires an error")
        _positive_integer(message.lease_token, "lease_token")
        payload = {
            "type": "output.completed",
            "version": PROTOCOL_VERSION,
            "session_id": _identifier(message.session_id, "session_id"),
            "action_id": _identifier(message.action_id, "action_id"),
            "resource": message.resource.value,
            "lease_token": message.lease_token,
            "state": state,
            "error": error,
        }
    elif isinstance(message, AudioChunkMessage):
        pcm = _audio_pcm(message.pcm)
        payload = {
            "type": "audio.chunk",
            "version": PROTOCOL_VERSION,
            "session_id": _identifier(message.session_id, "session_id"),
            "sequence": _nonnegative_integer(message.sequence, "sequence"),
            "captured_monotonic_ns": _nonnegative_integer(
                message.captured_monotonic_ns,
                "captured_monotonic_ns",
            ),
            "sample_rate": 16000,
            "channels": 1,
            "encoding": "pcm_s16le",
            "pcm": base64.b64encode(pcm).decode("ascii"),
        }
    elif isinstance(message, ScreenshotRequestMessage):
        payload = {
            "type": "screenshot.request",
            "version": PROTOCOL_VERSION,
            "request_id": _identifier(message.request_id, "request_id"),
            "session_id": _identifier(message.session_id, "session_id"),
        }
    elif isinstance(message, ScreenshotResponseMessage):
        error = _optional_text(message.error, "error")
        jpeg = message.jpeg
        if (jpeg is None) == (error is None):
            raise ProtocolError(
                "screenshot response requires exactly one of jpeg or error"
            )
        encoded_jpeg = None
        if jpeg is not None:
            encoded_jpeg = base64.b64encode(_jpeg(jpeg)).decode("ascii")
        payload = {
            "type": "screenshot.response",
            "version": PROTOCOL_VERSION,
            "request_id": _identifier(message.request_id, "request_id"),
            "session_id": _identifier(message.session_id, "session_id"),
            "jpeg": encoded_jpeg,
            "error": error,
        }
    elif isinstance(message, HeartbeatMessage):
        sender = message.sender
        if sender not in {"mac", "windows"}:
            raise ProtocolError("heartbeat sender is invalid")
        payload = {
            "type": "heartbeat",
            "version": PROTOCOL_VERSION,
            "session_id": _identifier(message.session_id, "session_id"),
            "sender": sender,
            "sequence": _nonnegative_integer(message.sequence, "sequence"),
        }
    else:
        raise TypeError(f"unsupported message class: {type(message).__name__}")

    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError("message is not JSON serializable") from exc
    if len(encoded.encode("utf-8")) > MAX_MESSAGE_BYTES:
        raise ProtocolError("message exceeds the size limit")
    return encoded


def decode_message(raw: str | bytes) -> StreamMessage:
    if isinstance(raw, bytes):
        if len(raw) > MAX_MESSAGE_BYTES:
            raise ProtocolError("message exceeds the size limit")
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError("message must be UTF-8 JSON") from exc
    elif not isinstance(raw, str):
        raise ProtocolError("message must be text or bytes")
    elif len(raw.encode("utf-8")) > MAX_MESSAGE_BYTES:
        raise ProtocolError("message exceeds the size limit")

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError("message must be valid JSON") from exc
    root = _mapping(value, "message")
    message_type = root.get("type")
    version = _plain_integer(root.get("version"), "version")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version: {version}")

    if message_type == "hello":
        _exact(root, {"type", "version", "client_instance_id"})
        return HelloMessage(
            client_instance_id=_identifier(
                root["client_instance_id"],
                "client_instance_id",
            )
        )
    if message_type == "session.open":
        _exact(root, {"type", "version", "session_id"})
        return SessionOpenMessage(
            session_id=_identifier(root["session_id"], "session_id")
        )
    if message_type == "output.authorize":
        _exact(
            root,
            {
                "type",
                "version",
                "message_id",
                "session_id",
                "resource",
                "lease_token",
                "action_id",
            },
        )
        return AuthorizeMessage(
            message_id=_identifier(root["message_id"], "message_id"),
            session_id=_identifier(root["session_id"], "session_id"),
            lease=ResourceLease(
                resource=_resource(root["resource"]),
                token=_positive_integer(root["lease_token"], "lease_token"),
                action_id=_identifier(root["action_id"], "action_id"),
            ),
        )
    if message_type == "output.neutralize":
        _exact(
            root,
            {
                "type",
                "version",
                "message_id",
                "session_id",
                "resource",
                "lease_token",
            },
        )
        return OutputNeutralizeMessage(
            message_id=_identifier(root["message_id"], "message_id"),
            session_id=_identifier(root["session_id"], "session_id"),
            resource=_resource(root["resource"]),
            lease_token=_nonnegative_integer(root["lease_token"], "lease_token"),
        )
    if message_type == "output.data":
        _exact(
            root,
            {
                "type",
                "version",
                "session_id",
                "action_id",
                "resource",
                "lease_token",
                "sequence",
                "ttl_ms",
                "payload",
            },
        )
        return OutputDataMessage(
            envelope=OutputEnvelope(
                session_id=_identifier(root["session_id"], "session_id"),
                action_id=_identifier(root["action_id"], "action_id"),
                resource=_resource(root["resource"]),
                lease_token=_positive_integer(root["lease_token"], "lease_token"),
                sequence=_nonnegative_integer(root["sequence"], "sequence"),
                ttl_ms=_positive_integer(root["ttl_ms"], "ttl_ms"),
                payload=dict(_mapping(root["payload"], "payload")),
            )
        )
    if message_type == "ack":
        _exact(root, {"type", "version", "message_id", "accepted", "reason"})
        accepted = root["accepted"]
        if not isinstance(accepted, bool):
            raise ProtocolError("accepted must be a boolean")
        return AckMessage(
            message_id=_identifier(root["message_id"], "message_id"),
            accepted=accepted,
            reason=_optional_text(root["reason"], "reason"),
        )
    if message_type == "output.completed":
        _exact(
            root,
            {
                "type",
                "version",
                "session_id",
                "action_id",
                "resource",
                "lease_token",
                "state",
                "error",
            },
        )
        state = _completion_state(root["state"])
        error = _optional_text(root["error"], "error")
        if state == "failed" and error is None:
            raise ProtocolError("failed completion requires an error")
        return OutputCompletedMessage(
            session_id=_identifier(root["session_id"], "session_id"),
            action_id=_identifier(root["action_id"], "action_id"),
            resource=_resource(root["resource"]),
            lease_token=_positive_integer(root["lease_token"], "lease_token"),
            state=state,
            error=error,
        )
    if message_type == "audio.chunk":
        _exact(
            root,
            {
                "type",
                "version",
                "session_id",
                "sequence",
                "captured_monotonic_ns",
                "sample_rate",
                "channels",
                "encoding",
                "pcm",
            },
        )
        if root["sample_rate"] != 16000:
            raise ProtocolError("audio sample_rate must be 16000")
        if root["channels"] != 1:
            raise ProtocolError("audio channels must be 1")
        if root["encoding"] != "pcm_s16le":
            raise ProtocolError("audio encoding must be pcm_s16le")
        return AudioChunkMessage(
            session_id=_identifier(root["session_id"], "session_id"),
            sequence=_nonnegative_integer(root["sequence"], "sequence"),
            captured_monotonic_ns=_nonnegative_integer(
                root["captured_monotonic_ns"],
                "captured_monotonic_ns",
            ),
            pcm=_audio_pcm(_decode_base64(root["pcm"], "pcm")),
        )
    if message_type == "screenshot.request":
        _exact(root, {"type", "version", "request_id", "session_id"})
        return ScreenshotRequestMessage(
            request_id=_identifier(root["request_id"], "request_id"),
            session_id=_identifier(root["session_id"], "session_id"),
        )
    if message_type == "screenshot.response":
        _exact(
            root,
            {
                "type",
                "version",
                "request_id",
                "session_id",
                "jpeg",
                "error",
            },
        )
        encoded_jpeg = root["jpeg"]
        error = _optional_text(root["error"], "error")
        if (encoded_jpeg is None) == (error is None):
            raise ProtocolError(
                "screenshot response requires exactly one of jpeg or error"
            )
        jpeg = None
        if encoded_jpeg is not None:
            jpeg = _jpeg(_decode_base64(encoded_jpeg, "jpeg"))
        return ScreenshotResponseMessage(
            request_id=_identifier(root["request_id"], "request_id"),
            session_id=_identifier(root["session_id"], "session_id"),
            jpeg=jpeg,
            error=error,
        )
    if message_type == "heartbeat":
        _exact(root, {"type", "version", "session_id", "sender", "sequence"})
        sender = root["sender"]
        if sender not in {"mac", "windows"}:
            raise ProtocolError("heartbeat sender is invalid")
        return HeartbeatMessage(
            session_id=_identifier(root["session_id"], "session_id"),
            sender=str(sender),
            sequence=_nonnegative_integer(root["sequence"], "sequence"),
        )
    raise ProtocolError(f"unsupported message type: {message_type!r}")


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise ProtocolError(f"{location} must be an object")
    return value


def _exact(value: Mapping[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        raise ProtocolError("message fields do not match its type")


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{name} must be a non-empty string")
    result = value.strip()
    if len(result) > 128 or any(character in result for character in "\r\n\0"):
        raise ProtocolError(f"{name} is invalid")
    return result


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{name} must be null or a non-empty string")
    result = value.strip()
    if len(result) > 1000:
        raise ProtocolError(f"{name} is too long")
    return result


def _plain_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{name} must be an integer")
    return value


def _positive_integer(value: object, name: str) -> int:
    result = _plain_integer(value, name)
    if result <= 0:
        raise ProtocolError(f"{name} must be positive")
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    result = _plain_integer(value, name)
    if result < 0:
        raise ProtocolError(f"{name} must be non-negative")
    return result


def _resource(value: object) -> ControlResource:
    try:
        return ControlResource(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("resource is invalid") from exc


def _completion_state(value: object) -> str:
    if value not in {"completed", "cancelled", "failed"}:
        raise ProtocolError("completion state is invalid")
    return str(value)


def _decode_base64(value: object, name: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{name} must be non-empty base64")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ProtocolError(f"{name} must be valid base64") from exc


def _audio_pcm(value: object) -> bytes:
    if not isinstance(value, bytes) or not value:
        raise ProtocolError("pcm must not be empty")
    if len(value) > MAX_AUDIO_CHUNK_BYTES:
        raise ProtocolError("pcm exceeds the audio chunk size limit")
    if len(value) % 2:
        raise ProtocolError("pcm_s16le must contain complete samples")
    return value


def _jpeg(value: object) -> bytes:
    if not isinstance(value, bytes) or not value:
        raise ProtocolError("jpeg must not be empty")
    if len(value) > MAX_SCREENSHOT_BYTES:
        raise ProtocolError("jpeg exceeds the screenshot size limit")
    if not (value.startswith(b"\xff\xd8") and value.endswith(b"\xff\xd9")):
        raise ProtocolError("jpeg payload is invalid")
    return value
