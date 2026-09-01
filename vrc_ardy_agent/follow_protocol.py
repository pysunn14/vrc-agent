from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Any


PROTOCOL_VERSION = 2
MAX_PACKET_BYTES = 4096


class FollowProtocolError(ValueError):
    pass


class TargetSource(str, Enum):
    BODY = "body"
    NAMEPLATE = "nameplate"
    FUSED = "fused"


def _require_plain_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FollowProtocolError(f"{field} must be an integer")
    return value


def _require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FollowProtocolError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FollowProtocolError(f"{field} must be finite")
    return result


@dataclass(frozen=True)
class TargetObservation:
    """One complete, authoritative target state produced from one video frame."""

    session_id: str
    sequence: int
    captured_at_ns: int
    visible: bool
    source: TargetSource | None
    center_x: float | None
    proximity: float | None
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise FollowProtocolError("session_id must be a non-empty string")
        if len(self.session_id) > 128:
            raise FollowProtocolError("session_id must not exceed 128 characters")
        _require_plain_int(self.sequence, "sequence")
        _require_plain_int(self.captured_at_ns, "captured_at_ns")
        if self.sequence < 0:
            raise FollowProtocolError("sequence must be non-negative")
        if self.captured_at_ns < 0:
            raise FollowProtocolError("captured_at_ns must be non-negative")
        if not isinstance(self.visible, bool):
            raise FollowProtocolError("visible must be a boolean")

        confidence = _require_number(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise FollowProtocolError("confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", confidence)

        if not self.visible:
            if self.source is not None or self.center_x is not None or self.proximity is not None:
                raise FollowProtocolError("an invisible observation must not contain target geometry")
            if confidence != 0.0:
                raise FollowProtocolError("an invisible observation must have zero confidence")
            return

        try:
            source = self.source if isinstance(self.source, TargetSource) else TargetSource(self.source)
        except (TypeError, ValueError) as exc:
            raise FollowProtocolError("a visible observation requires a valid source") from exc
        center_x = _require_number(self.center_x, "center_x")
        proximity = _require_number(self.proximity, "proximity")
        if not 0.0 <= center_x <= 1.0:
            raise FollowProtocolError("center_x must be in [0, 1]")
        if not 0.0 <= proximity <= 1.0:
            raise FollowProtocolError("proximity must be in [0, 1]")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "center_x", center_x)
        object.__setattr__(self, "proximity", proximity)


def encode_target_observation(observation: TargetObservation) -> bytes:
    payload = {
        "version": PROTOCOL_VERSION,
        "session": observation.session_id,
        "seq": observation.sequence,
        "captured_at_ns": observation.captured_at_ns,
        "visible": observation.visible,
        "source": observation.source.value if observation.source is not None else None,
        "center_x": observation.center_x,
        "proximity": observation.proximity,
        "confidence": observation.confidence,
    }
    packet = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    if len(packet) > MAX_PACKET_BYTES:
        raise FollowProtocolError("encoded packet exceeds the size limit")
    return packet


def decode_target_observation(packet: bytes) -> TargetObservation:
    if not isinstance(packet, bytes):
        raise FollowProtocolError("packet must be bytes")
    if not packet or len(packet) > MAX_PACKET_BYTES:
        raise FollowProtocolError("packet size is invalid")
    try:
        payload = json.loads(packet.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FollowProtocolError("packet is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise FollowProtocolError("packet root must be an object")

    expected_keys = {
        "version",
        "session",
        "seq",
        "captured_at_ns",
        "visible",
        "source",
        "center_x",
        "proximity",
        "confidence",
    }
    if set(payload) != expected_keys:
        raise FollowProtocolError("packet fields do not match protocol version 2")
    if _require_plain_int(payload["version"], "version") != PROTOCOL_VERSION:
        raise FollowProtocolError(f"unsupported protocol version: {payload['version']!r}")

    try:
        return TargetObservation(
            session_id=payload["session"],
            sequence=_require_plain_int(payload["seq"], "seq"),
            captured_at_ns=_require_plain_int(payload["captured_at_ns"], "captured_at_ns"),
            visible=payload["visible"],
            source=payload["source"],
            center_x=payload["center_x"],
            proximity=payload["proximity"],
            confidence=_require_number(payload["confidence"], "confidence"),
        )
    except (TypeError, ValueError, FollowProtocolError) as exc:
        if isinstance(exc, FollowProtocolError):
            raise
        raise FollowProtocolError("packet contains invalid field types") from exc
