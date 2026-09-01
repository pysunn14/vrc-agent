from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any


PROTOCOL_VERSION = 1
MAX_PACKET_BYTES = 4096


class FollowProtocolError(ValueError):
    pass


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
    bbox: tuple[float, float, float, float] | None
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
            if self.bbox is not None:
                raise FollowProtocolError("an invisible observation must not contain a bbox")
            if confidence != 0.0:
                raise FollowProtocolError("an invisible observation must have zero confidence")
            return

        if self.bbox is None or len(self.bbox) != 4:
            raise FollowProtocolError("a visible observation requires a four-value bbox")
        x1, y1, x2, y2 = (
            _require_number(value, f"bbox[{index}]")
            for index, value in enumerate(self.bbox)
        )
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise FollowProtocolError("bbox must be ordered and normalized to [0, 1]")
        object.__setattr__(self, "bbox", (x1, y1, x2, y2))

    @property
    def center_x(self) -> float | None:
        if self.bbox is None:
            return None
        return (self.bbox[0] + self.bbox[2]) * 0.5

    @property
    def height(self) -> float | None:
        if self.bbox is None:
            return None
        return self.bbox[3] - self.bbox[1]


def encode_target_observation(observation: TargetObservation) -> bytes:
    payload = {
        "version": PROTOCOL_VERSION,
        "session": observation.session_id,
        "seq": observation.sequence,
        "captured_at_ns": observation.captured_at_ns,
        "visible": observation.visible,
        "bbox": list(observation.bbox) if observation.bbox is not None else None,
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
        "bbox",
        "confidence",
    }
    if set(payload) != expected_keys:
        raise FollowProtocolError("packet fields do not match protocol version 1")
    if _require_plain_int(payload["version"], "version") != PROTOCOL_VERSION:
        raise FollowProtocolError(f"unsupported protocol version: {payload['version']!r}")

    raw_bbox = payload["bbox"]
    bbox = None
    if raw_bbox is not None:
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            raise FollowProtocolError("bbox must be null or a four-value array")
        bbox = tuple(raw_bbox)

    try:
        return TargetObservation(
            session_id=payload["session"],
            sequence=_require_plain_int(payload["seq"], "seq"),
            captured_at_ns=_require_plain_int(payload["captured_at_ns"], "captured_at_ns"),
            visible=payload["visible"],
            bbox=bbox,  # type: ignore[arg-type]
            confidence=_require_number(payload["confidence"], "confidence"),
        )
    except (TypeError, FollowProtocolError) as exc:
        if isinstance(exc, FollowProtocolError):
            raise
        raise FollowProtocolError("packet contains invalid field types") from exc
