from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping, Sequence
import math
from typing import Any

from .asset_contract import face_values
from .opentrack_bridge import OpenTrackFrame
from .six_point_bridge import SixPointFrame, TrackerActivation
from .vmt_bridge import VmtFrame


MAX_WAV_PAYLOAD_BYTES = 16 * 1024 * 1024
_TRACKER_NAMES = ("left", "right", "hips", "left_foot", "right_foot")


class DevicePayloadError(ValueError):
    """Raised when an output payload cannot safely reach a device sink."""


def encode_pose_payload(frame: SixPointFrame) -> dict[str, object]:
    if not isinstance(frame, SixPointFrame):
        raise TypeError("frame must be a SixPointFrame")

    def tracker(value: VmtFrame) -> dict[str, object]:
        return {
            "position": list(value.position),
            "quaternion_xyzw": list(value.quaternion_xyzw),
        }

    payload: dict[str, object] = {
        "kind": "six_point_pose",
        "head": {
            "xyz_cm": list(frame.head.xyz_cm),
            "ypr_deg": list(frame.head.ypr_deg),
        },
        "trackers": {
            "left": tracker(frame.left),
            "right": tracker(frame.right),
            "hips": tracker(frame.hips),
            "left_foot": tracker(frame.left_foot),
            "right_foot": tracker(frame.right_foot),
        },
        "face": dict(frame.face),
        "fps": frame.fps,
        "scale": frame.scale,
        "locomotion": [
            frame.locomotion_x,
            frame.locomotion_y,
            frame.locomotion_turn,
        ],
        "tracker_activation": {
            name: getattr(frame.tracker_activation, name)
            for name in _TRACKER_NAMES
        },
    }
    # Encoding also validates locally so malformed numeric state never reaches
    # the network boundary.
    decode_pose_payload(payload)
    return payload


def decode_pose_payload(payload: object) -> SixPointFrame:
    root = _object(payload, "pose payload")
    _exact(
        {key: value for key, value in root.items() if key not in ("face", "yawn")},
        {
            "kind",
            "head",
            "trackers",
            "fps",
            "scale",
            "locomotion",
            "tracker_activation",
        },
    )
    if root["kind"] != "six_point_pose":
        raise DevicePayloadError("pose payload kind is invalid")
    fps = _positive(root["fps"], "fps")
    scale = _positive(root["scale"], "scale")

    head = _object(root["head"], "head")
    _exact(head, {"xyz_cm", "ypr_deg"})
    head_frame = OpenTrackFrame(
        xyz_cm=_vector(head["xyz_cm"], 3, "head.xyz_cm"),
        ypr_deg=_vector(head["ypr_deg"], 3, "head.ypr_deg"),
        fps=fps,
    )

    trackers = _object(root["trackers"], "trackers")
    _exact(trackers, set(_TRACKER_NAMES))

    def tracker(name: str) -> VmtFrame:
        value = _object(trackers[name], f"trackers.{name}")
        _exact(value, {"position", "quaternion_xyzw"})
        return VmtFrame(
            position=_vector(value["position"], 3, f"trackers.{name}.position"),
            quaternion_xyzw=_vector(
                value["quaternion_xyzw"],
                4,
                f"trackers.{name}.quaternion_xyzw",
            ),
            fps=fps,
        )

    locomotion = _vector(root["locomotion"], 3, "locomotion")
    if any(abs(value) > 1.0 for value in locomotion):
        raise DevicePayloadError("locomotion values must be in [-1, 1]")
    activation = _object(root["tracker_activation"], "tracker_activation")
    _exact(activation, set(_TRACKER_NAMES))
    # Recorded calibration clips still use a scalar face field. Keep the adapter
    # at the file/transport boundary; all live producers emit parameter maps.
    if "yawn" in root and "face" in root:
        raise DevicePayloadError("pose cannot mix face formats")
    face = root.get("face", {"ArdyYawn": root["yawn"]} if root.get("yawn") else {})
    if "yawn" in root:
        weight = _vector([root["yawn"]], 1, "face weight")[0]
        if not 0 <= weight <= 1: raise DevicePayloadError("face weight must be in [0, 1]")
    try: face_values(face)
    except ValueError as exc: raise DevicePayloadError(str(exc)) from exc
    return SixPointFrame(
        face=tuple(sorted(face.items())),
        head=head_frame,
        left=tracker("left"),
        right=tracker("right"),
        hips=tracker("hips"),
        left_foot=tracker("left_foot"),
        right_foot=tracker("right_foot"),
        fps=fps,
        scale=scale,
        locomotion_x=locomotion[0],
        locomotion_y=locomotion[1],
        locomotion_turn=locomotion[2],
        tracker_activation=TrackerActivation(
            **{
                name: _boolean(activation[name], f"tracker_activation.{name}")
                for name in _TRACKER_NAMES
            }
        ),
    )


def encode_wav_payload(wav_bytes: bytes) -> dict[str, object]:
    if not isinstance(wav_bytes, bytes) or not wav_bytes:
        raise DevicePayloadError("WAV payload must contain bytes")
    if len(wav_bytes) > MAX_WAV_PAYLOAD_BYTES:
        raise DevicePayloadError("WAV payload exceeds the size limit")
    return {
        "kind": "wav",
        "data": base64.b64encode(wav_bytes).decode("ascii"),
    }


def decode_wav_payload(payload: object) -> bytes:
    root = _object(payload, "WAV payload")
    _exact(root, {"kind", "data"})
    if root["kind"] != "wav":
        raise DevicePayloadError("WAV payload kind is invalid")
    data = root["data"]
    if not isinstance(data, str) or not data:
        raise DevicePayloadError("WAV payload data must be base64 text")
    try:
        wav = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise DevicePayloadError("WAV payload data is invalid base64") from exc
    if not wav:
        raise DevicePayloadError("WAV payload must not be empty")
    if len(wav) > MAX_WAV_PAYLOAD_BYTES:
        raise DevicePayloadError("WAV payload exceeds the size limit")
    return wav


def _object(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise DevicePayloadError(f"{location} must be an object")
    return value


def _exact(value: Mapping[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        raise DevicePayloadError("payload fields do not match the contract")


def _number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DevicePayloadError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DevicePayloadError(f"{location} must be finite")
    return result


def _positive(value: object, location: str) -> float:
    result = _number(value, location)
    if result <= 0:
        raise DevicePayloadError(f"{location} must be positive")
    return result


def _boolean(value: object, location: str) -> bool:
    if type(value) is not bool:
        raise DevicePayloadError(f"{location} must be a bool")
    return value


def _vector(value: object, length: int, location: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DevicePayloadError(f"{location} must be an array")
    if len(value) != length:
        raise DevicePayloadError(f"{location} must contain {length} numbers")
    return tuple(_number(item, f"{location}[{index}]") for index, item in enumerate(value))
