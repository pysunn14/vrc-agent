from __future__ import annotations

import math
import socket
import struct
import time


def _osc_string(value: str) -> bytes:
    raw = value.encode("utf-8") + b"\x00"
    return raw + (b"\x00" * ((-len(raw)) % 4))


def _encode_vector3(address: str, value: tuple[float, float, float]) -> bytes:
    if len(value) != 3:
        raise ValueError("OSC Vector3 must contain exactly three values")
    values = tuple(float(component) for component in value)
    if any(not math.isfinite(component) for component in values):
        raise ValueError("OSC Vector3 must contain only finite values")
    return b"".join(
        [
            _osc_string(address),
            _osc_string(",fff"),
            struct.pack(">fff", *values),
        ]
    )


def _tracker_address(index: int, component: str) -> str:
    if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= 8:
        raise ValueError("VRChat OSC tracker index must be an integer from 1 to 8")
    return f"/tracking/trackers/{index}/{component}"


def encode_vrchat_tracker_position(
    index: int,
    position: tuple[float, float, float],
) -> bytes:
    return _encode_vector3(_tracker_address(index, "position"), position)


def encode_vrchat_tracker_rotation(
    index: int,
    euler_deg: tuple[float, float, float],
) -> bytes:
    return _encode_vector3(_tracker_address(index, "rotation"), euler_deg)


def encode_vrchat_head_tracker_position(
    position: tuple[float, float, float],
) -> bytes:
    return _encode_vector3("/tracking/trackers/head/position", position)


def encode_vrchat_axis(name: str, value: float) -> bytes:
    value = float(value)
    if not -1.0 <= value <= 1.0:
        raise ValueError(f"axis value must be in [-1, 1], got {value}")
    return b"".join(
        [
            _osc_string(f"/input/{name}"),
            _osc_string(",f"),
            struct.pack(">f", value),
        ]
    )


def encode_vrchat_button(name: str, pressed: bool) -> bytes:
    return b"".join(
        [
            _osc_string(f"/input/{name}"),
            _osc_string(",i"),
            struct.pack(">i", 1 if pressed else 0),
        ]
    )


def hold_axis(
    *,
    host: str,
    name: str,
    value: float,
    duration: float,
    port: int = 9000,
    rate_hz: float = 20.0,
) -> int:
    if duration <= 0:
        raise ValueError("duration must be positive")
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive")

    active = encode_vrchat_axis(name, value)
    neutral = encode_vrchat_axis(name, 0.0)
    interval = 1.0 / rate_hz
    deadline = time.perf_counter() + duration
    sent = 0
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        while time.perf_counter() < deadline:
            sock.sendto(active, (host, port))
            sent += 1
            time.sleep(interval)
    finally:
        sock.sendto(neutral, (host, port))
        sock.close()
    return sent


def press_button(
    *,
    host: str,
    name: str,
    port: int = 9000,
    hold_seconds: float = 0.08,
) -> None:
    if hold_seconds <= 0:
        raise ValueError("hold_seconds must be positive")
    pressed = encode_vrchat_button(name, True)
    released = encode_vrchat_button(name, False)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Explicit release first so repeated button presses always create a clean edge.
        sock.sendto(released, (host, port))
        sock.sendto(pressed, (host, port))
        time.sleep(hold_seconds)
        sock.sendto(released, (host, port))
    finally:
        sock.close()
