from __future__ import annotations

import socket
import struct
import time


def _osc_string(value: str) -> bytes:
    raw = value.encode("utf-8") + b"\x00"
    return raw + (b"\x00" * ((-len(raw)) % 4))


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