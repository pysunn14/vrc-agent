from __future__ import annotations

import socket
import struct
import time


VMT_INPUT_JOYSTICK = "/VMT/Input/Joystick"
VMT_INPUT_JOYSTICK_CLICK = "/VMT/Input/Joystick/Click"
VMT_INPUT_TRIGGER = "/VMT/Input/Trigger"
VMT_INPUT_BUTTON = "/VMT/Input/Button"


def _osc_string(value: str) -> bytes:
    raw = value.encode("utf-8") + b"\x00"
    return raw + (b"\x00" * ((-len(raw)) % 4))


def _unit_axis(value: float, name: str) -> float:
    value = float(value)
    if not -1.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [-1, 1], got {value}")
    return value


def encode_vmt_joystick(
    *,
    index: int,
    joystick_index: int,
    timeoffset: float,
    x: float,
    y: float,
) -> bytes:
    if joystick_index != 0:
        raise ValueError("VMT supports joystick_index 0 only")
    x = _unit_axis(x, "x")
    y = _unit_axis(y, "y")
    return b"".join(
        [
            _osc_string(VMT_INPUT_JOYSTICK),
            _osc_string(",iifff"),
            struct.pack(">ii", int(index), int(joystick_index)),
            struct.pack(">fff", float(timeoffset), x, y),
        ]
    )


def encode_vmt_trigger(
    *,
    index: int,
    trigger_index: int,
    timeoffset: float,
    value: float,
) -> bytes:
    if trigger_index not in (0, 1):
        raise ValueError(f"trigger_index must be 0 or 1, got {trigger_index}")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"trigger value must be in [0, 1], got {value}")
    return b"".join(
        [
            _osc_string(VMT_INPUT_TRIGGER),
            _osc_string(",iiff"),
            struct.pack(">ii", int(index), int(trigger_index)),
            struct.pack(">ff", float(timeoffset), value),
        ]
    )


def encode_vmt_joystick_click(
    *,
    index: int,
    joystick_index: int,
    timeoffset: float,
    pressed: bool,
) -> bytes:
    if not 0 <= joystick_index <= 3:
        raise ValueError(f"joystick_index must be in [0, 3], got {joystick_index}")
    return b"".join(
        [
            _osc_string(VMT_INPUT_JOYSTICK_CLICK),
            _osc_string(",iifi"),
            struct.pack(">ii", int(index), int(joystick_index)),
            struct.pack(">fi", float(timeoffset), 1 if pressed else 0),
        ]
    )


def encode_vmt_button(
    *,
    index: int,
    button_index: int,
    timeoffset: float,
    pressed: bool,
) -> bytes:
    if not 0 <= button_index <= 7:
        raise ValueError(f"button_index must be in [0, 7], got {button_index}")
    return b"".join(
        [
            _osc_string(VMT_INPUT_BUTTON),
            _osc_string(",iifi"),
            struct.pack(">ii", int(index), int(button_index)),
            struct.pack(">fi", float(timeoffset), 1 if pressed else 0),
        ]
    )


def hold_joystick(
    *,
    host: str,
    index: int,
    x: float,
    y: float,
    duration: float,
    port: int = 39570,
    rate_hz: float = 20.0,
) -> int:
    if duration <= 0:
        raise ValueError("duration must be positive")
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive")

    active = encode_vmt_joystick(index=index, joystick_index=0, timeoffset=0.0, x=x, y=y)
    neutral = encode_vmt_joystick(index=index, joystick_index=0, timeoffset=0.0, x=0.0, y=0.0)
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
