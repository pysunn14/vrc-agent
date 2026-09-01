from __future__ import annotations

import socket
import threading
import time
from typing import Any, Callable

from .follow_control import FollowDecision
from .vmt_input import encode_vmt_joystick
from .vrchat_osc import encode_vrchat_axis, encode_vrchat_button


class _TurnPulse:
    def __init__(
        self,
        *,
        interval_seconds: float,
        clock: Callable[[], float],
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("turn pulse interval must be positive")
        self.interval_seconds = float(interval_seconds)
        self._clock = clock
        self._pressed = False
        self._direction = 0
        self._next_pulse_monotonic = float("-inf")

    def next_direction(self, value: float) -> int:
        direction = 1 if value > 0 else -1 if value < 0 else 0
        now = self._clock()
        press_direction = 0
        if direction == 0:
            self._pressed = False
            self._direction = 0
            self._next_pulse_monotonic = now
        elif self._pressed:
            self._pressed = False
            if direction != self._direction:
                self._direction = direction
                self._next_pulse_monotonic = now
        else:
            if direction != self._direction:
                self._direction = direction
                self._next_pulse_monotonic = now
            if now >= self._next_pulse_monotonic:
                press_direction = direction
                self._pressed = True
                self._next_pulse_monotonic = now + self.interval_seconds
        return press_direction


class VrchatLocomotionSink:
    """Send follower axes independently of ARDY body-frame production."""

    def __init__(
        self,
        *,
        host: str,
        port: int = 9000,
        dry_run: bool = False,
        turn_buttons: bool = False,
        turn_pulse_interval_seconds: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
        socket_factory: Callable[[], Any] | None = None,
    ) -> None:
        if not host:
            raise ValueError("host must not be empty")
        if not 1 <= port <= 65535:
            raise ValueError("port must be in [1, 65535]")
        if turn_pulse_interval_seconds <= 0:
            raise ValueError("turn_pulse_interval_seconds must be positive")
        self.host = host
        self.port = int(port)
        self.dry_run = bool(dry_run)
        self.turn_buttons = bool(turn_buttons)
        self.turn_pulse_interval_seconds = float(turn_pulse_interval_seconds)
        self._turn_pulse = _TurnPulse(
            interval_seconds=turn_pulse_interval_seconds,
            clock=clock,
        )
        self._lock = threading.Lock()
        self._closed = False
        self.packets_sent = 0
        if self.dry_run:
            self._socket = None
        else:
            factory = socket_factory or (
                lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            )
            self._socket = factory()

    def send(self, decision: FollowDecision) -> None:
        self._send_axes(
            horizontal=decision.horizontal,
            vertical=decision.vertical,
            look_horizontal=decision.look_horizontal,
        )

    def neutralize(self) -> None:
        self._send_axes(horizontal=0.0, vertical=0.0, look_horizontal=0.0)

    def _send_axes(
        self,
        *,
        horizontal: float,
        vertical: float,
        look_horizontal: float,
    ) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot send through a closed locomotion sink")
            packets = self._packets(
                horizontal=horizontal,
                vertical=vertical,
                look_horizontal=look_horizontal,
            )
            if self.dry_run:
                return
            if self._socket is None:
                raise RuntimeError("UDP socket is unavailable")
            for packet in packets:
                self._socket.sendto(packet, (self.host, self.port))
                self.packets_sent += 1

    def _packets(
        self,
        *,
        horizontal: float,
        vertical: float,
        look_horizontal: float,
    ) -> tuple[bytes, ...]:
        if not self.turn_buttons:
            return (
                encode_vrchat_axis("Horizontal", horizontal),
                encode_vrchat_axis("Vertical", vertical),
                encode_vrchat_axis("LookHorizontal", look_horizontal),
            )

        press_direction = self._turn_pulse.next_direction(look_horizontal)
        return (
            encode_vrchat_axis("Horizontal", horizontal),
            encode_vrchat_axis("Vertical", vertical),
            encode_vrchat_axis("LookHorizontal", 0.0),
            encode_vrchat_button("LookLeft", press_direction < 0),
            encode_vrchat_button("LookRight", press_direction > 0),
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            sock = self._socket
            try:
                if not self.dry_run and sock is not None:
                    for name in ("Horizontal", "Vertical", "LookHorizontal"):
                        sock.sendto(encode_vrchat_axis(name, 0.0), (self.host, self.port))
                        self.packets_sent += 1
                    if self.turn_buttons:
                        for name in ("LookLeft", "LookRight"):
                            sock.sendto(
                                encode_vrchat_button(name, False),
                                (self.host, self.port),
                            )
                            self.packets_sent += 1
            finally:
                self._closed = True
                if sock is not None:
                    sock.close()


class VmtLocomotionSink:
    """Drive VR locomotion through the same VMT hand controllers as ARDY."""

    def __init__(
        self,
        *,
        host: str,
        port: int = 39570,
        left_controller_index: int = 1,
        right_controller_index: int = 2,
        dry_run: bool = False,
        turn_pulse_interval_seconds: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
        socket_factory: Callable[[], Any] | None = None,
    ) -> None:
        if not host:
            raise ValueError("host must not be empty")
        if not 1 <= port <= 65535:
            raise ValueError("port must be in [1, 65535]")
        if left_controller_index == right_controller_index:
            raise ValueError("left and right controller indexes must differ")
        self.host = host
        self.port = int(port)
        self.left_controller_index = int(left_controller_index)
        self.right_controller_index = int(right_controller_index)
        self.dry_run = bool(dry_run)
        self._turn_pulse = _TurnPulse(
            interval_seconds=turn_pulse_interval_seconds,
            clock=clock,
        )
        self._lock = threading.Lock()
        self._closed = False
        self.packets_sent = 0
        if self.dry_run:
            self._socket = None
        else:
            factory = socket_factory or (
                lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            )
            self._socket = factory()

    def send(self, decision: FollowDecision) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot send through a closed locomotion sink")
            turn_direction = self._turn_pulse.next_direction(decision.look_horizontal)
            packets = (
                encode_vmt_joystick(
                    index=self.left_controller_index,
                    joystick_index=0,
                    timeoffset=0.0,
                    x=decision.horizontal,
                    y=decision.vertical,
                ),
                encode_vmt_joystick(
                    index=self.right_controller_index,
                    joystick_index=0,
                    timeoffset=0.0,
                    # Full deflection works with both smooth and comfort turning.
                    x=float(turn_direction),
                    y=0.0,
                ),
            )
            self._send_packets(packets)

    def neutralize(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._turn_pulse.next_direction(0.0)
            self._send_packets(self._neutral_packets())

    def _neutral_packets(self) -> tuple[bytes, bytes]:
        return (
            encode_vmt_joystick(
                index=self.left_controller_index,
                joystick_index=0,
                timeoffset=0.0,
                x=0.0,
                y=0.0,
            ),
            encode_vmt_joystick(
                index=self.right_controller_index,
                joystick_index=0,
                timeoffset=0.0,
                x=0.0,
                y=0.0,
            ),
        )

    def _send_packets(self, packets: tuple[bytes, ...]) -> None:
        if self.dry_run:
            return
        if self._socket is None:
            raise RuntimeError("UDP socket is unavailable")
        for packet in packets:
            self._socket.sendto(packet, (self.host, self.port))
            self.packets_sent += 1

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            sock = self._socket
            try:
                self._send_packets(self._neutral_packets())
            finally:
                self._closed = True
                if sock is not None:
                    sock.close()
