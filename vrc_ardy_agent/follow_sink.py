from __future__ import annotations

import socket
import threading
import time
from typing import Any, Callable

from .follow_control import FollowDecision
from .vrchat_osc import encode_vrchat_axis, encode_vrchat_button


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
        self._clock = clock
        self._turn_button_pressed = False
        self._turn_direction = 0
        self._next_turn_pulse_monotonic = float("-inf")
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

        direction = 1 if look_horizontal > 0 else -1 if look_horizontal < 0 else 0
        now = self._clock()
        press_direction = 0
        if direction == 0:
            self._turn_button_pressed = False
            self._turn_direction = 0
            self._next_turn_pulse_monotonic = now
        elif self._turn_button_pressed:
            self._turn_button_pressed = False
            if direction != self._turn_direction:
                self._turn_direction = direction
                self._next_turn_pulse_monotonic = now
        else:
            if direction != self._turn_direction:
                self._turn_direction = direction
                self._next_turn_pulse_monotonic = now
            if now >= self._next_turn_pulse_monotonic:
                press_direction = direction
                self._turn_button_pressed = True
                self._next_turn_pulse_monotonic = (
                    now + self.turn_pulse_interval_seconds
                )
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
