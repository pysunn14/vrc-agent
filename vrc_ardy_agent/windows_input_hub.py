from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from .stream_protocol import ScreenshotRequestMessage, StreamMessage
from .windows_input_transport import WindowsInputSnapshot, WindowsInputTransport


class WindowsInputHubError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class WindowsInputHubSnapshot:
    sensor_attached: bool
    sensor_id: str | None
    sensor_generation: int
    stale_audio_drops: int
    heartbeat_monotonic: float
    transport: WindowsInputSnapshot


class WindowsInputHub:
    """Stable bridge input boundary with one replaceable VRChat sensor."""

    def __init__(
        self,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._monotonic = monotonic
        self._lock = threading.RLock()
        self._sensor_id: str | None = None
        self._screenshot_provider: Callable[[], bytes] | None = None
        self._sensor_generation = 0
        self._stale_audio_drops = 0
        self._heartbeat_monotonic = monotonic()
        self._transport = WindowsInputTransport(
            screenshot_provider=self._capture_screenshot,
        )

    def attach_sensor(
        self,
        sensor_id: str,
        screenshot_provider: Callable[[], bytes],
    ) -> int:
        sensor_id = sensor_id.strip()
        if not sensor_id:
            raise ValueError("sensor_id must not be empty")
        if not callable(screenshot_provider):
            raise TypeError("screenshot_provider must be callable")
        with self._lock:
            self._sensor_generation += 1
            self._sensor_id = sensor_id
            self._screenshot_provider = screenshot_provider
            self._touch_locked()
            return self._sensor_generation

    def detach_sensor(self, sensor_id: str) -> bool:
        with self._lock:
            if sensor_id != self._sensor_id:
                return False
            self._sensor_id = None
            self._screenshot_provider = None
            self._touch_locked()
            return True

    def publish_audio(
        self,
        sensor_id: str,
        pcm: bytes,
        *,
        captured_monotonic_ns: int,
    ) -> bool:
        with self._lock:
            if sensor_id != self._sensor_id:
                self._stale_audio_drops += 1
                self._touch_locked()
                return False
            published = self._transport.publish_audio(
                pcm,
                captured_monotonic_ns=captured_monotonic_ns,
            )
            self._touch_locked()
            return published

    def open_session(
        self,
        session_id: str,
        sender: Callable[[StreamMessage], None],
    ) -> None:
        self._transport.open_session(session_id, sender)

    def close_session(self, session_id: str) -> bool:
        return self._transport.close_session(session_id)

    def handle_screenshot_request(
        self,
        request: ScreenshotRequestMessage,
    ) -> None:
        self._transport.handle_screenshot_request(request)

    def send_heartbeat(self) -> None:
        self._transport.send_heartbeat()

    def snapshot(self) -> WindowsInputHubSnapshot:
        transport = self._transport.snapshot()
        with self._lock:
            return WindowsInputHubSnapshot(
                sensor_attached=self._sensor_id is not None,
                sensor_id=self._sensor_id,
                sensor_generation=self._sensor_generation,
                stale_audio_drops=self._stale_audio_drops,
                heartbeat_monotonic=max(
                    self._heartbeat_monotonic,
                    transport.heartbeat_monotonic,
                ),
                transport=transport,
            )

    def _capture_screenshot(self) -> bytes:
        # Hold the sensor lock through capture. Detach then cannot close a
        # capture source while a bridge request is still reading from it.
        with self._lock:
            provider = self._screenshot_provider
            if self._sensor_id is None or provider is None:
                raise WindowsInputHubError("sensor_not_attached")
            return provider()

    def _touch_locked(self) -> None:
        self._heartbeat_monotonic = self._monotonic()
