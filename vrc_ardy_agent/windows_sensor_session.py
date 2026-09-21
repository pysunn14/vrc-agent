from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True, slots=True)
class VrchatSensorSnapshot:
    running: bool
    sensor_id: str
    components: Mapping[str, object]
    heartbeat_monotonic: float
    last_errors: tuple[str, ...]


class VrchatSensorSession:
    """Own capture and audio resources tied to one VRChat process.

    This session deliberately has no reference to the tracker writer, device
    gateway, or Mac bridge. Replacing a VRChat process therefore cannot tear
    down virtual tracker registration.
    """

    def __init__(
        self,
        *,
        sensor_id: str,
        capture: Any,
        screenshot_provider: Any,
        audio: Any,
        input_hub: Any,
        status_sources: Mapping[str, Any] | None = None,
        monotonic: Any = time.monotonic,
    ) -> None:
        sensor_id = sensor_id.strip()
        if not sensor_id:
            raise ValueError("sensor_id must not be empty")
        if not callable(screenshot_provider):
            raise TypeError("screenshot_provider must be callable")
        self.sensor_id = sensor_id
        self._capture = capture
        self._screenshot_provider = screenshot_provider
        self._audio = audio
        self._input_hub = input_hub
        self._status_sources = dict(status_sources or {})
        self._monotonic = monotonic
        self._lock = threading.RLock()
        self._started = False
        self._closed = False
        self._capture_attempted = False
        self._sensor_attached = False
        self._audio_attempted = False
        self._status = VrchatSensorSnapshot(
            running=False,
            sensor_id=sensor_id,
            components={},
            heartbeat_monotonic=self._monotonic(),
            last_errors=(),
        )

    def start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("VRChat sensor session is already started")
            if self._closed:
                raise RuntimeError("VRChat sensor session is closed")
            self._started = True
        try:
            # A screenshot provider cannot read a capture source before capture
            # is live. Publishing it to the stable hub is therefore second.
            self._capture_attempted = True
            self._capture.start()
            self._input_hub.attach_sensor(
                self.sensor_id,
                self._screenshot_provider,
            )
            self._sensor_attached = True
            self._audio_attempted = True
            self._audio.start()
        except BaseException as exc:
            self._append_error(f"start failed: {type(exc).__name__}: {exc}")
            self._shutdown(raise_errors=False)
            raise
        with self._lock:
            self._status = replace(
                self._status,
                running=True,
                heartbeat_monotonic=self._monotonic(),
            )

    def stop(self) -> None:
        self._shutdown(raise_errors=True)

    def snapshot(self) -> VrchatSensorSnapshot:
        components: dict[str, object] = {}
        for name, source in self._status_sources.items():
            try:
                reader = getattr(source, "snapshot", None)
                components[name] = reader() if callable(reader) else source()
            except Exception as exc:
                components[name] = {"status_error": f"{type(exc).__name__}: {exc}"}
        with self._lock:
            self._status = replace(
                self._status,
                components=components,
                heartbeat_monotonic=self._monotonic(),
            )
            return replace(self._status, components=dict(components))

    def _shutdown(self, *, raise_errors: bool) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            sensor_attached = self._sensor_attached
            audio_attempted = self._audio_attempted
            capture_attempted = self._capture_attempted
            self._sensor_attached = False
        errors: list[BaseException] = []

        # Detach first. The hub locks an in-flight screenshot through capture,
        # so after this returns neither screenshot nor audio can enter again.
        if sensor_attached:
            try:
                self._input_hub.detach_sensor(self.sensor_id)
            except BaseException as exc:
                errors.append(exc)
                self._append_error(f"sensor detach failed: {type(exc).__name__}: {exc}")
        if audio_attempted:
            try:
                self._audio.close()
            except BaseException as exc:
                errors.append(exc)
                self._append_error(f"audio close failed: {type(exc).__name__}: {exc}")
        if capture_attempted:
            try:
                self._capture.close()
            except BaseException as exc:
                errors.append(exc)
                self._append_error(f"capture close failed: {type(exc).__name__}: {exc}")
        with self._lock:
            self._status = replace(
                self._status,
                running=False,
                heartbeat_monotonic=self._monotonic(),
            )
        if errors and raise_errors:
            raise RuntimeError(
                "; ".join(f"{type(exc).__name__}: {exc}" for exc in errors)
            )

    def _append_error(self, error: str) -> None:
        with self._lock:
            self._status = replace(
                self._status,
                last_errors=(*self._status.last_errors, error)[-50:],
                heartbeat_monotonic=self._monotonic(),
            )
