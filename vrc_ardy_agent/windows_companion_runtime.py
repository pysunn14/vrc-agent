from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True, slots=True)
class WindowsCompanionSnapshot:
    running: bool
    late_registration: bool | None
    device_plane: object
    sensor_supervisor: object
    components: Mapping[str, object]
    heartbeat_monotonic: float
    last_errors: tuple[str, ...]


class WindowsCompanionRuntime:
    """Root lifecycle: stable devices outside replaceable VRChat sensors."""

    def __init__(
        self,
        *,
        device_plane: Any,
        sensor_supervisor: Any,
        monotonic: Any = time.monotonic,
    ) -> None:
        self.device_plane = device_plane
        self.sensor_supervisor = sensor_supervisor
        self._monotonic = monotonic
        self._lock = threading.RLock()
        self._started = False
        self._closed = False
        self._device_plane_attempted = False
        self._sensors_attempted = False
        self._status = WindowsCompanionSnapshot(
            running=False,
            late_registration=None,
            device_plane={},
            sensor_supervisor={},
            components={},
            heartbeat_monotonic=self._monotonic(),
            last_errors=(),
        )

    def start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("Windows companion is already started")
            if self._closed:
                raise RuntimeError("Windows companion is closed")
            self._started = True
        try:
            # Virtual tracker streaming must predate every VRChat sensor epoch.
            self.device_plane.start()
            self._device_plane_attempted = True
            self.sensor_supervisor.start()
            self._sensors_attempted = True
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

    def snapshot(self) -> WindowsCompanionSnapshot:
        plane = self._safe_snapshot("device plane", self.device_plane)
        sensors = self._safe_snapshot("sensor supervisor", self.sensor_supervisor)
        late_registration = _detect_late_registration(plane, sensors)
        components = {
            "device_plane": plane,
            "sensor_supervisor": sensors,
        }
        with self._lock:
            self._status = replace(
                self._status,
                late_registration=late_registration,
                device_plane=plane,
                sensor_supervisor=sensors,
                components=components,
                heartbeat_monotonic=self._monotonic(),
            )
            return replace(self._status, components=dict(components))

    def _shutdown(self, *, raise_errors: bool) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            sensors_attempted = self._sensors_attempted
            device_plane_attempted = self._device_plane_attempted
        errors: list[BaseException] = []
        if sensors_attempted:
            try:
                self.sensor_supervisor.stop()
            except BaseException as exc:
                errors.append(exc)
                self._append_error(
                    f"sensor supervisor stop failed: {type(exc).__name__}: {exc}"
                )
        if device_plane_attempted:
            try:
                self.device_plane.stop()
            except BaseException as exc:
                errors.append(exc)
                self._append_error(
                    f"device plane stop failed: {type(exc).__name__}: {exc}"
                )
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

    def _safe_snapshot(self, name: str, source: Any) -> object:
        try:
            return source.snapshot()
        except Exception as exc:
            error = f"{name} snapshot failed: {type(exc).__name__}: {exc}"
            self._append_error(error)
            return {"status_error": error}

    def _append_error(self, error: str) -> None:
        with self._lock:
            self._status = replace(
                self._status,
                last_errors=(*self._status.last_errors, error)[-50:],
                heartbeat_monotonic=self._monotonic(),
            )


def _detect_late_registration(
    plane_snapshot: object,
    sensor_snapshot: object,
) -> bool | None:
    stream_started_at = _read_field(
        plane_snapshot,
        "streaming_started_wall_time",
    )
    target = _read_field(sensor_snapshot, "target")
    process_started_at = _read_field(target, "process_started_at")
    if stream_started_at is None or process_started_at is None:
        return None
    try:
        return float(process_started_at) < float(stream_started_at)
    except (TypeError, ValueError):
        return None


def _read_field(value: object, name: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)
