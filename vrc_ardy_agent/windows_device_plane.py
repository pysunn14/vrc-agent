from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True, slots=True)
class WindowsDevicePlaneSnapshot:
    running: bool
    streaming_started_wall_time: float | None
    components: Mapping[str, object]
    vmt_alive: bool | None
    registration_verified: bool
    heartbeat_monotonic: float
    last_errors: tuple[str, ...]


class WindowsDevicePlane:
    """Long-lived owner of every final Windows output device.

    The tracker writer starts before the Mac bridge. The bridge may switch
    sessions and the VRChat sensor may be replaced, but only final service stop
    closes the devices and disables their trackers.
    """

    def __init__(
        self,
        *,
        tracker_controller: Any,
        bridge: Any,
        devices: Any,
        status_sources: Mapping[str, Any] | None = None,
        calibration: Any | None = None,
        wall_time: Any = time.time,
        monotonic: Any = time.monotonic,
    ) -> None:
        self.calibration = calibration
        self._tracker_controller = tracker_controller
        self._bridge = bridge
        self._devices = devices
        self._status_sources = dict(status_sources or {})
        self._wall_time = wall_time
        self._monotonic = monotonic
        self._lock = threading.RLock()
        self._started = False
        self._closed = False
        self._tracker_attempted = False
        self._bridge_attempted = False
        self._status = WindowsDevicePlaneSnapshot(
            running=False,
            streaming_started_wall_time=None,
            components={},
            # VMT's UDP listener and device registration are not observable
            # through the current protocol. Unknown must not masquerade as OK.
            vmt_alive=None,
            registration_verified=False,
            heartbeat_monotonic=self._monotonic(),
            last_errors=(),
        )

    def start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("Windows device plane is already started")
            if self._closed:
                raise RuntimeError("Windows device plane is closed")
            self._started = True
        try:
            self._tracker_attempted = True
            self._tracker_controller.start()
            if self.calibration is not None: self.calibration.start_watchdog()
            with self._lock:
                self._status = replace(
                    self._status,
                    streaming_started_wall_time=float(self._wall_time()),
                    heartbeat_monotonic=self._monotonic(),
                )
            self._bridge_attempted = True
            self._bridge.start()
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

    def snapshot(self) -> WindowsDevicePlaneSnapshot:
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
            bridge_attempted = self._bridge_attempted
        errors: list[BaseException] = []
        if self.calibration is not None:
            try: self.calibration.close()
            except BaseException as exc:
                errors.append(exc)
                self._append_error(f"calibration stop failed: {exc}")
        if bridge_attempted:
            try:
                self._bridge.stop()
            except BaseException as exc:
                errors.append(exc)
                self._append_error(f"bridge stop failed: {type(exc).__name__}: {exc}")
        # WindowsDeviceSink owns the tracker controller and wave player. Its
        # close is the sole final path that disables trackers.
        try:
            self._devices.close()
        except BaseException as exc:
            errors.append(exc)
            self._append_error(f"device close failed: {type(exc).__name__}: {exc}")
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
