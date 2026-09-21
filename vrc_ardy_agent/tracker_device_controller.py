from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from .six_point_bridge import SixPointFrame


class TrackerDeviceState(StrEnum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    STREAMING = "STREAMING"
    FAILED = "FAILED"


class TrackerOutputMode(StrEnum):
    SAFE = "SAFE"
    ACTION = "ACTION"


@dataclass(frozen=True, slots=True)
class TrackerDeviceSnapshot:
    running: bool
    state: TrackerDeviceState
    mode: TrackerOutputMode
    frames_sent: int
    target_revision: int
    safe_transitions: int
    heartbeat_monotonic: float
    last_error: str | None
    applied_revision: int = -1


class TrackerDeviceController:
    """Continuously writes one authoritative tracker target at a fixed rate.

    Losing an action lease selects the safe target. It does not unregister the
    virtual trackers. Tracker disablement is reserved for final service close.
    """

    def __init__(
        self,
        *,
        sink: Any,
        safe_frame: SixPointFrame,
        rate_hz: float = 20.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(safe_frame, SixPointFrame):
            raise TypeError("safe_frame must be a SixPointFrame")
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        self._sink = sink
        self._safe_frame = safe_frame
        self._target_frame = safe_frame
        self._period_seconds = 1.0 / float(rate_hz)
        self._monotonic = monotonic
        self._condition = threading.Condition(threading.RLock())
        self._io_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._status = TrackerDeviceSnapshot(
            running=False,
            state=TrackerDeviceState.STOPPED,
            mode=TrackerOutputMode.SAFE,
            frames_sent=0,
            target_revision=0,
            safe_transitions=0,
            heartbeat_monotonic=self._monotonic(),
            last_error=None,
        )

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                raise RuntimeError("tracker device controller is already started")
            if self._closed:
                raise RuntimeError("tracker device controller is closed")
            self._stop_event.clear()
            self._status = replace(
                self._status,
                state=TrackerDeviceState.STARTING,
                heartbeat_monotonic=self._monotonic(),
                last_error=None,
            )
        try:
            self._write_frame(self._safe_frame)
        except BaseException as exc:
            self._record_failure("initial safe pose failed", exc)
            raise

        with self._condition:
            self._status = replace(
                self._status,
                running=True,
                state=TrackerDeviceState.STREAMING,
                mode=TrackerOutputMode.SAFE,
                heartbeat_monotonic=self._monotonic(),
            )
            thread = threading.Thread(
                target=self._run,
                name="tracker-device-controller",
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def send(self, frame: SixPointFrame) -> None:
        if not isinstance(frame, SixPointFrame):
            raise TypeError("tracker target must be a SixPointFrame")
        with self._condition:
            if self._closed or not self._status.running:
                raise RuntimeError("tracker device controller is not running")
            self._target_frame = frame
            self._status = replace(
                self._status,
                mode=TrackerOutputMode.ACTION,
                target_revision=self._status.target_revision + 1,
                heartbeat_monotonic=self._monotonic(),
            )
            self._condition.notify_all()

    def neutralize_inputs(self) -> None:
        with self._io_lock:
            self._sink.neutralize_inputs()

    def neutralize_pose(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._target_frame = self._safe_frame
            self._status = replace(
                self._status,
                mode=TrackerOutputMode.SAFE,
                target_revision=self._status.target_revision + 1,
                safe_transitions=self._status.safe_transitions + 1,
                heartbeat_monotonic=self._monotonic(),
            )
            running = self._status.running
            self._condition.notify_all()
        if not running:
            return
        try:
            with self._io_lock:
                self._sink.neutralize_inputs()
                self._sink.send(self._safe_frame)
            self._count_frame()
        except BaseException as exc:
            self._record_failure("safe pose transition failed", exc)
            raise

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
            self._stop_event.set()
            self._condition.notify_all()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self._period_seconds * 4.0))
            if thread.is_alive():
                raise TimeoutError("tracker device controller did not stop")

        errors: list[BaseException] = []
        with self._io_lock:
            try:
                self._sink.neutralize_pose()
            except BaseException as exc:
                errors.append(exc)
            try:
                self._sink.close()
            except BaseException as exc:
                errors.append(exc)
        with self._condition:
            self._thread = None
            self._status = replace(
                self._status,
                running=False,
                state=(
                    TrackerDeviceState.FAILED if errors else TrackerDeviceState.STOPPED
                ),
                heartbeat_monotonic=self._monotonic(),
                last_error=(
                    self._status.last_error
                    if not errors
                    else "; ".join(
                        f"{type(error).__name__}: {error}" for error in errors
                    )
                ),
            )
        if errors:
            raise RuntimeError(self._status.last_error)

    def snapshot(self) -> TrackerDeviceSnapshot:
        with self._condition:
            return replace(self._status)

    def _run(self) -> None:
        deadline = self._monotonic() + self._period_seconds
        try:
            while not self._stop_event.is_set():
                remaining = max(0.0, deadline - self._monotonic())
                if self._stop_event.wait(remaining):
                    return
                try:
                    self.flush_target()
                except BaseException as exc:
                    self._record_failure("tracker stream failed", exc)
                    self._stop_event.set()
                    return
                deadline += self._period_seconds
                now = self._monotonic()
                if deadline <= now:
                    deadline = now + self._period_seconds
        finally:
            with self._condition:
                if (
                    not self._closed
                    and self._status.state is not TrackerDeviceState.FAILED
                ):
                    self._status = replace(
                        self._status,
                        running=False,
                        state=TrackerDeviceState.STOPPED,
                        heartbeat_monotonic=self._monotonic(),
                    )
                self._condition.notify_all()

    def flush_target(self) -> int:
        """Write the current target and report its observed output revision.

        Read the target under the I/O lock so an older queued loop iteration
        cannot overwrite a completed calibration update or SAFE transition.
        This observes local device output, not VRChat's rendered IK result.
        """
        with self._io_lock:
            with self._condition:
                if not self._status.running or self._closed:
                    raise RuntimeError("tracker output is not running")
                frame = self._target_frame
                revision = self._status.target_revision
            try:
                self._sink.send(frame)
            except BaseException as exc:
                self._record_failure("tracker output failed", exc)
                raise
            with self._condition:
                self._status = replace(self._status, applied_revision=revision)
            self._count_frame()
            return revision

    def _write_frame(self, frame: SixPointFrame) -> None:
        with self._io_lock:
            self._sink.send(frame)
        self._count_frame()

    def _count_frame(self) -> None:
        with self._condition:
            self._status = replace(
                self._status,
                frames_sent=self._status.frames_sent + 1,
                heartbeat_monotonic=self._monotonic(),
            )

    def _record_failure(self, context: str, exc: BaseException) -> None:
        with self._condition:
            self._status = replace(
                self._status,
                running=False,
                state=TrackerDeviceState.FAILED,
                heartbeat_monotonic=self._monotonic(),
                last_error=f"{context}: {type(exc).__name__}: {exc}",
            )
            self._condition.notify_all()
