from __future__ import annotations

from dataclasses import dataclass, replace
import threading
import time
from typing import Any, Callable


@dataclass(frozen=True)
class PoseStreamStatus:
    running: bool = False
    frames_sent: int = 0
    started_monotonic: float = 0.0
    heartbeat_monotonic: float = 0.0
    last_error: str | None = None


class PoseStreamRunner:
    """Continuously publish one calibrated tracking pose with a heartbeat."""

    def __init__(
        self,
        *,
        frame: Any,
        sink: Any,
        rate_hz: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        self.frame = frame
        self.sink = sink
        self.rate_hz = float(rate_hz)
        self._clock = clock
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._started = False
        self._status = PoseStreamStatus()

    @property
    def status(self) -> PoseStreamStatus:
        with self._lock:
            return replace(self._status)

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(
        self,
        *,
        duration_seconds: float | None = None,
        max_frames: int | None = None,
        realtime: bool = True,
        heartbeat_interval_seconds: float = 1.0,
        heartbeat: Callable[[PoseStreamStatus], None] | None = None,
    ) -> PoseStreamStatus:
        if duration_seconds is not None and duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")
        with self._lock:
            if self._started:
                raise RuntimeError("pose stream runner is one-shot")
            self._started = True
            started_at = self._clock()
            self._status = PoseStreamStatus(
                running=True,
                started_monotonic=started_at,
                heartbeat_monotonic=started_at,
            )

        next_frame_deadline = started_at
        next_heartbeat = started_at
        try:
            while not self._stop_event.is_set():
                now = self._clock()
                if duration_seconds is not None and now - started_at >= duration_seconds:
                    break
                if max_frames is not None and self.status.frames_sent >= max_frames:
                    break

                self.sink.send(self.frame)
                now = self._clock()
                with self._lock:
                    self._status = replace(
                        self._status,
                        frames_sent=self._status.frames_sent + 1,
                        heartbeat_monotonic=now,
                    )

                if heartbeat is not None and now >= next_heartbeat:
                    heartbeat(self.status)
                    next_heartbeat = now + heartbeat_interval_seconds

                if realtime:
                    next_frame_deadline += 1.0 / self.rate_hz
                    self._stop_event.wait(max(0.0, next_frame_deadline - self._clock()))
        except Exception as exc:
            with self._lock:
                self._status = replace(self._status, last_error=str(exc))
            raise
        finally:
            try:
                self.sink.close()
            finally:
                with self._lock:
                    self._status = replace(
                        self._status,
                        running=False,
                        heartbeat_monotonic=self._clock(),
                    )
        return self.status
