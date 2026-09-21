from __future__ import annotations

from collections.abc import Callable
import threading
from typing import Any

from .live_session import LiveSessionStatus, PromptPlaybackStarted


class MotionSessionPlayback:
    """Threaded playback lifecycle for one already-prefilled live session."""

    def __init__(
        self,
        *,
        session: Any,
        realtime: bool,
        heartbeat: Callable[[LiveSessionStatus], None],
        prompt_started: Callable[[PromptPlaybackStarted], None],
        finished: Callable[[], None],
    ) -> None:
        self.session = session
        self._realtime = realtime
        self._heartbeat = heartbeat
        self._prompt_started = prompt_started
        self._finished = finished
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._stop_reason: str | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="ardy-motion-playback",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def request_stop(self, reason: str) -> None:
        with self._lock:
            self._stop_reason = self._stop_reason or reason
        self._cancel.set()
        self.session.request_stop()

    def fail(self, error: BaseException) -> None:
        with self._lock:
            self._error = self._error or error
        self.request_stop("failed")

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def result(self) -> tuple[BaseException | None, str | None]:
        self._thread.join()
        with self._lock:
            return self._error, self._stop_reason

    def _run(self) -> None:
        error: BaseException | None = None
        try:
            self.session.run_started(
                realtime=self._realtime,
                cancel_event=self._cancel,
                heartbeat=self._heartbeat,
                prompt_started=self._prompt_started,
            )
        except BaseException as exc:
            error = exc
        finally:
            with self._lock:
                self._error = self._error or error
            self._finished()
