from __future__ import annotations

from dataclasses import dataclass, replace
import threading
import time
from typing import Any


@dataclass(frozen=True, slots=True)
class CompanionHttpSnapshot:
    running: bool
    bound_host: str
    bound_port: int
    heartbeat_monotonic: float
    last_error: str | None


class CompanionHttpRuntime:
    """Observed lifecycle wrapper for the exploratory companion HTTP API."""

    def __init__(self, server: Any) -> None:
        host, port = server.server_address[:2]
        self._server = server
        self._condition = threading.Condition(threading.RLock())
        self._thread: threading.Thread | None = None
        self._closed = False
        self._status = CompanionHttpSnapshot(
            running=False,
            bound_host=str(host),
            bound_port=int(port),
            heartbeat_monotonic=time.monotonic(),
            last_error=None,
        )

    def start(self) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("companion HTTP runtime is closed")
            if self._thread is not None:
                raise RuntimeError("companion HTTP runtime is already started")
            thread = threading.Thread(
                target=self._serve,
                name="companion-control-http",
                daemon=True,
            )
            self._thread = thread
            self._status = replace(
                self._status,
                running=True,
                heartbeat_monotonic=time.monotonic(),
                last_error=None,
            )
            thread.start()
            if not thread.is_alive():
                self._thread = None
                self._status = replace(self._status, running=False)
                raise RuntimeError("companion HTTP server thread did not start")

    def stop(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
        if thread is not None:
            self._server.shutdown()
        self._server.server_close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
            if thread.is_alive():
                raise TimeoutError("companion HTTP server did not stop")
        with self._condition:
            self._thread = None
            self._status = replace(
                self._status,
                running=False,
                heartbeat_monotonic=time.monotonic(),
            )
            self._condition.notify_all()

    def snapshot(self) -> CompanionHttpSnapshot:
        with self._condition:
            thread = self._thread
            running = thread is not None and thread.is_alive()
            return replace(self._status, running=running)

    def _serve(self) -> None:
        try:
            self._server.serve_forever()
        except Exception as exc:
            with self._condition:
                self._status = replace(
                    self._status,
                    running=False,
                    heartbeat_monotonic=time.monotonic(),
                    last_error=f"{type(exc).__name__}: {exc}",
                )
                self._condition.notify_all()
        finally:
            with self._condition:
                self._status = replace(
                    self._status,
                    running=False,
                    heartbeat_monotonic=time.monotonic(),
                )
                self._condition.notify_all()
