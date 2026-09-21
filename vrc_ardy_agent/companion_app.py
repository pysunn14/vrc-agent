from __future__ import annotations

from dataclasses import dataclass, replace
import threading
import time
from typing import Any


@dataclass(frozen=True, slots=True)
class MacCompanionSnapshot:
    running: bool
    heartbeat_monotonic: float
    last_errors: tuple[str, ...]


class MacCompanionApp:
    """Owns the Mac ingress, control API, turns, and action shutdown order."""

    def __init__(
        self,
        *,
        bridge: Any,
        control: Any,
        pipeline: Any,
        interaction: Any,
        supervisor: Any,
        motion_director: Any,
        idle_timeout_seconds: float = 10.0,
    ) -> None:
        if idle_timeout_seconds <= 0:
            raise ValueError("idle_timeout_seconds must be positive")
        self._bridge = bridge
        self._control = control
        self._pipeline = pipeline
        self._interaction = interaction
        self._supervisor = supervisor
        self._motion_director = motion_director
        self._idle_timeout_seconds = float(idle_timeout_seconds)
        self._lock = threading.RLock()
        self._attempted: list[tuple[str, Any]] = []
        self._closed = False
        self._status = MacCompanionSnapshot(
            running=False,
            heartbeat_monotonic=time.monotonic(),
            last_errors=(),
        )

    def start(self) -> None:
        with self._lock:
            if self._attempted or self._closed:
                raise RuntimeError("Mac companion cannot be started again")
        try:
            self._attempted.append(("motion", self._motion_director))
            self._motion_director.start()
            self._attempted.append(("bridge", self._bridge))
            self._bridge.start()
            self._attempted.append(("control", self._control))
            self._control.start()
        except BaseException as exc:
            self._append_error(f"start failed: {type(exc).__name__}: {exc}")
            self._shutdown(raise_errors=False)
            raise
        with self._lock:
            self._status = replace(
                self._status,
                running=True,
                heartbeat_monotonic=time.monotonic(),
            )

    def stop(self) -> None:
        self._shutdown(raise_errors=True)

    def snapshot(self) -> MacCompanionSnapshot:
        with self._lock:
            return replace(
                self._status,
                heartbeat_monotonic=time.monotonic(),
            )

    def _shutdown(self, *, raise_errors: bool) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            attempted_names = {name for name, _ in self._attempted}
        errors: list[BaseException] = []

        if "control" in attempted_names:
            self._try("control stop", self._control.stop, errors)
        # Closing the pipeline first rejects new audio while keeping the bridge
        # available for in-flight screenshots and authoritative neutralization.
        self._try(
            "pipeline close",
            lambda: self._pipeline.close(timeout=self._idle_timeout_seconds),
            errors,
        )
        self._try(
            "interaction halt",
            lambda: self._interaction.halt_all(reason="companion_shutdown"),
            errors,
        )
        try:
            idle = self._supervisor.wait_until_idle(
                timeout=self._idle_timeout_seconds
            )
            if not idle:
                raise TimeoutError("character supervisor did not become idle")
        except BaseException as exc:
            errors.append(exc)
            self._append_error(
                f"supervisor wait failed: {type(exc).__name__}: {exc}"
            )
        if "motion" in attempted_names:
            self._try("motion stop", self._motion_director.stop, errors)
        if "bridge" in attempted_names:
            self._try("bridge stop", self._bridge.stop, errors)

        with self._lock:
            self._status = replace(
                self._status,
                running=False,
                heartbeat_monotonic=time.monotonic(),
            )
        if errors and raise_errors:
            raise RuntimeError(
                "; ".join(f"{type(error).__name__}: {error}" for error in errors)
            )

    def _try(
        self,
        label: str,
        operation: Any,
        errors: list[BaseException],
    ) -> None:
        try:
            operation()
        except BaseException as exc:
            errors.append(exc)
            self._append_error(f"{label} failed: {type(exc).__name__}: {exc}")

    def _append_error(self, error: str) -> None:
        with self._lock:
            self._status = replace(
                self._status,
                heartbeat_monotonic=time.monotonic(),
                last_errors=(*self._status.last_errors, error)[-50:],
            )
